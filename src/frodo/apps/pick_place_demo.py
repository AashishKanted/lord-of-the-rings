"""First manipulation demo: pick the ring off a moving kit and seat it on the bottle.

ORACLE MODE: the robot reads the true ring/bottle poses and belt speed from the
simulator, and starts already docked at station A. This proves kinematics, belt
tracking, grasp and seating physics only. It is not a sensor-driven result.

    uv run python -m frodo.apps.pick_place_demo --video results/demo/pick_place.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from ..cell import CELL, STATIONS
from ..config import get_config
from ..log import get_logger, setup_logging
from ..plant.inspection import SeatMonitor
from ..robot.description import ARM_HOME
from ..robot.manipulation.ik import WALL_MID, ArmIK, tcp_in_site, tool_rotation
from ..robot.manipulation.motion import Slew
from ..sim.grasp import activate_grasp_weld
from ..sim.scene import build_spec, reset

log = get_logger(__name__)
M = get_config().manipulation
CONTROL_DT = M.control_dt_s
NECK_TOP = CELL["bottle"]["height_m"]
WORK_ENTRY_X = CELL["conveyor"]["work_entry_local_x_m"]



def run(belt_speed=0.03, video: Path | None = None, kit_start_x=-0.55, verbose=True):
    st, kit = STATIONS["A"], "kit_A0"
    model = build_spec({st.name: [kit_start_x]}).compile()
    data = mujoco.MjData(model)
    rot = tool_rotation(st)
    belt = np.array([np.cos(st.yaw), np.sin(st.yaw), 0.0])
    bottle_xy, ring_xy = CELL["kit"]["bottle_xy_in_kit_m"], CELL["kit"]["ring_xy_in_kit_m"]
    ring_from_bottle = (ring_xy[0] - bottle_xy[0]) * belt  # along the belt on this kit

    # Start docked with the arm hovering over the ring lane (setup shortcut for this demo).
    reset(model, data, st.dock, ARM_HOME)
    ik = ArmIK(model, data.qpos, rot)
    ring_at_entry = WORK_ENTRY_X + ring_xy[0] - bottle_xy[0]
    hover = st.to_world([ring_at_entry + 0.03 + WALL_MID, ring_xy[1]], 0.775 + 0.06)
    q_ready = ik.solve_static(hover)
    reset(model, data, st.dock, q_ready)
    grip = model.actuator("fingers_actuator").id
    data.ctrl[grip] = M.grip_preshape
    data.ctrl[model.actuator(f"{kit}_drive").id] = belt_speed

    ring_body, bottle_body = model.body(f"{kit}_ring").id, model.body(f"{kit}_bottle").id
    weld = model.equality(f"{kit}_grasp").id
    pads = {model.geom(n).id: n[0] for n in ("left_pad1", "left_pad2", "right_pad1", "right_pad2")}
    site = model.site("pinch_site").id
    monitor = SeatMonitor(model, kit)  # plant side; the controller never reads it

    def tcp_now():
        return data.site_xpos[site] + data.site_xmat[site].reshape(3, 3) @ tcp_in_site(rot)

    def pads_on_ring():
        sides = set()
        for c in data.contact[:data.ncon]:
            for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g in pads and model.geom_bodyid[other] == ring_body:
                    sides.add(pads[g])
        return sides == {"l", "r"}

    frames, renderers = [], None
    if video:
        renderers = (mujoco.Renderer(model, 720, 960), mujoco.Renderer(model, 240, 320))
    phases, phase, phase_t0 = [], "WAIT", 0.0
    slew = Slew(hover)            # commanded TCP, expressed relative to `origin`
    origin = np.zeros(3)          # moving reference: ring grasp point, then bottle base
    rel = hover.copy()
    grasp_offset = np.zeros(3)    # TCP minus ring centre, captured at grasp
    place_trim = np.zeros(3)      # integral correction of ring-over-neck error
    centred_ticks = 0
    entry_t = None
    steps_per_tick = round(CONTROL_DT / model.opt.timestep)
    track_err = []

    def enter(new):
        nonlocal phase, phase_t0
        phases.append((round(data.time, 2), new))
        if verbose:
            log.info("t=%6.2fs  %9s -> %s", data.time, phase, new)
        phase, phase_t0 = new, data.time

    def rebase(new_origin):
        """Change the moving reference without moving the commanded point."""
        nonlocal origin
        slew.p = slew.p + origin - new_origin
        origin = new_origin

    while data.time < 40.0 and phase != "DONE":
        # ---------- oracle "perception" ----------
        ring_pos, bottle_pos = data.xpos[ring_body].copy(), data.xpos[bottle_body].copy()
        lead = belt * belt_speed * M.belt_lead_s
        grasp_point = ring_pos + belt * WALL_MID + [0, 0, 0.001]  # downstream wall of the ring
        in_phase = data.time - phase_t0

        # ---------- state machine ----------
        if phase == "WAIT":
            goal = slew.p
            if (bottle_pos - st.center) @ belt > WORK_ENTRY_X + 0.01:
                rebase(grasp_point)
                enter("PREGRASP")
        if phase == "PREGRASP":
            origin, goal = grasp_point, np.array([0, 0, 0.04])
            if np.linalg.norm(slew.p - goal) < 1e-4:
                enter("DESCEND")
        elif phase == "DESCEND":
            origin, goal = grasp_point, np.zeros(3)
            if np.linalg.norm(slew.p) < 1e-4 and np.linalg.norm(tcp_now() - grasp_point) < 0.003:
                data.ctrl[grip] = M.grip_closed
                enter("CLOSE")
        elif phase == "CLOSE":
            origin, goal = grasp_point, np.zeros(3)
            # Grasp-constraint activation conditions: both pads physically touching this
            # ring, tool within 6 mm of the planned grasp point, fingers commanded closed.
            if in_phase > 0.15 and pads_on_ring() and np.linalg.norm(tcp_now() - grasp_point) < 0.006:
                before = data.xpos[ring_body].copy()
                activate_grasp_weld(model, data, weld)
                grasp_offset = tcp_now() - before
                rebase(bottle_pos)
                enter("LIFT")
            elif in_phase > 2.0:
                enter("FAILED")
                break
        elif phase in ("LIFT", "TRANSFER", "LOWER"):
            origin = bottle_pos
            lift_z, release_z = NECK_TOP + 0.022, NECK_TOP + 0.007  # ring centre above bottle base
            ring_goal = {"LIFT": ring_from_bottle + [0, 0, lift_z],
                         "TRANSFER": np.array([0, 0, lift_z]),
                         "LOWER": np.array([0, 0, release_z])}[phase]
            goal = ring_goal + grasp_offset + place_trim
            if np.linalg.norm(slew.p - goal) < 1e-4:
                if phase == "LOWER":
                    # Closed-loop trim on the measured ring-over-neck error: the seat only
                    # self-centres within ~2 mm, so servo it out before letting go.
                    err_xy = (bottle_pos - ring_pos)[:2]
                    radial = np.linalg.norm(err_xy)
                    place_trim[:2] += 0.15 * err_xy
                    centred_ticks = centred_ticks + 1 if radial < M.release_radial_m else 0
                    if centred_ticks >= 10:
                        data.eq_active[weld] = 0
                        data.ctrl[grip] = M.grip_preshape
                        if verbose:
                            tilt = np.degrees(np.arccos(abs(data.xmat[ring_body][8])))
                            log.info("release: ring %.1f mm off bottle axis, %.1f deg tilt "
                                     "in hand", 1000 * radial, tilt)
                        enter("RELEASE")
                else:
                    enter({"LIFT": "TRANSFER", "TRANSFER": "LOWER"}[phase])
        elif phase == "RELEASE":
            origin = bottle_pos
            if in_phase > 0.3:
                enter("RETREAT")
        elif phase == "RETREAT":
            origin = bottle_pos
            goal = np.array([0, 0, NECK_TOP + 0.08]) + grasp_offset - 0.05 * np.array(
                [np.cos(st.dock[2]), np.sin(st.dock[2]), 0])
            if np.linalg.norm(slew.p - goal) < 1e-4 and monitor.seat_verified_t is not None:
                enter("DONE")
            elif in_phase > 6.0:
                enter("DONE")

        # ---------- arm command ----------
        if phase not in ("WAIT", "DONE", "FAILED"):
            rel = slew.step(goal)
        target = origin + rel + (lead if phase != "WAIT" else 0)
        data.ctrl[ik.arm_ctrl] = ik.step(target)
        if phase in ("DESCEND", "CLOSE", "LOWER"):
            track_err.append(np.linalg.norm(tcp_now() - (origin + rel)))

        # ---------- simulate one control period; plant-side measurement every step ----------
        for _ in range(steps_per_tick):
            mujoco.mj_step(model, data)
            measurement = monitor.update(data)
            x_bottle = (data.xpos[bottle_body] - st.center) @ belt
            if entry_t is None and x_bottle >= CELL["conveyor"]["work_entry_local_x_m"]:
                entry_t = data.time
        if video and len(frames) < data.time * 30:
            renderers[0].update_scene(data, camera=f"close_{st.name}")
            renderers[1].update_scene(data, camera="wrist")
            frame = renderers[0].render().copy()
            frame[8:248, 632:952] = renderers[1].render()
            frames.append(frame)

    result = {
        "mode": "ORACLE poses, robot pre-docked",
        "belt_speed_mps": belt_speed,
        "phases": phases,
        "bottle_entered_work_zone_t": entry_t,
        "seat_verified_t": monitor.seat_verified_t,
        "cycle_s": None if monitor.seat_verified_t is None or entry_t is None
        else round(monitor.seat_verified_t - entry_t, 2),
        "tracking_error_mm_max": round(1000 * max(track_err), 2) if track_err else None,
        "final_measurement": {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in measurement.items()},
    }
    if video:
        import imageio.v2 as imageio
        video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video, frames, fps=30, macro_block_size=None)
        for r in renderers:
            r.close()
        result["video"] = str(video)
    return result


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--belt-speed", type=float, default=0.03)
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    setup_logging()
    print(json.dumps(run(args.belt_speed, args.video), indent=2))


if __name__ == "__main__":
    main()
