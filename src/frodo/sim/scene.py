"""Digital twin of the customer cell: room, two conveyors, kits and the TidyBot.

Everything is generated from ``spec/customer_cell.json`` so the scene cannot drift
from the customer dimensions. Kits ride a slide joint along their belt and are
driven by a velocity actuator (the README's "carrier along its track"); the bottle
is retained on the kit; the ring is a free rigid body resting on the kit.

    uv run python -m frodo.sim.scene --view          # interactive viewer
    uv run python -m frodo.sim.scene --render out.png  # headless snapshot
"""
from __future__ import annotations

import argparse
from math import cos, pi, radians, sin
from pathlib import Path

import mujoco
import numpy as np

from ..cell import CELL, KIT_TOP_Z, STATIONS, Station
from ..config import ROOT, get_config
from ..robot.description import ARM_HOME, ARM_JOINTS, ARM_STOWED, BASE_JOINTS

TIDYBOT_XML = ROOT / "third_party" / "mujoco_menagerie" / "stanford_tidybot" / "tidybot.xml"

# The stock TidyBot mounts the arm 0.335 m above the floor, which cannot reach the
# 0.78-1.0 m work height from the specified dock (see reach study in the report).
# Documented modification: a plain spacer column under the arm.
ARM_RISER_M = get_config().robot.arm_riser_m

# Collision bits: 1 = robot + cell structure, 2 = floor + loose parts.
# The base has no vertical freedom, so it must not fight the floor plane.
ROBOT, PARTS = 1, 2
PART_CONTACT = dict(friction=[.5, .005, .0001], solref=[.01, 1], solimp=[.95, .99, .001, .5, 2])

BOX, CYL, MESH, PLANE = (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER,
                         mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_PLANE)
VISUAL_GROUP = 2          # geoms in this group are decoration: no collision, invisible to lidar
LIDAR_SITE_PREFIX = "lidar_"
BASE_TOP_Z = 0.325        # top plate of the TidyBot base (Menagerie mesh), where the mast stands


def _yaw_quat(yaw):
    return [cos(yaw / 2), 0, 0, sin(yaw / 2)]


def _add_room(spec):
    w = spec.worldbody
    lx, ly = CELL["cell_size_m"]
    w.add_geom(name="floor", type=PLANE, size=[0, 0, .05], pos=[lx / 2, ly / 2, 0],
               rgba=[.32, .34, .36, 1], contype=PARTS, conaffinity=PARTS)
    t, h = .05, .4
    for name, pos, size in (("wall_s", [lx / 2, -t, h], [lx / 2 + 2 * t, t, h]),
                            ("wall_n", [lx / 2, ly + t, h], [lx / 2 + 2 * t, t, h]),
                            ("wall_w", [-t, ly / 2, h], [t, ly / 2, h]),
                            ("wall_e", [lx + t, ly / 2, h], [t, ly / 2, h])):
        w.add_geom(name=name, type=BOX, pos=pos, size=size, rgba=[.75, .75, .78, .35])
    w.add_light(pos=[lx / 2, ly / 2, 4], dir=[0, 0, -1], diffuse=[.8, .8, .8])
    target = w.add_body(name="cell_center", pos=[lx / 2, ly / 2, .5])
    w.add_camera(name="overview", pos=[lx / 2, -3.2, 5.5], fovy=50,
                 mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody=target.name)
    for st in STATIONS.values():
        w.add_body(name=f"view_{st.name}", pos=st.to_world([-0.05, 0.1], 0.9))
        for cam, xy, z in (("station", [-1.1, 1.3], 1.7), ("close", [0.35, -0.75], 1.25)):
            w.add_camera(name=f"{cam}_{st.name}", pos=st.to_world(xy, z), fovy=50,
                         mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
                         targetbody=f"conveyor_{st.name}" if cam == "station"
                         else f"view_{st.name}")


def _add_conveyor(spec, st: Station):
    c = CELL["conveyor"]
    h = c["height_m"]
    body = spec.worldbody.add_body(name=f"conveyor_{st.name}", pos=[*st.center[:2], h],
                                   quat=_yaw_quat(st.yaw))
    body.add_geom(name=f"belt_{st.name}", type=BOX, pos=[0, 0, -h / 2],
                  size=[c["length_m"] / 2, c["width_m"] / 2, h / 2], rgba=[.15, .15, .17, 1],
                  contype=ROBOT | PARTS, conaffinity=ROBOT | PARTS)
    # Visual markers: work-zone entry/exit and inspection line (no collision).
    for key, rgba in (("work_entry_local_x_m", [.1, .8, .2, 1]), ("work_exit_local_x_m", [.9, .7, .1, 1]),
                      ("inspection_local_x_m", [.9, .2, .2, 1])):
        body.add_geom(type=BOX, pos=[c[key], 0, .0005], size=[.003, c["width_m"] / 2, .0005],
                      rgba=rgba, contype=0, conaffinity=0)


def add_kit(spec, st: Station, kit_name: str, local_x: float = 0.0):
    """Kit carrier + retained bottle + free ring. Returns the names created."""
    k, b, r = CELL["kit"], CELL["bottle"], CELL["ring"]
    # Extra collision bits used only by kits: ring segments that rest on a tray; ring-to-ring.
    RING_SUPPORT, RING = 4, 8
    both = dict(contype=ROBOT | PARTS, conaffinity=ROBOT | PARTS, **PART_CONTACT)
    # Kit frame: plan centre on its top surface, x along belt travel.
    # ref=local_x makes the slide joint value equal the kit's station-local x.
    kit = spec.worldbody.add_body(name=kit_name, pos=st.to_world([local_x, 0], KIT_TOP_Z),
                                  quat=_yaw_quat(st.yaw))
    kit.add_joint(name=f"{kit_name}_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0],
                  ref=local_x)
    # The tray collides with the robot and with the ring's support segments only (see ring).
    kit.add_geom(name=f"{kit_name}_tray", type=BOX, pos=[0, 0, -k["height_m"] / 2], mass=1.0,
                 size=[k["length_m"] / 2, k["width_m"] / 2, k["height_m"] / 2],
                 rgba=[.55, .45, .3, 1], contype=0, conaffinity=ROBOT | RING_SUPPORT,
                 **PART_CONTACT)
    bx, by = k["bottle_xy_in_kit_m"]
    rb, rn = b["body_diameter_m"] / 2, b["neck_diameter_m"] / 2
    zb, zs = b["body_height_m"], b["body_height_m"] + b["shoulder_height_m"]
    bottle = kit.add_body(name=f"{kit_name}_bottle", pos=[bx, by, 0])
    rgba = [.85, .9, .95, 1]
    bottle.add_geom(name=f"{kit_name}_bottle_body", type=CYL, pos=[0, 0, zb / 2],
                    size=[rb, zb / 2, 0], mass=b["mass_kg"], rgba=rgba, **both)
    bottle.add_geom(name=f"{kit_name}_bottle_shoulder", type=MESH, meshname="bottle_shoulder",
                    pos=[0, 0, 0], mass=0, rgba=rgba, **both)
    bottle.add_geom(name=f"{kit_name}_bottle_neck", type=CYL, pos=[0, 0, (zs + b["height_m"]) / 2],
                    size=[rn, (b["height_m"] - zs) / 2, 0], mass=0, rgba=rgba, **both)
    # Belt drive: velocity servo on the carrier; ctrl is belt speed in m/s.
    act = spec.add_actuator(name=f"{kit_name}_drive", target=f"{kit_name}_slide",
                            trntype=mujoco.mjtTrn.mjTRN_JOINT)
    kv = 2000.0
    act.gaintype, act.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
    act.gainprm[0], act.biasprm[2] = kv, -kv

    # Ring: free body; 48 boxes preserve the hole (same construction as the starter fixture).
    ro, ri, th = r["outer_diameter_m"] / 2, r["inner_diameter_m"] / 2, r["thickness_m"]
    rx, ry = k["ring_xy_in_kit_m"]
    ring = spec.worldbody.add_body(name=f"{kit_name}_ring",
                                   pos=st.to_world([local_x + rx, ry], KIT_TOP_Z + th / 2))
    ring.add_freejoint(name=f"{kit_name}_ring_free")
    # All 48 segments collide with the robot, bottle, belt, floor and other rings. Only every
    # `support_every`-th one also collides with a kit tray: a flat ring on a flat tray would
    # otherwise make 192 contacts per kit, which dominates the step time of a full kit pool.
    n, mid = 48, (ro + ri) / 2
    support_every = n // get_config().plant.ring_tray_contact_segments
    for i in range(n):
        a = i * 2 * pi / n
        ring.add_geom(type=BOX, pos=[mid * cos(a), mid * sin(a), 0], quat=_yaw_quat(a),
                      size=[(ro - ri) / 2, mid * sin(pi / n), th / 2], mass=r["mass_kg"] / n,
                      rgba=[.1, .55, .9, 1],
                      contype=RING | (RING_SUPPORT if i % support_every == 0 else 0),
                      conaffinity=ROBOT | PARTS | RING, **PART_CONTACT)
    # Documented grasp constraint (README "What you may simplify"): a weld between the
    # gripper and this ring. It starts inactive; the robot may only activate it after a
    # verified physical grasp (see frodo.grasp) and must deactivate it on release.
    weld = spec.add_equality(name=f"{kit_name}_grasp", type=mujoco.mjtEq.mjEQ_WELD,
                             objtype=mujoco.mjtObj.mjOBJ_BODY, name1="base", name2=ring.name)
    weld.active = False
    return kit_name


def mast_camera_quat(yaw_deg: float, pitch_deg: float):
    """Orientation (in the base frame) of a MuJoCo camera whose optical axis has heading
    `yaw_deg` and is tilted `pitch_deg` below the horizontal, image upright."""
    yaw, pitch = radians(yaw_deg), radians(pitch_deg)
    right = [sin(yaw), -cos(yaw), 0.0]                                   # camera +x
    up = [cos(yaw) * sin(pitch), sin(yaw) * sin(pitch), cos(pitch)]      # camera +y
    back = [-cos(yaw) * cos(pitch), -sin(yaw) * cos(pitch), sin(pitch)]  # camera +z
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.array([right, up, back]).T.flatten())
    return quat


def _add_robot_sensors(spec):
    """Sensor mounts on the mobile base: lidar sites + pucks and the mast camera.

    Poses come from [sensors.lidar] and [sensors.cameras.mast]. The simulated sensors in
    ``frodo.sim.sensors`` read these mounts back from the compiled model.
    """
    sensors = get_config().sensors
    base = spec.body("base_link")
    decoration = dict(contype=0, conaffinity=0, group=VISUAL_GROUP, mass=0)
    lidar = sensors.lidar
    for mount in lidar.mounts:
        pos = [mount.x, mount.y, lidar.height_m]
        base.add_site(name=f"{LIDAR_SITE_PREFIX}{mount.name}", pos=pos,
                      quat=_yaw_quat(radians(mount.yaw_deg)))
        base.add_geom(name=f"lidar_{mount.name}_puck", type=CYL, pos=pos, size=[.03, .02, 0],
                      rgba=[.9, .45, .05, 1], **decoration)
    # Wrist camera: the Menagerie model points it along the tool axis. The gripper works
    # pitched by [manipulation] tilt_deg, so the camera is pitched back by `pitch_back_deg`
    # (about its own x axis) to look straight down at the belt from the working posture.
    wrist = sensors.cameras.wrist
    camera = spec.camera("wrist")
    back = np.zeros(4)
    mujoco.mju_axisAngle2Quat(back, [1.0, 0.0, 0.0], -radians(wrist.pitch_back_deg))
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, np.array(camera.quat, dtype=float), back)
    camera.quat, camera.fovy = quat, wrist.fovy_deg
    mast = sensors.cameras.mast
    x, y, z = mast.pos_m
    # The post is a real obstacle for the arm, so it collides like the rest of the robot.
    base.add_geom(name="mast_post", type=CYL, mass=0.5, rgba=[.2, .2, .22, 1], size=[.012, 0, 0],
                  fromto=[x, y, BASE_TOP_Z, x, y, z - 0.03])
    base.add_geom(name="mast_head", type=BOX, pos=[x, y, z], size=[.02, .045, .015],
                  rgba=[.1, .1, .1, 1], **decoration)
    base.add_camera(name="mast", pos=[x, y, z], fovy=mast.fovy_deg,
                    quat=mast_camera_quat(mast.yaw_deg, mast.pitch_deg))


def build_spec(kits: dict[str, list[float]] | None = None) -> mujoco.MjSpec:
    """kits maps station name -> list of initial kit local-x positions."""
    spec = mujoco.MjSpec.from_file(str(TIDYBOT_XML))
    spec.modelname = "frodo_cell"
    for key in list(spec.keys):
        spec.delete(key)
    spec.option.timestep = get_config().sim.timestep_s
    # Arm joint servos in the Menagerie model are plain PD position loops, which sag
    # ~10 mm at full reach. The real Gen3 controller compensates gravity; model that.
    for body in spec.bodies:
        if body.name not in ("world", "base_link"):
            body.gravcomp = 1.0
    b = CELL["bottle"]
    verts = []
    for z, radius in ((b["body_height_m"], b["body_diameter_m"] / 2),
                      (b["body_height_m"] + b["shoulder_height_m"], b["neck_diameter_m"] / 2)):
        for i in range(48):
            a = i * 2 * pi / 48
            verts.extend([radius * cos(a), radius * sin(a), z])
    spec.add_mesh(name="bottle_shoulder", uservert=verts)
    for st in STATIONS.values():
        _add_conveyor(spec, st)
    _add_room(spec)
    arm_base = spec.body("gen3/base_link")
    spec.body("base_link").add_geom(name="arm_riser", type=BOX, mass=4.0, rgba=[.35, .37, .37, 1],
                                    pos=[0, 0, arm_base.pos[2] + ARM_RISER_M / 2],
                                    size=[.06, .06, ARM_RISER_M / 2])
    arm_base.pos[2] += ARM_RISER_M
    _add_robot_sensors(spec)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
    for station, xs in (kits or {}).items():
        for i, x in enumerate(xs):
            add_kit(spec, STATIONS[station], f"kit_{station}{i}", x)
    return spec


def reset(model, data, base_pose=None, arm=ARM_STOWED):
    """Put the robot at a base pose (default: mission start) with the given arm config."""
    mujoco.mj_resetData(model, data)
    pose = CELL["start_pose_world"] if base_pose is None else base_pose
    for name, value in zip(BASE_JOINTS + ARM_JOINTS, [*pose, *arm], strict=True):
        data.qpos[model.joint(name).qposadr[0]] = value
        data.ctrl[model.actuator(name).id] = value
    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--view", action="store_true", help="open the interactive viewer")
    parser.add_argument("--render", type=Path, help="write a headless PNG snapshot")
    parser.add_argument("--camera", default="overview")
    parser.add_argument("--write-xml", type=Path)
    parser.add_argument("--dock", choices=list(STATIONS), help="place robot at a dock, arm at home")
    args = parser.parse_args()

    spec = build_spec({"A": [0.0], "B": [0.0]})
    model = spec.compile()
    data = mujoco.MjData(model)
    if args.dock:
        reset(model, data, STATIONS[args.dock].dock, ARM_HOME)
    else:
        reset(model, data)
    print(f"compiled: nq={model.nq} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom} "
          f"dt={model.opt.timestep}")
    if args.write_xml:
        args.write_xml.write_text(spec.to_xml())
    if args.render:
        from PIL import Image
        with mujoco.Renderer(model, 960, 1280) as renderer:
            renderer.update_scene(data, camera=args.camera)
            Image.fromarray(renderer.render()).save(args.render)
        print(f"wrote {args.render}")
    if args.view:
        from mujoco import viewer
        viewer.launch(model, data)


if __name__ == "__main__":
    main()
