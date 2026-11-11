"""The customer line in MuJoCo physics, headless.

`plant_only` is a PLANT-ONLY run: the robot never moves, so every inspection must be
rejected. `test_oracle_...` reuses the ORACLE pick-and-place demo (true poses, robot
pre-docked) on a Line-managed kit; it shows the plant's accept path on real ring physics
and is not a sensor-driven robot result.
"""
import json

import mujoco
import numpy as np
import pytest

from advaitic_systems.contract import inspection_passes
from advaitic_systems.evidence import check
from advaitic_systems.modbus import PLCClient
from frodo.apps import pick_place_demo
from frodo.cell import CELL, KIT_TOP_Z, STATIONS
from frodo.config import get_config, with_overrides
from frodo.plant import feeder
from frodo.plant.line import Line
from frodo.sim.scene import build_spec, reset

READY, NOT_READY, ASSEMBLY_DONE = 1, 2, 3
SPEED_A, SPEED_B, PHASE_A = 0.035, 0.027, 0.4
# Longest run-in: a lead beyond the belt is clamped to the upstream end, so a kit covers the
# whole belt and four are on it at once (the default lead is much shorter).
FULL_BELT = with_overrides(get_config(), {"plant": {"release_lead_m": 10.0}})


@pytest.fixture(scope="module")
def plant_only(tmp_path_factory):
    """52.5 s of the line alone at 35 mm/s, READY sent for A and B over Modbus."""
    out = tmp_path_factory.mktemp("plant-only")
    # Station A's pool only, to keep the test short. B's belt still runs (empty); B's
    # gating behind A is covered by the feeder-rules test in test_plant_units.py.
    model = build_spec(feeder.pool_layout(stations=["A"])).compile()
    data = mujoco.MjData(model)
    reset(model, data)
    obs = {"polled_ids": [], "accepted_polls": set(), "spacing": [], "kit_speed": []}
    a0 = model.joint("kit_A0_slide")
    a1 = model.joint("kit_A1_slide")
    ring0 = model.body("kit_A0_ring").id
    with Line(model, FULL_BELT, belt_speed_mps={"A": SPEED_A, "B": SPEED_B},
              release_phase_s={"A": PHASE_A}, events_path=out / "events.jsonl", port=0) as line:
        plc = PLCClient(port=line.port)
        plc.command(1, 1, 0, READY)
        plc.command(2, 2, 0, READY)   # B is ready too, but may not be fed before A is done
        step = 0
        while data.time < 52.5:
            mujoco.mj_step(model, data)
            line.step(data)
            step += 1
            if step % 250 == 0:       # robot-style poll, 2 Hz, over real TCP
                latest, _, _, accepted = plc.read(11, 4)
                if latest and latest not in obs["polled_ids"]:
                    obs["polled_ids"].append(latest)
                obs["accepted_polls"].add(accepted)
                if 20 < data.time < 40:   # kits 1 and 2 are both on the belt
                    obs["spacing"].append(float(data.qpos[a0.qposadr[0]] - data.qpos[a1.qposadr[0]]))
                    obs["kit_speed"].append(float(data.qvel[a0.dofadr[0]]))
            if abs(data.time - 30.0) < 1e-4:
                obs["ring_on_moving_kit"] = (data.xpos[ring0] - data.xpos[model.body("kit_A0").id])
                obs["encoder_at_30"] = line.conveyors()
        obs["registers"] = plc.read(0, 27)
        obs["belt_speed_hook"] = (line.belt_speed("A", data), line.belt_speed("B", data),
                                  line.belt_speed("nowhere", data))
        obs["ring0_final"] = data.xpos[ring0].copy()
        obs["a0_x_final"] = float(data.qpos[a0.qposadr[0]])
        obs["kits"] = [dict(k) for k in line.kits]
        obs["events"] = list(line.events.events)
        obs["counts"] = line.counts()
    obs["file_events"] = [json.loads(x) for x in (out / "events.jsonl").read_text().splitlines()]
    obs["raw_file"] = out / "seat_samples.jsonl"
    return obs


def test_kits_enter_at_customer_pitch_and_run_speed(plant_only):
    kits = plant_only["kits"]
    assert [k["kit_id"] for k in kits] == [1, 2, 3, 4, 5]
    assert all(k["station"] == "A" for k in kits)       # B ready, but A has no five accepted
    assert kits[0]["released_t"] == pytest.approx(PHASE_A, abs=0.005)
    releases = np.diff([k["released_t"] for k in kits])
    assert releases == pytest.approx(feeder.PITCH_M / SPEED_A, abs=0.005)
    # measured on the bodies, not on the feeder's own bookkeeping
    assert plant_only["spacing"] == pytest.approx([feeder.PITCH_M] * len(plant_only["spacing"]),
                                                  abs=5e-4)
    assert plant_only["kit_speed"] == pytest.approx([SPEED_A] * len(plant_only["kit_speed"]),
                                                    abs=1e-4)
    # the free ring rides its nest on the moving kit without slipping
    rx, ry = CELL["kit"]["ring_xy_in_kit_m"]
    assert plant_only["ring_on_moving_kit"][:2] == pytest.approx([rx, ry], abs=5e-4)


def test_arrival_is_registered_in_plc_over_tcp_at_the_bottle_crossing(plant_only):
    kits, reg = plant_only["kits"], plant_only["registers"]
    entry_travel = feeder.WORK_ENTRY_X - feeder.BOTTLE_X_IN_KIT - feeder.release_x(FULL_BELT)
    assert kits[0]["arrived_t"] - kits[0]["released_t"] == pytest.approx(
        entry_travel / SPEED_A, abs=0.01)
    arrivals = [k["arrived_t"] for k in kits if k["arrived_t"] is not None]
    assert len(arrivals) == 3
    assert np.diff(arrivals) == pytest.approx(feeder.PITCH_M / SPEED_A, abs=0.005)
    assert plant_only["polled_ids"] == [1, 2, 3]         # seen by the 2 Hz Modbus poll
    assert reg[0] == 1 and reg[3] == 3 and reg[11] == 3  # schema, total arrivals, latest at A
    assert reg[10] == 1 and reg[20] == 1                 # READY commands were applied
    assert reg[21] == 0 and reg[23] == 0                 # nothing arrived or inspected at B


def test_untouched_ring_is_rejected_at_the_inspection_line(plant_only):
    kits, reg = plant_only["kits"], plant_only["registers"]
    inspections = [e for e in plant_only["events"] if e["type"] == "inspection"]
    assert [e["kit_id"] for e in inspections] == [1]
    e = inspections[0]
    assert e["t"] - kits[0]["arrived_t"] == pytest.approx(0.85 / SPEED_A, abs=0.01)
    assert e["kit_contact"] is True and e["bottle_contact"] is False and e["hold_s"] == 0.0
    assert e["released"] is True and e["belt_speed_mps"] == SPEED_A
    assert not inspection_passes(e)
    assert kits[0]["accepted"] is False and kits[0]["robot_done"] is False
    assert (reg[12], reg[13], reg[14], reg[15]) == (1, 2, 0, 1)   # inspected id, REJECTED, 0, 1
    assert plant_only["accepted_polls"] == {0}
    assert not [e for e in plant_only["events"] if e["type"] == "seat_verified"]
    assert plant_only["counts"]["A"] == {"accepted": 0, "rejected": 1, "in_flight": 4}


def test_kit_is_recycled_with_its_ring_back_on_the_nest(plant_only):
    kits = plant_only["kits"]
    assert kits[0]["carrier"] == "kit_A0" and kits[4]["carrier"] == "kit_A0"
    x = plant_only["a0_x_final"]
    assert feeder.release_x(FULL_BELT) < x < feeder.release_x(FULL_BELT) + 0.05   # just released again
    rx, ry = CELL["kit"]["ring_xy_in_kit_m"]
    expected = STATIONS["A"].to_world([x + rx, ry], KIT_TOP_Z + CELL["ring"]["thickness_m"] / 2)
    assert plant_only["ring0_final"] == pytest.approx(expected, abs=1e-3)


def test_belt_encoder_accessors(plant_only):
    assert plant_only["belt_speed_hook"] == (SPEED_A, SPEED_B, None)
    travel = plant_only["encoder_at_30"]
    assert travel["A"] == pytest.approx((30.0 * SPEED_A, SPEED_A), abs=1e-4)
    assert travel["B"] == pytest.approx((30.0 * SPEED_B, SPEED_B), abs=1e-4)  # runs empty


def test_events_file_matches_and_checker_accepts_the_plant_events(plant_only):
    assert plant_only["file_events"] == plant_only["events"]
    assert {e["type"] for e in plant_only["events"]} == {"kit_arrived", "inspection"}
    assert plant_only["raw_file"].read_text() == ""       # ring never left its nest
    violations = check(plant_only["events"])["violations"]
    # Expected: the run has no robot, so counts and motion samples are missing. Nothing
    # about the plant's own events (schema, pitch, belt speed, inspection crossing).
    assert sorted(violations) == ["Expected five accepted per station, got {1: 0, 2: 0}",
                                  "Motion samples do not cover mission"]


class _SteppedByLine:
    """Stands in for the `mujoco` module inside the oracle demo so that every physics step
    is followed by `Line.step`, and plays the robot's PLC adapter for that demo."""

    def __init__(self):
        self.line = self.model = self.data = self.plc = None
        self.sequence, self.was_held, self.accepted_before_done = 0, False, None

    def __getattr__(self, name):
        return getattr(mujoco, name)

    def command(self, station, kit_id, command):
        self.sequence += 1
        self.plc.command(self.sequence, station, kit_id, command)

    def mj_step(self, model, data):
        if self.line is None:
            self.model, self.data = model, data
            self.line = Line(model, belt_speed_mps=SPEED_A, port=0).start()
            self.plc = PLCClient(port=self.line.port)
            self.command(1, 0, READY)
        mujoco.mj_step(model, data)
        self.line.step(data)
        if self.sequence == 1 and self.line.kits:
            self.command(1, 0, NOT_READY)      # one kit is enough for this test
        held = bool(data.eq_active[model.equality("kit_A0_grasp").id])
        if self.was_held and not held:         # the demo just released the ring
            self.accepted_before_done = self.plc.read(14, 1)[0]
            self.command(1, self.line.kits[0]["kit_id"], ASSEMBLY_DONE)
        self.was_held = held


def test_oracle_pick_place_on_line_managed_kit_is_accepted(monkeypatch):
    """ORACLE manipulation (true poses, pre-docked robot) on a kit released by the Line."""
    sim = _SteppedByLine()
    monkeypatch.setattr(pick_place_demo, "mujoco", sim)
    monkeypatch.setattr(pick_place_demo, "build_spec",
                        lambda kits: build_spec(feeder.pool_layout(stations=["A"])))
    try:
        result = pick_place_demo.run(belt_speed=SPEED_A, verbose=False)
        assert result["seat_verified_t"] is not None, result
        kit = sim.line.kits[0]
        accepted_after_done = sim.plc.read(14, 1)[0]
        while kit["inspected_t"] is None and sim.data.time < 60:
            sim.mj_step(sim.model, sim.data)   # arm holds its last pose; belt carries on
        registers = sim.plc.read(0, 27)
        events = list(sim.line.events.events)
        ledger = sim.line.plc.command_ledger()
    finally:
        if sim.line:
            sim.line.close()

    assert len(sim.line.kits) == 1                       # NOT_READY stopped the feeder
    assert [e["type"] for e in events] == ["kit_arrived", "seat_verified", "inspection"]
    arrived, seat, inspection = events
    assert inspection_passes(seat) and inspection_passes(inspection)
    assert kit["cycle_s"] == pytest.approx(seat["t"] - arrived["t"])
    assert kit["cycle_s"] < CELL["timing"]["assembly_target_s"] and kit["seat_in_window"]
    assert inspection["t"] - arrived["t"] == pytest.approx(0.85 / SPEED_A, abs=0.11)
    assert inspection["hold_s"] > seat["hold_s"] >= 0.5  # hold kept accumulating to the line
    # ASSEMBLY_DONE alone accepted nothing; the inspection did.
    assert sim.accepted_before_done == 0 and accepted_after_done == 0
    assert (registers[12], registers[13], registers[14], registers[15]) == (1, 1, 1, 0)
    assert kit["accepted"] is True and kit["robot_done"] is True
    assert [c["command"] for c in ledger] == [READY, NOT_READY, ASSEMBLY_DONE]
