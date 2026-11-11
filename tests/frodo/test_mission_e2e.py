"""End-to-end nominal mission in MuJoCo: start pose -> dock A -> five rings seated and
accepted -> stow -> dock B -> five more.

ORACLE PERCEPTION: kit poses given to the robot are simulator truth (perception.method =
"oracle"). Localization, navigation, the arm controller, the line protocol and the customer
line with its independent seat measurement are the real modules. This is a development
test of the integrated stack, not a customer acceptance run.

Takes about 1.5 minutes. Set FRODO_SKIP_E2E=1 to skip it.
"""
import json
import os

import pytest

from advaitic_systems.evidence import check
from frodo.apps.mission import run
from frodo.cell import CELL

pytestmark = pytest.mark.skipif(os.environ.get("FRODO_SKIP_E2E") == "1",
                                reason="FRODO_SKIP_E2E=1")
TIMING, ROBOT = CELL["timing"], CELL["robot"]


@pytest.fixture(scope="module")
def mission(tmp_path_factory):
    out = tmp_path_factory.mktemp("nominal_seed0")
    summary = run("nominal", out, seed=0, port=0)
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    return summary, events


def test_run_is_labelled_as_oracle_perception(mission):
    summary, _ = mission
    assert summary["perception"].startswith("ORACLE")
    assert "rudimentary" in summary["safety_monitor"]


def test_five_accepted_at_each_station(mission):
    """docs/ACCEPTANCE.md: nominal runs may include rejects but must reach the final counts.
    Ring seating has a thin tilt margin (see the report), so a reject can occur; a rejected
    kit must then have been replaced, not hidden."""
    summary, _ = mission
    assert summary["robot_finished"] and summary["robot_mode_at_end"] == "done"
    rejected = 0
    for station in ("A", "B"):
        counts = summary["counts"][station]
        assert counts["accepted"] == 5 and counts["in_flight"] == 0
        rejected += counts["rejected"]
    kits = summary["kits"]
    assert [k["kit_id"] for k in kits] == list(range(1, 11 + rejected))
    assert sum(not k["accepted"] for k in kits) == rejected <= 2


def test_acceptance_comes_from_the_plant_measurement_not_from_the_robot(mission):
    summary, events = mission
    accepted = [k for k in summary["kits"] if k["accepted"]]
    assert len(accepted) == 10
    for kit in accepted:
        done = next(e for e in events if e["type"] == "assembly_done" and e["kit_id"] == kit["kit_id"])
        seat = next(e for e in events if e["type"] == "seat_verified" and e["kit_id"] == kit["kit_id"])
        # the robot reports DONE when it lets go; the plant verifies the seat only after its
        # own 0.5 s settled hold, and accepts only at the downstream inspection
        assert done["t"] < seat["t"] < kit["inspected_t"]
        assert kit["robot_done"] and kit["seat_in_window"]
        assert seat["released"] and seat["bottle_contact"] and not seat["kit_contact"]
    for kit in summary["kits"]:
        if not kit["accepted"]:      # the robot said DONE, the plant measurement said no
            assert kit["robot_done"] and kit["inspected_t"] is not None


def test_cycle_and_mission_time_limits(mission):
    summary, _ = mission
    assert summary["accepted_cycle_s"]["max"] <= TIMING["assembly_target_s"]
    assert summary["simulated_s"] <= TIMING["nominal_mission_target_s"]


def test_docks_are_inside_the_customer_tolerance_and_b_follows_five_accepted_at_a(mission):
    summary, _ = mission
    assert [d["station"] for d in summary["docks"]] == ["A", "B"]
    for dock in summary["docks"]:
        assert dock["true_error_mm"] <= 1e3 * ROBOT["dock_position_tolerance_m"]
    fifth_a = sorted(k["inspected_t"] for k in summary["kits"] if k["station"] == "A")[-1]
    assert summary["docks"][1]["t"] > fifth_a


def test_base_obeys_speed_limits_and_stands_still_while_the_arm_works(mission):
    summary, _ = mission
    assert summary["max_base_speed_mps"] <= ROBOT["max_transit_speed_mps"]
    assert summary["max_base_speed_arm_unstowed_mps"] <= ROBOT["assembly_base_speed_max_mps"]


def test_starter_checker_finds_the_log_consistent(mission):
    _, events = mission
    verdict = check(events, "nominal")
    assert verdict["violations"] == [] and verdict["accepted"] == {1: 5, 2: 5}
