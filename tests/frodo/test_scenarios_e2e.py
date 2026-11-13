"""End-to-end fault scenarios in MuJoCo (ORACLE kit poses to keep each run to ~100 s; the
fault handling under test does not involve perception). FRODO_SKIP_E2E=1 skips them."""
import json
import os

import pytest

from advaitic_systems.evidence import check
from frodo.apps.mission import run

pytestmark = pytest.mark.skipif(os.environ.get("FRODO_SKIP_E2E") == "1",
                                reason="FRODO_SKIP_E2E=1")


def mission(scenario, tmp_path_factory):
    out = tmp_path_factory.mktemp(scenario)
    summary = run(scenario, out, seed=0, port=0, perception="oracle")
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    return summary, events, out


@pytest.fixture(scope="module")
def human(tmp_path_factory):
    return mission("human_stop", tmp_path_factory)


@pytest.fixture(scope="module")
def retry(tmp_path_factory):
    return mission("plc_retry", tmp_path_factory)


def test_human_stop_passes_the_starter_checker_and_finishes_in_time(human):
    summary, events, _ = human
    verdict = check(events, "human_stop")
    assert verdict["violations"] == [] and verdict["accepted"] == {1: 5, 2: 5}
    assert summary["simulated_s"] <= 360


def test_stop_is_sensor_triggered_fast_and_holds_until_acknowledged(human):
    _, events, _ = human
    t = {e["type"]: e["t"] for e in events
         if e["type"] in ("human_enter", "stop_command", "human_exit", "resume")}
    assert 0 <= t["stop_command"] - t["human_enter"] <= 0.2
    assert t["human_exit"] < t["resume"]
    still = [e for e in events if e["type"] == "sample"
             and t["human_enter"] + 1.0 <= e["t"] <= t["resume"]]
    assert still and max(max(e["base_speed"], e["ee_speed"]) for e in still) <= 0.01
    # nothing was completed while the zone was occupied
    assert not [e for e in events if e["type"] == "assembly_done"
                and t["human_enter"] <= e["t"] <= t["resume"]]


def test_plc_retry_is_recorded_with_real_frames_and_applied_once(retry):
    summary, events, out = retry
    verdict = check(events, "plc_retry")
    assert verdict["violations"] == [] and verdict["accepted"] == {1: 5, 2: 5}
    [event] = [e for e in events if e["type"] == "plc_retry"]
    assert event["controller_applied_count"] == 1
    assert event["accepted_before"] == event["accepted_after"]
    assert event["first_request_hex"][4:] == event["retry_request_hex"][4:]
    trace = [json.loads(line) for line in (out / "plc_socket_trace.jsonl").read_text().splitlines()]
    assert any(row.get("event", "").startswith("acknowledgement dropped") for row in trace)
    ledger = json.loads((out / "plc_ledger.json").read_text())
    sequences = [c["sequence"] for c in ledger]
    assert len(sequences) == len(set(sequences))          # no command applied twice
