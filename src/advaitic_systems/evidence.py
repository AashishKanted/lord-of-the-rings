"""Consistency checks on exported evidence. Clean simulation reruns remain required."""
from __future__ import annotations

import argparse
import json
import struct
from math import atan2, cos, hypot, isfinite, sin
from pathlib import Path

from .contract import inspection_passes

DOCKS = {1: (3., 2., -1.5707963267948966), 2: (3., 4., 1.5707963267948966)}
EVENT_TOLERANCE_S = .11  # crossings may be reported at 10 Hz


def _completion_request(hex_frame):
    if not isinstance(hex_frame, str) or len(hex_frame) != 42:
        raise ValueError("PLC retry needs a complete Modbus/TCP request frame")
    frame = bytes.fromhex(hex_frame)
    transaction, protocol, length, unit = struct.unpack(">HHHB", frame[:7])
    function, address, count, byte_count, sequence, station, kit_id, command = struct.unpack(
        ">BHHB4H", frame[7:])
    if (protocol, length, unit, function, address, count, byte_count, command) != (
            0, 15, 1, 16, 100, 4, 8, 3):
        raise ValueError("PLC retry frame is not an ASSEMBLY_DONE write")
    return transaction, sequence, station, kit_id


def check(events, scenario="nominal"):
    """Return violations. All timestamps are simulation seconds from mission start."""
    errors, kits, accepted, samples = [], {}, {1: 0, 2: 0}, []
    last_t, docked, dock_station, human = -1., False, None, None
    hazards, previous_sample = [], None
    last_arrival = {1: None, 2: None}
    saw_retry = False
    for index, event in enumerate(events):
        try:
            t = float(event["t"])
            if not isfinite(t) or t < 0 or t < last_t:
                raise ValueError("timestamps must be finite, nonnegative and ordered")
            last_t = t
            event = {**event, "t": t}
            kind = event["type"]
            if kind == "sample":
                values = [float(event[k]) for k in ("x", "y", "yaw", "base_speed", "ee_speed")]
                if not all(isfinite(v) for v in values):
                    raise ValueError("nonfinite motion sample")
                x, y, yaw, base_speed, ee_speed = values
                event.update(zip(("x", "y", "yaw", "base_speed", "ee_speed"), values, strict=True))
                if not (0 <= x <= 6 and 0 <= y <= 6):
                    errors.append("Robot left the cell")
                if base_speed < 0 or ee_speed < 0:
                    errors.append("Speed must be a nonnegative magnitude")
                if base_speed > .351:
                    errors.append("Transit speed exceeded")
                if base_speed > .01 and event["arm_stowed"] is not True:
                    errors.append("Arm not stowed during base movement")
                if previous_sample is not None:
                    dt = t - previous_sample["t"]
                    if dt <= 0 or dt > .101:
                        errors.append("Motion samples must be at least 10 Hz")
                    if hypot(x - previous_sample["x"], y - previous_sample["y"]) > .35 * dt + .003:
                        errors.append("Base displacement exceeds motion limit")
                    angle = abs(atan2(sin(yaw - previous_sample["yaw"]), cos(yaw - previous_sample["yaw"])))
                    if angle > .6 * dt + .003:
                        errors.append("Base yaw rate exceeded")
                if human is not None and t >= human["enter"] + 1.0:
                    if base_speed > .01 or ee_speed > .01:
                        errors.append("Motion continued in occupied human zone")
                previous_sample = {"t": t, "x": x, "y": y, "yaw": yaw}
                samples.append(event)
                if base_speed > .01:
                    docked = False
            elif kind == "dock":
                station = event["station"]
                expected = DOCKS[station]
                x, y, yaw = [float(event[k]) for k in ("x", "y", "yaw")]
                if not all(isfinite(v) for v in (x, y, yaw)):
                    raise ValueError("Nonfinite dock pose")
                yaw_error = atan2(sin(yaw - expected[2]), cos(yaw - expected[2]))
                if hypot(x - expected[0], y - expected[1]) > .025 or abs(yaw_error) > .034906586:
                    errors.append("Dock outside tolerance")
                if previous_sample is None or t - previous_sample["t"] > .101:
                    errors.append("Dock has no recent motion sample")
                elif (hypot(x - previous_sample["x"], y - previous_sample["y"]) > .025
                      or float(samples[-1]["base_speed"]) > .01):
                    errors.append("Dock inconsistent with measured base pose or speed")
                if station == 2 and accepted[1] != 5:
                    errors.append("Switched before five accepted at A")
                docked, dock_station = True, station
            elif kind == "kit_arrived":
                station, kit_id = event["station"], event["kit_id"]
                if (type(station) is not int or station not in DOCKS
                        or type(kit_id) is not int or not 1 <= kit_id <= 65535
                        or kit_id in kits):
                    raise ValueError("Invalid station or duplicate kit")
                if station == 2 and accepted[1] != 5:
                    errors.append("Station B feeder started before five accepted at A")
                speed = float(event["belt_speed_mps"])
                if not .025 <= speed <= .035:
                    raise ValueError("Invalid arrival belt speed")
                previous = last_arrival[station]
                if previous is not None:
                    if abs(speed - previous["speed"]) > 1e-6:
                        errors.append("Belt speed changed within a station run")
                    if t - previous["t"] < .45 / speed - EVENT_TOLERANCE_S:
                        errors.append("Kit arrivals violated 450 mm pitch")
                last_arrival[station] = {"t": t, "speed": speed}
                in_flight = sum(k["station"] == station and not k["inspected"]
                                for k in kits.values())
                if accepted[station] + in_flight >= 5:
                    errors.append("Feeder exceeded five accepted plus in-flight kits")
                kits[kit_id] = {"station": station, "arrived": t, "done": None,
                                "speed": speed, "seat": None, "inspected": False}
            elif kind == "assembly_done":
                kit = kits[event["kit_id"]]
                if kit["station"] != event["station"] or kit["done"] is not None:
                    raise ValueError("Wrong station or duplicate completion")
                if not docked or dock_station != kit["station"] or human is not None:
                    errors.append("Assembly completed without a valid clear dock")
                kit["done"] = t
                if t - kit["arrived"] > .6 / kit["speed"] + .02:
                    errors.append("Assembly completed after work window")
            elif kind == "seat_verified":
                kit = kits[event["kit_id"]]
                if (kit["station"] != event["station"] or kit["seat"] is not None
                        or kit["inspected"]):
                    raise ValueError("Wrong station or duplicate/late seat verification")
                if not inspection_passes(event):
                    errors.append("Seat verification failed geometry or hold")
                hold = float(event["hold_s"])
                if hold > t - kit["arrived"] + EVENT_TOLERANCE_S:
                    errors.append("Seat hold began before kit work entry")
                if t - kit["arrived"] > .6 / kit["speed"] + .02:
                    errors.append("Seat verification occurred after work window")
                kit["seat"] = t
            elif kind == "inspection":
                kit = kits[event["kit_id"]]
                if kit["station"] != event["station"] or kit["inspected"]:
                    raise ValueError("Wrong station or duplicate inspection")
                speed = float(event["belt_speed_mps"])
                if not .025 <= speed <= .035:
                    raise ValueError("Belt speed outside declared test range")
                if abs(speed - kit["speed"]) > 1e-6:
                    errors.append("Inspection belt speed differs from arrival")
                # Bottle, not kit centre, defines entry and inspection crossings.
                if abs(t - kit["arrived"] - .85 / kit["speed"]) > EVENT_TOLERANCE_S:
                    errors.append("Inspection did not occur at downstream crossing")
                kit["inspected"] = True
                if kit["done"] is not None and kit["seat"] is not None and inspection_passes(event):
                    accepted[kit["station"]] += 1
                    if scenario == "nominal" and kit["seat"] - kit["arrived"] > 12.0001:
                        errors.append("Assembly cycle missed")
                elif kit["done"] is not None and kit["seat"] is None and inspection_passes(event):
                    errors.append("Accepted inspection lacks measured seat verification")
            elif kind == "human_enter":
                if human is not None:
                    raise ValueError("Nested human entry")
                if not any(kit["arrived"] <= t and kit["done"] is None for kit in kits.values()):
                    errors.append("Human interruption outside an active kit")
                human = {"enter": t, "stop": None, "clear": None, "resume": None}
                hazards.append(human)
            elif kind == "stop_command":
                if human is None:
                    raise ValueError("Stop command without human episode")
                if human["stop"] is None:
                    human["stop"] = t
            elif kind == "human_exit":
                if human is None or human["clear"] is not None:
                    raise ValueError("Human exit without entry or duplicate exit")
                human["clear"] = t
            elif kind == "resume":
                if human is None or human["clear"] is None:
                    raise ValueError("Resume before human cleared")
                human["resume"] = t
                human = None
            elif kind == "plc_retry":
                first = _completion_request(event["first_request_hex"])
                retry = _completion_request(event["retry_request_hex"])
                if first[1:] != retry[1:]:
                    errors.append("PLC retry did not preserve sequence and payload")
                kit = kits.get(first[3])
                if (kit is None or kit["station"] != first[2] or kit["done"] is None
                        or first[1] == 0):
                    errors.append("PLC retry did not reference a completed kit")
                if (type(event["controller_applied_count"]) is not int
                        or event["controller_applied_count"] != 1):
                    errors.append("PLC completion was not applied exactly once")
                if event["accepted_before"] != event["accepted_after"]:
                    errors.append("PLC retry changed acceptance count")
                saw_retry = True
            else:
                raise ValueError(f"Unknown event type: {kind}")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            errors.append(f"Event {index}: {exc}")
    if accepted != {1: 5, 2: 5}:
        errors.append(f"Expected five accepted per station, got {accepted}")
    if scenario == "nominal" and last_t > 270.0001:
        errors.append("Nominal mission exceeded 270 s")
    if scenario != "nominal" and last_t > 360.0001:
        errors.append("Fault mission exceeded 360 s")
    if not samples or float(samples[0]["t"]) > .1 or last_t - float(samples[-1]["t"]) > .101:
        errors.append("Motion samples do not cover mission")
    if samples:
        first = samples[0]
        if hypot(float(first["x"]) - 1., float(first["y"]) - 3.) > .025:
            errors.append("Incorrect initial robot pose")
    for hazard in hazards:
        if hazard["stop"] is None or hazard["stop"] - hazard["enter"] > .2001:
            errors.append("Human stop command missed 200 ms deadline")
        if hazard["resume"] is None:
            errors.append("Human episode has no explicit resume")
        if (scenario == "human_stop" and hazard["clear"] is not None
                and not 2 <= hazard["clear"] - hazard["enter"] <= 4):
            errors.append("Human interruption must last 2–4 s")
        deadline = hazard["enter"] + 1.
        candidates = [s for s in samples if deadline <= s["t"] <= deadline + .101]
        if not candidates or any(s["base_speed"] > .01 or s["ee_speed"] > .01 for s in candidates):
            errors.append("Motion did not settle within one second")
    if scenario == "human_stop" and not hazards:
        errors.append("Missing human interruption")
    if scenario == "plc_retry" and not saw_retry:
        errors.append("Missing PLC retry evidence")
    return {"passed": not errors, "accepted": accepted, "violations": sorted(set(errors)),
            "limitation": "Export consistency only; independently rerun simulation and inspect measurement bridge."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--scenario", choices=["nominal", "human_stop", "plc_retry"], default="nominal")
    args = parser.parse_args()
    try:
        events = [json.loads(line) for line in args.events.read_text().splitlines() if line.strip()]
        result = check(events, args.scenario)
    except (OSError, ValueError) as exc:
        result = {"passed": False, "violations": [str(exc)]}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
