"""Customer-side evaluator: checks a run against the simulator's own state.

The starter `systems-check` only checks that the evidence log is consistent. This module
looks at the physics directly, on the plant side, and never touches robot code:

- collisions: every contact between the robot and anything else, by pair. The only
  expected ones are the gripper fingers on a ring.
- belt: the belts never pause or slow down: every kit on a belt stays within
  ``BELT_POSITION_TOLERANCE_M`` of where the belt's own travel puts it, and arrivals are at
  least one pitch apart. The largest momentary speed deviation is reported too (the gripper
  closing on a ring nudges its kit for a fraction of a second; the carrier pulls it back).
- seating hold: the 0.5 s hold is recomputed from the raw per-step seat samples and must
  give the same times as the logged ``seat_verified`` events.
- stale work: every kit that was not seated inside its work window was rejected, and every
  accepted kit was seated inside it.
- recovery: after a human stop, the robot moved again and finished; after a PLC fault, the
  retried command was applied once.

    evaluator = Evaluator(model, line)
    every control tick:  evaluator.tick(data)
    at the end:          evaluator.finish(events, kits, seat_samples_path, scenario)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from advaitic_systems.contract import inspection_passes

from ..cell import CELL
from .inspection import HOLD_S, SETTLED_ANG_RADPS, SETTLED_LIN_MPS

BELT_POSITION_TOLERANCE_M = 0.001
GRIPPER_BODY = "base"          # Robotiq 2F-85 palm; its subtree is the gripper


class Evaluator:
    def __init__(self, model, line):
        self.model, self.line = model, line
        robot_root = model.body("base_link").id
        self._robot = {b for b in range(model.nbody) if model.body_rootid[b] == robot_root}
        palm = model.body(GRIPPER_BODY).id
        self._gripper = {b for b in self._robot if self._descends(b, palm)}
        self._rings = {b for b in range(model.nbody) if model.body(b).name.endswith("_ring")}
        self.contacts: Counter = Counter()        # (robot part, other body) -> control ticks
        self.belt_error_max = 0.0                  # speed, m/s
        self.belt_error_at: tuple[float, str] | None = None
        self.belt_position_error_max = 0.0         # m, kit vs the belt's own travel
        self.belt_samples = 0

    def _descends(self, body: int, ancestor: int) -> bool:
        while body != ancestor and body != 0:
            body = int(self.model.body_parentid[body])
        return body == ancestor

    def tick(self, data) -> None:
        m = self.model
        for g1, g2 in data.contact.geom[:data.ncon]:
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if (b1 in self._robot) == (b2 in self._robot):
                continue
            robot, other = (b1, b2) if b1 in self._robot else (b2, b1)
            if robot in self._gripper and other in self._rings:
                continue                           # fingers on a ring: the job
            part = "gripper" if robot in self._gripper else m.body(robot).name
            self.contacts[(part, m.body(other).name)] += 1
        for belt in self.line._belts.values():
            for carrier in belt.pool:
                if carrier.on_belt:
                    error = abs(float(data.qvel[carrier.slide_dof]) - belt.speed)
                    if error > self.belt_error_max:
                        self.belt_error_max, self.belt_error_at = error, (round(float(
                            data.time), 3), carrier.name)
                    on_belt = self.line._release_x + belt.travel - carrier.release_travel
                    self.belt_position_error_max = max(self.belt_position_error_max,
                                                       abs(carrier.x(data) - on_belt))
                    self.belt_samples += 1

    # ---- verdicts ----
    def finish(self, events: list[dict], kits: list[dict], seat_samples: Path | None,
               scenario: str) -> dict:
        checks = {
            "collisions": self._collisions(),
            "belt_motion": self._belt(events),
            "seating_hold": self._hold(events, seat_samples),
            "stale_work_rejected": self._stale(kits),
        }
        if scenario == "human_stop":
            checks["recovery_after_human_stop"] = self._human(events)
        if scenario == "plc_retry":
            checks["plc_retry_applied_once"] = self._retry(events)
        return {"passed": all(c["passed"] for c in checks.values()), "checks": checks,
                "note": "computed from simulator state by the plant side, not by robot code"}

    def _collisions(self) -> dict:
        pairs = {f"{a} <-> {b}": n for (a, b), n in sorted(self.contacts.items())}
        return {"passed": not pairs, "unexpected_contacts_control_ticks": pairs,
                "allowed": "gripper fingers touching a ring",
                "sampling": "contacts checked every 10 ms control tick"}

    def _belt(self, events: list[dict]) -> dict:
        speeds = {e["station"]: e["belt_speed_mps"] for e in events if e["type"] == "kit_arrived"}
        pitch_ok, pitch = True, CELL["conveyor"]["kit_pitch_m"]
        for station, speed in speeds.items():
            times = [e["t"] for e in events if e["type"] == "kit_arrived" and e["station"] == station]
            gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
            pitch_ok &= all(g >= pitch / speed - 0.011 for g in gaps)
        ok = self.belt_position_error_max <= BELT_POSITION_TOLERANCE_M and pitch_ok
        return {"passed": ok,
                "kit_position_error_max_m": round(self.belt_position_error_max, 6),
                "tolerance_m": BELT_POSITION_TOLERANCE_M,
                "kit_speed_deviation_max_mps": round(self.belt_error_max, 6),
                "speed_deviation_worst_at": self.belt_error_at, "samples": self.belt_samples,
                "arrivals_at_least_one_pitch_apart": pitch_ok,
                "declared_speed_mps": speeds}

    def _hold(self, events: list[dict], path: Path | None) -> dict:
        if path is None or not Path(path).exists():
            return {"passed": False, "why": "no raw seat samples"}
        step = self.model.opt.timestep
        holds: dict[int, float] = {}
        first: dict[int, float] = {}
        last_t: dict[int, float] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                kit, t = row["kit_id"], row["t"]
                valid = (row["rel_lin_mps"] <= SETTLED_LIN_MPS
                         and row["rel_ang_radps"] <= SETTLED_ANG_RADPS
                         and inspection_passes({**row, "hold_s": HOLD_S}))
                continuous = kit in last_t and abs(t - last_t[kit] - step) < step / 2
                holds[kit] = (holds.get(kit, 0.0) if continuous else 0.0) + step if valid else 0.0
                last_t[kit] = t
                if kit not in first and holds[kit] >= HOLD_S - 1e-9:
                    first[kit] = t
        logged = {e["kit_id"]: e["t"] for e in events if e["type"] == "seat_verified"}
        mismatch = {k: (logged.get(k), first.get(k)) for k in set(logged) | set(first)
                    if logged.get(k) is None or first.get(k) is None
                    or abs(logged[k] - first[k]) > step}
        return {"passed": not mismatch, "kits_recomputed": len(first),
                "mismatches_logged_vs_recomputed": mismatch}

    def _stale(self, kits: list[dict]) -> dict:
        bad = [k["kit_id"] for k in kits if k["inspected_t"] is not None
               and bool(k["accepted"]) != bool(k["seat_in_window"])]
        given_up = [k["kit_id"] for k in kits if k["inspected_t"] is not None
                    and not k["accepted"]]
        return {"passed": not bad, "rejected_kits": given_up,
                "accepted_without_in_window_seat_or_rejected_with_one": bad}

    def _human(self, events: list[dict]) -> dict:
        t = {e["type"]: e["t"] for e in events
             if e["type"] in ("human_enter", "stop_command", "human_exit", "resume")}
        if len(t) < 4:
            return {"passed": False, "why": f"missing events, have {sorted(t)}"}
        moved_after = any(e["type"] == "sample" and e["t"] > t["resume"] and e["ee_speed"] > 0.01
                          for e in events)
        done_after = any(e["type"] == "assembly_done" and e["t"] > t["resume"] for e in events)
        latency = t["stop_command"] - t["human_enter"]
        return {"passed": moved_after and done_after and latency <= 0.2,
                "stop_latency_s": round(latency, 3), "robot_moved_after_resume": moved_after,
                "assemblies_after_resume": done_after}

    def _retry(self, events: list[dict]) -> dict:
        retries = [e for e in events if e["type"] == "plc_retry"]
        ok = len(retries) == 1 and retries[0]["controller_applied_count"] == 1 \
            and retries[0]["accepted_before"] == retries[0]["accepted_after"]
        return {"passed": ok, "plc_retry_events": len(retries)}
