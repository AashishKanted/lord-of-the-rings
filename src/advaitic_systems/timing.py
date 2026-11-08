"""Transparent sizing estimate, not a robot benchmark."""
import json
from math import sqrt


def move_time(distance, velocity=.25, acceleration=.5):
    if distance < 0 or velocity <= 0 or acceleration <= 0:
        raise ValueError("Invalid motion limits")
    return (distance / velocity + velocity / acceleration
            if distance >= velocity * velocity / acceleration
            else 2 * sqrt(distance / acceleration))


def budget():
    segments = [.20, .12, .22, .08, .10, .15]
    motion = sum(move_time(d) for d in segments)
    overhead = .7 + .5 + .5 + .25 + 1.0
    return {
        "status": "Engineering estimate; no full robot measured",
        "arm_segments_m": segments,
        "arm_motion_s": round(motion, 3),
        "grip_release_settle_perception_tracking_s": overhead,
        "cycle_with_25_percent_reserve_s": round((motion + overhead) * 1.25, 3),
        "proposed_cycle_s": 12,
        "nominal_kit_interval_s": .45 / .03,
        "nominal_work_window_s": .6 / .03,
        "fastest_test_work_window_s": round(.6 / .035, 3),
        "fifth_kit_inspected_from_first_arrival_s": round(4 * 15 + .85 / .03, 3),
        "two_station_dwell_s": round(2 * (4 * 15 + .85 / .03), 3),
        "slowest_test_two_station_dwell_s": 2 * (4 * .45 / .025 + .85 / .025),
        "travel_dock_budget_s": 45,
        "proposed_mission_limit_s": 270,
    }


def main():
    print(json.dumps(budget(), indent=2))
