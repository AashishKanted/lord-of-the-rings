"""Localizer + Navigator closed around the MuJoCo cell: SensorSuite in, ActuatorBridge out.

Base-only navigation (arm stowed, no kits, plant or PLC). The true base pose is read from
the simulator for SCORING ONLY; the robot modules see nothing but `SensorFrame`s. This is
an integration test of navigation, not an acceptance run.
"""
from math import degrees

import pytest

from frodo.apps.nav_demo import MujocoCell, run_mission, summarize
from frodo.cell import CELL
from frodo.robot.navigation import NavStatus


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mujoco_start_to_dock_a_to_dock_b(seed):
    sim = MujocoCell(seed=seed)
    legs, trace = run_mission(seed, sim=sim)
    stats = summarize(trace, legs)
    print(f"\n[MuJoCo, base-only navigation] seed {seed}")
    for leg in legs:
        print(f"  dock {leg.station}: {leg.status.value} in {leg.travel_s:.2f} s, TRUE error "
              f"{leg.true_error_mm:.2f} mm / {leg.true_yaw_error_deg:.3f} deg, estimate error "
              f"{leg.estimate_error_mm:.2f} mm / {leg.estimate_yaw_error_deg:.3f} deg")
    print(f"  EKF max {stats['ekf_pos_max_mm']:.1f} mm / {stats['ekf_yaw_max_deg']:.3f} deg, "
          f"odometry-only max {stats['odo_pos_max_mm']:.1f} mm / "
          f"{stats['odo_yaw_max_deg']:.3f} deg, max speed {stats['max_true_speed_mps']:.3f} m/s, "
          f"max yaw rate {stats['max_true_yaw_rate_radps']:.3f} rad/s, contacts "
          f"{sorted(sim.contacts) or 'none'}")
    robot = CELL["robot"]
    for leg in legs:
        assert leg.status is NavStatus.DOCKED
        assert leg.true_error_mm < 10.0                 # customer limit: 25 mm
        assert leg.true_yaw_error_deg < 0.5             # customer limit: 2 deg
        assert leg.true_yaw_error_deg < degrees(robot["dock_yaw_tolerance_rad"])
        assert leg.travel_s < 20.0
    assert not sim.contacts                             # the robot touched nothing in the cell
    assert stats["max_true_speed_mps"] <= robot["max_transit_speed_mps"]
    assert stats["max_true_yaw_rate_radps"] <= robot["max_yaw_rate_radps"]
    assert stats["min_footprint_clearance_m"] > 0.05
    assert stats["unhealthy_ticks"] == 0
    assert stats["ekf_pos_max_mm"] < 10.0
    # Docked means at rest: the base must be below the assembly speed limit at the end.
    final = trace.true_twist[-1]
    assert abs(final.vx) + abs(final.vy) < robot["assembly_base_speed_max_mps"]
