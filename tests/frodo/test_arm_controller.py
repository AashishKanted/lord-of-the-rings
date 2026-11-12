"""Arm controller pieces that need no physics: joint-space moves, station setup, stow logic."""
import numpy as np
import pytest

from frodo.cell import CELL, KIT_TOP_Z, STATIONS
from frodo.config import get_config
from frodo.interfaces import JointState, Pose2D
from frodo.robot.description import ARM_HOME, ARM_STOWED
from frodo.robot.frames import StationFrame
from frodo.robot.manipulation.arm_controller import ArmController, ArmPhase
from frodo.robot.manipulation.motion import JointMove

CFG = get_config()
DT = CFG.sim.control_dt_s


def test_joint_move_passes_through_the_via_point_within_the_speed_limit():
    move = JointMove([ARM_STOWED, ARM_HOME, ARM_STOWED], max_speed=1.0)
    points = [ARM_STOWED.copy()]
    while not move.done:
        points.append(move.step(DT))
    points = np.array(points)
    assert points[-1] == pytest.approx(ARM_STOWED)
    assert np.min(np.max(np.abs(points - ARM_HOME), axis=1)) < 0.02     # reached the via point
    assert np.max(np.abs(np.diff(points, axis=0))) / DT <= 1.0 + 1e-6
    # starts and ends at rest
    assert np.max(np.abs(points[1] - points[0])) / DT < 0.05


@pytest.fixture(scope="module")
def controller(empty_model):
    arm = ArmController(empty_model, CFG)
    dock = Pose2D(*(float(v) for v in STATIONS["A"].dock))
    arm.set_station(StationFrame.from_pose("A", dock))
    return arm


def test_waiting_pose_is_reachable_and_clears_the_bottles(controller):
    arm = controller
    assert arm.ik.tcp() == pytest.approx(arm.hover, abs=1e-3)           # IK converged on it
    assert arm.fk.tcp(arm.q_ready) == pytest.approx(arm.hover, abs=1e-3)
    assert arm.hover[2] > KIT_TOP_Z + CELL["bottle"]["height_m"] + 0.03
    assert arm.hover[0] == pytest.approx(0.8 - CELL["kit"]["ring_xy_in_kit_m"][1])   # ring lane


def test_unstow_then_stow_runs_through_the_phases(controller):
    arm = controller
    assert arm.stowed and not arm.waiting
    with pytest.raises(RuntimeError):
        arm.begin(1, 0.0)                  # no cycle from the stowed pose
    arm.unstow()
    t, q = 0.0, ARM_STOWED
    while not arm.waiting and t < 15:
        q = arm.step(JointState(t, q, 0.47), None, t, DT).arm_q
        t += DT
    assert arm.phase is ArmPhase.WAITING and q == pytest.approx(arm.q_ready, abs=1e-6)
    held = arm.step(JointState(t, q, 0.47), None, t, DT)          # waits at the hover pose
    assert held.arm_q == pytest.approx(arm.q_ready, abs=1e-3) and not held.request_grasp
    arm.stow()
    while not arm.stowed and t < 30:
        q = arm.step(JointState(t, q, 0.47), None, t, DT).arm_q
        t += DT
    assert arm.stowed and q == pytest.approx(ARM_STOWED)
    with pytest.raises(RuntimeError):
        arm.begin(1, t)


def test_tool_target_is_kept_inside_the_workspace_box(controller):
    arm, m = controller, CFG.manipulation
    inside = arm.hover.copy()
    assert arm._limit(inside) == pytest.approx(inside)
    into_the_belt = arm._limit(inside - [0, 0, 0.5])
    assert into_the_belt[2] == pytest.approx(KIT_TOP_Z - m.workspace_below_kit_top_m)
    assert into_the_belt[:2] == pytest.approx(inside[:2])
    far_side = arm._limit(inside + [0.6, 0, 0])                 # beyond the belt, away from the robot
    assert arm.station.center[0] - far_side[0] == pytest.approx(m.workspace_local_y_m[0])
    downstream = arm._limit(inside + [0, 2.0, 0])
    assert arm.station.local_x(downstream) == pytest.approx(m.workspace_local_x_m[1])


def test_blocked_tool_aborts_the_cycle(controller):
    from frodo.robot.perception import KitObservation
    arm = controller
    arm.unstow()
    t, q = 0.0, ARM_STOWED
    while not arm.waiting:
        q = arm.step(JointState(t, q, 0.0), None, t, DT).arm_q
        t += DT
    bottle = arm.station.point(-0.30, 0.045, KIT_TOP_Z)
    obs = KitObservation(t, 1, bottle + [0, 0.14, 0.004], bottle, np.zeros(3), 0.0, "test")
    arm.begin(1, t)
    stuck = q.copy()                      # the joints never follow: something holds the arm
    for _ in range(200):
        arm.step(JointState(t, stuck, 0.0), obs, t, DT)
        t += DT
        if arm.phase is ArmPhase.RETURN:
            break
    assert arm.phase is ArmPhase.RETURN and "blocked" in arm.cycle.failure
    assert arm.cycle.grasped_t is None
