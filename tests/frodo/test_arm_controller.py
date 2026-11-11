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
