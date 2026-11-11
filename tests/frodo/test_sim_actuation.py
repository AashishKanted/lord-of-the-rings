"""Actuator bridge: twist limits, base tracking, arm/gripper pass-through, grasp gating."""
from math import pi

import mujoco
import numpy as np
import pytest

from frodo.cell import CELL, STATIONS
from frodo.config import get_config
from frodo.interfaces import RobotCommand, Twist2D
from frodo.robot.description import ARM_HOME, ARM_STOWED
from frodo.sim.actuation import ActuatorBridge

CFG = get_config()
ACT = CFG.actuation
DT = CFG.sim.control_dt_s


def test_speed_limits(empty_model):
    bridge = ActuatorBridge(empty_model)
    for _ in range(1000):
        bridge.twist = bridge.limit(Twist2D(3.0, 4.0, -9.0), DT)
    assert np.hypot(bridge.twist.vx, bridge.twist.vy) == pytest.approx(ACT.max_linear_mps)
    assert bridge.twist.vx / bridge.twist.vy == pytest.approx(3 / 4)      # direction kept
    assert bridge.twist.wz == pytest.approx(-ACT.max_angular_radps)


def test_acceleration_limits(empty_model):
    bridge = ActuatorBridge(empty_model)
    first = bridge.limit(Twist2D(vx=1.0, wz=1.0), DT)
    assert first.vx == pytest.approx(ACT.max_linear_accel_mps2 * DT)
    assert first.wz == pytest.approx(ACT.max_angular_accel_radps2 * DT)
    history = []
    for _ in range(200):
        bridge.twist = bridge.limit(Twist2D(vx=0.3, vy=-0.2, wz=0.5), DT)
        history.append((bridge.twist.vx, bridge.twist.vy, bridge.twist.wz))
    for _ in range(200):
        bridge.twist = bridge.limit(Twist2D(), DT)
        history.append((bridge.twist.vx, bridge.twist.vy, bridge.twist.wz))
    h = np.array([(0, 0, 0), *history])
    d = np.diff(h, axis=0)
    assert np.hypot(d[:, 0], d[:, 1]).max() <= ACT.max_linear_accel_mps2 * DT + 1e-12
    assert np.abs(d[:, 2]).max() <= ACT.max_angular_accel_radps2 * DT + 1e-12
    assert h[200] == pytest.approx([0.3, -0.2, 0.5])
    assert h[-1] == pytest.approx([0, 0, 0], abs=1e-12)


def test_base_follows_commanded_velocity_within_limits(empty_model, fresh, step):
    data = fresh(empty_model)
    bridge = ActuatorBridge(empty_model)
    speeds = []

    def sample():
        speeds.append((np.hypot(*data.qvel[:2]), abs(data.qvel[2])))

    step(empty_model, data, bridge, RobotCommand(Twist2D(vx=5.0, wz=5.0)), 3.0, on_tick=sample)
    s = np.array(speeds)
    assert s[:, 0].max() < 1.1 * ACT.max_linear_mps
    assert s[:, 1].max() < 1.15 * ACT.max_angular_radps
    assert s[-1, 0] > 0.9 * ACT.max_linear_mps


def test_velocity_tracking_and_stop_accuracy(empty_model, fresh, step):
    """Mean base velocity equals the command; after stopping the base sits on the integrated
    target to well under a millimetre."""
    start = np.array(CELL["start_pose_world"])
    for twist, axis in ((Twist2D(vx=0.3), 0), (Twist2D(vy=0.3), 1), (Twist2D(wz=0.5), 2)):
        data = fresh(empty_model)
        bridge = ActuatorBridge(empty_model)
        step(empty_model, data, bridge, RobotCommand(twist), 1.0)
        before = data.qpos[:3].copy()
        step(empty_model, data, bridge, RobotCommand(twist), 1.0)
        rate = (data.qpos[:3] - before)[axis]
        assert rate == pytest.approx([twist.vx, twist.vy, twist.wz][axis], rel=1e-3)
        assert np.abs(data.qpos[:3] - bridge.target_pose).max() < 0.005
        step(empty_model, data, bridge, RobotCommand(), 1.5)
        assert np.abs(data.qpos[:3] - bridge.target_pose).max() < 1e-5
        assert np.abs(data.qvel[:3]).max() < 1e-5
        moved = data.qpos[:3] - start
        assert moved[axis] == pytest.approx(2.0 * [twist.vx, twist.vy, twist.wz][axis], rel=1e-3)
        assert np.abs(np.delete(moved, axis)).max() < 1e-5


def test_twist_is_interpreted_in_the_base_frame(empty_model, fresh, step):
    data = fresh(empty_model, (3.0, 3.0, pi / 2))
    bridge = ActuatorBridge(empty_model)
    step(empty_model, data, bridge, RobotCommand(Twist2D(vx=0.2)), 1.0)
    step(empty_model, data, bridge, RobotCommand(), 1.0)
    assert data.qpos[1] - 3.0 == pytest.approx(0.2, abs=1e-3)   # forward = world +y here
    assert data.qpos[0] == pytest.approx(3.0, abs=1e-4)


def test_bridge_only_writes_controls(empty_model, fresh):
    data = fresh(empty_model)
    bridge = ActuatorBridge(empty_model)
    qpos, qvel = data.qpos.copy(), data.qvel.copy()
    bridge.apply(data, RobotCommand(Twist2D(0.3, 0.1, 0.2), arm_q=ARM_HOME, gripper=0.5,
                                    request_grasp=True))
    assert np.array_equal(qpos, data.qpos) and np.array_equal(qvel, data.qvel)


def test_blocked_base_does_not_wind_up(empty_model, fresh, step):
    """Driving into conveyor A: the servo target stays within the following-error limit."""
    data = fresh(empty_model, (3.0, 1.85, -pi / 2))
    bridge = ActuatorBridge(empty_model)
    step(empty_model, data, bridge, RobotCommand(Twist2D(vx=0.3)), 3.0)
    assert data.qpos[1] > 1.6                                   # the conveyor stopped it
    gap = np.hypot(*(bridge.target_pose[:2] - data.qpos[:2]))
    assert gap <= ACT.max_following_error_m + 1e-6


def test_arm_and_gripper_pass_through(empty_model, fresh):
    data = fresh(empty_model)
    bridge = ActuatorBridge(empty_model)
    arm_ids = [empty_model.actuator(f"joint_{i}").id for i in range(1, 8)]
    grip = empty_model.actuator("fingers_actuator").id
    bridge.apply(data, RobotCommand())                          # None = hold
    assert np.allclose(data.ctrl[arm_ids], ARM_STOWED)
    assert data.ctrl[grip] == 0
    bridge.apply(data, RobotCommand(arm_q=ARM_HOME, gripper=0.5))
    assert np.allclose(data.ctrl[arm_ids], ARM_HOME)
    assert data.ctrl[grip] == pytest.approx(127.5)
    wild = np.full(7, 10.0)
    bridge.apply(data, RobotCommand(arm_q=wild, gripper=7.0))
    assert data.ctrl[grip] == 255
    assert data.ctrl[arm_ids[1]] == pytest.approx(empty_model.actuator_ctrlrange[arm_ids[1], 1])


def test_grasp_refused_when_not_holding_a_ring(kit_model, fresh, step):
    data = fresh(kit_model, STATIONS["A"].dock, ARM_HOME)
    bridge = ActuatorBridge(kit_model)
    weld = kit_model.equality("kit_A0_grasp").id
    # Gripper open: refused for that reason.
    step(kit_model, data, bridge, RobotCommand(arm_q=ARM_HOME, gripper=0.0, request_grasp=True),
         0.1)
    assert data.eq_active[weld] == 0 and bridge.holding is None
    assert "gripper commanded" in bridge.last_grasp_refusal
    # Gripper closed on thin air, far from the ring: still refused.
    step(kit_model, data, bridge, RobotCommand(arm_q=ARM_HOME, gripper=1.0, request_grasp=True),
         1.0)
    assert data.eq_active[weld] == 0 and bridge.holding is None
    assert "not both in contact" in bridge.last_grasp_refusal
    step(kit_model, data, bridge, RobotCommand(arm_q=ARM_HOME, gripper=1.0), 0.05)
    assert bridge.last_grasp_refusal is None


def test_grasp_engages_on_a_physical_pinch_and_releases(kit_model, fresh, step):
    """Close the fingers on the ring wall with the arm (IK from the manipulation module),
    then the weld is granted; opening the gripper or asking for release drops it."""
    from frodo.robot.manipulation.ik import WALL_MID, ArmIK, tcp_in_site, tool_rotation
    st = STATIONS["A"]
    data = fresh(kit_model, st.dock, ARM_HOME)
    bridge = ActuatorBridge(kit_model)
    weld = kit_model.equality("kit_A0_grasp").id
    ring = kit_model.body("kit_A0_ring").id
    rot = tool_rotation(st)
    ik = ArmIK(kit_model, data.qpos, rot)
    grasp_point = data.xpos[ring] + np.array([WALL_MID, 0, 0.001])
    preshape = CFG.manipulation.grip_preshape / 255
    q_above = ik.solve_static(grasp_point + [0, 0, 0.05])
    step(kit_model, data, bridge, RobotCommand(arm_q=q_above, gripper=preshape), 2.0)
    q_grasp = ik.solve_static(grasp_point)
    step(kit_model, data, bridge, RobotCommand(arm_q=q_grasp, gripper=preshape,
                                               request_grasp=True), 2.0)
    assert bridge.holding is None                      # at the ring but fingers not closed
    site = kit_model.site("pinch_site").id
    tcp = data.site_xpos[site] + data.site_xmat[site].reshape(3, 3) @ tcp_in_site(rot)
    assert np.linalg.norm(tcp - grasp_point) < 0.006, "test setup: tool did not reach the ring"
    step(kit_model, data, bridge, RobotCommand(arm_q=q_grasp, gripper=1.0, request_grasp=True),
         1.0)
    assert bridge.holding == "kit_A0_ring" and data.eq_active[weld] == 1
    # Lift: the ring comes along.
    z0 = data.xpos[ring][2]
    step(kit_model, data, bridge, RobotCommand(arm_q=q_above, gripper=1.0), 1.5)
    assert data.xpos[ring][2] > z0 + 0.03
    # Opening the gripper drops the weld without an explicit release.
    step(kit_model, data, bridge, RobotCommand(arm_q=q_above, gripper=preshape), 0.05)
    assert bridge.holding is None and data.eq_active[weld] == 0


def test_release_request_drops_the_weld(kit_model, fresh):
    """Unit check of the release path with the weld state forced (no physics)."""
    data = fresh(kit_model, STATIONS["A"].dock, ARM_HOME)
    bridge = ActuatorBridge(kit_model)
    weld = kit_model.equality("kit_A0_grasp").id
    bridge.apply(data, RobotCommand(gripper=1.0))
    bridge._held_weld = weld
    data.eq_active[weld] = 1
    bridge.apply(data, RobotCommand(gripper=1.0, release_grasp=True))
    assert bridge.holding is None and data.eq_active[weld] == 0


def test_reset_resyncs_targets(empty_model, fresh, step):
    data = fresh(empty_model)
    bridge = ActuatorBridge(empty_model)
    step(empty_model, data, bridge, RobotCommand(Twist2D(vx=0.3)), 0.5)
    from frodo.sim.scene import reset
    reset(empty_model, data, (3.0, 3.0, 1.0))
    bridge.apply(data, RobotCommand())          # time went backwards -> bridge re-syncs
    assert bridge.target_pose == pytest.approx([3.0, 3.0, 1.0])
    mujoco.mj_step(empty_model, data)
    assert data.qpos[:3] == pytest.approx([3.0, 3.0, 1.0], abs=1e-6)
