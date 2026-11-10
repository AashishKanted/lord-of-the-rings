"""What the robot knows about its own body: joint names and named arm postures."""
from __future__ import annotations

import numpy as np

BASE_JOINTS = ("joint_x", "joint_y", "joint_th")
ARM_JOINTS = tuple(f"joint_{i}" for i in range(1, 8))
GRIPPER_ACTUATOR = "fingers_actuator"
TOOL_SITE = "pinch_site"
# Arm configurations copied from the Menagerie keyframes (keyframes themselves are
# dropped from the scene because adding kits changes the size of qpos).
ARM_HOME = np.array([0, 0.26179939, 3.14159265, -2.26892803, 0, 0.95993109, 1.57079633])
ARM_STOWED = np.array([0, -0.34906585, 3.14159265, -2.54818071, 0, -0.87266463, 1.57079633])
