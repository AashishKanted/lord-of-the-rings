"""Documented grasp constraint (README "What you may simplify").

A weld between the gripper and a ring. The robot may only request it after a valid
physical grasp (both pads touching, tool at the grasp pose, fingers closed), and it
must be released when the gripper opens.
"""
from __future__ import annotations

import mujoco
import numpy as np


def activate_grasp_weld(model, data, eq_id):
    """Freeze the current gripper<-ring relative pose into the weld, then enable it."""
    b1, b2 = model.eq_obj1id[eq_id], model.eq_obj2id[eq_id]
    neg, rel_quat, rel_pos = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_negQuat(neg, data.xquat[b1])
    mujoco.mju_mulQuat(rel_quat, neg, data.xquat[b2])
    mujoco.mju_rotVecQuat(rel_pos, data.xpos[b2] - data.xpos[b1], neg)
    model.eq_data[eq_id, 0:3] = 0          # anchor at the ring origin
    model.eq_data[eq_id, 3:6] = rel_pos    # ring pose in the gripper frame
    model.eq_data[eq_id, 6:10] = rel_quat
    data.eq_active[eq_id] = 1


def release_grasp_weld(data, eq_id):
    data.eq_active[eq_id] = 0
