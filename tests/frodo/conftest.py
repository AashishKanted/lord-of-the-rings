"""Shared fixtures for the simulator tests. Sets the off-screen GL backend before MuJoCo
is first imported (it is read at import time)."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import pytest  # noqa: E402

mujoco = pytest.importorskip("mujoco")

from frodo.sim.scene import build_spec, reset  # noqa: E402


@pytest.fixture(scope="session")
def empty_model():
    """Cell with the robot and no kits."""
    return build_spec().compile()


@pytest.fixture(scope="session")
def kit_model():
    """Cell with one kit at the centre of station A."""
    return build_spec({"A": [0.0]}).compile()


@pytest.fixture
def step():
    """step(model, data, bridge, command, seconds, suite=None) -> list of SensorFrames."""
    from frodo.config import get_config
    dt = get_config().sim.control_dt_s

    def run(model, data, bridge, command, seconds, suite=None, on_tick=None):
        frames = []
        for _ in range(round(seconds / dt)):
            bridge.apply(data, command)
            for _ in range(round(dt / model.opt.timestep)):
                mujoco.mj_step(model, data)
            if suite is not None:
                frames.append(suite.read(data))
            if on_tick is not None:
                on_tick()
        return frames
    return run


@pytest.fixture
def fresh():
    """fresh(model, pose=None, arm=None) -> MjData reset to a base pose."""
    def make(model, pose=None, arm=None):
        data = mujoco.MjData(model)
        if arm is None:
            reset(model, data, pose)
        else:
            reset(model, data, pose, arm)
        return data
    return make
