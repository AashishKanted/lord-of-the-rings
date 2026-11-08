import pytest


def test_rigid_ring_can_seat_on_bottle_shoulder():
    pytest.importorskip("mujoco")
    from advaitic_systems.parts import validate_parts
    assert validate_parts()["passed"]
