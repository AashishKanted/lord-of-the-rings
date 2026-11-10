"""Guards the plant/robot separation required by the integration contract."""
import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2] / "src" / "frodo"
FORBIDDEN = ("sim", "plant", "apps")


def _imports(path: Path):
    depth = len(path.relative_to(PKG).parts)  # 1 for frodo/x.py
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:  # relative: resolve to an absolute frodo.* name
                parts = ["frodo", *path.relative_to(PKG).parts[:depth - node.level]]
                module = ".".join([*parts, module]) if module else ".".join(parts)
            yield module


@pytest.mark.parametrize("path", sorted((PKG / "robot").rglob("*.py")), ids=str)
def test_robot_code_does_not_import_simulator_or_plant(path):
    for module in _imports(path):
        for side in FORBIDDEN:
            assert not (module == f"frodo.{side}" or module.startswith(f"frodo.{side}.")), (
                f"{path.name} imports {module}: robot code must not see ground truth")


def test_config_loads():
    from frodo.config import get_config
    cfg = get_config()
    assert cfg.sensors.cameras.perception_source in cfg.sensors.cameras.enabled
    assert cfg.safety.stop_distance_m < cfg.safety.slow_distance_m
