"""Kit pool, belt geometry and per-run line parameters for the customer feeder.

MuJoCo bodies must exist at compile time, so each station owns a fixed pool of kit
carriers. A kit that is not on the belt is parked in a "magazine": a row of slots on the
same slide axis, upstream of the belt and outside the conveyor. Two plant-side fixture
resets act on parts that are outside the work area:

- release: a parked kit is put on the belt ``plant.release_lead_m`` before the work-entry
  line (see `release_x`) with its ring on its nest, moving at belt speed;
- recycle: a kit that has run off the downstream end (after inspection) is put back in
  its magazine slot and its ring is put back on its nest.

Nothing is repositioned between the release point and the belt's downstream end, so
everything the robot and the inspection see is plain physics.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from ..cell import CELL, KIT_TOP_Z, STATIONS, Station
from ..config import get_config

_C, _K = CELL["conveyor"], CELL["kit"]
PITCH_M = _C["kit_pitch_m"]
SPEED_RANGE_MPS = tuple(_C["test_speed_range_mps"])
NOMINAL_SPEED_MPS = _C["nominal_speed_mps"]
BOTTLE_X_IN_KIT = _K["bottle_xy_in_kit_m"][0]
# Kit-centre positions (station-local x) at which a whole kit is just on / just off the belt.
INFEED_X = -_C["length_m"] / 2 + _K["length_m"] / 2
OUTFEED_X = _C["length_m"] / 2 - _K["length_m"] / 2
# Bottle-centre crossings defined by the customer packet.
WORK_ENTRY_X = _C["work_entry_local_x_m"]
WORK_EXIT_X = _C["work_exit_local_x_m"]
INSPECTION_X = _C["inspection_local_x_m"]
BATCH = CELL["timing"]["accepted_per_station_before_switch"]


def release_x(config=None) -> float:
    """Kit-centre position (station-local x) at which the feeder puts a kit on the belt.

    The customer documents do not fix where a kit joins the belt. ``plant.release_lead_m``
    is the distance the bottle still has to travel to the work-entry line; the position is
    never upstream of the point where the whole kit is on the belt.
    """
    plant = (config or get_config()).plant
    return max(INFEED_X, WORK_ENTRY_X - plant.release_lead_m - BOTTLE_X_IN_KIT)


def min_pool_size() -> int:
    """Kits that can be on one belt at once when released one pitch apart."""
    return ceil((OUTFEED_X - INFEED_X) / PITCH_M)


def magazine_slots(count: int, config=None) -> list[float]:
    """Station-local x of the parked kit centres, nearest the belt first."""
    plant = (config or get_config()).plant
    first = -_C["length_m"] / 2 - plant.magazine_gap_m - _K["length_m"] / 2
    return [first - i * (_K["length_m"] + plant.magazine_spacing_m) for i in range(count)]


def pool_layout(config=None, stations=None) -> dict[str, list[float]]:
    """The ``kits`` argument for ``frodo.sim.scene.build_spec``: a parked pool per station."""
    plant = (config or get_config()).plant
    if plant.pool_size < min_pool_size():
        raise ValueError(f"plant.pool_size={plant.pool_size} is below the {min_pool_size()} kits "
                         "that fit on a belt at the customer pitch")
    return {name: magazine_slots(plant.pool_size, config) for name in (stations or STATIONS)}


@dataclass(frozen=True)
class RunParameters:
    """Per-run line settings, constant for the whole run. Keys are station names."""
    belt_speed_mps: dict[str, float]
    release_phase_s: dict[str, float]

    def __post_init__(self):
        lo, hi = SPEED_RANGE_MPS
        for name, speed in self.belt_speed_mps.items():
            if not lo <= speed <= hi:
                raise ValueError(f"belt speed {speed} m/s at {name} is outside the customer "
                                 f"test range {lo}-{hi} m/s")
        if any(phase < 0 for phase in self.release_phase_s.values()):
            raise ValueError("release phase must be nonnegative")


def run_parameters(belt_speed_mps: float | dict | None = None,
                   release_phase_s: float | dict | None = None) -> RunParameters:
    """Explicit settings; defaults are the nominal customer speed and zero phase."""
    def per_station(value, default):
        if value is None:
            value = default
        if isinstance(value, dict):
            return {name: float(value.get(name, default)) for name in STATIONS}
        return {name: float(value) for name in STATIONS}
    return RunParameters(per_station(belt_speed_mps, NOMINAL_SPEED_MPS),
                         per_station(release_phase_s, 0.0))


def draw_run_parameters(seed: int, config=None) -> RunParameters:
    """Seeded belt speed (customer test range) and release phase per station.

    docs/ACCEPTANCE.md: "vary static belt speed within 25-35 mm/s and kit release phase
    within 0-1 s after readiness. Use the seed to select values reproducibly."
    """
    plant = (config or get_config()).plant
    rng = np.random.default_rng(seed)
    speeds = {name: round(float(rng.uniform(*SPEED_RANGE_MPS)), 4) for name in STATIONS}
    phases = {name: round(float(rng.uniform(*plant.release_phase_range_s)), 3)
              for name in STATIONS}
    return RunParameters(speeds, phases)


@dataclass
class Carrier:
    """One pooled kit body and what the line currently knows about it."""
    name: str
    station: Station
    slot_x: float
    slide_qpos: int
    slide_dof: int
    drive: int
    ring_qpos: int
    ring_dof: int
    weld: int
    on_belt: bool = False
    ring_on_nest: bool = True      # False while a recycle waits for the robot to let go
    record: dict | None = None     # ledger entry of the kit it currently carries
    release_travel: float = 0.0
    monitor: object | None = None  # SeatMonitor between work entry and inspection

    @classmethod
    def from_model(cls, model, station: Station, name: str, slot_x: float) -> Carrier:
        slide = model.joint(f"{name}_slide")
        ring = model.joint(f"{name}_ring_free")
        return cls(name, station, slot_x, int(slide.qposadr[0]), int(slide.dofadr[0]),
                   model.actuator(f"{name}_drive").id, int(ring.qposadr[0]),
                   int(ring.dofadr[0]), model.equality(f"{name}_grasp").id)

    def x(self, data) -> float:
        """Kit centre, station-local x (the slide joint is referenced to it)."""
        return float(data.qpos[self.slide_qpos])

    def bottle_x(self, data) -> float:
        return self.x(data) + BOTTLE_X_IN_KIT

    def place_kit(self, data, x: float, speed: float) -> None:
        data.qpos[self.slide_qpos], data.qvel[self.slide_dof] = x, speed

    def ring_held(self, data) -> bool:
        return bool(data.eq_active[self.weld])

    def place_ring_on_nest(self, data, x: float, speed: float) -> None:
        """Fixture reset: ring flat on its nest of a kit whose centre is at local x."""
        rx, ry = _K["ring_xy_in_kit_m"]
        st = self.station
        pos = st.to_world([x + rx, ry], KIT_TOP_Z + CELL["ring"]["thickness_m"] / 2)
        data.qpos[self.ring_qpos:self.ring_qpos + 7] = [*pos, 1, 0, 0, 0]
        data.qvel[self.ring_dof:self.ring_dof + 6] = [speed * np.cos(st.yaw),
                                                      speed * np.sin(st.yaw), 0, 0, 0, 0]
        self.ring_on_nest = True


def find_pool(model, station: Station, config=None) -> list[Carrier]:
    """Carriers ``kit_<station><i>`` present in the compiled model, in index order."""
    names = []
    while True:
        name = f"kit_{station.name}{len(names)}"
        try:
            model.body(name)
        except KeyError:
            break
        names.append(name)
    slots = magazine_slots(len(names), config)
    return [Carrier.from_model(model, station, n, x) for n, x in zip(names, slots, strict=True)]
