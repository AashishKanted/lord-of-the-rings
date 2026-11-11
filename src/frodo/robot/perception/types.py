"""What perception hands to the rest of the robot software."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KitDetection:
    """One kit found in one sensor sample. Poses are in the robot base frame."""
    t: float                     # capture time of the sample
    ring_in_base: np.ndarray     # xyz of the ring centre
    bottle_in_base: np.ndarray   # xyz of the bottle axis at the bottle base (kit top surface)


@dataclass(frozen=True)
class KitObservation:
    """Tracked kit, valid at time ``t``: the last detection carried forward along the belt
    with the conveyor encoder."""
    t: float
    track_id: int
    ring_in_base: np.ndarray
    bottle_in_base: np.ndarray
    belt_velocity: np.ndarray    # base frame xyz, m/s
    age_s: float                 # time since the detection this is based on
    source: str                  # detector name, e.g. "oracle"
