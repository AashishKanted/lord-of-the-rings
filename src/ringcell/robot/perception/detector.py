"""Ring and bottle-neck detection from one RGB-D frame. PLACEHOLDER.

Planned classical pipeline: colour threshold for the ring ([perception] HSV range),
depth to 3D points, circle fit for the ring centre; bottle neck from the depth blob at
the known neck height; belt-speed-aware tracking between frames. The detector is
selected by ``perception.method`` so a learned detector can be swapped in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...interfaces import RgbdFrame


@dataclass(frozen=True)
class KitObservation:
    t: float
    ring_in_base: np.ndarray | None      # xyz of the ring centre, base frame
    bottle_in_base: np.ndarray | None    # xyz of the neck top centre, base frame


class Detector:
    def __init__(self, config):
        self.config = config

    def detect(self, frame: RgbdFrame) -> KitObservation:
        raise NotImplementedError
