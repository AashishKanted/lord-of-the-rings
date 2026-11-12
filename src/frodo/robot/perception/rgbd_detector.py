"""RGB-D kit detector: segment, fit ring and bottle from depth, verify, pair into kits.

    masks   = segmenter.masks(frame)                 # proposals (may include junk)
    fits    = ring / bottle circle fit per mask      # geometry.py, with validity checks
    kits    = bottle + the ring found one ring-offset downstream of it on the same kit

A fit that fails its checks is dropped, and a bottle without its ring (or the reverse) does
not make a kit, so a wrong proposal has to pass three geometric tests to get through.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...cell import CELL
from ...interfaces import SensorFrame
from ..frames import StationFrame
from .geometry import CircleFit, fit_bottle, fit_ring
from .segmenters import Segmenter, make_segmenter
from .types import KitDetection

_K = CELL["kit"]
RING_FROM_BOTTLE = np.array(_K["ring_xy_in_kit_m"]) - np.array(_K["bottle_xy_in_kit_m"])


@dataclass
class DetectionReport:
    """Everything one frame produced, for debugging and the bake-off."""
    rings: list[CircleFit] = field(default_factory=list)
    bottles: list[CircleFit] = field(default_factory=list)
    rejected: list[CircleFit] = field(default_factory=list)
    kits: list[KitDetection] = field(default_factory=list)


class RgbdDetector:
    def __init__(self, config, segmenter: Segmenter | None = None):
        self.cfg = config.perception
        self.segmenter = segmenter or make_segmenter(config)
        self.name = f"rgbd/{self.segmenter.name}"
        self.last = DetectionReport()

    def reset(self) -> None:
        self.last = DetectionReport()

    def detect(self, frame: SensorFrame, station: StationFrame) -> list[KitDetection] | None:
        if frame.camera is None:
            return None
        self.last = report = self.analyse(frame.camera, station)
        return report.kits

    def analyse(self, image, station: StationFrame) -> DetectionReport:
        report = DetectionReport()
        for mask in self.segmenter.masks(image):
            # A mask is tried as both shapes; the height check tells them apart.
            fits = (fit_ring(image, mask, self.cfg), fit_bottle(image, mask, self.cfg))
            good = [f for f in fits if f.valid]
            if not good:
                report.rejected.append(min(fits, key=lambda f: f.rms_m))
            for fit in good:
                (report.rings if fit.kind == "ring" else report.bottles).append(fit)
        report.rings, report.bottles = _unique(report.rings), _unique(report.bottles)
        offset = RING_FROM_BOTTLE[0] * station.belt_dir + RING_FROM_BOTTLE[1] * station.lateral
        for bottle in report.bottles:
            expected = bottle.center[:2] + offset[:2]
            ring = min(report.rings, default=None,
                       key=lambda r: np.linalg.norm(r.center[:2] - expected))
            if ring is None or np.linalg.norm(ring.center[:2] - expected) > self.cfg.pair_tolerance_m:
                continue
            base = bottle.center - [0, 0, CELL["bottle"]["height_m"]]
            report.kits.append(KitDetection(image.t, ring.center, base))
        return report


def _unique(fits: list[CircleFit], radius: float = 0.01) -> list[CircleFit]:
    """Several masks can cover one object; keep the best fit per location."""
    kept: list[CircleFit] = []
    for fit in sorted(fits, key=lambda f: f.rms_m):
        if all(np.linalg.norm(fit.center[:2] - k.center[:2]) > radius for k in kept):
            kept.append(fit)
    return kept
