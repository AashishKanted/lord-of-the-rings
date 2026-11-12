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
from scipy import ndimage

from ...cell import CELL
from ...interfaces import SensorFrame
from ..frames import StationFrame
from .geometry import CircleFit, fit_bottle, fit_ring
from .segmenters import Segmenter, height_map, make_segmenter
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
            found = ring is not None and \
                np.linalg.norm(ring.center[:2] - expected) <= self.cfg.pair_tolerance_m
            base = bottle.center - [0, 0, CELL["bottle"]["height_m"]]
            # Without its ring a bottle can still update a kit that is already tracked.
            report.kits.append(KitDetection(image.t, ring.center if found else None, base))
        return report

    def find_held_ring(self, frame: SensorFrame, expected: np.ndarray):
        """The ring in the gripper, looked for near where the robot believes it is holding
        it (`expected`, ring centre in the base frame). Returns (capture time, centre) or
        None. Depth only: the ring is the level annulus at that height."""
        image = frame.camera
        if image is None:
            return None
        z = height_map(image)
        top = expected[2] + CELL["ring"]["thickness_m"] / 2
        labels, count = ndimage.label(np.abs(z - top) < self.cfg.held_ring_band_m)
        best = None
        for index in range(1, count + 1):
            mask = labels == index
            if mask.sum() < self.cfg.min_mask_pixels:
                continue
            fit = fit_ring(image, mask, self.cfg, center_z=expected[2])
            near = np.linalg.norm(fit.center[:2] - expected[:2]) < self.cfg.held_ring_search_m
            if fit.valid and near and (best is None or fit.rms_m < best.rms_m):
                best = fit
        return None if best is None else (image.t, best.center)


def _unique(fits: list[CircleFit], radius: float = 0.01) -> list[CircleFit]:
    """Several masks can cover one object; keep the best fit per location."""
    kept: list[CircleFit] = []
    for fit in sorted(fits, key=lambda f: f.rms_m):
        if all(np.linalg.norm(fit.center[:2] - k.center[:2]) > radius for k in kept):
            kept.append(fit)
    return kept
