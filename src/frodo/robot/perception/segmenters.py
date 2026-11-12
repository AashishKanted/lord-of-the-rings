"""Segmentation sources: one RGB-D frame in, candidate object masks out.

A segmenter only proposes image regions. Whether a region is a ring or a bottle, and where
it is, is decided afterwards by `geometry` from the depth inside the mask, so a segmenter
may over-propose.

- `DepthBandSegmenter`: no model. Groups pixels by height above the kit surface.
- `FastSamSegmenter`:   class-agnostic "segment everything" network on the colour image.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy import ndimage

from ...interfaces import RgbdFrame
from .geometry import BOTTLE_TOP_Z, RING_TOP_Z


class Segmenter(Protocol):
    name: str

    def masks(self, frame: RgbdFrame) -> list[np.ndarray]:
        """Boolean HxW masks of candidate objects."""


def _base_z(frame: RgbdFrame) -> np.ndarray:
    """Height above the floor of every pixel (NaN where depth is invalid)."""
    h, w = frame.depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = frame.deproject(u + 0.5, v + 0.5, frame.depth)[..., 2]
    return np.where(frame.depth > 0, z, np.nan)


class DepthBandSegmenter:
    """Connected regions whose height matches the top of a ring on its kit, or the top of a
    bottle neck. Colour is not used."""
    name = "depth_bands"

    def __init__(self, config):
        self.cfg = config.perception

    def masks(self, frame: RgbdFrame) -> list[np.ndarray]:
        z, tol, out = _base_z(frame), self.cfg.band_tolerance_m, []
        for level in (RING_TOP_Z, BOTTLE_TOP_Z):
            labels, count = ndimage.label(np.abs(z - level) < tol)
            sizes = ndimage.sum_labels(np.ones_like(labels), labels, np.arange(1, count + 1))
            out += [labels == i + 1 for i in np.nonzero(sizes >= self.cfg.min_mask_pixels)[0]]
        return out


class FastSamSegmenter:
    """Ultralytics FastSAM in "everything" mode. Weights download on first use."""
    name = "fastsam"

    def __init__(self, config):
        from ultralytics import FastSAM  # optional dependency (group "perception")

        from ...config import ROOT
        self.cfg = config.perception
        self._model = FastSAM(str(ROOT / self.cfg.fastsam_weights))

    def masks(self, frame: RgbdFrame) -> list[np.ndarray]:
        result = self._model(frame.rgb[..., ::-1], device=self.cfg.fastsam_device,
                             retina_masks=True, imgsz=self.cfg.fastsam_imgsz,
                             conf=self.cfg.fastsam_conf, iou=0.9, verbose=False)[0]
        if result.masks is None:
            return []
        masks = result.masks.data.cpu().numpy().astype(bool)
        return [m for m in masks if m.sum() >= self.cfg.min_mask_pixels]


def make_segmenter(config, name: str | None = None) -> Segmenter:
    name = name or config.perception.segmenter
    if name == "depth_bands":
        return DepthBandSegmenter(config)
    if name == "fastsam":
        return FastSamSegmenter(config)
    raise ValueError(f"unknown segmenter '{name}' (have: depth_bands, fastsam)")
