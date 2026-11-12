"""Ring and bottle poses from segmented depth points.

Whatever produced a mask (depth bands, a learned segmenter), the pose comes from the same
geometry here, using only customer dimensions:

- ring:   flat annulus, inner / outer diameter from the spec. The mask's rim pixels are
          deprojected and a centre is fitted so every rim point lies on one of the two rims.
- bottle: the top of the neck is a disc of the neck diameter at the bottle height. The
          highest points of the mask are taken and a centre fitted to their rim.

Each fit returns a `CircleFit` with the numbers used to accept or reject it (`valid`):
residual, how much of the circumference was seen, and the height above the kit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, optimize

from ...cell import CELL, KIT_TOP_Z
from ...interfaces import RgbdFrame

_R, _B = CELL["ring"], CELL["bottle"]
RING_RADII = (_R["inner_diameter_m"] / 2, _R["outer_diameter_m"] / 2)
NECK_RADIUS = _B["neck_diameter_m"] / 2
RING_TOP_Z = KIT_TOP_Z + _R["thickness_m"]          # ring lying on its kit
BOTTLE_TOP_Z = KIT_TOP_Z + _B["height_m"]
SECTORS = 12


@dataclass(frozen=True)
class CircleFit:
    kind: str                 # "ring" | "bottle"
    center: np.ndarray        # xyz in the base frame: ring centre / top-of-neck centre
    rms_m: float              # RMS distance of the rim points from the fitted rim(s)
    coverage: float           # share of the circumference with rim points (0..1)
    points: int
    valid: bool
    why: str = ""             # reason when not valid


def pixel_points(frame: RgbdFrame, u, v) -> np.ndarray:
    """Base-frame xyz of pixels given by integer column `u` and row `v`. A pixel's depth
    belongs to its centre, which is half a pixel from its index."""
    return frame.deproject(u + 0.5, v + 0.5, frame.depth[v, u])


def mask_points(frame: RgbdFrame, mask: np.ndarray) -> np.ndarray:
    """Base-frame xyz of the mask's pixels that have a valid depth."""
    v, u = np.nonzero(mask & (frame.depth > 0))
    return pixel_points(frame, u, v) if len(u) else np.zeros((0, 3))


def rim(mask: np.ndarray) -> np.ndarray:
    """The mask's boundary pixels."""
    return mask & ~ndimage.binary_erosion(mask)


def _fit_center(xy: np.ndarray, radii, inlier_m: float) -> tuple[np.ndarray, float, float, int]:
    """Centre such that the points lie on the circles `radii`. Robust to points that are on
    neither (the edge where something hides part of the object). Returns centre, RMS of the
    inliers, share of the circumference they cover, and their number."""
    radii = np.asarray(radii)

    def residual(c):
        r = np.linalg.norm(xy - c, axis=1)
        return np.min(np.abs(r[:, None] - radii), axis=1)

    result = optimize.least_squares(residual, xy.mean(axis=0), loss="soft_l1", f_scale=0.001)
    res = residual(result.x)
    inlier = res < inlier_m
    angles = np.arctan2(*(xy[inlier] - result.x).T[::-1])
    sectors = np.unique(((angles + np.pi) / (2 * np.pi) * SECTORS).astype(int) % SECTORS)
    rms = float(np.sqrt(np.mean(res[inlier] ** 2))) if inlier.any() else np.inf
    return result.x, rms, len(sectors) / SECTORS, int(inlier.sum())


def _judge(kind, center, rms, coverage, n, z_expected, cfg, surface_xy, r_min, r_max) -> CircleFit:
    """Accept or reject a fit. `surface_xy` are all the object's points, which must lie
    between `r_min` and `r_max` of the centre: a centre fitted to the wrong rim of a partly
    hidden ring passes the rim checks but leaves much of the ring outside the annulus."""
    r = np.linalg.norm(surface_xy - center[:2], axis=1)
    inside = float(np.mean((r > r_min - cfg.rim_inlier_m) & (r < r_max + cfg.rim_inlier_m)))
    why = ""
    if inside < cfg.min_shape_fraction:
        why = f"only {100 * inside:.0f}% of the region fits the {kind} outline"
    elif n < cfg.min_rim_points:
        why = f"only {n} rim points"
    elif rms > cfg.max_fit_rms_m:
        why = f"rim fit rms {1e3 * rms:.1f} mm"
    elif coverage < cfg.min_coverage:
        why = f"only {100 * coverage:.0f}% of the circumference seen"
    elif abs(center[2] - z_expected) > cfg.max_height_error_m:
        why = f"height off by {1e3 * (center[2] - z_expected):.0f} mm"
    return CircleFit(kind, center, rms, coverage, n, not why, why)


def top_surface(frame: RgbdFrame, mask: np.ndarray, band_m: float, percentile: float):
    """Depth clean-up of a mask: keep only its pixels within `band_m` of its upper surface.
    A colour segmenter's mask may include the hole of the ring, a shadow or the side of an
    object; those pixels are at a different height. Returns (refined mask, its points)."""
    v, u = np.nonzero(mask & (frame.depth > 0))
    if len(u) < 3:
        return np.zeros_like(mask), np.zeros((0, 3))
    points = pixel_points(frame, u, v)
    keep = np.abs(points[:, 2] - np.percentile(points[:, 2], percentile)) < band_m
    refined = np.zeros_like(mask)
    refined[v[keep], u[keep]] = True
    return refined, points[keep]


def fit_ring(frame: RgbdFrame, mask: np.ndarray, cfg, center_z: float | None = None) -> CircleFit:
    """Level ring; `center` z is its mid-height. It is expected lying on its kit unless
    `center_z` gives another height (a ring held by the gripper)."""
    center_z = RING_TOP_Z - _R["thickness_m"] / 2 if center_z is None else center_z
    mask, surface = top_surface(frame, mask, cfg.surface_band_m, 75)
    points = mask_points(frame, rim(mask))
    if len(points) < 3:
        return CircleFit("ring", np.zeros(3), np.inf, 0.0, len(points), False, "no rim points")
    xy, rms, coverage, n = _fit_center(points[:, :2], RING_RADII, cfg.rim_inlier_m)
    center = np.array([*xy, float(np.median(surface[:, 2])) - _R["thickness_m"] / 2])
    return _judge("ring", center, rms, coverage, n, center_z, cfg, surface[:, :2], *RING_RADII)


def fit_bottle(frame: RgbdFrame, mask: np.ndarray, cfg) -> CircleFit:
    """Top of the bottle neck. `center` z is the top surface of the neck."""
    disc, top = top_surface(frame, mask, cfg.surface_band_m, 95)
    edge = mask_points(frame, rim(disc))
    if len(edge) < 3:
        return CircleFit("bottle", np.zeros(3), np.inf, 0.0, len(edge), False, "no rim points")
    xy, rms, coverage, n = _fit_center(edge[:, :2], (NECK_RADIUS,), cfg.rim_inlier_m)
    center = np.array([*xy, float(np.median(top[:, 2]))])
    return _judge("bottle", center, rms, coverage, n, BOTTLE_TOP_Z, cfg, top[:, :2], 0.0,
                  NECK_RADIUS)
