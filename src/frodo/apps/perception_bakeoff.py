"""Perception bake-off: the same wrist-camera frames through each segmenter, scored
against simulator truth.

    scripts/run.sh perception --frames 200 --segmenters depth_bands fastsam

Each frame: the robot is docked with the arm at its waiting pose (camera looking straight
down), one or two kits are placed at random positions along the belt, and the simulated
RGB-D camera (with its depth noise) is rendered. Every segmenter's masks go through the
same geometry and checks (`RgbdDetector`). Truth is used for scoring only.

This measures detection on still frames. It is not a mission run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from ..cell import CELL, STATIONS
from ..config import ROOT, get_config
from ..interfaces import Pose2D
from ..log import get_logger, setup_logging
from ..plant.feeder import PITCH_M, find_pool
from ..robot.frames import StationFrame
from ..robot.manipulation.arm_controller import ArmController
from ..robot.perception.rgbd_detector import RgbdDetector
from ..robot.perception.segmenters import make_segmenter
from ..sim.oracle import OracleKitSource
from ..sim.scene import build_spec, reset
from ..sim.sensors import SensorSuite

log = get_logger(__name__)
MATCH_M = 0.010       # a fit within this of a true object counts as that object
BOTTLE_H = CELL["bottle"]["height_m"]


def _visible(image, point, margin_px: float) -> bool:
    u, v, z = image.project(point)
    h, w = image.depth.shape
    return bool(z > 0 and margin_px <= u < w - margin_px and margin_px <= v < h - margin_px)


def _score(fits, visible, everything) -> tuple[list[float], int, int]:
    """(errors of the fully visible objects found, fully visible objects missed, fits that
    are on no real object at all)."""
    errors = []
    for truth in visible:
        d = [np.linalg.norm(f.center[:2] - truth[:2]) for f in fits]
        if d and min(d) < MATCH_M:
            errors.append(float(min(d)))
    false = sum(all(np.linalg.norm(f.center[:2] - t[:2]) >= MATCH_M for t in everything)
                for f in fits)
    return errors, len(visible) - len(errors), false


def run(frames: int = 200, segmenters=("depth_bands",), seed: int = 0, station: str = "A",
        out_dir: Path | None = None, video: bool = False, fps: int = 5) -> dict:
    cfg = get_config()
    out = Path(out_dir or ROOT / "results" / "perception_bakeoff")
    out.mkdir(parents=True, exist_ok=True)
    st = STATIONS[station]
    frame_of_station = StationFrame.from_pose(station, Pose2D(*(float(v) for v in st.dock)))
    arm = ArmController(build_spec().compile(), cfg)
    arm.set_station(frame_of_station)
    model = build_spec({station: [0.0, 0.0]}).compile()
    data = mujoco.MjData(model)
    kits, oracle = find_pool(model, st, cfg), OracleKitSource(model)
    suite = SensorSuite(model, cfg, seed=seed)
    detectors = {name: RgbdDetector(cfg, make_segmenter(cfg, name)) for name in segmenters}
    stats = {name: {"ring": [], "bottle": [], "ring_missed": 0, "bottle_missed": 0,
                    "false_fits": 0, "kits": 0, "seconds": []} for name in segmenters}
    writers = {}
    if video:
        import imageio.v2 as imageio
        writers = {name: imageio.get_writer(out / f"detections_{name}.mp4", fps=fps,
                                            macro_block_size=None) for name in segmenters}
    rng, truth_count = np.random.default_rng(seed), {"ring": 0, "bottle": 0, "kits": 0}

    for index in range(frames):
        reset(model, data, st.dock, arm.q_ready)
        x = rng.uniform(-0.42, 0.05)                     # kit centre, station-local x
        for kit, kit_x in zip(kits, (x, x - PITCH_M if rng.random() < 0.5 else -5.0),
                              strict=True):
            kit.place_kit(data, kit_x, 0.0)
            kit.place_ring_on_nest(data, kit_x, 0.0)
        mujoco.mj_forward(model, data)
        image = suite.render_camera("wrist", data)
        rings, bottles, all_rings, all_bottles, whole = [], [], [], [], 0
        for truth in oracle.read(data):
            top = truth.bottle_in_base + [0, 0, BOTTLE_H]
            # margins: the object's radius in pixels, so "visible" means entirely in the image
            ring_seen, bottle_seen = _visible(image, truth.ring_in_base, 45), _visible(image, top, 35)
            all_rings.append(truth.ring_in_base)
            all_bottles.append(top)
            rings += [truth.ring_in_base] if ring_seen else []
            bottles += [top] if bottle_seen else []
            whole += ring_seen and bottle_seen
        truth_count["ring"] += len(rings)
        truth_count["bottle"] += len(bottles)
        truth_count["kits"] += whole
        for name, detector in detectors.items():
            tic = time.perf_counter()
            report = detector.analyse(image, frame_of_station)
            s = stats[name]
            s["seconds"].append(time.perf_counter() - tic)
            for kind, fits, truths, everything in (("ring", report.rings, rings, all_rings),
                                                   ("bottle", report.bottles, bottles, all_bottles)):
                errors, missed, false = _score(fits, truths, everything)
                s[kind] += errors
                s[f"{kind}_missed"] += missed
                s["false_fits"] += false
            s["kits"] += len(report.kits)
            if video or index == 0:
                picture = _overlay(image, detector, report, all_rings + all_bottles,
                                   _caption(index, name, report, all_rings, all_bottles))
                if index == 0:
                    picture.save(out / f"example_{name}.png")
                if video:
                    writers[name].append_data(np.asarray(picture))
    suite.close()
    for writer in writers.values():
        writer.close()

    summary = {"what": "still-frame detection at the waiting pose; truth used for scoring only",
               "frames": frames, "seed": seed, "camera": "wrist (top-down)",
               "depth_noise_std_m_at_1m": cfg.sensors.cameras.depth_noise_std_m,
               "fully_visible": truth_count, "segmenters": {}}
    for name, s in stats.items():
        row = {"kits_detected": s["kits"], "false_fits": s["false_fits"],
               "ms_per_frame_median": round(1e3 * float(np.median(s["seconds"][1:] or s["seconds"])), 1)}
        for kind in ("ring", "bottle"):
            e = 1e3 * np.array(s[kind])
            row[kind] = {"found": len(e), "missed": s[f"{kind}_missed"],
                         "error_mm_mean": round(float(e.mean()), 3) if len(e) else None,
                         "error_mm_p95": round(float(np.percentile(e, 95)), 3) if len(e) else None,
                         "error_mm_max": round(float(e.max()), 3) if len(e) else None}
        summary["segmenters"][name] = row
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _overlay(image, detector, report, truths, title: str):
    """Colour image with proposed regions outlined (yellow), accepted fits drawn as their
    spec circles (ring green, neck top red) and true centres as white crosses."""
    from PIL import Image, ImageDraw

    from ..robot.perception.geometry import NECK_RADIUS, RING_RADII, rim
    rgb = image.rgb.copy()
    for mask in detector.segmenter.masks(image):
        rgb[rim(mask)] = (255, 255, 0)
    picture = Image.fromarray(rgb)
    draw = ImageDraw.Draw(picture)
    angles = np.linspace(0, 2 * np.pi, 73)
    circle = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
    for fits, radii, colour in ((report.rings, RING_RADII, (0, 255, 0)),
                                (report.bottles, (NECK_RADIUS,), (255, 40, 40))):
        for fit in fits:
            for radius in radii:
                u, v, _ = image.project(fit.center + radius * circle)
                draw.line(list(zip(u.tolist(), v.tolist(), strict=True)), fill=colour, width=2)
    for truth in truths:
        u, v, _ = image.project(truth)
        draw.line([u - 5, v, u + 5, v], fill="white")
        draw.line([u, v - 5, u, v + 5], fill="white")
    draw.rectangle([0, 0, image.rgb.shape[1], 30], fill=(0, 0, 0))
    draw.text((6, 4), title, fill="white")
    return picture


def _caption(index, name, report, rings, bottles) -> str:
    def worst(fits, truths):
        d = [min(np.linalg.norm(f.center[:2] - t[:2]) for t in truths) for f in fits] if truths else []
        return f"{1e3 * max(d):.2f} mm" if d else "-"
    return (f"frame {index}  {name}  rings {len(report.rings)} (err {worst(report.rings, rings)})  "
            f"neck tops {len(report.bottles)} (err {worst(report.bottles, bottles)})  "
            f"kits {len(report.kits)}  rejected regions {len(report.rejected)}\n"
            "yellow: proposed regions   green/red: accepted ring / neck-top fit   white +: truth")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--segmenters", nargs="+", default=["depth_bands"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--video", action="store_true",
                        help="write detections_<segmenter>.mp4 with every frame's result drawn")
    args = parser.parse_args()
    setup_logging()
    print(json.dumps(run(args.frames, tuple(args.segmenters), args.seed, out_dir=args.out,
                         video=args.video), indent=2))


if __name__ == "__main__":
    main()
