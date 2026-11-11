"""Simulated RGB-D camera: colour + metric depth from any named MuJoCo camera.

    uv run python -m frodo.sim.camera --camera wrist --out results/camera

MuJoCo camera frame convention: looks along -z, +x is image right, +y is image up.
Depth images are metres along the optical axis (not ray length). `RgbdFrame.cam_in_base`
(what robot code receives) uses the optical convention instead: x right, y down, z forward.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

# MuJoCo camera axes (x right, y up, z backward) -> optical axes (x right, y down, z forward).
MJ_TO_OPTICAL = np.diag([1.0, -1.0, -1.0])


class RgbdCamera:
    """One named MuJoCo camera. Several cameras of the same resolution can share one
    `mujoco.Renderer` (pass it in); otherwise a renderer is created on the first read."""

    def __init__(self, model, name: str, width: int = 640, height: int = 480, renderer=None):
        self.model, self.name, self.width, self.height = model, name, width, height
        self.cam_id = model.camera(name).id
        self._renderer, self._owns_renderer = renderer, renderer is None
        # Pinhole intrinsics from the vertical field of view (square pixels).
        fovy = np.deg2rad(model.cam_fovy[self.cam_id])
        self.fy = self.fx = 0.5 * height / np.tan(fovy / 2)
        self.cx, self.cy = width / 2, height / 2

    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        return float(self.fx), float(self.fy), float(self.cx), float(self.cy)

    def read(self, data):
        """Return (rgb uint8 HxWx3, depth float32 HxW in metres along the optical axis).
        Raw render: no noise, depth at the far clip plane where nothing is hit."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.height, self.width)
        r = self._renderer
        r.update_scene(data, camera=self.cam_id)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        try:
            depth = r.render().copy()
        finally:
            r.disable_depth_rendering()
        return rgb, depth

    def pose(self, data):
        """Camera position and 3x3 rotation (camera axes as columns) in the world."""
        return data.cam_xpos[self.cam_id].copy(), data.cam_xmat[self.cam_id].reshape(3, 3).copy()

    def cam_in_base(self, data, base_body_id: int) -> np.ndarray:
        """4x4 pose of the camera OPTICAL frame (x right, y down, z forward) in the frame of
        body `base_body_id`. For a camera on the arm this is the robot's forward kinematics."""
        pos, rot = self.pose(data)
        base_pos, base_rot = data.xpos[base_body_id], data.xmat[base_body_id].reshape(3, 3)
        out = np.eye(4)
        out[:3, :3] = base_rot.T @ rot @ MJ_TO_OPTICAL
        out[:3, 3] = base_rot.T @ (pos - base_pos)
        return out

    def deproject(self, data, u, v, depth):
        """Pixel(s) (u right, v down) + depth -> world xyz."""
        u, v, depth = np.asarray(u, float), np.asarray(v, float), np.asarray(depth, float)
        cam = np.stack([(u - self.cx) / self.fx * depth, -(v - self.cy) / self.fy * depth, -depth],
                       axis=-1)
        pos, rot = self.pose(data)
        return pos + cam @ rot.T

    def project(self, data, world_xyz):
        """World xyz -> (u, v, depth)."""
        pos, rot = self.pose(data)
        cam = (np.asarray(world_xyz, float) - pos) @ rot
        depth = -cam[..., 2]
        return (self.cx + self.fx * cam[..., 0] / depth, self.cy - self.fy * cam[..., 1] / depth,
                depth)

    def close(self):
        if self._owns_renderer and self._renderer is not None:
            self._renderer.close()
        self._renderer = None


def depth_to_image(depth, max_m=2.0):
    """Greyscale visualisation: near = bright."""
    return (255 * (1 - np.clip(depth / max_m, 0, 1))).astype(np.uint8)


def main():
    from PIL import Image

    from ..cell import STATIONS
    from ..robot.description import ARM_HOME
    from .scene import build_spec, reset

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", default="wrist")
    parser.add_argument("--out", type=Path, default=Path("results/camera"))
    args = parser.parse_args()

    model = build_spec({"A": [0.0]}).compile()
    data = mujoco.MjData(model)
    reset(model, data, STATIONS["A"].dock, ARM_HOME)
    cam = RgbdCamera(model, args.camera)
    rgb, depth = cam.read(data)
    args.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(args.out / f"{args.camera}_rgb.png")
    Image.fromarray(depth_to_image(depth)).save(args.out / f"{args.camera}_depth.png")
    np.save(args.out / f"{args.camera}_depth_m.npy", depth)
    print(f"{args.camera}: {rgb.shape[1]}x{rgb.shape[0]}, fx=fy={cam.fx:.1f}px, "
          f"depth range {depth.min():.3f}-{depth.max():.3f} m -> {args.out}/")

    # Self-check: project a known point into the image, read depth there, deproject back.
    truth = data.body("kit_A0_bottle").xpos + [0, 0, 0.18]  # centre of the bottle top
    u, v, d = cam.project(data, truth)
    if 0 <= u < cam.width and 0 <= v < cam.height and d > 0:
        back = cam.deproject(data, u, v, depth[int(v), int(u)])
        print(f"bottle top at pixel ({u:.0f},{v:.0f}); depth-deprojected error "
              f"{1000 * np.linalg.norm(back - truth):.1f} mm")
    else:
        print("bottle top is outside this camera's view")
    cam.close()


if __name__ == "__main__":
    main()
