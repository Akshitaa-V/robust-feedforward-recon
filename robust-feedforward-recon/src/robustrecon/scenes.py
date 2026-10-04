"""Procedural dynamic driving scenes with exact ground truth.

Every sequence is ray-cast analytically, so each frame comes with a dense
z-depth map, semantic labels, a moving-object mask and the camera pose.
Camera convention: x right, y down, z forward. The ground is the plane
y = +camera_height in world coordinates (the camera starts at the origin).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SKY, GROUND, BUILDING, DYNAMIC = 0, 1, 2, 3
CLASS_NAMES = {SKY: "sky", GROUND: "ground", BUILDING: "static", DYNAMIC: "dynamic"}


@dataclass
class Box:
    lo: np.ndarray            # (3,) min corner at t = 0
    hi: np.ndarray            # (3,) max corner at t = 0
    color: np.ndarray         # (3,) base albedo
    label: int
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        shift = self.velocity * t
        return self.lo + shift, self.hi + shift


@dataclass
class SceneConfig:
    height: int = 64
    width: int = 96
    focal_scale: float = 0.8      # fx = fy = focal_scale * width
    camera_height: float = 1.5
    max_depth: float = 60.0
    n_frames: int = 5


def intrinsics(cfg: SceneConfig) -> np.ndarray:
    f = cfg.focal_scale * cfg.width
    return np.array([[f, 0.0, cfg.width / 2.0],
                     [0.0, f, cfg.height / 2.0],
                     [0.0, 0.0, 1.0]])


def _hash(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = np.sin(a * 12.9898 + b * 78.233) * 43758.5453
    return v - np.floor(v)


def rot_y(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def sample_layout(rng: np.random.Generator, cfg: SceneConfig) -> dict:
    """Static buildings, parked cars and moving road users for one sequence."""
    g = cfg.camera_height
    boxes: list[Box] = []
    for side in (-1.0, 1.0):
        z = rng.uniform(-5.0, 2.0)
        while z < 140.0:
            length = rng.uniform(6.0, 16.0)
            near = rng.uniform(6.0, 8.0)
            depth = rng.uniform(4.0, 8.0)
            h = rng.uniform(4.0, 16.0)
            x0, x1 = sorted((side * near, side * (near + depth)))
            col = rng.uniform(0.25, 0.85, 3)
            boxes.append(Box(np.array([x0, g - h, z]), np.array([x1, g, z + length]), col, BUILDING))
            z += length + rng.uniform(0.0, 5.0)
        for _ in range(rng.integers(1, 4)):        # parked cars
            zc = rng.uniform(8.0, 60.0)
            xc = side * rng.uniform(4.2, 5.0)
            col = rng.uniform(0.1, 0.9, 3)
            boxes.append(Box(np.array([xc - 0.9, g - 1.5, zc]), np.array([xc + 0.9, g, zc + 4.2]), col, BUILDING))
    for _ in range(rng.integers(2, 6)):            # moving vehicles
        lane = rng.choice([-1.75, 1.75])
        zc = rng.uniform(6.0, 45.0)
        vz = rng.uniform(0.3, 1.6) if lane > 0 else -rng.uniform(0.5, 1.6)
        w, h, l = rng.uniform(1.7, 2.1), rng.uniform(1.4, 2.6), rng.uniform(3.8, 6.0)
        col = rng.uniform(0.05, 0.95, 3)
        boxes.append(Box(np.array([lane - w / 2, g - h, zc]), np.array([lane + w / 2, g, zc + l]),
                         col, DYNAMIC, np.array([0.0, 0.0, vz])))
    for _ in range(rng.integers(0, 3)):            # pedestrians crossing
        xc = rng.uniform(-5.0, 5.0)
        zc = rng.uniform(6.0, 25.0)
        vx = rng.choice([-1.0, 1.0]) * rng.uniform(0.1, 0.25)
        col = rng.uniform(0.1, 0.9, 3)
        boxes.append(Box(np.array([xc - 0.25, g - 1.75, zc]), np.array([xc + 0.25, g, zc + 0.4]),
                         col, DYNAMIC, np.array([vx, 0.0, 0.0])))
    speed = rng.uniform(0.6, 1.3)
    yaw0 = rng.uniform(-0.04, 0.04)
    yaw_rate = rng.uniform(-0.01, 0.01)
    x_drift = rng.uniform(-0.05, 0.05)
    sun = np.array([rng.uniform(-0.6, 0.6), -1.0, rng.uniform(-0.3, 0.6)])
    return {"boxes": boxes, "speed": speed, "yaw0": yaw0, "yaw_rate": yaw_rate,
            "x_drift": x_drift, "sun": sun / np.linalg.norm(sun)}


def camera_pose(layout: dict, t: int) -> np.ndarray:
    """4x4 camera-to-world transform for frame t."""
    yaw = layout["yaw0"] + layout["yaw_rate"] * t
    T = np.eye(4)
    T[:3, :3] = rot_y(yaw)
    T[:3, 3] = [layout["x_drift"] * t, 0.0, layout["speed"] * t]
    return T


def _intersect_boxes(origin: np.ndarray, dirs: np.ndarray, los: np.ndarray, his: np.ndarray):
    """Slab test. dirs (N,3) unnormalised, returns t_hit (N,B) and hit axis (N,B)."""
    inv = 1.0 / np.where(np.abs(dirs) < 1e-9, 1e-9, dirs)          # (N,3)
    t0 = (los[None] - origin[None, None]) * inv[:, None]           # (N,B,3)
    t1 = (his[None] - origin[None, None]) * inv[:, None]
    tmin = np.minimum(t0, t1)
    tmax = np.maximum(t0, t1)
    t_near = tmin.max(axis=2)
    t_far = tmax.min(axis=2)
    axis = tmin.argmax(axis=2)
    hit = (t_far >= t_near) & (t_near > 1e-3)
    return np.where(hit, t_near, np.inf), axis


def render_frame(layout: dict, cfg: SceneConfig, t: int) -> dict:
    H, W = cfg.height, cfg.width
    K = intrinsics(cfg)
    pose = camera_pose(layout, t)
    R, c = pose[:3, :3], pose[:3, 3]
    v, u = np.mgrid[0:H, 0:W]
    pix = np.stack([u + 0.5, v + 0.5, np.ones_like(u, dtype=float)], -1).reshape(-1, 3)
    d_cam = pix @ np.linalg.inv(K).T                     # z component = 1, so ray t = z-depth
    d_world = d_cam @ R.T
    N = d_world.shape[0]

    boxes = layout["boxes"]
    los = np.stack([b.at(t)[0] for b in boxes])
    his = np.stack([b.at(t)[1] for b in boxes])
    t_box, axis = _intersect_boxes(c, d_world, los, his)
    bidx = t_box.argmin(axis=1)
    t_b = t_box[np.arange(N), bidx]

    g = cfg.camera_height
    dy = d_world[:, 1]
    t_g = np.where(dy > 1e-6, (g - c[1]) / np.where(dy > 1e-6, dy, 1.0), np.inf)

    depth = np.minimum(t_b, t_g)
    is_box = t_b < t_g
    label = np.where(np.isinf(depth), SKY, np.where(is_box, -1, GROUND))
    labels_b = np.array([b.label for b in boxes])
    label = np.where(is_box & np.isfinite(depth), labels_b[bidx], label)
    depth = np.where(depth > cfg.max_depth, np.inf, depth)
    label = np.where(np.isinf(depth), SKY, label)

    p = c[None] + d_world * np.where(np.isfinite(depth), depth, 0.0)[:, None]
    normal = np.zeros((N, 3))
    normal[:, 1] = -1.0                                   # ground faces up (-y)
    hit_axis = axis[np.arange(N), bidx]
    sign = -np.sign(d_world[np.arange(N), hit_axis])
    box_n = np.zeros((N, 3))
    box_n[np.arange(N), hit_axis] = sign
    normal = np.where(is_box[:, None], box_n, normal)

    rgb = np.zeros((N, 3))
    sun = layout["sun"]
    shade = 0.45 + 0.55 * np.clip(normal @ (-sun), 0.0, 1.0)

    # ground texture: asphalt, lane markings, pavement
    x, z = p[:, 0], p[:, 2]
    asphalt = 0.33 + 0.08 * _hash(np.floor(x * 3), np.floor(z * 3))
    marking = (np.abs(x) < 0.08) & (np.mod(z, 6.0) < 3.0) | (np.abs(np.abs(x) - 3.5) < 0.1)
    pave = np.abs(x) > 4.0
    gval = np.where(marking, 0.9, np.where(pave, 0.55 + 0.05 * _hash(np.floor(x * 2), np.floor(z * 2)), asphalt))
    ground_rgb = np.stack([gval, gval, gval * 0.97], -1)

    # box texture: albedo plus window grid on facades
    base = np.stack([b.color for b in boxes])[bidx]
    facade_u = np.where(hit_axis == 0, z, x)
    h_above = g - p[:, 1]
    win = (np.mod(facade_u, 2.0) < 1.0) & (np.mod(h_above, 3.0) > 1.4)
    is_bld = labels_b[bidx] == BUILDING
    tall = (his[bidx, 1] - los[bidx, 1]) > 3.0
    box_rgb = np.where((win & is_bld & tall)[:, None], base * 0.35 + np.array([0.05, 0.08, 0.15]), base)
    car_glass = (labels_b[bidx] == DYNAMIC) & (h_above > 0.9)
    box_rgb = np.where(car_glass[:, None], box_rgb * 0.4, box_rgb)
    box_rgb = box_rgb * (0.92 + 0.08 * _hash(np.floor(p[:, 0] * 5 + p[:, 2] * 5), np.floor(p[:, 1] * 5)))[:, None]

    rgb = np.where(is_box[:, None], box_rgb, ground_rgb) * shade[:, None]
    sky_t = np.clip((pix[:, 1] / H), 0, 1)[:, None]
    sky = np.array([0.45, 0.65, 0.95]) * (1 - sky_t) + np.array([0.85, 0.9, 0.95]) * sky_t
    sky_mask = label == SKY
    rgb = np.where(sky_mask[:, None], sky, rgb)

    return {
        "rgb": np.clip(rgb, 0, 1).reshape(H, W, 3).astype(np.float32),
        "depth": depth.reshape(H, W).astype(np.float32),
        "label": label.reshape(H, W).astype(np.int8),
        "pose": pose.astype(np.float32),
    }


def render_sequence(seed: int, cfg: SceneConfig) -> dict:
    rng = np.random.default_rng(seed)
    layout = sample_layout(rng, cfg)
    frames = [render_frame(layout, cfg, t) for t in range(cfg.n_frames)]
    out = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
    out["K"] = intrinsics(cfg).astype(np.float32)
    return out
