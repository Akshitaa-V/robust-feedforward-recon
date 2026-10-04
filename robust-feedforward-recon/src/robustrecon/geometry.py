"""Back-projection, reprojection, point-cloud and occupancy utilities (NumPy)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def backproject(depth: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Depth map (H,W) -> camera-frame points (N,3) for pixels in mask."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    if mask is None:
        mask = np.isfinite(depth)
    z = depth[mask]
    x = (u[mask] + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v[mask] + 0.5 - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], -1)


def transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def reprojection_consistency(d_a, d_b, gt_a, gt_b, pose_a, pose_b, K, static_a, rel_tol=0.03):
    """Mean relative depth disagreement between frame a's prediction, warped
    into frame b with the true poses, and frame b's prediction.

    Only static pixels that are truly visible in both frames are used: a pixel
    counts if warping the ground-truth depth lands on ground truth within
    rel_tol, which removes disocclusions. Returns (error, n_pixels).
    """
    H, W = d_a.shape
    mask = static_a & np.isfinite(gt_a)
    v, u = np.mgrid[0:H, 0:W]
    uu, vv = u[mask], v[mask]
    rel = np.linalg.inv(pose_b) @ pose_a

    def warp(depth_a):
        z = depth_a[mask]
        pts = np.stack([(uu + 0.5 - K[0, 2]) / K[0, 0] * z, (vv + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1)
        pb = transform(pts, rel)
        ub = np.floor(pb[:, 0] / pb[:, 2] * K[0, 0] + K[0, 2]).astype(int)
        vb = np.floor(pb[:, 1] / pb[:, 2] * K[1, 1] + K[1, 2]).astype(int)
        return pb[:, 2], ub, vb

    z_gt, ub, vb = warp(gt_a)
    inside = (z_gt > 0) & (ub >= 0) & (ub < W) & (vb >= 0) & (vb < H)
    ub_c, vb_c = np.clip(ub, 0, W - 1), np.clip(vb, 0, H - 1)
    target_gt = gt_b[vb_c, ub_c]
    visible = inside & np.isfinite(target_gt) & (np.abs(z_gt - target_gt) < rel_tol * target_gt)
    if visible.sum() == 0:
        return float("nan"), 0
    # warp the prediction of frame a with its own predicted depth
    z_pred, ubp, vbp = warp(d_a)
    ok = visible & (ubp >= 0) & (ubp < W) & (vbp >= 0) & (vbp < H) & (z_pred > 0)
    target = d_b[np.clip(vbp, 0, H - 1), np.clip(ubp, 0, W - 1)]
    ok &= np.isfinite(target) & (target > 0)
    err = np.abs(z_pred[ok] - target[ok]) / target[ok]
    return float(err.mean()) if ok.any() else float("nan"), int(ok.sum())


def chamfer(p: np.ndarray, q: np.ndarray, max_points: int = 4000, seed: int = 0) -> float:
    """Symmetric mean nearest-neighbour distance in metres."""
    rng = np.random.default_rng(seed)
    if len(p) > max_points:
        p = p[rng.choice(len(p), max_points, replace=False)]
    if len(q) > max_points:
        q = q[rng.choice(len(q), max_points, replace=False)]
    d_pq, _ = cKDTree(q).query(p)
    d_qp, _ = cKDTree(p).query(q)
    return float(0.5 * (d_pq.mean() + d_qp.mean()))


def bev_occupancy(points: np.ndarray, camera_height: float, x_range=(-10.0, 10.0), z_range=(0.0, 30.0),
                  cell=1.0, min_height=0.3) -> np.ndarray:
    """Bird's-eye-view occupancy grid of obstacles (points above the ground).

    1 m cells over the near 30 m: with 0.5 m cells to 40 m, a uniform 3 %
    depth error on ground truth already drops IoU to about 0.2, so the
    finer grid mostly measures pixel quantisation at range.
    """
    nx = int((x_range[1] - x_range[0]) / cell)
    nz = int((z_range[1] - z_range[0]) / cell)
    grid = np.zeros((nz, nx), dtype=bool)
    above = points[:, 1] < camera_height - min_height
    p = points[above]
    ix = np.floor((p[:, 0] - x_range[0]) / cell).astype(int)
    iz = np.floor((p[:, 2] - z_range[0]) / cell).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
    grid[iz[ok], ix[ok]] = True
    return grid


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")
