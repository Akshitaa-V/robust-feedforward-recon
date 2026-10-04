"""Standard monocular depth metrics (Eigen et al.) on valid pixels."""
from __future__ import annotations

import numpy as np


def depth_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> dict:
    valid = np.isfinite(gt) & (gt > 0)
    if mask is not None:
        valid &= mask
    if valid.sum() == 0:
        return {"abs_rel": float("nan"), "rmse": float("nan"), "delta1": float("nan"), "n": 0}
    p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)
    ratio = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "n": int(valid.sum()),
    }


def aggregate(rows: list[dict]) -> dict:
    """Pixel-weighted mean over images."""
    keys = [k for k in rows[0] if k != "n"]
    w = np.array([r["n"] for r in rows], dtype=float)
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        ok = np.isfinite(v) & (w > 0)
        if k == "rmse":
            out[k] = float(np.sqrt(np.sum(w[ok] * v[ok] ** 2) / w[ok].sum())) if ok.any() else float("nan")
        else:
            out[k] = float(np.sum(w[ok] * v[ok]) / w[ok].sum()) if ok.any() else float("nan")
    out["n"] = int(w.sum())
    return out
