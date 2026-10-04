"""Evaluate trained variants and the ground-plane baseline under every condition.

    python -m robustrecon.evaluate --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import torch

from . import corruptions
from .data import build_split, to_tensor
from .geometry import backproject, bev_occupancy, chamfer, iou, reprojection_consistency
from .metrics import aggregate, depth_metrics
from .scenes import DYNAMIC, SKY, SceneConfig
from .train import build_model, load_config

VALID_PRED_DEPTH = 60.0


def corrupt_split(data: dict, name: str, severity: int) -> np.ndarray:
    out = np.empty_like(data["rgb"])
    for s in range(len(out)):
        rng = np.random.default_rng(zlib.crc32(f"{name}-{severity}-{s}".encode()))
        out[s] = corruptions.apply(name, data["rgb"][s], data["depth"][s], severity, rng)
    return out


def ground_plane_depth(cfg: SceneConfig, K: np.ndarray) -> np.ndarray:
    """Geometry-only prior: every pixel below the horizon lies on the road."""
    v = np.arange(cfg.height)[:, None] + 0.5 - K[1, 2]
    d = np.where(v > 0, cfg.camera_height * K[1, 1] / np.maximum(v, 1e-6), 80.0)
    return np.broadcast_to(np.minimum(d, 80.0), (cfg.height, cfg.width)).astype(np.float32)


@torch.no_grad()
def predict_sequences(model, rgb: np.ndarray, context: int, centres: list[int], batch: int = 64) -> np.ndarray:
    """Returns depth (S, len(centres), H, W)."""
    S = rgb.shape[0]
    windows = [to_tensor(rgb[s, t - context:t + context + 1]) for s in range(S) for t in centres]
    preds = []
    for i in range(0, len(windows), batch):
        preds.append(torch.exp(model(torch.stack(windows[i:i + batch]))).numpy())
    return np.concatenate(preds).reshape(S, len(centres), *rgb.shape[2:4])


def score(pred: np.ndarray, data: dict, centres: list[int], cfg: SceneConfig) -> dict:
    K = data["K"]
    all_rows, dyn_rows, cons, chs, ious = [], [], [], [], []
    for s in range(pred.shape[0]):
        for j, t in enumerate(centres):
            gt = data["depth"][s, t]
            lab = data["label"][s, t]
            all_rows.append(depth_metrics(pred[s, j], gt))
            dyn_rows.append(depth_metrics(pred[s, j], gt, lab == DYNAMIC))
        for j in range(len(centres) - 1):
            ta, tb = centres[j], centres[j + 1]
            e, n = reprojection_consistency(pred[s, j], pred[s, j + 1], data["depth"][s, ta], data["depth"][s, tb],
                                            data["pose"][s, ta], data["pose"][s, tb], K,
                                            data["label"][s, ta] != DYNAMIC)
            if n:
                cons.append(e)
        jm = len(centres) // 2
        tm = centres[jm]
        gt = data["depth"][s, tm]
        p_gt = backproject(gt, K, np.isfinite(gt) & (gt < 40))
        p_pr = backproject(pred[s, jm], K, pred[s, jm] < 40)
        if len(p_pr) and len(p_gt):
            chs.append(chamfer(p_pr, p_gt, seed=s))
        p_pr_all = backproject(pred[s, jm], K, pred[s, jm] < VALID_PRED_DEPTH)
        p_gt_all = backproject(gt, K, np.isfinite(gt))
        ious.append(iou(bev_occupancy(p_pr_all, cfg.camera_height), bev_occupancy(p_gt_all, cfg.camera_height)))
    out = aggregate(all_rows)
    out["dynamic_abs_rel"] = aggregate(dyn_rows)["abs_rel"]
    out["temporal_inconsistency"] = float(np.nanmean(cons))
    out["chamfer_m"] = float(np.mean(chs))
    out["bev_iou"] = float(np.nanmean(ious))
    out.pop("n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    scene_cfg = SceneConfig(**cfg["scene"])
    data = build_split("test", cfg["data"]["n_test"], scene_cfg, cfg["data"]["cache_dir"])
    centres = [1, 2, 3]

    models = {}
    for name, variant in cfg["variants"].items():
        for seed in cfg["seeds"]:
            path = Path(cfg["runs_dir"]) / f"{name}_seed{seed}" / "model.pt"
            if not path.exists():
                continue
            m = build_model(variant)
            m.load_state_dict(torch.load(path, map_location="cpu"))
            m.eval()
            models[(name, seed)] = (m, variant["context"])
    gp = ground_plane_depth(scene_cfg, data["K"])

    sev = cfg["eval_severity"]
    jobs = [(c, sev if c != "clean" else 0) for c in cfg["eval_conditions"]]
    jobs += [(c, s) for c in cfg["severity_sweep"] for s in (1, 2, 3) if s != sev]
    results = []
    for cond, s in jobs:
        rgb = data["rgb"] if cond == "clean" else corrupt_split(data, cond, s)
        gp_pred = np.broadcast_to(gp, (rgb.shape[0], len(centres), *gp.shape))
        r = score(gp_pred, data, centres, scene_cfg)
        results.append({"variant": "ground_plane", "seed": 0, "condition": cond, "severity": s, **r})
        for (name, seed), (m, ctx) in models.items():
            pred = predict_sequences(m, rgb, ctx, centres)
            r = score(pred, data, centres, scene_cfg)
            results.append({"variant": name, "seed": seed, "condition": cond, "severity": s, **r})
            print(f"{cond:10s} s{s} {name:13s} seed{seed} abs_rel {r['abs_rel']:.3f} d1 {r['delta1']:.3f} "
                  f"tc {r['temporal_inconsistency']:.3f} iou {r['bev_iou']:.3f}", flush=True)
    out = Path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics_raw.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {out / 'metrics_raw.json'} ({len(results)} rows)")


if __name__ == "__main__":
    main()
