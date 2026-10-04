"""Summarise metrics_raw.json into tables and figures.

    python -m robustrecon.report --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import build_split
from .evaluate import corrupt_split, predict_sequences
from .scenes import SceneConfig
from .train import build_model, load_config

# The ground-plane prior is kept in metrics_raw.json but left out of the tables:
# it assigns far depth to every pixel above the horizon, so its AbsRel (about 2.1)
# says more about buildings than about robustness.
ORDER = ["single_clean", "multi_clean", "multi_aug", "robust"]
LABELS = {
    "ground_plane": "Ground-plane prior (no image)",
    "single_clean": "Single frame, clean training",
    "multi_clean": "3 frames, clean training",
    "multi_aug": "3 frames + corruption augmentation",
    "robust": "3 frames + aug + photometric norm + consistency",
}
COLORS = {"ground_plane": "#9a9993", "single_clean": "#2a78d6", "multi_clean": "#eb6834",
          "multi_aug": "#1baf7a", "robust": "#eda100"}
METRICS = ["abs_rel", "rmse", "delta1", "dynamic_abs_rel", "temporal_inconsistency", "chamfer_m", "bev_iou"]
INK, MUTED, GRID = "#2b2b29", "#6b6a65", "#e4e3de"


def summarise(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["variant"], r["condition"], r["severity"])].append(r)
    out = {}
    for (v, c, s), rs in groups.items():
        out[f"{v}|{c}|{s}"] = {m: {"mean": float(np.mean([r[m] for r in rs])),
                                   "spread": float((max(r[m] for r in rs) - min(r[m] for r in rs)) / 2)}
                               for m in METRICS}
        out[f"{v}|{c}|{s}"]["n_seeds"] = len(rs)
    return out


def get(summary, v, c, s, m="abs_rel"):
    return summary[f"{v}|{c}|{s}"][m]["mean"]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def write_tables(summary, cfg, out: Path) -> str:
    sev = cfg["eval_severity"]
    conds = cfg["eval_conditions"]
    variants = [v for v in ORDER if f"{v}|clean|0" in summary]
    lines = [f"### AbsRel by condition (severity {sev}; lower is better; mean over seeds)\n",
             "| Method | " + " | ".join(c + (" (unseen)" if c not in cfg["train_corruptions"] + ["clean"] else "")
                                       for c in conds) + " | Mean corrupted |",
             "|---" * (len(conds) + 2) + "|"]
    for v in variants:
        vals = [get(summary, v, c, 0 if c == "clean" else sev) for c in conds]
        corr = np.mean(vals[1:])
        lines.append(f"| {LABELS[v]} | " + " | ".join(f"{x:.3f}" for x in vals) + f" | {corr:.3f} |")
    lines += ["", f"### All metrics, averaged over the six corruptions (severity {sev})\n",
              "| Method | AbsRel | RMSE (m) | delta<1.25 | AbsRel moving objects | Temporal inconsistency | Chamfer (m) | BEV occupancy IoU |",
              "|---|---|---|---|---|---|---|---|"]
    for v in variants:
        vals = [np.mean([get(summary, v, c, sev, m) for c in conds[1:]]) for m in METRICS]
        lines.append(f"| {LABELS[v]} | " + " | ".join(f"{x:.3f}" for x in vals) + " |")
    lines += ["", "### Clean images\n",
              "| Method | AbsRel | RMSE (m) | delta<1.25 | AbsRel moving objects | Temporal inconsistency | Chamfer (m) | BEV occupancy IoU |",
              "|---|---|---|---|---|---|---|---|"]
    for v in variants:
        lines.append(f"| {LABELS[v]} | " + " | ".join(f"{get(summary, v, 'clean', 0, m):.3f}" for m in METRICS) + " |")
    lines += ["", "### Severity sweep (AbsRel)\n", "| Condition | Method | 1 | 2 | 3 |", "|---|---|---|---|---|"]
    for c in cfg["severity_sweep"]:
        for v in variants:
            lines.append(f"| {c} | {LABELS[v]} | " + " | ".join(f"{get(summary, v, c, s):.3f}" for s in (1, 2, 3)) + " |")
    text = "\n".join(lines) + "\n"
    (out / "results.md").write_text(text)
    return text


def fig_conditions(summary, cfg, out: Path):
    sev = cfg["eval_severity"]
    conds = cfg["eval_conditions"]
    variants = [v for v in ORDER if f"{v}|clean|0" in summary]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    n = len(variants)
    w = 0.8 / n
    x = np.arange(len(conds))
    for i, v in enumerate(variants):
        vals = [get(summary, v, c, 0 if c == "clean" else sev) for c in conds]
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.9, color=COLORS[v], label=LABELS[v])
    ax.set_xticks(x, [c.replace("_", " ") + ("\n(unseen)" if c not in cfg["train_corruptions"] + ["clean"] else "")
                      for c in conds], color=INK)
    ax.set_ylabel("AbsRel (lower is better)", color=INK)
    ax.set_title(f"Depth error per condition, severity {sev}", color=INK, loc="left", fontsize=11)
    style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "abs_rel_by_condition.png", dpi=140)
    plt.close(fig)


def fig_severity(summary, cfg, out: Path):
    variants = [v for v in ORDER if f"{v}|clean|0" in summary]
    conds = cfg["severity_sweep"]
    fig, axes = plt.subplots(1, len(conds), figsize=(10, 3.7), sharey=True)
    for ax, c in zip(axes, conds):
        for v in variants:
            ys = [get(summary, v, "clean", 0)] + [get(summary, v, c, s) for s in (1, 2, 3)]
            ax.plot([0, 1, 2, 3], ys, color=COLORS[v], linewidth=2, marker="o", markersize=5, label=LABELS[v])
        ax.set_title(c.replace("_", " ") + (" (unseen)" if c not in cfg["train_corruptions"] else ""),
                     color=INK, fontsize=10, loc="left")
        ax.set_xticks([0, 1, 2, 3], ["clean", "1", "2", "3"])
        ax.set_xlabel("severity", color=MUTED, fontsize=9)
        style(ax)
    axes[0].set_ylabel("AbsRel", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=2, loc="lower center", labelcolor=INK)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out / "severity_sweep.png", dpi=140)
    plt.close(fig)


def fig_qualitative(cfg, out: Path, seq: int = 4):
    scene_cfg = SceneConfig(**cfg["scene"])
    data = build_split("test", cfg["data"]["n_test"], scene_cfg, cfg["data"]["cache_dir"])
    sub = {k: (v[seq:seq + 1] if k != "K" else v) for k, v in data.items()}
    shown = ["single_clean", "robust"]
    models = {}
    for v in shown:
        m = build_model(cfg["variants"][v])
        m.load_state_dict(torch.load(Path(cfg["runs_dir"]) / f"{v}_seed0" / "model.pt", map_location="cpu"))
        models[v] = m.eval()
    conds = ["clean", "fog", "low_light", "snow", "occlusion"]
    gt = np.where(np.isfinite(sub["depth"][0, 2]), sub["depth"][0, 2], 80.0)
    fig, axes = plt.subplots(len(conds), 4, figsize=(9, 1.75 * len(conds)))
    kw = dict(cmap="magma_r", vmin=np.log(2), vmax=np.log(80))
    for i, c in enumerate(conds):
        rgb = sub["rgb"] if c == "clean" else corrupt_split(sub, c, 3 if c == "low_light" else 2)
        axes[i, 0].imshow(rgb[0, 2])
        axes[i, 1].imshow(np.log(gt), **kw)
        for j, v in enumerate(shown):
            ctx = cfg["variants"][v]["context"]
            axes[i, 2 + j].imshow(np.log(predict_sequences(models[v], rgb, ctx, [2])[0, 0]), **kw)
        axes[i, 0].set_ylabel(c.replace("_", " "), color=INK, fontsize=9)
    for j, t in enumerate(["input (centre frame)", "ground truth", "single frame, clean", "robust (ours)"]):
        axes[0, j].set_title(t, color=INK, fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "qualitative.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["results_dir"])
    rows = json.loads((out / "metrics_raw.json").read_text())
    summary = summarise(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(write_tables(summary, cfg, out))
    fig_conditions(summary, cfg, out)
    fig_severity(summary, cfg, out)
    fig_qualitative(cfg, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
