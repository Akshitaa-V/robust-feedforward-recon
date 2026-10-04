"""Train one model variant from a YAML config.

    python -m robustrecon.train --config configs/default.yaml --variant robust --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import WindowDataset, build_split
from .model import DepthUNet, count_parameters, depth_loss
from .scenes import SceneConfig


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(variant: dict) -> DepthUNet:
    n_views = 2 * variant["context"] + 1
    return DepthUNet(n_views=n_views, photometric_norm=variant.get("photometric_norm", False))


def train(cfg: dict, variant_name: str, seed: int, out_dir: str | Path) -> dict:
    variant = cfg["variants"][variant_name]
    set_seed(seed)
    scene_cfg = SceneConfig(**cfg["scene"])
    data = build_split("train", cfg["data"]["n_train"], scene_cfg, cfg["data"]["cache_dir"])
    consistency = float(variant.get("consistency_weight", 0.0))
    ds = WindowDataset(data, context=variant["context"], augment=variant.get("augment", False),
                       train_corruptions=cfg["train_corruptions"], p_clean=cfg["train"]["p_clean"],
                       return_clean=consistency > 0, seed=seed)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    model = build_model(variant)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-4)
    epochs = cfg["train"]["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["train"]["lr"], total_steps=epochs * len(loader))
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for batch in loader:
            log_pred = model(batch["frames"])
            loss = depth_loss(log_pred, batch["depth"])
            if consistency > 0:
                log_clean = model(batch["clean"])
                loss = loss + depth_loss(log_clean, batch["depth"])
                loss = loss + consistency * (log_pred - log_clean.detach()).abs().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(batch["depth"])
            n += len(batch["depth"])
        history.append({"epoch": epoch + 1, "loss": total / n, "seconds": round(time.time() - t0, 1)})
        print(f"[{variant_name} seed {seed}] epoch {epoch + 1}/{epochs} loss {total / n:.4f} ({time.time() - t0:.0f}s)",
              flush=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    meta = {"variant": variant_name, "seed": seed, "params": count_parameters(model),
            "train_windows": len(ds), "history": history, "config": variant}
    (out_dir / "train_log.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    train(cfg, args.variant, args.seed, Path(cfg["runs_dir"]) / f"{args.variant}_seed{args.seed}")


if __name__ == "__main__":
    main()
