"""Dataset generation, caching and PyTorch datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import corruptions
from .scenes import SKY, SceneConfig, render_sequence

SPLIT_SEED_OFFSET = {"train": 0, "val": 100_000, "test": 200_000}
SKY_TARGET = 80.0


def build_split(split: str, n_sequences: int, cfg: SceneConfig, cache_dir: str | Path) -> dict:
    """Render (or load from cache) n_sequences for one split."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{split}_{n_sequences}_{cfg.height}x{cfg.width}_T{cfg.n_frames}.npz"
    if path.exists():
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    seqs = [render_sequence(SPLIT_SEED_OFFSET[split] + i, cfg) for i in range(n_sequences)]
    data = {k: np.stack([s[k] for s in seqs]) for k in ("rgb", "depth", "label", "pose")}
    data["K"] = seqs[0]["K"]
    np.savez_compressed(path, **data)
    return data


def window_indices(n_frames: int, context: int) -> list[int]:
    """Centre frames that have `context` neighbours on each side."""
    return list(range(context, n_frames - context))


class WindowDataset(Dataset):
    """Yields (frames, target) where frames stacks 2*context+1 views.

    With augment=True a random corruption from `train_corruptions` is applied
    to the whole sequence before cropping the window, so neighbouring frames
    share fog density or occluder position, as they would in a real video.
    """

    def __init__(self, data: dict, context: int, augment: bool = False,
                 train_corruptions: list[str] | None = None, p_clean: float = 0.25,
                 return_clean: bool = False, seed: int = 0):
        self.data = data
        self.context = context
        self.augment = augment
        self.train_corruptions = train_corruptions or []
        self.p_clean = p_clean
        self.return_clean = return_clean
        self.rng = np.random.default_rng(seed)
        n_seq, n_frames = data["rgb"].shape[:2]
        self.items = [(s, t) for s in range(n_seq) for t in window_indices(n_frames, max(context, 1))]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        s, t = self.items[idx]
        sl = slice(t - self.context, t + self.context + 1)
        clean = self.data["rgb"][s, sl]
        depth = self.data["depth"][s, sl]
        rgb = clean
        if self.augment and self.rng.uniform() > self.p_clean and self.train_corruptions:
            name = self.rng.choice(self.train_corruptions)
            rgb = corruptions.apply(str(name), clean, depth, int(self.rng.integers(1, 4)), self.rng)
        target = self.data["depth"][s, t].copy()
        target[self.data["label"][s, t] == SKY] = SKY_TARGET
        target = np.where(np.isfinite(target), target, SKY_TARGET)
        out = {"frames": to_tensor(rgb), "depth": torch.from_numpy(target.astype(np.float32))}
        if self.return_clean:
            out["clean"] = to_tensor(clean)
        return out


def to_tensor(frames: np.ndarray) -> torch.Tensor:
    """(V,H,W,3) -> (3V,H,W) float tensor."""
    V, H, W, _ = frames.shape
    return torch.from_numpy(np.ascontiguousarray(frames.transpose(0, 3, 1, 2).reshape(V * 3, H, W)))
