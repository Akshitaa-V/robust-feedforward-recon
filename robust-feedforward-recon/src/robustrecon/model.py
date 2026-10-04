"""Feed-forward multi-view depth network (U-Net with early frame fusion)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MIN_DEPTH, MAX_DEPTH = 1.0, 80.0


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class PhotometricNorm(nn.Module):
    """Per-frame, per-channel standardisation of the input images.

    Removes global exposure and contrast, which low light, fog and exposure
    changes mostly alter, before the network sees the frames.
    """

    def __init__(self, n_views: int, eps: float = 1e-3):
        super().__init__()
        self.n_views = n_views
        self.eps = eps

    def forward(self, x):
        B, C, H, W = x.shape
        v = x.view(B, self.n_views, 3, H, W)
        mean = v.mean(dim=(3, 4), keepdim=True)
        std = v.std(dim=(3, 4), keepdim=True)
        return ((v - mean) / (std + self.eps)).view(B, C, H, W)


class DepthUNet(nn.Module):
    def __init__(self, n_views: int = 3, widths=(24, 48, 72, 96), photometric_norm: bool = False):
        super().__init__()
        self.n_views = n_views
        self.norm = PhotometricNorm(n_views) if photometric_norm else None
        cin = 3 * n_views + 2                     # +2: normalised pixel coordinates (geometric prior)
        self.enc = nn.ModuleList()
        for w in widths:
            self.enc.append(ConvBlock(cin, w))
            cin = w
        self.dec = nn.ModuleList()
        for w_skip, w in zip(reversed(widths[:-1]), reversed(widths[1:])):
            self.dec.append(ConvBlock(w + w_skip, w_skip))
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        else:
            x = x * 2.0 - 1.0
        B, _, H, W = x.shape
        ys = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        xs = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        x = torch.cat([x, xs, ys], 1)
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for blk in self.dec:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = blk(torch.cat([x, skip], 1))
        s = torch.sigmoid(self.head(x))[:, 0]
        return s * (math.log(MAX_DEPTH) - math.log(MIN_DEPTH)) + math.log(MIN_DEPTH)   # log depth


def depth_loss(log_pred: torch.Tensor, depth: torch.Tensor, grad_weight: float = 0.5) -> torch.Tensor:
    log_gt = torch.log(depth)
    diff = log_pred - log_gt
    l1 = diff.abs().mean()
    gx = (diff[:, :, 1:] - diff[:, :, :-1]).abs().mean()
    gy = (diff[:, 1:] - diff[:, :-1]).abs().mean()
    return l1 + grad_weight * (gx + gy)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
