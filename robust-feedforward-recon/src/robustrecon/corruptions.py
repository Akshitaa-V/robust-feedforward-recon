"""Physically motivated, temporally coherent image corruptions.

Each corruption takes a clean sequence (T,H,W,3) in [0,1] and its depth
(T,H,W, inf = sky) and returns a corrupted sequence. Severity runs 1..3.
Fog follows the Koschmieder model I = J t + A (1 - t), t = exp(-beta d), so
it depends on the true depth, as it does outdoors.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

SKY_DEPTH = 80.0


def _depth_for_scattering(depth: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(depth), depth, SKY_DEPTH)[..., None]


def fog(rgb, depth, severity, rng):
    beta = [0.03, 0.06, 0.10][severity - 1]
    airlight = rng.uniform(0.75, 0.9)
    t = np.exp(-beta * _depth_for_scattering(depth))
    return rgb * t + airlight * (1.0 - t)


def low_light(rgb, depth, severity, rng):
    gain = [0.30, 0.15, 0.07][severity - 1]
    noise = [0.015, 0.025, 0.035][severity - 1]
    dark = (rgb ** 1.4) * gain
    shot = rng.normal(0.0, 1.0, rgb.shape) * np.sqrt(np.maximum(dark, 1e-6)) * 0.08
    read = rng.normal(0.0, noise, rgb.shape)
    return dark + shot + read


def _smooth_noise(rng, T, H, W, cells=4, drift=0.6):
    """Low-frequency field that drifts over time (clouds, moving shadows)."""
    gh, gw = cells + 2, cells * 2 + 2
    grid = rng.uniform(0, 1, (gh, gw + T))
    ys = np.linspace(0, cells, H)
    out = np.empty((T, H, W))
    for t in range(T):
        xs = np.linspace(0, cells * 1.5, W) + drift * t
        x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
        fx, fy = xs - x0, ys - y0
        g = grid
        a = g[y0][:, x0] * (1 - fx) + g[y0][:, x0 + 1] * fx
        b = g[y0 + 1][:, x0] * (1 - fx) + g[y0 + 1][:, x0 + 1] * fx
        out[t] = a * (1 - fy[:, None]) + b * fy[:, None]
    return out


def shadows(rgb, depth, severity, rng):
    """Moving cast shadows plus frame-to-frame exposure changes."""
    T, H, W, _ = rgb.shape
    strength = [0.45, 0.6, 0.75][severity - 1]
    field = _smooth_noise(rng, T, H, W)
    mask = np.clip((field - 0.45) * 6.0, 0.0, 1.0)
    exposure = rng.uniform(1 - 0.25 * severity, 1 + 0.15 * severity, (T, 1, 1, 1))
    return rgb * (1.0 - strength * mask[..., None]) * exposure


def rain(rgb, depth, severity, rng):
    T, H, W, _ = rgb.shape
    n = [60, 140, 260][severity - 1]
    out = fog(rgb, depth, 1, rng) * 0.9 if severity > 1 else rgb * 0.92
    out = out.copy()
    slope = rng.uniform(-0.3, 0.3)
    for t in range(T):
        ys = rng.integers(0, H, n)
        xs = rng.integers(0, W, n)
        length = rng.integers(3, 7, n)
        for y, x, L in zip(ys, xs, length):
            for k in range(L):
                yy, xx = y + k, int(round(x + slope * k))
                if 0 <= yy < H and 0 <= xx < W:
                    out[t, yy, xx] = 0.65 * out[t, yy, xx] + 0.35 * 0.85
    return out


def snow(rgb, depth, severity, rng):
    T, H, W, _ = rgb.shape
    n = [80, 180, 320][severity - 1]
    haze = 0.15 + 0.1 * severity
    out = rgb * (1 - haze) + haze * 0.9
    out = out.copy()
    for t in range(T):
        ys = rng.integers(0, H, n)
        xs = rng.integers(0, W, n)
        big = rng.uniform(0, 1, n) < 0.25
        out[t, ys, xs] = 0.97
        yb, xb = ys[big], np.clip(xs[big] + 1, 0, W - 1)
        out[t, yb, xb] = 0.95
    return out


def occlusion(rgb, depth, severity, rng):
    """Long-term occluder fixed in the image (dirt or an object on the lens
    housing) plus short-lived occluders that appear in single frames."""
    T, H, W, _ = rgb.shape
    out = rgb.copy()
    frac = [0.08, 0.15, 0.25][severity - 1]
    h = int(np.sqrt(frac * H * W * 0.6))
    w = int(frac * H * W / max(h, 1))
    y0, x0 = rng.integers(H // 4, H - h), rng.integers(0, W - w)
    col = rng.uniform(0.15, 0.4, 3)
    out[:, y0:y0 + h, x0:x0 + w] = col
    for t in range(T):
        if rng.uniform() < 0.5:
            hh, ww = rng.integers(6, 14), rng.integers(6, 18)
            yy, xx = rng.integers(0, H - hh), rng.integers(0, W - ww)
            out[t, yy:yy + hh, xx:xx + ww] = rng.uniform(0.1, 0.9, 3)
    return out


CORRUPTIONS: dict[str, Callable] = {
    "fog": fog,
    "rain": rain,
    "snow": snow,
    "low_light": low_light,
    "shadows": shadows,
    "occlusion": occlusion,
}


def apply(name: str, rgb: np.ndarray, depth: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    if name == "clean":
        return rgb.copy()
    if name not in CORRUPTIONS:
        raise KeyError(f"unknown corruption '{name}'")
    if severity not in (1, 2, 3):
        raise ValueError("severity must be 1, 2 or 3")
    out = CORRUPTIONS[name](rgb.astype(np.float64), depth, severity, rng)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
