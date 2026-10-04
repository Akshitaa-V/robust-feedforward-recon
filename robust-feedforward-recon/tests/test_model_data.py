import math

import numpy as np
import torch

from robustrecon.data import SKY_TARGET, WindowDataset, build_split
from robustrecon.evaluate import ground_plane_depth, predict_sequences
from robustrecon.model import MAX_DEPTH, MIN_DEPTH, DepthUNet, PhotometricNorm, depth_loss
from robustrecon.scenes import SceneConfig

CFG = SceneConfig()


def test_build_split_caches(tmp_path):
    a = build_split("val", 3, CFG, tmp_path)
    b = build_split("val", 3, CFG, tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 1
    np.testing.assert_array_equal(a["depth"], b["depth"])


def test_splits_do_not_overlap(tmp_path):
    tr = build_split("train", 2, CFG, tmp_path)
    te = build_split("test", 2, CFG, tmp_path)
    assert not np.array_equal(tr["rgb"][0], te["rgb"][0])


def test_window_dataset_shapes(tmp_path):
    data = build_split("val", 2, CFG, tmp_path)
    ds = WindowDataset(data, context=1, augment=True, train_corruptions=["fog"], return_clean=True)
    item = ds[0]
    assert item["frames"].shape == (9, CFG.height, CFG.width)
    assert item["clean"].shape == (9, CFG.height, CFG.width)
    assert torch.isfinite(item["depth"]).all() and item["depth"].max() <= SKY_TARGET
    assert len(ds) == 2 * 3


def test_single_view_dataset(tmp_path):
    data = build_split("val", 2, CFG, tmp_path)
    ds = WindowDataset(data, context=0)
    assert ds[0]["frames"].shape == (3, CFG.height, CFG.width)


def test_model_output_range():
    for views, norm in [(1, False), (3, True)]:
        m = DepthUNet(n_views=views, photometric_norm=norm)
        out = m(torch.rand(2, 3 * views, CFG.height, CFG.width))
        assert out.shape == (2, CFG.height, CFG.width)
        assert out.min() >= math.log(MIN_DEPTH) - 1e-5 and out.max() <= math.log(MAX_DEPTH) + 1e-5


def test_photometric_norm_removes_exposure():
    x = torch.rand(1, 9, 16, 16)
    norm = PhotometricNorm(3)
    torch.testing.assert_close(norm(x), norm(x * 0.5 + 0.05), atol=1e-2, rtol=1e-2)


def test_loss_zero_for_perfect_prediction():
    d = torch.rand(2, 8, 8) * 10 + 1
    assert depth_loss(torch.log(d), d).item() < 1e-6


def test_model_can_overfit_one_batch(tmp_path):
    torch.manual_seed(0)
    data = build_split("val", 2, CFG, tmp_path)
    ds = WindowDataset(data, context=1)
    batch = torch.stack([ds[i]["frames"] for i in range(4)])
    depth = torch.stack([ds[i]["depth"] for i in range(4)])
    m = DepthUNet(n_views=3, widths=(16, 32))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        loss = depth_loss(m(batch), depth)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


def test_predict_sequences_shape(tmp_path):
    data = build_split("val", 2, CFG, tmp_path)
    m = DepthUNet(n_views=3, widths=(16, 32)).eval()
    pred = predict_sequences(m, data["rgb"], 1, [1, 2, 3])
    assert pred.shape == (2, 3, CFG.height, CFG.width)


def test_ground_plane_baseline_exact_on_road(tmp_path):
    data = build_split("val", 1, CFG, tmp_path)
    gp = ground_plane_depth(CFG, data["K"])
    lab, d = data["label"][0, 0], data["depth"][0, 0]
    road = lab == 1
    np.testing.assert_allclose(gp[road], d[road], rtol=0.03)
