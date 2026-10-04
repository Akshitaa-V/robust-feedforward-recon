import numpy as np
import pytest

from robustrecon.geometry import backproject, bev_occupancy, chamfer, iou, reprojection_consistency, transform
from robustrecon.scenes import DYNAMIC, GROUND, SKY, SceneConfig, camera_pose, intrinsics, render_sequence

CFG = SceneConfig()


@pytest.fixture(scope="module")
def seq():
    return render_sequence(7, CFG)


def test_shapes_and_ranges(seq):
    T, H, W = CFG.n_frames, CFG.height, CFG.width
    assert seq["rgb"].shape == (T, H, W, 3)
    assert seq["depth"].shape == (T, H, W)
    assert seq["rgb"].min() >= 0 and seq["rgb"].max() <= 1
    finite = seq["depth"][np.isfinite(seq["depth"])]
    assert finite.min() > 0 and finite.max() <= CFG.max_depth


def test_rendering_is_deterministic():
    a, b = render_sequence(11, CFG), render_sequence(11, CFG)
    np.testing.assert_array_equal(a["rgb"], b["rgb"])
    np.testing.assert_array_equal(a["depth"], b["depth"])


def test_sky_has_no_depth_and_ground_matches_plane(seq):
    d, lab = seq["depth"][0], seq["label"][0]
    assert np.all(np.isinf(d[lab == SKY]))
    K = seq["K"]
    v = np.nonzero(lab == GROUND)[0]
    expected = CFG.camera_height * K[1, 1] / (v + 0.5 - K[1, 2])   # frame 0: camera yaw is tiny
    np.testing.assert_allclose(d[lab == GROUND], expected, rtol=0.02)


def test_dynamic_objects_present_and_moving():
    found = False
    for s in range(5):
        seq = render_sequence(s, CFG)
        if (seq["label"] == DYNAMIC).sum() > 20:
            found = True
            assert not np.array_equal(seq["label"][0] == DYNAMIC, seq["label"][-1] == DYNAMIC)
    assert found


def test_camera_moves_forward():
    layout = {"yaw0": 0.0, "yaw_rate": 0.0, "speed": 1.0, "x_drift": 0.0}
    assert camera_pose(layout, 3)[2, 3] == pytest.approx(3.0)


def test_backproject_roundtrip(seq):
    d = seq["depth"][0]
    K = seq["K"]
    pts = backproject(d, K)
    proj = pts @ K.T
    u = proj[:, 0] / proj[:, 2]
    v, uu = np.mgrid[0:CFG.height, 0:CFG.width]
    np.testing.assert_allclose(u, uu[np.isfinite(d)] + 0.5, atol=1e-3)


def test_ground_truth_is_temporally_consistent(seq):
    """Warping true depth with true poses must agree with itself."""
    gt = seq["depth"]
    err, n = reprojection_consistency(gt[1], gt[2], gt[1], gt[2], seq["pose"][1], seq["pose"][2], seq["K"],
                                      seq["label"][1] != DYNAMIC)
    assert n > 500
    assert err < 0.02


def test_scaled_depth_is_inconsistent(seq):
    gt = seq["depth"]
    err, _ = reprojection_consistency(gt[1] * 1.3, gt[2], gt[1], gt[2], seq["pose"][1], seq["pose"][2], seq["K"],
                                      seq["label"][1] != DYNAMIC)
    assert err > 0.1


def test_chamfer_zero_for_identical_and_grows_with_shift():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 10, (500, 3))
    assert chamfer(p, p) == pytest.approx(0.0)
    assert chamfer(p, p + [0.5, 0, 0]) > chamfer(p, p + [0.1, 0, 0])


def test_bev_occupancy_ignores_ground(seq):
    gt = seq["depth"][0]
    pts = backproject(gt, seq["K"], seq["label"][0] == GROUND)
    assert bev_occupancy(pts, CFG.camera_height).sum() == 0


def test_iou():
    a = np.zeros((4, 4), bool)
    a[:2] = True
    assert iou(a, a) == 1.0
    assert iou(a, ~a) == 0.0


def test_transform_identity():
    p = np.ones((3, 3))
    np.testing.assert_array_equal(transform(p, np.eye(4)), p)


def test_intrinsics_centre():
    K = intrinsics(CFG)
    assert K[0, 2] == CFG.width / 2 and K[1, 2] == CFG.height / 2
