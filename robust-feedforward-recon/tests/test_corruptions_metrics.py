import numpy as np
import pytest

from robustrecon import corruptions
from robustrecon.metrics import aggregate, depth_metrics
from robustrecon.scenes import SceneConfig, render_sequence

SEQ = render_sequence(3, SceneConfig())


@pytest.mark.parametrize("name", list(corruptions.CORRUPTIONS))
@pytest.mark.parametrize("severity", [1, 2, 3])
def test_corruption_output_valid(name, severity):
    out = corruptions.apply(name, SEQ["rgb"], SEQ["depth"], severity, np.random.default_rng(0))
    assert out.shape == SEQ["rgb"].shape and out.dtype == np.float32
    assert out.min() >= 0 and out.max() <= 1
    assert not np.allclose(out, SEQ["rgb"])


@pytest.mark.parametrize("name", list(corruptions.CORRUPTIONS))
def test_corruption_is_reproducible(name):
    a = corruptions.apply(name, SEQ["rgb"], SEQ["depth"], 2, np.random.default_rng(5))
    b = corruptions.apply(name, SEQ["rgb"], SEQ["depth"], 2, np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", ["fog", "low_light", "snow"])
def test_severity_increases_damage(name):
    errs = [np.abs(corruptions.apply(name, SEQ["rgb"], SEQ["depth"], s, np.random.default_rng(0)) - SEQ["rgb"]).mean()
            for s in (1, 2, 3)]
    assert errs[0] < errs[1] < errs[2]


def test_fog_hides_far_more_than_near():
    out = corruptions.apply("fog", SEQ["rgb"], SEQ["depth"], 3, np.random.default_rng(0))
    d = SEQ["depth"]
    diff = np.abs(out - SEQ["rgb"]).mean(-1)
    near = np.isfinite(d) & (d < 6)
    far = np.isfinite(d) & (d > 30)
    assert diff[far].mean() > diff[near].mean()


def test_occluder_is_persistent_over_time():
    out = corruptions.apply("occlusion", SEQ["rgb"], SEQ["depth"], 3, np.random.default_rng(1))
    flat = np.all(out == out[0], axis=(0, 3))      # pixels identical in every frame
    assert flat.mean() > 0.15


def test_clean_is_identity():
    np.testing.assert_array_equal(corruptions.apply("clean", SEQ["rgb"], SEQ["depth"], 1, None), SEQ["rgb"])


def test_bad_inputs_raise():
    with pytest.raises(KeyError):
        corruptions.apply("hail", SEQ["rgb"], SEQ["depth"], 1, np.random.default_rng(0))
    with pytest.raises(ValueError):
        corruptions.apply("fog", SEQ["rgb"], SEQ["depth"], 4, np.random.default_rng(0))


def test_perfect_prediction_metrics():
    gt = np.array([[1.0, 2.0], [np.inf, 4.0]])
    m = depth_metrics(gt.copy(), gt)
    assert m["abs_rel"] == 0 and m["rmse"] == 0 and m["delta1"] == 1 and m["n"] == 3


def test_metrics_known_values():
    gt = np.array([2.0, 4.0])
    m = depth_metrics(np.array([3.0, 4.0]), gt)
    assert m["abs_rel"] == pytest.approx(0.25)
    assert m["delta1"] == pytest.approx(0.5)


def test_aggregate_is_pixel_weighted():
    rows = [{"abs_rel": 0.1, "rmse": 1.0, "delta1": 1.0, "n": 1},
            {"abs_rel": 0.4, "rmse": 1.0, "delta1": 0.0, "n": 3}]
    assert aggregate(rows)["abs_rel"] == pytest.approx(0.325)
