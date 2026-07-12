"""Save/load round-trip tests for depthba.depth.source."""

import numpy as np
import pytest

from depthba.depth.source import DepthBundle, load_bundle, save_bundle

H, W, K = 6, 8, 4


def full_bundle():
    """MDA-style: all channels present. float64 inputs to exercise coercion."""
    rng = np.random.default_rng(0)
    return DepthBundle(
        estimated_depth=rng.uniform(1, 10, (H, W)),
        means=rng.uniform(1, 10, (K, H, W)),
        weights=np.full((K, H, W), 1.0 / K),
        sigmas=rng.uniform(0.01, 1, (K, H, W)),
        confidence=rng.uniform(0, 1, (H, W)),
        sky_mask=rng.uniform(0, 1, (H, W)) > 0.8,
    )


def minimal_bundle():
    """DepthPro-style: committed map only."""
    return DepthBundle(
        estimated_depth=np.ones((H, W), np.float32),
        means=None,
        weights=None,
        sigmas=None,
        confidence=None,
        sky_mask=None,
    )


def test_full_round_trip(tmp_path):
    b = full_bundle()
    path = tmp_path / "img.npz"
    save_bundle(path, b)
    got = load_bundle(path, (H, W))
    for name in ("estimated_depth", "means", "weights", "sigmas", "confidence"):
        want = getattr(b, name).astype(np.float32)
        np.testing.assert_array_equal(getattr(got, name), want)
        assert getattr(got, name).dtype == np.float32
    np.testing.assert_array_equal(got.sky_mask, b.sky_mask)
    assert got.sky_mask.dtype == bool


def test_minimal_round_trip(tmp_path):
    path = tmp_path / "img.npz"
    save_bundle(path, minimal_bundle())
    got = load_bundle(path, (H, W))
    np.testing.assert_array_equal(got.estimated_depth, np.ones((H, W), np.float32))
    assert got.means is None
    assert got.weights is None
    assert got.sigmas is None
    assert got.confidence is None
    assert got.sky_mask is None


def test_stale_grid_rejected(tmp_path):
    path = tmp_path / "img.npz"
    save_bundle(path, minimal_bundle())
    with pytest.raises(ValueError, match="stale dump"):
        load_bundle(path, (H + 1, W))


def test_means_without_weights_rejected(tmp_path):
    b = full_bundle()
    b.weights = None
    with pytest.raises(ValueError, match="present together"):
        save_bundle(tmp_path / "img.npz", b)


def test_sigmas_without_means_rejected(tmp_path):
    b = full_bundle()
    b.means = None
    b.weights = None
    with pytest.raises(ValueError, match="sigmas present without means"):
        save_bundle(tmp_path / "img.npz", b)


def test_unknown_array_rejected(tmp_path):
    path = tmp_path / "img.npz"
    np.savez_compressed(
        path, estimated_depth=np.ones((H, W), np.float32), bogus=np.zeros(3)
    )
    with pytest.raises(ValueError, match="bogus"):
        load_bundle(path, (H, W))


def test_non_npz_path_rejected(tmp_path):
    with pytest.raises(ValueError, match=r"\.npz"):
        save_bundle(tmp_path / "img", minimal_bundle())
    with pytest.raises(ValueError, match=r"\.npz"):
        load_bundle(tmp_path / "img.npy", (H, W))


def test_k_mismatch_rejected(tmp_path):
    b = full_bundle()
    b.sigmas = b.sigmas[:2]
    with pytest.raises(ValueError, match="disagree on K"):
        save_bundle(tmp_path / "img.npz", b)


def test_hw_mismatch_rejected(tmp_path):
    b = full_bundle()
    b.confidence = b.confidence[:, :-1]
    with pytest.raises(ValueError, match="confidence"):
        save_bundle(tmp_path / "img.npz", b)


def test_bad_ndim_rejected(tmp_path):
    b = minimal_bundle()
    b.estimated_depth = b.estimated_depth[None]
    with pytest.raises(ValueError, match="estimated_depth"):
        save_bundle(tmp_path / "img.npz", b)
