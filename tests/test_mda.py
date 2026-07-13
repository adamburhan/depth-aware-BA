"""mda_native extractor tests. Pixel fixture copied from test_unimodal.py,
extended per mode: plane k holds k*10000 + u + 1000*v, so mode order
surviving the (K, N) -> (N, K) transpose is visible in the values."""

import numpy as np
import pytest

from depthba.depth.extractors import EXTRACTORS
from depthba.depth.extractors.mda import extract
from depthba.depth.source import DepthBundle

H, W, K = 5, 7, 4


def make_bundle(**overrides):
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    k = np.arange(K, dtype=np.float32)
    pixel = u[None, :] + 1000.0 * v[:, None]                # (H, W)
    planes = 10000.0 * k[:, None, None] + pixel[None]       # (K, H, W)
    fields = dict(
        estimated_depth=pixel,
        means=planes,
        weights=np.full((K, H, W), 0.25, np.float32),
        sigmas=0.1 * (planes + 1.0),
        confidence=None,
        sky_mask=None,
    )
    fields.update(overrides)
    return DepthBundle(**fields)


def kps(*xys):
    out = np.zeros((len(xys), 6), dtype=np.float32)
    out[:, :2] = xys
    return out


def test_mixture_readout():
    m = extract(make_bundle(), kps((0.6, 0.0), (1.0, 0.0), (6.9, 4.9)), {})
    pix = np.float32([0.0, 1.0, 4006.0])
    expected_modes = 10000.0 * np.arange(K, dtype=np.float32)[None, :] + pix[:, None]
    assert m.modes.shape == (3, K)
    np.testing.assert_array_equal(m.modes, expected_modes)
    np.testing.assert_array_equal(m.weights, np.full((3, K), 0.25, np.float32))
    np.testing.assert_array_equal(m.sigmas, 0.1 * (expected_modes + 1.0))
    np.testing.assert_array_equal(m.estimated_depth, pix)
    assert m.confidence is None
    assert m.is_sky is None


def test_sigmaless_mixture():
    m = extract(make_bundle(sigmas=None), kps((0.0, 0.0)), {})
    assert m.sigmas is None
    assert m.modes.shape == (1, K)


def test_unimodal_bundle_rejected():
    b = make_bundle(
        means=np.ones((1, H, W), np.float32),
        weights=np.ones((1, H, W), np.float32),
        sigmas=None,
    )
    with pytest.raises(ValueError, match="K>=2"):
        extract(b, kps((0.0, 0.0)), {})
    no_mixture = make_bundle(means=None, weights=None, sigmas=None)
    with pytest.raises(ValueError, match="K>=2"):
        extract(no_mixture, kps((0.0, 0.0)), {})


def test_registry():
    assert EXTRACTORS["mda_native"] is extract
