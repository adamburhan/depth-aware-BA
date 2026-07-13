"""Half-pixel convention test for the unimodal extractor.

Canonical fixture for the COLMAP pixel convention: pixel (0,0) spans
[0,1)x[0,1) with center (0.5, 0.5), so coordinate x samples column
floor(x). Later extractors inherit the convention by copying this fixture.
"""

import numpy as np
import pytest

from depthba.depth.extractors import EXTRACTORS
from depthba.depth.extractors.unimodal import extract
from depthba.depth.source import DepthBundle

H, W = 5, 7


def make_bundle(**overrides):
    """estimated_depth[v, u] = u + 1000*v — sampled column/row read off the value."""
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    fields = dict(
        estimated_depth=u[None, :] + 1000.0 * v[:, None],
        means=None,
        weights=None,
        sigmas=None,
        confidence=None,
        sky_mask=None,
    )
    fields.update(overrides)
    return DepthBundle(**fields)


def kps(*xys):
    """(N, 6) COLMAP keypoint rows; affine shape columns zeroed."""
    out = np.zeros((len(xys), 6), dtype=np.float32)
    out[:, :2] = xys
    return out


def test_half_pixel_convention():
    keypoints = kps(
        (0.4, 0.0),   # -> column 0
        (0.6, 0.0),   # -> still column 0 (center of pixel 0 is 0.5)
        (1.0, 0.0),   # -> column 1
        (0.0, 0.0),   # corner
        (0.0, 0.6),   # -> row 0
        (0.0, 1.0),   # -> row 1
        (6.9, 4.9),   # max valid pixel
        (7.2, 5.3),   # past the edge -> clipped to (6, 4)
        (-0.5, -2.0), # negative -> clipped to (0, 0)
    )
    m = extract(make_bundle(), keypoints, {})
    expected = [0.0, 0.0, 1.0, 0.0, 0.0, 1000.0, 4006.0, 4006.0, 0.0]
    np.testing.assert_array_equal(m.estimated_depth, np.float32(expected))
    np.testing.assert_array_equal(m.modes, np.float32(expected)[:, None])
    assert m.modes.shape == (9, 1)
    assert m.modes.dtype == np.float32
    np.testing.assert_array_equal(m.weights, np.ones((9, 1), np.float32))


def test_optional_channels_passthrough():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, (H, W)).astype(np.float32)
    sky = np.zeros((H, W), dtype=bool)
    sky[2, 3] = True
    b = make_bundle(confidence=conf, sky_mask=sky)
    m = extract(b, kps((3.4, 2.6), (0.0, 0.0)), {})
    np.testing.assert_array_equal(m.confidence, conf[[2, 0], [3, 0]])
    np.testing.assert_array_equal(m.is_sky, [True, False])
    assert m.sigmas is None


def test_k1_sigmas_passthrough():
    sig = np.full((1, H, W), 0.25, dtype=np.float32)
    b = make_bundle(
        means=np.ones((1, H, W), np.float32),
        weights=np.ones((1, H, W), np.float32),
        sigmas=sig,
    )
    m = extract(b, kps((1.5, 2.5)), {})
    np.testing.assert_array_equal(m.sigmas, [[0.25]])


def test_unimodal_on_mixture_bundle():
    """The unimodal control condition: same MDA dump, mixture off.

    Per-mode channels are ignored (the committed map has no per-mode sigma);
    per-pixel channels still travel — sky in particular, so factor
    construction can exclude fabricated sky-pixel depths later.
    """
    rng = np.random.default_rng(1)
    conf = rng.uniform(0, 1, (H, W)).astype(np.float32)
    sky = np.zeros((H, W), dtype=bool)
    sky[0, 0] = True
    b = make_bundle(
        means=rng.uniform(1, 10, (4, H, W)).astype(np.float32),
        weights=np.full((4, H, W), 0.25, np.float32),
        sigmas=rng.uniform(0.01, 1, (4, H, W)).astype(np.float32),
        confidence=conf,
        sky_mask=sky,
    )
    m = extract(b, kps((0.4, 0.0), (6.9, 4.9)), {})
    np.testing.assert_array_equal(m.modes[:, 0], np.float32([0.0, 4006.0]))
    np.testing.assert_array_equal(m.estimated_depth, m.modes[:, 0])
    assert m.sigmas is None
    np.testing.assert_array_equal(m.confidence, conf[[0, 4], [0, 6]])
    np.testing.assert_array_equal(m.is_sky, [True, False])


def test_registry():
    assert EXTRACTORS["unimodal"] is extract
    with pytest.raises(NotImplementedError):
        EXTRACTORS["gmm_patch"](make_bundle(), kps((0.0, 0.0)), {})
