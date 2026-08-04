"""MAD robust-scale fit in DepthContext.

Residuals are prescribed exactly: with alpha=1 and a single mode mu, a
target whitened residual r is realized by z = mu * exp(r * sigma), so each
test states the dispersion it expects to be recovered.

"""

import numpy as np
import pytest

from depthba.backends.depth_context import DepthContext
from depthba.config import DepthBAConfig
from depthba.depth import schema

MU = 5.0
SIGMA = 0.1


class _Point2D:
    def __init__(self, point3D_id):
        self.point3D_id = point3D_id

    def has_point3D(self):
        return self.point3D_id is not None


class _Point3D:
    def __init__(self, z):
        self.xyz = np.array([0.0, 0.0, float(z)])


class _CamFromWorld:
    """Identity pose: camera-frame z is the world z."""

    def __mul__(self, xyz):
        return np.asarray(xyz, dtype=float)


class _Image:
    def __init__(self, points2D):
        self.points2D = points2D
        self.cam_from_world = _CamFromWorld()


class _Reconstruction:
    def __init__(self, images, points3D):
        self.images = images
        self.points3D = points3D


def build(residuals, *, second_mode=None, is_sky=False, triangulated=True, alpha=1.0,
          sigma_scale=1.0, stored_sigmas=True):
    """DepthContext whose image 1 carries rows with the given whitened
    residuals. second_mode adds a decoy first mode, with z placed on the
    SECOND mode (so the max-mixture winner is the one that fits)."""
    rows, points2D, points3D = {}, [], {}
    for i, r in enumerate(residuals):
        z = MU * np.exp(r * SIGMA)
        modes = np.array([MU]) if second_mode is None else np.array([second_mode, MU])
        k = len(modes)
        rows[i] = schema.KeypointDepth(
            image_id=1, point2D_idx=i, sensor="s",
            modes=modes,
            weights=np.full(k, 1.0 / k),
            estimated_depth=MU,
            sigmas=np.full(k, SIGMA) if stored_sigmas else None,
            confidence=None,
            is_sky=is_sky,
        )
        points3D[i] = _Point3D(z)
        points2D.append(_Point2D(i if triangulated else None))

    ctx = DepthContext(
        DepthBAConfig(sensor="s", sigma=SIGMA, sigma_scale=sigma_scale), rows={1: rows}
    )
    ctx.affine(1, alpha)
    return ctx, _Reconstruction({1: _Image(points2D)}, points3D)


def test_recovers_known_dispersion():
    rng = np.random.default_rng(0)
    ctx, rec = build(rng.normal(0.0, 3.0, 4000))
    assert ctx.update_robust_scale(rec, [1]) == pytest.approx(3.0, rel=0.1)


def test_floors_at_one_when_residuals_are_tight():
    """Better-than-modeled residuals must NOT tighten the loss below
    huber_scale — the floor is what keeps a clean bundle from rejecting its
    own good depth factors."""
    rng = np.random.default_rng(1)
    ctx, rec = build(rng.normal(0.0, 0.3, 4000))
    assert ctx.update_robust_scale(rec, [1]) == 1.0


def test_second_mode_does_not_inflate_the_scale():
    """The failure mode this design exists to avoid: a keypoint sitting on
    its SECOND mode is a good fit, not an outlier. Scored against mode 0
    alone these residuals are ~46 sigma."""
    ctx, rec = build(np.zeros(500), second_mode=MU * 0.63)
    r = ctx._sampled_residuals(rec, [1], 20_000)
    assert np.abs(r).max() < 1e-9                      # winner is the fitting mode
    naive = (np.log(MU) - np.log(MU * 0.63)) / SIGMA   # what mode 0 would have given
    assert naive > 4.0
    assert ctx.update_robust_scale(rec, [1]) == 1.0


def test_too_few_samples_keeps_previous_estimate():
    ctx, rec = build(np.full(10, 5.0))
    ctx.robust_scale = 2.5
    assert ctx.update_robust_scale(rec, [1]) == 2.5


def test_images_without_alpha_are_skipped():
    """First global BA: no alpha exists yet, so there is no residual to
    fit — must return the nominal scale, not crash."""
    ctx, rec = build(np.full(100, 5.0))
    ctx.alphas.clear()
    assert ctx.update_robust_scale(rec, [1]) == 1.0
    assert len(ctx._sampled_residuals(rec, [1], 20_000)) == 0


def test_sky_and_untriangulated_rows_are_excluded():
    for kwargs in (dict(is_sky=True), dict(triangulated=False)):
        ctx, rec = build(np.full(500, 5.0), **kwargs)
        assert len(ctx._sampled_residuals(rec, [1], 20_000)) == 0


def test_sampling_is_capped_and_deterministic():
    ctx, rec = build(np.full(5000, 2.0))
    first = ctx._sampled_residuals(rec, [1], 100)
    assert len(first) <= 100
    assert np.array_equal(first, ctx._sampled_residuals(rec, [1], 100))
    assert len(ctx._sampled_residuals(rec, [1], 20_000)) == 5000


@pytest.mark.parametrize("stored_sigmas", [True, False])
def test_sigma_scale_applies_to_both_sensor_kinds(stored_sigmas):
    """One sweep axis with the same meaning whether the sensor stored its own
    sigmas (gmm) or falls back to the config constant (unimodal)."""
    ctx, rec = build(np.full(500, 2.0), sigma_scale=2.0, stored_sigmas=stored_sigmas)
    assert np.allclose(ctx._sampled_residuals(rec, [1], 20_000), 1.0)


def test_alpha_is_applied():
    """alpha rescales the prediction, so residuals must be measured against
    alpha*mu — a stale alpha would otherwise read as pure dispersion."""
    ctx, rec = build(np.zeros(500), alpha=1.5)
    r = ctx._sampled_residuals(rec, [1], 20_000)
    assert np.allclose(r, -np.log(1.5) / SIGMA)
