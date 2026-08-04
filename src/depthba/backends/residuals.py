"""Python mirrors of the fork's depth-factor math.

These formulas MUST match the C++ factors in the pyceres fork wheel — they
are what the MAD scale estimate is computed from, so a silent divergence
would fit the robust loss to the wrong dispersion. Parity is pinned against
the shipped wheel by tests/test_maxmix_factor.py (Linux-only).

Deliberately free of the pyceres import so they stay importable, and
testable, where the fork wheel is absent (macOS).

LOG SPACE ONLY: LogDepthError / LogDepthErrorMaxMix. The linear and inverse
families have different residual definitions and no parity coverage yet —
callers must gate on depth_space until they do.
"""

import numpy as np


def whitened_residuals(z, modes, sigmas, alpha=1.0, beta=0.0):
    """Per-mode whitened residual: (log z - log(alpha*mu + beta)) / sigma.

    The affine acts in LINEAR depth (alpha multiplicative, beta metric
    offset), then the comparison is taken in log space.
    """
    mu = alpha * np.asarray(modes, dtype=float) + beta
    return (np.log(z) - np.log(mu)) / np.asarray(sigmas, dtype=float)


def maxmix_scores(residuals, sigmas, weights):
    """Max-mixture selection score: whitened^2 + 2 log(sigma_k / w_k).

    The factor's residual is the winning mode's PLAIN whitened error; the
    log term only decides the winner (see tests/test_maxmix_factor.py).
    """
    return np.asarray(residuals, dtype=float) ** 2 + 2.0 * np.log(
        np.asarray(sigmas, dtype=float) / np.asarray(weights, dtype=float)
    )
