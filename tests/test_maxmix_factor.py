"""Known-answer tests for LogDepthErrorMaxMix (fork wheel behavior check).

The selection score is whitened_k^2 + 2*log(sigma_k / w_k) — per-mode sigma
normalizer in the selection only, winning mode's plain whitened error as the
residual (confirmed from bindings.h source; these tests pin the shipped wheel
to that spec). Selection flips exactly where
    delta whitened^2 = 2*log(sigma_B * w_A / (sigma_A * w_B)).

Linux-only (macOS cannot co-import; these tests need only pyceres anyway,
but skip uniformly where the fork wheel is absent).
"""

import numpy as np
import pytest

pyceres = pytest.importorskip("pyceres")

POSE7_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # [quat xyzw | t]


def maxmix_cost(z, modes, sigmas, weights, alpha=1.0, beta=0.0):
    """0.5 * whitened_winner^2, via a 0-iteration solve at a synthetic state
    (identity pose, point at [0, 0, z] -> z_cam = z)."""
    problem = pyceres.Problem()
    cost = pyceres.factors.LogDepthErrorMaxMix(
        np.asarray(modes, float), np.asarray(sigmas, float), np.asarray(weights, float)
    )
    blocks = [POSE7_IDENTITY.copy(), np.array([0.0, 0.0, float(z)]),
              np.array([alpha]), np.array([beta])]
    problem.add_residual_block(cost, None, blocks)
    options = pyceres.SolverOptions()
    options.max_num_iterations = 0
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    return summary.initial_cost


def whitened(z, mu, sigma, alpha=1.0, beta=0.0):
    return (np.log(z) - np.log(alpha * mu + beta)) / sigma


def score(z, mu, sigma, w):
    return whitened(z, mu, sigma) ** 2 + 2.0 * np.log(sigma / w)


def test_zero_residual_at_winning_mode():
    assert maxmix_cost(2.0, [2.0, 5.0], [0.1, 0.1], [0.9, 0.1]) < 1e-20


def test_log_sigma_term_in_selection():
    """z sits EXACTLY on mode B, yet tight-sigma mode A must win the
    selection: score_A = whitened_A^2 + 2 log sigma_A beats score_B = 2 log
    sigma_B. Without the log-sigma term B would win with cost 0."""
    modes, sigmas, weights = [2.0, 2.2], [0.05, 0.5], [0.5, 0.5]
    z = 2.2
    assert score(z, modes[0], sigmas[0], weights[0]) < score(z, modes[1], sigmas[1], weights[1])
    expected = 0.5 * whitened(z, modes[0], sigmas[0]) ** 2
    got = maxmix_cost(z, modes, sigmas, weights)
    assert got == pytest.approx(expected, rel=1e-9)
    assert got > 1.0  # emphatically not the "B wins, cost 0" outcome


def test_weight_flip_at_analytic_threshold():
    """Equal sigmas, unequal weights: the winner flips where
    whitened_B^2 - whitened_A^2 = 2 log(w_A / w_B). Solve the threshold in
    log-depth closed form and probe both sides."""
    mu_a, mu_b, sigma, w_a, w_b = 2.0, 3.0, 0.1, 0.8, 0.2
    a, b = np.log(mu_a), np.log(mu_b)
    # (L-b)^2 - (L-a)^2 = sigma^2 * 2 log(w_a/w_b)  =>  linear in L
    rhs = sigma**2 * 2.0 * np.log(w_a / w_b)
    L_star = 0.5 * ((rhs / (a - b)) + a + b)
    for dL, want_winner in [(-0.01, None), (+0.01, None)]:
        z = float(np.exp(L_star + dL))
        s_a = score(z, mu_a, sigma, w_a)
        s_b = score(z, mu_b, sigma, w_b)
        want = mu_a if s_a < s_b else mu_b
        expected = 0.5 * whitened(z, want, sigma) ** 2
        assert maxmix_cost(z, [mu_a, mu_b], [sigma, sigma], [w_a, w_b]) == pytest.approx(
            expected, rel=1e-9
        )
    # and the two sides pick DIFFERENT modes
    below = score(np.exp(L_star - 0.01), mu_a, sigma, w_a) < score(
        np.exp(L_star - 0.01), mu_b, sigma, w_b)
    above = score(np.exp(L_star + 0.01), mu_a, sigma, w_a) < score(
        np.exp(L_star + 0.01), mu_b, sigma, w_b)
    assert below != above


def test_affine_transform_slots():
    """Residual compares z against alpha*mu + beta (linear domain, then log):
    each parameter alone can zero the residual — pins the alpha/beta slots."""
    assert maxmix_cost(3.0, [2.0], [0.1], [1.0], alpha=1.5, beta=0.0) < 1e-20
    assert maxmix_cost(3.0, [2.0], [0.1], [1.0], alpha=1.0, beta=1.0) < 1e-20


def test_construction_validation():
    bad = [
        dict(modes=[0.0, 2.0], sigmas=[0.1, 0.1], weights=[0.5, 0.5]),   # mode <= 0
        dict(modes=[2.0, 3.0], sigmas=[0.0, 0.1], weights=[0.5, 0.5]),   # sigma <= 0
        dict(modes=[2.0, 3.0], sigmas=[0.1, 0.1], weights=[0.0, 0.5]),   # weight <= 0
        dict(modes=[2.0, 3.0], sigmas=[0.1], weights=[0.5, 0.5]),        # size mismatch
    ]
    for kwargs in bad:
        with pytest.raises((ValueError, RuntimeError)):
            pyceres.factors.LogDepthErrorMaxMix(
                np.asarray(kwargs["modes"], float),
                np.asarray(kwargs["sigmas"], float),
                np.asarray(kwargs["weights"], float),
            )
