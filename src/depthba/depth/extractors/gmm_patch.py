"""K=2 patch-GMM extractor: fit a 2-mode mixture to the depth values in each
keypoint's SIFT support region.
"""

import numpy as np

from depthba.depth.extractors import DepthMeasurements, _pixel_indices
from depthba.depth.source import DepthBundle

_TINY = 1e-12


def _disk(r: int) -> tuple[np.ndarray, np.ndarray]:
    """Integer (dv, du) offsets of a filled disk of radius r."""
    d = np.arange(-r, r + 1)
    dv, du = np.meshgrid(d, d, indexing="ij")
    m = dv * dv + du * du <= r * r
    return dv[m], du[m]


def _robust_anchor(depth: np.ndarray, v: int, u: int, h: int, w: int, fallback: float) -> float:
    """Median of the valid inner 3x3 around (v, u); fallback if none valid."""
    block = depth[max(v - 1, 0):min(v + 2, h), max(u - 1, 0):min(u + 2, w)].reshape(-1)
    block = block[np.isfinite(block) & (block > 0)]
    d = float(np.median(block)) if block.size else fallback
    return max(d, _TINY)


def _weighted_em(y, sw, mu0, sig_abs, sig_rel, wmin, sep_rel, max_iter):
    """Anchored, spatially-weighted 2-comp 1D EM in linear depth. mu0 fixed.

    Returns (mu1, s0, s1, w0, w1). Collapses to unimodal
    (mu1 = mu0, w1 = wmin) when the patch is flat / undersampled, or when the
    fitted second mode is unsupported or too close to the anchor.

    Floors and gate are RELATIVE to the mode depths (plus a small absolute
    backstop), restoring the scale invariance the log-space fit had for free:
    sigma_k >= max(sig_abs, sig_rel * mu_k), separation >= sep_rel * mu0.
    The relative floor also encodes sensor error: patch variance on a flat
    surface says nothing about network accuracy, so a near-delta sigma there
    would overweight the depth factor against reprojection terms.
    """
    floor0 = max(sig_abs, sig_rel * mu0)

    W = sw.sum()
    ybar = (sw * y).sum() / max(W, _TINY)
    yvar = (sw * (y - ybar) ** 2).sum() / max(W, _TINY)
    s_init = max(np.sqrt(yvar), floor0)
    # degenerate: too few samples or flat patch -> unimodal
    if y.size < 2 or yvar < floor0 * floor0:
        return mu0, s_init, s_init, 1.0 - wmin, wmin

    mu1 = y[np.argmax(np.abs(y - mu0))]       # farthest sample seeds mode 1
    s0 = s1 = s_init
    p0 = p1 = 0.5
    for _ in range(max_iter):
        g0 = p0 * np.exp(-0.5 * ((y - mu0) / s0) ** 2) / s0
        g1 = p1 * np.exp(-0.5 * ((y - mu1) / s1) ** 2) / s1
        den = g0 + g1 + _TINY
        r0, r1 = g0 / den, g1 / den
        n0, n1 = (sw * r0).sum(), (sw * r1).sum()
        p0, p1 = n0 / W, n1 / W
        mu1 = (sw * r1 * y).sum() / max(n1, _TINY)   # mu0 stays fixed
        floor1 = max(sig_abs, sig_rel * mu1)         # mu1 moved; re-floor
        s0 = max(np.sqrt((sw * r0 * (y - mu0) ** 2).sum() / max(n0, _TINY)), floor0)
        s1 = max(np.sqrt((sw * r1 * (y - mu1) ** 2).sum() / max(n1, _TINY)), floor1)

    # joint gate: keep the second mode only if supported AND separated
    # (relative to the anchor depth)
    if p1 < wmin or abs(mu1 - mu0) < sep_rel * mu0:
        return mu0, s0, s0, 1.0 - wmin, wmin
    return mu1, s0, s1, p0, p1


def extract(
    bundle: DepthBundle, keypoints: np.ndarray, params: dict
) -> DepthMeasurements:
    depth = bundle.estimated_depth
    h, w = depth.shape
    c = params.get("patch_scale", 4.0)          # r = c * sqrt(det A)
    r_min = params.get("r_min", 2)
    sig_abs = params.get("sigma_min_abs", 0.02)     # absolute sigma backstop (meters)
    sig_rel = params.get("sigma_min_rel", 0.05)     # sigma floor as fraction of mode depth
    wmin = params.get("wmin", 0.05)                 # min 2nd-mode weight
    sep_rel = params.get("sep_min_rel", 0.1)        # min |mu1 - mu0| / mu0
    max_iter = params.get("em_iters", 30)

    v_kp, u_kp = _pixel_indices(keypoints, (h, w))
    d_kp = depth[v_kp, u_kp].astype(np.float64)     # committed map value (== unimodal)

    A = keypoints[:, 2:6]
    detA = A[:, 0] * A[:, 3] - A[:, 1] * A[:, 2]
    sigma_s = np.sqrt(np.maximum(detA, 0.0))
    radii = np.maximum(np.round(c * sigma_s).astype(int), r_min)

    n = len(keypoints)
    modes = np.empty((n, 2), np.float32)
    weights = np.empty((n, 2), np.float32)
    sigmas = np.empty((n, 2), np.float32)

    disk_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i in range(n):
        vi, ui = int(v_kp[i]), int(u_kp[i])
        mu0 = _robust_anchor(depth, vi, ui, h, w, float(d_kp[i]))

        dv, du = disk_cache.setdefault(int(radii[i]), _disk(int(radii[i])))
        vv, uu = vi + dv, ui + du
        inb = (vv >= 0) & (vv < h) & (uu >= 0) & (uu < w)
        dvi, dui = dv[inb], du[inb]
        dpatch = depth[vv[inb], uu[inb]].astype(np.float64)
        valid = np.isfinite(dpatch) & (dpatch > 0)
        dpatch, dvi, dui = dpatch[valid], dvi[valid], dui[valid]

        ss = max(sigma_s[i], _TINY)
        sw = np.exp(-(dvi * dvi + dui * dui) / (2.0 * ss * ss))
        mu1, s0, s1, p0, p1 = _weighted_em(
            dpatch, sw, mu0, sig_abs, sig_rel, wmin, sep_rel, max_iter
        )
        modes[i] = (mu0, mu1)
        sigmas[i] = (s0, s1)
        weights[i] = (p0, p1)

    return DepthMeasurements(
        modes=modes,
        weights=weights,
        estimated_depth=d_kp.astype(np.float32),
        sigmas=sigmas,
        confidence=None if bundle.confidence is None else bundle.confidence[v_kp, u_kp],
        is_sky=None if bundle.sky_mask is None else bundle.sky_mask[v_kp, u_kp],
    )
