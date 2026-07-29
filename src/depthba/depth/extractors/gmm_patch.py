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


def _weighted_em(y, sw, mu0, sig_floor, wmin, sep_min, max_iter):
    """Anchored, spatially-weighted 2-comp 1D EM in LOG depth. mu0 fixed.

    sw are per-sample spatial weights — Gaussian in pixel distance from the
    keypoint with sigma = the SIFT scale — so pixels near the keypoint carry
    the fit and the disk rim (at ~patch_scale sigmas) only registers when the
    support region is genuinely large. Sufficient statistics are the
    sw-weighted responsibility sums, so p0/p1 are weighted mass fractions,
    not sample counts.

    Returns (mu1, s0, s1, w0, w1) in log space. Collapses to unimodal
    (mu1 = mu0, w1 = wmin) when the patch is flat / undersampled, or when the
    fitted second mode is unsupported or too close to the anchor.
    """
    # degenerate: too few samples -> unimodal (yvar would be 0, so s = floor)
    if y.size < 2:
        return mu0, sig_floor, sig_floor, 1.0 - wmin, wmin
    W = max(sw.sum(), _TINY)
    ybar = (sw * y).sum() / W
    yvar = (sw * (y - ybar) ** 2).sum() / W
    s_init = max(np.sqrt(yvar), sig_floor)
    if yvar < sig_floor * sig_floor:          # flat patch -> unimodal
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
        s0 = max(np.sqrt((sw * r0 * (y - mu0) ** 2).sum() / max(n0, _TINY)), sig_floor)
        s1 = max(np.sqrt((sw * r1 * (y - mu1) ** 2).sum() / max(n1, _TINY)), sig_floor)

    # joint gate: keep the second mode only if supported AND separated
    if p1 < wmin or abs(mu1 - mu0) < sep_min:
        return mu0, s0, s0, 1.0 - wmin, wmin
    return mu1, s0, s1, p0, p1


def extract(
    bundle: DepthBundle, keypoints: np.ndarray, params: dict
) -> DepthMeasurements:
    depth = bundle.estimated_depth
    h, w = depth.shape
    c = params.get("patch_scale", 4.0)          # r = c * sqrt(det A)
    r_min = params.get("r_min", 2)
    wmin = params.get("wmin", 0.05)                 # min 2nd-mode weight
    max_iter = params.get("em_iters", 30)
    uniform_prior = params.get("uniform_prior", True)  # uniform prior on pi's
    
    # depth_space = params.get("depth_space", "linear")
    # if depth_space == 'linear':
    #     sig_floor = params.get("sigma_linear_min", 0.02)
    # elif depth_space == 'log':
    sig_floor = params.get("sigma_log_min", 0.05)   # log sigma floor (~5% relative)
    sep_min = params.get("sep_log_min", 0.1)        # min |mu1 - mu0| in log (~10%)
        
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
        #mu0 = np.log(_robust_anchor(depth, vi, ui, h, w, float(d_kp[i])))
        mu0 = np.log(max(d_kp[i], _TINY))  

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
            np.log(dpatch), sw, mu0, sig_floor, wmin, sep_min, max_iter
        )
        modes[i] = (np.exp(mu0), np.exp(mu1))
        sigmas[i] = (s0, s1)
        
        if uniform_prior:
            weights[i] = (0.5, 0.5)
        else:
            weights[i] = (p0, p1)

    return DepthMeasurements(
        modes=modes,
        weights=weights,
        estimated_depth=d_kp.astype(np.float32),
        sigmas=sigmas,
        confidence=None if bundle.confidence is None else bundle.confidence[v_kp, u_kp],
        is_sky=None if bundle.sky_mask is None else bundle.sky_mask[v_kp, u_kp],
    )
