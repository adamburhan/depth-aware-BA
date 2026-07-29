"""Sim(3) trajectory alignment (Umeyama), shared by the pose evals and the
ScanNet++ NVS pose transfer """

import numpy as np


def umeyama_sim3(src, dst, with_scale=True):
    """Least-squares Sim(3): dst ~ s * R @ src + t (Umeyama 1991, with scale).

    src, dst: (N, 3) corresponding points, N >= 3. Fit direction matters:
    swapping src and dst yields the inverse transform. Returns (s, R, t).
    """
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[0] >= 3
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    x_src, x_dst = src - mu_src, dst - mu_dst
    cov = x_dst.T @ x_src / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) * len(src) / (x_src**2).sum() if with_scale else 1.0
    t = mu_dst - scale * R @ mu_src
    return scale, R, t


def robust_umeyama_sim3(src, dst, with_scale=True, trim=3.0):
    """Umeyama with one outlier-trim pass: fit, drop correspondences whose
    residual exceeds trim * median residual, refit on the survivors.

    Returns (s, R, t, inliers) with inliers the boolean mask of kept
    correspondences. Residuals are ||dst - (s R src + t)|| in dst units.
    """
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    s, R, t = umeyama_sim3(src, dst, with_scale)
    res = np.linalg.norm(dst - (s * (R @ src.T)).T - t, axis=1)
    inliers = res <= trim * np.median(res)
    if 3 <= inliers.sum() < len(src):
        s, R, t = umeyama_sim3(src[inliers], dst[inliers], with_scale)
    return s, R, t, inliers
