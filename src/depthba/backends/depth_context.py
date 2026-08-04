"""
Runtime depth state for depth-aware BA: the per-image row cache (loaded once
per pipeline run from the depthba tables) and the persistent per-image
alpha/beta affine blocks that survive across BA calls.

Kept separate from the problem builder: everything here is depth-domain
logic (which factor class, which sigma, warm starts); the builder only
walks it. pyceres is imported lazily inside make_cost — everything else
here (row cache, affine blocks, robust-scale fit) is plain numpy, and
keeping it importable without the Linux-only fork wheel is what lets those
parts be tested off-cluster.
"""

import sqlite3
from pathlib import Path

import numpy as np

from depthba.backends.residuals import maxmix_scores, whitened_residuals
from depthba.config import DepthBAConfig
from depthba.depth import schema

_PLAIN_FACTORS = {
    "log": "LogDepthError",
    "linear": "DepthError",
    "inverse": "InvDepthError",
}
_MAXMIX_FACTORS = {
    "log": "LogDepthErrorMaxMix",
    "linear": "DepthErrorMaxMix",
    "inverse": "InvDepthErrorMaxMix",
}


class DepthContext:
    """One per pipeline run.

    The alpha/beta arrays are the actual ceres parameter blocks: created at
    an image's first BA appearance, kept alive here, and reused (and further
    refined) by every later BA the image participates in.
    """

    def __init__(self, config: DepthBAConfig, meta=None, rows=None):
        self.config = config
        self.meta = meta
        self.rows = rows if rows is not None else {}  # image_id -> {point2D_idx: KeypointDepth}
        self.alphas: dict[int, np.ndarray] = {}
        self.betas: dict[int, np.ndarray] = {}
        # Huber multiplier from the last global-BA MAD fit. Persisting it here
        # is what lets local BAs inherit the global estimate instead of
        # refitting on their own (young, badly-triangulated) points.
        self.robust_scale = 1.0

    @classmethod
    def load(cls, config: DepthBAConfig, database_path: Path) -> "DepthContext":
        if config.sensor is None:
            return cls(config)
        conn = sqlite3.connect(database_path)
        try:
            meta = schema.read_meta(conn, config.sensor)
            if meta.sigma_space is not None and meta.sigma_space != config.depth_space:
                raise NotImplementedError(
                    f"sensor stores sigmas in {meta.sigma_space!r} but depth_space "
                    f"is {config.depth_space!r}; sigma-space conversion is not implemented"
                )
            image_ids = [
                i for (i,) in conn.execute(
                    "SELECT DISTINCT image_id FROM depthba_keypoint_depths WHERE sensor=?",
                    (config.sensor,),
                )
            ]
            rows = {
                image_id: schema.read_depths_for_image(
                    conn, image_id, config.sensor, meta.num_modes
                )
                for image_id in image_ids
            }
        finally:
            conn.close()
        return cls(config, meta, rows)

    def active(self, in_global: bool) -> bool:
        if self.config.sensor is None:
            return False
        return self.config.depth_in_global if in_global else self.config.depth_in_local

    def _affine_key(self, image_id: int) -> int:
        # shared_scale: every image resolves to one global block (sentinel key)
        return -1 if self.config.shared_scale else image_id

    def has_affine(self, image_id: int) -> bool:
        return self._affine_key(image_id) in self.alphas

    def affine(self, image_id: int, alpha0: float = 1.0):
        """Get-or-create the persistent alpha/beta blocks for an image.
        alpha0 is used only at creation (the image's first BA appearance;
        under shared_scale, the whole map's first). beta is created at 0 and
        held there — the affine is scale-only; the block exists because the
        fork's factors take it."""
        key = self._affine_key(image_id)
        if key not in self.alphas:
            self.alphas[key] = np.array([float(alpha0)], dtype=np.float64)
            self.betas[key] = np.array([0.0], dtype=np.float64)
        return self.alphas[key], self.betas[key]

    def rescale_affine(self, scale: float) -> None:
        """Keep alpha consistent when reconstruction.normalize() rescales the
        map: z' = s*z  =>  alpha' = s*alpha. Without this, persistent alpha
        blocks go stale after every normalization (harmless for free alpha,
        which re-converges, but wrong for frozen ones). beta is constant at 0,
        so it needs no rescaling."""
        for alpha in self.alphas.values():
            alpha *= scale

    def _sampled_residuals(self, reconstruction, image_ids, max_samples) -> np.ndarray:
        """Whitened depth residuals at the CURRENT state, one per sampled
        factor, taking each row's max-mixture winner — a keypoint sitting
        correctly on its second mode must read as a small residual, not
        inflate the scale estimate it is then judged against."""
        cfg = self.config
        ids = [i for i in sorted(image_ids) if self.rows.get(i) and self.has_affine(i)]
        if not ids:
            return np.empty(0)  # nothing has an alpha yet: first global BA
        budget = max(1, max_samples // len(ids))

        out = []
        for image_id in ids:
            rows = self.rows[image_id]
            alpha, beta = self.affine(image_id)
            a, b = float(alpha[0]), float(beta[0])
            image = reconstruction.images[image_id]
            cam_from_world = image.cam_from_world
            if callable(cam_from_world):
                cam_from_world = cam_from_world()
            points2D = image.points2D
            # Fixed stride, spread over the whole image: a prefix would bias
            # toward one region, and an RNG would cost bit-identical re-runs.
            # Most keypoints are untriangulated, so this undershoots the
            # budget — fine, MAD's error falls as 1/sqrt(n).
            stride = max(1, len(points2D) // budget)
            for idx in range(0, len(points2D), stride):
                row = rows.get(idx)
                p2d = points2D[idx]
                if row is None or row.is_sky or not p2d.has_point3D():
                    continue
                z = (cam_from_world * reconstruction.points3D[p2d.point3D_id].xyz)[2]
                if z <= 0:
                    continue
                sigmas = (
                    row.sigmas if row.sigmas is not None
                    else np.full(len(row.modes), cfg.sigma)
                )
                r = whitened_residuals(z, row.modes, sigmas, a, b)
                if len(r) == 1:
                    out.append(r[0])
                else:
                    weights = np.maximum(row.weights.astype(np.float64), 1e-20)
                    out.append(r[int(np.argmin(maxmix_scores(r, sigmas, weights)))])
        return np.asarray(out)

    def update_robust_scale(
        self, reconstruction, image_ids, max_samples: int = 20_000, min_samples: int = 50
    ) -> float:
        """Refit the Huber multiplier from the current residual dispersion.

        Called at global BA only; the result is frozen for that solve and
        inherited by every local BA until the next global one. Too few
        samples (early map) keeps the previous estimate.
        """
        r = self._sampled_residuals(reconstruction, image_ids, max_samples)
        if len(r) < min_samples:
            return self.robust_scale
        sigma = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        # Floor at 1: the fitted dispersion may only LOOSEN the loss relative
        # to huber_scale. Tightening below nominal would let a well-behaved
        # bundle start rejecting its own good depth factors.
        self.robust_scale = max(sigma, 1.0)
        return self.robust_scale

    def make_cost(self, row):
        import pyceres  # lazy: see module docstring

        cfg = self.config
        num_modes = len(row.modes)
        if num_modes == 1:
            factor = getattr(pyceres.factors, _PLAIN_FACTORS[cfg.depth_space])
            sigma = float(row.sigmas[0]) if row.sigmas is not None else cfg.sigma
            # modes[0] == estimated_depth for K=1 rows by extractor
            # construction; modes[0] keeps the K=1/K>1 paths consistent.
            return factor(float(row.modes[0]), sigma)
        factor = getattr(pyceres.factors, _MAXMIX_FACTORS[cfg.depth_space])
        sigmas = row.sigmas if row.sigmas is not None else np.full(num_modes, cfg.sigma)
        # The C++ MaxMix factor requires weights > 0 (it takes their log);
        # ingest allows weight == 0 and float32 storage can underflow a tiny
        # softmax weight to exactly 0. Clamp to a floor far below any real
        # weight — a near-zero-weight mode gets a huge log-penalty and is
        # never selected, which is the correct "absent mode" behavior.
        weights = np.maximum(row.weights.astype(np.float64), 1e-20)
        return factor(
            row.modes.astype(np.float64),
            sigmas.astype(np.float64),
            weights,
        )


def median_depth_ratio(image, reconstruction, rows) -> float:
    """Alpha warm start: median z_cam / mu over the image's triangulated,
    non-sky depth rows (per-image auto-scale). The multiplicative slot of
    the affine is alpha — never initialize beta with a ratio.

    Uses estimated_depth deliberately (also for multimodal rows): the schema
    defines it as the sensor's committed single-value answer, which is the
    right scale reference; mode order carries no primacy guarantee."""
    cam_from_world = image.cam_from_world
    if callable(cam_from_world):
        cam_from_world = cam_from_world()
    ratios = []
    for idx, p2d in enumerate(image.points2D):
        row = rows.get(idx)
        if row is None or row.is_sky or not p2d.has_point3D():
            continue
        z = (cam_from_world * reconstruction.points3D[p2d.point3D_id].xyz)[2]
        if z > 0 and row.estimated_depth > 0:
            ratios.append(z / row.estimated_depth)
    return float(np.median(ratios)) if ratios else 1.0
