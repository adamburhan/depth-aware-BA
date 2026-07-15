"""
Runtime depth state for depth-aware BA: the per-image row cache (loaded once
per pipeline run from the depthba tables) and the persistent per-image
alpha/beta affine blocks that survive across BA calls.

Kept separate from the problem builder: everything here is depth-domain
logic (which factor class, which sigma, warm starts); the builder only
walks it. Imports pyceres, so Linux-only in practice.
"""

import sqlite3
from pathlib import Path

import numpy as np

import pyceres

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

    def affine(self, image_id: int, alpha0: float = 1.0):
        """Get-or-create the persistent alpha/beta blocks for an image.
        alpha0 is used only at creation (the image's first BA appearance)."""
        if image_id not in self.alphas:
            self.alphas[image_id] = np.array([float(alpha0)])
            self.betas[image_id] = np.array([0.0])
        return self.alphas[image_id], self.betas[image_id]

    def make_cost(self, row) -> "pyceres.CostFunction":
        cfg = self.config
        num_modes = len(row.modes)
        if num_modes == 1:
            factor = getattr(pyceres.factors, _PLAIN_FACTORS[cfg.depth_space])
            sigma = float(row.sigmas[0]) if row.sigmas is not None else cfg.sigma
            return factor(float(row.estimated_depth), sigma)
        factor = getattr(pyceres.factors, _MAXMIX_FACTORS[cfg.depth_space])
        sigmas = row.sigmas if row.sigmas is not None else np.full(num_modes, cfg.sigma)
        return factor(
            row.modes.astype(np.float64),
            sigmas.astype(np.float64),
            row.weights.astype(np.float64),
        )


def median_depth_ratio(image, reconstruction, rows) -> float:
    """Alpha warm start: median z_cam / mu over the image's triangulated,
    non-sky depth rows (per-image auto-scale). The multiplicative slot of
    the affine is alpha — never initialize beta with a ratio."""
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
