"""K=1 extractor: read the committed depth map at each keypoint's pixel.

Runs on any bundle; per-mode channels (means/weights/sigmas) are ignored
unless K=1, in which case sigmas pass through. On a mixture bundle this is
the unimodal control condition: same dump, mixture off — the committed map
has no per-mode sigma, so sigmas=None there is the truth, not a loss.
Per-pixel channels (confidence, sky_mask) always pass through.
"""

import numpy as np

from depthba.depth.extractors import DepthMeasurements, _pixel_indices
from depthba.depth.source import DepthBundle


def extract(
    bundle: DepthBundle, keypoints: np.ndarray, params: dict
) -> DepthMeasurements:
    v, u = _pixel_indices(keypoints, bundle.estimated_depth.shape)
    depth = bundle.estimated_depth[v, u]  # (N,)
    k1_sigmas = bundle.sigmas is not None and bundle.sigmas.shape[0] == 1
    return DepthMeasurements(
        modes=depth[:, None],
        weights=np.ones((len(depth), 1), dtype=np.float32),
        estimated_depth=depth,
        sigmas=bundle.sigmas[0, v, u][:, None] if k1_sigmas else None,
        confidence=None if bundle.confidence is None else bundle.confidence[v, u],
        is_sky=None if bundle.sky_mask is None else bundle.sky_mask[v, u],
    )
