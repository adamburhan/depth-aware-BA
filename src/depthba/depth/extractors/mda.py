"""mda_native extractor: read the mixture stack at each keypoint's pixel.

means/weights/sigmas sampled per keypoint as (N, K); committed map and
per-pixel channels pass through as in unimodal. Requires a mixture bundle
(K >= 2) — pairing mda_native with a unimodal dump is a wrong-pairing
configuration error, the mirror image of unimodal's ignore-extras rule.
"""

import numpy as np

from depthba.depth.extractors import DepthMeasurements, _pixel_indices
from depthba.depth.source import DepthBundle


def extract(
    bundle: DepthBundle, keypoints: np.ndarray, params: dict
) -> DepthMeasurements:
    if bundle.means is None or bundle.means.shape[0] < 2:
        k = None if bundle.means is None else bundle.means.shape[0]
        raise ValueError(f"mda_native requires a mixture bundle (K>=2), got K={k}")
    v, u = _pixel_indices(keypoints, bundle.estimated_depth.shape)
    return DepthMeasurements(
        modes=bundle.means[:, v, u].T,
        weights=bundle.weights[:, v, u].T,
        estimated_depth=bundle.estimated_depth[v, u],
        sigmas=None if bundle.sigmas is None else bundle.sigmas[:, v, u].T,
        confidence=None if bundle.confidence is None else bundle.confidence[v, u],
        is_sky=None if bundle.sky_mask is None else bundle.sky_mask[v, u],
    )
