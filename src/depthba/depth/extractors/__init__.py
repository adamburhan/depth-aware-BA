"""
Extractor interface: (DepthBundle, keypoints, params) -> DepthMeasurements.

Extractors are the method axis of the design: several can consume the same
bundle (dump) without re-running the network. attach_depths picks one from
EXTRACTORS by name; that name + params become the sensor identity in
depthba_depth_meta.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from depthba.depth.source import DepthBundle


@dataclass
class DepthMeasurements:
    """Per-image extraction result, arrays over N keypoints."""

    modes: np.ndarray              # (N, K) float32, linear meters
    weights: np.ndarray            # (N, K)
    estimated_depth: np.ndarray    # (N,)
    sigmas: np.ndarray | None      # (N, K); None if sensor provides no sigmas
    confidence: np.ndarray | None  # (N,)
    is_sky: np.ndarray | None      # (N,) bool


# keypoints are (N, 6) full COLMAP keypoint rows: x, y in columns 0-1
# (extractors read only those today; the affine shape columns are passed
# through so scale-adaptive methods need no interface change).
ExtractorFn = Callable[[DepthBundle, np.ndarray, dict], DepthMeasurements]


def _pixel_indices(
    keypoints: np.ndarray, hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP keypoint (x, y) -> integer (v, u): floor, then clip to bounds.

    Pixel (0, 0) spans [0,1)x[0,1) with center (0.5, 0.5), so continuous
    coordinate x lands in column floor(x). Canonical convention for all
    extractors; pinned by the half-pixel test in test_unimodal.py.
    """
    h, w = hw
    u = np.clip(np.floor(keypoints[:, 0]).astype(np.int64), 0, w - 1)
    v = np.clip(np.floor(keypoints[:, 1]).astype(np.int64), 0, h - 1)
    return v, u


def _not_implemented(name: str) -> ExtractorFn:
    def stub(bundle: DepthBundle, keypoints: np.ndarray, params: dict) -> DepthMeasurements:
        raise NotImplementedError(f"{name} extractor not yet implemented")

    return stub


from depthba.depth.extractors import mda, unimodal  

EXTRACTORS: dict[str, ExtractorFn] = {
    "unimodal": unimodal.extract,
    "mda_native": mda.extract,
    "gmm_patch": _not_implemented("gmm_patch"),
}
