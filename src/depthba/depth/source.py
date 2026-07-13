"""
DepthBundle and its on-disk npz dump format. save_bundle/load_bundle are
the only code that reads or writes dump files; extractors consume
DepthBundle in memory and never touch disk. One bundle format per dump,
i.e. per (network, scene); the bundle is method-blind — how it is turned
into KeypointDepth rows is the extractor's choice at ingest.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DepthBundle:
    """One image's dense depth-network outputs, verbatim, at native resolution.

    Method-agnostic: extractors consume this; nothing here encodes how it
    will be analyzed. Nullable fields = network doesn't produce them.
    """
    estimated_depth: np.ndarray # (H, W) float32, linear meters — the committed map
    means: np.ndarray | None # (K, H, W), mixture networks only
    weights: np.ndarray | None # (K, H, W)
    sigmas: np.ndarray | None # (K, H, W), sensor-native space
    confidence: np.ndarray | None # (H, W)
    sky_mask: np.ndarray | None # (H, W) bool


_FIELDS = [f.name for f in dataclasses.fields(DepthBundle)]


def _validated(bundle: DepthBundle) -> DepthBundle:
    """Coerce dtypes and enforce shape invariants; returns a new bundle."""
    est = np.ascontiguousarray(bundle.estimated_depth, dtype=np.float32)
    if est.ndim != 2:
        raise ValueError(f"estimated_depth must be (H, W), got shape {est.shape}")
    hw = est.shape

    def coerce(name: str, arr: np.ndarray | None, ndim: int, dtype) -> np.ndarray | None:
        if arr is None:
            return None
        arr = np.ascontiguousarray(arr, dtype=dtype)
        if arr.ndim != ndim or arr.shape[-2:] != hw:
            raise ValueError(
                f"{name} shape {arr.shape} inconsistent with estimated_depth {hw}"
            )
        return arr

    means = coerce("means", bundle.means, 3, np.float32)
    weights = coerce("weights", bundle.weights, 3, np.float32)
    sigmas = coerce("sigmas", bundle.sigmas, 3, np.float32)
    confidence = coerce("confidence", bundle.confidence, 2, np.float32)
    sky_mask = coerce("sky_mask", bundle.sky_mask, 2, bool)

    if (means is None) != (weights is None):
        raise ValueError("means and weights must be present together")
    if sigmas is not None and means is None:
        raise ValueError("sigmas present without means — per-mode spreads require modes")
    ks = {a.shape[0] for a in (means, weights, sigmas) if a is not None}
    if len(ks) > 1:
        raise ValueError(f"means/weights/sigmas disagree on K: {sorted(ks)}")

    return DepthBundle(
        estimated_depth=est,
        means=means,
        weights=weights,
        sigmas=sigmas,
        confidence=confidence,
        sky_mask=sky_mask,
    )


def _check_npz_suffix(path: Path) -> Path:
    """np.savez silently appends .npz to other suffixes; make the convention
    explicit instead of numpy-implicit."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"bundle paths must end in .npz, got {path}")
    return path


def save_bundle(path: Path, bundle: DepthBundle) -> None:
    path = _check_npz_suffix(path)
    bundle = _validated(bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    present = {
        name: arr for name in _FIELDS if (arr := getattr(bundle, name)) is not None
    }
    np.savez_compressed(path, **present)


class DepthSource:
    """Directory of per-image bundles for one dump: <dump_dir>/<image_stem>.npz.

    Keyed by COLMAP image name (stem-matched: DSC_6489.JPG -> DSC_6489.npz).
    image_stems() lets attach_depths pre-check coverage of the whole database
    before writing anything, instead of dying mid-ingest.
    """

    def __init__(self, dump_dir: Path):
        if not Path(dump_dir).is_dir():
            raise FileNotFoundError(f"dump dir does not exist: {dump_dir}")
        self.dump_dir = Path(dump_dir)

    def bundle_path(self, image_name: str) -> Path:
        return self.dump_dir / f"{Path(image_name).stem}.npz"

    def load(self, image_name: str, expected_hw: tuple[int, int]) -> DepthBundle:
        p = self.bundle_path(image_name)
        if not p.exists():
            raise FileNotFoundError(
                f"no bundle for image {image_name!r} in {self.dump_dir} "
                f"(expected {p.name})"
            )
        return load_bundle(p, expected_hw)

    def image_stems(self) -> set[str]:
        files = list(self.dump_dir.glob("*.npz"))
        stems = {p.stem for p in files}
        if len(stems) != len(files):
            raise ValueError(
                f"stem collisions among bundles in {self.dump_dir} — "
                "images differing only in extension?"
            )
        return stems


def load_bundle(path: Path, expected_hw: tuple[int, int]) -> DepthBundle:
    """expected_hw is the image's native (H, W) as recorded in the COLMAP
    database; a mismatch means the dump was produced on a different grid."""
    path = _check_npz_suffix(path)
    with np.load(path) as npz:
        unknown = set(npz.files) - set(_FIELDS)
        if unknown:
            raise ValueError(f"Unknown arrays {sorted(unknown)} in {path}")
        if "estimated_depth" not in npz.files:
            raise ValueError(f"{path} has no estimated_depth array")
        arrays = {n: npz[n] if n in npz.files else None for n in _FIELDS}
    bundle = _validated(DepthBundle(**arrays))
    if bundle.estimated_depth.shape != tuple(expected_hw):
        raise ValueError(
            f"{path}: bundle grid {bundle.estimated_depth.shape} != expected "
            f"{tuple(expected_hw)} — stale dump from a different resolution?"
        )
    return bundle
