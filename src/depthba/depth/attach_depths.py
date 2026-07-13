"""
Ingest one sensor into the COLMAP database: for every image, load its
DepthBundle from the dump, extract per-keypoint measurements, and write
KeypointDepth rows plus the SensorMeta row.

Orchestration only — file I/O lives in source.py, sampling in extractors,
SQL in schema.py. The value-level sanity checks that source.py deliberately
skips (finiteness, positivity) happen here, on non-sky keypoints; sky rows
carry fabricated values by design and are excluded downstream via the
is_sky flag. Weight-row sums are only bounded above: deficits are
legitimate (MDA-style dumps may carry sky mass outside the stored K
weights), while sums above 1 are wiring bugs under any reading.

The whole ingest is one transaction: a failure at image 200 rolls back to
zero rows, which is what makes write_depths' plain-INSERT semantics livable.
"""

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

import pycolmap

from depthba.config import AttachConfig
from depthba.depth import schema
from depthba.depth.extractors import EXTRACTORS, DepthMeasurements
from depthba.depth.source import DepthSource


def compute_dump_hash(source: DepthSource) -> str:
    """Cheap provenance: sha256 over sorted (filename, size) of the dump's
    bundles. Catches added/removed/resized files, not byte-level edits."""
    h = hashlib.sha256()
    for p in sorted(source.dump_dir.glob("*.npz")):
        h.update(f"{p.name}:{p.stat().st_size}\n".encode())
    return h.hexdigest()


def check_measurements(image_name: str, m: DepthMeasurements) -> None:
    """Value-level sanity at ingest; sky keypoints are exempt (fabricated)."""
    ok = slice(None) if m.is_sky is None else ~m.is_sky
    modes, weights = m.modes[ok], m.weights[ok]
    est = m.estimated_depth[ok]

    def bad(what: str):
        raise ValueError(f"{image_name}: {what}")

    if not np.isfinite(modes).all() or (modes <= 0).any():
        bad("modes must be finite and > 0 (linear meters)")
    if not np.isfinite(est).all() or (est <= 0).any():
        bad("estimated_depth must be finite and > 0")
    if not np.isfinite(weights).all() or (weights < 0).any():
        bad("weights must be finite and >= 0")
    if len(weights) and weights.sum(axis=1).max() > 1 + 1e-3:
        bad("weight rows must not sum above 1 (transposed or double-counted weights?)")
    if m.sigmas is not None:
        sigmas = m.sigmas[ok]
        if not np.isfinite(sigmas).all() or (sigmas <= 0).any():
            bad("sigmas must be finite and > 0")


def _to_rows(
    image_id: int, sensor: str, m: DepthMeasurements
) -> list[schema.KeypointDepth]:
    return [
        schema.KeypointDepth(
            image_id=image_id,
            point2D_idx=i,
            sensor=sensor,
            modes=m.modes[i],
            weights=m.weights[i],
            estimated_depth=float(m.estimated_depth[i]),
            sigmas=None if m.sigmas is None else m.sigmas[i],
            confidence=None if m.confidence is None else float(m.confidence[i]),
            is_sky=None if m.is_sky is None else bool(m.is_sky[i]),
        )
        for i in range(len(m.estimated_depth))
    ]


def run(
    config: AttachConfig, db_path: Path, dump_dir: Path, force: bool = False
) -> None:
    if config.method not in EXTRACTORS:
        raise ValueError(
            f"unknown method {config.method!r} (available: {sorted(EXTRACTORS)})"
        )
    extractor_fn = EXTRACTORS[config.method]
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"database does not exist: {db_path}")
    source = DepthSource(dump_dir)

    # Read everything needed from the COLMAP side up front, so no pycolmap
    # handle is open once the write transaction starts.
    with pycolmap.Database.open(db_path) as db:
        images = db.read_all_images()
        cameras = {c.camera_id: c for c in db.read_all_cameras()}
        keypoints = {im.image_id: db.read_keypoints(im.image_id) for im in images}
    if not images:
        raise ValueError(f"no images in {db_path}")

    # Coverage pre-check: fail before writing anything.
    missing = {Path(im.name).stem for im in images} - source.image_stems()
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images have no bundle in {source.dump_dir}: "
            f"{sorted(missing)[:10]}"
        )

    conn = sqlite3.connect(db_path)
    try:
        schema.create_tables(conn)  # executescript autocommits; keep outside txn
        with conn:
            if config.sensor in schema.list_sensors(conn):
                if not force:
                    raise ValueError(
                        f"sensor {config.sensor!r} already ingested; "
                        "use --force to re-ingest"
                    )
                schema.delete_sensor(conn, config.sensor)

            num_modes, has_confidence, has_sigmas = None, None, None
            for im in images:
                cam = cameras[im.camera_id]
                bundle = source.load(im.name, (cam.height, cam.width))
                m = extractor_fn(bundle, keypoints[im.image_id], config.method_params)
                # Row i of DepthMeasurements IS keypoint i (point2D_idx = i);
                # this count check is the only guard on that positional
                # identity — a filtering extractor would misattribute every
                # row after the drop. (A future gt sensor with raycast holes
                # will need a rows<=keypoints carve-out plus dropped-count
                # provenance; strict equality is correct for dense sensors.)
                n_kp = len(keypoints[im.image_id])
                if len(m.estimated_depth) != n_kp:
                    raise ValueError(
                        f"{im.name}: extractor returned {len(m.estimated_depth)} "
                        f"measurements for {n_kp} keypoints"
                    )
                check_measurements(im.name, m)
                if num_modes is None:  # first image fixes the sensor shape
                    num_modes = m.modes.shape[1]
                    has_confidence = m.confidence is not None
                    has_sigmas = m.sigmas is not None
                    if has_sigmas != (config.sigma_space is not None):
                        raise ValueError(
                            f"config sigma_space={config.sigma_space!r} but extractor "
                            f"{'emits' if has_sigmas else 'does not emit'} sigmas"
                        )
                elif (
                    m.modes.shape[1] != num_modes
                    or (m.confidence is not None) != has_confidence
                    or (m.sigmas is not None) != has_sigmas
                ):
                    raise ValueError(
                        f"{im.name}: measurement shape (K={m.modes.shape[1]}, "
                        f"confidence={m.confidence is not None}, "
                        f"sigmas={m.sigmas is not None}) differs from earlier images "
                        f"(K={num_modes}, confidence={has_confidence}, sigmas={has_sigmas})"
                    )
                schema.write_depths(conn, _to_rows(im.image_id, config.sensor, m))

            schema.write_meta(
                conn,
                schema.SensorMeta(
                    sensor=config.sensor,
                    num_modes=num_modes,
                    method=config.method,
                    method_params=config.method_params,
                    sigma_space=config.sigma_space,
                    has_confidence=has_confidence,
                    dump_dir=str(source.dump_dir),
                    dump_hash=compute_dump_hash(source),
                    created_at=schema.now_iso(),
                ),
            )
    finally:
        conn.close()
