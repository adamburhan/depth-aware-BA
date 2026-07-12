"""
All DDL, blob layouts, and SQL for depthba_keypoint_depths and
depthba_depth_meta tables live here. Writers (attach_depths) and
readers go through these helpers. No other module contains SQL
or frombuffer calls for these tables.   
"""


import sqlite3, json
from datetime import datetime, timezone
import numpy as np
from dataclasses import dataclass

@dataclass
class KeypointDepth:
    """One sensor's depth measurement at one COLMAP keypoint.

    Keyed by (image_id, point2D_idx, sensor). point2D_idx is the row index
    into the image's COLMAP keypoints blob — the same index used in
    Image.points2D and in point3D track elements.

    Conventions:
    - modes are LINEAR depth in meters (raw measurement; log/inverse/linear
      residual parameterization is a BAConfig choice applied at factor
      construction, never here).
    - sigmas are in the SENSOR'S NATIVE space; which space is recorded once
      per sensor in depthba_depth_meta.sigma_space. Factor construction
      converts to the active residual space (e.g. sigma_log ~ sigma_z / z).
    - None means "sensor does not provide this" — distinct from 0.0/False.
    """
    image_id: int
    point2D_idx: int
    sensor: str # for example depthpro_gmm
    modes: np.ndarray
    weights: np.ndarray
    estimated_depth: float
    sigmas: np.ndarray | None
    confidence: float | None
    is_sky: bool | None
    
    
@dataclass
class SensorMeta:
    """Identity and provenance of one ingested sensor"""
    sensor: str
    num_modes: int
    method: str # extractor name, e.g. "mda", "gmm_patch"
    method_params: dict # json-serialized in the table
    sigma_space: str | None # "log", "linear", "inverse"; None if sensor provides no sigmas
    has_confidence: bool
    dump_dir: str
    dump_hash: str
    created_at: str
    

# DDL
CREATE_TABLES = """
-- Layout deliberately per-row (inspectability, PK-enforced identity); convertible
-- to per-image blobs behind this module's API if profiling ever demands it.
CREATE TABLE IF NOT EXISTS depthba_keypoint_depths (
    image_id        INTEGER NOT NULL,
    point2D_idx     INTEGER NOT NULL,
    sensor          TEXT    NOT NULL,
    modes           BLOB    NOT NULL,   -- K float32, linear depth [m]
    weights         BLOB    NOT NULL,   -- K float32, sum ~ 1
    estimated_depth REAL    NOT NULL,   -- sensor's committed depth [m]
    sigmas          BLOB,               -- K float32 in sensor-native space; NULL if none
    confidence      REAL,               -- NULL if sensor provides none
    is_sky          INTEGER,            -- 0/1; NULL if sensor has no sky head
    PRIMARY KEY (image_id, point2D_idx, sensor)
);
 
CREATE TABLE IF NOT EXISTS depthba_depth_meta (
    sensor         TEXT PRIMARY KEY,
    num_modes      INTEGER NOT NULL,
    method         TEXT    NOT NULL,
    method_params  TEXT    NOT NULL,    -- JSON
    sigma_space    TEXT,                -- NULL when sensor provides no sigmas
    has_confidence INTEGER NOT NULL,
    dump_dir       TEXT    NOT NULL,
    dump_hash      TEXT    NOT NULL,
    created_at     TEXT    NOT NULL     -- ISO 8601 UTC
);
"""

def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(CREATE_TABLES)

    
# Meta
def write_meta(conn: sqlite3.Connection, meta: SensorMeta) -> None:
    if meta.sigma_space is not None and meta.sigma_space not in ("log", "linear", "inverse"):
        raise ValueError(f"Invalid sigma_space {meta.sigma_space}")
    
    conn.execute(
        "INSERT INTO depthba_depth_meta "
        "(sensor, num_modes, method, method_params, sigma_space,"
        " has_confidence, dump_dir, dump_hash, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            meta.sensor,
            meta.num_modes,
            meta.method,
            json.dumps(meta.method_params, sort_keys=True),
            meta.sigma_space,
            int(meta.has_confidence),
            meta.dump_dir,
            meta.dump_hash,
            meta.created_at,
        ),
    )
    
def read_meta(conn: sqlite3.Connection, sensor: str) -> SensorMeta:
    row = conn.execute(
        "SELECT sensor, num_modes, method, method_params, sigma_space,"
        " has_confidence, dump_dir, dump_hash, created_at "
        "FROM depthba_depth_meta WHERE sensor=?",
        (sensor,),
    ).fetchone()
    if row is None:
        available = [r[0] for r in conn.execute("SELECT sensor FROM depthba_depth_meta")]
        raise KeyError(
            f"sensor {sensor!r} not ingested into this database "
            f"(available: {available or 'none'})"
        )
    (sensor, num_modes, method, params_json, sigma_space,
     has_conf, dump_dir, dump_hash, created_at) = row
    return SensorMeta(
        sensor=sensor,
        num_modes=num_modes,
        method=method,
        method_params=json.loads(params_json),
        sigma_space=sigma_space,
        has_confidence=bool(has_conf),
        dump_dir=dump_dir,
        dump_hash=dump_hash,
        created_at=created_at,
    )
 
def list_sensors(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT sensor FROM depthba_depth_meta ORDER BY sensor")]
 
 
def delete_sensor(conn: sqlite3.Connection, sensor: str) -> None:
    """Remove a sensor entirely (rows + meta). Used by the --force re-ingest path."""
    conn.execute("DELETE FROM depthba_keypoint_depths WHERE sensor=?", (sensor,))
    conn.execute("DELETE FROM depthba_depth_meta WHERE sensor=?", (sensor,))
 
 
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# depth rows
def _encode_optional_float(x: float | None, what: str) -> float | None:
    """SQLite silently converts NaN REAL values to NULL; forbid the ambiguity."""
    if x is None:
        return None
    x = float(x)
    if np.isnan(x):
        raise ValueError(
            f"{what} is NaN — use None for 'sensor does not provide this' "
            "(SQLite would silently store NaN as NULL)"
        )
    return x
 
 
def write_depths(conn: sqlite3.Connection, rows: list[KeypointDepth]) -> None:
    """Plain INSERT: colliding with existing (image_id, point2D_idx, sensor)
    raises IntegrityError. Re-ingest requires delete_sensor() first."""
    conn.executemany(
        "INSERT INTO depthba_keypoint_depths "
        "(image_id, point2D_idx, sensor, modes, weights, estimated_depth,"
        " sigmas, confidence, is_sky) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (
                r.image_id,
                r.point2D_idx,
                r.sensor,
                np.ascontiguousarray(r.modes, dtype=np.float32).tobytes(),
                np.ascontiguousarray(r.weights, dtype=np.float32).tobytes(),
                _encode_optional_float(r.estimated_depth, "estimated_depth"),
                None if r.sigmas is None
                else np.ascontiguousarray(r.sigmas, dtype=np.float32).tobytes(),
                _encode_optional_float(r.confidence, "confidence"),
                None if r.is_sky is None else int(r.is_sky),
            )
            for r in rows
        ],
    )
 
 
def _decode_blob(blob: bytes, num_modes: int, what: str, image_id: int, idx: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float32).copy()
    if arr.shape != (num_modes,):
        raise ValueError(
            f"image {image_id} kp {idx}: {what} blob has {arr.size} floats, "
            f"meta says K={num_modes} — table/meta inconsistency"
        )
    return arr
 
 
def read_depths_for_image(
    conn: sqlite3.Connection, image_id: int, sensor: str, num_modes: int
) -> dict[int, KeypointDepth]:
    """Read all rows for one image and sensor, keyed by point2D_idx.
 
    num_modes comes from read_meta(); every blob is validated against it.
    """
    out: dict[int, KeypointDepth] = {}
    for (iid, idx, s, modes_b, weights_b, est, sigmas_b, conf, sky) in conn.execute(
        "SELECT image_id, point2D_idx, sensor, modes, weights, estimated_depth,"
        " sigmas, confidence, is_sky "
        "FROM depthba_keypoint_depths WHERE image_id=? AND sensor=?",
        (image_id, sensor),
    ):
        out[idx] = KeypointDepth(
            image_id=iid,
            point2D_idx=idx,
            sensor=s,
            modes=_decode_blob(modes_b, num_modes, "modes", iid, idx),
            weights=_decode_blob(weights_b, num_modes, "weights", iid, idx),
            estimated_depth=est,
            sigmas=None if sigmas_b is None
            else _decode_blob(sigmas_b, num_modes, "sigmas", iid, idx),
            confidence=conf,
            is_sky=None if sky is None else bool(sky),
        )
    return out
 
 
def count_rows(conn: sqlite3.Connection, sensor: str) -> int:
    (n,) = conn.execute(
        "SELECT COUNT(*) FROM depthba_keypoint_depths WHERE sensor=?", (sensor,)
    ).fetchone()
    return n