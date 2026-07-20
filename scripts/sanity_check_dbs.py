"""Post-attach sanity sweep over the T&T amb3r databases.

Re-verifies, directly against the persisted blobs, everything attach_depths'
check_measurements guaranteed at ingest time (value validity), plus things
that can only be checked in aggregate afterward (row-count completeness
against the COLMAP keypoints table, cross-scene bimodal rate, value ranges).
Read-only against depth data; only writes are idempotent CREATE TABLE IF NOT
EXISTS calls so a never-attached db reports "MISSING" instead of crashing.

  python scripts/sanity_check_dbs.py \
      --root $SCRATCH/experiments/depth-aware-ba/tt_amb3r
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np

from depthba.depth import schema

SEQUENCES = ["Barn", "Caterpillar", "Church", "Courthouse", "Ignatius", "Meetingroom", "Truck"]

# sensor -> (expected num_modes, expected sigma_space)
EXPECTED = {
    "amb3r": (1, None),
    "amb3r_gmm": (2, "log"),
}


def _keypoint_counts(conn: sqlite3.Connection) -> dict[int, int]:
    """image_id -> num keypoints, straight from COLMAP's own table."""
    return dict(conn.execute("SELECT image_id, rows FROM keypoints").fetchall())


def _load_sensor_arrays(conn: sqlite3.Connection, sensor: str, num_modes: int):
    rows = conn.execute(
        "SELECT modes, weights, estimated_depth, sigmas, confidence, is_sky "
        "FROM depthba_keypoint_depths WHERE sensor=?",
        (sensor,),
    ).fetchall()
    n = len(rows)
    if n == 0:
        return None
    modes = np.frombuffer(b"".join(r[0] for r in rows), dtype=np.float32).reshape(n, num_modes)
    weights = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(n, num_modes)
    est = np.array([r[2] for r in rows], dtype=np.float64)
    sigmas = None
    if rows[0][3] is not None:
        sigmas = np.frombuffer(b"".join(r[3] for r in rows), dtype=np.float32).reshape(n, num_modes)
    conf = None
    if rows[0][4] is not None:
        conf = np.array([r[4] for r in rows], dtype=np.float64)
    is_sky = None
    if rows[0][5] is not None:
        is_sky = np.array([bool(r[5]) for r in rows])
    return modes, weights, est, sigmas, conf, is_sky


def check_sensor(conn: sqlite3.Connection, sensor: str, kp_counts: dict[int, int]) -> list[str]:
    problems = []
    try:
        meta = schema.read_meta(conn, sensor)
    except KeyError:
        return [f"MISSING: sensor {sensor!r} not ingested"]

    exp_modes, exp_sigma_space = EXPECTED.get(sensor, (meta.num_modes, meta.sigma_space))
    if meta.num_modes != exp_modes:
        problems.append(f"num_modes={meta.num_modes}, expected {exp_modes}")
    if meta.sigma_space != exp_sigma_space:
        problems.append(f"sigma_space={meta.sigma_space!r}, expected {exp_sigma_space!r}")

    n_rows = schema.count_rows(conn, sensor)
    n_kp = sum(kp_counts.values())
    if n_rows != n_kp:
        problems.append(f"row count {n_rows} != total COLMAP keypoints {n_kp}")

    # per-image completeness (row count == keypoint count for every image)
    per_image = dict(
        conn.execute(
            "SELECT image_id, COUNT(*) FROM depthba_keypoint_depths WHERE sensor=? "
            "GROUP BY image_id",
            (sensor,),
        ).fetchall()
    )
    mismatched = [
        iid for iid, n in kp_counts.items() if per_image.get(iid, 0) != n
    ]
    if mismatched:
        problems.append(
            f"{len(mismatched)} images have row count != keypoint count "
            f"(e.g. image_id={mismatched[0]}: {per_image.get(mismatched[0], 0)} rows vs "
            f"{kp_counts[mismatched[0]]} keypoints)"
        )

    arrays = _load_sensor_arrays(conn, sensor, meta.num_modes)
    if arrays is None:
        problems.append("0 rows persisted despite meta present")
        return problems
    modes, weights, est, sigmas, conf, is_sky = arrays
    ok = slice(None) if is_sky is None else ~is_sky

    if not np.isfinite(modes[ok]).all() or (modes[ok] <= 0).any():
        problems.append("non-finite or <=0 modes in non-sky rows")
    if not np.isfinite(est[ok]).all() or (est[ok] <= 0).any():
        problems.append("non-finite or <=0 estimated_depth in non-sky rows")
    if not np.isfinite(weights[ok]).all() or (weights[ok] < 0).any():
        problems.append("non-finite or negative weights in non-sky rows")
    wsum = weights[ok].sum(axis=1)
    if len(wsum) and wsum.max() > 1 + 1e-3:
        problems.append(f"weight rows sum above 1 (max={wsum.max():.4f})")
    if sigmas is not None:
        if not np.isfinite(sigmas[ok]).all() or (sigmas[ok] <= 0).any():
            problems.append("non-finite or <=0 sigmas in non-sky rows")

    # informational, not a failure
    print(
        f"    {sensor:12s} rows={len(modes):>8d}  "
        f"depth[{modes[ok].min():.2f},{modes[ok].max():.2f}]m  "
        f"est_depth median={np.median(est[ok]):.2f}m"
        + (f"  sigma median={np.median(sigmas[ok]):.4f}" if sigmas is not None else "")
        + (
            f"  bimodal_rate={(np.abs(np.log(modes[ok, 1] / modes[ok, 0])) > 1e-6).mean():.1%}"
            if meta.num_modes > 1
            else ""
        )
    )
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="dir containing <sequence>/database.db")
    ap.add_argument("--sequences", nargs="+", default=SEQUENCES)
    ap.add_argument("--sensors", nargs="+", default=list(EXPECTED))
    args = ap.parse_args()

    any_problems = False
    for seq in args.sequences:
        db_path = args.root / seq / "database.db"
        print(f"\n{seq}  ({db_path})")
        if not db_path.exists():
            print("  MISSING: no database.db")
            any_problems = True
            continue
        conn = sqlite3.connect(db_path)
        try:
            schema.create_tables(conn)  # no-op if already present; lets a
            # never-attached db report cleanly instead of raising on a
            # missing depthba_depth_meta table
            kp_counts = _keypoint_counts(conn)
            print(f"  {len(kp_counts)} images, {sum(kp_counts.values())} keypoints total")
            for sensor in args.sensors:
                problems = check_sensor(conn, sensor, kp_counts)
                for p in problems:
                    print(f"    [{sensor}] PROBLEM: {p}")
                    any_problems = True
        finally:
            conn.close()

    print("\n" + ("SOME PROBLEMS FOUND — see above" if any_problems else "all checks passed"))


if __name__ == "__main__":
    main()
