"""Against the laser mesh, is the second mode ever the right answer?

Casts one ray per live keypoint (no rendering). The sensor's global scale is fitted
TO the mesh, so this asks "given the best possible global scale, which mode is
nearer the truth" -- independent of BA's frozen alpha.

  python scripts/oracle_modes.py --db database.db --model ba/<variant>/0 \
      --gt $DATA/<scene>/dslr/colmap --mesh $DATA/<scene>/scans/mesh_aligned_0.05.ply
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np

import pycolmap

from depthba.backends.residuals import maxmix_scores, whitened_residuals
from depthba.depth import schema
from depthba.eval import mesh_oracle


def read_rows(db: Path, sensor: str):
    conn = sqlite3.connect(db)
    meta = schema.read_meta(conn, sensor)
    if meta.num_modes < 2:
        raise SystemExit(f"{sensor} is unimodal (K={meta.num_modes})")
    ids = [i for (i,) in conn.execute(
        "SELECT DISTINCT image_id FROM depthba_keypoint_depths WHERE sensor=?", (sensor,))]
    rows = {i: schema.read_depths_for_image(conn, i, sensor, meta.num_modes) for i in ids}
    conn.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True, help="dslr/colmap, scan-aligned")
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--sensor", default="amb3r_gmm")
    ap.add_argument("--sep_min", type=float, default=0.1)
    ap.add_argument("--anchor_tol", type=float, default=0.1,
                    help="|log(mesh/alpha*mode0)| above this = the anchor was wrong")
    args = ap.parse_args()
    pycolmap.logging.minloglevel = 2

    rec, gt = pycolmap.Reconstruction(args.model), pycolmap.Reconstruction(args.gt)
    rows_by_image = read_rows(args.db, args.sensor)
    scale, R, t, n_common = mesh_oracle.align_to(rec, gt)
    print(f"{n_common} common images, SfM->scan Sim(3) scale {scale:.6f}")

    origins, dirs, norms, records = mesh_oracle.live_rays(
        rec, rows_by_image, args.sep_min, (scale, R, t))
    if not len(records):
        raise SystemExit("no live keypoints")
    print(f"{len(records)} live keypoints; casting rays")
    depth = mesh_oracle.cast(args.mesh, origins, dirs, norms)

    hit = np.isfinite(depth)
    m0 = np.array([r[2].modes[0] for r in records])
    m1 = np.array([r[2].modes[1] for r in records])
    alpha = float(np.median(depth[hit] / m0[hit]))
    e0 = np.abs(np.log(depth) - np.log(alpha * m0))
    e1 = np.abs(np.log(depth) - np.log(alpha * m1))
    better1 = (e1 < e0) & hit
    wrong = (e0 > args.anchor_tol) & hit

    print(f"mesh hit on {hit.sum()}/{len(records)}; alpha fitted to mesh = {alpha:.4f}")
    print(f"\nmode 1 closer to mesh than mode 0: {better1[hit].mean():.1%}")
    print(f"  anchor wrong (|log err| > {args.anchor_tol}): {wrong[hit].mean():.1%} of live")
    if wrong.any():
        print(f"    of those, mode 1 is closer: {better1[wrong].mean():.1%}  <-- the population")
    ok = hit & ~wrong
    if ok.any():
        print(f"  anchor fine: mode 1 closer anyway {better1[ok].mean():.1%} "
              "(should be low; high = the EM is fitting noise)")

    z = np.array([r[3] if r[3] is not None else np.nan for r in records])
    tri = hit & np.isfinite(z)
    if tri.any():
        alpha_z = float(np.median(z[tri] / m0[tri]))
        ba_win = np.zeros(len(records), bool)
        for i in np.flatnonzero(tri):
            row = records[i][2]
            r = whitened_residuals(z[i], row.modes, row.sigmas, alpha_z)
            ba_win[i] = int(np.argmin(maxmix_scores(
                r, row.sigmas, np.maximum(row.weights, 1e-20)))) != 0
        tp = (ba_win & better1 & tri).sum()
        fp = (ba_win & ~better1 & tri).sum()
        fn = (~ba_win & better1 & tri).sum()
        print(f"\nBA's selection on {tri.sum()} triangulated live keypoints:")
        print(f"  switched {tp + fp} ({tp} right, {fp} wrong), missed {fn}")
        print(f"  precision {tp / max(tp + fp, 1):.1%}  recall {tp / max(tp + fn, 1):.1%}  "
              f"net {tp - fp:+d} (never-switch = 0)")


if __name__ == "__main__":
    main()
