"""How often does the max-mixture actually pick mode 1, among keypoints where a second mode exists?

FINAL-STATE ONLY: mode flips during incremental mapping are invisible here.

Alpha under shared_scale is frozen very early in mapping and never recomputed, so
runs predating alpha logging cannot have it reproduced exactly -- pass --alpha from
the BA log if you have it. Otherwise it is re-estimated from the final
reconstruction and reported across a jitter range, since the winner of a
near-midpoint keypoint depends on it.

  python scripts/winner_stats.py --db .../database.db \
      --model .../ba/<variant>/0 --sensor amb3r_gmm
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np

import pycolmap

from depthba.backends.residuals import maxmix_scores, whitened_residuals
from depthba.depth import schema


def collect(rec, rows_by_image):
    """Triangulated non-sky keypoints: (z_cam, modes, sigmas, weights)."""
    z, modes, sigmas, weights = [], [], [], []
    for image in rec.images.values():
        rows = rows_by_image.get(image.image_id)
        if not rows:
            continue
        cam_from_world = image.cam_from_world
        if callable(cam_from_world):
            cam_from_world = cam_from_world()
        for idx, p2d in enumerate(image.points2D):
            row = rows.get(idx)
            if row is None or row.is_sky or not p2d.has_point3D():
                continue
            depth = (cam_from_world * rec.points3D[p2d.point3D_id].xyz)[2]
            if depth <= 0:
                continue
            z.append(depth)
            modes.append(row.modes)
            sigmas.append(row.sigmas if row.sigmas is not None else np.full(len(row.modes), np.nan))
            weights.append(row.weights)
    return (np.array(z), np.array(modes), np.array(sigmas), np.array(weights))


def winner_fraction(z, modes, sigmas, weights, alpha):
    """Fraction of the given factors whose max-mixture winner is not mode 0."""
    wins = 0
    for i in range(len(z)):
        r = whitened_residuals(z[i], modes[i], sigmas[i], alpha)
        k = int(np.argmin(maxmix_scores(r, sigmas[i], np.maximum(weights[i], 1e-20))))
        wins += k != 0
    return wins / max(len(z), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--sensor", default="amb3r_gmm")
    ap.add_argument("--sep_min", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=None,
                    help="the frozen alpha from the BA log; re-estimated if omitted")
    ap.add_argument("--alpha_jitter", type=float, default=0.05,
                    help="report sensitivity over alpha*(1 +/- this)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    meta = schema.read_meta(conn, args.sensor)
    if meta.num_modes < 2:
        raise SystemExit(f"{args.sensor} is unimodal (K={meta.num_modes})")
    image_ids = [i for (i,) in conn.execute(
        "SELECT DISTINCT image_id FROM depthba_keypoint_depths WHERE sensor=?",
        (args.sensor,))]
    rows_by_image = {
        i: schema.read_depths_for_image(conn, i, args.sensor, meta.num_modes)
        for i in image_ids
    }
    conn.close()

    rec = pycolmap.Reconstruction(args.model)
    z, modes, sigmas, weights = collect(rec, rows_by_image)
    if not len(z):
        raise SystemExit("no triangulated non-sky rows in this reconstruction")

    alpha = args.alpha
    if alpha is None:
        alpha = float(np.median(z / modes[:, 0]))
        print(f"alpha re-estimated from the final reconstruction: {alpha:.6f} "
              "(NOT the frozen value BA used -- see --alpha)")
    else:
        print(f"alpha from BA log: {alpha:.6f}")

    sep = np.abs(np.log(modes[:, 1]) - np.log(modes[:, 0]))
    live = sep >= args.sep_min
    print(f"{len(z)} triangulated non-sky factors, {live.sum()} live "
          f"(separation >= {args.sep_min}) = {live.mean():.1%}")

    for scale in (1 - args.alpha_jitter, 1.0, 1 + args.alpha_jitter):
        a = alpha * scale
        frac_live = winner_fraction(
            z[live], modes[live], sigmas[live], weights[live], a)
        print(f"  alpha x{scale:.2f}: mode 1 wins {frac_live:.2%} of live factors "
              f"({int(frac_live * live.sum())} keypoints), "
              f"{frac_live * live.mean():.3%} of all")


if __name__ == "__main__":
    main()
