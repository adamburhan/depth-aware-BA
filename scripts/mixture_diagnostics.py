"""How much the gmm mixture actually differs from the unimodal prior.

Two questions, both cheap:

  prior side (database only) -- what fraction of rows carry a second mode at
  all, how far it sits from the first, and whether mode 0 is the unimodal
  sensor's depth. A row whose modes coincide is bit-identical to a unimodal
  factor: the max-mixture residual is the winning mode's plain whitened
  error, so two equal modes with equal weights tie and mode 0 wins.

  solution side (needs a finished ba) -- how often the mixture actually
  selects mode 1 at convergence. That is the only channel through which gmm
  can produce different geometry than unimodal.

  python scripts/mixture_diagnostics.py \
      --root $SCRATCH/experiments/depth-aware-ba/scannetpp_amb3r \
      --sequence 0d2ee665be --variant amb3r_gmm_all_sigma_scale1.0
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np

import pycolmap

from depthba.backends.residuals import maxmix_scores, whitened_residuals
from depthba.depth import schema


def largest_submodel(variant_dir: Path) -> Path:
    subs = sorted(p for p in variant_dir.iterdir() if p.is_dir() and p.name.isdigit())
    if not subs:
        raise FileNotFoundError(f"{variant_dir}: no sub-models")
    if len(subs) == 1:
        return subs[0]
    sizes = [pycolmap.Reconstruction(p).num_reg_images() for p in subs]
    print(f"{variant_dir.name}: sub-models {sizes}, using {subs[sizes.index(max(sizes))].name}")
    return subs[sizes.index(max(sizes))]


def prior_stats(conn, sensor: str, unimodal_sensor: str, max_images: int) -> None:
    meta = schema.read_meta(conn, sensor)
    if meta.num_modes < 2:
        print(f"{sensor}: K={meta.num_modes}, nothing to compare")
        return
    image_ids = [
        i for (i,) in conn.execute(
            "SELECT DISTINCT image_id FROM depthba_keypoint_depths WHERE sensor=?",
            (sensor,),
        )
    ][:max_images]
    try:
        uni_meta = schema.read_meta(conn, unimodal_sensor)
    except Exception:
        uni_meta = None

    sep, agree = [], []
    for image_id in image_ids:
        rows = schema.read_depths_for_image(conn, image_id, sensor, meta.num_modes)
        uni = (
            schema.read_depths_for_image(conn, image_id, unimodal_sensor, uni_meta.num_modes)
            if uni_meta is not None else {}
        )
        for idx, row in rows.items():
            sigma = float(row.sigmas[0]) if row.sigmas is not None else 0.05
            sep.append(abs(np.log(row.modes[1] / row.modes[0])) / sigma)
            if idx in uni:
                agree.append(abs(np.log(row.modes[0] / uni[idx].modes[0])) / sigma)

    sep = np.asarray(sep)
    print(f"\nprior ({sensor}, K={meta.num_modes}, {len(sep)} rows over "
          f"{len(image_ids)} images)")
    print(f"  rows with a second mode : {(sep > 0).mean():.4f}")
    print(f"  separation [sigma] pctl : "
          f"{np.percentile(sep, [50, 75, 90, 95, 99]).round(2)}")
    if len(sep[sep > 0]):
        print(f"  separation | detected   : median {np.median(sep[sep > 0]):.2f}")
    if agree:
        agree = np.asarray(agree)
        print(f"  |mode0 - {unimodal_sensor}| [sigma] : median {np.median(agree):.4f}, "
              f"frac > 1: {(agree > 1).mean():.4f}")


def winner_stats(conn, model_dir: Path, sensor: str) -> None:
    meta = schema.read_meta(conn, sensor)
    rec = pycolmap.Reconstruction(model_dir)

    obs, ratios = [], []
    for image in rec.images.values():
        rows = schema.read_depths_for_image(conn, image.image_id, sensor, meta.num_modes)
        if not rows:
            continue
        cam_from_world = image.cam_from_world
        if callable(cam_from_world):
            cam_from_world = cam_from_world()
        for idx, p2d in enumerate(image.points2D):
            row = rows.get(idx)
            if row is None or row.is_sky or not p2d.has_point3D():
                continue
            z = (cam_from_world * rec.points3D[p2d.point3D_id].xyz)[2]
            if z > 0:
                obs.append((z, row))
                ratios.append(z / row.modes[0])

    if not obs:
        print(f"\nsolution ({model_dir}): no depth-carrying observations")
        return
    # The ba freezes one shared alpha that is never written out; the global
    # median ratio reproduces it closely enough for a winner rate.
    alpha = float(np.median(ratios))

    wins1 = degenerate = 0
    for z, row in obs:
        sigmas = (
            row.sigmas if row.sigmas is not None
            else np.full(meta.num_modes, 0.05)
        )
        residuals = whitened_residuals(z, row.modes, sigmas, alpha, 0.0)
        weights = np.maximum(row.weights.astype(np.float64), 1e-20)
        wins1 += int(np.argmin(maxmix_scores(residuals, sigmas, weights))) == 1
        degenerate += row.modes[0] == row.modes[1]

    n = len(obs)
    live = max(n - degenerate, 1)
    print(f"\nsolution ({model_dir.parent.name}/{model_dir.name}, alpha {alpha:.4f})")
    print(f"  observations            : {n}")
    print(f"  degenerate rows         : {degenerate / n:.4f}")
    print(f"  mode 1 wins             : {wins1 / n:.4f} of all, "
          f"{wins1 / live:.4f} of non-degenerate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--variant", default="amb3r_gmm_all_sigma_scale1.0")
    ap.add_argument("--sensor", default="amb3r_gmm")
    ap.add_argument("--unimodal_sensor", default="amb3r")
    ap.add_argument("--max_images", type=int, default=50, help="prior stats only")
    args = ap.parse_args()

    dslr = args.root / args.sequence / "dslr"
    conn = sqlite3.connect(dslr / "database.db")
    try:
        prior_stats(conn, args.sensor, args.unimodal_sensor, args.max_images)
        winner_stats(conn, largest_submodel(dslr / "ba" / args.variant), args.sensor)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
