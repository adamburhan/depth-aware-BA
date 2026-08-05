"""Does a sensor's stored data actually carry bimodality, and are its sigmas informative?

Answers, from the database alone, why a mixture arm might be indistinguishable from
a unimodal one: if the EM's gate collapsed mode 1 onto mode 0 almost everywhere, or
if every sigma sits on its floor, the two factor families are numerically the same.

  python scripts/inspect_sensor.py --db .../database.db --sensor amb3r_gmm
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np

from depthba.depth import schema


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--sensor", default="amb3r_gmm")
    ap.add_argument("--sep_min", type=float, default=0.1,
                    help="log separation the extractor's gate used")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    meta = schema.read_meta(conn, args.sensor)
    print(f"{args.sensor}: {meta.method}, K={meta.num_modes}, "
          f"sigma_space={meta.sigma_space}, params={meta.method_params}")

    image_ids = [i for (i,) in conn.execute(
        "SELECT DISTINCT image_id FROM depthba_keypoint_depths WHERE sensor=?",
        (args.sensor,))]
    modes, sigmas, weights, sky, per_image = [], [], [], [], []
    for image_id in image_ids:
        rows = schema.read_depths_for_image(conn, image_id, args.sensor, meta.num_modes)
        m = np.array([r.modes for r in rows.values()])
        s = np.array([r.is_sky for r in rows.values()], dtype=bool)
        modes.append(m)
        sky.append(s)
        if meta.num_modes > 1:
            sep = np.abs(np.log(m[:, 1]) - np.log(m[:, 0]))
            per_image.append((sep[~s] >= args.sep_min).mean() if (~s).any() else 0.0)
        if rows and next(iter(rows.values())).sigmas is not None:
            sigmas.append(np.array([r.sigmas for r in rows.values()]))
            weights.append(np.array([r.weights for r in rows.values()]))
    conn.close()

    modes = np.concatenate(modes)
    sky = np.concatenate(sky)
    print(f"{len(modes)} rows over {len(image_ids)} images ({sky.sum()} sky)")

    if meta.num_modes > 1:
        sep = np.abs(np.log(modes[:, 1]) - np.log(modes[:, 0]))[~sky]
        live = sep >= args.sep_min
        print(f"\nMODE SEPARATION |log(m1/m0)| over non-sky rows")
        print(f"  collapsed (mode1 == mode0 exactly): {(sep == 0).mean():.1%}")
        print(f"  live (>= sep_min={args.sep_min}):    {live.mean():.1%}")
        if live.any():
            q = np.percentile(sep[live], [50, 90, 99])
            print(f"  live separation med/p90/p99:       "
                  f"{q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f} log units")
        pim = np.array(per_image)
        print(f"  per-image live fraction med/min/max: "
              f"{np.median(pim):.1%} / {pim.min():.1%} / {pim.max():.1%}")
        print("  -> if 'live' is a few %, the mixture is numerically unimodal "
              "almost everywhere and gmm-vs-unimodal cannot separate here.")

    if sigmas:
        sig = np.concatenate(sigmas)[~sky]
        floor = sig.min()
        print(f"\nSIGMAS (stored, {meta.sigma_space} space)")
        print(f"  min {floor:.4f}  med {np.median(sig):.4f}  p90 {np.percentile(sig, 90):.4f}")
        print(f"  at the floor (within 1%): {(sig <= floor * 1.01).mean():.1%}")
        print("  -> if nearly all sit on the floor, stored sigmas carry no "
              "per-keypoint information and act as a constant.")
        w = np.concatenate(weights)[~sky]
        print(f"\nWEIGHTS  med {np.median(w, axis=0)}  "
              f"(uniform_prior stores 0.5/0.5 by design)")


if __name__ == "__main__":
    main()
