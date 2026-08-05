"""How often does the max-mixture pick mode 1, and on which population?

FINAL-STATE ONLY: mode flips during incremental mapping are invisible here. The
selection rule is invariant to a uniform sigma scaling (equal sigmas/weights make
it argmin|r|), so engagement differences across sigma_scale cells reflect where
points converged, not a change in the rule.

Alpha under shared_scale is frozen very early in mapping and never recomputed, so
runs predating alpha logging cannot have it reproduced exactly -- pass --alpha from
the BA log if you have it.

  python scripts/winner_stats.py --db database.db --sensor amb3r_gmm \
      --models ba/*/0 [--detail]
"""

import argparse
import collections
import sqlite3
from pathlib import Path

import numpy as np

import pycolmap

from depthba.backends.residuals import maxmix_scores, whitened_residuals
from depthba.depth import schema


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


def collect(rec, rows_by_image):
    """Per triangulated non-sky observation: depth, modes, sigmas, weights, track id."""
    z, modes, sigmas, weights, track = [], [], [], [], []
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
            sigmas.append(row.sigmas)
            weights.append(row.weights)
            track.append(p2d.point3D_id)
    return (np.array(z), np.array(modes), np.array(sigmas, dtype=float),
            np.array(weights, dtype=float), np.array(track))


def winners(z, modes, sigmas, weights, alpha):
    r = whitened_residuals(z[:, None], modes, sigmas, alpha)
    return np.argmin(maxmix_scores(r, sigmas, np.maximum(weights, 1e-20)), axis=1)


def rates(win, label, edges_of):
    """Win rate within quartile bins of some quantity, over the live subset."""
    q = np.percentile(edges_of, [25, 50, 75])
    bins = np.digitize(edges_of, q)
    return label + " " + " ".join(
        f"q{b + 1}:{win[bins == b].mean():.0%}" if (bins == b).any() else f"q{b + 1}:-"
        for b in range(4)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--models", type=Path, nargs="+", required=True)
    ap.add_argument("--sensor", default="amb3r_gmm")
    ap.add_argument("--sep_min", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--detail", action="store_true",
                    help="per-model breakdowns by depth, separation and track")
    args = ap.parse_args()
    pycolmap.logging.minloglevel = 2

    rows_by_image = read_rows(args.db, args.sensor)
    print(f"{'model':<52}{'factors':>8}{'live':>7}{'win':>7}{'farther':>8}")
    for model in args.models:
        rec = pycolmap.Reconstruction(model)
        z, modes, sigmas, weights, track = collect(rec, rows_by_image)
        if not len(z):
            continue
        alpha = args.alpha if args.alpha is not None else float(np.median(z / modes[:, 0]))
        sep = np.abs(np.log(modes[:, 1]) - np.log(modes[:, 0]))
        live = sep >= args.sep_min
        k = winners(z[live], modes[live], sigmas[live], weights[live], alpha)
        win = k != 0
        # is the winning second mode BEHIND the anchor? a systematic depth bias
        # would push wins to one side; genuine surface confusion should not.
        farther = (modes[live][win][:, 1] > modes[live][win][:, 0]).mean() if win.any() else np.nan
        name = str(model).split("/ba/")[-1]
        print(f"{name:<52}{len(z):>8}{live.mean():>7.1%}{win.mean():>7.1%}{farther:>8.0%}")

        if args.detail:
            print(f"    alpha {alpha:.4f}"
                  + ("" if args.alpha is not None else " (re-estimated, not BA's frozen value)"))
            print("    " + rates(win, "by depth   ", z[live]))
            print("    " + rates(win, "by sep     ", sep[live]))
            tracks = collections.defaultdict(list)
            for t, w in zip(track[live], win):
                tracks[t].append(w)
            multi = [v for v in tracks.values() if len(v) > 1]
            if multi:
                agree = np.mean([all(v) or not any(v) for v in multi])
                print(f"    tracks with >1 live observation: {len(multi)}, "
                      f"{agree:.0%} unanimous on the winner")


if __name__ == "__main__":
    main()
