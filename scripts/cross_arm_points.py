"""Where do two arms' 3D points differ, and is the difference concentrated on mode-1 keypoints?

Tracks are matched through shared (image_id, point2D_idx) observations, then the two
reconstructions are Sim(3)-aligned on their common camera centres, so displacements
are in the reference arm's units and gauge-free.

  python scripts/cross_arm_points.py --db database.db \
      --ref ba/amb3r_unimodal_all_.../0 --arm ba/amb3r_gmm_all_.../0
"""

import argparse
import collections
import sqlite3
from pathlib import Path

import numpy as np

import pycolmap

from depthba.backends.residuals import maxmix_scores, whitened_residuals
from depthba.depth import schema
from depthba.eval.alignment import umeyama_sim3


def centers(rec):
    """World-frame camera centres keyed by image name."""
    out = {}
    for image in rec.images.values():
        cfw = image.cam_from_world
        if callable(cfw):
            cfw = cfw()
        R, t = cfw.rotation.matrix(), np.asarray(cfw.translation, dtype=float)
        out[Path(image.name).stem] = -R.T @ t
    return out


def observations(rec):
    """{(image_name, point2D_idx): point3D_id} for triangulated observations."""
    out = {}
    for image in rec.images.values():
        name = Path(image.name).stem
        for idx, p2d in enumerate(image.points2D):
            if p2d.has_point3D():
                out[(name, idx)] = p2d.point3D_id
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True, help="reference arm (e.g. unimodal)")
    ap.add_argument("--arm", type=Path, required=True, help="arm under test (e.g. gmm)")
    ap.add_argument("--sensor", default="amb3r_gmm")
    ap.add_argument("--sep_min", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()
    pycolmap.logging.minloglevel = 2

    ref, arm = pycolmap.Reconstruction(args.ref), pycolmap.Reconstruction(args.arm)
    c_ref, c_arm = centers(ref), centers(arm)
    common = sorted(c_ref.keys() & c_arm.keys())
    if len(common) < 3:
        raise SystemExit(f"only {len(common)} common images -- cannot align")
    scale, R, t = umeyama_sim3(np.array([c_arm[n] for n in common]),
                               np.array([c_ref[n] for n in common]))
    print(f"{len(common)} common images, arm->ref Sim(3) scale {scale:.6f}")

    # per-observation winner for the arm under test
    conn = sqlite3.connect(args.db)
    meta = schema.read_meta(conn, args.sensor)
    ids = [i for (i,) in conn.execute(
        "SELECT DISTINCT image_id FROM depthba_keypoint_depths WHERE sensor=?", (args.sensor,))]
    rows_by_image = {i: schema.read_depths_for_image(conn, i, args.sensor, meta.num_modes)
                     for i in ids}
    conn.close()

    obs_ref, obs_arm = observations(ref), observations(arm)
    shared = obs_ref.keys() & obs_arm.keys()

    # match tracks by their most common counterpart across shared observations
    votes = collections.defaultdict(collections.Counter)
    live_wins = collections.defaultdict(list)
    alpha_num = []
    for image in arm.images.values():
        name = Path(image.name).stem
        rows = rows_by_image.get(image.image_id)
        if not rows:
            continue
        cfw = image.cam_from_world
        if callable(cfw):
            cfw = cfw()
        for idx, p2d in enumerate(image.points2D):
            if (name, idx) not in shared:
                continue
            row = rows.get(idx)
            if row is None or row.is_sky:
                continue
            votes[obs_arm[(name, idx)]][obs_ref[(name, idx)]] += 1
            depth = (cfw * arm.points3D[p2d.point3D_id].xyz)[2]
            if depth <= 0:
                continue
            alpha_num.append(depth / row.modes[0])
            sep = abs(np.log(row.modes[1]) - np.log(row.modes[0]))
            if sep >= args.sep_min:
                live_wins[obs_arm[(name, idx)]].append((depth, row))

    alpha = args.alpha if args.alpha is not None else float(np.median(alpha_num))
    print(f"alpha {alpha:.4f}"
          + ("" if args.alpha is not None else " (re-estimated, not BA's frozen value)"))

    disp, is_mode1 = [], []
    for pid_arm, counter in votes.items():
        pid_ref = counter.most_common(1)[0][0]
        p = arm.points3D[pid_arm].xyz
        d = np.linalg.norm((scale * (R @ p) + t) - ref.points3D[pid_ref].xyz)
        wins = []
        for depth, row in live_wins.get(pid_arm, []):
            r = whitened_residuals(depth, row.modes, row.sigmas, alpha)
            wins.append(int(np.argmin(maxmix_scores(r, row.sigmas,
                                                    np.maximum(row.weights, 1e-20)))) != 0)
        disp.append(d)
        is_mode1.append(bool(wins) and sum(wins) * 2 >= len(wins))
    disp, is_mode1 = np.array(disp), np.array(is_mode1)

    print(f"\n{len(disp)} matched tracks, {is_mode1.sum()} mode-1 majority")
    for label, mask in (("mode-0 tracks", ~is_mode1), ("mode-1 tracks", is_mode1)):
        if not mask.any():
            continue
        d = disp[mask] * 1000
        print(f"  {label:<14} n={mask.sum():>6}  "
              f"med {np.median(d):>8.2f}  p90 {np.percentile(d, 90):>9.2f}  "
              f"max {d.max():>10.2f}  [mm]")


if __name__ == "__main__":
    main()
