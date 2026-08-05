"""Against the laser mesh, is the second mode ever the right answer?

Casts one ray per live keypoint (no rendering). Frames: the mesh and dslr/colmap
share ScanNet++'s scan-aligned frame; our SfM gauge is mapped into it by a Sim(3)
fitted on common camera centres. The sensor's global scale is fitted TO the mesh
(alpha = median(mesh / mode0)), so this asks "given the best possible global
scale, which mode is nearer the truth" and is independent of BA's frozen alpha.

  python scripts/oracle_modes.py --db database.db --model ba/<variant>/0 \
      --gt $DATA/<scene>/dslr/colmap --mesh $DATA/<scene>/scans/mesh_aligned_0.05.ply
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import trimesh

import pycolmap

from depthba.backends.residuals import maxmix_scores, whitened_residuals
from depthba.depth import schema
from depthba.eval.alignment import umeyama_sim3


def pose(image):
    cfw = image.cam_from_world
    if callable(cfw):
        cfw = cfw()
    R = cfw.rotation.matrix()
    return R, np.asarray(cfw.translation, dtype=float)


def centers(rec):
    out = {}
    for image in rec.images.values():
        R, t = pose(image)
        out[Path(image.name).stem] = -R.T @ t
    return out


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
    c_rec, c_gt = centers(rec), centers(gt)
    common = sorted(c_rec.keys() & c_gt.keys())
    if len(common) < 3:
        raise SystemExit(f"only {len(common)} common images with GT -- cannot align")
    scale, Rs, ts = umeyama_sim3(np.array([c_rec[n] for n in common]),
                                 np.array([c_gt[n] for n in common]))
    print(f"{len(common)} common images, SfM->scan Sim(3) scale {scale:.6f}")

    conn = sqlite3.connect(args.db)
    meta = schema.read_meta(conn, args.sensor)
    ids = [i for (i,) in conn.execute(
        "SELECT DISTINCT image_id FROM depthba_keypoint_depths WHERE sensor=?", (args.sensor,))]
    rows_by_image = {i: schema.read_depths_for_image(conn, i, args.sensor, meta.num_modes)
                     for i in ids}
    conn.close()

    origins, dirs, norms, keep = [], [], [], []
    for image in rec.images.values():
        rows = rows_by_image.get(image.image_id)
        if not rows:
            continue
        camera = rec.cameras[image.camera_id]
        if camera.model.name != "PINHOLE":
            raise SystemExit(f"expected PINHOLE cameras, got {camera.model.name}")
        fx, fy, cx, cy = camera.params
        R, t = pose(image)
        center = Rs @ (-R.T @ t) * scale + ts
        for idx, p2d in enumerate(image.points2D):
            row = rows.get(idx)
            if row is None or row.is_sky:
                continue
            if abs(np.log(row.modes[1]) - np.log(row.modes[0])) < args.sep_min:
                continue
            u, v = p2d.xy
            d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
            d_world = Rs @ (R.T @ d_cam)
            origins.append(center)
            norms.append(np.linalg.norm(d_cam))
            dirs.append(d_world / np.linalg.norm(d_world))
            keep.append((row, image.image_id, idx, p2d.has_point3D(),
                         None if not p2d.has_point3D()
                         else (R @ rec.points3D[p2d.point3D_id].xyz + t)[2]))
    if not keep:
        raise SystemExit("no live keypoints")
    origins, dirs, norms = np.array(origins), np.array(dirs), np.array(norms)
    print(f"{len(keep)} live keypoints; casting rays "
          f"(embreex {'in use' if trimesh.ray.has_embree else 'NOT installed - may be slow'})")

    mesh = trimesh.load(args.mesh, process=False, force="mesh")
    loc, idx_ray, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    depth = np.full(len(keep), np.nan)
    dist = np.linalg.norm(loc - origins[idx_ray], axis=1)
    for r, d in zip(idx_ray, dist):  # keep the nearest hit per ray
        if np.isnan(depth[r]) or d < depth[r]:
            depth[r] = d
    depth = depth / norms  # along-ray distance -> camera-frame z

    hit = np.isfinite(depth)
    m0 = np.array([k[0].modes[0] for k in keep])
    m1 = np.array([k[0].modes[1] for k in keep])
    alpha = float(np.median(depth[hit] / m0[hit]))
    e0 = np.abs(np.log(depth) - np.log(alpha * m0))
    e1 = np.abs(np.log(depth) - np.log(alpha * m1))
    better1 = (e1 < e0) & hit
    wrong = (e0 > args.anchor_tol) & hit

    print(f"mesh hit on {hit.sum()}/{len(keep)}; alpha fitted to mesh = {alpha:.4f}")
    print(f"\nmode 1 closer to mesh than mode 0: {better1[hit].mean():.1%}")
    print(f"  anchor wrong (|log err| > {args.anchor_tol}): {wrong[hit].mean():.1%} of live")
    if wrong.any():
        print(f"    of those, mode 1 is closer: {better1[wrong].mean():.1%}  <-- the population")
    ok = hit & ~wrong
    if ok.any():
        print(f"  anchor fine: mode 1 closer anyway {better1[ok].mean():.1%} "
              "(should be low; high = the EM is fitting noise)")

    tri = np.array([k[3] for k in keep]) & hit
    if tri.any():
        z = np.array([k[4] if k[4] is not None else np.nan for k in keep])
        ba_win = np.zeros(len(keep), bool)
        for i in np.flatnonzero(tri):
            row = keep[i][0]
            r = whitened_residuals(z[i], row.modes, row.sigmas, np.median(z[tri] / m0[tri]))
            ba_win[i] = int(np.argmin(maxmix_scores(
                r, row.sigmas, np.maximum(row.weights, 1e-20)))) != 0
        print(f"\ncross-tab on {tri.sum()} triangulated live keypoints "
              "(BA's choice vs the mesh's):")
        for b in (True, False):
            sel = tri & (ba_win == b)
            if sel.any():
                print(f"  BA chose mode {'1' if b else '0'}: {sel.sum():>6}  "
                      f"mesh agrees {(better1[sel] == b).mean():.1%}")


if __name__ == "__main__":
    main()
