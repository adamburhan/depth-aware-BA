"""Gate A': prove the pycolmap+pyceres seam and the factor conventions on a
converged reconstruction, before any depth-aware experiments.

Stages (each prints PASS/FAIL evidence; later stages assume earlier ones):
  1. convention  — evaluate LogDepthError at a real observation's state for
     all (quaternion order) x (alpha/beta order) combos; exactly one should
     give residual ~ 0. Pins the pose->factor convention empirically.
  2. reproj      — assemble reprojection-only problem from the converged sfm
     via pycolmap.cost_functions in a pyceres.Problem. Initial cost should be
     ~ the final cost of the pipeline's last global BA; solving should move
     poses by numerical noise only.
  3. depth (--depth) — same problem + LogDepthError factors from the ingested
     sensor rows, per-image alpha/beta blocks with NormalPriors, timed.

Run on Linux (macOS cannot co-import pycolmap+pyceres):
  python scripts/gate_a_prime.py --sfm $OUT/sfm/0 --db $OUT/database.db [--depth]
"""

import argparse
import sqlite3
import time

import numpy as np

import pycolmap
import pyceres

from depthba.depth import schema


def _get(obj, name):
    """pycolmap 4.1 exposes some accessors as methods, others as properties."""
    attr = getattr(obj, name)
    return attr() if callable(attr) else attr


def solve(problem, max_iters):
    options = pyceres.SolverOptions()
    options.max_num_iterations = max_iters
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
    options.minimizer_progress_to_stdout = False
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    return summary


def stage_convention(rec, sigma=0.1):
    print("=== stage 1: convention ===")
    image = next(im for im in rec.images.values() if _get(im, "has_pose"))
    p2d = next(p for p in image.points2D if p.has_point3D())
    X = rec.points3D[p2d.point3D_id].xyz.copy()
    cfw = _get(image, "cam_from_world")
    z = (cfw * X)[2]
    q = np.asarray(cfw.rotation.quat, dtype=np.float64).copy()
    t = np.asarray(cfw.translation, dtype=np.float64).copy()
    print(f"observation: image {image.image_id}, z_cam = {z:.4f} m, quat(xyzw as stored) = {q}")

    # 7-param fused pose block (fork >= 2.7.2); quat order xyzw and alpha-
    # before-beta were already pinned on the 5-block wheel (exact 0 cost).
    combos = {
        "pose7=[quat(xyzw)|t]": np.r_[q, t],
        "pose7=[t|quat(xyzw)]": np.r_[t, q],
    }
    results = {}
    for name, pose7 in combos.items():
        problem = pyceres.Problem()
        cost = pyceres.factors.LogDepthError(float(z), sigma)
        blocks = [pose7.copy(), X.copy(), np.array([1.0]), np.array([0.0])]
        problem.add_residual_block(cost, None, blocks)
        summary = solve(problem, 0)
        results[name] = summary.initial_cost
        print(f"  {name}: initial_cost = {summary.initial_cost:.6e}")
    winner = min(results, key=results.get)
    ok = results[winner] < 1e-12
    print(f"{'PASS' if ok else 'FAIL'}: convention = {winner}")
    return winner if ok else None


def build_reproj_problem(rec, blocks=None, quiet=False):
    """Copy-in assembly: local arrays per block (no reliance on pycolmap views);
    returns (problem, blocks) — hold `blocks` alive for the problem's lifetime.

    Pass an existing `blocks` dict to build a problem over the SAME arrays —
    used to evaluate the reprojection objective alone at a later solution."""
    problem = pyceres.Problem()
    loss = None  # trivial, matching the pipeline's global BA
    blocks = blocks if blocks is not None else {}
    poses = blocks.setdefault("poses", {})
    points = blocks.setdefault("points", {})
    cams = blocks.setdefault("cams", {})
    for camera_id, camera in rec.cameras.items():
        if camera_id not in cams:
            cams[camera_id] = camera.params.astype(np.float64).copy()
    num_obs = 0
    for image_id, image in rec.images.items():
        if not _get(image, "has_pose"):
            continue
        if image_id not in poses:
            cfw = _get(image, "cam_from_world")
            poses[image_id] = np.r_[  # pose7 = [quat(xyzw) | t], Rigid3d layout
                np.asarray(cfw.rotation.quat, dtype=np.float64),
                np.asarray(cfw.translation, dtype=np.float64),
            ]
        camera = rec.cameras[image.camera_id]
        for p2d in image.points2D:
            if not p2d.has_point3D():
                continue
            pid = p2d.point3D_id
            if pid not in points:
                points[pid] = rec.points3D[pid].xyz.astype(np.float64).copy()
            cost = pycolmap.cost_functions.ReprojErrorCost(camera.model, p2d.xy)
            # NOTE: reprojection cost is point-first [3, 7, 4]; the depth
            # factors are pose-first [7, 3, 1, 1] — do not copy the ordering.
            problem.add_residual_block(
                cost, loss,
                [points[pid], poses[image_id], cams[image.camera_id]],
            )
            num_obs += 1

    manifold = pyceres.ProductManifold(
        pyceres.EigenQuaternionManifold(), pyceres.EuclideanManifold(3)
    )
    for image_id in poses:
        problem.set_manifold(poses[image_id], manifold)
    # gauge: two lowest registered image ids fully constant (over-constrained
    # relative to COLMAP's TWO_CAMS_FROM_WORLD, harmless at a converged state)
    for image_id in sorted(poses)[:2]:
        problem.set_parameter_block_constant(poses[image_id])
    for arr in cams.values():
        problem.set_parameter_block_constant(arr)

    if not quiet:
        print(f"assembled: {num_obs} reprojection residuals, "
              f"{len(poses)} poses, {len(points)} points")
    return problem, blocks


def stage_reproj(rec):
    print("=== stage 2: reprojection-only A' ===")
    problem, blocks = build_reproj_problem(rec)
    p0 = {i: p.copy() for i, p in blocks["poses"].items()}
    tic = time.time()
    summary = solve(problem, 20)
    toc = time.time() - tic
    dq = max(np.abs(blocks["poses"][i][:4] - p0[i][:4]).max() for i in p0)
    dt = max(np.abs(blocks["poses"][i][4:] - p0[i][4:]).max() for i in p0)
    print(f"initial cost: {summary.initial_cost:.6e}  (compare to the pipeline's "
          "final global BA cost)")
    print(f"final cost:   {summary.final_cost:.6e}  in {toc:.1f}s")
    print(f"max pose delta: quat {dq:.2e}, translation {dt:.2e}")
    rel = (summary.initial_cost - summary.final_cost) / summary.initial_cost
    print(f"{'PASS' if rel < 0.01 and dt < 1e-3 else 'CHECK'}: "
          f"relative cost drop {rel:.2%} (want ~0), poses ~immobile")
    return problem, blocks, summary.final_cost


def stage_depth(rec, problem, blocks, db_path, sensor, sigma_log,
                prior_sigma_alpha, prior_sigma_beta, reproj_baseline,
                save_affine=None):
    print(f"=== stage 3: + depth factors (sensor {sensor!r}) ===")
    conn = sqlite3.connect(db_path)
    meta = schema.read_meta(conn, sensor)
    alphas, betas = {}, {}
    num_depth, num_skipped = 0, 0
    for image_id, image in rec.images.items():
        if image_id not in blocks["poses"]:
            continue
        rows = schema.read_depths_for_image(conn, image_id, sensor, meta.num_modes)
        if not rows:
            continue
        alphas[image_id] = np.array([1.0])
        betas[image_id] = np.array([0.0])
        for idx, p2d in enumerate(image.points2D):
            row = rows.get(idx)
            if row is None or not p2d.has_point3D() or row.is_sky:
                num_skipped += row is not None
                continue
            cost = pyceres.factors.LogDepthError(row.estimated_depth, sigma_log)
            problem.add_residual_block(
                cost, None,
                [blocks["poses"][image_id], blocks["points"][p2d.point3D_id],
                 alphas[image_id], betas[image_id]],
            )
            num_depth += 1
        problem.add_residual_block(
            pyceres.factors.NormalPrior([1.0], [[prior_sigma_alpha**2]]),
            None, [alphas[image_id]])
        problem.add_residual_block(
            pyceres.factors.NormalPrior([0.0], [[prior_sigma_beta**2]]),
            None, [betas[image_id]])
    conn.close()
    blocks["alphas"], blocks["betas"] = alphas, betas
    print(f"added {num_depth} depth factors ({num_skipped} rows skipped: "
          "sky/untriangulated), 2x{} affine priors".format(len(alphas)))

    tic = time.time()
    summary = solve(problem, 30)
    toc = time.time() - tic
    print(f"initial cost: {summary.initial_cost:.6e}")
    print(f"final cost:   {summary.final_cost:.6e}  in {toc:.1f}s")

    # Split objective: reprojection blocks alone, evaluated at the depth
    # solution (same arrays, fresh problem, 0-iteration solve). "What did the
    # depth factors' fit cost in reprojection quality?"
    eval_problem, _ = build_reproj_problem(rec, blocks, quiet=True)
    reproj_at_sol = solve(eval_problem, 0).initial_cost
    print(f"reproj-only cost at depth solution: {reproj_at_sol:.6e} "
          f"(baseline {reproj_baseline:.6e}, "
          f"degradation {(reproj_at_sol - reproj_baseline) / reproj_baseline:+.2%})")
    print(f"depth(+priors) cost at solution:    "
          f"{summary.final_cost - reproj_at_sol:.6e}")

    a = np.array([alphas[i][0] for i in sorted(alphas)])
    b = np.array([betas[i][0] for i in sorted(betas)])
    print(f"alpha: mean {a.mean():.4f} std {a.std():.4f} | "
          f"beta: mean {b.mean():.4f} std {b.std():.4f}")
    print("per-image affine:")
    for i in sorted(alphas):
        print(f"  {i:3d} {rec.images[i].name:24s} "
              f"alpha={alphas[i][0]:.4f} beta={betas[i][0]:+.4f}")
    if save_affine:
        np.savez(save_affine,
                 image_ids=np.array(sorted(alphas)),
                 names=np.array([rec.images[i].name for i in sorted(alphas)]),
                 alpha=a, beta=b)
        print(f"affine params saved to {save_affine}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfm", required=True, help="converged reconstruction dir, e.g. $OUT/sfm/0")
    ap.add_argument("--db", required=True, help="COLMAP db with depthba tables")
    ap.add_argument("--depth", action="store_true", help="run stage 3")
    ap.add_argument("--sensor", default="depthpro")
    ap.add_argument("--sigma_log", type=float, default=0.15)
    # Weak by default: gauge is pinned by poses; an informative linear-space
    # alpha prior fights the (arbitrary) SfM scale. Tighten deliberately, as
    # an experimental condition, not by habit.
    ap.add_argument("--prior_sigma_alpha", type=float, default=1.0)
    ap.add_argument("--prior_sigma_beta", type=float, default=1.0)
    ap.add_argument("--save_affine", default=None,
                    help="optional .npz path for per-image alpha/beta")
    args = ap.parse_args()

    rec = pycolmap.Reconstruction(args.sfm)
    print(rec.summary())

    if stage_convention(rec) is None:
        print("convention unresolved — stopping before assembly")
        return
    problem, blocks, reproj_baseline = stage_reproj(rec)
    if args.depth:
        stage_depth(rec, problem, blocks, args.db, args.sensor, args.sigma_log,
                    args.prior_sigma_alpha, args.prior_sigma_beta,
                    reproj_baseline, save_affine=args.save_affine)


if __name__ == "__main__":
    main()
