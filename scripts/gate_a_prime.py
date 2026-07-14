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

    combos = {
        "quat=xyzw, blocks=[..,alpha,beta]": (q, [1.0], [0.0], False),
        "quat=wxyz, blocks=[..,alpha,beta]": (np.r_[q[3], q[:3]], [1.0], [0.0], False),
        "quat=xyzw, blocks=[..,beta,alpha]": (q, [0.0], [1.0], True),
        "quat=wxyz, blocks=[..,beta,alpha]": (np.r_[q[3], q[:3]], [0.0], [1.0], True),
    }
    results = {}
    for name, (qq, s4, s5, _swapped) in combos.items():
        problem = pyceres.Problem()
        cost = pyceres.factors.LogDepthError(float(z), sigma)
        blocks = [qq.copy(), t.copy(), X.copy(), np.array(s4), np.array(s5)]
        problem.add_residual_block(cost, None, blocks)
        summary = solve(problem, 0)
        results[name] = summary.initial_cost
        print(f"  {name}: initial_cost = {summary.initial_cost:.6e}")
    winner = min(results, key=results.get)
    ok = results[winner] < 1e-12
    print(f"{'PASS' if ok else 'FAIL'}: convention = {winner}")
    return winner if ok else None


def build_reproj_problem(rec):
    """Copy-in assembly: local arrays per block (no reliance on pycolmap views);
    returns (problem, blocks) — hold `blocks` alive for the problem's lifetime."""
    problem = pyceres.Problem()
    loss = None  # trivial, matching the pipeline's global BA
    quats, tvecs, points, cams = {}, {}, {}, {}
    for camera_id, camera in rec.cameras.items():
        cams[camera_id] = camera.params.astype(np.float64).copy()
    num_obs = 0
    for image_id, image in rec.images.items():
        if not _get(image, "has_pose"):
            continue
        cfw = _get(image, "cam_from_world")
        quats[image_id] = np.asarray(cfw.rotation.quat, dtype=np.float64).copy()
        tvecs[image_id] = np.asarray(cfw.translation, dtype=np.float64).copy()
        camera = rec.cameras[image.camera_id]
        for p2d in image.points2D:
            if not p2d.has_point3D():
                continue
            pid = p2d.point3D_id
            if pid not in points:
                points[pid] = rec.points3D[pid].xyz.astype(np.float64).copy()
            cost = pycolmap.cost_functions.ReprojErrorCost(camera.model, p2d.xy)
            problem.add_residual_block(
                cost, loss,
                [quats[image_id], tvecs[image_id], points[pid], cams[image.camera_id]],
            )
            num_obs += 1

    for image_id in quats:
        problem.set_manifold(quats[image_id], pyceres.EigenQuaternionManifold())
    # gauge: two lowest registered image ids fully constant (over-constrained
    # relative to COLMAP's TWO_CAMS_FROM_WORLD, harmless at a converged state)
    for image_id in sorted(quats)[:2]:
        problem.set_parameter_block_constant(quats[image_id])
        problem.set_parameter_block_constant(tvecs[image_id])
    for arr in cams.values():
        problem.set_parameter_block_constant(arr)

    print(f"assembled: {num_obs} reprojection residuals, "
          f"{len(quats)} poses, {len(points)} points")
    blocks = dict(quats=quats, tvecs=tvecs, points=points, cams=cams)
    return problem, blocks


def stage_reproj(rec):
    print("=== stage 2: reprojection-only A' ===")
    problem, blocks = build_reproj_problem(rec)
    q0 = {i: q.copy() for i, q in blocks["quats"].items()}
    t0 = {i: t.copy() for i, t in blocks["tvecs"].items()}
    tic = time.time()
    summary = solve(problem, 20)
    toc = time.time() - tic
    dq = max(np.abs(blocks["quats"][i] - q0[i]).max() for i in q0)
    dt = max(np.abs(blocks["tvecs"][i] - t0[i]).max() for i in t0)
    print(f"initial cost: {summary.initial_cost:.6e}  (compare to the pipeline's "
          "final global BA cost)")
    print(f"final cost:   {summary.final_cost:.6e}  in {toc:.1f}s")
    print(f"max pose delta: quat {dq:.2e}, translation {dt:.2e}")
    rel = (summary.initial_cost - summary.final_cost) / summary.initial_cost
    print(f"{'PASS' if rel < 0.01 and dt < 1e-3 else 'CHECK'}: "
          f"relative cost drop {rel:.2%} (want ~0), poses ~immobile")
    return problem, blocks


def stage_depth(rec, problem, blocks, db_path, sensor, sigma_log,
                prior_sigma_alpha, prior_sigma_beta):
    print(f"=== stage 3: + depth factors (sensor {sensor!r}) ===")
    conn = sqlite3.connect(db_path)
    meta = schema.read_meta(conn, sensor)
    alphas, betas = {}, {}
    num_depth, num_skipped = 0, 0
    for image_id, image in rec.images.items():
        if image_id not in blocks["quats"]:
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
                [blocks["quats"][image_id], blocks["tvecs"][image_id],
                 blocks["points"][p2d.point3D_id],
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
    a = np.array([alphas[i][0] for i in sorted(alphas)])
    b = np.array([betas[i][0] for i in sorted(betas)])
    print(f"alpha: mean {a.mean():.4f} std {a.std():.4f} | "
          f"beta: mean {b.mean():.4f} std {b.std():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfm", required=True, help="converged reconstruction dir, e.g. $OUT/sfm/0")
    ap.add_argument("--db", required=True, help="COLMAP db with depthba tables")
    ap.add_argument("--depth", action="store_true", help="run stage 3")
    ap.add_argument("--sensor", default="depthpro")
    ap.add_argument("--sigma_log", type=float, default=0.15)
    ap.add_argument("--prior_sigma_alpha", type=float, default=0.2)
    ap.add_argument("--prior_sigma_beta", type=float, default=0.2)
    args = ap.parse_args()

    rec = pycolmap.Reconstruction(args.sfm)
    print(rec.summary())

    if stage_convention(rec) is None:
        print("convention unresolved — stopping before assembly")
        return
    problem, blocks = stage_reproj(rec)
    if args.depth:
        stage_depth(rec, problem, blocks, args.db, args.sensor, args.sigma_log,
                    args.prior_sigma_alpha, args.prior_sigma_beta)


if __name__ == "__main__":
    main()
