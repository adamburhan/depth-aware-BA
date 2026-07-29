"""Evaluate a reconstruction against a Tanks and Temples trajectory
(*_COLMAP_SfM.log, Choi et al. format: per frame a metadata line plus a 4x4
camera-to-world matrix over four lines; frame order == sorted image names).

Sim(3)-aligns estimated camera centers to the log trajectory (Umeyama) and
reports per-image position/rotation errors — same metrics as
evaluate_vs_gt.py. Position errors are in LOG units unless --trans is given
(the *_trans.txt maps the log frame into the metric laser-scan frame; only
its scale matters for error magnitudes).

  python scripts/evaluate_vs_tnt.py \
      --log $DATA/Meetingroom/Meetingroom_COLMAP_SfM.log \
      --images $DATA/Meetingroom/images \
      [--trans $DATA/Meetingroom/Meetingroom_trans.txt] \
      $OUT/meetingroom/sfm/0
"""

import argparse
from pathlib import Path

import numpy as np

import pycolmap

from depthba.eval.alignment import umeyama_sim3


def read_tnt_log(path: Path) -> list[np.ndarray]:
    lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
    if len(lines) % 5 != 0:
        raise ValueError(f"{path}: {len(lines)} lines, expected a multiple of 5")
    poses = []
    for i in range(0, len(lines), 5):
        mat = np.array(
            [[float(x) for x in lines[i + row].split()] for row in range(1, 5)]
        )
        poses.append(mat)  # camera-to-world
    return poses


def pose_auc(errors: np.ndarray, thresholds: list[float]) -> list[float]:
    """AUC of the error recall curve at each threshold (IMC/SuperGlue convention):
    (1/t) * integral_0^t recall(x) dx, where recall(x) = fraction of errors <= x.
    1.0 = every pose exact; 0.0 = every pose at/beyond the threshold."""
    errors = np.sort(np.asarray(errors))
    recall = np.arange(1, len(errors) + 1) / len(errors)
    errors = np.concatenate([[0.0], errors])
    recall = np.concatenate([[0.0], recall])
    aucs = []
    for t in thresholds:
        idx = np.searchsorted(errors, t, side="right")
        e = np.concatenate([errors[:idx], [t]])
        r = np.concatenate([recall[:idx], [recall[idx - 1] if idx > 0 else 0.0]])
        aucs.append(float(np.trapezoid(r, e) / t))
    return aucs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="*_COLMAP_SfM.log GT trajectory")
    ap.add_argument("--images", required=True, help="image dir (defines frame order)")
    ap.add_argument("--trans", default=None,
                    help="*_trans.txt: scales position errors into metric units")
    ap.add_argument("reconstruction", help="estimated model dir, e.g. .../sfm/0")
    args = ap.parse_args()

    gt_poses = read_tnt_log(Path(args.log))
    names = sorted(p.name for p in Path(args.images).iterdir() if p.is_file())
    if len(names) != len(gt_poses):
        raise ValueError(f"{len(names)} images but {len(gt_poses)} log poses")
    gt = dict(zip(names, gt_poses))

    metric_scale = 1.0
    if args.trans:
        rows = [
            [float(x) for x in l.split()]
            for l in Path(args.trans).read_text().splitlines() if l.strip()
        ]
        trans = np.array(rows)[:4, :4]
        metric_scale = float(np.cbrt(abs(np.linalg.det(trans[:3, :3]))))

    rec = pycolmap.Reconstruction(args.reconstruction)
    print(rec.summary())

    est_centers, gt_centers, est_R_cw, gt_R_cw, used = [], [], [], [], []
    est_R_by_name, gt_R_by_name = {}, {}
    est_t_by_name, gt_t_by_name = {}, {}
    for image in rec.images.values():
        if image.name not in gt:
            continue
        cfw = image.cam_from_world
        if callable(cfw):
            cfw = cfw()
        R_cw = np.asarray(cfw.rotation.matrix())
        t_cw = np.asarray(cfw.translation)
        est_centers.append(-R_cw.T @ t_cw)
        est_R_cw.append(R_cw)
        c2w = gt[image.name]
        gt_R_cw_i = c2w[:3, :3].T
        gt_centers.append(c2w[:3, 3])
        gt_R_cw.append(gt_R_cw_i)
        used.append(image.name)
        est_R_by_name[image.name] = R_cw
        gt_R_by_name[image.name] = gt_R_cw_i
        est_t_by_name[image.name] = t_cw
        gt_t_by_name[image.name] = -gt_R_cw_i @ c2w[:3, 3]

    est_centers = np.array(est_centers)
    gt_centers = np.array(gt_centers)
    print(f"\nimages compared: {len(used)} / {len(gt)} GT frames")

    # Relative pose between consecutive frames: gauge-free AND scale-free, so
    # it needs no alignment. Distinguishes local geometric fidelity from
    # global ATE effects. Tight RPE (~0.1-0.3deg) + large ATE => the ATE gap
    # is global/alignment/intrinsics-scale, not a locally bad reconstruction.
    #
    # Pose AUC follows MP-SfM (Pataki et al., CVPR'25) exactly: relative pose
    # error = max(rotation angular error, translation angular error) per
    # consecutive pair, translation compared by direction only (scale-free),
    # AUC at 1/5/20deg, reported as a percentage triplet.
    ordered = sorted(n for n in used)
    rpe, pose_err = [], []
    for a, b in zip(ordered[:-1], ordered[1:]):
        est_rel_R = est_R_by_name[b] @ est_R_by_name[a].T
        gt_rel_R = gt_R_by_name[b] @ gt_R_by_name[a].T
        cos_r = np.clip((np.trace(est_rel_R @ gt_rel_R.T) - 1.0) / 2.0, -1.0, 1.0)
        rot_err_deg = np.degrees(np.arccos(cos_r))
        rpe.append(rot_err_deg)

        est_t_ab = est_t_by_name[b] - est_R_by_name[b] @ est_R_by_name[a].T @ est_t_by_name[a]
        gt_t_ab = gt_t_by_name[b] - gt_R_by_name[b] @ gt_R_by_name[a].T @ gt_t_by_name[a]
        n_est, n_gt = np.linalg.norm(est_t_ab), np.linalg.norm(gt_t_ab)
        if n_est < 1e-9 or n_gt < 1e-9:
            continue  # near-zero baseline: direction undefined, skip this pair
        cos_t = np.clip(np.dot(est_t_ab / n_est, gt_t_ab / n_gt), -1.0, 1.0)
        trans_err_deg = np.degrees(np.arccos(cos_t))

        pose_err.append(max(rot_err_deg, trans_err_deg))
    rpe = np.array(rpe)
    pose_err = np.array(pose_err)
    print(f"RPE rotation [deg] (consecutive, alignment-free): "
          f"mean {rpe.mean():.4f}  median {np.median(rpe):.4f}  "
          f"p90 {np.percentile(rpe, 90):.4f}  max {rpe.max():.4f}")
    pose_thresholds = [1, 5, 20]
    aucs = pose_auc(pose_err, pose_thresholds)
    print(f"pose AUC @ {pose_thresholds}deg (relative, max(rot,trans) err, "
          f"n={len(pose_err)} pairs): " + "/".join(f"{a * 100:.1f}" for a in aucs))

    scale, R_align, t_align = umeyama_sim3(est_centers, gt_centers)
    print(f"sim3 est->log: scale {scale:.4f}"
          + (f" | log->metric scale {metric_scale:.4f}" if args.trans else ""))

    aligned = (scale * (R_align @ est_centers.T)).T + t_align
    pos_err = np.linalg.norm(aligned - gt_centers, axis=1) * metric_scale

    rot_err = np.empty(len(used))
    for i in range(len(used)):
        R_err = (est_R_cw[i] @ R_align.T) @ gt_R_cw[i].T
        cos = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        rot_err[i] = np.degrees(np.arccos(cos))

    unit = "m" if args.trans else "log-units"
    print(f"position error [{unit}]: mean {pos_err.mean():.4f}  "
          f"median {np.median(pos_err):.4f}  p90 {np.percentile(pos_err, 90):.4f}  "
          f"max {pos_err.max():.4f}")
    print(f"rotation error [deg]:  mean {rot_err.mean():.4f}  "
          f"median {np.median(rot_err):.4f}  p90 {np.percentile(rot_err, 90):.4f}  "
          f"max {rot_err.max():.4f}")

    worst = int(np.argmax(pos_err))
    print(f"worst image: {used[worst]} ({pos_err[worst]:.4f} {unit}, "
          f"{rot_err[worst]:.3f} deg)")

    # Drift profile: median error per 10%-of-trajectory bin, in frame order.
    # Smoothly growing / locally bulging bins = accumulated drift; flat noisy
    # bins = alignment or matching artifact.
    order = np.argsort(used)
    bins = np.array_split(order, 10)
    profile = " ".join(f"{np.median(pos_err[b]):.3f}" for b in bins)
    print(f"drift profile (median {unit} per trajectory tenth): {profile}")


if __name__ == "__main__":
    main()
