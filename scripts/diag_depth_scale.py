"""Diagnose depth-sensor scale consistency against a known-good reconstruction.

For every registered image: alpha_i = median(z_cam / mu) over its triangulated
keypoints (mu = bundle depth at the keypoint pixel, sky/invalid excluded).
If the sensor is scale-consistent, alpha_i is constant across images up to
triangulation noise; the spread of alpha_i is the scale-inconsistency the
frozen-per-image-alpha config would bake into depth factors. The within-image
residual std of log z - log(alpha_i * mu) is the empirically honest sigma_log
for factor whitening.

  python scripts/diag_depth_scale.py \
      --model $out/sfm_baseline/0 --dump_dir $amb/depth_bundles \
      [--out plots/meetingroom_alpha.png]
"""

import argparse
from pathlib import Path

import numpy as np
import pycolmap

from depthba.depth.source import DepthSource


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="reconstruction dir, e.g. .../sfm_baseline/0")
    ap.add_argument("--dump_dir", required=True, type=Path)
    ap.add_argument("--out", default=None, help="optional plot path")
    args = ap.parse_args()

    rec = pycolmap.Reconstruction(args.model)
    source = DepthSource(args.dump_dir)

    names, alphas, within_stds, n_obs = [], [], [], []
    all_logratio = []  # log(z/mu) per observation, for the global view
    for image in sorted(rec.images.values(), key=lambda im: im.name):
        cam = rec.cameras[image.camera_id]
        cfw = image.cam_from_world
        if callable(cfw):
            cfw = cfw()
        R = np.asarray(cfw.rotation.matrix())
        t = np.asarray(cfw.translation)

        xy, xyz = [], []
        for p in image.points2D:
            if p.has_point3D():
                xy.append(p.xy)
                xyz.append(rec.points3D[p.point3D_id].xyz)
        if len(xy) < 20:
            continue
        xy = np.asarray(xy)
        z = (np.asarray(xyz) @ R.T + t)[:, 2]

        bundle = source.load(image.name, (cam.height, cam.width))
        u = np.clip(np.floor(xy[:, 0]).astype(int), 0, cam.width - 1)
        v = np.clip(np.floor(xy[:, 1]).astype(int), 0, cam.height - 1)
        mu = bundle.estimated_depth[v, u].astype(np.float64)

        ok = np.isfinite(mu) & (mu > 0) & (z > 0)
        if bundle.sky_mask is not None:
            ok &= ~bundle.sky_mask[v, u]
        if ok.sum() < 20:
            continue
        logratio = np.log(z[ok]) - np.log(mu[ok])

        a = float(np.exp(np.median(logratio)))   # robust per-image scale
        names.append(image.name)
        alphas.append(a)
        within_stds.append(float(np.std(logratio - np.log(a))))
        n_obs.append(int(ok.sum()))
        all_logratio.append(logratio)

    alphas = np.asarray(alphas)
    within_stds = np.asarray(within_stds)
    log_alpha = np.log(alphas)
    print(f"{len(alphas)} images, {sum(n_obs)} observations")
    print(f"alpha (map-units per meter): median {np.median(alphas):.4f}")
    # across-image scale spread, in relative (log) units -- THE consistency test
    spread = np.std(log_alpha)
    p5, p95 = np.percentile(log_alpha, [5, 95])
    print(f"across-image scale spread: std {100*spread:.2f}%  "
          f"p5..p95 {100*(p5-np.median(log_alpha)):+.2f}%..{100*(p95-np.median(log_alpha)):+.2f}%  "
          f"max dev {100*np.max(np.abs(log_alpha-np.median(log_alpha))):.2f}%")
    # within-image residual spread -- the honest per-factor sigma_log
    print(f"within-image sigma_log: median {np.median(within_stds):.3f}  "
          f"p10 {np.percentile(within_stds,10):.3f}  p90 {np.percentile(within_stds,90):.3f}")
    glob = np.concatenate(all_logratio) - np.median(log_alpha)
    print(f"single-global-alpha residual std: {np.std(glob):.3f} log-units "
          f"(robust: {1.4826*np.median(np.abs(glob-np.median(glob))):.3f})")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        idx = np.arange(len(alphas))
        rel = 100 * (log_alpha - np.median(log_alpha))
        axes[0].plot(idx, rel, ".", ms=3)
        axes[0].axhline(0, color="k", lw=0.5)
        axes[0].set_xlabel("image (sorted)"); axes[0].set_ylabel("alpha dev [%]")
        axes[0].set_title("per-image scale vs global")
        axes[1].hist(within_stds, bins=40)
        axes[1].set_xlabel("within-image sigma_log"); axes[1].set_title("honest per-factor sigma")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
