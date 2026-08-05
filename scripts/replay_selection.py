"""Score candidate mode-selection rules against the mesh oracle, without re-running BA.

Evidence comes from the BASELINE (depth-off) reconstruction, so the replay is not
contaminated by any depth prior -- rays, alignment and converged depths all come
from that one model.

`net` is the headline: right switches minus wrong ones. Never switching (i.e. the
unimodal arm) scores 0, so a rule is only worth shipping if net > 0.

Only `margin` is explorable on a sensor ingested with uniform_prior (weights are
0.5/0.5 and sigmas are floored, so those columns cannot vary). To test informative
weights or unfloored sigmas, re-ingest under a NEW sensor name with different
method_params -- that is an attach run (~30s/scene), not a BA run -- and re-run this.

  python scripts/replay_selection.py --db database.db --sensor amb3r_gmm \
      --baseline ba/baseline/0 --gt $D/dslr/colmap --mesh $D/scans/mesh_aligned_0.05.ply
"""

import argparse
from pathlib import Path

import numpy as np

import pycolmap

from depthba.backends.residuals import whitened_residuals
from depthba.eval import mesh_oracle
from oracle_modes import read_rows


def score(z, modes, sigmas, weights, alpha, use_sigmas: bool, use_weights: bool):
    """Per-mode max-mixture score, with either non-residual term switchable off."""
    s = whitened_residuals(z[:, None], modes, sigmas, alpha) ** 2
    if use_sigmas:
        s = s + 2.0 * np.log(sigmas)
    if use_weights:
        s = s - 2.0 * np.log(np.maximum(weights, 1e-20))
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True,
                    help="depth-off reconstruction: uncontaminated evidence")
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--sensor", default="amb3r_gmm")
    ap.add_argument("--sep_min", type=float, default=0.1)
    ap.add_argument("--margins", type=float, nargs="+",
                    default=[0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    args = ap.parse_args()
    pycolmap.logging.minloglevel = 2

    rec = pycolmap.Reconstruction(args.baseline)
    gt = pycolmap.Reconstruction(args.gt)
    rows_by_image = read_rows(args.db, args.sensor)
    scale, R, t, n_common = mesh_oracle.align_to(rec, gt)
    origins, dirs, norms, records = mesh_oracle.live_rays(
        rec, rows_by_image, args.sep_min, (scale, R, t))
    if not len(records):
        raise SystemExit("no live keypoints")
    depth = mesh_oracle.cast(args.mesh, origins, dirs, norms)

    modes = np.array([r[2].modes for r in records], dtype=float)
    sigmas = np.array([r[2].sigmas for r in records], dtype=float)
    weights = np.array([r[2].weights for r in records], dtype=float)
    z = np.array([r[3] if r[3] is not None else np.nan for r in records])

    usable = np.isfinite(depth) & np.isfinite(z)
    if not usable.any():
        raise SystemExit("no keypoints both triangulated in baseline and hit by a ray")
    modes, sigmas, weights = modes[usable], sigmas[usable], weights[usable]
    depth, z = depth[usable], z[usable]

    alpha_mesh = float(np.median(depth / modes[:, 0]))
    alpha_z = float(np.median(z / modes[:, 0]))
    truth = (np.abs(np.log(depth) - np.log(alpha_mesh * modes[:, 1]))
             < np.abs(np.log(depth) - np.log(alpha_mesh * modes[:, 0])))
    print(f"{n_common} common images, Sim(3) scale {scale:.6f}; "
          f"{usable.sum()} keypoints usable of {len(records)} live")
    print(f"alpha: mesh {alpha_mesh:.4f}, baseline evidence {alpha_z:.4f}")
    print(f"mode 1 is correct for {truth.mean():.1%} (the ceiling: net {truth.sum():+d})\n")

    print(f"{'rule':<34}{'switch':>8}{'right':>7}{'wrong':>7}"
          f"{'prec':>7}{'rec':>7}{'net':>7}")
    uniform_w = np.allclose(weights, weights[0, 0])
    flat_s = np.allclose(sigmas, sigmas.min())
    for use_sigmas, use_weights in ((False, False), (True, False), (False, True), (True, True)):
        if (use_weights and uniform_w) or (use_sigmas and flat_s):
            continue  # that term is constant in this sensor; identical to leaving it out
        for margin in args.margins:
            s = score(z, modes, sigmas, weights, alpha_z, use_sigmas, use_weights)
            pick1 = (s[:, 1] + margin) < s[:, 0]
            tp = int((pick1 & truth).sum())
            fp = int((pick1 & ~truth).sum())
            fn = int((~pick1 & truth).sum())
            name = (f"margin={margin:g}"
                    f"{' +sigma' if use_sigmas else ''}{' +weights' if use_weights else ''}")
            print(f"{name:<34}{tp + fp:>8}{tp:>7}{fp:>7}"
                  f"{tp / max(tp + fp, 1):>7.1%}{tp / max(tp + fn, 1):>7.1%}{tp - fp:>+7d}")
    if uniform_w:
        print("\nweights are uniform in this sensor -- the +weights rules are unavailable; "
              "re-ingest with uniform_prior: false under a new sensor name to test them")
    if flat_s:
        print("sigmas are all at the floor -- the +sigma rules are unavailable; "
              "re-ingest with a lower sigma_log_min under a new sensor name to test them")


if __name__ == "__main__":
    main()
