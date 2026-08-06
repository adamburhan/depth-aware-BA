"""Pool the ba sweep into one row per condition, paired against each scene's own baseline.

Why paired: an arm that holds a scene together is scored on the hard images the
baseline shed into a second sub-model, so unpaired per-image error penalises
completeness. Deltas are computed per image on the intersection, then per scene,
then aggregated.

  dbase   median Sim(3) distance to that scene's baseline, mm. Engagement check
          -- the full-mapper repeat floor is ~1mm, so anything near that is a no-op.
  dpos50  median over scenes of the per-scene median paired position delta, mm.
          Negative = better than baseline.
  dworst  worst scene's paired delta, mm. The do-no-harm claim lives here.
  auc     mean pose AUC (position, up to --auc_max) over ALL GT images, so
          unregistered images count as failures. Higher is better.
  dauc    mean per-scene AUC delta vs baseline. Positive = better.
  frag    scenes whose reconstruction split; sizes listed under the table.
  fail    scenes where Sim(3) alignment failed outright (silently dropped
          otherwise, which flatters exactly the worst arms).

  python scripts/summarize_sweep.py \
      --root $SCRATCH/experiments/depth-aware-ba/scannetpp_amb3r \
      --gt_root $SCRATCH/datasets/scannetpp/data
"""

import argparse
import collections
from pathlib import Path

import numpy as np

import pycolmap


def load(model_dir: Path):
    if not any((model_dir / f"cameras.{e}").exists() for e in ("bin", "txt")):
        return None
    rec = pycolmap.Reconstruction(model_dir)
    for image in rec.images.values():
        image.name = Path(image.name).stem  # GT names are .JPG, ours .png
    return rec


def best_submodel(variant_dir: Path):
    """Largest sub-model, with every sub-model's size. COLMAP does not
    guarantee that sub-model 0 is the biggest."""
    recs = [
        load(p) for p in sorted(variant_dir.iterdir())
        if p.is_dir() and p.name.isdigit()
    ]
    recs = [r for r in recs if r is not None]
    if not recs:
        return None, []
    sizes = [r.num_reg_images() for r in recs]
    return recs[int(np.argmax(sizes))], sizes


def errors(rec, ref, max_err: float):
    """Per-image {name: (position m, rotation deg)} after Sim(3) alignment."""
    result = pycolmap.compare_reconstructions(
        rec, ref, alignment_error="proj_center", max_proj_center_error=max_err
    )
    if result is None:
        return None
    errs = result.get("errors", result.get("image_alignment_error"))
    return {e.image_name: (e.proj_center_error, e.rotation_error_deg) for e in errs}


def auc(err, n_total: int, max_err: float, steps: int = 20) -> float:
    """Mean fraction of ALL images under a sweep of thresholds; images that
    never registered are in the denominator, so completeness counts."""
    if not n_total:
        return np.nan
    pos = np.array([v[0] for v in err.values()])
    return float(np.mean([
        (pos <= t).sum() / n_total for t in np.linspace(0, max_err, steps + 1)[1:]
    ]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--gt_root", type=Path, default=None)
    ap.add_argument("--gt_subdir", default="dslr/colmap",
                    help="under <gt_root>/<sequence>; eth3d ships its GT poses "
                         "in dslr_calibration_undistorted")
    ap.add_argument("--sequences", nargs="+", default=None)
    ap.add_argument("--max_proj_center_error", type=float, default=0.2)
    ap.add_argument("--auc_max", type=float, default=0.05, help="AUC threshold, m")
    args = ap.parse_args()
    pycolmap.logging.minloglevel = 2  # glog: alignment output is very noisy

    sequences = args.sequences or sorted(
        p.name for p in args.root.iterdir() if (p / "dslr" / "ba").is_dir()
    )

    cells = collections.defaultdict(
        lambda: {"dbase": [], "dpos": [], "auc": [], "dauc": [], "n": 0,
                 "frag": [], "fail": 0}
    )
    for seq in sequences:
        ba = args.root / seq / "dslr" / "ba"
        gt = load(args.gt_root / seq / args.gt_subdir) if args.gt_root else None
        n_gt = len(gt.images) if gt is not None else 0
        # baseline may carry an exp_id suffix (baseline_0)
        base_dirs = sorted(p for p in ba.iterdir()
                           if p.is_dir() and p.name.startswith("baseline"))
        baseline, _ = best_submodel(base_dirs[0]) if base_dirs else (None, [])
        # errors() aligns its FIRST argument in place, and the alignment's inlier
        # threshold is absolute -- so a reconstruction that has already been
        # aligned once lands somewhere else the second time (86mm on facade).
        # Every comparison therefore gets its own load.
        base_err = errors(best_submodel(base_dirs[0])[0], gt, args.max_proj_center_error) \
            if (baseline is not None and gt is not None) else None
        base_auc = auc(base_err, n_gt, args.auc_max) if base_err else None

        for variant in sorted(p.name for p in ba.iterdir() if p.is_dir()):
            rec, sizes = best_submodel(ba / variant)
            if rec is None:
                continue
            cell = cells[variant]
            cell["n"] += 1
            if len(sizes) > 1:
                cell["frag"].append(f"    {variant:<50}{seq:<14}{sizes}")
            if baseline is not None:
                d = errors(best_submodel(ba / variant)[0], baseline,
                           args.max_proj_center_error)
                if d is not None:
                    cell["dbase"].append(np.median([v[0] for v in d.values()]))
            if gt is None:
                continue
            err = errors(rec, gt, args.max_proj_center_error)  # rec still pristine
            if err is None:
                cell["fail"] += 1
                continue
            cell["auc"].append(auc(err, n_gt, args.auc_max))
            if base_err is not None:
                common = err.keys() & base_err.keys()
                if common:
                    cell["dpos"].append(np.median(
                        [err[k][0] - base_err[k][0] for k in common]
                    ))
                    cell["dauc"].append(cell["auc"][-1] - base_auc)

    def stat(values, fn, scale=1.0):
        return fn(values) * scale if values else np.nan

    print(f"{'condition':<50}{'n':>3}{'frg':>4}{'fal':>4}{'dbase':>8}"
          f"{'dpos50':>8}{'dworst':>8}{'auc':>7}{'dauc':>7}   [mm]")
    for name in sorted(cells):
        c = cells[name]
        print(f"{name:<50}{c['n']:>3}{len(c['frag']):>4}{c['fail']:>4}"
              f"{stat(c['dbase'], np.median, 1000):>8.2f}"
              f"{stat(c['dpos'], np.median, 1000):>8.2f}"
              f"{stat(c['dpos'], np.max, 1000):>8.2f}"
              f"{stat(c['auc'], np.mean):>7.3f}"
              f"{stat(c['dauc'], np.mean):>7.3f}")

    frag = [line for c in cells.values() for line in c["frag"]]
    if frag:
        print(f"\nsub-model sizes where the reconstruction split ({len(frag)}):")
        for line in sorted(frag):
            print(line)


if __name__ == "__main__":
    main()
