"""Pool the ba sweep into one row per condition.

Columns: n sequences, mean registered images, dbase = median Sim(3) distance
to that sequence's own baseline (did the knob engage at all — compare against
the ~0.4mm run-to-run floor), then position/rotation against the official
ScanNet++ COLMAP model. P90 matters as much as the median: the do-no-harm
claim lives in the tail, not the centre.

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
    if not (model_dir / "cameras.bin").exists() and not (model_dir / "cameras.txt").exists():
        return None
    rec = pycolmap.Reconstruction(model_dir)
    for image in rec.images.values():
        image.name = Path(image.name).stem  # GT names are .JPG, ours .png
    return rec


def errors(rec, ref):
    """Per-image position (m) and rotation (deg) after Sim(3) alignment."""
    result = pycolmap.compare_reconstructions(
        rec, ref, alignment_error="proj_center", max_proj_center_error=0.2
    )
    if result is None:
        return None, None
    errs = result.get("errors", result.get("image_alignment_error"))
    return (np.array([e.proj_center_error for e in errs]),
            np.array([e.rotation_error_deg for e in errs]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--gt_root", type=Path, default=None)
    ap.add_argument("--sequences", nargs="+", default=None)
    args = ap.parse_args()
    pycolmap.logging.minloglevel = 2  # glog: alignment output is very noisy

    sequences = args.sequences or sorted(
        p.name for p in args.root.iterdir() if (p / "dslr" / "ba").is_dir()
    )

    cells = collections.defaultdict(
        lambda: {"pos": [], "rot": [], "dbase": [], "reg": [], "dreg": [], "frag": []}
    )
    for seq in sequences:
        ba = args.root / seq / "dslr" / "ba"
        gt = load(args.gt_root / seq / "dslr" / "colmap") if args.gt_root else None
        # baseline may carry an exp_id suffix (baseline_0)
        base_dirs = sorted(
            p for p in ba.iterdir() if p.is_dir() and p.name.startswith("baseline")
        )
        baseline = load(base_dirs[0] / "0") if base_dirs else None
        for variant in sorted(p.name for p in ba.iterdir() if p.is_dir()):
            subs = sorted(
                p for p in (ba / variant).iterdir() if p.is_dir() and p.name.isdigit()
            )
            rec = load(subs[0]) if subs else None
            if rec is None:
                continue
            cell = cells[variant]
            cell["reg"].append(rec.num_reg_images())
            if len(subs) > 1:
                cell["frag"].append(f"{seq}({len(subs)})")
            if baseline is not None:
                # registrations relative to this sequence's own baseline —
                # absolute counts are not comparable across scenes
                cell["dreg"].append(rec.num_reg_images() - baseline.num_reg_images())
                pos, _ = errors(rec, baseline)
                if pos is not None:
                    cell["dbase"].append(np.median(pos))
            if gt is not None:
                pos, rot = errors(rec, gt)
                if pos is not None:
                    cell["pos"].append(pos)
                    cell["rot"].append(rot)

    print(f"{'condition':<50}{'n':>3}{'dreg':>6}{'dbase':>8}"
          f"{'pos50':>8}{'pos90':>8}{'rot50':>8}   [mm, deg]")
    for name in sorted(cells):
        cell = cells[name]
        pos = np.concatenate(cell["pos"]) if cell["pos"] else np.array([np.nan])
        rot = np.concatenate(cell["rot"]) if cell["rot"] else np.array([np.nan])
        dbase = np.median(cell["dbase"]) * 1000 if cell["dbase"] else np.nan
        dreg = np.mean(cell["dreg"]) if cell["dreg"] else np.nan
        print(f"{name:<50}{len(cell['reg']):>3}{dreg:>6.1f}{dbase:>8.2f}"
              f"{np.median(pos) * 1000:>8.2f}{np.percentile(pos, 90) * 1000:>8.2f}"
              f"{np.median(rot):>8.3f}"
              + (f"   FRAGMENTED {','.join(cell['frag'])}" if cell["frag"] else ""))


if __name__ == "__main__":
    main()
