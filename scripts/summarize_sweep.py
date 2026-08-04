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
        lambda: {"pos": [], "rot": [], "dbase": [], "reg": [], "frag": 0}
    )
    for seq in sequences:
        ba = args.root / seq / "dslr" / "ba"
        gt = load(args.gt_root / seq / "dslr" / "colmap") if args.gt_root else None
        baseline = load(ba / "baseline" / "0")
        for variant in sorted(p.name for p in ba.iterdir() if p.is_dir()):
            subs = sorted(
                p for p in (ba / variant).iterdir() if p.is_dir() and p.name.isdigit()
            )
            rec = load(subs[0]) if subs else None
            if rec is None:
                continue
            cell = cells[variant]
            cell["reg"].append(rec.num_reg_images())
            cell["frag"] += len(subs) > 1
            if baseline is not None:
                pos, _ = errors(rec, baseline)
                if pos is not None:
                    cell["dbase"].append(np.median(pos))
            if gt is not None:
                pos, rot = errors(rec, gt)
                if pos is not None:
                    cell["pos"].append(pos)
                    cell["rot"].append(rot)

    print(f"{'condition':<50}{'n':>3}{'reg':>6}{'dbase':>8}"
          f"{'pos50':>8}{'pos90':>8}{'rot50':>8}   [mm, deg]")
    for name in sorted(cells):
        cell = cells[name]
        pos = np.concatenate(cell["pos"]) if cell["pos"] else np.array([np.nan])
        rot = np.concatenate(cell["rot"]) if cell["rot"] else np.array([np.nan])
        dbase = np.median(cell["dbase"]) * 1000 if cell["dbase"] else np.nan
        print(f"{name:<50}{len(cell['reg']):>3}{np.mean(cell['reg']):>6.0f}{dbase:>8.2f}"
              f"{np.median(pos) * 1000:>8.2f}{np.percentile(pos, 90) * 1000:>8.2f}"
              f"{np.median(rot):>8.3f}"
              + ("   FRAGMENTED" if cell["frag"] else ""))


if __name__ == "__main__":
    main()
