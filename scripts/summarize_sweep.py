"""Sweep-level sanity table for the ScanNet++ depth-BA arms.

Three checks per (sequence, arm), cheapest first:

  models/reg  did the reconstruction stay in one piece, and did the depth
              factors cost us registrations? A depth arm registering far
              fewer images than baseline is the first red flag.
  vs_base     Sim(3) distance from the arm to its OWN baseline. This is the
              "did the feature engage at all" check: ~0 here means the depth
              factors were a silent no-op (config loaded, rows read, but
              nothing ever entered the ceres problem) and every vs-GT number
              below is measuring nothing.
  vs_gt       pose error against the official ScanNet++ COLMAP model, plus
              the Sim(3) scale. The scale is only interpretable if the GT is
              metric, so the camera-center extent is printed per sequence —
              a room should read a handful of metres.

  python scripts/summarize_sweep.py \
      --root $SCRATCH/experiments/depth-aware-ba/scannetpp_amb3r \
      --gt_root $SCRATCH/datasets/scannetpp/data
"""

import argparse
from pathlib import Path

import numpy as np

import pycolmap

ARMS = [
    "baseline",
    "gmm",
    "unimodal",
    "gmm_all",
    "unimodal_all",
    "gmm_local",
    "unimodal_local",
]


def normalize_image_names(rec: pycolmap.Reconstruction) -> None:
    """Match by stem: our dumps are DSC_x.png, ScanNet++ GT is DSC_x.JPG."""
    for image in rec.images.values():
        image.name = Path(image.name).stem


def load(model_dir: Path) -> pycolmap.Reconstruction | None:
    if not (model_dir / "cameras.bin").exists() and not (
        model_dir / "cameras.txt"
    ).exists():
        return None
    rec = pycolmap.Reconstruction(model_dir)
    normalize_image_names(rec)
    return rec


def submodels(arm_dir: Path) -> list[Path]:
    """Sub-model dirs written by the pipeline: 0, 1, ... — more than one means
    the reconstruction fragmented."""
    if not arm_dir.is_dir():
        return []
    return sorted((p for p in arm_dir.iterdir() if p.is_dir() and p.name.isdigit()),
                  key=lambda p: int(p.name))


def compare(rec, ref, max_proj_center_error: float) -> dict | None:
    """Sim(3)-align rec to ref and reduce to medians. Returns None if the
    alignment itself failed (too few inlier correspondences)."""
    result = pycolmap.compare_reconstructions(
        rec, ref, alignment_error="proj_center",
        max_proj_center_error=max_proj_center_error,
    )
    if result is None:
        return None
    errors = result.get("errors", result.get("image_alignment_error"))
    if errors is None:
        return None
    pos = np.array([e.proj_center_error for e in errors])
    rot = np.array([e.rotation_error_deg for e in errors])
    sim3 = result.get("rec2_from_rec1")
    return {
        "n": len(errors),
        "pos_med": float(np.median(pos)),
        "pos_mean": float(pos.mean()),
        "rot_med": float(np.median(rot)),
        "scale": None if sim3 is None else float(sim3.scale),
    }


def center_extent(rec) -> float:
    """Bounding-box diagonal of the registered camera centres, in model units.
    Uses camera centres rather than points3D so stray far-field points can't
    dominate the number."""
    centers = np.array([rec.images[i].projection_center() for i in rec.reg_image_ids()])
    if len(centers) < 2:
        return float("nan")
    return float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))


def fmt(value, spec: str, missing: str = "-") -> str:
    return missing if value is None else format(value, spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="dir containing <sequence>/dslr/sfm_<arm>/0")
    ap.add_argument("--gt_root", type=Path, default=None,
                    help="ScanNet++ data root containing <sequence>/dslr/colmap; "
                         "omit to run the no-GT checks only")
    ap.add_argument("--sequences", nargs="+", default=None,
                    help="default: every subdir of --root")
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--max_proj_center_error", type=float, default=0.2,
                    help="alignment inlier threshold in GT units (m). The "
                         "ETH3D-tuned 0.2 is not obviously right for indoor "
                         "room scale — vary it if alignments fail")
    args = ap.parse_args()

    sequences = args.sequences or sorted(
        p.name for p in args.root.iterdir() if (p / "dslr").is_dir()
    )

    header = (f"{'arm':<16} {'mdl':>3} {'reg':>5} "
              f"{'base_n':>6} {'base_pos':>9} "
              f"{'gt_n':>5} {'gt_pos':>8} {'gt_rot':>7} {'scale':>8}")
    problems: list[str] = []

    for seq in sequences:
        seq_dir = args.root / seq / "dslr"
        gt = None
        if args.gt_root is not None:
            gt_dir = args.gt_root / seq / "dslr" / "colmap"
            gt = load(gt_dir)
            if gt is None:
                problems.append(f"{seq}: no GT model at {gt_dir}")

        print(f"\n=== {seq} ===")
        if gt is not None:
            print(f"GT: {gt.num_reg_images()} reg images, "
                  f"camera-centre extent {center_extent(gt):.2f} "
                  f"(expect a few metres if the GT is metric)")
        print(header)

        baseline = None
        rows = {}
        for arm in args.arms:
            models = submodels(seq_dir / f"sfm_{arm}")
            rows[arm] = (models, load(models[0]) if models else None)
            if not models:
                problems.append(f"{seq}/{arm}: no sub-model directory")
            elif rows[arm][1] is None:
                problems.append(f"{seq}/{arm}: sub-model {models[0]} has no cameras file")
            elif len(models) > 1:
                problems.append(
                    f"{seq}/{arm}: fragmented into {len(models)} sub-models"
                )
        if rows.get("baseline"):
            baseline = rows["baseline"][1]
        if baseline is None:
            problems.append(f"{seq}: no baseline reconstruction — vs_base column is blank")

        for arm in args.arms:
            models, rec = rows[arm]
            if rec is None:
                print(f"{arm:<16} {'-':>3} {'MISSING':>5}")
                continue

            base = None
            if baseline is not None and arm != "baseline":
                base = compare(rec, baseline, args.max_proj_center_error)
                if base is None:
                    problems.append(f"{seq}/{arm}: alignment to baseline FAILED")
                elif base["pos_med"] < 1e-9:
                    problems.append(
                        f"{seq}/{arm}: identical to baseline "
                        f"(median {base['pos_med']:.2e}) — depth factors were a no-op"
                    )

            vs_gt = compare(rec, gt, args.max_proj_center_error) if gt else None
            if gt is not None and vs_gt is None:
                problems.append(f"{seq}/{arm}: alignment to GT FAILED")

            print(
                f"{arm:<16} {len(models):>3} {rec.num_reg_images():>5} "
                f"{fmt(base and base['n'], 'd'):>6} "
                f"{fmt(base and base['pos_med'], '.5f'):>9} "
                f"{fmt(vs_gt and vs_gt['n'], 'd'):>5} "
                f"{fmt(vs_gt and vs_gt['pos_med'], '.5f'):>8} "
                f"{fmt(vs_gt and vs_gt['rot_med'], '.4f'):>7} "
                f"{fmt(vs_gt and vs_gt['scale'], '.4f'):>8}"
            )

    print("\n" + "-" * 72)
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  {p}")
    else:
        print("no structural problems found")


if __name__ == "__main__":
    main()
