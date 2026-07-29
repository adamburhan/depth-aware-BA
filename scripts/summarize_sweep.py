"""Sweep-level sanity + accuracy table for the ScanNet++ depth-BA arms.

Reports the DISTRIBUTION of pose error, not just its median. On ScanNet++ the
baseline's median is excellent while its P90 is catastrophic (a minority of
grossly misregistered cameras), so a median-only summary inverts the
conclusion: depth looks harmful when it is in fact removing the tail. Every
row therefore carries med/mean/P90/max plus bad%, the fraction of cameras
beyond --tail_threshold.

Per (sequence, arm):
  mdl/reg  did the reconstruction stay in one piece, and did the depth
           factors cost registrations? Only sub-model 0 is evaluated, so a
           fragmented arm (mdl > 1) is flagged and its row is not strictly
           comparable to the others.
  dbase    median Sim(3) distance to that sequence's OWN baseline — the "did
           the feature engage" check. ~0 means the depth factors were a
           silent no-op and the vs-GT columns measure nothing.
  pos_*/rot_*  error against the official ScanNet++ COLMAP model.
  scale    Sim(3) scale est->GT. NOT a metricness measure under
           shared_scale: the frozen alpha absorbs the arbitrary gauge scale,
           so depth enforces internal consistency at that scale rather than
           absolute metres. Useful only for cross-arm agreement.

  python scripts/summarize_sweep.py \
      --root $SCRATCH/experiments/depth-aware-ba/scannetpp_amb3r \
      --gt_root $SCRATCH/datasets/scannetpp/data
"""

import argparse
import collections
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

HEADER = (f"{'arm':<16}{'mdl':>4}{'reg':>5}{'dbase':>9}"
          f"{'n':>5}{'pos_med':>8}{'pos_mean':>9}{'pos_p90':>9}{'pos_max':>9}"
          f"{'bad%':>7}{'rot_med':>9}{'rot_p90':>9}{'scale':>8}")
POOLED_HEADER = (f"{'arm':<16}{'nseq':>4}{'ncam':>5}{'dbase':>9}"
                 f"{'n':>5}{'pos_med':>8}{'pos_mean':>9}{'pos_p90':>9}{'pos_max':>9}"
                 f"{'bad%':>7}{'rot_med':>9}{'rot_p90':>9}{'scale':>8}")


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
    """Sim(3)-align rec to ref; return the raw per-camera error arrays (so
    they can be pooled across sequences) plus the alignment scale. None if
    the alignment itself failed."""
    result = pycolmap.compare_reconstructions(
        rec, ref, alignment_error="proj_center",
        max_proj_center_error=max_proj_center_error,
    )
    if result is None:
        return None
    errors = result.get("errors", result.get("image_alignment_error"))
    if errors is None:
        return None
    sim3 = result.get("rec2_from_rec1")
    return {
        "pos": np.array([e.proj_center_error for e in errors]),
        "rot": np.array([e.rotation_error_deg for e in errors]),
        "scale": None if sim3 is None else float(sim3.scale),
    }


def center_extent(rec) -> float:
    """Bounding-box diagonal of the registered camera centres, in model units.
    Camera centres rather than points3D, so stray far-field points can't
    dominate the number."""
    centers = np.array([rec.images[i].projection_center() for i in rec.reg_image_ids()])
    if len(centers) < 2:
        return float("nan")
    return float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))


def metrics_cells(vs_gt: dict | None, tail: float) -> str:
    """The shared pos/rot/scale column block."""
    if vs_gt is None:
        return f"{'-':>5}{'-':>8}{'-':>9}{'-':>9}{'-':>9}{'-':>7}{'-':>9}{'-':>9}{'-':>8}"
    pos, rot = vs_gt["pos"], vs_gt["rot"]
    scale = vs_gt["scale"]
    return (
        f"{len(pos):>5}"
        f"{np.median(pos):>8.4f}{pos.mean():>9.4f}"
        f"{np.percentile(pos, 90):>9.4f}{pos.max():>9.4f}"
        f"{100.0 * (pos > tail).mean():>7.1f}"
        f"{np.median(rot):>9.4f}{np.percentile(rot, 90):>9.4f}"
        + (f"{'-':>8}" if scale is None else f"{scale:>8.4f}")
    )


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
    ap.add_argument("--tail_threshold", type=float, default=0.25,
                    help="bad%% column: fraction of cameras with position "
                         "error above this, in GT metres")
    ap.add_argument("--verbose_colmap", action="store_true",
                    help="keep COLMAP's per-alignment glog output (very noisy)")
    args = ap.parse_args()

    if not args.verbose_colmap:
        pycolmap.logging.minloglevel = 2  # glog: 0=INFO 1=WARNING 2=ERROR

    sequences = args.sequences or sorted(
        p.name for p in args.root.iterdir() if (p / "dslr").is_dir()
    )

    problems: list[str] = []
    # arm -> pooled raw errors across sequences
    pooled = collections.defaultdict(
        lambda: {"pos": [], "rot": [], "scale": [], "dbase": [], "nseq": 0}
    )

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
            print(f"GT: {gt.num_reg_images()} reg images, camera-centre extent "
                  f"{center_extent(gt):.2f} (expect a few metres if GT is metric)")
        print(HEADER)

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
                    f"{seq}/{arm}: fragmented into {len(models)} sub-models — only "
                    f"sub-model 0 evaluated, row not comparable to the others"
                )
        baseline = rows["baseline"][1] if "baseline" in rows else None
        if baseline is None:
            problems.append(f"{seq}: no baseline reconstruction — dbase column is blank")

        for arm in args.arms:
            models, rec = rows[arm]
            if rec is None:
                print(f"{arm:<16}{'-':>4}{'-':>5}{'-':>9}"
                      + metrics_cells(None, args.tail_threshold) + "  MISSING")
                continue

            dbase = None
            if baseline is not None and arm != "baseline":
                base = compare(rec, baseline, args.max_proj_center_error)
                if base is None:
                    problems.append(f"{seq}/{arm}: alignment to baseline FAILED")
                else:
                    dbase = float(np.median(base["pos"]))
                    if dbase < 1e-9:
                        problems.append(
                            f"{seq}/{arm}: identical to baseline (median "
                            f"{dbase:.2e}) — depth factors were a no-op"
                        )

            vs_gt = compare(rec, gt, args.max_proj_center_error) if gt else None
            if gt is not None and vs_gt is None:
                problems.append(f"{seq}/{arm}: alignment to GT FAILED")

            print(
                f"{arm:<16}{len(models):>4}{rec.num_reg_images():>5}"
                + (f"{'-':>9}" if dbase is None else f"{dbase:>9.5f}")
                + metrics_cells(vs_gt, args.tail_threshold)
            )

            acc = pooled[arm]
            acc["nseq"] += 1
            if dbase is not None:
                acc["dbase"].append(dbase)
            if vs_gt is not None:
                acc["pos"].append(vs_gt["pos"])
                acc["rot"].append(vs_gt["rot"])
                if vs_gt["scale"] is not None:
                    acc["scale"].append(vs_gt["scale"])

    # ---- pooled across sequences -------------------------------------------
    # Cameras are pooled rather than per-sequence medians averaged: GT is
    # metric in every sequence so the errors share units, and the tail is the
    # quantity of interest. Consequence: sequences are weighted by their
    # registered-image count, not equally.
    if len(sequences) > 1:
        print(f"\n=== POOLED over {len(sequences)} sequences "
              f"(cameras pooled; weights by image count) ===")
        print(POOLED_HEADER)
        for arm in args.arms:
            acc = pooled.get(arm)
            if not acc or not acc["pos"]:
                continue
            vs_gt = {
                "pos": np.concatenate(acc["pos"]),
                "rot": np.concatenate(acc["rot"]),
                "scale": float(np.median(acc["scale"])) if acc["scale"] else None,
            }
            dbase = float(np.median(acc["dbase"])) if acc["dbase"] else None
            print(
                f"{arm:<16}{acc['nseq']:>4}{len(vs_gt['pos']):>5}"
                + (f"{'-':>9}" if dbase is None else f"{dbase:>9.5f}")
                + metrics_cells(vs_gt, args.tail_threshold)
            )
        print(f"\nbad% = cameras with position error > {args.tail_threshold} m.  "
              "scale in the pooled row is the median over sequences.")

    print("\n" + "-" * 98)
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  {p}")
    else:
        print("no structural problems found")


if __name__ == "__main__":
    main()
