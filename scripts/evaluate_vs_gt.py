"""Evaluate reconstructions against ETH3D ground truth.

Sim(3)-aligns each estimate to the GT model (gauge-free comparison, per
design), then reports per-camera position/rotation errors and the alignment
scale — the latter measures how metric the estimate's own scale is (1.0 =
perfectly metric; only meaningful for depth-anchored conditions).

  python scripts/evaluate_vs_gt.py \
      --gt $DATA/kicker/dslr_calibration_undistorted \
      $OUT/sfm_baseline/0 $OUT/sfm_depthpro/0
"""

import argparse
from pathlib import Path

import numpy as np

import pycolmap


def normalize_image_names(rec: pycolmap.Reconstruction) -> None:
    """Strip directory prefixes: ETH3D GT stores 'dslr_images_undistorted/DSC_x.JPG'
    while our db (built with image_path inside that dir) stores bare names —
    comparison matches by exact string."""
    for image in rec.images.values():
        image.name = Path(image.name).name


def evaluate(gt: pycolmap.Reconstruction, rec_path: Path) -> None:
    rec = pycolmap.Reconstruction(rec_path)
    normalize_image_names(rec)
    print(f"\n=== {rec_path} ===")
    print(f"registered: {rec.num_reg_images()} (GT: {gt.num_reg_images()})")

    result = pycolmap.compare_reconstructions(
        rec, gt, alignment_error="proj_center", max_proj_center_error=0.2
    )
    if result is None:
        print("ALIGNMENT FAILED — reconstructions not comparable")
        return
    if "rec2_from_rec1" in result:
        sim3 = result["rec2_from_rec1"]
        print(f"sim3 est->GT: scale {sim3.scale:.4f} "
              "(est scale is arbitrary unless depth-anchored)")
    errors = result.get("errors", result.get("image_alignment_error"))
    if errors is None:
        print("unexpected result keys:", sorted(result.keys()))
        return

    pos = np.array([e.proj_center_error for e in errors])
    rot = np.array([e.rotation_error_deg for e in errors])
    print(f"images compared: {len(errors)}")
    print(f"position error [m]:  mean {pos.mean():.4f}  median {np.median(pos):.4f}  "
          f"max {pos.max():.4f}")
    print(f"rotation error [deg]: mean {rot.mean():.4f}  median {np.median(rot):.4f}  "
          f"max {rot.max():.4f}")
    worst = max(errors, key=lambda e: e.proj_center_error)
    print(f"worst image: {worst.image_name} "
          f"({worst.proj_center_error:.4f} m, {worst.rotation_error_deg:.3f} deg)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True,
                        help="GT model dir (e.g. kicker/dslr_calibration_undistorted)")
    parser.add_argument("reconstructions", nargs="+",
                        help="estimated reconstruction dirs (e.g. .../sfm_baseline/0)")
    args = parser.parse_args()

    gt = pycolmap.Reconstruction(args.gt)
    normalize_image_names(gt)
    for rec_path in args.reconstructions:
        evaluate(gt, Path(rec_path))


if __name__ == "__main__":
    main()
