"""Stage named test views for eyeballing: every arm, every repeat, plus GT.

Renders are indexed per arm (00000.png), and the index for a given photo
differs between arms, so a view can only be collected by name through each
arm's own holdout order.

  python scripts/stage_views.py --root $SCRATCH/.../scannetpp_amb3r \
      --out $SCRATCH/view_dump --diff \
      13c3e046d7:DSC01448 1ada7a0617:DSC03596 1ada7a0617:DSC03716
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

from summarize_nvs import ITER, held_out

ARMS = ["amb3r_gmm_all_{cell}", "amb3r_unimodal_all_{cell}", "baseline"]


def diff_map(render: Path, gt: Path, out: Path) -> None:
    """|render - GT| as a heatmap — puts the defect where the eye goes."""
    from PIL import Image

    a = np.asarray(Image.open(render).convert("RGB"), dtype=np.float32)
    b = np.asarray(Image.open(gt).convert("RGB"), dtype=np.float32)
    err = np.abs(a - b).mean(axis=2)
    err = np.clip(err / max(err.max(), 1e-6), 0, 1)
    # blue -> red ramp, no colormap dependency
    rgb = np.stack([err, np.zeros_like(err), 1.0 - err], axis=2)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("views", nargs="+", help="scene:VIEWNAME, e.g. 13c3e046d7:DSC01448")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cell", default="sigma_scale2.0")
    ap.add_argument("--diff", action="store_true", help="also write |render-GT| maps")
    args = ap.parse_args()

    for spec in args.views:
        seq, _, view = spec.partition(":")
        dest = args.out / seq / f"{args.cell}_{view}"
        dest.mkdir(parents=True, exist_ok=True)
        got_gt = False
        for template in ARMS:
            variant = template.format(cell=args.cell)
            for gs in sorted((args.root / seq / "dslr" / "ba" / variant).glob("gs_repeat*")):
                if not (gs / "test" / ITER / "renders").is_dir():
                    continue
                names = held_out(gs)
                if view not in names:
                    print(f"  {variant} {gs.name}: {view} not in this arm's holdout")
                    continue
                idx = names.index(view)
                src = gs / "test" / ITER / "renders" / f"{idx:05d}.png"
                tag = f"{variant}__{gs.name}"
                shutil.copy2(src, dest / f"{tag}.png")
                if args.diff:
                    diff_map(src, gs / "test" / ITER / "gt" / f"{idx:05d}.png",
                             dest / f"{tag}__diff.png")
                if not got_gt:
                    shutil.copy2(gs / "test" / ITER / "gt" / f"{idx:05d}.png",
                                 dest / "gt.png")
                    got_gt = True
        print(f"{dest}: {len(list(dest.glob('*.png')))} images")


if __name__ == "__main__":
    main()
