"""Find views where two arms disagree LOCALLY, and contact-sheet the crops.

Ranking by image-level PSNR delta finds diffuse differences (exposure, blur)
and buries localized ones. So: difference the two arms' RENDERS against each
other -- not against GT, which is dominated by error both arms share -- and
score each view by how concentrated that disagreement is.

    flagrancy = p99 patch error / p50 patch error

A floater spikes one patch and leaves the rest quiet (high ratio); a global
shift moves every patch together (ratio ~1) however large it is. Top views
are cropped around the offending patch and tiled into one sheet per scene.

  python scripts/mine_views.py --root $SCRATCH/.../scannetpp_amb3r \\
      --cell sigma_scale2.0 --out plots/mined
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from summarize_nvs import ITER, held_out

PATCH = 32


def render_path(gs_dir: Path, view: str, kind: str = "renders") -> Path | None:
    names = held_out(gs_dir)
    if view not in names:
        return None
    return gs_dir / "test" / ITER / kind / f"{names.index(view):05d}.png"


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def patch_errors(a: np.ndarray, b: np.ndarray, patch: int) -> np.ndarray:
    """Mean |a-b| per patch, as a 2D grid."""
    err = np.abs(a - b).mean(axis=2)
    h, w = (err.shape[0] // patch) * patch, (err.shape[1] // patch) * patch
    grid = err[:h, :w].reshape(h // patch, patch, w // patch, patch)
    return grid.mean(axis=(1, 3))


def flagrancy(grid: np.ndarray) -> tuple[float, float, tuple[int, int]]:
    """Concentration of disagreement, its peak, and where the peak is."""
    hi, mid = np.percentile(grid, 99), np.percentile(grid, 50)
    peak = np.unravel_index(int(np.argmax(grid)), grid.shape)
    return float(hi / max(mid, 1e-3)), float(grid.max()), (int(peak[0]), int(peak[1]))


def crop(path: Path, peak: tuple[int, int], patch: int, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    cy, cx = (peak[0] + 0.5) * patch, (peak[1] + 0.5) * patch
    left = int(np.clip(cx - size / 2, 0, max(image.width - size, 0)))
    top = int(np.clip(cy - size / 2, 0, max(image.height - size, 0)))
    return image.crop((left, top, left + size, top + size))


def sheet(rows, labels, out: Path, size: int) -> None:
    """One row per view, one column per arm, labelled."""
    band = 16
    canvas = Image.new("RGB", (size * len(labels), (size + band) * len(rows)), "black")
    draw = ImageDraw.Draw(canvas)
    for r, (caption, images) in enumerate(rows):
        y = r * (size + band)
        draw.text((3, y + 3), caption, fill="white")
        for c, image in enumerate(images):
            if image is not None:
                canvas.paste(image, (c * size, y + band))
            draw.text((c * size + 3, y + band + 3), labels[c], fill="yellow")
    canvas.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sequences", nargs="+", default=None)
    ap.add_argument("--cell", default="sigma_scale2.0")
    ap.add_argument("--contrast", default="amb3r_gmm_all,amb3r_unimodal_all")
    ap.add_argument("--repeat", default="gs_repeat0", help="screening pass: one repeat")
    ap.add_argument("--patch", type=int, default=PATCH)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--min_peak", type=float, default=4.0,
                    help="ignore views whose worst patch is this quiet (0-255)")
    ap.add_argument("--top", type=int, default=8, help="views per scene")
    args = ap.parse_args()

    lhs, _, rhs = args.contrast.partition(",")
    sequences = args.sequences or sorted(
        p.name for p in args.root.iterdir() if (p / "dslr" / "ba").is_dir()
    )
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"{'scene':<12}{'view':<12}{'flagrancy':>10}{'peak':>7}  (patch {args.patch})")
    def arm_dir(ba: Path, name: str) -> Path | None:
        # baseline carries no sigma cell, so fall back to the bare name
        for candidate in (f"{name}_{args.cell}" if args.cell else name, name):
            if (ba / candidate / args.repeat / "per_view.json").exists():
                return ba / candidate / args.repeat
        return None

    for seq in sequences:
        ba = args.root / seq / "dslr" / "ba"
        a_dir, b_dir = arm_dir(ba, lhs), arm_dir(ba, rhs)
        if a_dir is None or b_dir is None:
            continue
        views = sorted(set(held_out(a_dir)) & set(held_out(b_dir)))

        scored = []
        for view in views:
            pa, pb = render_path(a_dir, view), render_path(b_dir, view)
            grid = patch_errors(load(pa), load(pb), args.patch)
            ratio, peak_err, peak = flagrancy(grid)
            if peak_err >= args.min_peak:
                scored.append((ratio, peak_err, peak, view, pa, pb))
        if not scored:
            print(f"{seq:<12}nothing above --min_peak")
            continue

        scored.sort(reverse=True)
        rows = []
        for ratio, peak_err, peak, view, pa, pb in scored[:args.top]:
            print(f"{seq:<12}{view:<12}{ratio:>10.1f}{peak_err:>7.1f}")
            gt = render_path(a_dir, view, "gt")
            rows.append((f"{view}  flag {ratio:.1f}  peak {peak_err:.1f}",
                         [crop(p, peak, args.patch, args.crop) for p in (pa, pb, gt)]))
        sheet(rows, [lhs, rhs, "GT"], args.out / f"{seq}_{args.cell}.png", args.crop)
        print(f"  -> {args.out / f'{seq}_{args.cell}.png'}")


if __name__ == "__main__":
    main()
