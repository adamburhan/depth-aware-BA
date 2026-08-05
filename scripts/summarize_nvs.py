"""Paired NVS table on the test views every arm of a scene actually held out.

3DGS derives its holdout from the arm's OWN registered set (sorted by name,
every 8th), so two arms that register different images hold out different
views and their scene means are not comparable. per_view.json is keyed by
render index, which hides this — index 3 is a different photo in each arm.
So each cell's indices are mapped back to image names through the sparse
model train.py read, and every comparison is restricted to the names all
arms of that scene held out.

  python scripts/summarize_nvs.py --root $SCRATCH/.../scannetpp_amb3r
"""

import argparse
import collections
import json
from pathlib import Path

import numpy as np

import pycolmap

LLFFHOLD = 8
ITER = "ours_30000"


def held_out(gs_dir: Path) -> list[str]:
    """Image names this arm held out, in render order — the split
    readColmapSceneInfo builds: sort the registered set by name, take every
    LLFFHOLD-th."""
    rec = pycolmap.Reconstruction(gs_dir / "source" / "sparse" / "0")
    names = sorted(Path(image.name).stem for image in rec.images.values())
    return names[::LLFFHOLD]


def cell(gs_dir: Path, metric: str) -> dict[str, float]:
    """{image name: metric} for one repeat, undoing the index keying."""
    raw = json.loads((gs_dir / "per_view.json").read_text())
    key = ITER if ITER in raw else sorted(raw)[-1]
    scores = raw[key][metric]
    names = held_out(gs_dir)
    if len(scores) != len(names):
        raise ValueError(
            f"{gs_dir}: {len(scores)} rendered views but {len(names)} held out "
            "— the llffhold reconstruction does not match this run"
        )
    return {names[int(Path(k).stem)]: v for k, v in scores.items()}


def collect(ba_dir: Path, metric: str) -> dict[str, list[dict[str, float]]]:
    """{variant: [per-repeat {name: metric}]} for every finished gs run."""
    out = collections.defaultdict(list)
    for variant in sorted(p for p in ba_dir.iterdir() if p.is_dir()):
        for gs in sorted(variant.glob("gs_repeat*")):
            if (gs / "per_view.json").exists():
                out[variant.name].append(cell(gs, metric))
    return out


def stats(repeats: list[dict[str, float]], views: list[str]) -> tuple[float, float]:
    """Mean over the common views, then mean and sd over repeats."""
    per_repeat = [float(np.mean([r[v] for v in views])) for r in repeats]
    return float(np.mean(per_repeat)), float(np.std(per_repeat, ddof=1)) if len(per_repeat) > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--metric", default="PSNR", help="PSNR | SSIM | LPIPS")
    ap.add_argument("--sequences", nargs="+", default=None)
    ap.add_argument("--reference", default="baseline")
    args = ap.parse_args()

    sequences = args.sequences or sorted(
        p.name for p in args.root.iterdir() if (p / "dslr" / "ba").is_dir()
    )

    deltas = collections.defaultdict(list)   # variant -> [(seq, delta)]
    print(f"{'scene':<12}{'variant':<36}{'n':>3}{'views':>6}"
          f"{args.metric:>9}{'sd':>7}{'vs ' + args.reference:>10}")
    for seq in sequences:
        cells = collect(args.root / seq / "dslr" / "ba", args.metric)
        if not cells:
            continue
        # Every repeat of a cell shares a split (same reconstruction), so the
        # intersection is over variants; taking it over repeats too is free.
        common = set.intersection(
            *(set(r) for repeats in cells.values() for r in repeats)
        )
        if not common:
            print(f"{seq:<12}no views held out by every arm")
            continue
        views = sorted(common)
        full = max(len(r) for repeats in cells.values() for r in repeats)

        ref = stats(cells[args.reference], views)[0] if args.reference in cells else None
        for variant in sorted(cells):
            mean, sd = stats(cells[variant], views)
            delta = "" if ref is None else f"{mean - ref:+10.3f}"
            if ref is not None and variant != args.reference:
                deltas[variant].append((seq, mean - ref))
            print(f"{seq:<12}{variant:<36}{len(cells[variant]):>3}"
                  f"{len(views):>4}/{full:<2}{mean:>8.3f}{sd:>7.3f}{delta}")
        print()

    if not deltas:
        return
    print(f"\npooled paired delta vs {args.reference} "
          f"({args.metric}, one observation per scene)")
    print(f"{'variant':<36}{'n':>3}{'mean':>9}{'sd':>8}{'worst':>9}{'sign':>7}")
    for variant in sorted(deltas):
        values = np.array([d for _, d in deltas[variant]])
        positive = int((values > 0).sum())
        print(f"{variant:<36}{len(values):>3}{values.mean():>9.3f}"
              f"{values.std(ddof=1) if len(values) > 1 else 0.0:>8.3f}"
              f"{values.min():>9.3f}{positive:>4}/{len(values)}")


if __name__ == "__main__":
    main()
