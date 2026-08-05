"""Paired NVS table on the test views the compared arms actually held out.

3DGS derives its holdout from the arm's OWN registered set (sorted by name,
every 8th), so two arms that register different images hold out different
views and their scene means are not comparable. per_view.json is keyed by
render index, which hides this — index 3 is a different photo in each arm.
So each cell's indices are mapped back to image names through the sparse
model train.py read, and every comparison is restricted to the names both
sides of that comparison held out. Intersections are taken per COMPARISON
PAIR, not per scene: one arm registering the other half of a split scene
must not shrink the contrast between two arms that agree.

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


def views_of(repeats: list[dict[str, float]]) -> set[str]:
    return set.intersection(*(set(r) for r in repeats))


def score(repeats: list[dict[str, float]], views: list[str]) -> tuple[float, float]:
    """Mean over the given views, then mean and sd over repeats."""
    per_repeat = [float(np.mean([r[v] for v in views])) for r in repeats]
    sd = float(np.std(per_repeat, ddof=1)) if len(per_repeat) > 1 else 0.0
    return float(np.mean(per_repeat)), sd


def contrast(cells, a: str, b: str, min_views: int):
    """Delta between two arms on the views BOTH held out. None if either arm
    is missing or the overlap is too thin to mean anything."""
    if a not in cells or b not in cells:
        return None
    views = sorted(views_of(cells[a]) & views_of(cells[b]))
    if len(views) < min_views:
        return None
    return score(cells[a], views)[0] - score(cells[b], views)[0], len(views)


def pooled(title: str, deltas: dict[str, list], thin: list[str]) -> None:
    if not deltas:
        return
    print(f"\n{title}")
    print(f"{'variant':<36}{'n':>3}{'views':>7}{'mean':>9}{'sd':>8}{'worst':>9}{'sign':>7}")
    for variant in sorted(deltas):
        values = np.array([d for d, _ in deltas[variant]])
        nviews = np.array([n for _, n in deltas[variant]])
        sd = values.std(ddof=1) if len(values) > 1 else 0.0
        print(f"{variant:<36}{len(values):>3}{int(nviews.min()):>7}{values.mean():>9.3f}"
              f"{sd:>8.3f}{values.min():>9.3f}{int((values > 0).sum()):>4}/{len(values)}")
    if thin:
        print(f"  excluded for < min_views: {', '.join(sorted(set(thin)))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--metric", default="PSNR", help="PSNR | SSIM | LPIPS")
    ap.add_argument("--sequences", nargs="+", default=None)
    ap.add_argument("--reference", default="baseline")
    ap.add_argument("--contrast", default="amb3r_gmm_all,amb3r_unimodal_all",
                    help="prefix pair for the arm-vs-arm block, '' to skip")
    ap.add_argument("--min_views", type=int, default=10,
                    help="drop a scene from a comparison below this overlap")
    args = ap.parse_args()

    sequences = args.sequences or sorted(
        p.name for p in args.root.iterdir() if (p / "dslr" / "ba").is_dir()
    )
    lhs, _, rhs = args.contrast.partition(",")

    vs_ref = collections.defaultdict(list)
    vs_arm = collections.defaultdict(list)
    thin_ref, thin_arm = [], []

    print(f"{'scene':<12}{'variant':<36}{'n':>3}{'views':>8}{args.metric:>9}{'sd':>7}"
          f"{'vs ' + args.reference:>10}")
    for seq in sequences:
        cells = collect(args.root / seq / "dslr" / "ba", args.metric)
        if not cells:
            continue
        for variant in sorted(cells):
            # Display row: the arm on its own split, so the number is the one
            # its results.json reports. Comparisons below are view-matched.
            own = sorted(views_of(cells[variant]))
            mean, sd = score(cells[variant], own)
            delta = contrast(cells, variant, args.reference, args.min_views)
            if variant != args.reference:
                vs_ref[variant].append(delta) if delta else thin_ref.append(seq)
            shown = f"{delta[0]:+10.3f}" if delta else (
                "" if variant == args.reference else f"{'thin':>10}")
            print(f"{seq:<12}{variant:<36}{len(cells[variant]):>3}{len(own):>8}"
                  f"{mean:>9.3f}{sd:>7.3f}{shown}")

            if lhs and rhs and variant.startswith(lhs):
                other = rhs + variant[len(lhs):]
                pair = contrast(cells, variant, other, args.min_views)
                vs_arm[variant].append(pair) if pair else thin_arm.append(seq)
        print()

    pooled(f"pooled paired delta vs {args.reference} ({args.metric}, per scene, "
           f"view-matched per pair)", vs_ref, thin_ref)
    if lhs and rhs:
        pooled(f"pooled paired delta {lhs} - {rhs} ({args.metric}, per scene)",
               vs_arm, thin_arm)


if __name__ == "__main__":
    main()
