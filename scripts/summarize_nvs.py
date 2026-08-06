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

--csv writes one long-format row per (scene, variant, reference) for the
notebook; `delta` is in metric units and `improvement` is sign-normalised so
that positive always means better, LPIPS included.

  python scripts/summarize_nvs.py --root $SCRATCH/.../scannetpp_amb3r \\
      --csv nvs_psnr.csv
"""

import argparse
import collections
import csv
import json
import re
from pathlib import Path

import numpy as np

import pycolmap

LLFFHOLD = 8
ITER = "ours_30000"
# sign that turns the metric into higher-is-better
BETTER = {"PSNR": 1.0, "SSIM": 1.0, "LPIPS": -1.0}
CELL = re.compile(r"^(.*)_sigma_scale([0-9.]+)$")

FIELDS = ["scene", "variant", "sensor", "sigma_scale", "metric", "reference",
          "n_reps", "n_views", "value", "sd", "ref_value", "ref_sd",
          "delta", "improvement"]


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


def record(cells, scene, a, b, metric, min_views):
    """One paired comparison on the views BOTH arms held out. None if either
    arm is missing or the overlap is too thin to mean anything."""
    if a not in cells or b not in cells:
        return None
    views = sorted(views_of(cells[a]) & views_of(cells[b]))
    if len(views) < min_views:
        return None
    value, sd = score(cells[a], views)
    ref_value, ref_sd = score(cells[b], views)
    sensor, sigma = (CELL.match(a).groups() if CELL.match(a) else (a, ""))
    return dict(scene=scene, variant=a, sensor=sensor, sigma_scale=sigma,
                metric=metric, reference=b, n_reps=len(cells[a]),
                n_views=len(views), value=round(value, 4), sd=round(sd, 4),
                ref_value=round(ref_value, 4), ref_sd=round(ref_sd, 4),
                delta=round(value - ref_value, 4),
                improvement=round(BETTER[metric] * (value - ref_value), 4))


def pooled(title: str, records: list[dict], thin: list[str]) -> None:
    if not records:
        return
    by_variant = collections.defaultdict(list)
    for r in records:
        by_variant[r["variant"]].append(r)
    print(f"\n{title} — improvement is sign-normalised, + is always better")
    print(f"{'variant':<36}{'n':>3}{'views':>7}{'mean':>9}{'sd':>8}{'worst':>9}{'sign':>7}")
    for variant in sorted(by_variant):
        rows = by_variant[variant]
        values = np.array([r["improvement"] for r in rows])
        sd = values.std(ddof=1) if len(values) > 1 else 0.0
        print(f"{variant:<36}{len(values):>3}{min(r['n_views'] for r in rows):>7}"
              f"{values.mean():>9.3f}{sd:>8.3f}{values.min():>9.3f}"
              f"{int((values > 0).sum()):>4}/{len(values)}")
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
    ap.add_argument("--csv", type=Path, default=None, help="long-format dump")
    args = ap.parse_args()

    sequences = args.sequences or sorted(
        p.name for p in args.root.iterdir() if (p / "dslr" / "ba").is_dir()
    )
    lhs, _, rhs = args.contrast.partition(",")

    vs_ref, vs_arm, thin_ref, thin_arm = [], [], [], []
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
            row = None
            if variant != args.reference:
                row = record(cells, seq, variant, args.reference,
                             args.metric, args.min_views)
                vs_ref.append(row) if row else thin_ref.append(seq)
            shown = f"{row['delta']:+10.3f}" if row else (
                "" if variant == args.reference else f"{'thin':>10}")
            print(f"{seq:<12}{variant:<36}{len(cells[variant]):>3}{len(own):>8}"
                  f"{mean:>9.3f}{sd:>7.3f}{shown}")

            if lhs and rhs and variant.startswith(lhs):
                pair = record(cells, seq, variant, rhs + variant[len(lhs):],
                              args.metric, args.min_views)
                vs_arm.append(pair) if pair else thin_arm.append(seq)
        print()

    pooled(f"pooled paired delta vs {args.reference} ({args.metric}, per scene, "
           f"view-matched per pair)", vs_ref, thin_ref)
    if lhs and rhs:
        pooled(f"pooled paired delta {lhs} - {rhs} ({args.metric}, per scene)",
               vs_arm, thin_arm)

    if args.csv:
        with args.csv.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(vs_ref + vs_arm)
        print(f"\nwrote {len(vs_ref) + len(vs_arm)} rows to {args.csv}")


if __name__ == "__main__":
    main()
