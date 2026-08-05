"""Per-view outliers: single views where one arm beats another by a lot.

A localized mechanism (a few hundred switched points per scene) should show
up as a handful of views with large deltas, not as a shifted scene mean --
so this censuses both tails and ranks the extremes for eyeballing. A
lopsided count is the evidence; a balanced one is noise.

Deltas are scored in units of the cell's pooled replicate sd, NOT the view's
own 3-repeat spread: with 2 degrees of freedom that spread invents outliers
wherever three runs happen to agree. The per-view spread is printed anyway,
as an instability check.

  python scripts/extreme_views.py --root $SCRATCH/.../scannetpp_amb3r --paths
"""

import argparse
import collections
from pathlib import Path

import numpy as np

from summarize_nvs import ITER, cell, held_out

# sign that turns the metric into higher-is-better
BETTER = {"PSNR": 1.0, "SSIM": 1.0, "LPIPS": -1.0}

_splits: dict[Path, list[str]] = {}


def split_of(gs_dir: Path) -> list[str]:
    if gs_dir not in _splits:
        _splits[gs_dir] = held_out(gs_dir)
    return _splits[gs_dir]


def arm(ba_dir: Path, variant: str, metric: str):
    """[(gs_dir, {view: metric})] over this arm's repeats."""
    out = []
    for gs in sorted((ba_dir / variant).glob("gs_repeat*")):
        if (gs / "per_view.json").exists():
            out.append((gs, cell(gs, metric)))
    return out


def per_view(repeats, views):
    """{view: (mean, sd)} over repeats, plus the cell's pooled sd."""
    stats = {}
    for v in views:
        values = [scores[v] for _, scores in repeats]
        stats[v] = (float(np.mean(values)),
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
    pooled = float(np.median([sd for _, sd in stats.values()]))
    return stats, pooled


def render_paths(repeats, view: str) -> str:
    gs, _ = repeats[0]
    idx = split_of(gs).index(view)
    return str(gs / "test" / ITER / "renders" / f"{idx:05d}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--metric", default="PSNR", help="PSNR | SSIM | LPIPS")
    ap.add_argument("--sequences", nargs="+", default=None)
    ap.add_argument("--reference", default="baseline")
    ap.add_argument("--contrast", default="amb3r_gmm_all,amb3r_unimodal_all")
    ap.add_argument("--min_delta", type=float, default=0.5)
    ap.add_argument("--min_z", type=float, default=3.0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--paths", action="store_true", help="print render paths to eyeball")
    args = ap.parse_args()

    flip = BETTER[args.metric]
    lhs, _, rhs = args.contrast.partition(",")
    sequences = args.sequences or sorted(
        p.name for p in args.root.iterdir() if (p / "dslr" / "ba").is_dir()
    )

    census = []   # (scene, suffix, contrast, n_views, n_pos, n_neg)
    rows = []     # extreme views
    for seq in sequences:
        ba = args.root / seq / "dslr" / "ba"
        if not ba.is_dir():
            continue
        present = {p.name for p in ba.iterdir() if p.is_dir()}
        for suffix in sorted(v[len(lhs):] for v in present if v.startswith(lhs)):
            names = {"gmm": lhs + suffix, "uni": rhs + suffix, "base": args.reference}
            if not all(n in present for n in names.values()):
                continue
            arms = {k: arm(ba, n, args.metric) for k, n in names.items()}
            if not all(arms.values()):
                continue
            seen = {k: set.intersection(*(set(s) for _, s in reps))
                    for k, reps in arms.items()}

            for a, b in (("gmm", "uni"), ("gmm", "base"), ("uni", "base")):
                label = f"{a}-{b}"
                # Pairwise, never three-way: baseline holding the other half of
                # a split scene must not delete the gmm-uni contrast.
                views = sorted(seen[a] & seen[b])
                if not views:
                    continue
                stats_a, pooled_a = per_view(arms[a], views)
                stats_b, pooled_b = per_view(arms[b], views)
                stats = {a: stats_a, b: stats_b}
                denom = max(np.hypot(pooled_a, pooled_b), 1e-6)
                pos = neg = 0
                for v in views:
                    delta = flip * (stats[a][v][0] - stats[b][v][0])
                    z = delta / denom
                    if abs(delta) < args.min_delta or abs(z) < args.min_z:
                        continue
                    pos, neg = pos + (delta > 0), neg + (delta < 0)
                    rows.append((abs(z), seq, suffix.lstrip("_"), v, label, delta, z,
                                 stats[a][v], stats[b][v], arms[a], arms[b]))
                census.append((seq, suffix.lstrip("_"), label, len(views), pos, neg))

    print(f"census: |delta| > {args.min_delta} and |z| > {args.min_z}  "
          f"({args.metric}, sign flipped so + = first arm better)")
    print(f"{'scene':<12}{'cell':<20}{'contrast':<12}{'views':>6}{'+':>5}{'-':>5}")
    for scene, suffix, label, n, pos, neg in census:
        if pos or neg:
            print(f"{scene:<12}{suffix:<20}{label:<12}{n:>6}{pos:>5}{neg:>5}")

    totals = collections.Counter()
    for _, _, label, _, pos, neg in census:
        totals[label] += 0
        totals[(label, "+")] += pos
        totals[(label, "-")] += neg
    print("\npooled tails (all scenes and cells)")
    for label in ("gmm-uni", "gmm-base", "uni-base"):
        print(f"  {label:<12}{totals[(label, '+')]:>4} + / {totals[(label, '-')]:>4} -")

    rows.sort(reverse=True)
    print(f"\ntop {args.top} views by |z|")
    print(f"{'scene':<12}{'cell':<18}{'view':<12}{'contrast':<11}"
          f"{'delta':>8}{'z':>7}{'a':>8}{'b':>8}{'a_sd':>6}{'b_sd':>6}")
    for _, seq, suffix, view, label, delta, z, a, b, reps_a, reps_b in rows[:args.top]:
        print(f"{seq:<12}{suffix:<18}{view:<12}{label:<11}"
              f"{delta:>8.2f}{z:>7.1f}{a[0]:>8.2f}{b[0]:>8.2f}{a[1]:>6.2f}{b[1]:>6.2f}")
        if args.paths:
            print(f"    a: {render_paths(reps_a, view)}")
            print(f"    b: {render_paths(reps_b, view)}")


if __name__ == "__main__":
    main()
