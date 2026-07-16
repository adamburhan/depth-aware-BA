"""Visualize the fitted 2-mode patch-GMM for a chosen image's keypoints.

Reconstructs, per keypoint, exactly what the gmm_patch extractor saw (the disk
from sqrt(detA)*patch_scale, the spatial weights) and overlays the fit stored in
the database (modes, weights, sigmas). Nothing is re-fit -- the DB rows are the
source of truth; the dump is reloaded only to recover the patch pixels.

Selection: all keypoints of the image, a random sample, or the top-k ranked by
ambiguity = min(weight) * |log(mode0 / mode1)| (collapsed/unimodal ~ 0).

    uv run python scripts/viz_gmm_patch.py --db $OUT/database.db \
        --sensor depthpro_gmm --image DSC_0123.JPG -n 8 --rank ambiguity \
        --images_dir $SCRATCH/datasets/eth3d/kicker/images --out plots/kicker_0123.pdf
"""

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pycolmap

from depthba.depth import schema
from depthba.depth.extractors.gmm_patch import _disk
from depthba.depth.source import DepthSource

_TINY = 1e-12
ANCHOR_C, FREE_C = "#0173B2", "#D55E00"  # colorblind-safe: mode 0 (anchor) vs mode 1 (free)


def _find_image(images, name):
    by_name = {im.name: im for im in images}
    if name in by_name:
        return by_name[name]
    by_stem = {Path(im.name).stem: im for im in images}
    if Path(name).stem in by_stem:
        return by_stem[Path(name).stem]
    raise SystemExit(f"image {name!r} not found (have e.g. {list(by_name)[:5]})")


def _ambiguity(kd):
    """min(weight) * |log(mode0/mode1)|; ~0 for collapsed/unimodal rows."""
    m0, m1 = kd.modes
    return float(kd.weights.min() * abs(np.log(max(m0, _TINY) / max(m1, _TINY))))


def _gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _patch(depth, v, u, r):
    """Return (disk depths, dv, du, valid-disk mask on a square crop) for imshow."""
    h, w = depth.shape
    dv, du = _disk(r)
    vv, uu = v + dv, u + du
    inb = (vv >= 0) & (vv < h) & (uu >= 0) & (uu < w)
    return depth[vv[inb], uu[inb]], dv[inb], du[inb]


def _panel_depth(ax, depth, v, u, r):
    h, w = depth.shape
    v0, v1 = max(v - r, 0), min(v + r + 1, h)
    u0, u1 = max(u - r, 0), min(u + r + 1, w)
    crop = depth[v0:v1, u0:u1].astype(float).copy()
    # mask pixels outside the disk so the support region is legible
    yy, xx = np.mgrid[v0:v1, u0:u1]
    crop[(yy - v) ** 2 + (xx - u) ** 2 > r * r] = np.nan
    im = ax.imshow(crop, origin="upper", extent=[u0, u1, v1, v0], cmap="viridis")
    ax.scatter([u + 0.5], [v + 0.5], c="red", s=40, marker="+", linewidths=1.5)
    ax.add_patch(plt.Circle((u + 0.5, v + 0.5), r, fill=False, ec="white", lw=1))
    ax.set_title(f"depth patch  r={r}px", fontsize=9)
    ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, label="m")


def _panel_hist(ax, dpatch, sw, kd):
    y = np.log(dpatch)
    ax.hist(y, bins=24, weights=sw, density=True, color="0.8", edgecolor="0.6")
    xs = np.linspace(y.min() - 0.1, y.max() + 0.1, 400)
    (m0, m1), (s0, s1), (p0, p1) = kd.modes, kd.sigmas, kd.weights
    mix = np.zeros_like(xs)
    for mu, s, p, c, lab, ls in [
        (np.log(m0), s0, p0, ANCHOR_C, f"mode0 (anchor) {m0:.2f}m  w={p0:.2f}", "-"),
        (np.log(m1), s1, p1, FREE_C, f"mode1 (free) {m1:.2f}m  w={p1:.2f}", "--"),
    ]:
        g = p * _gauss(xs, mu, s)
        mix += g
        ax.plot(xs, g, color=c, ls=ls, lw=1.8, label=lab)
        ax.axvline(mu, color=c, ls=":", lw=1, alpha=0.7)
    ax.plot(xs, mix, color="0.3", lw=1, alpha=0.7, label="mixture")
    ax.set_xlabel("log depth", fontsize=9)
    ax.set_ylabel("weighted density", fontsize=9)
    ax.set_title(f"ambiguity={_ambiguity(kd):.3f}", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper right")


def _panel_rgb(ax, images_dir, name, v, u, r):
    p = Path(images_dir) / name
    if not p.exists():
        p = next((q for q in Path(images_dir).rglob(Path(name).name)), None)
    if p is None or not Path(p).exists():
        ax.text(0.5, 0.5, "no RGB", ha="center", va="center"); ax.axis("off"); return
    img = plt.imread(p)
    pad = 2 * r
    h, w = img.shape[:2]
    v0, v1 = max(v - pad, 0), min(v + pad + 1, h)
    u0, u1 = max(u - pad, 0), min(u + pad + 1, w)
    ax.imshow(img[v0:v1, u0:u1], extent=[u0, u1, v1, v0])
    ax.scatter([u + 0.5], [v + 0.5], c="red", s=40, marker="+", linewidths=1.5)
    ax.add_patch(plt.Circle((u + 0.5, v + 0.5), r, fill=False, ec="yellow", lw=1))
    ax.set_title("RGB", fontsize=9)
    ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--sensor", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("-n", "--num", type=int, default=8)
    ap.add_argument("--rank", choices=["ambiguity", "random", "all"], default="ambiguity")
    ap.add_argument("--images_dir", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with pycolmap.Database.open(args.db) as db:
        images = db.read_all_images()
        cameras = {c.camera_id: c for c in db.read_all_cameras()}
        im = _find_image(images, args.image)
        cam = cameras[im.camera_id]
        kps = np.asarray(db.read_keypoints(im.image_id))

    conn = sqlite3.connect(args.db)
    try:
        meta = schema.read_meta(conn, args.sensor)
        if meta.num_modes != 2:
            raise SystemExit(f"viz expects a 2-mode sensor, {args.sensor} has K={meta.num_modes}")
        rows = schema.read_depths_for_image(conn, im.image_id, args.sensor, 2)
    finally:
        conn.close()

    depth = DepthSource(meta.dump_dir).load(im.name, (cam.height, cam.width)).estimated_depth
    c = meta.method_params.get("patch_scale", 4.0)
    r_min = meta.method_params.get("r_min", 2)

    idxs = sorted(rows)
    if args.rank == "ambiguity":
        idxs = sorted(idxs, key=lambda i: _ambiguity(rows[i]), reverse=True)[:args.num]
    elif args.rank == "random":
        idxs = list(np.random.default_rng(args.seed).choice(idxs, min(args.num, len(idxs)), replace=False))
    print(f"{im.name}: {len(rows)} keypoints, showing {len(idxs)} ({args.rank})")

    ncols = 3 if args.images_dir else 2
    fig, axes = plt.subplots(len(idxs), ncols, figsize=(ncols * 3.4, len(idxs) * 2.7),
                             squeeze=False, constrained_layout=True)
    for row, i in enumerate(idxs):
        kd = rows[i]
        u, v = int(np.floor(kps[i, 0])), int(np.floor(kps[i, 1]))
        u = min(max(u, 0), cam.width - 1); v = min(max(v, 0), cam.height - 1)
        detA = kps[i, 2] * kps[i, 5] - kps[i, 3] * kps[i, 4]
        sigma_s = np.sqrt(max(detA, 0.0))
        r = max(int(round(c * sigma_s)), r_min)

        dpatch, dv, du = _patch(depth, v, u, r)
        ok = np.isfinite(dpatch) & (dpatch > 0)
        sw = np.exp(-(dv[ok] ** 2 + du[ok] ** 2) / (2.0 * max(sigma_s, _TINY) ** 2))

        col = 0
        if args.images_dir:
            _panel_rgb(axes[row, col], args.images_dir, im.name, v, u, r); col += 1
        _panel_depth(axes[row, col], depth, v, u, r); col += 1
        _panel_hist(axes[row, col], dpatch[ok], sw, kd)
        axes[row, 0].set_ylabel(f"kp {i}", fontsize=9)

    out = args.out or Path("plots") / f"gmm_patch_{Path(im.name).stem}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
