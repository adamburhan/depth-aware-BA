"""Live-fit patch-GMM analysis: run the CURRENT gmm_patch extractor on one
image's keypoints (no attached DB rows needed -- stage-1 database only) and
visualize what the fit saw and what it produced.

Outputs two figures:
  1. gallery: per-keypoint RGB patch / depth patch / weighted histogram + fit
  2. population: separation, weight, sigma distributions + spatial map of
     bimodal keypoints over the image

    uv run python scripts/viz_gmm_live.py --db data/Meetingroom/database.db \
        --data data/Meetingroom --image 000001.png -n 10
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycolmap

from depthba.depth.extractors import _pixel_indices
from depthba.depth.extractors.gmm_patch import _disk, extract
from depthba.depth.source import DepthSource

_TINY = 1e-12
ANCHOR_C, FREE_C = "#0173B2", "#D55E00"


def _ambiguity(modes, weights):
    m0, m1 = modes
    return float(weights.min() * abs(np.log(max(m0, _TINY) / max(m1, _TINY))))


def _gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _patch(depth, v, u, r):
    h, w = depth.shape
    dv, du = _disk(r)
    vv, uu = v + dv, u + du
    inb = (vv >= 0) & (vv < h) & (uu >= 0) & (uu < w)
    return depth[vv[inb], uu[inb]], dv[inb], du[inb]


def _panel_rgb(ax, img, v, u, r):
    pad = 2 * r
    h, w = img.shape[:2]
    v0, v1 = max(v - pad, 0), min(v + pad + 1, h)
    u0, u1 = max(u - pad, 0), min(u + pad + 1, w)
    ax.imshow(img[v0:v1, u0:u1], extent=[u0, u1, v1, v0])
    ax.scatter([u + 0.5], [v + 0.5], c="red", s=40, marker="+", linewidths=1.5)
    ax.add_patch(plt.Circle((u + 0.5, v + 0.5), r, fill=False, ec="yellow", lw=1))
    ax.set_title("RGB", fontsize=9)
    ax.tick_params(labelsize=7)


def _panel_depth(ax, depth, v, u, r):
    h, w = depth.shape
    v0, v1 = max(v - r, 0), min(v + r + 1, h)
    u0, u1 = max(u - r, 0), min(u + r + 1, w)
    crop = depth[v0:v1, u0:u1].astype(float).copy()
    yy, xx = np.mgrid[v0:v1, u0:u1]
    crop[(yy - v) ** 2 + (xx - u) ** 2 > r * r] = np.nan
    im = ax.imshow(crop, origin="upper", extent=[u0, u1, v1, v0], cmap="viridis")
    ax.scatter([u + 0.5], [v + 0.5], c="red", s=40, marker="+", linewidths=1.5)
    ax.set_title(f"depth patch  r={r}px", fontsize=9)
    ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, label="m")


def _panel_hist(ax, dpatch, sw, modes, sigmas, weights):
    """Weighted histogram + fitted mixture, all in LINEAR meters (fit space)."""
    ax.hist(dpatch, bins=24, weights=sw, density=True, color="0.8", edgecolor="0.6")
    lo, hi = dpatch.min(), dpatch.max()
    pad = 0.05 * (hi - lo + 1e-3)
    xs = np.linspace(lo - pad, hi + pad, 400)
    (m0, m1), (s0, s1), (p0, p1) = modes, sigmas, weights
    mix = np.zeros_like(xs)
    for mu, s, p, c, lab, ls in [
        (m0, s0, p0, ANCHOR_C, f"mode0 (anchor) {m0:.2f}m  w={p0:.2f}", "-"),
        (m1, s1, p1, FREE_C, f"mode1 (free) {m1:.2f}m  w={p1:.2f}", "--"),
    ]:
        g = p * _gauss(xs, mu, s)
        mix += g
        ax.plot(xs, g, color=c, ls=ls, lw=1.8, label=lab)
        ax.axvline(mu, color=c, ls=":", lw=1, alpha=0.7)
    ax.plot(xs, mix, color="0.3", lw=1, alpha=0.7, label="mixture")
    ax.set_xlabel("depth [m]", fontsize=9)
    ax.set_title(f"ambiguity={_ambiguity(modes, weights):.3f}", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="upper right")


def _population_fig(depth, img, kps, dm, wmin, sep_min, out):
    sep = np.abs(dm.modes[:, 0] - dm.modes[:, 1])
    rel_sep = sep / np.maximum(dm.modes[:, 0], _TINY)
    bimodal = (dm.weights.min(1) >= wmin) & (sep > 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.hist(sep[bimodal], bins=40, color=FREE_C, alpha=0.8)
    ax.set_xlabel("|mode1 - mode0| [m]"); ax.set_title(f"separation (bimodal, n={bimodal.sum()})")

    ax = axes[0, 1]
    ax.hist(rel_sep[bimodal], bins=40, color=FREE_C, alpha=0.8)
    ax.axvline(sep_min, color="k", ls="--", lw=1, label=f"sep_min_rel={sep_min}")
    ax.set_xlabel("separation / mode0"); ax.set_title("relative separation")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.hist(dm.weights[bimodal, 1], bins=40, color=FREE_C, alpha=0.8)
    ax.set_xlabel("mode1 weight"); ax.set_title("2nd-mode support")

    ax = axes[1, 0]
    ax.hist(dm.sigmas[bimodal, 0], bins=40, color=ANCHOR_C, alpha=0.6, label="sigma0")
    ax.hist(dm.sigmas[bimodal, 1], bins=40, color=FREE_C, alpha=0.6, label="sigma1")
    ax.set_xlabel("sigma [m]"); ax.set_title("per-mode sigma (bimodal)")
    ax.legend(fontsize=8)

    v_kp, u_kp = _pixel_indices(kps, depth.shape)
    ax = axes[1, 1]
    ax.imshow(img)
    ax.scatter(u_kp[~bimodal], v_kp[~bimodal], s=3, c="lime", alpha=0.5, label="unimodal")
    ax.scatter(u_kp[bimodal], v_kp[bimodal], s=5, c="red", alpha=0.8, label="bimodal")
    ax.set_title(f"keypoints ({100 * bimodal.mean():.1f}% bimodal)")
    ax.legend(fontsize=8, loc="lower right"); ax.axis("off")

    ax = axes[1, 2]
    gy, gx = np.gradient(depth)
    ax.imshow(np.hypot(gx, gy), cmap="magma", vmax=np.percentile(np.hypot(gx, gy), 99))
    ax.scatter(u_kp[bimodal], v_kp[bimodal], s=4, c="cyan", alpha=0.8)
    ax.set_title("bimodal kps over |grad depth|"); ax.axis("off")

    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path,
                    help="sequence dir containing images/ and depth_bundles/")
    ap.add_argument("--image", required=True)
    ap.add_argument("-n", "--num", type=int, default=10)
    ap.add_argument("--rank", choices=["ambiguity", "random"], default="ambiguity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--params", default="{}", help="method_params dict as python literal")
    args = ap.parse_args()
    params = eval(args.params)  # analysis tool; trusted input

    with pycolmap.Database.open(args.db) as db:
        images = {im.name: im for im in db.read_all_images()}
        cameras = {c.camera_id: c for c in db.read_all_cameras()}
        im = images.get(args.image) or images.get(Path(args.image).stem + ".png")
        if im is None:
            raise SystemExit(f"image {args.image!r} not in db")
        cam = cameras[im.camera_id]
        kps = np.asarray(db.read_keypoints(im.image_id))

    bundle = DepthSource(args.data / "depth_bundles").load(im.name, (cam.height, cam.width))
    depth = bundle.estimated_depth
    img = plt.imread(args.data / "images" / im.name)

    dm = extract(bundle, kps, params)
    print(f"{im.name}: {len(kps)} keypoints")

    c = params.get("patch_scale", 4.0)
    r_min = params.get("r_min", 2)
    wmin = params.get("wmin", 0.05)
    sep_min = params.get("sep_min_rel", 0.1)

    amb = np.array([_ambiguity(dm.modes[i], dm.weights[i]) for i in range(len(kps))])
    if args.rank == "ambiguity":
        idxs = np.argsort(-amb)[: args.num]
    else:
        idxs = np.random.default_rng(args.seed).choice(len(kps), args.num, replace=False)

    v_all, u_all = _pixel_indices(kps, depth.shape)
    fig, axes = plt.subplots(len(idxs), 3, figsize=(3 * 3.4, len(idxs) * 2.7),
                             squeeze=False, constrained_layout=True)
    for row, i in enumerate(idxs):
        v, u = int(v_all[i]), int(u_all[i])
        detA = kps[i, 2] * kps[i, 5] - kps[i, 3] * kps[i, 4]
        sigma_s = np.sqrt(max(detA, 0.0))
        r = max(int(round(c * sigma_s)), r_min)

        dpatch, dv, du = _patch(depth, v, u, r)
        ok = np.isfinite(dpatch) & (dpatch > 0)
        sw = np.exp(-(dv[ok] ** 2 + du[ok] ** 2) / (2.0 * max(sigma_s, _TINY) ** 2))

        _panel_rgb(axes[row, 0], img, v, u, r)
        _panel_depth(axes[row, 1], depth, v, u, r)
        _panel_hist(axes[row, 2], dpatch[ok], sw, dm.modes[i], dm.sigmas[i], dm.weights[i])
        axes[row, 0].set_ylabel(f"kp {i}", fontsize=9)

    stem = Path(im.name).stem
    out = Path("plots") / f"gmm_live_{stem}_{args.rank}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

    _population_fig(depth, img, kps, dm, wmin, sep_min,
                    Path("plots") / f"gmm_live_{stem}_population.png")


if __name__ == "__main__":
    main()
