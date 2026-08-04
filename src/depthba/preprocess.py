"""amb3r npz -> canonical images + intrinsics.json + per-frame DepthBundle npzs.

The npz is amb3r's own resized working set, so images, intrinsics and depth
land on one grid (canonical 518x392) with no resampling. Depth is recovered
by undoing the cam2world wrap on the point maps.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image


def unpack_amb3r(npz_path: Path, out_dir: Path, rgb_dir: Path) -> None:
    names = sorted(p for p in Path(rgb_dir).iterdir() if p.suffix.lower() == ".jpg")

    out = Path(out_dir)
    (out / "depth_bundles").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)

    d = np.load(npz_path)
    pts, pose = d["pts"], d["pose"]              # (T,H,W,3), (T,4,4) cam2world
    conf, sky = d["conf"], d["sky_mask"]         # (T,H,W)  conf = sig scale [0,1)
    imgs = d["images"]                           # (T,3,H,W) in [-1,1], RGB
    unmapped = set(d["unmapped_frames"].tolist())
    intrinsics = d["intrinsics"]                 # (3,3) K on the canonical grid
    with open(out / "intrinsics.json", "w") as f:
        json.dump({
            "fx": float(intrinsics[0, 0]),
            "fy": float(intrinsics[1, 1]),
            "cx": float(intrinsics[0, 2]),
            "cy": float(intrinsics[1, 2]),
        }, f, indent=2)
    T = pts.shape[0]

    assert len(names) == T, f"{len(names)} images vs {T} npz frames"

    # ---- depth: undo the cam2world wrap ----
    R, t = pose[:, :3, :3], pose[:, :3, 3]
    pts_cam = np.einsum("nji,nhwj->nhwi", R, pts - t[:, None, None, :])
    depth = np.ascontiguousarray(pts_cam[..., 2]).astype(np.float32)

    # ---- per-scene sanity ----
    valid = ~sky & (conf > 1e-4)
    pos_frac = (depth[valid] > 0).mean()
    assert pos_frac > 0.99, f"depth positivity {pos_frac:.4f} — pose convention?"
    assert np.isfinite(depth).all(), "non-finite depth"

    # ---- images: [-1,1] CHW float -> [0,255] HWC uint8, PNG via PIL (RGB-safe) ----
    imgs_u8 = np.clip((imgs.transpose(0, 2, 3, 1) + 1.0) / 2.0 * 255.0 + 0.5,
                      0, 255).astype(np.uint8)

    n_saved = 0
    for i in range(T):
        stem = Path(names[i].name).stem
        Image.fromarray(imgs_u8[i]).save(out / "images" / f"{stem}.png")
        if i in unmapped:
            continue                             # image saved (COLMAP can use it); no bundle
        np.savez_compressed(
            out / "depth_bundles" / f"{stem}.npz",
            estimated_depth=depth[i],
            confidence=conf[i].astype(np.float32),
            sky_mask=sky[i],
        )
        n_saved += 1

    print(f"{npz_path}: {T} images, {n_saved} bundles, {len(unmapped)} unmapped, "
          f"depth pos frac {pos_frac:.4f}")
