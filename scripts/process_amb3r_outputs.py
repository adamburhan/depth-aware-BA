"""Process AMB3R npz -> per-frame DepthBundle npz files + canonical PNG images."""
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

def name_for_index(i: int) -> str:
    return f"{i+1:06d}"          # T&T convention: npz idx i <-> file 000001-based

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out = Path(args.out)
    (out / 'depth_bundles').mkdir(parents=True, exist_ok=True)
    (out / 'images').mkdir(parents=True, exist_ok=True)

    d = np.load(args.npz)
    pts, pose = d['pts'], d['pose']              # (T,H,W,3), (T,4,4) cam2world
    conf, sky = d['conf'], d['sky_mask']         # (T,H,W)  conf = sig scale [0,1)
    imgs = d['images']                           # (T,3,H,W) in [-1,1], RGB
    unmapped = set(d['unmapped_frames'].tolist())
    T = pts.shape[0]

    # ---- depth: undo the cam2world wrap ----
    R, t = pose[:, :3, :3], pose[:, :3, 3]
    pts_cam = np.einsum('nji,nhwj->nhwi', R, pts - t[:, None, None, :])
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
        stem = name_for_index(i)
        Image.fromarray(imgs_u8[i]).save(out / 'images' / f'{stem}.png')
        if i in unmapped:
            continue                             # image saved (COLMAP can use it); no bundle
        np.savez_compressed(
            out / 'depth_bundles' / f'{stem}.npz',
            estimated_depth=depth[i],
            confidence=conf[i].astype(np.float32),
            sky_mask=sky[i],
        )
        n_saved += 1

    print(f"{args.npz}: {T} images, {n_saved} bundles, {len(unmapped)} unmapped, "
          f"depth pos frac {pos_frac:.4f}")

if __name__ == '__main__':
    main()