#!/usr/bin/env python
"""Side-by-side RGB / depth / confidence check for one DepthBundle.
Usage: python viz_bundle.py --npz depth_bundles/000042.npz --img images/000042.png [--out check.png]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')                      # headless cluster
import matplotlib.pyplot as plt
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--npz', required=True)
ap.add_argument('--img', required=True)
ap.add_argument('--out', default=None)
args = ap.parse_args()

d = np.load(args.npz)
depth = d['estimated_depth']
conf  = d['confidence'] if 'confidence' in d.files else None
sky   = d['sky_mask'] if 'sky_mask' in d.files else np.zeros_like(depth, bool)
rgb   = np.asarray(Image.open(args.img))

# robust range from valid pixels only (sky garbage would wreck the colormap otherwise)
valid = ~sky & np.isfinite(depth) & (depth > 0)
vmin, vmax = np.percentile(depth[valid], [2, 98])

depth_vis = np.ma.masked_where(~valid, depth)   # masked pixels render blank

fig, axes = plt.subplots(1, 3 if conf is not None else 2, figsize=(16, 4.5))
axes[0].imshow(rgb);  axes[0].set_title('RGB')
im1 = axes[1].imshow(depth_vis, cmap='turbo', vmin=vmin, vmax=vmax)
axes[1].set_title(f'depth [{vmin:.1f}, {vmax:.1f}] (sky/invalid masked)')
plt.colorbar(im1, ax=axes[1], fraction=0.03)
if conf is not None:
    im2 = axes[2].imshow(conf, cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title('confidence (sig scale)')
    plt.colorbar(im2, ax=axes[2], fraction=0.03)
for ax in axes: ax.axis('off')
plt.tight_layout()

out = args.out or 'bundle_check.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f'saved {out}   depth: min {depth[valid].min():.2f}  med {np.median(depth[valid]):.2f}  max {depth[valid].max():.2f}')
