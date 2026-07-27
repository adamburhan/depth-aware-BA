import pycolmap
import viser
import numpy as np
import re
import time
import matplotlib

BASE = "/Users/adam/repos/depth-aware-BA/Example/Meetingroom"
GT_DIR = f"{BASE}/gt"  # adjust: wherever you copied the GT files locally

VARIANTS = {
    "baseline": f"{BASE}/sfm_baseline/0",
    "gmm":      f"{BASE}/sfm_gmm/0",
    "unimodal": f"{BASE}/sfm_unimodal/0",
}
COLORS = {
    "baseline": (255, 80, 80),
    "gmm":      (80, 255, 80),
    "unimodal": (80, 80, 255),
}


# ---------- GT parsing ----------

def load_colmap_sfm_log(path):
    """Parse T&T .log trajectory file -> dict {view_id: 4x4 cam-to-world np.array}."""
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    poses = {}
    i = 0
    while i < len(lines):
        header = lines[i].split()
        view_id = int(header[0])
        mat = np.array([
            [float(x) for x in lines[i + 1].split()],
            [float(x) for x in lines[i + 2].split()],
            [float(x) for x in lines[i + 3].split()],
            [float(x) for x in lines[i + 4].split()],
        ])
        poses[view_id] = mat
        i += 5
    return poses


def load_trans_txt(path):
    with open(path, "r") as f:
        vals = [float(x) for x in f.read().split()]
    return np.array(vals).reshape(4, 4)


gt_poses_raw = load_colmap_sfm_log(f"{GT_DIR}/Meetingroom_COLMAP_SfM.log")
trans = load_trans_txt(f"{GT_DIR}/Meetingroom_trans.txt")

# apply trans.txt to bring GT poses into the laser-scan / point-cloud frame
gt_poses = {vid: trans @ mat for vid, mat in gt_poses_raw.items()}
gt_centers = {vid: mat[:3, 3] for vid, mat in gt_poses.items()}

print(f"Loaded {len(gt_centers)} GT camera poses")


# ---------- match your recon image names to GT view ids ----------

def extract_id(name):
    """Pull the leading/only integer out of a filename, e.g. '000369.jpg' -> 369."""
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def umeyama_alignment(src, dst):
    n, dim = src.shape
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / n
    scale = np.trace(np.diag(D) @ S) / var_src
    t = mu_dst - scale * R @ mu_src
    return scale, R, t


def apply_transform(points, scale, R, t):
    return (scale * (R @ points.T).T) + t


def error_colormap(values, vmax, cmap_name="turbo"):
    """values -> uint8 RGB, clipped to [0, vmax] so outliers don't wash out the scale."""
    norm = np.clip(values / max(vmax, 1e-9), 0.0, 1.0)
    rgba = matplotlib.colormaps[cmap_name](norm)
    return (rgba[:, :3] * 255).astype(np.uint8)


# ---------- load variants, align each to GT ----------

server = viser.ViserServer()

# render GT point cloud as static reference (subsample if huge)
import trimesh
gt_cloud = trimesh.load(f"{GT_DIR}/Meetingroom.ply")
gt_xyz = np.array(gt_cloud.vertices)
if len(gt_xyz) > 2_000_000:
    idx = np.random.choice(len(gt_xyz), 2_000_000, replace=False)
    gt_xyz = gt_xyz[idx]
gt_rgb = (np.array(gt_cloud.visual.vertex_colors)[:, :3]
          if hasattr(gt_cloud.visual, "vertex_colors") else
          np.full((len(gt_xyz), 3), 180, dtype=np.uint8))
if len(gt_rgb) != len(gt_xyz):
    gt_rgb = gt_rgb[idx]

server.scene.add_point_cloud(
    name="/gt/points", points=gt_xyz, colors=gt_rgb, point_size=0.003,
)

# ---------- pass 1: load + align every variant, collect errors, before making any handles ----------
# (need pooled error ranges across variants for a shared, comparable color scale)

loaded = {}
all_point_errors, all_cam_errors = [], []

for name, path in VARIANTS.items():
    recon = pycolmap.Reconstruction(path)

    src_pts, dst_pts = [], []
    for img in recon.images.values():
        vid = extract_id(img.name)
        if vid is not None and vid in gt_centers:
            world_from_cam = img.cam_from_world().inverse()
            src_pts.append(world_from_cam.translation)
            dst_pts.append(gt_centers[vid])

    if len(src_pts) < 3:
        print(f"WARNING: only {len(src_pts)} matches for {name} — check extract_id() "
              f"against your actual image filenames")
        continue
    if len(src_pts) < 0.8 * len(recon.images):
        print(f"WARNING: {name} only matched {len(src_pts)}/{len(recon.images)} images to "
              f"GT — alignment is based on a partial subset")

    src_pts, dst_pts = np.array(src_pts), np.array(dst_pts)
    scale, R, t = umeyama_alignment(src_pts, dst_pts)
    print(f"{name}: aligned to GT using {len(src_pts)} images, scale={scale:.4f}")

    xyz = np.array([p.xyz for p in recon.points3D.values()])
    rgb = np.array([p.color for p in recon.points3D.values()], dtype=np.uint8)
    point_err = np.array([
        p.error if p.has_error else np.nan for p in recon.points3D.values()
    ])
    xyz_aligned = apply_transform(xyz, scale, R, t)

    cam_names, cam_ids, cam_centers_aligned, cam_wxyz_aligned, cam_err = [], [], [], [], []
    for img in recon.images.values():
        world_from_cam = img.cam_from_world().inverse()
        center_aligned = apply_transform(world_from_cam.translation[None, :], scale, R, t)[0]
        rot_aligned = R @ world_from_cam.rotation.matrix()
        wxyz_aligned = pycolmap.Rotation3d(rot_aligned).quat[[3, 0, 1, 2]]

        vid = extract_id(img.name)
        err = np.linalg.norm(center_aligned - gt_centers[vid]) if vid in gt_centers else np.nan

        cam_names.append(img.name)
        cam_ids.append(img.camera_id)
        cam_centers_aligned.append(center_aligned)
        cam_wxyz_aligned.append(wxyz_aligned)
        cam_err.append(err)

    loaded[name] = dict(
        recon=recon, xyz_aligned=xyz_aligned, rgb=rgb, point_err=point_err,
        cam_names=cam_names, cam_ids=cam_ids, cam_centers=np.array(cam_centers_aligned),
        cam_wxyz=cam_wxyz_aligned, cam_err=np.array(cam_err),
    )
    all_point_errors.append(point_err[np.isfinite(point_err)])
    all_cam_errors.append(np.array(cam_err)[np.isfinite(cam_err)])

# shared color scale (95th percentile pooled across variants) so redness is
# directly comparable between baseline/gmm/unimodal, not renormalized per-variant
point_err_vmax = np.percentile(np.concatenate(all_point_errors), 95) if all_point_errors else 1.0
cam_err_vmax = np.percentile(np.concatenate(all_cam_errors), 95) if all_cam_errors else 1.0
print(f"color scale: point error vmax={point_err_vmax:.4f} (px), "
      f"camera position error vmax={cam_err_vmax:.4f} (m) [95th pct pooled across variants]")

# ---------- pass 2: build scene ----------

for name, d in loaded.items():
    handles = []
    with server.gui.add_folder(name):
        pc_rgb = server.scene.add_point_cloud(
            name=f"/{name}/points_rgb", points=d["xyz_aligned"], colors=d["rgb"],
            point_size=0.005, visible=True,
        )
        finite = np.isfinite(d["point_err"])
        err_colors = np.where(
            finite[:, None],
            error_colormap(np.nan_to_num(d["point_err"]), point_err_vmax),
            np.uint8(60),  # dim gray for points with no error (shouldn't normally happen)
        )
        pc_err = server.scene.add_point_cloud(
            name=f"/{name}/points_err", points=d["xyz_aligned"], colors=err_colors,
            point_size=0.005, visible=False,
        )
        handles += [pc_rgb, pc_err]

        color_mode = server.gui.add_dropdown(
            f"{name} point color", options=("rgb", "reprojection error"), initial_value="rgb"
        )

        @color_mode.on_update
        def _(event, pc_rgb=pc_rgb, pc_err=pc_err):
            is_rgb = event.target.value == "rgb"
            pc_rgb.visible = is_rgb
            pc_err.visible = not is_rgb

        cam_colors = error_colormap(np.nan_to_num(d["cam_err"]), cam_err_vmax)
        for img_name, cam_id, center, wxyz, err, color in zip(
            d["cam_names"], d["cam_ids"], d["cam_centers"], d["cam_wxyz"], d["cam_err"], cam_colors
        ):
            cam = d["recon"].cameras[cam_id]
            frustum_handle = server.scene.add_camera_frustum(
                name=f"/{name}/cameras/{img_name}",
                fov=2 * np.arctan(cam.height / (2 * cam.focal_length_y)),
                aspect=cam.width / cam.height,
                scale=0.05,
                wxyz=wxyz,
                position=center,
                color=tuple(int(c) for c in color) if np.isfinite(err) else (80, 80, 80),
            )
            handles.append(frustum_handle)

        traj_handle = server.scene.add_spline_catmull_rom(
            name=f"/{name}/trajectory", positions=d["cam_centers"],
            color=COLORS[name], line_width=2.0,
        )
        handles.append(traj_handle)

        vis_checkbox = server.gui.add_checkbox(f"{name} visible", initial_value=True)

        @vis_checkbox.on_update
        def _(event, handles=handles, pc_rgb=pc_rgb, pc_err=pc_err, color_mode=color_mode):
            for h in handles:
                if h is pc_rgb or h is pc_err:
                    continue  # keep obeying color_mode instead of forcing both on
                h.visible = event.target.value
            is_rgb = color_mode.value == "rgb"
            pc_rgb.visible = event.target.value and is_rgb
            pc_err.visible = event.target.value and not is_rgb

# GT trajectory too, for visual reference
gt_traj_handle = server.scene.add_spline_catmull_rom(
    name="/gt/trajectory",
    positions=np.array(list(gt_centers.values())),
    color=(255, 255, 0),
    line_width=3.0,
)

print("Point clouds: per-variant dropdown toggles RGB vs. reprojection-error heatmap "
      "(turbo colormap, shared scale across variants).")
print("Camera frustums: colored by aligned position error vs. matched GT camera "
      "(same shared scale); gray = no GT match.")
print("Open the printed URL, e.g. http://localhost:8080")
while True:
    time.sleep(1)
