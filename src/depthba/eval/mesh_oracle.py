"""Mesh depth at keypoints: Sim(3) into the scan frame, then one ray per keypoint."""

from pathlib import Path

import numpy as np

from depthba.eval.alignment import umeyama_sim3


def pose(image):
    cfw = image.cam_from_world
    if callable(cfw):
        cfw = cfw()
    return cfw.rotation.matrix(), np.asarray(cfw.translation, dtype=float)


def centers(rec) -> dict:
    out = {}
    for image in rec.images.values():
        R, t = pose(image)
        out[Path(image.name).stem] = -R.T @ t
    return out


def align_to(rec, gt):
    """Sim(3) taking rec's world frame onto gt's, from common camera centres."""
    c_rec, c_gt = centers(rec), centers(gt)
    common = sorted(c_rec.keys() & c_gt.keys())
    if len(common) < 3:
        raise ValueError(f"only {len(common)} common images -- cannot align")
    scale, R, t = umeyama_sim3(np.array([c_rec[n] for n in common]),
                               np.array([c_gt[n] for n in common]))
    return scale, R, t, len(common)


def live_rays(rec, rows_by_image, sep_min, sim3):
    """Rays for every non-sky keypoint whose two modes are separated.

    Returns (origins, unit dirs, |unnormalised dir|, records), where a record is
    (image_id, point2D_idx, row, z_cam or None). Dividing the along-ray hit
    distance by the third value converts it to camera-frame depth.
    """
    scale, Rs, ts = sim3
    origins, dirs, norms, records = [], [], [], []
    for image in rec.images.values():
        rows = rows_by_image.get(image.image_id)
        if not rows:
            continue
        camera = rec.cameras[image.camera_id]
        if camera.model.name != "PINHOLE":
            raise ValueError(f"expected PINHOLE cameras, got {camera.model.name}")
        fx, fy, cx, cy = camera.params
        R, t = pose(image)
        center = Rs @ (-R.T @ t) * scale + ts
        for idx, p2d in enumerate(image.points2D):
            row = rows.get(idx)
            if row is None or row.is_sky:
                continue
            if abs(np.log(row.modes[1]) - np.log(row.modes[0])) < sep_min:
                continue
            u, v = p2d.xy
            d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
            d_world = Rs @ (R.T @ d_cam)
            origins.append(center)
            dirs.append(d_world / np.linalg.norm(d_world))
            norms.append(np.linalg.norm(d_cam))
            z = None
            if p2d.has_point3D():
                z = float((R @ rec.points3D[p2d.point3D_id].xyz + t)[2])
            records.append((image.image_id, idx, row, z))
    return np.array(origins), np.array(dirs), np.array(norms), records


def cast(mesh_path, origins, dirs, norms):
    """Camera-frame mesh depth per ray; NaN where the ray missed."""
    import trimesh

    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    loc, idx_ray, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    depth = np.full(len(origins), np.nan)
    for r, d in zip(idx_ray, np.linalg.norm(loc - origins[idx_ray], axis=1)):
        if np.isnan(depth[r]) or d < depth[r]:
            depth[r] = d
    return depth / norms
