"""ScanNet++ NVS eval, pose side: express the DSLR test cameras in our SfM frame.

For ScanNet++, the eval for nvs works as follows:
- The test images are taken from DSLR, iphone used for training.
- DSLR images are transformed into iPhone image space. This is exact and
  depth-free because the DSLR image and the synthetic iPhone view share the
  same optical center — a pure camera-model/rotation change, no parallax.
- The DSLR poses are in ScanNet++'s scan-aligned COLMAP frame; the Sim(3)
  into our SfM frame is estimated from iPhone poses (present in both frames:
  our SfM and ScanNet++'s iphone/colmap), then applied to the DSLR poses.
- With the new poses and intrinsics, we can render the images using our model
  and compare with the ground-truth DSLR images (can use scannetpp eval.nvs).

NOTE (convention traps): nerfstudio poses (dslr/nerfstudio/transforms.json)
are cam-to-world in OpenGL axes — everything here is COLMAP world-to-cam in
OpenCV axes. Align camera *centers* C = -R^T t, never raw w2c translations.

  python -m depthba.eval.nvs.scannetpp.render \
      --our_sfm $EXP/scannetpp_amb3r/<scene>/sfm_<arm>/0 \
      --iphone_colmap $DATA/<scene>/iphone/colmap \
      --dslr_colmap $DATA/<scene>/dslr/colmap \
      --out $EXP/.../dslr_in_sfm_<arm>

Writes a COLMAP text model (cameras.txt/images.txt/points3D.txt) holding every
DSLR frame posed in our SfM frame, all assigned our arm's iPhone camera. The
robust-fit RMS it prints (converted to meters via the fitted scale) doubles as
the arm's trajectory-accuracy number on this scene.
"""

import argparse
from pathlib import Path

import numpy as np

import pycolmap

from depthba.eval.alignment import robust_umeyama_sim3


def camera_centers(rec: pycolmap.Reconstruction) -> dict[str, np.ndarray]:
    """{image stem: camera center C = -R^T t}. Keyed by stem because our SfM
    ran on the AMB3R re-exported .png frames while ScanNet++ names them .jpg."""
    centers = {}
    for image in rec.images.values():
        cfw = image.cam_from_world
        if callable(cfw):
            cfw = cfw()
        R = np.asarray(cfw.rotation.matrix())
        t = np.asarray(cfw.translation)
        centers[Path(image.name).stem] = -R.T @ t
    return centers


def w2c_matrix(image) -> np.ndarray:
    cfw = image.cam_from_world
    if callable(cfw):
        cfw = cfw()
    w2c = np.eye(4)
    w2c[:3, :3] = np.asarray(cfw.rotation.matrix())
    w2c[:3, 3] = np.asarray(cfw.translation)
    return w2c


def apply_sim3_to_pose(w2c: np.ndarray, s: float, R: np.ndarray,
                       t: np.ndarray) -> np.ndarray:
    """World-to-cam 4x4 after remapping the world by x -> s R x + t.

    Compose on the cam-to-world side, strip the scale so the rotation block
    stays orthonormal, invert back: the new center is s R C + t and the new
    orientation is R @ R_c2w.
    """
    S = np.eye(4)
    S[:3, :3] = s * R
    S[:3, 3] = t
    c2w = S @ np.linalg.inv(w2c)
    c2w[:3, :3] /= s
    return np.linalg.inv(c2w)


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """(qw, qx, qy, qz), COLMAP order, via the largest-component method."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s]
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        q = [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s]
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        q = [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        q = [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--our_sfm", required=True,
                    help="our COLMAP model dir, e.g. .../sfm_baseline/0")
    ap.add_argument("--iphone_colmap", required=True,
                    help="ScanNet++ <scene>/iphone/colmap (pose-only GT model)")
    ap.add_argument("--dslr_colmap", required=True,
                    help="ScanNet++ <scene>/dslr/colmap")
    ap.add_argument("--out", required=True,
                    help="output dir for the DSLR-in-SfM COLMAP text model")
    args = ap.parse_args()

    ours = pycolmap.Reconstruction(args.our_sfm)
    iphone = pycolmap.Reconstruction(args.iphone_colmap)

    c_ours = camera_centers(ours)
    c_data = camera_centers(iphone)
    common = sorted(set(c_ours) & set(c_data))
    print(f"iphone frames: ours {len(c_ours)}, scannetpp {len(c_data)}, "
          f"common {len(common)}")
    if len(common) < 3:
        raise SystemExit("fewer than 3 common frames — check image naming")

    # Fit dataset -> ours so the transform applies directly to the DSLR poses.
    p = np.array([c_data[n] for n in common])
    q = np.array([c_ours[n] for n in common])
    s, R, t, inliers = robust_umeyama_sim3(p, q)
    res = np.linalg.norm(q - (s * (R @ p.T)).T - t, axis=1)
    rms = float(np.sqrt(np.mean(res[inliers] ** 2)))
    # Residuals live in SfM units; the dataset frame is metric, so /s -> meters.
    print(f"sim3 scannetpp->sfm: scale {s:.6f}, inliers {int(inliers.sum())}/"
          f"{len(common)}, rms {rms:.4f} sfm-units = {rms / s:.4f} m")

    if len(ours.cameras) != 1:
        raise SystemExit(f"expected 1 camera in our SfM, got {len(ours.cameras)}")
    cam = next(iter(ours.cameras.values()))

    dslr = pycolmap.Reconstruction(args.dslr_colmap)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    params = " ".join(repr(float(v)) for v in cam.params)
    (out / "cameras.txt").write_text(
        f"1 {cam.model.name} {cam.width} {cam.height} {params}\n")

    lines = []
    images = sorted(dslr.images.values(), key=lambda im: im.name)
    for i, image in enumerate(images, start=1):
        w2c = apply_sim3_to_pose(w2c_matrix(image), s, R, t)
        qw, qx, qy, qz = rotmat_to_quat(w2c[:3, :3])
        tx, ty, tz = w2c[:3, 3]
        lines.append(f"{i} {qw!r} {qx!r} {qy!r} {qz!r} "
                     f"{tx!r} {ty!r} {tz!r} 1 {image.name}")
        lines.append("")  # empty POINTS2D line
    (out / "images.txt").write_text("\n".join(lines) + "\n")
    (out / "points3D.txt").write_text("")

    print(f"wrote {len(images)} DSLR poses (camera: our {cam.model.name} "
          f"{cam.width}x{cam.height}) to {out}")


if __name__ == "__main__":
    main()
