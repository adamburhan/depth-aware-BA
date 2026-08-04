"""unpack_amb3r round-trip on a synthetic npz: known depth wrapped to world
point maps must come back exactly, with images/intrinsics/bundles written
and unmapped frames skipped."""

import json

import numpy as np
import pytest

from depthba.preprocess import unpack_amb3r


T, H, W = 3, 6, 8


def _random_pose(rng):
    """Random cam2world SE(3) via Gram-Schmidt."""
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    P = np.eye(4)
    P[:3, :3] = Q
    P[:3, 3] = rng.normal(size=3)
    return P


@pytest.fixture
def synthetic(tmp_path):
    rng = np.random.default_rng(0)
    depth = rng.uniform(1.0, 10.0, size=(T, H, W)).astype(np.float64)

    # back-project on a unit-focal grid, wrap cam -> world with random poses
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(u - W / 2), (v - H / 2), np.ones_like(u)], -1).astype(np.float64)
    pts_cam = rays[None] * depth[..., None]
    pose = np.stack([_random_pose(rng) for _ in range(T)])
    R, t = pose[:, :3, :3], pose[:, :3, 3]
    pts = np.einsum("nij,nhwj->nhwi", R, pts_cam) + t[:, None, None, :]

    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()
    for i in range(T):
        (rgb_dir / f"frame_{i:03d}.jpg").touch()  # names only; content unused

    npz = tmp_path / "results.npz"
    np.savez(
        npz,
        pts=pts, pose=pose,
        conf=np.full((T, H, W), 0.5, np.float32),
        sky_mask=np.zeros((T, H, W), bool),
        images=rng.uniform(-1, 1, size=(T, 3, H, W)).astype(np.float32),
        unmapped_frames=np.array([1]),
        intrinsics=np.array([[100.0, 0, W / 2], [0, 100.0, H / 2], [0, 0, 1]]),
    )
    return npz, tmp_path / "out", rgb_dir, depth


def test_roundtrip(synthetic):
    npz, out, rgb_dir, depth = synthetic
    unpack_amb3r(npz, out, rgb_dir)

    # unmapped frame 1: image yes, bundle no
    assert sorted(p.name for p in (out / "images").iterdir()) == [
        "frame_000.png", "frame_001.png", "frame_002.png"]
    assert sorted(p.name for p in (out / "depth_bundles").iterdir()) == [
        "frame_000.npz", "frame_002.npz"]

    for i in (0, 2):
        b = np.load(out / "depth_bundles" / f"frame_{i:03d}.npz")
        np.testing.assert_allclose(b["estimated_depth"], depth[i], rtol=1e-5)
        assert b["estimated_depth"].dtype == np.float32
        assert set(b.files) == {"estimated_depth", "confidence", "sky_mask"}

    k = json.loads((out / "intrinsics.json").read_text())
    assert (k["fx"], k["fy"], k["cx"], k["cy"]) == (100.0, 100.0, W / 2, H / 2)


def test_name_count_mismatch(synthetic, tmp_path):
    npz, out, rgb_dir, _ = synthetic
    (rgb_dir / "frame_999.jpg").touch()
    with pytest.raises(AssertionError, match="images vs"):
        unpack_amb3r(npz, out, rgb_dir)
