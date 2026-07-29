"""Synthetic round-trip tests for the shared Sim(3) alignment: generate points,
apply a known (s, R, t), recover it — kills the whole class of sign/scale/
fit-direction bugs — then corrupt 10% grossly and check the trim loop."""

import numpy as np
import pytest

from depthba.eval.alignment import robust_umeyama_sim3, umeyama_sim3
from depthba.eval.nvs.scannetpp.render import apply_sim3_to_pose, rotmat_to_quat


def random_rotation(rng):
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


@pytest.fixture
def sim3(rng=np.random.default_rng(0)):
    return 2.37, random_rotation(np.random.default_rng(1)), np.array([0.5, -3.0, 1.2])


def test_umeyama_roundtrip(sim3):
    rng = np.random.default_rng(2)
    s, R, t = sim3
    p = rng.standard_normal((100, 3))
    q = (s * (R @ p.T)).T + t
    s_hat, R_hat, t_hat = umeyama_sim3(p, q)
    np.testing.assert_allclose(s_hat, s, atol=1e-10)
    np.testing.assert_allclose(R_hat, R, atol=1e-10)
    np.testing.assert_allclose(t_hat, t, atol=1e-9)


def test_umeyama_without_scale(sim3):
    rng = np.random.default_rng(3)
    _, R, t = sim3
    p = rng.standard_normal((50, 3))
    q = (R @ p.T).T + t
    s_hat, R_hat, t_hat = umeyama_sim3(p, q, with_scale=False)
    assert s_hat == 1.0
    np.testing.assert_allclose(R_hat, R, atol=1e-10)
    np.testing.assert_allclose(t_hat, t, atol=1e-9)


def test_robust_recovers_under_gross_outliers(sim3):
    rng = np.random.default_rng(4)
    s, R, t = sim3
    p = rng.standard_normal((200, 3))
    q = (s * (R @ p.T)).T + t
    corrupt = rng.choice(len(p), size=20, replace=False)  # 10%, grossly wrong
    q[corrupt] += 50.0 * rng.standard_normal((len(corrupt), 3))
    s_hat, R_hat, t_hat, inliers = robust_umeyama_sim3(p, q)
    assert not inliers[corrupt].any()
    assert inliers.sum() >= 150
    np.testing.assert_allclose(s_hat, s, atol=1e-8)
    np.testing.assert_allclose(R_hat, R, atol=1e-8)
    np.testing.assert_allclose(t_hat, t, atol=1e-8)


def test_apply_sim3_to_pose(sim3):
    rng = np.random.default_rng(5)
    s, R, t = sim3
    R_cw = random_rotation(rng)
    C = rng.standard_normal(3)
    w2c = np.eye(4)
    w2c[:3, :3] = R_cw
    w2c[:3, 3] = -R_cw @ C
    new = apply_sim3_to_pose(w2c, s, R, t)
    # New center is the sim3-mapped center; orientation composes without scale.
    new_C = -new[:3, :3].T @ new[:3, 3]
    np.testing.assert_allclose(new_C, s * R @ C + t, atol=1e-9)
    np.testing.assert_allclose(new[:3, :3], R_cw @ R.T, atol=1e-10)
    np.testing.assert_allclose(new[:3, :3] @ new[:3, :3].T, np.eye(3), atol=1e-10)


def test_rotmat_quat_roundtrip():
    rng = np.random.default_rng(6)
    for _ in range(20):
        R = random_rotation(rng)
        np.testing.assert_allclose(quat_to_rotmat(rotmat_to_quat(R)), R,
                                   atol=1e-10)
