"""End-to-end ingest tests: synthetic COLMAP db + dump -> attach -> read back."""

import sqlite3

import numpy as np
import pytest

import pycolmap

from depthba.config import AttachConfig
from depthba.depth import schema
from depthba.depth.attach_depths import check_measurements, run
from depthba.depth.extractors import DepthMeasurements
from depthba.depth.source import DepthBundle, save_bundle

H, W = 5, 7
IMAGES = {"a.JPG": [(0.4, 0.0), (1.0, 0.0), (6.9, 4.9)], "b.JPG": [(2.0, 3.0)]}


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "database.db"
    db = pycolmap.Database.open(path)
    cam = pycolmap.Camera(model="PINHOLE", width=W, height=H, params=[1.0, 1.0, 3.5, 2.5])
    camera_id = db.write_camera(cam)
    for name, xys in IMAGES.items():
        image_id = db.write_image(pycolmap.Image(name=name, camera_id=camera_id))
        kps = np.zeros((len(xys), 6), np.float32)
        kps[:, :2] = xys
        db.write_keypoints(image_id, kps)
    db.close()
    return path


@pytest.fixture
def dump_dir(tmp_path):
    d = tmp_path / "dump"
    for offset, name in enumerate(IMAGES):
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        depth = u[None, :] + 1000.0 * v[:, None] + 1.0 + 100000.0 * offset
        save_bundle(
            d / f"{name.rsplit('.', 1)[0]}.npz",
            DepthBundle(
                estimated_depth=depth,
                means=None, weights=None, sigmas=None,
                confidence=None, sky_mask=None,
            ),
        )
    return d


CONFIG = AttachConfig(sensor="depthpro", method="unimodal", sigma_space=None)


def test_ingest_round_trip(db_path, dump_dir):
    run(CONFIG, db_path, dump_dir)
    conn = sqlite3.connect(db_path)
    ids = dict(conn.execute("SELECT name, image_id FROM images"))
    meta = schema.read_meta(conn, "depthpro")
    assert meta.num_modes == 1
    assert meta.method == "unimodal"
    assert meta.has_confidence is False
    assert meta.sigma_space is None
    assert schema.count_rows(conn, "depthpro") == 4

    rows = schema.read_depths_for_image(conn, ids["a.JPG"], "depthpro", meta.num_modes)
    # image a: pixels (0,0), (1,0), (6,4) of depth map u + 1000*v + 1
    assert [rows[i].estimated_depth for i in range(3)] == [1.0, 2.0, 4007.0]
    rows_b = schema.read_depths_for_image(conn, ids["b.JPG"], "depthpro", meta.num_modes)
    assert rows_b[0].estimated_depth == 100000.0 + 2.0 + 3000.0 + 1.0
    assert rows_b[0].sigmas is None and rows_b[0].is_sky is None
    conn.close()


def test_reingest_needs_force(db_path, dump_dir):
    run(CONFIG, db_path, dump_dir)
    with pytest.raises(ValueError, match="--force"):
        run(CONFIG, db_path, dump_dir)
    run(CONFIG, db_path, dump_dir, force=True)  # succeeds, rows replaced not doubled
    conn = sqlite3.connect(db_path)
    assert schema.count_rows(conn, "depthpro") == 4
    conn.close()


def test_missing_bundle_fails_before_writing(db_path, dump_dir):
    (dump_dir / "b.npz").unlink()
    with pytest.raises(FileNotFoundError, match="b"):
        run(CONFIG, db_path, dump_dir)
    conn = sqlite3.connect(db_path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    if "depthba_keypoint_depths" in tables:
        assert schema.count_rows(conn, "depthpro") == 0
    conn.close()


def test_sigma_space_mismatch_rejected(db_path, dump_dir):
    bad = AttachConfig(sensor="s", method="unimodal", sigma_space="log")
    with pytest.raises(ValueError, match="sigma_space"):
        run(bad, db_path, dump_dir)
    # transaction rolled back: no meta row, no depth rows
    conn = sqlite3.connect(db_path)
    assert schema.list_sensors(conn) == []
    assert schema.count_rows(conn, "s") == 0
    conn.close()


def test_extractor_row_count_enforced(db_path, dump_dir, monkeypatch):
    """Row i of DepthMeasurements is keypoint i; a filtering extractor would
    silently misattribute every row after the drop. The count check is the
    only guard on that positional identity."""
    from depthba.depth import extractors

    def dropping(bundle, keypoints, params):
        m = extractors.unimodal.extract(bundle, keypoints, params)
        return DepthMeasurements(
            modes=m.modes[:-1],
            weights=m.weights[:-1],
            estimated_depth=m.estimated_depth[:-1],
            sigmas=None,
            confidence=None,
            is_sky=None,
        )

    monkeypatch.setitem(extractors.EXTRACTORS, "dropping", dropping)
    cfg = AttachConfig(sensor="s2", method="dropping")
    with pytest.raises(ValueError, match="measurements for"):
        run(cfg, db_path, dump_dir)
    conn = sqlite3.connect(db_path)
    assert schema.count_rows(conn, "s2") == 0  # transaction rolled back
    conn.close()


def make_measurements(**overrides):
    fields = dict(
        modes=np.array([[2.0], [3.0]], np.float32),
        weights=np.ones((2, 1), np.float32),
        estimated_depth=np.array([2.0, 3.0], np.float32),
        sigmas=None,
        confidence=None,
        is_sky=None,
    )
    fields.update(overrides)
    return DepthMeasurements(**fields)


def test_check_measurements():
    check_measurements("img", make_measurements())  # clean passes
    with pytest.raises(ValueError, match="modes"):
        check_measurements("img", make_measurements(modes=np.float32([[2.0], [-1.0]])))
    with pytest.raises(ValueError, match="estimated_depth"):
        check_measurements(
            "img", make_measurements(estimated_depth=np.float32([np.inf, 3.0]))
        )
    with pytest.raises(ValueError, match="sum above 1"):
        check_measurements("img", make_measurements(weights=np.full((2, 1), 1.2, np.float32)))
    # deficit is legitimate (sky mass outside the stored K): sums < 1 pass
    check_measurements("img", make_measurements(weights=np.full((2, 1), 0.7, np.float32)))
    with pytest.raises(ValueError, match="sigmas"):
        check_measurements("img", make_measurements(sigmas=np.zeros((2, 1), np.float32)))
    # sky keypoints are exempt: fabricated values on flagged rows pass
    check_measurements(
        "img",
        make_measurements(
            modes=np.float32([[2.0], [-5.0]]),
            estimated_depth=np.float32([2.0, -5.0]),
            is_sky=np.array([False, True]),
        ),
    )
