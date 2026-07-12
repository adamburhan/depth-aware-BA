"""Round-trip tests for depthba.depth.schema against in-memory SQLite."""

import sqlite3

import numpy as np
import pytest

from depthba.depth import schema
from depthba.depth.schema import KeypointDepth, SensorMeta


@pytest.fixture
def conn():
    conn = sqlite3.connect(":memory:")
    schema.create_tables(conn)
    yield conn
    conn.close()


def make_meta(sensor="depthpro_gmm", num_modes=4, sigma_space="log"):
    return SensorMeta(
        sensor=sensor,
        num_modes=num_modes,
        method="gmm_patch",
        method_params={"patch": 7, "gmm": {"K": num_modes, "reg": 1e-6}},
        sigma_space=sigma_space,
        has_confidence=True,
        dump_dir="/dumps/depthpro",
        dump_hash="abc123",
        created_at=schema.now_iso(),
    )


def make_row(sensor="depthpro_gmm", K=4, image_id=1, idx=0, **overrides):
    fields = dict(
        image_id=image_id,
        point2D_idx=idx,
        sensor=sensor,
        modes=np.linspace(1.0, 4.0, K),
        weights=np.full(K, 1.0 / K),
        estimated_depth=2.5,
        sigmas=np.full(K, 0.1),
        confidence=0.9,
        is_sky=False,
    )
    fields.update(overrides)
    return KeypointDepth(**fields)


def test_full_row_round_trip(conn):
    row = make_row(K=4)
    schema.write_depths(conn, [row])
    out = schema.read_depths_for_image(conn, 1, "depthpro_gmm", num_modes=4)
    assert set(out) == {0}
    got = out[0]
    np.testing.assert_array_equal(got.modes, row.modes.astype(np.float32))
    np.testing.assert_array_equal(got.weights, row.weights.astype(np.float32))
    np.testing.assert_array_equal(got.sigmas, row.sigmas.astype(np.float32))
    assert got.modes.dtype == np.float32
    assert got.weights.dtype == np.float32
    assert got.sigmas.dtype == np.float32
    assert got.estimated_depth == pytest.approx(2.5)
    assert got.confidence == pytest.approx(0.9)
    assert got.is_sky is False


def test_k1_row_optionals_none(conn):
    row = make_row(K=1, sigmas=None, confidence=None, is_sky=None)
    schema.write_depths(conn, [row])
    got = schema.read_depths_for_image(conn, 1, "depthpro_gmm", num_modes=1)[0]
    assert got.sigmas is None
    assert got.confidence is None
    assert got.is_sky is None
    np.testing.assert_array_equal(got.modes, row.modes.astype(np.float32))


def test_nan_confidence_rejected(conn):
    row = make_row(confidence=float("nan"))
    with pytest.raises(ValueError, match="NaN"):
        schema.write_depths(conn, [row])


def test_duplicate_key_rejected(conn):
    schema.write_depths(conn, [make_row()])
    with pytest.raises(sqlite3.IntegrityError):
        schema.write_depths(conn, [make_row()])


def test_num_modes_mismatch(conn):
    schema.write_depths(conn, [make_row(K=4)])
    with pytest.raises(ValueError, match="table/meta inconsistency"):
        schema.read_depths_for_image(conn, 1, "depthpro_gmm", num_modes=2)


@pytest.mark.parametrize("sigma_space", ["log", None])
def test_meta_round_trip(conn, sigma_space):
    meta = make_meta(sigma_space=sigma_space)
    schema.write_meta(conn, meta)
    got = schema.read_meta(conn, meta.sensor)
    assert got.method_params == {"patch": 7, "gmm": {"K": 4, "reg": 1e-6}}
    assert got.sigma_space == sigma_space
    assert got == meta


def test_invalid_sigma_space_rejected(conn):
    with pytest.raises(ValueError, match="sigma_space"):
        schema.write_meta(conn, make_meta(sigma_space="banana"))


def test_read_meta_missing_sensor_lists_available(conn):
    schema.write_meta(conn, make_meta(sensor="depthpro_gmm"))
    with pytest.raises(KeyError, match="depthpro_gmm"):
        schema.read_meta(conn, "nonexistent")


def test_delete_sensor_then_reingest(conn):
    schema.write_meta(conn, make_meta())
    schema.write_depths(conn, [make_row()])
    schema.delete_sensor(conn, "depthpro_gmm")
    assert schema.count_rows(conn, "depthpro_gmm") == 0
    assert schema.list_sensors(conn) == []
    schema.write_meta(conn, make_meta())
    schema.write_depths(conn, [make_row()])
    assert schema.count_rows(conn, "depthpro_gmm") == 1
    assert schema.list_sensors(conn) == ["depthpro_gmm"]
