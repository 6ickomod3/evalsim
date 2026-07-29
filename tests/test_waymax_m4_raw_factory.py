"""Synthetic pre-JAX boundary test for the pinned Waymax WOMD factory.

The invented arrays below contain no WOMD payload or source-derived values.
"""
from __future__ import annotations

import numpy as np
import pytest


def _invented_pre_jax_example() -> dict[str, np.ndarray]:
    """Build a small NumPy mapping with the dtypes emitted before JAX."""
    int32_min = np.iinfo(np.int32).min
    int32_max = np.iinfo(np.int32).max
    valid = np.asarray(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    float_values = np.asarray(
        [
            [10.0, 11.0, -1.0, -1.0],
            [20.0, -1.0, 22.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    )
    length = np.asarray(
        [
            [4.0, 6.0, np.nan, -8.0],
            [2.0, np.inf, 4.0, -9.0],
            [np.nan, np.inf, -3.0, 99.0],
        ],
        dtype=np.float32,
    )
    width = np.asarray(
        [
            [2.0, 4.0, np.inf, -3.0],
            [1.0, np.nan, 2.0, -7.0],
            [np.inf, np.nan, -5.0, 101.0],
        ],
        dtype=np.float32,
    )

    return {
        # WOMD stores these semantically integral fields as float32. 2**24 is
        # the exact-integer precision boundary for float32; -1 is padding.
        "state/id": np.asarray(
            [0.0, float(2**24), -1.0],
            dtype=np.float32,
        ),
        "state/type": np.asarray([1.0, 4.0, -1.0], dtype=np.float32),
        "state/tracks_to_predict": np.asarray([1, 0, -1], dtype=np.int64),
        "state/is_sdc": np.asarray([1, 0, -1], dtype=np.int64),
        "state/objects_of_interest": np.asarray([0, 1, 0], dtype=np.int64),
        "state/all/x": float_values.copy(),
        "state/all/y": (float_values + np.float32(0.25)).astype(np.float32),
        "state/all/z": np.zeros_like(float_values),
        "state/all/velocity_x": np.ones_like(float_values),
        "state/all/velocity_y": np.zeros_like(float_values),
        "state/all/bbox_yaw": np.zeros_like(float_values),
        "state/all/valid": valid,
        "state/all/length": length,
        "state/all/width": width,
        "state/all/height": np.asarray(
            [
                [1.0, 1.0, np.nan, -1.0],
                [2.0, np.inf, 2.0, -1.0],
                [np.nan, np.inf, -1.0, -2.0],
            ],
            dtype=np.float32,
        ),
        "state/all/timestamp_micros": np.asarray(
            [
                [int32_min, -1, 0, int32_max],
                [1, 2, 3, -1],
                [-1, -1, -1, -1],
            ],
            dtype=np.int64,
        ),
        "roadgraph_samples/xyz": np.asarray(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [-1.0, -1.0, -1.0],
            ],
            dtype=np.float32,
        ),
        "roadgraph_samples/dir": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, -1.0, -1.0],
            ],
            dtype=np.float32,
        ),
        "roadgraph_samples/type": np.asarray(
            [[1], [19], [-1]],
            dtype=np.int64,
        ),
        "roadgraph_samples/id": np.asarray(
            [[int32_min], [int32_max], [-1]],
            dtype=np.int64,
        ),
        "roadgraph_samples/valid": np.asarray(
            [[1], [1], [0]],
            dtype=np.int64,
        ),
        "traffic_light_state/all/x": np.asarray(
            [[1.0, 2.0], [1.5, -1.0], [2.0, 3.0], [-1.0, -1.0]],
            dtype=np.float32,
        ),
        "traffic_light_state/all/y": np.zeros((4, 2), dtype=np.float32),
        "traffic_light_state/all/z": np.zeros((4, 2), dtype=np.float32),
        "traffic_light_state/all/state": np.asarray(
            [[1, 8], [3, -1], [6, 4], [-1, -1]],
            dtype=np.int64,
        ),
        "traffic_light_state/all/id": np.asarray(
            [
                [int32_min, int32_max],
                [7, -1],
                [8, 9],
                [-1, -1],
            ],
            dtype=np.int64,
        ),
        "traffic_light_state/all/valid": np.asarray(
            [[1, 1], [1, 0], [1, 1], [0, 0]],
            dtype=np.int64,
        ),
    }


def test_pinned_waymax_factory_preserves_pre_jax_discrete_semantics() -> None:
    dataloader = pytest.importorskip(
        "waymax.dataloader",
        reason="the synthetic factory test requires the optional Waymo stack",
    )
    raw = _invented_pre_jax_example()
    snapshot = {key: value.copy() for key, value in raw.items()}
    assert raw["state/id"].dtype == np.float32
    assert raw["state/type"].dtype == np.float32
    int64_fields = (
        "state/tracks_to_predict",
        "state/is_sdc",
        "state/objects_of_interest",
        "state/all/valid",
        "state/all/timestamp_micros",
        "roadgraph_samples/type",
        "roadgraph_samples/id",
        "roadgraph_samples/valid",
        "traffic_light_state/all/state",
        "traffic_light_state/all/id",
        "traffic_light_state/all/valid",
    )
    assert all(raw[key].dtype == np.int64 for key in int64_fields)

    # SDC paths are intentionally disabled: route fields are unrelated to every
    # metadata, roadgraph, traffic-light, and trajectory field asserted here.
    state = dataloader.simulator_state_from_womd_dict(
        raw,
        include_sdc_paths=False,
    )

    metadata = state.object_metadata
    np.testing.assert_array_equal(
        np.asarray(metadata.ids),
        raw["state/id"].astype(np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(metadata.object_types),
        raw["state/type"].astype(np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(metadata.is_valid),
        raw["state/tracks_to_predict"] != -1,
    )
    np.testing.assert_array_equal(
        np.asarray(metadata.is_modeled),
        raw["state/tracks_to_predict"] == 1,
    )
    np.testing.assert_array_equal(
        np.asarray(metadata.is_sdc),
        raw["state/is_sdc"] == 1,
    )
    assert not raw["state/all/valid"][2].any()
    assert raw["state/is_sdc"][2] == -1
    assert not np.asarray(metadata.is_sdc)[2]
    np.testing.assert_array_equal(
        np.asarray(metadata.objects_of_interest),
        raw["state/objects_of_interest"] == 1,
    )
    np.testing.assert_array_equal(
        np.asarray(metadata.is_controlled),
        np.zeros(3, dtype=np.bool_),
    )
    assert np.asarray(metadata.ids).dtype == np.int32
    assert np.asarray(metadata.object_types).dtype == np.int32
    assert np.asarray(metadata.is_valid).dtype == np.bool_
    assert np.asarray(metadata.is_modeled).dtype == np.bool_
    assert np.asarray(metadata.is_sdc).dtype == np.bool_
    assert np.asarray(metadata.objects_of_interest).dtype == np.bool_
    assert np.asarray(metadata.is_controlled).dtype == np.bool_

    roadgraph = state.roadgraph_points
    np.testing.assert_array_equal(
        np.asarray(roadgraph.types),
        raw["roadgraph_samples/type"][..., 0].astype(np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(roadgraph.ids),
        raw["roadgraph_samples/id"][..., 0].astype(np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(roadgraph.valid),
        raw["roadgraph_samples/valid"][..., 0].astype(np.bool_),
    )
    assert np.asarray(roadgraph.types).dtype == np.int32
    assert np.asarray(roadgraph.ids).dtype == np.int32
    assert np.asarray(roadgraph.valid).dtype == np.bool_

    traffic_lights = state.log_traffic_light
    np.testing.assert_array_equal(
        np.asarray(traffic_lights.state),
        raw["traffic_light_state/all/state"].T.astype(np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(traffic_lights.lane_ids),
        raw["traffic_light_state/all/id"].T.astype(np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(traffic_lights.valid),
        raw["traffic_light_state/all/valid"].T.astype(np.bool_),
    )
    assert np.asarray(traffic_lights.state).dtype == np.int32
    assert np.asarray(traffic_lights.lane_ids).dtype == np.int32
    assert np.asarray(traffic_lights.valid).dtype == np.bool_

    trajectory = state.log_trajectory
    np.testing.assert_array_equal(
        np.asarray(trajectory.valid),
        raw["state/all/valid"].astype(np.bool_),
    )
    np.testing.assert_array_equal(
        np.asarray(trajectory.timestamp_micros),
        raw["state/all/timestamp_micros"].astype(np.int32),
    )
    assert np.asarray(trajectory.valid).dtype == np.bool_
    assert np.asarray(trajectory.timestamp_micros).dtype == np.int32
    expected_length = np.broadcast_to(
        np.asarray([5.0, 3.0, -1.0], dtype=np.float32)[:, np.newaxis],
        raw["state/all/length"].shape,
    )
    expected_width = np.broadcast_to(
        np.asarray([3.0, 1.5, -1.0], dtype=np.float32)[:, np.newaxis],
        raw["state/all/width"].shape,
    )
    np.testing.assert_array_equal(
        np.asarray(trajectory.length),
        expected_length,
    )
    np.testing.assert_array_equal(
        np.asarray(trajectory.width),
        expected_width,
    )
    assert np.asarray(trajectory.length).dtype == np.float32
    assert np.asarray(trajectory.width).dtype == np.float32

    # The simulator trajectory starts with the exact first logged frame and
    # uses typed zero padding for every later frame.
    np.testing.assert_array_equal(
        np.asarray(state.sim_trajectory.timestamp_micros)[:, 0],
        raw["state/all/timestamp_micros"][:, 0].astype(np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(state.sim_trajectory.timestamp_micros)[:, 1:],
        np.zeros((3, 3), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(state.sim_trajectory.valid)[:, 0],
        raw["state/all/valid"][:, 0].astype(np.bool_),
    )
    np.testing.assert_array_equal(
        np.asarray(state.sim_trajectory.valid)[:, 1:],
        np.zeros((3, 3), dtype=np.bool_),
    )
    assert state.sdc_paths is None

    # The pinned factory must treat the pre-JAX mapping as read-only input.
    np.testing.assert_array_equal(
        raw["state/is_sdc"],
        np.asarray([1, 0, -1], dtype=np.int64),
    )
    assert tuple(raw) == tuple(snapshot)
    for key, expected in snapshot.items():
        assert raw[key].dtype == expected.dtype
        np.testing.assert_array_equal(raw[key], expected)
