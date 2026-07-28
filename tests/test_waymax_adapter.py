"""Pure M3 Waymax-state adapter tests using only in-memory upstream datatypes."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from evalsim import AgentType, MapType
from evalsim.sources.waymax import (
    DEFAULT_WAYMAX_TEMPORAL_PROFILE,
    MAX_MAP_SEGMENT_METERS,
    WAYMAX_ADAPTER_VERSION,
    WAYMAX_COMMIT,
    WOMD_DATASET_VERSION,
    WaymaxConversionError,
    WaymaxTemporalProfile,
    scenario_from_waymax_state,
    waymax_adapter_fingerprint,
)
from evalsim.sources.waymax_loader import (
    WaymaxDataError,
    WaymaxRecord,
    validate_record_parity,
)

jnp = pytest.importorskip(
    "jax.numpy",
    reason="Waymax adapter fixtures require the optional waymo dependencies",
)
pytest.importorskip(
    "waymax",
    reason="Waymax adapter fixtures require the optional waymo dependencies",
)

from waymax.datatypes.object_state import ObjectMetadata, Trajectory
from waymax.datatypes.roadgraph import RoadgraphPoints
from waymax.datatypes.simulator_state import SimulatorState
from waymax.datatypes.traffic_lights import TrafficLights


def _provenance() -> dict[str, object]:
    return {
        "shard_suffix": "00000",
        "record_ordinal": 7,
        "shard_sha256": "a" * 64,
        "dataset_config_fingerprint": "b" * 64,
    }


def _make_state() -> SimulatorState:
    num_objects = 4
    num_steps = DEFAULT_WAYMAX_TEMPORAL_PROFILE.horizon
    valid = np.zeros((num_objects, num_steps), dtype=bool)
    valid[0, :] = True
    valid[1, 5:25] = True
    valid[2, 10:] = True

    step = np.arange(num_steps, dtype=np.float32)[np.newaxis, :]
    x = np.tile(step * 0.5, (num_objects, 1))
    x += np.arange(num_objects, dtype=np.float32)[:, np.newaxis] * 10.0
    y = np.tile(step * 0.1, (num_objects, 1))
    yaw = np.full((num_objects, num_steps), 3.0 * np.pi, dtype=np.float32)
    vel_x = np.full((num_objects, num_steps), 5.0, dtype=np.float32)
    vel_y = np.full((num_objects, num_steps), 1.0, dtype=np.float32)
    for values in (x, y, yaw, vel_x, vel_y):
        values[~valid] = np.nan

    timestamp_micros = np.tile(
        (1_000_000 + np.arange(num_steps) * 100_000).astype(np.int32),
        (num_objects, 1),
    )
    timestamp_micros[~valid] = -1
    length = np.tile(
        np.array([4.5, 0.8, 2.0, 1.0], dtype=np.float32)[:, np.newaxis],
        (1, num_steps),
    )
    width = np.tile(
        np.array([2.0, 0.5, 0.8, 1.0], dtype=np.float32)[:, np.newaxis],
        (1, num_steps),
    )
    trajectory = Trajectory(
        x=jnp.asarray(x),
        y=jnp.asarray(y),
        z=jnp.zeros((num_objects, num_steps), dtype=jnp.float32),
        vel_x=jnp.asarray(vel_x),
        vel_y=jnp.asarray(vel_y),
        yaw=jnp.asarray(yaw),
        valid=jnp.asarray(valid),
        timestamp_micros=jnp.asarray(timestamp_micros),
        length=jnp.asarray(length),
        width=jnp.asarray(width),
        height=jnp.ones((num_objects, num_steps), dtype=jnp.float32),
    )
    metadata = ObjectMetadata(
        ids=jnp.asarray([101, 202, 303, -1], dtype=jnp.int32),
        object_types=jnp.asarray([1, 2, 4, 0], dtype=jnp.int32),
        is_sdc=jnp.asarray([True, False, False, False]),
        is_modeled=jnp.asarray([True, False, True, False]),
        is_valid=jnp.asarray(np.any(valid, axis=1)),
        objects_of_interest=jnp.asarray([False, True, True, False]),
        is_controlled=jnp.zeros((num_objects,), dtype=jnp.bool_),
    )

    # Two supported groups pass.  Four other valid groups exercise explicit omission:
    # crosswalk, spacing, direction, and mixed-source-type semantics.  The final point
    # is invalid and never joins a feature group.
    roadgraph = RoadgraphPoints(
        x=jnp.asarray(
            [
                0.0,
                0.5,
                1.0,  # id 10, lane
                2.0,
                2.5,  # id 20, road edge
                3.0,
                3.5,  # id 30, unsupported crosswalk
                4.0,
                5.0,  # id 40, segment too long
                6.0,
                6.5,  # id 50, misaligned direction
                7.0,
                7.5,  # id 60, mixed types
                99.0,  # invalid sample
            ],
            dtype=jnp.float32,
        ),
        y=jnp.zeros((14,), dtype=jnp.float32),
        z=jnp.zeros((14,), dtype=jnp.float32),
        dir_x=jnp.asarray(
            [1, 1, 0, 1, 0, 1, 0, 1, 0, -1, 0, 1, 0, 0],
            dtype=jnp.float32,
        ),
        dir_y=jnp.zeros((14,), dtype=jnp.float32),
        dir_z=jnp.zeros((14,), dtype=jnp.float32),
        types=jnp.asarray(
            [1, 1, 1, 15, 15, 18, 18, 1, 1, 1, 1, 1, 15, 1],
            dtype=jnp.int32,
        ),
        ids=jnp.asarray(
            [10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70],
            dtype=jnp.int32,
        ),
        valid=jnp.asarray([True] * 13 + [False]),
    )
    traffic_lights = TrafficLights(
        x=jnp.zeros((0, num_steps), dtype=jnp.float32),
        y=jnp.zeros((0, num_steps), dtype=jnp.float32),
        z=jnp.zeros((0, num_steps), dtype=jnp.float32),
        state=jnp.zeros((0, num_steps), dtype=jnp.int32),
        lane_ids=jnp.zeros((0, num_steps), dtype=jnp.int32),
        valid=jnp.zeros((0, num_steps), dtype=jnp.bool_),
    )
    return SimulatorState(
        sim_trajectory=trajectory,
        log_trajectory=trajectory,
        log_traffic_light=traffic_lights,
        object_metadata=metadata,
        timestep=jnp.asarray(0, dtype=jnp.int32),
        roadgraph_points=roadgraph,
    )


def _convert(state: SimulatorState | None = None):
    return scenario_from_waymax_state(
        _make_state() if state is None else state,
        scenario_id="abc123",
        provenance=_provenance(),
    )


def _audit_from_state(state: SimulatorState) -> dict[str, np.ndarray]:
    trajectory = state.log_trajectory
    roadgraph = state.roadgraph_points
    metadata = state.object_metadata
    return {
        "state/id": np.asarray(metadata.ids),
        "state/type": np.asarray(metadata.object_types),
        "state/is_sdc": np.asarray(metadata.is_sdc),
        "state/which_time": np.concatenate(
            (
                -np.ones(10, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                np.ones(80, dtype=np.float32),
            )
        ),
        "state/all/valid": np.asarray(trajectory.valid),
        "state/all/x": np.asarray(trajectory.x),
        "state/all/y": np.asarray(trajectory.y),
        "state/all/velocity_x": np.asarray(trajectory.vel_x),
        "state/all/velocity_y": np.asarray(trajectory.vel_y),
        "state/all/bbox_yaw": np.asarray(trajectory.yaw),
        "state/all/timestamp_micros": np.asarray(trajectory.timestamp_micros),
        "state/all/length": np.asarray(trajectory.length),
        "state/all/width": np.asarray(trajectory.width),
        "roadgraph_samples/xyz": np.stack(
            (
                np.asarray(roadgraph.x),
                np.asarray(roadgraph.y),
                np.asarray(roadgraph.z),
            ),
            axis=-1,
        ),
        "roadgraph_samples/dir": np.stack(
            (
                np.asarray(roadgraph.dir_x),
                np.asarray(roadgraph.dir_y),
                np.asarray(roadgraph.dir_z),
            ),
            axis=-1,
        ),
        "roadgraph_samples/type": np.asarray(roadgraph.types)[:, np.newaxis],
        "roadgraph_samples/id": np.asarray(roadgraph.ids)[:, np.newaxis],
        "roadgraph_samples/valid": np.asarray(roadgraph.valid)[:, np.newaxis],
    }


def _record_from_state(state: SimulatorState) -> WaymaxRecord:
    provenance = _provenance()
    return WaymaxRecord(
        scenario_id="abc123",
        state=state,
        audit=_audit_from_state(state),
        shard_suffix=str(provenance["shard_suffix"]),
        record_ordinal=int(provenance["record_ordinal"]),
        shard_sha256=str(provenance["shard_sha256"]),
        dataset_config_fingerprint=str(
            provenance["dataset_config_fingerprint"]
        ),
    )


def _assert_same_scenario(left, right) -> None:
    assert left.scenario_id == right.scenario_id
    assert left.ego_index == right.ego_index
    assert left.metadata == right.metadata
    np.testing.assert_array_equal(left.timestamps, right.timestamps)
    for left_agent, right_agent in zip(left.agents, right.agents, strict=True):
        assert left_agent.id == right_agent.id
        assert left_agent.type == right_agent.type
        assert left_agent.length == right_agent.length
        assert left_agent.width == right_agent.width
        for field in ("valid", "x", "y", "heading", "vx", "vy"):
            np.testing.assert_array_equal(
                getattr(left_agent, field),
                getattr(right_agent, field),
            )
    for left_map, right_map in zip(left.map, right.map, strict=True):
        assert left_map.type == right_map.type
        np.testing.assert_array_equal(left_map.xy, right_map.xy)


def test_independent_parity_checks_real_time_partition_and_contradiction() -> None:
    state = _make_state()
    scenario = _convert(state)
    record = _record_from_state(state)

    assert validate_record_parity(record, scenario)["time_boundary"] is True

    drifted_audit = dict(record.audit)
    drifted_time = np.array(drifted_audit["state/which_time"], copy=True)
    drifted_time[9], drifted_time[10] = drifted_time[10], drifted_time[9]
    drifted_audit["state/which_time"] = drifted_time
    with pytest.raises(WaymaxDataError, match="parity_time_boundary"):
        validate_record_parity(
            dataclasses.replace(record, audit=drifted_audit),
            scenario,
        )


def test_conversion_preserves_supported_agents_time_dimensions_and_order() -> None:
    state = _make_state()
    before_x = np.asarray(state.log_trajectory.x).copy()
    scenario = _convert(state)

    assert scenario.scenario_id == "abc123"
    assert [agent.id for agent in scenario.agents] == [101, 202, 303]
    assert [agent.type for agent in scenario.agents] == [
        AgentType.VEHICLE,
        AgentType.PEDESTRIAN,
        AgentType.UNKNOWN,
    ]
    assert scenario.ego_index == 0
    assert scenario.metadata["current_index"] == 10
    assert scenario.agents[0].length == 4.5
    assert scenario.agents[1].width == 0.5
    np.testing.assert_allclose(
        scenario.timestamps,
        np.arange(scenario.num_steps, dtype=float) * 0.1,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        scenario.agents[1].valid,
        np.asarray(state.log_trajectory.valid)[1],
    )
    assert np.all(scenario.agents[1].x[~scenario.agents[1].valid] == 0.0)
    assert np.all(scenario.agents[1].heading[~scenario.agents[1].valid] == 0.0)
    assert np.all(scenario.agents[0].heading >= -np.pi)
    assert np.all(scenario.agents[0].heading < np.pi)
    np.testing.assert_array_equal(np.asarray(state.log_trajectory.x), before_x)

    for left_index, left in enumerate(scenario.agents):
        for right in scenario.agents[left_index + 1 :]:
            for field in ("valid", "x", "y", "heading", "vx", "vy"):
                assert not np.shares_memory(
                    getattr(left, field),
                    getattr(right, field),
                )


def test_map_gating_preserves_source_order_and_accounts_for_omissions() -> None:
    scenario = _convert()

    assert [feature.type for feature in scenario.map] == [
        MapType.LANE,
        MapType.ROAD_EDGE,
    ]
    np.testing.assert_array_equal(
        scenario.map[0].xy,
        np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]),
    )
    np.testing.assert_array_equal(
        scenario.map[1].xy,
        np.array([[2.0, 0.0], [2.5, 0.0]]),
    )
    accounting = scenario.metadata["map_conversion"]
    assert accounting["source_point_count"] == 14
    assert accounting["valid_point_count"] == 13
    assert accounting["invalid_point_count"] == 1
    assert accounting["retained_group_count"] == 2
    assert accounting["retained_point_count"] == 5
    assert accounting["omitted_group_count"] == 4
    assert accounting["omitted_point_count"] == 8
    assert accounting["omitted_by_reason"] == {
        "unsupported_map_type": {"groups": 1, "points": 2},
        "segment_too_long": {"groups": 1, "points": 2},
        "direction_misaligned": {"groups": 1, "points": 2},
        "mixed_source_types": {"groups": 1, "points": 2},
    }
    assert accounting["unsupported_source_types"] == {
        "18": {"groups": 1, "points": 2}
    }


def test_provenance_is_json_native_bounded_and_deterministic() -> None:
    first = _convert()
    second = _convert()
    _assert_same_scenario(first, second)

    assert first.metadata["source"] == "womd"
    assert first.metadata["source_version"] == WOMD_DATASET_VERSION
    assert first.metadata["adapter_version"] == WAYMAX_ADAPTER_VERSION
    assert first.metadata["waymax_commit"] == WAYMAX_COMMIT
    assert first.metadata["adapter_fingerprint"] == waymax_adapter_fingerprint()
    assert (
        first.metadata["adapter_fingerprint"]
        != first.metadata["dataset_config_fingerprint"]
    )
    assert first.metadata["source_fingerprint"] not in {
        first.metadata["adapter_fingerprint"],
        first.metadata["dataset_config_fingerprint"],
        first.metadata["shard_sha256"],
    }
    assert "scenario_id" not in first.metadata
    json.dumps(first.metadata, allow_nan=False)


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda state: state.replace(
                object_metadata=state.object_metadata.replace(
                    ids=jnp.asarray([101, 101, 303, -1], dtype=jnp.int32)
                )
            ),
            "duplicate_agent_id",
        ),
        (
            lambda state: state.replace(
                object_metadata=state.object_metadata.replace(
                    is_sdc=jnp.zeros((4,), dtype=jnp.bool_)
                )
            ),
            "sdc_count_invalid",
        ),
        (
            lambda state: state.replace(
                object_metadata=state.object_metadata.replace(
                    is_valid=jnp.asarray([True, False, True, False])
                )
            ),
            "object_validity_mismatch",
        ),
    ],
)
def test_agent_contract_failures_have_stable_codes(mutate, code: str) -> None:
    with pytest.raises(WaymaxConversionError) as caught:
        _convert(mutate(_make_state()))
    assert caught.value.code == code
    assert str(caught.value).startswith(f"{code}:")


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("timestamp_disagreement", "timestamp_disagreement"),
        ("timestamp_no_contributor", "timestamp_no_contributor"),
        ("non_finite_valid_value", "trajectory_value_invalid"),
        ("varying_dimension", "dimension_not_constant"),
    ],
)
def test_trajectory_contract_failures_have_stable_codes(
    field: str,
    code: str,
) -> None:
    state = _make_state()
    trajectory = state.log_trajectory
    if field == "timestamp_disagreement":
        values = np.asarray(trajectory.timestamp_micros).copy()
        values[1, 5] += 1
        trajectory = trajectory.replace(timestamp_micros=jnp.asarray(values))
    elif field == "timestamp_no_contributor":
        values = np.asarray(trajectory.valid).copy()
        values[:, 50] = False
        trajectory = trajectory.replace(valid=jnp.asarray(values))
    elif field == "non_finite_valid_value":
        values = np.asarray(trajectory.x).copy()
        values[0, 0] = np.nan
        trajectory = trajectory.replace(x=jnp.asarray(values))
    else:
        values = np.asarray(trajectory.length).copy()
        values[0, 1] += 0.25
        trajectory = trajectory.replace(length=jnp.asarray(values))

    with pytest.raises(WaymaxConversionError) as caught:
        _convert(state.replace(log_trajectory=trajectory))
    assert caught.value.code == code


def test_batched_state_and_temporal_profile_drift_are_rejected() -> None:
    state = _make_state()
    batched = state.replace(
        log_trajectory=state.log_trajectory.replace(
            x=state.log_trajectory.x[jnp.newaxis, ...]
        ),
        object_metadata=state.object_metadata.replace(
            ids=state.object_metadata.ids[jnp.newaxis, ...]
        ),
        timestep=state.timestep[jnp.newaxis],
    )
    with pytest.raises(WaymaxConversionError) as caught:
        _convert(batched)
    assert caught.value.code == "batched_state"

    drifted = WaymaxTemporalProfile(
        past_steps=10,
        current_steps=1,
        future_steps=79,
        init_steps=11,
    )
    with pytest.raises(WaymaxConversionError) as caught:
        scenario_from_waymax_state(
            state,
            scenario_id="abc123",
            temporal_profile=drifted,
            provenance=_provenance(),
        )
    assert caught.value.code == "temporal_profile_drift"


@pytest.mark.parametrize(
    ("key", "value", "code"),
    [
        ("source", "womd", "provenance_reserved_key"),
        ("local_path", "/private/tmp/a.tfrecord", "provenance_absolute_path"),
        ("native_scenario_id", "abc123", "provenance_raw_id"),
        ("tensor", [1, 2, 3], "provenance_non_scalar"),
    ],
)
def test_provenance_rejects_overrides_paths_ids_and_payloads(
    key: str,
    value: object,
    code: str,
) -> None:
    provenance = _provenance()
    provenance[key] = value
    with pytest.raises(WaymaxConversionError) as caught:
        scenario_from_waymax_state(
            _make_state(),
            scenario_id="abc123",
            provenance=provenance,
        )
    assert caught.value.code == code


def test_exact_maximum_map_spacing_is_retained() -> None:
    state = _make_state()
    roadgraph = state.roadgraph_points
    assert roadgraph is not None
    roadgraph = roadgraph.replace(
        x=jnp.asarray([0.0, MAX_MAP_SEGMENT_METERS], dtype=jnp.float32),
        y=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        z=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        dir_x=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        dir_y=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        dir_z=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        types=jnp.asarray([3, 3], dtype=jnp.int32),
        ids=jnp.asarray([88, 88], dtype=jnp.int32),
        valid=jnp.asarray([True, True]),
    )
    scenario = _convert(state.replace(roadgraph_points=roadgraph))
    assert len(scenario.map) == 1
    assert scenario.map[0].type == MapType.LANE
