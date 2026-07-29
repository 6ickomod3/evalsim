"""Pure M3 Waymax-state adapter tests using only in-memory upstream datatypes."""
from __future__ import annotations

import dataclasses
import json
import math

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

from waymax.dataloader import womd_factories
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


def _float32_masked_mean(values: np.ndarray, valid: np.ndarray) -> np.float32:
    selected = np.asarray(values, dtype=np.float32)[valid]
    if selected.size == 0:
        return np.float32(-1.0)
    return np.float32(
        np.sum(selected, dtype=np.float32) / np.float32(selected.size)
    )


def _reference_dimension_mean_and_bound(
    values: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float]:
    selected = np.asarray(values, dtype=np.float32)[valid]
    reference_mean = (
        math.fsum(float(value) for value in selected) / selected.size
    )
    epsilon = float(np.finfo(np.float32).eps)
    reduction_steps = int(selected.size) - 1
    gamma = (
        reduction_steps
        * epsilon
        / (1.0 - reduction_steps * epsilon)
    )
    return reference_mean, reference_mean * (
        gamma + epsilon * (1.0 + gamma)
    )


def _base_factory_dimension_tensors() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    trajectory = _make_state().log_trajectory
    return (
        np.array(trajectory.valid, dtype=bool, copy=True),
        np.array(trajectory.length, dtype=np.float32, copy=True),
        np.array(trajectory.width, dtype=np.float32, copy=True),
    )


def _factory_state_and_record(
    *,
    raw_length: np.ndarray,
    raw_width: np.ndarray,
    valid: np.ndarray,
) -> tuple[SimulatorState, WaymaxRecord]:
    """Build the adapter boundary through the actual pinned WOMD factory."""
    state = _make_state()
    source = state.log_trajectory
    trajectory = womd_factories.trajectory_from_womd_dict(
        {
            "state/all/x": np.asarray(source.x),
            "state/all/y": np.asarray(source.y),
            "state/all/z": np.asarray(source.z),
            "state/all/velocity_x": np.asarray(source.vel_x),
            "state/all/velocity_y": np.asarray(source.vel_y),
            "state/all/bbox_yaw": np.asarray(source.yaw),
            "state/all/valid": valid,
            "state/all/length": raw_length,
            "state/all/width": raw_width,
            "state/all/height": np.asarray(source.height),
            "state/all/timestamp_micros": np.asarray(
                source.timestamp_micros
            ),
        }
    )
    state = state.replace(
        log_trajectory=trajectory,
        sim_trajectory=trajectory,
    )
    audit = _audit_from_state(state)
    audit["state/all/length"] = raw_length
    audit["state/all/width"] = raw_width
    record = dataclasses.replace(_record_from_state(state), audit=audit)
    return state, record


@pytest.fixture
def varied_raw_dimension_record() -> tuple[SimulatorState, WaymaxRecord]:
    """Exercise the actual pinned factory with invented raw dimensions."""
    valid, _, _ = _base_factory_dimension_tensors()
    valid[2] &= np.arange(valid.shape[1]) % 3 != 0
    shape = valid.shape
    raw_length = np.full(shape, np.nan, dtype=np.float32)
    raw_width = np.full(shape, np.inf, dtype=np.float32)
    target_length = np.asarray([4.5, 0.8, 2.0, -1.0], dtype=np.float32)
    target_width = np.asarray([2.0, 0.5, 0.8, -1.0], dtype=np.float32)

    for source_index in range(shape[0]):
        valid_indices = np.flatnonzero(valid[source_index])
        if valid_indices.size == 0:
            continue
        for values, target in (
            (raw_length, target_length[source_index]),
            (raw_width, target_width[source_index]),
        ):
            values[source_index, valid_indices] = target
            values[source_index, valid_indices[0]] = target - np.float32(0.25)
            values[source_index, valid_indices[1]] = target + np.float32(0.25)

    raw_length[2] = (
        np.float32(5.0)
        + np.arange(shape[1], dtype=np.float32) * np.float32(0.0037)
    )

    # These values exercise the fact that invalid raw dimension payloads do not
    # contribute to the factory's masked mean.
    raw_length[1, 0:3] = np.asarray([np.inf, -17.0, np.nan], dtype=np.float32)
    raw_width[1, 0:3] = np.asarray([np.nan, -19.0, np.inf], dtype=np.float32)
    raw_length[3, 0:4] = np.asarray(
        [np.nan, np.inf, -23.0, 101.0],
        dtype=np.float32,
    )
    raw_width[3, 0:4] = np.asarray(
        [np.inf, np.nan, -29.0, 103.0],
        dtype=np.float32,
    )

    return _factory_state_and_record(
        raw_length=raw_length,
        raw_width=raw_width,
        valid=valid,
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


def test_pinned_factory_preserves_minimum_normal_dimension_at_91_steps() -> None:
    valid, raw_length, raw_width = _base_factory_dimension_tensors()
    minimum_normal = np.finfo(np.float32).tiny
    assert valid.shape[1] == 91
    assert np.all(valid[0])
    raw_length[0] = minimum_normal

    state, record = _factory_state_and_record(
        raw_length=raw_length,
        raw_width=raw_width,
        valid=valid,
    )
    factory_length = np.asarray(state.log_trajectory.length[0])
    assert factory_length.dtype == np.float32
    assert np.all(np.isfinite(factory_length))
    assert np.all(factory_length > 0.0)
    np.testing.assert_array_equal(
        factory_length,
        np.full(91, minimum_normal, dtype=np.float32),
    )

    scenario = _convert(state)
    assert scenario.agents[0].length == float(minimum_normal)
    assert validate_record_parity(record, scenario)["agents"] is True

    expected, analytic_bound = _reference_dimension_mean_and_bound(
        record.audit["state/all/length"][0],
        np.asarray(record.audit["state/all/valid"][0], dtype=bool),
    )
    fixed_atol_counterexample = 5.0e-7
    assert analytic_bound < fixed_atol_counterexample < 1.0e-6
    contradicted_agents = list(scenario.agents)
    contradicted_agents[0] = dataclasses.replace(
        contradicted_agents[0],
        length=expected + fixed_atol_counterexample,
    )
    contradicted = dataclasses.replace(
        scenario,
        agents=contradicted_agents,
    )
    with pytest.raises(WaymaxDataError, match="parity_dimensions"):
        validate_record_parity(record, contradicted)


def test_pinned_factory_flushes_positive_subnormal_dimension_at_91_steps() -> None:
    valid, raw_length, raw_width = _base_factory_dimension_tensors()
    minimum_normal = np.finfo(np.float32).tiny
    positive_subnormal = np.nextafter(
        minimum_normal,
        np.float32(0.0),
    )
    assert np.float32(0.0) < positive_subnormal < minimum_normal
    assert np.all(valid[0])
    raw_length[0] = positive_subnormal

    state, _ = _factory_state_and_record(
        raw_length=raw_length,
        raw_width=raw_width,
        valid=valid,
    )
    factory_length = np.asarray(state.log_trajectory.length[0])
    np.testing.assert_array_equal(
        factory_length,
        np.zeros(91, dtype=np.float32),
    )

    with pytest.raises(WaymaxConversionError) as error:
        _convert(state)
    assert error.value.code == "dimension_not_constant"


def test_pinned_factory_preserves_guarded_high_magnitude_dimension() -> None:
    valid, raw_length, raw_width = _base_factory_dimension_tensors()
    float32 = np.finfo(np.float32)
    sample_count = 91
    epsilon = float(float32.eps)
    reduction_steps = sample_count - 1
    gamma = (
        reduction_steps
        * epsilon
        / (1.0 - reduction_steps * epsilon)
    )
    weights = np.linspace(
        np.float32(0.5),
        np.float32(1.5),
        91,
        dtype=np.float32,
    )
    guarded_sum_target = float(float32.max) * 0.99 / (1.0 + gamma)
    scale = np.float32(
        guarded_sum_target
        / math.fsum(float(value) for value in weights)
    )
    safe_values = (weights * scale).astype(np.float32)
    assert np.all(valid[0])
    raw_length[0] = safe_values

    reference_sum = math.fsum(float(value) for value in safe_values)
    guard_ratio = (
        reference_sum * (1.0 + gamma) / float(float32.max)
    )
    assert 0.98 < guard_ratio < 1.0

    state, record = _factory_state_and_record(
        raw_length=raw_length,
        raw_width=raw_width,
        valid=valid,
    )
    factory_length = np.asarray(state.log_trajectory.length[0])
    assert factory_length.dtype == np.float32
    assert np.all(np.isfinite(factory_length))
    assert np.all(factory_length > 0.0)
    np.testing.assert_array_equal(
        factory_length,
        np.full(91, factory_length[0], dtype=np.float32),
    )

    scenario = _convert(state)
    assert validate_record_parity(record, scenario)["agents"] is True


def test_pinned_factory_rejects_repeated_float32_max_dimension() -> None:
    valid, raw_length, raw_width = _base_factory_dimension_tensors()
    maximum = np.finfo(np.float32).max
    assert np.all(valid[0])
    raw_length[0] = maximum

    state, _ = _factory_state_and_record(
        raw_length=raw_length,
        raw_width=raw_width,
        valid=valid,
    )
    factory_length = np.asarray(state.log_trajectory.length[0])
    assert np.all(np.isinf(factory_length))

    with pytest.raises(WaymaxConversionError) as error:
        _convert(state)
    assert error.value.code == "dimension_not_constant"


def test_pinned_factory_single_sample_locks_k_zero_and_max_boundary() -> None:
    valid, raw_length, raw_width = _base_factory_dimension_tensors()
    source_index = 1
    source_step = 10
    valid[source_index] = False
    valid[source_index, source_step] = True
    maximum = np.finfo(np.float32).max
    raw_length[source_index] = np.nan
    raw_length[source_index, source_step] = maximum

    state, record = _factory_state_and_record(
        raw_length=raw_length,
        raw_width=raw_width,
        valid=valid,
    )
    factory_length = np.asarray(
        state.log_trajectory.length[source_index],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        factory_length,
        np.full(91, maximum, dtype=np.float32),
    )

    scenario = _convert(state)
    assert validate_record_parity(record, scenario)["agents"] is True
    expected, correct_bound = _reference_dimension_mean_and_bound(
        record.audit["state/all/length"][source_index],
        valid[source_index],
    )
    epsilon = float(np.finfo(np.float32).eps)
    wrong_gamma = epsilon / (1.0 - epsilon)
    wrong_k_one_bound = expected * (
        wrong_gamma + epsilon * (1.0 + wrong_gamma)
    )
    contradiction = (correct_bound + wrong_k_one_bound) / 2.0
    assert correct_bound < contradiction < wrong_k_one_bound

    contradicted_agents = list(scenario.agents)
    contradicted_agents[source_index] = dataclasses.replace(
        contradicted_agents[source_index],
        length=expected + contradiction,
    )
    contradicted = dataclasses.replace(
        scenario,
        agents=contradicted_agents,
    )
    with pytest.raises(WaymaxDataError, match="parity_dimensions"):
        validate_record_parity(record, contradicted)


def test_varied_raw_dimensions_match_factory_broadcast_and_parity(
    varied_raw_dimension_record,
) -> None:
    state, record = varied_raw_dimension_record
    valid = np.asarray(record.audit["state/all/valid"], dtype=bool)
    trajectory = state.log_trajectory

    for source_index in np.flatnonzero(np.any(valid, axis=1)):
        raw_length = record.audit["state/all/length"][source_index]
        raw_width = record.audit["state/all/width"][source_index]
        assert np.unique(raw_length[valid[source_index]]).size > 1
        assert np.unique(raw_width[valid[source_index]]).size > 1
        expected_length, length_bound = _reference_dimension_mean_and_bound(
            raw_length,
            valid[source_index],
        )
        expected_width, width_bound = _reference_dimension_mean_and_bound(
            raw_width,
            valid[source_index],
        )
        actual_length = np.asarray(trajectory.length[source_index])
        actual_width = np.asarray(trajectory.width[source_index])
        np.testing.assert_array_equal(
            actual_length,
            np.full(trajectory.num_timesteps, actual_length[0]),
        )
        np.testing.assert_array_equal(
            actual_width,
            np.full(trajectory.num_timesteps, actual_width[0]),
        )
        assert math.isclose(
            float(actual_length[0]),
            expected_length,
            rel_tol=0.0,
            abs_tol=length_bound,
        )
        assert math.isclose(
            float(actual_width[0]),
            expected_width,
            rel_tol=0.0,
            abs_tol=width_bound,
        )

    counterexample_index = 2
    assert np.count_nonzero(valid[counterexample_index]) == 54
    np.testing.assert_array_equal(
        record.audit["state/all/length"][counterexample_index],
        np.float32(5.0)
        + np.arange(91, dtype=np.float32) * np.float32(0.0037),
    )
    numpy_mean = _float32_masked_mean(
        record.audit["state/all/length"][counterexample_index],
        valid[counterexample_index],
    )
    factory_mean = np.asarray(
        trajectory.length[counterexample_index],
        dtype=np.float32,
    )[0]
    assert abs(float(numpy_mean) - float(factory_mean)) > 1e-6

    scenario = _convert(state)
    assert validate_record_parity(record, scenario)["agents"] is True


def test_dimension_parity_rejects_raw_valid_sample_mean_drift(
    varied_raw_dimension_record,
) -> None:
    state, record = varied_raw_dimension_record
    valid = np.asarray(record.audit["state/all/valid"], dtype=bool)
    source_index = 0
    source_step = int(np.flatnonzero(valid[source_index])[0])
    drifted_audit = dict(record.audit)
    drifted_length = np.array(
        drifted_audit["state/all/length"],
        copy=True,
    )
    before, before_bound = _reference_dimension_mean_and_bound(
        drifted_length[source_index],
        valid[source_index],
    )
    drifted_length[source_index, source_step] += np.float32(0.01)
    after, after_bound = _reference_dimension_mean_and_bound(
        drifted_length[source_index],
        valid[source_index],
    )
    actual = _convert(state).agents[source_index].length
    assert math.isclose(
        actual,
        before,
        rel_tol=0.0,
        abs_tol=before_bound,
    )
    assert not math.isclose(
        actual,
        after,
        rel_tol=0.0,
        abs_tol=after_bound,
    )
    drifted_audit["state/all/length"] = drifted_length

    with pytest.raises(WaymaxDataError, match="parity_dimensions"):
        validate_record_parity(
            dataclasses.replace(record, audit=drifted_audit),
            _convert(state),
        )


def test_dimension_parity_ignores_invalid_raw_payload(
    varied_raw_dimension_record,
) -> None:
    state, record = varied_raw_dimension_record
    valid = np.asarray(record.audit["state/all/valid"], dtype=bool)
    source_index = 1
    source_step = int(np.flatnonzero(~valid[source_index])[0])
    drifted_audit = dict(record.audit)
    drifted_length = np.array(
        drifted_audit["state/all/length"],
        copy=True,
    )
    drifted_width = np.array(
        drifted_audit["state/all/width"],
        copy=True,
    )
    expected_length = _reference_dimension_mean_and_bound(
        drifted_length[source_index],
        valid[source_index],
    )
    expected_width = _reference_dimension_mean_and_bound(
        drifted_width[source_index],
        valid[source_index],
    )
    drifted_length[source_index, source_step] = np.float32(-1.0e20)
    drifted_width[source_index, source_step] = np.float32(np.nan)
    assert (
        _reference_dimension_mean_and_bound(
            drifted_length[source_index],
            valid[source_index],
        )
        == expected_length
    )
    assert (
        _reference_dimension_mean_and_bound(
            drifted_width[source_index],
            valid[source_index],
        )
        == expected_width
    )
    drifted_audit["state/all/length"] = drifted_length
    drifted_audit["state/all/width"] = drifted_width

    assert (
        validate_record_parity(
            dataclasses.replace(record, audit=drifted_audit),
            _convert(state),
        )["agents"]
        is True
    )


@pytest.mark.parametrize(
    "invalid_sample",
    [
        pytest.param(np.float32(np.nan), id="nan"),
        pytest.param(np.float32(np.inf), id="positive-infinity"),
        pytest.param(np.float32(-np.inf), id="negative-infinity"),
        pytest.param(np.float32(0.0), id="zero"),
        pytest.param(np.float32(-1.0), id="negative"),
        pytest.param(
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            id="positive-subnormal",
        ),
    ],
)
def test_dimension_parity_rejects_invalid_valid_raw_samples(
    varied_raw_dimension_record,
    invalid_sample: np.float32,
) -> None:
    state, record = varied_raw_dimension_record
    valid = np.asarray(record.audit["state/all/valid"], dtype=bool)
    source_index = 0
    source_step = int(np.flatnonzero(valid[source_index])[0])
    invalid_audit = dict(record.audit)
    invalid_length = np.array(
        invalid_audit["state/all/length"],
        copy=True,
    )
    invalid_length[source_index, source_step] = invalid_sample
    invalid_audit["state/all/length"] = invalid_length

    with pytest.raises(WaymaxDataError, match="parity_dimensions"):
        validate_record_parity(
            dataclasses.replace(record, audit=invalid_audit),
            _convert(state),
        )


def test_dimension_parity_rejects_scalar_beyond_analytic_bound(
    varied_raw_dimension_record,
) -> None:
    state, record = varied_raw_dimension_record
    scenario = _convert(state)
    valid = np.asarray(record.audit["state/all/valid"], dtype=bool)
    source_index = 2
    expected, absolute_tolerance = _reference_dimension_mean_and_bound(
        record.audit["state/all/length"][source_index],
        valid[source_index],
    )
    contradicted_agents = list(scenario.agents)
    contradicted_agents[source_index] = dataclasses.replace(
        contradicted_agents[source_index],
        length=expected + 2.0 * absolute_tolerance,
    )
    contradicted = dataclasses.replace(
        scenario,
        agents=contradicted_agents,
    )

    assert (
        abs(contradicted.agents[source_index].length - expected)
        > absolute_tolerance
    )
    with pytest.raises(WaymaxDataError, match="parity_dimensions"):
        validate_record_parity(record, contradicted)


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
