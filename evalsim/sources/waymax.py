"""Pure, eager conversion from a Waymax state to the EvalSim contract.

This module intentionally has no Waymax, JAX, or TensorFlow imports.  Callers pass an
in-memory, unbatched ``waymax.datatypes.SimulatorState`` (or an object implementing the
same public attributes), and everything leaving this boundary is an EvalSim contract or
JSON-native provenance.  Dataset selection and TFRecord parsing belong to the separate
optional reader boundary.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import PureWindowsPath
from typing import Any, NoReturn

import numpy as np

from evalsim.contracts import Agent, AgentType, MapPolyline, MapType, Scenario

WAYMAX_COMMIT = "a64dfec9be8576b60d9cecc94f406d9812d4a7d0"
WOMD_DATASET_VERSION = "1.3.1"
WAYMAX_ADAPTER_VERSION = "0.1.0"
WAYMAX_ADAPTER_SCHEMA_VERSION = "1"

MIN_MAP_SEGMENT_METERS = 1e-6
MAX_MAP_SEGMENT_METERS = 0.75
MAX_MAP_DIRECTION_ERROR_DEGREES = 10.0

_LANE_TYPE_IDS = frozenset({0, 1, 2, 3})
_ROAD_EDGE_TYPE_IDS = frozenset({14, 15, 16})
_OBJECT_TYPE_MAP = {
    1: AgentType.VEHICLE,
    2: AgentType.PEDESTRIAN,
    3: AgentType.CYCLIST,
}
_REQUIRED_PROVENANCE_KEYS = frozenset(
    {
        "dataset_config_fingerprint",
        "record_ordinal",
        "shard_sha256",
        "shard_suffix",
    }
)
_RAW_ID_PROVENANCE_KEYS = frozenset(
    {"native_id", "native_scenario_id", "raw_id", "raw_scenario_id", "scenario_id"}
)


class WaymaxConversionError(ValueError):
    """Fail-closed source-contract error with a machine-stable reason code."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class WaymaxTemporalProfile:
    """Immutable WOMD temporal layout and matching Waymax initialization boundary."""

    past_steps: int = 10
    current_steps: int = 1
    future_steps: int = 80
    init_steps: int = 11

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise WaymaxConversionError(
                    "temporal_profile_invalid",
                    f"{name} must be an integer",
                )
            object.__setattr__(self, name, int(value))
        if self.past_steps < 0 or self.current_steps < 1 or self.future_steps < 0:
            raise WaymaxConversionError(
                "temporal_profile_invalid",
                "past/future steps must be non-negative and current_steps positive",
            )
        if self.init_steps < 1:
            raise WaymaxConversionError(
                "temporal_profile_invalid",
                "init_steps must be positive",
            )

    @property
    def horizon(self) -> int:
        return self.past_steps + self.current_steps + self.future_steps

    @property
    def current_index(self) -> int:
        return self.past_steps + self.current_steps - 1

    def to_dict(self) -> dict[str, int]:
        return {
            "past_steps": self.past_steps,
            "current_steps": self.current_steps,
            "future_steps": self.future_steps,
            "init_steps": self.init_steps,
        }


DEFAULT_WAYMAX_TEMPORAL_PROFILE = WaymaxTemporalProfile()

_ADAPTER_RULES = {
    "agent_types": {
        "1": AgentType.VEHICLE.value,
        "2": AgentType.PEDESTRIAN.value,
        "3": AgentType.CYCLIST.value,
        "other": AgentType.UNKNOWN.value,
    },
    "dimension_rule": "constant_waymax_broadcast_value",
    "heading_range": "[-pi,pi)",
    "invalid_fill": "zero_where_invalid",
    "map_direction_error_degrees": MAX_MAP_DIRECTION_ERROR_DEGREES,
    "map_lane_type_ids": sorted(_LANE_TYPE_IDS),
    "map_max_segment_meters": MAX_MAP_SEGMENT_METERS,
    "map_min_segment_meters_exclusive": MIN_MAP_SEGMENT_METERS,
    "map_road_edge_type_ids": sorted(_ROAD_EDGE_TYPE_IDS),
    "timestamp_rule": "exact_valid_object_consensus_normalized_to_first_frame",
}


def waymax_adapter_fingerprint(
    temporal_profile: WaymaxTemporalProfile = DEFAULT_WAYMAX_TEMPORAL_PROFILE,
) -> str:
    """Fingerprint every declared rule that affects adapter output."""

    payload = {
        "adapter_schema_version": WAYMAX_ADAPTER_SCHEMA_VERSION,
        "adapter_version": WAYMAX_ADAPTER_VERSION,
        "rules": _ADAPTER_RULES,
        "temporal_profile": temporal_profile.to_dict(),
        "waymax_commit": WAYMAX_COMMIT,
        "womd_dataset_version": WOMD_DATASET_VERSION,
    }
    return _fingerprint(payload)


def scenario_from_waymax_state(
    state: Any,
    *,
    scenario_id: str,
    temporal_profile: WaymaxTemporalProfile = DEFAULT_WAYMAX_TEMPORAL_PROFILE,
    provenance: Mapping[str, Any],
) -> Scenario:
    """Convert one unbatched Waymax ``SimulatorState`` without reading any data.

    Args:
        state: An eager, unbatched Waymax ``SimulatorState``.
        scenario_id: The decoded native WOMD scenario ID.
        temporal_profile: Locked WOMD past/current/future and initialization layout.
        provenance: Flat, JSON-scalar reader provenance.  It must contain the exact
            shard suffix, record ordinal, shard digest, and dataset-config fingerprint.

    Raises:
        WaymaxConversionError: If the source state cannot be represented faithfully by
            the supported M3 EvalSim contract.
    """

    if not isinstance(scenario_id, str) or not scenario_id:
        _fail("scenario_id_invalid", "native scenario_id must be a non-empty string")
    _validate_locked_profile(temporal_profile)
    clean_provenance = _validate_provenance(provenance, scenario_id=scenario_id)
    adapter_fingerprint = waymax_adapter_fingerprint(temporal_profile)
    source_fingerprint = _fingerprint(
        {
            "adapter_fingerprint": adapter_fingerprint,
            "dataset_config_fingerprint": clean_provenance[
                "dataset_config_fingerprint"
            ],
            "shard_sha256": clean_provenance["shard_sha256"],
            "shard_suffix": clean_provenance["shard_suffix"],
        }
    )
    _reject_batched_state(state)

    trajectory = _trajectory_arrays(state)
    num_objects, horizon = trajectory["x"].shape
    if horizon != temporal_profile.horizon:
        _fail(
            "horizon_mismatch",
            f"state has {horizon} steps; temporal profile requires "
            f"{temporal_profile.horizon}",
        )

    object_metadata = _object_metadata_arrays(state, num_objects=num_objects)
    active_from_trajectory = np.any(trajectory["valid"], axis=1)
    if not np.array_equal(object_metadata["is_valid"], active_from_trajectory):
        _fail(
            "object_validity_mismatch",
            "object_metadata.is_valid must equal trajectory valid-any semantics",
        )
    retained_indices = np.flatnonzero(active_from_trajectory)
    if retained_indices.size == 0:
        _fail("no_active_agents", "state contains no trajectory-valid object slots")

    retained_ids = object_metadata["ids"][retained_indices]
    if len(set(int(value) for value in retained_ids)) != retained_indices.size:
        _fail("duplicate_agent_id", "retained Waymax object IDs must be unique")

    retained_sdc = object_metadata["is_sdc"][retained_indices]
    if int(np.count_nonzero(retained_sdc)) != 1:
        _fail(
            "sdc_count_invalid",
            "exactly one retained object must be marked as the SDC",
        )
    if np.any(object_metadata["is_sdc"][~active_from_trajectory]):
        _fail(
            "sdc_count_invalid",
            "an SDC marker may not refer to a never-valid padding slot",
        )

    timestamps = _canonical_timestamps(
        trajectory["timestamp_micros"],
        trajectory["valid"],
        retained_indices,
    )
    agents = _convert_agents(trajectory, object_metadata, retained_indices)
    ego_index = int(np.flatnonzero(retained_sdc)[0])
    map_features, map_conversion = _convert_roadgraph(
        getattr(state, "roadgraph_points", None)
    )

    unknown_types = sum(agent.type == AgentType.UNKNOWN for agent in agents)
    metadata: dict[str, Any] = {
        "source": "womd",
        "source_version": WOMD_DATASET_VERSION,
        "source_fingerprint": source_fingerprint,
        "split": "validation",
        "current_index": temporal_profile.current_index,
        "adapter_version": WAYMAX_ADAPTER_VERSION,
        "adapter_schema_version": WAYMAX_ADAPTER_SCHEMA_VERSION,
        "adapter_fingerprint": adapter_fingerprint,
        "waymax_commit": WAYMAX_COMMIT,
        "temporal_profile": temporal_profile.to_dict(),
        "coordinate_frame": "global",
        "coordinate_unit": "meters",
        "source_time_unit": "microseconds",
        "time_unit": "seconds",
        "time_origin": "first_supported_frame",
        "invalid_fill": "finite_zero_where_invalid",
        "heading_normalization": "[-pi,pi)",
        "dimension_rule": "constant_waymax_broadcast_value",
        "conversion_counts": {
            "source_object_slots": int(num_objects),
            "retained_objects": len(agents),
            "dropped_never_valid_objects": int(num_objects - len(agents)),
            "unknown_object_types": int(unknown_types),
        },
        "map_conversion": map_conversion,
        **clean_provenance,
    }
    # This is an assertion about our boundary, not merely a serialization convenience.
    _json_native_tree(metadata, context="generated Scenario.metadata")

    return Scenario(
        scenario_id=scenario_id,
        timestamps=timestamps,
        agents=agents,
        map=map_features,
        ego_index=ego_index,
        metadata=metadata,
    )


def _validate_locked_profile(profile: WaymaxTemporalProfile) -> None:
    if not isinstance(profile, WaymaxTemporalProfile):
        _fail(
            "temporal_profile_invalid",
            "temporal_profile must be an immutable WaymaxTemporalProfile",
        )
    if profile != DEFAULT_WAYMAX_TEMPORAL_PROFILE:
        _fail(
            "temporal_profile_drift",
            "M3 requires the locked 10-past/1-current/80-future profile with "
            "11 initialization steps",
        )
    if profile.init_steps != profile.current_index + 1:
        _fail(
            "temporal_profile_drift",
            "Waymax init_steps must end at the derived current_index",
        )


def _validate_provenance(
    provenance: Mapping[str, Any],
    *,
    scenario_id: str,
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        _fail("provenance_invalid", "provenance must be a mapping")
    missing = sorted(_REQUIRED_PROVENANCE_KEYS.difference(provenance))
    if missing:
        _fail(
            "provenance_missing",
            f"missing required provenance keys: {', '.join(missing)}",
        )

    clean: dict[str, Any] = {}
    protected = {
        "adapter_fingerprint",
        "adapter_schema_version",
        "adapter_version",
        "conversion_counts",
        "coordinate_frame",
        "coordinate_unit",
        "current_index",
        "dimension_rule",
        "heading_normalization",
        "invalid_fill",
        "map_conversion",
        "source",
        "source_fingerprint",
        "source_time_unit",
        "source_version",
        "split",
        "temporal_profile",
        "time_origin",
        "time_unit",
        "waymax_commit",
    }
    for key, value in provenance.items():
        if not isinstance(key, str) or not key:
            _fail("provenance_invalid", "provenance keys must be non-empty strings")
        if key in protected:
            _fail(
                "provenance_reserved_key",
                f"provenance may not override generated metadata key {key!r}",
            )
        if key.lower() in _RAW_ID_PROVENANCE_KEYS:
            _fail(
                "provenance_raw_id",
                f"raw identity belongs only in Scenario.scenario_id, not {key!r}",
            )
        if value is not None and (
            isinstance(value, (np.generic, np.ndarray))
            or not isinstance(value, (str, int, float, bool))
        ):
            _fail(
                "provenance_non_scalar",
                f"provenance value {key!r} must be a JSON-native scalar",
            )
        if isinstance(value, float) and not math.isfinite(value):
            _fail(
                "provenance_non_finite",
                f"provenance value {key!r} must be finite",
            )
        if isinstance(value, str):
            if os.path.isabs(value) or PureWindowsPath(value).is_absolute():
                _fail(
                    "provenance_absolute_path",
                    f"provenance value {key!r} may not contain an absolute path",
                )
            if value == scenario_id:
                _fail(
                    "provenance_raw_id",
                    "native scenario ID belongs only in Scenario.scenario_id",
                )
        clean[key] = value

    suffix = clean["shard_suffix"]
    if (
        not isinstance(suffix, str)
        or len(suffix) != 5
        or not suffix.isascii()
        or not suffix.isdecimal()
    ):
        _fail(
            "provenance_invalid",
            "shard_suffix must be a canonical five-digit string",
        )
    ordinal = clean["record_ordinal"]
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        _fail(
            "provenance_invalid",
            "record_ordinal must be a non-negative integer",
        )
    shard_sha256 = clean["shard_sha256"]
    if (
        not isinstance(shard_sha256, str)
        or len(shard_sha256) != 64
        or any(char not in "0123456789abcdef" for char in shard_sha256)
    ):
        _fail(
            "provenance_invalid",
            "shard_sha256 must be a lowercase hexadecimal SHA-256 digest",
        )
    config_fingerprint = clean["dataset_config_fingerprint"]
    if not isinstance(config_fingerprint, str) or not config_fingerprint:
        _fail(
            "provenance_invalid",
            "dataset_config_fingerprint must be a non-empty string",
        )
    return clean


def _reject_batched_state(state: Any) -> None:
    try:
        batch_dims = tuple(state.batch_dims)
    except (AttributeError, TypeError) as exc:
        raise WaymaxConversionError(
            "state_contract_invalid",
            "state must expose Waymax batch_dims",
        ) from exc
    if batch_dims:
        _fail(
            "batched_state",
            f"adapter accepts one state, not batch_dims={batch_dims}",
        )
    timestep = _as_array(getattr(state, "timestep", None), "state.timestep")
    if timestep.ndim != 0:
        _fail("batched_state", "state.timestep must be scalar for an unbatched state")


def _trajectory_arrays(state: Any) -> dict[str, np.ndarray]:
    try:
        trajectory = state.log_trajectory
    except AttributeError as exc:
        raise WaymaxConversionError(
            "state_contract_invalid",
            "state must expose log_trajectory",
        ) from exc
    names = (
        "x",
        "y",
        "z",
        "vel_x",
        "vel_y",
        "yaw",
        "valid",
        "timestamp_micros",
        "length",
        "width",
        "height",
    )
    arrays = {
        name: _as_array(getattr(trajectory, name, None), f"log_trajectory.{name}")
        for name in names
    }
    shape = arrays["x"].shape
    if arrays["x"].ndim > 2:
        _fail("batched_state", f"log_trajectory has batched shape {shape}")
    if arrays["x"].ndim != 2:
        _fail(
            "trajectory_shape_invalid",
            f"log_trajectory fields must be [objects, timesteps], got {shape}",
        )
    for name, array in arrays.items():
        if array.shape != shape:
            _fail(
                "trajectory_shape_invalid",
                f"log_trajectory.{name} has shape {array.shape}, expected {shape}",
            )
    if arrays["valid"].dtype != np.bool_:
        _fail(
            "trajectory_dtype_invalid",
            "log_trajectory.valid must have boolean dtype",
        )
    if not np.issubdtype(arrays["timestamp_micros"].dtype, np.integer):
        _fail(
            "trajectory_dtype_invalid",
            "log_trajectory.timestamp_micros must have integer dtype",
        )
    return arrays


def _object_metadata_arrays(
    state: Any,
    *,
    num_objects: int,
) -> dict[str, np.ndarray]:
    try:
        metadata = state.object_metadata
    except AttributeError as exc:
        raise WaymaxConversionError(
            "state_contract_invalid",
            "state must expose object_metadata",
        ) from exc
    names = (
        "ids",
        "object_types",
        "is_sdc",
        "is_modeled",
        "is_valid",
        "objects_of_interest",
        "is_controlled",
    )
    arrays = {
        name: _as_array(getattr(metadata, name, None), f"object_metadata.{name}")
        for name in names
    }
    for name, array in arrays.items():
        if array.ndim > 1:
            _fail(
                "batched_state",
                f"object_metadata.{name} has batched shape {array.shape}",
            )
        if array.shape != (num_objects,):
            _fail(
                "object_metadata_shape_invalid",
                f"object_metadata.{name} has shape {array.shape}, expected "
                f"({num_objects},)",
            )
    if not np.issubdtype(arrays["ids"].dtype, np.integer):
        _fail("object_metadata_dtype_invalid", "object IDs must have integer dtype")
    if not np.issubdtype(arrays["object_types"].dtype, np.integer):
        _fail("object_metadata_dtype_invalid", "object types must have integer dtype")
    for name in (
        "is_sdc",
        "is_modeled",
        "is_valid",
        "objects_of_interest",
        "is_controlled",
    ):
        if arrays[name].dtype != np.bool_:
            _fail(
                "object_metadata_dtype_invalid",
                f"object_metadata.{name} must have boolean dtype",
            )
    return arrays


def _canonical_timestamps(
    timestamp_micros: np.ndarray,
    valid: np.ndarray,
    retained_indices: np.ndarray,
) -> np.ndarray:
    retained_timestamps = timestamp_micros[retained_indices]
    retained_valid = valid[retained_indices]
    canonical = np.empty(timestamp_micros.shape[1], dtype=np.int64)
    for step in range(timestamp_micros.shape[1]):
        contributors = retained_timestamps[retained_valid[:, step], step].astype(
            np.int64,
            copy=False,
        )
        if contributors.size == 0:
            _fail(
                "timestamp_no_contributor",
                f"no retained object is valid at timestep {step}",
            )
        if not np.all(contributors == contributors[0]):
            _fail(
                "timestamp_disagreement",
                f"valid objects disagree at timestep {step}",
            )
        canonical[step] = contributors[0]
    deltas_micros = canonical - canonical[0]
    if np.any(np.diff(deltas_micros) <= 0):
        _fail(
            "timestamps_not_increasing",
            "canonical source timestamps must be strictly increasing",
        )
    timestamps = deltas_micros.astype(np.float64) * 1e-6
    if not np.all(np.isfinite(timestamps)):
        _fail("timestamps_non_finite", "normalized timestamps must be finite")
    return timestamps


def _convert_agents(
    trajectory: Mapping[str, np.ndarray],
    object_metadata: Mapping[str, np.ndarray],
    retained_indices: np.ndarray,
) -> list[Agent]:
    agents: list[Agent] = []
    for source_index in retained_indices:
        source_index = int(source_index)
        valid = np.array(trajectory["valid"][source_index], dtype=bool, copy=True)
        converted_series: dict[str, np.ndarray] = {}
        source_names = {
            "x": "x",
            "y": "y",
            "heading": "yaw",
            "vx": "vel_x",
            "vy": "vel_y",
        }
        for target_name, source_name in source_names.items():
            source = np.array(
                trajectory[source_name][source_index],
                dtype=np.float64,
                copy=True,
            )
            if not np.all(np.isfinite(source[valid])):
                _fail(
                    "trajectory_value_invalid",
                    f"object slot {source_index} has non-finite valid {source_name}",
                )
            if target_name == "heading":
                source[valid] = _wrap_heading(source[valid])
            source[~valid] = 0.0
            converted_series[target_name] = np.array(source, copy=True)

        length = _constant_dimension(
            trajectory["length"][source_index],
            name="length",
            source_index=source_index,
        )
        width = _constant_dimension(
            trajectory["width"][source_index],
            name="width",
            source_index=source_index,
        )
        object_type = _OBJECT_TYPE_MAP.get(
            int(object_metadata["object_types"][source_index]),
            AgentType.UNKNOWN,
        )
        agents.append(
            Agent(
                id=int(object_metadata["ids"][source_index]),
                type=object_type,
                valid=valid,
                x=converted_series["x"],
                y=converted_series["y"],
                heading=converted_series["heading"],
                vx=converted_series["vx"],
                vy=converted_series["vy"],
                length=length,
                width=width,
            )
        )
    return agents


def _constant_dimension(
    values: np.ndarray,
    *,
    name: str,
    source_index: int,
) -> float:
    values = np.asarray(values)
    if values.ndim != 1:
        _fail(
            "dimension_shape_invalid",
            f"object slot {source_index} {name} must be one-dimensional",
        )
    if (
        not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
        or not np.all(values == values[0])
    ):
        _fail(
            "dimension_not_constant",
            f"object slot {source_index} {name} must be a finite, positive, "
            "constant Waymax broadcast value",
        )
    return float(values[0])


def _convert_roadgraph(
    roadgraph: Any | None,
) -> tuple[list[MapPolyline], dict[str, Any]]:
    if roadgraph is None:
        return [], _empty_map_conversion()

    names = ("x", "y", "z", "dir_x", "dir_y", "dir_z", "types", "ids", "valid")
    arrays = {
        name: _as_array(getattr(roadgraph, name, None), f"roadgraph_points.{name}")
        for name in names
    }
    point_count = arrays["x"].shape
    if arrays["x"].ndim > 1:
        _fail("batched_state", f"roadgraph has batched shape {point_count}")
    if arrays["x"].ndim != 1:
        _fail(
            "roadgraph_shape_invalid",
            f"roadgraph fields must be [points], got {point_count}",
        )
    for name, array in arrays.items():
        if array.shape != point_count:
            _fail(
                "roadgraph_shape_invalid",
                f"roadgraph_points.{name} has shape {array.shape}, expected "
                f"{point_count}",
            )
    if arrays["valid"].dtype != np.bool_:
        _fail("roadgraph_dtype_invalid", "roadgraph valid must have boolean dtype")
    for name in ("types", "ids"):
        if not np.issubdtype(arrays[name].dtype, np.integer):
            _fail(
                "roadgraph_dtype_invalid",
                f"roadgraph {name} must have integer dtype",
            )

    valid = arrays["valid"]
    groups: OrderedDict[int, list[int]] = OrderedDict()
    for source_index in np.flatnonzero(valid):
        feature_id = int(arrays["ids"][source_index])
        groups.setdefault(feature_id, []).append(int(source_index))

    features: list[MapPolyline] = []
    retained_points = 0
    retained_by_type = {MapType.LANE.value: 0, MapType.ROAD_EDGE.value: 0}
    omitted_by_reason: dict[str, dict[str, int]] = {}
    unsupported_source_types: dict[str, dict[str, int]] = {}
    for source_indices in groups.values():
        indices = np.asarray(source_indices, dtype=int)
        source_types = arrays["types"][indices]
        unique_types = np.unique(source_types)
        if unique_types.size != 1:
            _record_map_omission(
                omitted_by_reason,
                "mixed_source_types",
                len(indices),
            )
            continue
        source_type = int(unique_types[0])
        if source_type in _LANE_TYPE_IDS:
            target_type = MapType.LANE
        elif source_type in _ROAD_EDGE_TYPE_IDS:
            target_type = MapType.ROAD_EDGE
        else:
            _record_map_omission(
                omitted_by_reason,
                "unsupported_map_type",
                len(indices),
            )
            _record_map_omission(
                unsupported_source_types,
                str(source_type),
                len(indices),
            )
            continue

        xy = np.column_stack((arrays["x"][indices], arrays["y"][indices])).astype(
            np.float64,
            copy=True,
        )
        directions = np.column_stack(
            (arrays["dir_x"][indices], arrays["dir_y"][indices])
        ).astype(np.float64, copy=True)
        omission = _map_gate_reason(xy, directions)
        if omission is not None:
            _record_map_omission(omitted_by_reason, omission, len(indices))
            continue
        features.append(MapPolyline(type=target_type, xy=xy))
        retained_points += len(indices)
        retained_by_type[target_type.value] += 1

    omitted_groups = sum(item["groups"] for item in omitted_by_reason.values())
    omitted_points = sum(item["points"] for item in omitted_by_reason.values())
    return features, {
        "source_point_count": int(arrays["x"].size),
        "valid_point_count": int(np.count_nonzero(valid)),
        "invalid_point_count": int(valid.size - np.count_nonzero(valid)),
        "valid_feature_group_count": len(groups),
        "retained_group_count": len(features),
        "retained_point_count": int(retained_points),
        "retained_groups_by_type": retained_by_type,
        "omitted_group_count": int(omitted_groups),
        "omitted_point_count": int(omitted_points),
        "omitted_by_reason": omitted_by_reason,
        "unsupported_source_types": unsupported_source_types,
    }


def _empty_map_conversion() -> dict[str, Any]:
    return {
        "source_point_count": 0,
        "valid_point_count": 0,
        "invalid_point_count": 0,
        "valid_feature_group_count": 0,
        "retained_group_count": 0,
        "retained_point_count": 0,
        "retained_groups_by_type": {
            MapType.LANE.value: 0,
            MapType.ROAD_EDGE.value: 0,
        },
        "omitted_group_count": 0,
        "omitted_point_count": 0,
        "omitted_by_reason": {},
        "unsupported_source_types": {},
    }


def _map_gate_reason(
    xy: np.ndarray,
    directions: np.ndarray,
) -> str | None:
    if not np.all(np.isfinite(xy)):
        return "non_finite_xy"
    if not np.all(np.isfinite(directions)):
        return "non_finite_direction"
    if xy.shape[0] < 2 or np.unique(xy, axis=0).shape[0] < 2:
        return "fewer_than_two_distinct_points"
    segments = np.diff(xy, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    if np.any(segment_lengths <= MIN_MAP_SEGMENT_METERS):
        return "segment_too_short"
    if np.any(segment_lengths > MAX_MAP_SEGMENT_METERS):
        return "segment_too_long"

    # Direction at the terminal point has no outgoing segment and is intentionally
    # ignored.  A zero direction there is valid in WOMD.
    nonterminal_directions = directions[:-1]
    direction_norms = np.linalg.norm(nonterminal_directions, axis=1)
    if np.any(direction_norms <= 0.0):
        return "zero_nonterminal_direction"
    tangents = segments / segment_lengths[:, np.newaxis]
    unit_directions = nonterminal_directions / direction_norms[:, np.newaxis]
    cosines = np.sum(tangents * unit_directions, axis=1)
    threshold = math.cos(math.radians(MAX_MAP_DIRECTION_ERROR_DEGREES))
    if np.any(cosines < threshold):
        return "direction_misaligned"
    return None


def _record_map_omission(
    counts: dict[str, dict[str, int]],
    reason: str,
    point_count: int,
) -> None:
    item = counts.setdefault(reason, {"groups": 0, "points": 0})
    item["groups"] += 1
    item["points"] += int(point_count)


def _wrap_heading(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _as_array(value: Any, name: str) -> np.ndarray:
    if value is None:
        _fail("state_contract_invalid", f"{name} is required")
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise WaymaxConversionError(
            "state_contract_invalid",
            f"{name} must be an eager array",
        ) from exc


def _json_native_tree(value: Any, *, context: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail("json_metadata_invalid", f"{context} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _fail("json_metadata_invalid", f"{context} has a non-string key")
            normalized[key] = _json_native_tree(
                item,
                context=f"{context}.{key}",
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _json_native_tree(item, context=f"{context}[]")
            for item in value
        ]
    _fail(
        "json_metadata_invalid",
        f"{context} contains non-JSON-native type {type(value).__name__}",
    )


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _fail(code: str, message: str) -> NoReturn:
    raise WaymaxConversionError(code, message)


__all__ = [
    "DEFAULT_WAYMAX_TEMPORAL_PROFILE",
    "MAX_MAP_DIRECTION_ERROR_DEGREES",
    "MAX_MAP_SEGMENT_METERS",
    "MIN_MAP_SEGMENT_METERS",
    "WAYMAX_ADAPTER_SCHEMA_VERSION",
    "WAYMAX_ADAPTER_VERSION",
    "WAYMAX_COMMIT",
    "WOMD_DATASET_VERSION",
    "WaymaxConversionError",
    "WaymaxTemporalProfile",
    "scenario_from_waymax_state",
    "waymax_adapter_fingerprint",
]
