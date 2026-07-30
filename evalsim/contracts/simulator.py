"""Seam B: typed, access-audited ``SimulatorPolicy`` contracts.

The rollout engine owns time, validity-mask lifecycle, ego control, and kinematic
integration. History-only policies receive observed history plus realized current
state, while explicitly privileged policies may receive a defensive immutable copy of
the complete logged reference. Both act synchronously on one immutable current frame.
"""
from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .scenario import Scenario
from .types import AgentType, MapType

_NUMERIC_FRAME_FIELDS = ("x", "y", "heading", "vx", "vy")
_SOURCE_PROVENANCE_KEYS = ("source", "source_version", "source_time_unit")


def _readonly_array(
    value: Any,
    *,
    dtype: Any,
    name: str,
    ndim: int,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError(
            f"{name} must be a {ndim}-D array, got shape {array.shape}"
        )
    # A regular owning ndarray can reverse ``setflags(write=False)``. Back the public
    # view with immutable ``bytes`` so even a hostile policy cannot re-enable writes.
    immutable = np.frombuffer(
        array.tobytes(order="C"),
        dtype=array.dtype,
    ).reshape(array.shape)
    immutable.setflags(write=False)
    return immutable


def _readonly_vector(value: Any, *, dtype: Any, name: str) -> np.ndarray:
    return _readonly_array(value, dtype=dtype, name=name, ndim=1)


def _freeze_json(value: Any, *, path: str) -> Any:
    """Recursively freeze a JSON value and reject ambiguous/coerced keys."""

    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(
        f"{path} must contain only JSON-compatible values, got "
        f"{type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return copy.deepcopy(value)


def _freeze_source_provenance(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("scenario metadata must be a mapping")
    selected = {
        key: metadata[key]
        for key in _SOURCE_PROVENANCE_KEYS
        if key in metadata
    }
    frozen = _freeze_json(selected, path="source_provenance")
    # This boundary must never preserve non-standard NaN/Infinity JSON values.
    json.dumps(_thaw_json(frozen), allow_nan=False, sort_keys=True)
    return frozen


def _validate_policy_static_fields(
    *,
    scenario_id: str,
    agent_ids: tuple[int, ...],
    agent_types: tuple[AgentType, ...],
    lengths: np.ndarray,
    widths: np.ndarray,
    ego_index: int,
    timestamps: np.ndarray,
    frames: tuple["AgentFrame", ...],
    current_index: int,
    future_step_count: int,
) -> None:
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be a non-empty string")
    count = len(agent_ids)
    if count < 1:
        raise ValueError("policy context must contain at least one agent")
    if any(
        isinstance(agent_id, (bool, np.bool_))
        or not isinstance(agent_id, (int, np.integer))
        for agent_id in agent_ids
    ):
        raise ValueError("policy context agent_ids must be integers")
    if len(set(agent_ids)) != count:
        raise ValueError("policy context agent_ids must be unique")
    if any(
        len(value) != count
        for value in (agent_types, lengths, widths)
    ):
        raise ValueError("policy context static fields must share agent count")
    if (
        not np.all(np.isfinite(lengths))
        or not np.all(np.isfinite(widths))
        or np.any(lengths <= 0.0)
        or np.any(widths <= 0.0)
    ):
        raise ValueError("policy context dimensions must be finite and positive")
    if (
        isinstance(ego_index, (bool, np.bool_))
        or not isinstance(ego_index, (int, np.integer))
        or not 0 <= int(ego_index) < count
    ):
        raise ValueError("policy context ego_index must be an in-range integer")
    if (
        isinstance(current_index, (bool, np.bool_))
        or not isinstance(current_index, (int, np.integer))
        or int(current_index) < 0
    ):
        raise ValueError("policy context current_index must be non-negative")
    if (
        isinstance(future_step_count, (bool, np.bool_))
        or not isinstance(future_step_count, (int, np.integer))
        or int(future_step_count) < 0
    ):
        raise ValueError("policy context future_step_count must be non-negative")
    if len(timestamps) != len(frames):
        raise ValueError("policy context timestamps and frames must share horizon")
    if not frames:
        raise ValueError("policy context must contain at least one frame")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("policy context timestamps must be finite")
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0.0):
        raise ValueError(
            "policy context timestamps must be strictly increasing"
        )
    if any(
        not isinstance(frame, AgentFrame) or frame.num_agents != count
        for frame in frames
    ):
        raise ValueError("policy context frames must match agent count")


@dataclass(frozen=True, slots=True)
class PolicyMapFeature:
    """Immutable typed static map feature exposed at policy initialization."""

    type: MapType
    xy: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", MapType(self.type))
        xy = _readonly_array(
            self.xy,
            dtype=float,
            name="PolicyMapFeature.xy",
            ndim=2,
        )
        if xy.shape[1] != 2:
            raise ValueError(
                f"PolicyMapFeature.xy must be [P, 2], got {xy.shape}"
            )
        object.__setattr__(self, "xy", xy)


@dataclass(frozen=True, slots=True)
class AgentFrame:
    """Immutable per-agent state at one timestamp.

    Every array has shape ``[A]`` in scenario agent order. Invalid agents retain a
    finite payload for serialization, but policies must gate behavior on ``valid``.
    """

    valid: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    vx: np.ndarray
    vy: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "valid",
            _readonly_vector(self.valid, dtype=bool, name="AgentFrame.valid"),
        )
        for name in _NUMERIC_FRAME_FIELDS:
            object.__setattr__(
                self,
                name,
                _readonly_vector(
                    getattr(self, name),
                    dtype=float,
                    name=f"AgentFrame.{name}",
                ),
            )

        shapes = {
            getattr(self, name).shape
            for name in ("valid", *_NUMERIC_FRAME_FIELDS)
        }
        if len(shapes) != 1:
            raise ValueError(
                f"AgentFrame arrays must share one shape, got {sorted(shapes)}"
            )
        if any(
            not np.all(np.isfinite(getattr(self, name)))
            for name in _NUMERIC_FRAME_FIELDS
        ):
            raise ValueError("AgentFrame numeric arrays must be finite")
        if np.any(self.heading < -np.pi) or np.any(self.heading > np.pi):
            raise ValueError("AgentFrame headings must be in [-pi, pi]")

    @classmethod
    def from_scenario(cls, scenario: Scenario, step_index: int) -> "AgentFrame":
        """Copy one frame from a scenario without aliasing its arrays."""

        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, (int, np.integer))
            or not 0 <= int(step_index) < scenario.num_steps
        ):
            raise ValueError(
                f"step_index {step_index!r} is outside scenario horizon "
                f"[0, {scenario.num_steps})"
            )
        step_index = int(step_index)
        return cls(
            valid=np.array(
                [agent.valid[step_index] for agent in scenario.agents],
                dtype=bool,
            ),
            x=np.array([agent.x[step_index] for agent in scenario.agents]),
            y=np.array([agent.y[step_index] for agent in scenario.agents]),
            heading=np.array(
                [agent.heading[step_index] for agent in scenario.agents]
            ),
            vx=np.array([agent.vx[step_index] for agent in scenario.agents]),
            vy=np.array([agent.vy[step_index] for agent in scenario.agents]),
        )

    @property
    def num_agents(self) -> int:
        return int(self.valid.shape[0])

    def speed(self) -> np.ndarray:
        """Return a fresh scalar-speed vector."""

        return np.hypot(self.vx, self.vy)


def _copy_agent_frame(frame: AgentFrame) -> AgentFrame:
    if not isinstance(frame, AgentFrame):
        raise TypeError("policy context frames must contain AgentFrame values")
    return AgentFrame(
        valid=frame.valid,
        x=frame.x,
        y=frame.y,
        heading=frame.heading,
        vx=frame.vx,
        vy=frame.vy,
    )


def _copy_map_features(
    features: tuple[PolicyMapFeature, ...],
) -> tuple[PolicyMapFeature, ...]:
    copied: list[PolicyMapFeature] = []
    for feature in tuple(features):
        if not isinstance(feature, PolicyMapFeature):
            raise TypeError(
                "policy context map_features must contain PolicyMapFeature values"
            )
        copied.append(PolicyMapFeature(type=feature.type, xy=feature.xy))
    return tuple(copied)


def _normalize_context_provenance(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("source_provenance must be a mapping")
    unexpected = set(value) - set(_SOURCE_PROVENANCE_KEYS)
    if unexpected:
        raise ValueError(
            "source_provenance contains non-allowlisted keys: "
            f"{sorted(unexpected)}"
        )
    return _freeze_source_provenance(value)


def _scenario_current_index(scenario: Scenario) -> int:
    raw = scenario.metadata.get("current_index", 0)
    if (
        isinstance(raw, (bool, np.bool_))
        or not isinstance(raw, (int, np.integer))
        or not 0 <= int(raw) < scenario.num_steps
    ):
        raise ValueError(
            "scenario.metadata['current_index'] must be an integer inside "
            "the scenario horizon"
        )
    return int(raw)


def _scenario_static_fields(
    scenario: Scenario,
) -> tuple[
    tuple[int, ...],
    tuple[AgentType, ...],
    np.ndarray,
    np.ndarray,
    tuple[PolicyMapFeature, ...],
]:
    return (
        tuple(agent.id for agent in scenario.agents),
        tuple(agent.type for agent in scenario.agents),
        np.asarray([agent.length for agent in scenario.agents], dtype=float),
        np.asarray([agent.width for agent in scenario.agents], dtype=float),
        tuple(
            PolicyMapFeature(type=feature.type, xy=feature.xy)
            for feature in scenario.map
        ),
    )


@dataclass(frozen=True, slots=True)
class HistoryOnlyPolicyContext:
    """Immutable initialization input with no post-current motion payload.

    ``frames`` and ``timestamps`` end exactly at ``current_index``.
    ``future_step_count`` communicates only the remaining horizon length.
    """

    scenario_id: str
    agent_ids: tuple[int, ...]
    agent_types: tuple[AgentType, ...]
    lengths: np.ndarray
    widths: np.ndarray
    ego_index: int
    map_features: tuple[PolicyMapFeature, ...]
    timestamps: np.ndarray
    frames: tuple[AgentFrame, ...]
    current_index: int
    future_step_count: int
    source_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        agent_ids = tuple(self.agent_ids)
        agent_types = tuple(AgentType(value) for value in self.agent_types)
        lengths = _readonly_vector(
            self.lengths,
            dtype=float,
            name="HistoryOnlyPolicyContext.lengths",
        )
        widths = _readonly_vector(
            self.widths,
            dtype=float,
            name="HistoryOnlyPolicyContext.widths",
        )
        timestamps = _readonly_vector(
            self.timestamps,
            dtype=float,
            name="HistoryOnlyPolicyContext.timestamps",
        )
        frames = tuple(_copy_agent_frame(frame) for frame in self.frames)
        map_features = _copy_map_features(tuple(self.map_features))
        current_index = int(self.current_index)
        future_step_count = int(self.future_step_count)
        _validate_policy_static_fields(
            scenario_id=self.scenario_id,
            agent_ids=agent_ids,
            agent_types=agent_types,
            lengths=lengths,
            widths=widths,
            ego_index=self.ego_index,
            timestamps=timestamps,
            frames=frames,
            current_index=self.current_index,
            future_step_count=self.future_step_count,
        )
        if len(frames) != current_index + 1:
            raise ValueError(
                "history-only frames must end exactly at current_index"
            )

        object.__setattr__(self, "agent_ids", tuple(int(v) for v in agent_ids))
        object.__setattr__(self, "agent_types", agent_types)
        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "widths", widths)
        object.__setattr__(self, "ego_index", int(self.ego_index))
        object.__setattr__(self, "map_features", map_features)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "current_index", current_index)
        object.__setattr__(self, "future_step_count", future_step_count)
        object.__setattr__(
            self,
            "source_provenance",
            _normalize_context_provenance(self.source_provenance),
        )

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "HistoryOnlyPolicyContext":
        """Build the audited view without retaining ``scenario`` itself."""

        if not isinstance(scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        current_index = _scenario_current_index(scenario)
        ids, types, lengths, widths, map_features = _scenario_static_fields(
            scenario
        )
        return cls(
            scenario_id=scenario.scenario_id,
            agent_ids=ids,
            agent_types=types,
            lengths=lengths,
            widths=widths,
            ego_index=scenario.ego_index,
            map_features=map_features,
            timestamps=scenario.timestamps[: current_index + 1],
            frames=tuple(
                AgentFrame.from_scenario(scenario, step_index)
                for step_index in range(current_index + 1)
            ),
            current_index=current_index,
            future_step_count=scenario.num_steps - current_index - 1,
            source_provenance=_freeze_source_provenance(scenario.metadata),
        )


@dataclass(frozen=True, slots=True)
class PrivilegedPolicyContext:
    """Immutable initialization input containing the complete logged reference.

    This is an explicit privileged capability. It carries all logged frames and
    timestamps but never aliases the caller's mutable :class:`Scenario`.
    """

    scenario_id: str
    agent_ids: tuple[int, ...]
    agent_types: tuple[AgentType, ...]
    lengths: np.ndarray
    widths: np.ndarray
    ego_index: int
    map_features: tuple[PolicyMapFeature, ...]
    timestamps: np.ndarray
    frames: tuple[AgentFrame, ...]
    current_index: int
    future_step_count: int
    source_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        agent_ids = tuple(self.agent_ids)
        agent_types = tuple(AgentType(value) for value in self.agent_types)
        lengths = _readonly_vector(
            self.lengths,
            dtype=float,
            name="PrivilegedPolicyContext.lengths",
        )
        widths = _readonly_vector(
            self.widths,
            dtype=float,
            name="PrivilegedPolicyContext.widths",
        )
        timestamps = _readonly_vector(
            self.timestamps,
            dtype=float,
            name="PrivilegedPolicyContext.timestamps",
        )
        frames = tuple(_copy_agent_frame(frame) for frame in self.frames)
        map_features = _copy_map_features(tuple(self.map_features))
        current_index = int(self.current_index)
        future_step_count = int(self.future_step_count)
        _validate_policy_static_fields(
            scenario_id=self.scenario_id,
            agent_ids=agent_ids,
            agent_types=agent_types,
            lengths=lengths,
            widths=widths,
            ego_index=self.ego_index,
            timestamps=timestamps,
            frames=frames,
            current_index=self.current_index,
            future_step_count=self.future_step_count,
        )
        if len(frames) != current_index + future_step_count + 1:
            raise ValueError(
                "privileged frames must contain the complete reference horizon"
            )

        object.__setattr__(self, "agent_ids", tuple(int(v) for v in agent_ids))
        object.__setattr__(self, "agent_types", agent_types)
        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "widths", widths)
        object.__setattr__(self, "ego_index", int(self.ego_index))
        object.__setattr__(self, "map_features", map_features)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "current_index", current_index)
        object.__setattr__(self, "future_step_count", future_step_count)
        object.__setattr__(
            self,
            "source_provenance",
            _normalize_context_provenance(self.source_provenance),
        )

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "PrivilegedPolicyContext":
        """Build a complete defensive copy for an explicitly privileged policy."""

        if not isinstance(scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        current_index = _scenario_current_index(scenario)
        ids, types, lengths, widths, map_features = _scenario_static_fields(
            scenario
        )
        return cls(
            scenario_id=scenario.scenario_id,
            agent_ids=ids,
            agent_types=types,
            lengths=lengths,
            widths=widths,
            ego_index=scenario.ego_index,
            map_features=map_features,
            timestamps=scenario.timestamps,
            frames=tuple(
                AgentFrame.from_scenario(scenario, step_index)
                for step_index in range(scenario.num_steps)
            ),
            current_index=current_index,
            future_step_count=scenario.num_steps - current_index - 1,
            source_provenance=_freeze_source_provenance(scenario.metadata),
        )


@dataclass(frozen=True, slots=True)
class HistoryOnlyPolicyObservation:
    """One synchronous realized-state observation with no future lifecycle."""

    current_index: int
    next_index: int
    timestamp: float
    next_timestamp: float
    dt: float
    frame: AgentFrame
    agent_ids: tuple[int, ...]
    agent_types: tuple[AgentType, ...]
    lengths: np.ndarray
    widths: np.ndarray
    ego_index: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.current_index, (bool, np.bool_))
            or not isinstance(self.current_index, (int, np.integer))
            or int(self.current_index) < 0
        ):
            raise ValueError("current_index must be a non-negative integer")
        if (
            isinstance(self.next_index, (bool, np.bool_))
            or not isinstance(self.next_index, (int, np.integer))
            or int(self.next_index) != int(self.current_index) + 1
        ):
            raise ValueError("next_index must equal current_index + 1")
        object.__setattr__(self, "current_index", int(self.current_index))
        object.__setattr__(self, "next_index", int(self.next_index))

        for name in ("timestamp", "next_timestamp", "dt"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if not np.isclose(
            self.next_timestamp - self.timestamp,
            self.dt,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("dt must equal next_timestamp - timestamp")

        raw_agent_ids = tuple(self.agent_ids)
        if any(
            isinstance(agent_id, (bool, np.bool_))
            or not isinstance(agent_id, (int, np.integer))
            for agent_id in raw_agent_ids
        ):
            raise ValueError(
                "HistoryOnlyPolicyObservation agent_ids must be integers"
            )
        object.__setattr__(
            self,
            "agent_ids",
            tuple(int(agent_id) for agent_id in raw_agent_ids),
        )
        object.__setattr__(
            self,
            "agent_types",
            tuple(AgentType(agent_type) for agent_type in self.agent_types),
        )
        object.__setattr__(
            self,
            "lengths",
            _readonly_vector(
                self.lengths,
                dtype=float,
                name="HistoryOnlyPolicyObservation.lengths",
            ),
        )
        object.__setattr__(
            self,
            "widths",
            _readonly_vector(
                self.widths,
                dtype=float,
                name="HistoryOnlyPolicyObservation.widths",
            ),
        )

        count = self.frame.num_agents
        if any(
            len(value) != count
            for value in (
                self.agent_ids,
                self.agent_types,
                self.lengths,
                self.widths,
            )
        ):
            raise ValueError(
                "HistoryOnlyPolicyObservation fields must match frame agent count"
            )
        if len(set(self.agent_ids)) != count:
            raise ValueError(
                "HistoryOnlyPolicyObservation agent_ids must be unique"
            )
        if (
            not np.all(np.isfinite(self.lengths))
            or not np.all(np.isfinite(self.widths))
            or np.any(self.lengths <= 0.0)
            or np.any(self.widths <= 0.0)
        ):
            raise ValueError(
                "HistoryOnlyPolicyObservation dimensions must be finite and positive"
            )
        if (
            isinstance(self.ego_index, (bool, np.bool_))
            or not isinstance(self.ego_index, (int, np.integer))
            or not 0 <= int(self.ego_index) < count
        ):
            raise ValueError(
                "PolicyObservation ego_index must be an in-range integer"
            )
        object.__setattr__(self, "ego_index", int(self.ego_index))


# Deliberate source-compatibility path for callers importing the pre-M6 name. The
# aliased contract has M6 semantics: in particular, it has no ``next_valid`` field.
PolicyObservation = HistoryOnlyPolicyObservation


@dataclass(frozen=True, slots=True)
class PolicyStep:
    """A policy's proposed synchronous transition.

    Controls are longitudinal acceleration and yaw rate for point-mass integration.
    ``override`` plus ``override_mask`` supplies absolute next states (for example,
    log replay). Overrides bypass dynamics clamps. The engine always owns lifecycle
    masks and logged-ego control, regardless of what a policy proposes.
    """

    next_state: Any
    longitudinal_acceleration: np.ndarray
    yaw_rate: np.ndarray
    override: AgentFrame | None = None
    override_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        acceleration = _readonly_vector(
            self.longitudinal_acceleration,
            dtype=float,
            name="PolicyStep.longitudinal_acceleration",
        )
        yaw_rate = _readonly_vector(
            self.yaw_rate,
            dtype=float,
            name="PolicyStep.yaw_rate",
        )
        if acceleration.shape != yaw_rate.shape:
            raise ValueError(
                "PolicyStep acceleration and yaw_rate must share one shape"
            )
        if not np.all(np.isfinite(acceleration)):
            raise ValueError("PolicyStep acceleration must be finite")
        if not np.all(np.isfinite(yaw_rate)):
            raise ValueError("PolicyStep yaw_rate must be finite")
        object.__setattr__(self, "longitudinal_acceleration", acceleration)
        object.__setattr__(self, "yaw_rate", yaw_rate)

        if self.override_mask is None:
            override_mask = _readonly_vector(
                np.zeros(acceleration.shape, dtype=bool),
                dtype=bool,
                name="PolicyStep.override_mask",
            )
        else:
            override_mask = _readonly_vector(
                self.override_mask,
                dtype=bool,
                name="PolicyStep.override_mask",
            )
        object.__setattr__(self, "override_mask", override_mask)
        if override_mask.shape != acceleration.shape:
            raise ValueError("PolicyStep override_mask must match control shape")
        if np.any(override_mask) and self.override is None:
            raise ValueError("PolicyStep override_mask requires an override frame")
        if (
            self.override is not None
            and self.override.num_agents != acceleration.shape[0]
        ):
            raise ValueError("PolicyStep override frame must match control shape")

    @property
    def num_agents(self) -> int:
        return int(self.longitudinal_acceleration.shape[0])


@dataclass(frozen=True, slots=True)
class PolicyMetadata:
    """Validated, immutable simulator self-description and config snapshot."""

    name: str
    version: str
    deterministic: bool
    required_features: tuple[str, ...] = ()
    supported_agent_types: tuple[AgentType, ...] = (AgentType.VEHICLE,)
    params: Mapping[str, Any] = field(default_factory=dict)
    known_limitations: tuple[str, ...] = ()
    fallback_policy: str | None = None

    def __post_init__(self) -> None:
        for name in ("name", "version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"PolicyMetadata.{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        if not isinstance(self.deterministic, bool):
            raise ValueError("PolicyMetadata.deterministic must be bool")

        required_features = tuple(self.required_features)
        known_limitations = tuple(self.known_limitations)
        if any(
            not isinstance(item, str) or not item
            for item in (*required_features, *known_limitations)
        ):
            raise ValueError(
                "PolicyMetadata features and limitations must be non-empty strings"
            )
        object.__setattr__(self, "required_features", required_features)
        object.__setattr__(self, "known_limitations", known_limitations)

        supported = tuple(AgentType(item) for item in self.supported_agent_types)
        if len(set(supported)) != len(supported):
            raise ValueError("PolicyMetadata supported_agent_types must be unique")
        object.__setattr__(self, "supported_agent_types", supported)

        if not isinstance(self.params, Mapping):
            raise ValueError("PolicyMetadata.params must be a mapping")
        object.__setattr__(
            self,
            "params",
            _freeze_json(self.params, path="PolicyMetadata.params"),
        )

        if self.fallback_policy is not None and (
            not isinstance(self.fallback_policy, str)
            or not self.fallback_policy.strip()
        ):
            raise ValueError(
                "PolicyMetadata.fallback_policy must be None or a non-empty string"
            )
        if isinstance(self.fallback_policy, str):
            object.__setattr__(
                self,
                "fallback_policy",
                self.fallback_policy.strip(),
            )

        # Fail at the policy boundary rather than much later during Parquet metadata.
        json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-ready provenance snapshot."""

        return {
            "name": self.name,
            "version": self.version,
            "deterministic": self.deterministic,
            "required_features": list(self.required_features),
            "supported_agent_types": [
                agent_type.value for agent_type in self.supported_agent_types
            ],
            "params": _thaw_json(self.params),
            "known_limitations": list(self.known_limitations),
            "fallback_policy": self.fallback_policy,
        }


class SimulatorPolicy(ABC):
    """Common abstract world-agent simulator surface.

    Initialization is intentionally absent here. A runnable policy must declare exactly
    one audited initialization capability by deriving from
    :class:`HistoryOnlySimulatorPolicy` or :class:`PrivilegedSimulatorPolicy`.
    ``step`` computes all agents' proposals from the same immutable realized-state
    observation. Policies must not store run-local state on the policy instance, so one
    instance can be safely reused and interleaved across rollouts.
    """

    @abstractmethod
    def step(
        self,
        state: Any,
        observation: HistoryOnlyPolicyObservation,
    ) -> PolicyStep:
        """Propose the next synchronous world-agent transition."""

    @abstractmethod
    def metadata(self) -> PolicyMetadata:
        """Return immutable simulator metadata and configuration."""


class HistoryOnlySimulatorPolicy(SimulatorPolicy):
    """Policy capability initialized without post-current logged motion."""

    @abstractmethod
    def initialize(
        self,
        context: HistoryOnlyPolicyContext,
        seed: int,
    ) -> Any:
        """Return opaque run-local state from an audited history-only context."""


class PrivilegedSimulatorPolicy(SimulatorPolicy):
    """Policy capability explicitly initialized with the full logged reference."""

    @abstractmethod
    def initialize(
        self,
        context: PrivilegedPolicyContext,
        seed: int,
    ) -> Any:
        """Return opaque run-local state from a privileged full-reference context."""
