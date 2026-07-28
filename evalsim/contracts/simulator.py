"""Seam B: the typed ``SimulatorPolicy`` transition contract.

The rollout engine owns time, validity-mask lifecycle, ego control, and kinematic
integration. Policies synchronously inspect one immutable frame and return either
per-agent controls or an explicit absolute-state override. The latter is what makes
log replay exact without coupling the engine to a concrete policy class.
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
from .types import AgentType

_NUMERIC_FRAME_FIELDS = ("x", "y", "heading", "vx", "vy")


def _readonly_vector(value: Any, *, dtype: Any, name: str) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {array.shape}")
    # A regular owning ndarray can reverse ``setflags(write=False)``. Back the public
    # view with immutable ``bytes`` so even a hostile policy cannot re-enable writes.
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype)
    immutable.setflags(write=False)
    return immutable


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


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    """One synchronous transition observation supplied by the rollout engine."""

    current_index: int
    next_index: int
    timestamp: float
    next_timestamp: float
    dt: float
    frame: AgentFrame
    next_valid: np.ndarray
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

        next_valid = _readonly_vector(
            self.next_valid,
            dtype=bool,
            name="PolicyObservation.next_valid",
        )
        object.__setattr__(self, "next_valid", next_valid)
        raw_agent_ids = tuple(self.agent_ids)
        if any(
            isinstance(agent_id, (bool, np.bool_))
            or not isinstance(agent_id, (int, np.integer))
            for agent_id in raw_agent_ids
        ):
            raise ValueError("PolicyObservation agent_ids must be integers")
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
                name="PolicyObservation.lengths",
            ),
        )
        object.__setattr__(
            self,
            "widths",
            _readonly_vector(
                self.widths,
                dtype=float,
                name="PolicyObservation.widths",
            ),
        )

        count = self.frame.num_agents
        if any(
            len(value) != count
            for value in (
                self.next_valid,
                self.agent_ids,
                self.agent_types,
                self.lengths,
                self.widths,
            )
        ):
            raise ValueError("PolicyObservation fields must match frame agent count")
        if len(set(self.agent_ids)) != count:
            raise ValueError("PolicyObservation agent_ids must be unique")
        if (
            not np.all(np.isfinite(self.lengths))
            or not np.all(np.isfinite(self.widths))
            or np.any(self.lengths <= 0.0)
            or np.any(self.widths <= 0.0)
        ):
            raise ValueError("PolicyObservation dimensions must be finite and positive")
        if (
            isinstance(self.ego_index, (bool, np.bool_))
            or not isinstance(self.ego_index, (int, np.integer))
            or not 0 <= int(self.ego_index) < count
        ):
            raise ValueError(
                "PolicyObservation ego_index must be an in-range integer"
            )
        object.__setattr__(self, "ego_index", int(self.ego_index))


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
    """Abstract world-agent simulator.

    ``initialize`` returns run-local opaque memory. ``step`` must compute all agents'
    proposals from the same immutable observation and return a :class:`PolicyStep`.
    Policies must not store run-local state on the policy instance, so one instance can
    be safely reused and interleaved across rollouts.
    """

    @abstractmethod
    def initialize(self, scenario: Scenario, seed: int) -> Any:
        """Return opaque, run-local policy state."""

    @abstractmethod
    def step(
        self,
        state: Any,
        observation: PolicyObservation,
    ) -> PolicyStep:
        """Propose the next synchronous world-agent transition."""

    @abstractmethod
    def metadata(self) -> PolicyMetadata:
        """Return immutable simulator metadata and configuration."""
