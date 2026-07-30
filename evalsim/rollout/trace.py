"""Immutable policy-execution sidecars for audited M6 rollouts.

The sidecar is deliberately separate from :class:`~evalsim.contracts.Rollout` so the
legacy no-plan serialization and metadata remain byte-for-byte stable.  It records
only agent-order tensors and bounded policy provenance; source/native identities are
never copied into it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import struct
from typing import Any

import numpy as np

from evalsim.contracts import AgentFrame, Rollout
from evalsim.contracts.counterfactual import RolloutSnapshot

from .dynamics import ClampCounts, DynamicsLimits, integrate_point_mass

M6_POLICY_TRACE_SCHEMA_VERSION = "1.0.0"
_TRACE_DOMAIN = b"evalsim-m6-policy-execution-trace-v1"
_FLOAT_FIELDS = (
    "longitudinal_acceleration",
    "yaw_rate",
    "override_x",
    "override_y",
    "override_heading",
    "override_vx",
    "override_vy",
)
_BOOL_FIELDS = (
    "override_mask",
    "override_valid",
    "effective_control_mask",
    "lifecycle_birth_mask",
)
_ACCESS_ROLES = frozenset({"history_only", "privileged"})
_SHA256_HEX = frozenset("0123456789abcdef")
_STATE_FIELDS = ("x", "y", "heading", "vx", "vy")
_CLAMP_FIELDS = (
    "acceleration",
    "deceleration",
    "speed",
    "yaw_rate",
    "reverse_prevented",
)


def _u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _readonly_vector(
    value: Any,
    *,
    dtype: np.dtype[Any] | type[Any],
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype)
    frozen.setflags(write=False)
    return frozen


def _readonly_matrix(
    value: Any,
    *,
    dtype: np.dtype[Any] | type[Any],
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    frozen = np.frombuffer(
        array.tobytes(order="C"),
        dtype=array.dtype,
    ).reshape(array.shape)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, slots=True)
class PolicyExecutionTrace:
    """Exact agent-order controls and masks emitted during one rollout."""

    policy_name: str
    policy_version: str
    policy_access_role: str
    start_index: int
    stop_index: int
    ego_index: int
    timestamps: np.ndarray
    control_modes: tuple[str, ...]
    longitudinal_acceleration: np.ndarray
    yaw_rate: np.ndarray
    override_mask: np.ndarray
    override_valid: np.ndarray
    override_x: np.ndarray
    override_y: np.ndarray
    override_heading: np.ndarray
    override_vx: np.ndarray
    override_vy: np.ndarray
    effective_control_mask: np.ndarray
    lifecycle_birth_mask: np.ndarray
    perturbation_identity: str | None
    rollout_fingerprint: str
    _integrity_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_name",
            _text(self.policy_name, name="policy_name"),
        )
        object.__setattr__(
            self,
            "policy_version",
            _text(self.policy_version, name="policy_version"),
        )
        role = _text(self.policy_access_role, name="policy_access_role")
        if role not in _ACCESS_ROLES:
            raise ValueError("policy_access_role is not registered")
        object.__setattr__(self, "policy_access_role", role)
        start = _integer(self.start_index, name="start_index")
        stop = _integer(self.stop_index, name="stop_index")
        if stop <= start:
            raise ValueError("stop_index must follow start_index")
        object.__setattr__(self, "start_index", start)
        object.__setattr__(self, "stop_index", stop)

        timestamps = _readonly_vector(
            self.timestamps,
            dtype=np.dtype("<f8"),
            name="timestamps",
        )
        transition_count = stop - start
        if timestamps.shape != (transition_count + 1,):
            raise ValueError(
                "trace timestamps must span current through stop inclusive"
            )
        if not np.all(np.isfinite(timestamps)) or np.any(
            np.diff(timestamps) <= 0.0
        ):
            raise ValueError("trace timestamps must be finite and increasing")
        object.__setattr__(self, "timestamps", timestamps)

        modes = tuple(
            _text(value, name="control mode") for value in self.control_modes
        )
        if not modes:
            raise ValueError("control_modes must not be empty")
        object.__setattr__(self, "control_modes", modes)
        ego_index = _integer(self.ego_index, name="ego_index")
        if ego_index >= len(modes):
            raise ValueError("ego_index must be inside agent order")
        object.__setattr__(self, "ego_index", ego_index)
        shape = (transition_count, len(modes))
        for name in _FLOAT_FIELDS:
            values = _readonly_matrix(
                getattr(self, name),
                dtype=np.dtype("<f8"),
                name=name,
            )
            if values.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
            object.__setattr__(self, name, values)
        for name in _BOOL_FIELDS:
            values = _readonly_matrix(
                getattr(self, name),
                dtype=np.bool_,
                name=name,
            )
            if values.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            object.__setattr__(self, name, values)
        if np.any(self.lifecycle_birth_mask & self.effective_control_mask):
            raise ValueError("birth fallback and effective policy control must be disjoint")
        if np.any(self.override_valid != (self.override_mask & self.override_valid)):
            raise ValueError("override_valid must be false outside override_mask")
        for name in (
            "override_mask",
            "override_valid",
            "effective_control_mask",
            "lifecycle_birth_mask",
        ):
            if np.any(getattr(self, name)[:, self.ego_index]):
                raise ValueError(f"{name} must be false for the engine-owned ego")
        if self.policy_access_role == "history_only" and np.any(
            self.override_mask
        ):
            raise ValueError(
                "history-only policy traces cannot contain absolute overrides"
            )
        if self.perturbation_identity is not None:
            object.__setattr__(
                self,
                "perturbation_identity",
                _text(
                    self.perturbation_identity,
                    name="perturbation_identity",
                ),
            )
        if (
            not isinstance(self.rollout_fingerprint, str)
            or len(self.rollout_fingerprint) != 64
            or any(character not in _SHA256_HEX for character in self.rollout_fingerprint)
        ):
            raise ValueError(
                "rollout_fingerprint must be a lowercase SHA-256 digest"
            )
        expected_ego_mode = (
            "logged_ego"
            if self.perturbation_identity is None
            else "typed_ego_plan"
        )
        if self.control_modes[self.ego_index] != expected_ego_mode:
            raise ValueError("trace ego control mode contradicts plan provenance")
        object.__setattr__(
            self,
            "_integrity_fingerprint",
            self._compute_integrity_fingerprint(),
        )

    @property
    def transition_count(self) -> int:
        return self.stop_index - self.start_index

    @property
    def agent_count(self) -> int:
        return len(self.control_modes)

    @property
    def integrity_fingerprint(self) -> str:
        self.revalidate()
        return self._integrity_fingerprint

    def _compute_integrity_fingerprint(self) -> str:
        metadata = json.dumps(
            {
                "control_modes": list(self.control_modes),
                "ego_index": self.ego_index,
                "perturbation_identity": self.perturbation_identity,
                "policy_access_role": self.policy_access_role,
                "policy_name": self.policy_name,
                "policy_version": self.policy_version,
                "rollout_fingerprint": self.rollout_fingerprint,
                "schema_version": M6_POLICY_TRACE_SCHEMA_VERSION,
                "start_index": self.start_index,
                "stop_index": self.stop_index,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        parts = [
            metadata,
            np.asarray(self.timestamps, dtype="<f8").tobytes(order="C"),
            *(
                np.asarray(getattr(self, name), dtype="<f8").tobytes(order="C")
                for name in _FLOAT_FIELDS
            ),
            *(
                np.asarray(getattr(self, name), dtype=np.uint8).tobytes(order="C")
                for name in _BOOL_FIELDS
            ),
        ]
        digest = hashlib.sha256()
        digest.update(_TRACE_DOMAIN)
        digest.update(b"\x00")
        for part in parts:
            digest.update(_u64(len(part)))
            digest.update(part)
        return digest.hexdigest()

    def revalidate(self) -> None:
        if self._compute_integrity_fingerprint() != self._integrity_fingerprint:
            raise ValueError("policy execution trace was mutated")

    def validate_for_rollout(
        self,
        rollout: Rollout | RolloutSnapshot,
    ) -> None:
        """Rebind this trace to one exact rollout and replay its engine effects.

        The trace is not accepted merely because its policy labels match.  Lifecycle
        categories are derived again from the rollout validity tensors; absolute
        overrides must reproduce every effective next state; and kinematic controls
        are replayed under the recorded engine limits.  The resulting clamp totals
        must exactly equal the rollout metadata.
        """

        self.revalidate()
        if not isinstance(rollout, (Rollout, RolloutSnapshot)):
            raise TypeError("rollout must be a Rollout or RolloutSnapshot")
        rollout_snapshot = (
            rollout
            if isinstance(rollout, RolloutSnapshot)
            else RolloutSnapshot.from_rollout(rollout)
        )
        rollout_snapshot.revalidate()
        if (
            rollout_snapshot._integrity_fingerprint
            != self.rollout_fingerprint
        ):
            raise ValueError("trace rollout_fingerprint does not match rollout")
        if (
            rollout.sim_name != self.policy_name
            or rollout.sim_version != self.policy_version
        ):
            raise ValueError("trace policy identity does not match rollout")
        if rollout.perturbation != self.perturbation_identity:
            raise ValueError("trace perturbation identity does not match rollout")
        if self.stop_index != rollout.num_steps - 1:
            raise ValueError("trace stop_index does not match rollout horizon")
        expected_timestamps = rollout.timestamps[
            self.start_index : self.stop_index + 1
        ]
        if not np.array_equal(self.timestamps, expected_timestamps):
            raise ValueError("trace timestamps do not match rollout")

        metadata = rollout.metadata
        if metadata.get("rollout_start_index") != self.start_index:
            raise ValueError("trace start_index does not match rollout metadata")
        policy = metadata.get("policy")
        if not isinstance(policy, Mapping) or (
            policy.get("name") != self.policy_name
            or policy.get("version") != self.policy_version
        ):
            raise ValueError("trace policy metadata does not match rollout")
        modes = metadata.get("agent_control_modes")
        if not isinstance(modes, Mapping) or set(modes) != {
            str(agent.id) for agent in rollout.agents
        }:
            raise ValueError("rollout agent_control_modes is incomplete")
        ordered_modes = tuple(
            modes[str(agent.id)] for agent in rollout.agents
        )
        if ordered_modes != self.control_modes:
            raise ValueError("trace control modes do not match rollout")
        controlled_ids = metadata.get("controlled_agent_ids")
        if not isinstance(controlled_ids, Sequence) or isinstance(
            controlled_ids,
            (str, bytes),
        ):
            raise ValueError("rollout controlled_agent_ids is invalid")
        expected_world_ids = [
            agent.id
            for index, agent in enumerate(rollout.agents)
            if index != self.ego_index
        ]
        if list(controlled_ids) != expected_world_ids:
            raise ValueError("trace ego_index does not match rollout agent order")
        expected_ego_mode = (
            "logged_ego"
            if self.perturbation_identity is None
            else "typed_ego_plan"
        )
        if self.control_modes[self.ego_index] != expected_ego_mode:
            raise ValueError("trace ego control mode contradicts plan provenance")

        dynamics = metadata.get("dynamics")
        if not isinstance(dynamics, Mapping):
            raise ValueError("rollout dynamics metadata is invalid")
        limits_payload = dynamics.get("limits")
        if not isinstance(limits_payload, Mapping) or set(limits_payload) != {
            "max_acceleration_mps2",
            "max_deceleration_mps2",
            "max_speed_mps",
            "max_yaw_rate_radps",
        }:
            raise ValueError("rollout dynamics limits metadata is invalid")
        try:
            limits = DynamicsLimits(**dict(limits_payload))
        except Exception as exc:
            raise ValueError("rollout dynamics limits metadata is invalid") from exc
        recorded_clamps = dynamics.get("clamp_counts")
        if not isinstance(recorded_clamps, Mapping) or set(recorded_clamps) != set(
            _CLAMP_FIELDS
        ):
            raise ValueError("rollout dynamics clamp_counts metadata is invalid")
        if any(
            isinstance(recorded_clamps[name], (bool, np.bool_))
            or not isinstance(recorded_clamps[name], (int, np.integer))
            or int(recorded_clamps[name]) < 0
            for name in _CLAMP_FIELDS
        ):
            raise ValueError("rollout dynamics clamp_counts must be nonnegative integers")

        world_mask = np.ones(self.agent_count, dtype=bool)
        world_mask[self.ego_index] = False
        replayed_clamps = ClampCounts()
        for offset in range(self.transition_count):
            current_index = self.start_index + offset
            next_index = current_index + 1
            current = AgentFrame(
                valid=np.asarray(
                    [agent.valid[current_index] for agent in rollout.agents],
                    dtype=bool,
                ),
                **{
                    name: np.asarray(
                        [
                            getattr(agent, name)[current_index]
                            for agent in rollout.agents
                        ],
                        dtype=np.float64,
                    )
                    for name in _STATE_FIELDS
                },
            )
            next_valid = np.asarray(
                [agent.valid[next_index] for agent in rollout.agents],
                dtype=bool,
            )
            continuing = current.valid & next_valid & world_mask
            births = ~current.valid & next_valid & world_mask
            if not np.array_equal(
                self.effective_control_mask[offset],
                continuing,
            ):
                raise ValueError(
                    "trace effective_control_mask contradicts rollout validity"
                )
            if not np.array_equal(
                self.lifecycle_birth_mask[offset],
                births,
            ):
                raise ValueError(
                    "trace lifecycle_birth_mask contradicts rollout validity"
                )

            submitted_override = self.override_mask[offset]
            expected_override_valid = submitted_override & next_valid
            if not np.array_equal(
                self.override_valid[offset],
                expected_override_valid,
            ):
                raise ValueError(
                    "trace override_valid contradicts the applied lifecycle"
                )
            outside_override = ~submitted_override
            for name in _STATE_FIELDS:
                values = getattr(self, f"override_{name}")[offset]
                if np.any(values[outside_override] != 0.0):
                    raise ValueError(
                        f"trace override_{name} must be zero outside override_mask"
                    )
            effective_override = submitted_override & continuing
            for name in _STATE_FIELDS:
                expected_next = np.asarray(
                    [
                        getattr(agent, name)[next_index]
                        for agent in rollout.agents
                    ],
                    dtype=np.float64,
                )
                if not np.array_equal(
                    getattr(self, f"override_{name}")[offset][effective_override],
                    expected_next[effective_override],
                ):
                    raise ValueError(
                        f"trace effective override_{name} does not match rollout"
                    )

            kinematic = continuing & ~submitted_override
            replay = integrate_point_mass(
                current,
                self.longitudinal_acceleration[offset],
                self.yaw_rate[offset],
                float(self.timestamps[offset + 1] - self.timestamps[offset]),
                limits=limits,
                update_mask=kinematic,
            )
            replayed_clamps += replay.clamp_counts
            for name in _STATE_FIELDS:
                expected_next = np.asarray(
                    [
                        getattr(agent, name)[next_index]
                        for agent in rollout.agents
                    ],
                    dtype=np.float64,
                )
                if not np.array_equal(
                    getattr(replay.frame, name)[kinematic],
                    expected_next[kinematic],
                ):
                    raise ValueError(
                        f"trace kinematic controls do not reproduce rollout {name}"
                    )

        if replayed_clamps.to_dict() != {
            name: int(recorded_clamps[name]) for name in _CLAMP_FIELDS
        }:
            raise ValueError(
                "trace-replayed clamp counts do not match rollout metadata"
            )

    def replayed_clamp_counts_for_prefix(
        self,
        rollout: Rollout | RolloutSnapshot,
        transition_count: int,
    ) -> dict[str, int]:
        """Replay an exact leading transition prefix and return its clamp totals."""

        self.validate_for_rollout(rollout)
        count = _integer(
            transition_count,
            name="transition_count",
        )
        if count > self.transition_count:
            raise ValueError(
                "transition_count exceeds the trace execution horizon"
            )
        dynamics = rollout.metadata["dynamics"]
        assert isinstance(dynamics, Mapping)
        limits_payload = dynamics["limits"]
        assert isinstance(limits_payload, Mapping)
        limits = DynamicsLimits(**dict(limits_payload))
        replayed = ClampCounts()
        for offset in range(count):
            current_index = self.start_index + offset
            current = AgentFrame(
                valid=np.asarray(
                    [agent.valid[current_index] for agent in rollout.agents],
                    dtype=bool,
                ),
                **{
                    name: np.asarray(
                        [
                            getattr(agent, name)[current_index]
                            for agent in rollout.agents
                        ],
                        dtype=np.float64,
                    )
                    for name in _STATE_FIELDS
                },
            )
            kinematic = (
                self.effective_control_mask[offset]
                & ~self.override_mask[offset]
            )
            replay = integrate_point_mass(
                current,
                self.longitudinal_acceleration[offset],
                self.yaw_rate[offset],
                float(self.timestamps[offset + 1] - self.timestamps[offset]),
                limits=limits,
                update_mask=kinematic,
            )
            replayed += replay.clamp_counts
        return replayed.to_dict()


@dataclass(frozen=True, slots=True)
class TracedRollout:
    """One rollout paired with its separately versioned execution sidecar."""

    rollout: Rollout
    trace: PolicyExecutionTrace

    def __post_init__(self) -> None:
        if not isinstance(self.rollout, Rollout):
            raise TypeError("rollout must be a Rollout")
        if not isinstance(self.trace, PolicyExecutionTrace):
            raise TypeError("trace must be a PolicyExecutionTrace")
        self.trace.validate_for_rollout(self.rollout)

    def revalidate(self) -> None:
        self.trace.validate_for_rollout(self.rollout)


def policy_trace_prefix_equal(
    shorter: PolicyExecutionTrace,
    longer: PolicyExecutionTrace,
) -> bool:
    """Compare policy-side tensors while intentionally ignoring plan provenance."""

    if not isinstance(shorter, PolicyExecutionTrace) or not isinstance(
        longer,
        PolicyExecutionTrace,
    ):
        raise TypeError("both values must be PolicyExecutionTrace")
    shorter.revalidate()
    longer.revalidate()
    count = shorter.transition_count
    if (
        shorter.policy_name != longer.policy_name
        or shorter.policy_version != longer.policy_version
        or shorter.policy_access_role != longer.policy_access_role
        or shorter.start_index != longer.start_index
        or shorter.ego_index != longer.ego_index
        or shorter.agent_count != longer.agent_count
        or count > longer.transition_count
        or not np.array_equal(
            shorter.timestamps,
            longer.timestamps[: count + 1],
        )
    ):
        return False
    if any(
        left != right
        for index, (left, right) in enumerate(
            zip(
                shorter.control_modes,
                longer.control_modes,
                strict=True,
            )
        )
        if index != shorter.ego_index
    ):
        return False
    return all(
        np.array_equal(
            getattr(shorter, name),
            getattr(longer, name)[:count],
        )
        for name in (*_FLOAT_FIELDS, *_BOOL_FIELDS)
    )


__all__ = [
    "M6_POLICY_TRACE_SCHEMA_VERSION",
    "PolicyExecutionTrace",
    "TracedRollout",
    "policy_trace_prefix_equal",
]
