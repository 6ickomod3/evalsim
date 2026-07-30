"""Exact 20-transition M6 paired measures for the bounded Waymax role.

This module is deliberately separate from the 40-transition
``CounterfactualPair`` contract.  The public builder consumes a pair of compact
Waymax outputs only after the existing adapter has revalidated the complete source,
plan, qualification, target, actor, lifecycle, logged-world, and synchronous-order
gates.  It then takes an immutable source-neutral 21-frame snapshot and recomputes
the four preregistered paired measures over precisely those 20 transitions.

The eight-cell summary is secondary and descriptive.  It always contains exactly
two Waymax bundles crossed with four measures.  Fewer than eight complete paired
scenes is unsupported, eight or nine is insufficient and outcome-suppressed, and
10--16 receives a deterministic 10,000-resample pointwise 95% fixed-cohort
reweighting band.  No result from this module permits directional language.
"""
from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
import struct
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from evalsim.contracts import EgoTrajectoryPlan, Rollout, Scenario
from evalsim.metrics.m6 import (
    M6_PAIRED_METRIC_VERSION,
    M6_RESPONSE_ACCELERATION_THRESHOLD_MPS2,
    M6_RESPONSE_PERSISTENCE_S,
)
from evalsim.perturb.m6 import (
    PRIMARY_BRAKE_MAGNITUDE_MPS2,
    identity_spec,
    longitudinal_brake_pulse_spec,
)
from evalsim.simulators.waymax_m6 import (
    M6_WAYMAX_BUNDLES,
    M6_WAYMAX_CADENCE_MICROS,
    M6_WAYMAX_FRAME_COUNT,
    M6_WAYMAX_LOGGED_WORLD,
    M6_WAYMAX_MAX_SCENES,
    M6_WAYMAX_PRIVILEGED_IDM,
    M6_WAYMAX_TRANSITIONS,
    CompactM6WaymaxRollout,
    M6WaymaxEligibility,
    M6WaymaxPrimaryDomain,
    M6WaymaxSelection,
    WaymaxEgoPlanView,
    m6_waymax_to_rollout,
    source_state_mutation_sha256,
    validate_m6_waymax_pair,
)
from evalsim.simulators.waymax_reference import M4_MAX_OBJECTS

M6_WAYMAX_MEASURE_SCHEMA_VERSION = "m6-waymax-paired-measures-1.0.0"
M6_WAYMAX_STATISTICS_SCHEMA_VERSION = "m6-waymax-paired-statistics-1.0.0"
M6_WAYMAX_BASE_SEED = 20260729
M6_WAYMAX_RESAMPLES = 10_000
M6_WAYMAX_POINTWISE_LEVEL = 0.95
M6_WAYMAX_CELL_COUNT = 8
M6_WAYMAX_MIN_SUPPORTED_N = 8
M6_WAYMAX_MIN_DESCRIPTIVE_N = 10
M6_WAYMAX_DETERMINISM_CONDITIONS = ("identity", "primary_brake")
M6_WAYMAX_DETERMINISM_ROW_COUNT = (
    M6_WAYMAX_MAX_SCENES
    * len(M6_WAYMAX_BUNDLES)
    * len(M6_WAYMAX_DETERMINISM_CONDITIONS)
)
M6_WAYMAX_REWEIGHTING_INTERPRETATION = (
    "fixed_cohort_reweighting_sensitivity"
)

M6WaymaxCellStatus = Literal[
    "unsupported",
    "insufficient_n",
    "descriptive",
]
M6WaymaxScalarStatus = Literal["selected", "not_selected"]

_SHA256 = frozenset("0123456789abcdef")
_WORLD_FIELDS = ("valid", "x", "y", "heading", "vx", "vy")
_VIEW_FLOAT_FIELDS = (
    "baseline_ego_x",
    "baseline_ego_y",
    "baseline_ego_heading",
    "baseline_ego_vx",
    "baseline_ego_vy",
    "treatment_ego_x",
    "treatment_ego_y",
    "treatment_ego_heading",
    "treatment_ego_vx",
    "treatment_ego_vy",
    "baseline_target_x",
    "baseline_target_y",
    "baseline_target_heading",
    "baseline_target_vx",
    "baseline_target_vy",
    "treatment_target_x",
    "treatment_target_y",
    "treatment_target_heading",
    "treatment_target_vx",
    "treatment_target_vy",
)
_METRICS = (
    (
        "additional_target_braking_impulse_mps",
        M6_PAIRED_METRIC_VERSION,
        "m/s",
    ),
    (
        "response_timeliness_s",
        M6_PAIRED_METRIC_VERSION,
        "s",
    ),
    (
        "minimum_longitudinal_bumper_gap_change_m",
        M6_PAIRED_METRIC_VERSION,
        "m",
    ),
    (
        "target_progress_loss_m",
        M6_PAIRED_METRIC_VERSION,
        "m",
    ),
)
_METRICS_BY_NAME = {name: (version, unit) for name, version, unit in _METRICS}
_PAIR_VIEW_DOMAIN = b"evalsim-m6-waymax-20-transition-pair-view-v1"
_WORLD_GATE_DOMAIN = b"evalsim-m6-waymax-world-pair-gate-v1"
_MEASURE_RESULT_DOMAIN = b"evalsim-m6-waymax-paired-measure-result-v1"
_SAFE_SCALAR_DOMAIN = b"evalsim-m6-waymax-safe-scene-scalar-v1"
_CELL_RESULT_DOMAIN = b"evalsim-m6-waymax-cell-result-v1"
_MATRIX_RESULT_DOMAIN = b"evalsim-m6-waymax-matrix-result-v1"
_SELECTION_BINDING_DOMAIN = b"evalsim-m6-waymax-selection-binding-v1"
_ISSUED_SCALAR_TABLE_DOMAIN = b"evalsim-m6-waymax-issued-scalar-table-v1"
_PARSED_SCALAR_TABLE_DOMAIN = b"evalsim-m6-waymax-parsed-scalar-table-v1"
_STORED_RECONSTRUCTION_DOMAIN = (
    b"evalsim-m6-waymax-stored-reconstruction-v1"
)
_VERIFIED_STORED_SELECTION_DOMAIN = (
    b"evalsim-m6-waymax-verified-stored-selection-v1"
)
_NO_EXECUTION_DETERMINISM_ROW_DOMAIN = (
    b"evalsim-m6-waymax-no-execution-determinism-row-v1"
)
_NO_EXECUTION_DETERMINISM_TABLE_DOMAIN = (
    b"evalsim-m6-waymax-no-execution-determinism-table-v1"
)
_LIVE_DETERMINISM_ROW_DOMAIN = b"evalsim-m6-waymax-live-determinism-row-v1"
_LIVE_DETERMINISM_TABLE_DOMAIN = (
    b"evalsim-m6-waymax-live-determinism-table-v1"
)
_PAIR_VIEW_ISSUER = object()
_ISSUED_SCALAR_TABLE_ISSUER = object()
_PARSED_SCALAR_TABLE_ISSUER = object()
_STORED_RECONSTRUCTION_ISSUER = object()
_VERIFIED_STORED_SELECTION_ISSUER = object()
_NO_EXECUTION_DETERMINISM_ISSUER = object()
_LIVE_DETERMINISM_ISSUER = object()
_CELL_RESULT_ISSUER = object()
_MATRIX_RESULT_ISSUER = object()

_IDENTITY_CONFIGURATION_FINGERPRINT = (
    identity_spec().configuration_fingerprint
)
_PRIMARY_B2_CONFIGURATION_FINGERPRINT = (
    longitudinal_brake_pulse_spec(
        PRIMARY_BRAKE_MAGNITUDE_MPS2
    ).configuration_fingerprint
)
_BUNDLE_POLICY_ACCESS = {
    M6_WAYMAX_LOGGED_WORLD: (
        M6_WAYMAX_LOGGED_WORLD,
        "privileged_logged_world_negative_control",
    ),
    # This exact role name prevents the descriptive result from being relabeled
    # as a causal or history-only policy.
    M6_WAYMAX_PRIVILEGED_IDM: (
        M6_WAYMAX_PRIVILEGED_IDM,
        "privileged_logged_trajectory_waypoint_following",
    ),
}


class M6WaymaxMeasureError(ValueError):
    """Fail-closed bounded-Waymax measure error with a stable reason code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    raise M6WaymaxMeasureError(code, message)


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _readonly_vector(
    value: Any,
    *,
    dtype: np.dtype[Any] | type[Any],
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != (M6_WAYMAX_FRAME_COUNT,):
        raise ValueError(
            f"{name} must have shape [{M6_WAYMAX_FRAME_COUNT}]"
        )
    immutable = np.frombuffer(
        np.ascontiguousarray(array).tobytes(order="C"),
        dtype=np.dtype(dtype),
    )
    immutable.setflags(write=False)
    return immutable


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _selection_binding_sha256(selection: M6WaymaxSelection) -> str:
    """Bind the complete typed selection without exposing native scene identity."""

    payload = {
        "eligible_count": selection.eligible_count,
        "qualification_ledger_sha256": (
            selection.qualification_ledger_sha256
        ),
        "members": [
            {
                "cohort_index": member.cohort_index,
                "qualification_binding_sha256": (
                    member.qualification_binding_sha256
                ),
                "rank_sha256": member.rank_sha256,
            }
            for member in selection.members
        ],
        "primary_domain_member_count": (
            selection.primary_domain_member_count
        ),
        "primary_domain_sha256": selection.primary_domain_sha256,
        "supported": selection.supported,
        "selector_selection_sha256": selection.selection_sha256,
    }
    return hashlib.sha256(
        _SELECTION_BINDING_DOMAIN
        + b"\x00"
        + _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _validate_selection(
    selection: M6WaymaxSelection,
    *,
    primary_domain: M6WaymaxPrimaryDomain | None = None,
) -> str:
    """Revalidate a selection and, when supplied, its complete primary ledger."""

    if not isinstance(selection, M6WaymaxSelection):
        raise TypeError("selection must be an M6WaymaxSelection")
    if primary_domain is not None and not isinstance(
        primary_domain,
        M6WaymaxPrimaryDomain,
    ):
        raise TypeError("primary_domain must be an M6WaymaxPrimaryDomain")
    selection.revalidate(primary_domain=primary_domain)
    if primary_domain is not None:
        entries = primary_domain.entry_by_cohort_index
        for member in selection.members:
            entry = entries.get(member.cohort_index)
            if (
                entry is None
                or member.primary_entry_sha256 != entry.entry_sha256
                or member.source_binding_sha256
                != entry.source_state_sha256
                or member.scenario_id != entry.scenario_id
                or member.target_agent_id != entry.target_contract_id
                or member.target_index
                != entry.upstream_eligibility.target_index
            ):
                _fail(
                    "selection_domain",
                    "selected qualification differs from its primary-domain entry",
                )
    return _selection_binding_sha256(selection)


def _hash_array(hasher: Any, name: str, value: np.ndarray) -> None:
    encoded_name = name.encode("ascii")
    dtype = value.dtype.str.encode("ascii")
    hasher.update(len(encoded_name).to_bytes(4, "big"))
    hasher.update(encoded_name)
    hasher.update(len(dtype).to_bytes(4, "big"))
    hasher.update(dtype)
    hasher.update(value.ndim.to_bytes(4, "big"))
    for dimension in value.shape:
        hasher.update(int(dimension).to_bytes(8, "big"))
    payload = value.tobytes(order="C")
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def _compact_sha256(compact: CompactM6WaymaxRollout) -> str:
    if not isinstance(compact, CompactM6WaymaxRollout):
        raise TypeError("compact inputs must be CompactM6WaymaxRollout values")
    hasher = hashlib.sha256()
    hasher.update(b"evalsim-m6-waymax-compact-snapshot-v1\x00")
    for name in CompactM6WaymaxRollout._fields:
        _hash_array(hasher, name, np.asarray(getattr(compact, name)))
    return hasher.hexdigest()


def _world_rollout_sha256(
    rollout: Rollout,
    *,
    ego_index: int,
    start: int,
    stop: int,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(_WORLD_GATE_DOMAIN)
    hasher.update(b"\x00")
    for index, agent in enumerate(rollout.agents):
        if index == ego_index:
            continue
        hasher.update(struct.pack(">q", int(agent.id)))
        hasher.update(agent.type.value.encode("ascii"))
        hasher.update(np.asarray([agent.length, agent.width], dtype="<f8").tobytes())
        for name in _WORLD_FIELDS:
            array = np.asarray(getattr(agent, name))[start:stop]
            _hash_array(hasher, name, array)
    _hash_array(hasher, "timestamps", rollout.timestamps[start:stop])
    return hasher.hexdigest()


def _world_rollouts_equal(
    baseline: Rollout,
    treatment: Rollout,
    *,
    ego_index: int,
    start: int,
    stop: int,
) -> bool:
    if (
        len(baseline.agents) != len(treatment.agents)
        or not np.array_equal(
            baseline.timestamps[start:stop],
            treatment.timestamps[start:stop],
        )
    ):
        return False
    for index, (left, right) in enumerate(
        zip(baseline.agents, treatment.agents, strict=True)
    ):
        if index == ego_index:
            continue
        if (
            left.id != right.id
            or left.type != right.type
            or left.length != right.length
            or left.width != right.width
        ):
            return False
        if any(
            not np.array_equal(
                np.asarray(getattr(left, name))[start:stop],
                np.asarray(getattr(right, name))[start:stop],
            )
            for name in _WORLD_FIELDS
        ):
            return False
    return True


@dataclass(frozen=True, slots=True)
class M6WaymaxTwentyTransitionPairView:
    """Immutable exact-current-plus-20 snapshot of one validated compact pair."""

    selection_position: int
    cohort_index: int
    scenario_id: str
    bundle: str
    target_index: int
    target_agent_id: int
    target_slot: int
    ego_index: int
    ego_agent_id: int
    source_state_sha256: str
    qualification_binding_sha256: str
    primary_domain_sha256: str
    selection_binding_sha256: str
    selection_member_count: int
    baseline_plan_fingerprint: str
    treatment_plan_fingerprint: str
    baseline_configuration_fingerprint: str
    intervention_configuration_fingerprint: str
    baseline_perturbation_identity: str
    treatment_perturbation_identity: str
    target_length_m: float
    ego_length_m: float
    timestamps_micros: np.ndarray = field(repr=False, compare=False)
    target_valid: np.ndarray = field(repr=False, compare=False)
    ego_valid: np.ndarray = field(repr=False, compare=False)
    baseline_ego_x: np.ndarray = field(repr=False, compare=False)
    baseline_ego_y: np.ndarray = field(repr=False, compare=False)
    baseline_ego_heading: np.ndarray = field(repr=False, compare=False)
    baseline_ego_vx: np.ndarray = field(repr=False, compare=False)
    baseline_ego_vy: np.ndarray = field(repr=False, compare=False)
    treatment_ego_x: np.ndarray = field(repr=False, compare=False)
    treatment_ego_y: np.ndarray = field(repr=False, compare=False)
    treatment_ego_heading: np.ndarray = field(repr=False, compare=False)
    treatment_ego_vx: np.ndarray = field(repr=False, compare=False)
    treatment_ego_vy: np.ndarray = field(repr=False, compare=False)
    baseline_target_x: np.ndarray = field(repr=False, compare=False)
    baseline_target_y: np.ndarray = field(repr=False, compare=False)
    baseline_target_heading: np.ndarray = field(repr=False, compare=False)
    baseline_target_vx: np.ndarray = field(repr=False, compare=False)
    baseline_target_vy: np.ndarray = field(repr=False, compare=False)
    treatment_target_x: np.ndarray = field(repr=False, compare=False)
    treatment_target_y: np.ndarray = field(repr=False, compare=False)
    treatment_target_heading: np.ndarray = field(repr=False, compare=False)
    treatment_target_vx: np.ndarray = field(repr=False, compare=False)
    treatment_target_vy: np.ndarray = field(repr=False, compare=False)
    world_pair_gate_sha256: str
    view_binding_sha256: str | None = field(default=None, repr=False)
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _PAIR_VIEW_ISSUER:
            raise TypeError(
                "M6WaymaxTwentyTransitionPairView is builder-issued only"
            )
        for name in (
            "selection_position",
            "cohort_index",
            "target_index",
            "target_slot",
            "ego_index",
        ):
            object.__setattr__(
                self,
                name,
                _strict_int(getattr(self, name), name),
            )
        if self.selection_position >= M6_WAYMAX_MAX_SCENES:
            raise ValueError("selection_position must lie in [0, 15]")
        member_count = _strict_int(
            self.selection_member_count,
            "selection_member_count",
            minimum=M6_WAYMAX_MIN_SUPPORTED_N,
        )
        if (
            member_count > M6_WAYMAX_MAX_SCENES
            or self.selection_position >= member_count
        ):
            raise ValueError(
                "selection position must identify one member of a supported selection"
            )
        object.__setattr__(self, "selection_member_count", member_count)
        for name in ("target_agent_id", "ego_agent_id"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value,
                Integral,
            ):
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        _text(self.scenario_id, "scenario_id")
        if self.bundle not in M6_WAYMAX_BUNDLES:
            raise ValueError("bundle must be one of the exact bounded Waymax bundles")
        for name in (
            "source_state_sha256",
            "qualification_binding_sha256",
            "primary_domain_sha256",
            "selection_binding_sha256",
            "baseline_plan_fingerprint",
            "treatment_plan_fingerprint",
            "baseline_configuration_fingerprint",
            "intervention_configuration_fingerprint",
            "world_pair_gate_sha256",
        ):
            _sha256(getattr(self, name), name)
        if (
            self.baseline_configuration_fingerprint
            != _IDENTITY_CONFIGURATION_FINGERPRINT
            or self.intervention_configuration_fingerprint
            != _PRIMARY_B2_CONFIGURATION_FINGERPRINT
        ):
            raise ValueError(
                "pair view requires the exact registered identity and primary b=2 configs"
            )
        _text(self.baseline_perturbation_identity, "baseline_perturbation_identity")
        _text(self.treatment_perturbation_identity, "treatment_perturbation_identity")
        for name in ("target_length_m", "ego_length_m"):
            value = _finite_float(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "timestamps_micros",
            _readonly_vector(
                self.timestamps_micros,
                dtype=np.dtype("<i8"),
                name="timestamps_micros",
            ),
        )
        for name in ("target_valid", "ego_valid"):
            object.__setattr__(
                self,
                name,
                _readonly_vector(
                    getattr(self, name),
                    dtype=np.bool_,
                    name=name,
                ),
            )
        for name in _VIEW_FLOAT_FIELDS:
            object.__setattr__(
                self,
                name,
                _readonly_vector(
                    getattr(self, name),
                    dtype=np.dtype("<f8"),
                    name=name,
                ),
            )
        self._validate_semantics()
        expected = self._binding_sha256()
        if (
            self.view_binding_sha256 is not None
            and self.view_binding_sha256 != expected
        ):
            raise ValueError("view_binding_sha256 does not bind the complete view")
        object.__setattr__(self, "view_binding_sha256", expected)

    def _validate_semantics(self) -> None:
        if self.target_index == self.ego_index:
            raise ValueError("the frozen target must not be ego")
        if not bool(np.all(self.target_valid)) or not bool(np.all(self.ego_valid)):
            raise ValueError("target and ego must be valid at all 21 scored frames")
        if not all(
            bool(np.all(np.isfinite(getattr(self, name))))
            for name in _VIEW_FLOAT_FIELDS
        ):
            raise ValueError("the complete paired float view must be finite")
        if np.any(
            self.baseline_target_heading < -np.pi
        ) or np.any(self.baseline_target_heading > np.pi):
            raise ValueError("baseline target headings must lie in [-pi, pi]")
        if np.any(
            self.treatment_target_heading < -np.pi
        ) or np.any(self.treatment_target_heading > np.pi):
            raise ValueError("treatment target headings must lie in [-pi, pi]")
        deltas = np.diff(self.timestamps_micros)
        if not np.array_equal(
            deltas,
            np.full(M6_WAYMAX_TRANSITIONS, M6_WAYMAX_CADENCE_MICROS, dtype="<i8"),
        ):
            raise ValueError("the Waymax measure view requires exact 100 ms cadence")

        # The first simulated world frame was chosen from the unchanged current
        # observation.  A target response at current+1 would violate the frozen
        # synchronous t+2 floor.
        for suffix in ("x", "y", "heading", "vx", "vy"):
            if not np.array_equal(
                getattr(self, f"baseline_target_{suffix}")[:2],
                getattr(self, f"treatment_target_{suffix}")[:2],
            ):
                raise ValueError("the target violates the synchronous t+2 floor")

        # The logged-world bundle is a world-state negative control.  Relational gap
        # may still change because treatment ego moves, but target/world-only fields
        # may not.
        if self.bundle == M6_WAYMAX_LOGGED_WORLD:
            for suffix in ("x", "y", "heading", "vx", "vy"):
                if not np.array_equal(
                    getattr(self, f"baseline_target_{suffix}"),
                    getattr(self, f"treatment_target_{suffix}"),
                ):
                    raise ValueError(
                        "logged-world target fields must be exactly nonreactive"
                    )

    @property
    def transition_count(self) -> int:
        return M6_WAYMAX_TRANSITIONS

    @property
    def dt_s(self) -> np.ndarray:
        result = np.diff(self.timestamps_micros).astype(np.float64) * 1e-6
        result.setflags(write=False)
        return result

    def _binding_sha256(self) -> str:
        metadata = {
            "baseline_configuration_fingerprint": (
                self.baseline_configuration_fingerprint
            ),
            "baseline_perturbation_identity": (
                self.baseline_perturbation_identity
            ),
            "baseline_plan_fingerprint": self.baseline_plan_fingerprint,
            "bundle": self.bundle,
            "cohort_index": self.cohort_index,
            "ego_agent_id": self.ego_agent_id,
            "ego_index": self.ego_index,
            "ego_length_m": self.ego_length_m,
            "intervention_configuration_fingerprint": (
                self.intervention_configuration_fingerprint
            ),
            "primary_domain_sha256": self.primary_domain_sha256,
            "qualification_binding_sha256": (
                self.qualification_binding_sha256
            ),
            "scenario_id": self.scenario_id,
            "schema_version": M6_WAYMAX_MEASURE_SCHEMA_VERSION,
            "selection_position": self.selection_position,
            "selection_binding_sha256": self.selection_binding_sha256,
            "selection_member_count": self.selection_member_count,
            "source_state_sha256": self.source_state_sha256,
            "target_agent_id": self.target_agent_id,
            "target_index": self.target_index,
            "target_length_m": self.target_length_m,
            "target_slot": self.target_slot,
            "treatment_perturbation_identity": (
                self.treatment_perturbation_identity
            ),
            "treatment_plan_fingerprint": self.treatment_plan_fingerprint,
            "world_pair_gate_sha256": self.world_pair_gate_sha256,
        }
        hasher = hashlib.sha256()
        hasher.update(_PAIR_VIEW_DOMAIN)
        hasher.update(b"\x00")
        encoded = _canonical_json(metadata).encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
        for name in (
            "timestamps_micros",
            "target_valid",
            "ego_valid",
            *_VIEW_FLOAT_FIELDS,
        ):
            _hash_array(hasher, name, np.asarray(getattr(self, name)))
        return hasher.hexdigest()

    def revalidate(self) -> None:
        self._validate_semantics()
        if self._binding_sha256() != self.view_binding_sha256:
            _fail(
                "pair_view_mutated",
                "the immutable 20-transition pair view failed its binding",
            )


def _validate_reconstructed_rollout(
    rollout: Rollout,
    *,
    scenario: Scenario,
    plan: EgoTrajectoryPlan,
    bundle: str,
    current: int,
) -> None:
    if not isinstance(rollout, Rollout):
        raise TypeError("Waymax reconstruction must return a Rollout")
    if (
        rollout.scenario_id != scenario.scenario_id
        or rollout.sim_name != bundle
        or rollout.seed != 0
        or rollout.perturbation != plan.perturbation_identity
    ):
        _fail("rollout_relabel", "reconstructed rollout identity/provenance drifted")
    expected_frames = current + 1 + M6_WAYMAX_TRANSITIONS
    if rollout.num_steps != expected_frames:
        _fail("rollout_horizon", "reconstructed rollout has the wrong horizon")
    if len(rollout.agents) != len(scenario.agents):
        _fail("rollout_pairing", "reconstructed rollout changed agent cardinality")
    for source_agent, output_agent in zip(
        scenario.agents,
        rollout.agents,
        strict=True,
    ):
        if (
            source_agent.id != output_agent.id
            or source_agent.type != output_agent.type
            or source_agent.length != output_agent.length
            or source_agent.width != output_agent.width
        ):
            _fail("rollout_pairing", "reconstructed rollout changed agent identity")
    if (
        rollout.metadata.get("horizon_transitions") != M6_WAYMAX_TRANSITIONS
        or rollout.metadata.get("target_agent_id") is None
    ):
        _fail("rollout_provenance", "reconstructed rollout lacks exact M6 provenance")


def build_m6_waymax_twenty_transition_pair_view(
    baseline: CompactM6WaymaxRollout,
    treatment: CompactM6WaymaxRollout,
    *,
    selection_position: int,
    state: Any,
    scenario: Scenario,
    baseline_plan: EgoTrajectoryPlan,
    treatment_plan: EgoTrajectoryPlan,
    baseline_view: WaymaxEgoPlanView,
    treatment_view: WaymaxEgoPlanView,
    bundle: str,
    qualification: M6WaymaxEligibility,
    primary_domain: M6WaymaxPrimaryDomain,
    selection: M6WaymaxSelection,
) -> M6WaymaxTwentyTransitionPairView:
    """Validate a compact pair and freeze its exact 21-frame metric view."""

    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    selection_position = _strict_int(
        selection_position,
        "selection_position",
    )
    if selection_position >= M6_WAYMAX_MAX_SCENES:
        raise ValueError("selection_position must lie in [0, 15]")
    if not isinstance(qualification, M6WaymaxEligibility):
        raise TypeError("qualification must be an M6WaymaxEligibility")
    if not isinstance(primary_domain, M6WaymaxPrimaryDomain):
        raise TypeError("primary_domain must be an M6WaymaxPrimaryDomain")
    if bundle not in M6_WAYMAX_BUNDLES:
        raise ValueError("bundle must be one of the exact bounded Waymax bundles")
    qualification.revalidate()
    primary_domain.revalidate()
    selection_binding_sha256 = _validate_selection(
        selection,
        primary_domain=primary_domain,
    )
    if not selection.supported:
        _fail(
            "selection_unsupported",
            "fewer than eight selected scenes cannot construct outcomes",
        )
    if selection_position >= len(selection.members):
        _fail(
            "selection_position",
            "selection_position lies outside the exact selected subset",
        )
    selected_member = selection.members[selection_position]
    if (
        selected_member.cohort_index != qualification.cohort_index
        or selected_member.qualification_binding_sha256
        != qualification.qualification_binding_sha256
    ):
        _fail(
            "selection_member",
            "qualification is not the exact member at selection_position",
        )
    if not qualification.eligible:
        _fail("qualification", "an ineligible scene has no Waymax measure view")
    before_source = source_state_mutation_sha256(state)
    before_baseline = _compact_sha256(baseline)
    before_treatment = _compact_sha256(treatment)
    pair_validation = validate_m6_waymax_pair(
        baseline,
        treatment,
        state=state,
        scenario=scenario,
        baseline_plan=baseline_plan,
        treatment_plan=treatment_plan,
        baseline_view=baseline_view,
        treatment_view=treatment_view,
        bundle=bundle,
        qualification=qualification,
        primary_domain=primary_domain,
    )
    pair_validation.require_passed()
    baseline_rollout, baseline_validation = m6_waymax_to_rollout(
        baseline,
        state=state,
        scenario=scenario,
        plan=baseline_plan,
        view=baseline_view,
        bundle=bundle,
        qualification=qualification,
        primary_domain=primary_domain,
    )
    treatment_rollout, treatment_validation = m6_waymax_to_rollout(
        treatment,
        state=state,
        scenario=scenario,
        plan=treatment_plan,
        view=treatment_view,
        bundle=bundle,
        qualification=qualification,
        primary_domain=primary_domain,
    )
    baseline_validation.require_passed()
    treatment_validation.require_passed()
    baseline_plan.revalidate()
    treatment_plan.revalidate()
    baseline_view.revalidate()
    treatment_view.revalidate()
    qualification.revalidate()
    primary_domain.revalidate()
    if (
        source_state_mutation_sha256(state) != before_source
        or qualification.source_binding_sha256 != before_source
    ):
        _fail("source_mutated", "source state changed during measure construction")
    if (
        _compact_sha256(baseline) != before_baseline
        or _compact_sha256(treatment) != before_treatment
    ):
        _fail("compact_mutated", "compact output changed during validation")

    entry = primary_domain.entry_by_cohort_index.get(
        qualification.cohort_index
    )
    if (
        entry is None
        or qualification.primary_entry_sha256 != entry.entry_sha256
        or entry.scenario_id != scenario.scenario_id
        or entry.target_contract_id != qualification.target_agent_id
        or entry.upstream_eligibility.target_index != qualification.target_index
    ):
        _fail(
            "qualification_pairing",
            "qualification no longer matches its complete primary-domain entry",
        )
    if (
        qualification.scenario_id != scenario.scenario_id
        or qualification.target_index is None
        or qualification.target_agent_id is None
        or qualification.target_slot is None
    ):
        _fail("qualification_pairing", "qualification target identity is incomplete")
    target_index = qualification.target_index
    current = entry.upstream_eligibility.current_index
    if target_index >= scenario.num_agents or scenario.ego_index >= scenario.num_agents:
        _fail("target_scope", "frozen target or ego lies outside the scenario")
    target = scenario.agents[target_index]
    ego = scenario.agents[scenario.ego_index]
    if target.id != qualification.target_agent_id:
        _fail("target_scope", "frozen target contract identity drifted")
    _validate_reconstructed_rollout(
        baseline_rollout,
        scenario=scenario,
        plan=baseline_plan,
        bundle=bundle,
        current=current,
    )
    _validate_reconstructed_rollout(
        treatment_rollout,
        scenario=scenario,
        plan=treatment_plan,
        bundle=bundle,
        current=current,
    )
    if not np.array_equal(
        baseline_rollout.timestamps,
        treatment_rollout.timestamps,
    ):
        _fail("pairing", "paired rollout timestamps differ")

    start = current
    stop = current + M6_WAYMAX_FRAME_COUNT
    # Independently enforce the complete logged-world negative control and the
    # complete first-post-current t+2 floor on reconstructed contract agents.
    if bundle == M6_WAYMAX_LOGGED_WORLD:
        gate_start, gate_stop = current + 1, stop
    else:
        gate_start, gate_stop = current + 1, current + 2
    if not _world_rollouts_equal(
        baseline_rollout,
        treatment_rollout,
        ego_index=scenario.ego_index,
        start=gate_start,
        stop=gate_stop,
    ):
        code = (
            "logged_world_response"
            if bundle == M6_WAYMAX_LOGGED_WORLD
            else "synchronous_t_plus_2"
        )
        _fail(code, "reconstructed world pair violates its exact negative gate")
    baseline_world_hash = _world_rollout_sha256(
        baseline_rollout,
        ego_index=scenario.ego_index,
        start=gate_start,
        stop=gate_stop,
    )
    treatment_world_hash = _world_rollout_sha256(
        treatment_rollout,
        ego_index=scenario.ego_index,
        start=gate_start,
        stop=gate_stop,
    )
    if baseline_world_hash != treatment_world_hash:
        _fail("world_gate_hash", "world equality gate hashes disagree")
    world_gate_hash = hashlib.sha256(
        _WORLD_GATE_DOMAIN
        + b"\x00"
        + bundle.encode("ascii")
        + bytes.fromhex(baseline_world_hash)
    ).hexdigest()

    baseline_target = baseline_rollout.agents[target_index]
    treatment_target = treatment_rollout.agents[target_index]
    baseline_ego = baseline_rollout.agents[scenario.ego_index]
    treatment_ego = treatment_rollout.agents[scenario.ego_index]
    target_valid = np.asarray(baseline_target.valid[start:stop], dtype=bool)
    ego_valid = np.asarray(baseline_ego.valid[start:stop], dtype=bool)
    if (
        not np.array_equal(
            target_valid,
            treatment_target.valid[start:stop],
        )
        or not np.array_equal(ego_valid, treatment_ego.valid[start:stop])
    ):
        _fail("pair_validity", "paired target or ego validity differs")
    timestamps_micros = np.rint(
        baseline_rollout.timestamps[start:stop] * 1_000_000.0
    ).astype("<i8")
    if not np.allclose(
        timestamps_micros.astype(np.float64) * 1e-6,
        baseline_rollout.timestamps[start:stop],
        atol=1e-12,
        rtol=0.0,
    ):
        _fail("timestamp_units", "rollout timestamps do not map exactly to micros")

    return M6WaymaxTwentyTransitionPairView(
        selection_position=selection_position,
        cohort_index=qualification.cohort_index,
        scenario_id=scenario.scenario_id,
        bundle=bundle,
        target_index=target_index,
        target_agent_id=target.id,
        target_slot=qualification.target_slot,
        ego_index=scenario.ego_index,
        ego_agent_id=ego.id,
        source_state_sha256=before_source,
        qualification_binding_sha256=(
            qualification.qualification_binding_sha256
        ),
        primary_domain_sha256=primary_domain.domain_sha256,
        selection_binding_sha256=selection_binding_sha256,
        selection_member_count=len(selection.members),
        baseline_plan_fingerprint=baseline_plan.plan_fingerprint,
        treatment_plan_fingerprint=treatment_plan.plan_fingerprint,
        baseline_configuration_fingerprint=(
            baseline_plan.configuration_fingerprint
        ),
        intervention_configuration_fingerprint=(
            treatment_plan.configuration_fingerprint
        ),
        baseline_perturbation_identity=baseline_plan.perturbation_identity,
        treatment_perturbation_identity=treatment_plan.perturbation_identity,
        target_length_m=target.length,
        ego_length_m=ego.length,
        timestamps_micros=timestamps_micros,
        target_valid=target_valid,
        ego_valid=ego_valid,
        baseline_ego_x=baseline_ego.x[start:stop],
        baseline_ego_y=baseline_ego.y[start:stop],
        baseline_ego_heading=baseline_ego.heading[start:stop],
        baseline_ego_vx=baseline_ego.vx[start:stop],
        baseline_ego_vy=baseline_ego.vy[start:stop],
        treatment_ego_x=treatment_ego.x[start:stop],
        treatment_ego_y=treatment_ego.y[start:stop],
        treatment_ego_heading=treatment_ego.heading[start:stop],
        treatment_ego_vx=treatment_ego.vx[start:stop],
        treatment_ego_vy=treatment_ego.vy[start:stop],
        baseline_target_x=baseline_target.x[start:stop],
        baseline_target_y=baseline_target.y[start:stop],
        baseline_target_heading=baseline_target.heading[start:stop],
        baseline_target_vx=baseline_target.vx[start:stop],
        baseline_target_vy=baseline_target.vy[start:stop],
        treatment_target_x=treatment_target.x[start:stop],
        treatment_target_y=treatment_target.y[start:stop],
        treatment_target_heading=treatment_target.heading[start:stop],
        treatment_target_vx=treatment_target.vx[start:stop],
        treatment_target_vy=treatment_target.vy[start:stop],
        world_pair_gate_sha256=world_gate_hash,
        _issuance_capability=_PAIR_VIEW_ISSUER,
    )


@dataclass(frozen=True, slots=True)
class M6WaymaxPairedMeasureResult:
    """One exact per-scene scalar recomputed from a bound 20-transition view."""

    selection_position: int
    cohort_index: int
    scenario_id: str
    bundle: str
    metric_name: str
    metric_version: str
    value_unit: str
    value: float
    target_agent_id: int
    qualification_binding_sha256: str
    intervention_configuration_fingerprint: str
    view_binding_sha256: str
    responded: bool | None = None
    responder_latency_s: float | None = None
    result_binding_sha256: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selection_position",
            _strict_int(self.selection_position, "selection_position"),
        )
        if self.selection_position >= M6_WAYMAX_MAX_SCENES:
            raise ValueError("selection_position must lie in [0, 15]")
        object.__setattr__(
            self,
            "cohort_index",
            _strict_int(self.cohort_index, "cohort_index"),
        )
        _text(self.scenario_id, "scenario_id")
        if self.bundle not in M6_WAYMAX_BUNDLES:
            raise ValueError("result bundle is not registered")
        expected = _METRICS_BY_NAME.get(self.metric_name)
        if expected != (self.metric_version, self.value_unit):
            raise ValueError("result metric identity/version/unit is not registered")
        value = _finite_float(self.value, "value")
        object.__setattr__(self, "value", value)
        if isinstance(self.target_agent_id, (bool, np.bool_)) or not isinstance(
            self.target_agent_id,
            Integral,
        ):
            raise ValueError("target_agent_id must be an integer")
        object.__setattr__(self, "target_agent_id", int(self.target_agent_id))
        for name in (
            "qualification_binding_sha256",
            "intervention_configuration_fingerprint",
            "view_binding_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.metric_name == "response_timeliness_s":
            if type(self.responded) is not bool:
                raise ValueError("timeliness result requires responded")
            if self.responded:
                latency = _finite_float(
                    self.responder_latency_s,
                    "responder_latency_s",
                )
                if latency < 0.0:
                    raise ValueError("responder_latency_s must be non-negative")
                object.__setattr__(self, "responder_latency_s", latency)
            elif self.responder_latency_s is not None:
                raise ValueError("censored timeliness cannot have responder latency")
        elif self.responded is not None or self.responder_latency_s is not None:
            raise ValueError("response fields belong only to response timeliness")
        expected_hash = self._binding_sha256()
        if (
            self.result_binding_sha256 is not None
            and self.result_binding_sha256 != expected_hash
        ):
            raise ValueError("result_binding_sha256 does not bind the result")
        object.__setattr__(self, "result_binding_sha256", expected_hash)

    def _payload(self) -> dict[str, Any]:
        return {
            "bundle": self.bundle,
            "cohort_index": self.cohort_index,
            "intervention_configuration_fingerprint": (
                self.intervention_configuration_fingerprint
            ),
            "metric_name": self.metric_name,
            "metric_version": self.metric_version,
            "qualification_binding_sha256": (
                self.qualification_binding_sha256
            ),
            "responded": self.responded,
            "responder_latency_s": self.responder_latency_s,
            "scenario_id": self.scenario_id,
            "schema_version": M6_WAYMAX_MEASURE_SCHEMA_VERSION,
            "selection_position": self.selection_position,
            "target_agent_id": self.target_agent_id,
            "value": self.value,
            "value_unit": self.value_unit,
            "view_binding_sha256": self.view_binding_sha256,
        }

    def _binding_sha256(self) -> str:
        return hashlib.sha256(
            _MEASURE_RESULT_DOMAIN
            + b"\x00"
            + _canonical_json(self._payload()).encode("utf-8")
        ).hexdigest()

    def revalidate(self) -> None:
        if self._binding_sha256() != self.result_binding_sha256:
            _fail("measure_result_mutated", "paired measure result binding failed")


@dataclass(frozen=True, slots=True)
class M6WaymaxNoExecutionDeterminismRow:
    """One factory-issued strict-NA row; never execution evidence."""

    selection_position: int
    bundle: str
    condition: str
    cohort_index: None
    qualification_binding_sha256: None
    status: Literal["not_applicable"]
    eager_pass_1_sha256: None
    eager_pass_2_sha256: None
    jit_eager_sha256: None
    jit_compiled_sha256: None
    row_binding_sha256: str | None = field(default=None, repr=False)
    _issued_original_binding_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _issuance_capability: InitVar[object] = None

    STORE_FIELDS = frozenset(
        {
            "selection_position",
            "bundle",
            "condition",
            "cohort_index",
            "qualification_binding_sha256",
            "status",
            "eager_pass_1_sha256",
            "eager_pass_2_sha256",
            "jit_eager_sha256",
            "jit_compiled_sha256",
        }
    )

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _NO_EXECUTION_DETERMINISM_ISSUER:
            raise TypeError(
                "M6WaymaxNoExecutionDeterminismRow is factory-issued only"
            )
        position = _strict_int(
            self.selection_position,
            "selection_position",
        )
        if position >= M6_WAYMAX_MAX_SCENES:
            raise ValueError("selection_position must lie in [0, 15]")
        object.__setattr__(self, "selection_position", position)
        if self.bundle not in M6_WAYMAX_BUNDLES:
            raise ValueError("determinism row bundle is not registered")
        if self.condition not in M6_WAYMAX_DETERMINISM_CONDITIONS:
            raise ValueError("determinism row condition is not registered")
        if (
            self.status != "not_applicable"
            or any(
                value is not None
                for value in (
                    self.cohort_index,
                    self.qualification_binding_sha256,
                    self.eager_pass_1_sha256,
                    self.eager_pass_2_sha256,
                    self.jit_eager_sha256,
                    self.jit_compiled_sha256,
                )
            )
        ):
            raise ValueError(
                "no-execution determinism rows must be exact not_applicable/NA"
            )
        expected = self._binding_sha256()
        if (
            self.row_binding_sha256 is not None
            and self.row_binding_sha256 != expected
        ):
            raise ValueError(
                "row_binding_sha256 does not bind the no-execution row"
            )
        object.__setattr__(self, "row_binding_sha256", expected)
        object.__setattr__(
            self,
            "_issued_original_binding_sha256",
            expected,
        )

    def to_store_dict(self) -> dict[str, Any]:
        return {
            "selection_position": self.selection_position,
            "bundle": self.bundle,
            "condition": self.condition,
            "cohort_index": self.cohort_index,
            "qualification_binding_sha256": (
                self.qualification_binding_sha256
            ),
            "status": self.status,
            "eager_pass_1_sha256": self.eager_pass_1_sha256,
            "eager_pass_2_sha256": self.eager_pass_2_sha256,
            "jit_eager_sha256": self.jit_eager_sha256,
            "jit_compiled_sha256": self.jit_compiled_sha256,
        }

    def _binding_sha256(self) -> str:
        return hashlib.sha256(
            _NO_EXECUTION_DETERMINISM_ROW_DOMAIN
            + b"\x00"
            + _canonical_json(self.to_store_dict()).encode("utf-8")
        ).hexdigest()

    def revalidate(self) -> None:
        if (
            self._binding_sha256() != self.row_binding_sha256
            or self.row_binding_sha256
            != self._issued_original_binding_sha256
        ):
            _fail(
                "no_execution_determinism_row_mutated",
                "no-execution determinism row failed revalidation",
            )


def _no_execution_determinism_keys() -> tuple[tuple[int, str, str], ...]:
    return tuple(
        (position, bundle, condition)
        for position in range(M6_WAYMAX_MAX_SCENES)
        for bundle in M6_WAYMAX_BUNDLES
        for condition in M6_WAYMAX_DETERMINISM_CONDITIONS
    )


def _validate_no_execution_determinism_rows(
    rows: Sequence[M6WaymaxNoExecutionDeterminismRow],
) -> tuple[M6WaymaxNoExecutionDeterminismRow, ...]:
    normalized = tuple(rows)
    if len(normalized) != M6_WAYMAX_DETERMINISM_ROW_COUNT:
        raise ValueError("no-execution determinism table must contain 64 rows")
    if any(
        not isinstance(row, M6WaymaxNoExecutionDeterminismRow)
        for row in normalized
    ):
        raise TypeError(
            "no-execution determinism rows must be factory-issued row values"
        )
    for row in normalized:
        row.revalidate()
    actual_keys = tuple(
        (row.selection_position, row.bundle, row.condition)
        for row in normalized
    )
    if actual_keys != _no_execution_determinism_keys():
        raise ValueError(
            "no-execution determinism rows must be the canonical 16x2x2 grid"
        )
    return normalized


def _no_execution_determinism_table_sha256(
    *,
    reason: str,
    selection_binding_sha256: str | None,
    primary_domain_sha256: str | None,
    rows: Sequence[M6WaymaxNoExecutionDeterminismRow],
) -> str:
    digest = hashlib.sha256()
    digest.update(_NO_EXECUTION_DETERMINISM_TABLE_DOMAIN)
    digest.update(b"\x00")
    encoded_reason = reason.encode("ascii")
    digest.update(len(encoded_reason).to_bytes(4, "big"))
    digest.update(encoded_reason)
    for value in (selection_binding_sha256, primary_domain_sha256):
        digest.update(b"\x00" if value is None else b"\x01")
        if value is not None:
            digest.update(bytes.fromhex(value))
    digest.update(struct.pack(">I", len(rows)))
    for row in rows:
        digest.update(bytes.fromhex(row.row_binding_sha256))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M6WaymaxNoExecutionDeterminismTable(
    Sequence[M6WaymaxNoExecutionDeterminismRow]
):
    """Exact non-promotable data-free or unsupported 64-row placeholder."""

    reason: Literal["data_free", "unsupported_selection"]
    rows: tuple[M6WaymaxNoExecutionDeterminismRow, ...]
    selection_binding_sha256: str | None
    primary_domain_sha256: str | None
    table_binding_sha256: str | None = field(default=None, repr=False)
    promotable: bool = False
    _issued_original_binding_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _NO_EXECUTION_DETERMINISM_ISSUER:
            raise TypeError(
                "M6WaymaxNoExecutionDeterminismTable is factory-issued only"
            )
        if self.reason not in ("data_free", "unsupported_selection"):
            raise ValueError("no-execution determinism reason is not registered")
        if self.promotable is not False:
            raise ValueError(
                "no-execution determinism tables are permanently non-promotable"
            )
        if self.reason == "data_free":
            if (
                self.selection_binding_sha256 is not None
                or self.primary_domain_sha256 is not None
            ):
                raise ValueError(
                    "data-free determinism has no selection/domain binding"
                )
        else:
            _sha256(
                self.selection_binding_sha256,
                "selection_binding_sha256",
            )
            _sha256(
                self.primary_domain_sha256,
                "primary_domain_sha256",
            )
        rows = _validate_no_execution_determinism_rows(self.rows)
        object.__setattr__(self, "rows", rows)
        expected = _no_execution_determinism_table_sha256(
            reason=self.reason,
            selection_binding_sha256=self.selection_binding_sha256,
            primary_domain_sha256=self.primary_domain_sha256,
            rows=rows,
        )
        if (
            self.table_binding_sha256 is not None
            and self.table_binding_sha256 != expected
        ):
            raise ValueError(
                "table_binding_sha256 does not bind no-execution rows"
            )
        object.__setattr__(self, "table_binding_sha256", expected)
        object.__setattr__(
            self,
            "_issued_original_binding_sha256",
            expected,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int | slice,
    ) -> (
        M6WaymaxNoExecutionDeterminismRow
        | tuple[M6WaymaxNoExecutionDeterminismRow, ...]
    ):
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)

    def to_store_rows(self) -> tuple[dict[str, Any], ...]:
        self.revalidate()
        return tuple(row.to_store_dict() for row in self.rows)

    def revalidate(
        self,
        *,
        selection: M6WaymaxSelection | None = None,
        primary_domain: M6WaymaxPrimaryDomain | None = None,
    ) -> None:
        rows = _validate_no_execution_determinism_rows(self.rows)
        expected = _no_execution_determinism_table_sha256(
            reason=self.reason,
            selection_binding_sha256=self.selection_binding_sha256,
            primary_domain_sha256=self.primary_domain_sha256,
            rows=rows,
        )
        if (
            expected != self.table_binding_sha256
            or expected != self._issued_original_binding_sha256
            or self.promotable is not False
        ):
            _fail(
                "no_execution_determinism_table_mutated",
                "no-execution determinism table failed revalidation",
            )
        if self.reason == "data_free":
            if selection is not None or primary_domain is not None:
                raise ValueError(
                    "data-free determinism cannot bind a live selection/domain"
                )
            return
        if (selection is None) != (primary_domain is None):
            raise ValueError(
                "selection and primary_domain must be supplied together"
            )
        if selection is None:
            return
        assert primary_domain is not None
        selection_binding = _validate_selection(
            selection,
            primary_domain=primary_domain,
        )
        if selection.supported:
            _fail(
                "determinism_live_unavailable",
                "supported selections require the future official runner issuer",
            )
        if (
            selection_binding != self.selection_binding_sha256
            or primary_domain.domain_sha256 != self.primary_domain_sha256
        ):
            _fail(
                "no_execution_determinism_selection",
                "unsupported placeholder differs from canonical selection/domain",
            )


def _build_no_execution_determinism_rows(
) -> tuple[M6WaymaxNoExecutionDeterminismRow, ...]:
    return tuple(
        M6WaymaxNoExecutionDeterminismRow(
            selection_position=position,
            bundle=bundle,
            condition=condition,
            cohort_index=None,
            qualification_binding_sha256=None,
            status="not_applicable",
            eager_pass_1_sha256=None,
            eager_pass_2_sha256=None,
            jit_eager_sha256=None,
            jit_compiled_sha256=None,
            _issuance_capability=_NO_EXECUTION_DETERMINISM_ISSUER,
        )
        for position, bundle, condition in _no_execution_determinism_keys()
    )


def build_m6_waymax_data_free_determinism_table(
) -> M6WaymaxNoExecutionDeterminismTable:
    """Issue the sole caller-free data-free determinism placeholder."""

    return M6WaymaxNoExecutionDeterminismTable(
        reason="data_free",
        rows=_build_no_execution_determinism_rows(),
        selection_binding_sha256=None,
        primary_domain_sha256=None,
        _issuance_capability=_NO_EXECUTION_DETERMINISM_ISSUER,
    )


def build_m6_waymax_unsupported_determinism_table(
    *,
    selection: M6WaymaxSelection,
    primary_domain: M6WaymaxPrimaryDomain,
) -> M6WaymaxNoExecutionDeterminismTable:
    """Issue strict NA only for an authentic canonical unsupported selection."""

    selection_binding = _validate_selection(
        selection,
        primary_domain=primary_domain,
    )
    if selection.supported:
        _fail(
            "determinism_live_unavailable",
            "supported selections require the future official runner issuer",
        )
    table = M6WaymaxNoExecutionDeterminismTable(
        reason="unsupported_selection",
        rows=_build_no_execution_determinism_rows(),
        selection_binding_sha256=selection_binding,
        primary_domain_sha256=primary_domain.domain_sha256,
        _issuance_capability=_NO_EXECUTION_DETERMINISM_ISSUER,
    )
    table.revalidate(
        selection=selection,
        primary_domain=primary_domain,
    )
    return table


def validate_m6_waymax_no_execution_determinism_table(
    value: Any,
    *,
    selection: M6WaymaxSelection | None = None,
    primary_domain: M6WaymaxPrimaryDomain | None = None,
) -> M6WaymaxNoExecutionDeterminismTable:
    """Reject mappings/parsed rows and revalidate an authentic issued table."""

    if not isinstance(value, M6WaymaxNoExecutionDeterminismTable):
        raise TypeError(
            "determinism placeholder must be a factory-issued "
            "M6WaymaxNoExecutionDeterminismTable"
        )
    value.revalidate(
        selection=selection,
        primary_domain=primary_domain,
    )
    return value


_COMPACT_FLOAT_FIELDS = ("x", "y", "yaw", "vx", "vy")
_COMPACT_BOOLEAN_FIELDS = (
    "valid",
    "requested_control",
    "effective_control",
    "lifecycle_fallback",
    "initialized_overlap_excluded",
)


def _compact_storage_ids(value: CompactM6WaymaxRollout) -> set[int]:
    return {
        id(value),
        *(id(getattr(value, name)) for name in CompactM6WaymaxRollout._fields),
    }


def _compacts_share_storage(
    left: CompactM6WaymaxRollout,
    right: CompactM6WaymaxRollout,
) -> bool:
    if _compact_storage_ids(left) & _compact_storage_ids(right):
        return True
    return any(
        np.shares_memory(
            np.asarray(getattr(left, name)),
            np.asarray(getattr(right, name)),
        )
        for name in CompactM6WaymaxRollout._fields
    )


def _validated_live_compact_sha256(
    compact: CompactM6WaymaxRollout,
) -> str:
    """Validate and hash one defensive typed compact snapshot."""

    if not isinstance(compact, CompactM6WaymaxRollout):
        raise TypeError("live determinism inputs must be typed compact rollouts")
    object_shape = (M6_WAYMAX_TRANSITIONS, M4_MAX_OBJECTS)
    arrays = {
        name: np.asarray(getattr(compact, name))
        for name in CompactM6WaymaxRollout._fields
    }
    for name in _COMPACT_FLOAT_FIELDS:
        value = arrays[name]
        if value.shape != object_shape or value.dtype != np.dtype(np.float32):
            raise ValueError(
                f"live determinism compact {name} must be float32 [20, 128]"
            )
        if not bool(np.all(np.isfinite(value))):
            raise ValueError(f"live determinism compact {name} must be finite")
    for name in _COMPACT_BOOLEAN_FIELDS:
        value = arrays[name]
        if value.shape != object_shape or value.dtype != np.dtype(np.bool_):
            raise ValueError(
                f"live determinism compact {name} must be bool [20, 128]"
            )
    timestamps = arrays["timestamp_micros"]
    if timestamps.shape != object_shape or (
        np.issubdtype(timestamps.dtype, np.bool_)
        or not np.issubdtype(timestamps.dtype, np.integer)
    ):
        raise ValueError(
            "live determinism compact timestamp_micros must be integer [20, 128]"
        )
    timestep = arrays["timestep"]
    if timestep.shape != (M6_WAYMAX_TRANSITIONS,) or (
        np.issubdtype(timestep.dtype, np.bool_)
        or not np.issubdtype(timestep.dtype, np.integer)
    ):
        raise ValueError("live determinism compact timestep must be integer [20]")
    snapshots: dict[str, np.ndarray] = {}
    for name in CompactM6WaymaxRollout._fields:
        snapshot = np.array(arrays[name], copy=True, order="C")
        snapshot.setflags(write=False)
        snapshots[name] = snapshot
    frozen = CompactM6WaymaxRollout(
        *(snapshots[name] for name in CompactM6WaymaxRollout._fields)
    )
    digest = _compact_sha256(frozen)
    if _compact_sha256(compact) != digest:
        _fail(
            "live_determinism_compact_mutated",
            "a compact rollout changed while its digest was issued",
        )
    return digest


@dataclass(frozen=True, slots=True)
class M6WaymaxLiveDeterminismExecution:
    """Typed raw outputs for one selected position × bundle × condition."""

    selection_position: int
    bundle: str
    condition: str
    qualification: M6WaymaxEligibility = field(repr=False, compare=False)
    eager_pass_1: CompactM6WaymaxRollout = field(repr=False, compare=False)
    eager_pass_2: CompactM6WaymaxRollout = field(repr=False, compare=False)
    jit_eager: CompactM6WaymaxRollout | None = field(
        default=None, repr=False, compare=False
    )
    jit_compiled: CompactM6WaymaxRollout | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selection_position",
            _strict_int(self.selection_position, "selection_position"),
        )
        self.revalidate()

    def revalidate(self) -> None:
        if self.selection_position >= M6_WAYMAX_MAX_SCENES:
            raise ValueError("selection_position must lie in [0, 15]")
        if self.bundle not in M6_WAYMAX_BUNDLES:
            raise ValueError("live determinism bundle is not registered")
        if self.condition not in M6_WAYMAX_DETERMINISM_CONDITIONS:
            raise ValueError("live determinism condition is not registered")
        if not isinstance(self.qualification, M6WaymaxEligibility):
            raise TypeError("qualification must be M6WaymaxEligibility")
        self.qualification.revalidate()
        if not self.qualification.eligible:
            raise ValueError("live determinism requires an eligible qualification")
        compacts = (self.eager_pass_1, self.eager_pass_2)
        if any(not isinstance(value, CompactM6WaymaxRollout) for value in compacts):
            raise TypeError("eager passes must be typed compact rollouts")
        if _compacts_share_storage(*compacts):
            _fail(
                "live_determinism_eager_replay",
                "the two eager passes must be independently supplied outputs",
            )
        if self.selection_position == 0:
            if not isinstance(self.jit_eager, CompactM6WaymaxRollout) or not isinstance(
                self.jit_compiled, CompactM6WaymaxRollout
            ):
                raise TypeError(
                    "selection position zero requires eager and compiled JIT outputs"
                )
            all_compacts = (*compacts, self.jit_eager, self.jit_compiled)
            for index, left in enumerate(all_compacts):
                for right in all_compacts[index + 1 :]:
                    if _compacts_share_storage(left, right):
                        _fail(
                            "live_determinism_jit_replay",
                            "all eager/JIT outputs must be independently supplied",
                        )
        elif self.jit_eager is not None or self.jit_compiled is not None:
            raise ValueError(
                "only selection position zero may carry JIT evidence"
            )


@dataclass(frozen=True, slots=True)
class M6WaymaxLiveDeterminismRow:
    """One factory-issued live pass or exact unselected NA row."""

    selection_position: int
    bundle: str
    condition: str
    cohort_index: int | None
    qualification_binding_sha256: str | None
    status: Literal["passed", "not_applicable"]
    eager_pass_1_sha256: str | None
    eager_pass_2_sha256: str | None
    jit_eager_sha256: str | None
    jit_compiled_sha256: str | None
    row_binding_sha256: str | None = field(default=None, repr=False)
    _issued_original_binding_sha256: str = field(
        init=False, repr=False, compare=False
    )
    _issuance_capability: InitVar[object] = None

    STORE_FIELDS = M6WaymaxNoExecutionDeterminismRow.STORE_FIELDS

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _LIVE_DETERMINISM_ISSUER:
            raise TypeError("M6WaymaxLiveDeterminismRow is factory-issued only")
        object.__setattr__(
            self,
            "selection_position",
            _strict_int(self.selection_position, "selection_position"),
        )
        if self.cohort_index is not None:
            object.__setattr__(
                self,
                "cohort_index",
                _strict_int(self.cohort_index, "cohort_index"),
            )
        self._validate_semantics()
        expected = self._binding_sha256()
        if self.row_binding_sha256 is not None and self.row_binding_sha256 != expected:
            raise ValueError("row_binding_sha256 does not bind the live row")
        object.__setattr__(self, "row_binding_sha256", expected)
        object.__setattr__(self, "_issued_original_binding_sha256", expected)

    def _validate_semantics(self) -> None:
        if self.selection_position >= M6_WAYMAX_MAX_SCENES:
            raise ValueError("selection_position must lie in [0, 15]")
        if self.bundle not in M6_WAYMAX_BUNDLES:
            raise ValueError("live determinism row bundle is not registered")
        if self.condition not in M6_WAYMAX_DETERMINISM_CONDITIONS:
            raise ValueError("live determinism row condition is not registered")
        digests = (
            self.eager_pass_1_sha256,
            self.eager_pass_2_sha256,
            self.jit_eager_sha256,
            self.jit_compiled_sha256,
        )
        if self.status == "not_applicable":
            if self.cohort_index is not None or (
                self.qualification_binding_sha256 is not None
            ) or any(value is not None for value in digests):
                raise ValueError(
                    "unselected live determinism rows must be exact NA"
                )
            return
        if self.status != "passed" or self.cohort_index is None:
            raise ValueError("selected live determinism rows must pass")
        _sha256(
            self.qualification_binding_sha256,
            "qualification_binding_sha256",
        )
        eager_1 = _sha256(self.eager_pass_1_sha256, "eager_pass_1_sha256")
        eager_2 = _sha256(self.eager_pass_2_sha256, "eager_pass_2_sha256")
        if eager_1 != eager_2:
            raise ValueError("independent eager compact outputs disagree")
        if self.selection_position == 0:
            jit_eager = _sha256(self.jit_eager_sha256, "jit_eager_sha256")
            jit_compiled = _sha256(
                self.jit_compiled_sha256, "jit_compiled_sha256"
            )
            if eager_1 != jit_eager or jit_eager != jit_compiled:
                raise ValueError("position-zero eager and JIT outputs disagree")
        elif self.jit_eager_sha256 is not None or self.jit_compiled_sha256 is not None:
            raise ValueError("nonzero positions cannot carry JIT comparison hashes")

    def to_store_dict(self) -> dict[str, Any]:
        return {
            "selection_position": self.selection_position,
            "bundle": self.bundle,
            "condition": self.condition,
            "cohort_index": self.cohort_index,
            "qualification_binding_sha256": self.qualification_binding_sha256,
            "status": self.status,
            "eager_pass_1_sha256": self.eager_pass_1_sha256,
            "eager_pass_2_sha256": self.eager_pass_2_sha256,
            "jit_eager_sha256": self.jit_eager_sha256,
            "jit_compiled_sha256": self.jit_compiled_sha256,
        }

    def _binding_sha256(self) -> str:
        return hashlib.sha256(
            _LIVE_DETERMINISM_ROW_DOMAIN
            + b"\x00"
            + _canonical_json(self.to_store_dict()).encode("utf-8")
        ).hexdigest()

    def revalidate(self) -> None:
        try:
            self._validate_semantics()
            expected = self._binding_sha256()
            if expected != self.row_binding_sha256 or (
                expected != self._issued_original_binding_sha256
            ):
                raise ValueError("live row binding changed")
        except M6WaymaxMeasureError:
            raise
        except (TypeError, ValueError) as exc:
            _fail(
                "live_determinism_row_mutated",
                f"live determinism row failed revalidation: {exc}",
            )


def _validate_live_determinism_rows(
    rows: Sequence[M6WaymaxLiveDeterminismRow],
) -> tuple[M6WaymaxLiveDeterminismRow, ...]:
    normalized = tuple(rows)
    if len(normalized) != M6_WAYMAX_DETERMINISM_ROW_COUNT:
        raise ValueError("live determinism table must contain 64 rows")
    if any(not isinstance(row, M6WaymaxLiveDeterminismRow) for row in normalized):
        raise TypeError("live determinism rows must be factory-issued row values")
    for row in normalized:
        row.revalidate()
    keys = tuple(
        (row.selection_position, row.bundle, row.condition)
        for row in normalized
    )
    if keys != _no_execution_determinism_keys():
        raise ValueError("live determinism rows must be the canonical 16x2x2 grid")
    return normalized


def _validate_live_determinism_layout(
    rows: Sequence[M6WaymaxLiveDeterminismRow],
    selected_member_count: int,
) -> None:
    selected_identities: set[tuple[int, str]] = set()
    rows_per_position = len(M6_WAYMAX_BUNDLES) * len(
        M6_WAYMAX_DETERMINISM_CONDITIONS
    )
    for position in range(M6_WAYMAX_MAX_SCENES):
        start = position * rows_per_position
        group = rows[start : start + rows_per_position]
        if position < selected_member_count:
            if any(row.status != "passed" for row in group):
                raise ValueError("every selected determinism row must pass")
            identities = {
                (row.cohort_index, row.qualification_binding_sha256)
                for row in group
            }
            if len(identities) != 1:
                raise ValueError(
                    "one selected position must use one cohort/qualification"
                )
            identity = next(iter(identities))
            assert identity[0] is not None and identity[1] is not None
            if identity in selected_identities:
                raise ValueError("selected determinism identity was replayed")
            selected_identities.add((identity[0], identity[1]))
        elif any(row.status != "not_applicable" for row in group):
            raise ValueError("every unselected determinism slot must be exact NA")


def _live_determinism_table_sha256(
    *,
    selected_member_count: int,
    selection_binding_sha256: str,
    primary_domain_sha256: str,
    rows: Sequence[M6WaymaxLiveDeterminismRow],
) -> str:
    digest = hashlib.sha256()
    digest.update(_LIVE_DETERMINISM_TABLE_DOMAIN)
    digest.update(b"\x00")
    digest.update(struct.pack(">I", selected_member_count))
    digest.update(bytes.fromhex(selection_binding_sha256))
    digest.update(bytes.fromhex(primary_domain_sha256))
    digest.update(struct.pack(">I", len(rows)))
    for row in rows:
        assert row.row_binding_sha256 is not None
        digest.update(bytes.fromhex(row.row_binding_sha256))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M6WaymaxLiveDeterminismTable(Sequence[M6WaymaxLiveDeterminismRow]):
    """Factory-issued promotable live evidence over the exact 64-row grid."""

    selected_member_count: int
    selection_binding_sha256: str
    primary_domain_sha256: str
    rows: tuple[M6WaymaxLiveDeterminismRow, ...]
    table_binding_sha256: str | None = field(default=None, repr=False)
    promotable: bool = True
    _issued_original_binding_sha256: str = field(
        init=False, repr=False, compare=False
    )
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _LIVE_DETERMINISM_ISSUER:
            raise TypeError("M6WaymaxLiveDeterminismTable is factory-issued only")
        count = _strict_int(
            self.selected_member_count,
            "selected_member_count",
            minimum=M6_WAYMAX_MIN_SUPPORTED_N,
        )
        if count > M6_WAYMAX_MAX_SCENES:
            raise ValueError("selected_member_count must lie in [8, 16]")
        object.__setattr__(self, "selected_member_count", count)
        _sha256(self.selection_binding_sha256, "selection_binding_sha256")
        _sha256(self.primary_domain_sha256, "primary_domain_sha256")
        rows = _validate_live_determinism_rows(self.rows)
        _validate_live_determinism_layout(rows, count)
        object.__setattr__(self, "rows", rows)
        if self.promotable is not True:
            raise ValueError("live determinism evidence must remain promotable")
        expected = _live_determinism_table_sha256(
            selected_member_count=count,
            selection_binding_sha256=self.selection_binding_sha256,
            primary_domain_sha256=self.primary_domain_sha256,
            rows=rows,
        )
        if self.table_binding_sha256 is not None and (
            self.table_binding_sha256 != expected
        ):
            raise ValueError("table_binding_sha256 does not bind live rows")
        object.__setattr__(self, "table_binding_sha256", expected)
        object.__setattr__(self, "_issued_original_binding_sha256", expected)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int | slice):
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)

    def to_store_rows(self) -> tuple[dict[str, Any], ...]:
        self.revalidate()
        return tuple(row.to_store_dict() for row in self.rows)

    def revalidate(
        self,
        *,
        selection: M6WaymaxSelection | None = None,
        primary_domain: M6WaymaxPrimaryDomain | None = None,
    ) -> None:
        try:
            rows = _validate_live_determinism_rows(self.rows)
            _validate_live_determinism_layout(rows, self.selected_member_count)
            expected = _live_determinism_table_sha256(
                selected_member_count=self.selected_member_count,
                selection_binding_sha256=self.selection_binding_sha256,
                primary_domain_sha256=self.primary_domain_sha256,
                rows=rows,
            )
            if expected != self.table_binding_sha256 or (
                expected != self._issued_original_binding_sha256
            ) or self.promotable is not True:
                raise ValueError("live determinism table binding changed")
        except M6WaymaxMeasureError:
            raise
        except (TypeError, ValueError) as exc:
            _fail(
                "live_determinism_table_mutated",
                f"live determinism table failed revalidation: {exc}",
            )
        if (selection is None) != (primary_domain is None):
            raise ValueError("selection and primary_domain must be supplied together")
        if selection is None:
            return
        assert primary_domain is not None
        selection_binding = _validate_selection(
            selection, primary_domain=primary_domain
        )
        if not selection.supported:
            _fail(
                "live_determinism_selection_unsupported",
                "live evidence requires a supported canonical selection",
            )
        if (
            selection_binding != self.selection_binding_sha256
            or primary_domain.domain_sha256 != self.primary_domain_sha256
            or len(selection.members) != self.selected_member_count
        ):
            _fail(
                "live_determinism_selection_mismatch",
                "live evidence belongs to a different canonical selection",
            )
        rows_per_position = len(M6_WAYMAX_BUNDLES) * len(
            M6_WAYMAX_DETERMINISM_CONDITIONS
        )
        for position, member in enumerate(selection.members):
            start = position * rows_per_position
            group = self.rows[start : start + rows_per_position]
            if any(
                row.cohort_index != member.cohort_index
                or row.qualification_binding_sha256
                != member.qualification_binding_sha256
                for row in group
            ):
                _fail(
                    "live_determinism_selection_mismatch",
                    "live row cohort/qualification differs from selection",
                )


def build_m6_waymax_live_determinism_table(
    executions: Sequence[M6WaymaxLiveDeterminismExecution],
    *,
    selection: M6WaymaxSelection,
    primary_domain: M6WaymaxPrimaryDomain,
) -> M6WaymaxLiveDeterminismTable:
    """Hash independently produced typed outputs into live determinism evidence."""

    selection_binding = _validate_selection(
        selection, primary_domain=primary_domain
    )
    if not selection.supported:
        _fail(
            "live_determinism_selection_unsupported",
            "unsupported selections must use the no-execution placeholder",
        )
    if isinstance(executions, (str, bytes)) or not isinstance(executions, Sequence):
        raise TypeError("executions must be a sequence of typed live evidence")
    normalized = tuple(executions)
    if any(
        not isinstance(item, M6WaymaxLiveDeterminismExecution)
        for item in normalized
    ):
        raise TypeError(
            "executions must contain M6WaymaxLiveDeterminismExecution values"
        )
    expected_order = tuple(
        key
        for key in _no_execution_determinism_keys()
        if key[0] < len(selection.members)
    )
    by_key: dict[
        tuple[int, str, str], M6WaymaxLiveDeterminismExecution
    ] = {}
    seen_compacts: list[CompactM6WaymaxRollout] = []
    for item in normalized:
        item.revalidate()
        key = (item.selection_position, item.bundle, item.condition)
        if key in by_key:
            raise ValueError("live determinism execution key is duplicated")
        by_key[key] = item
        compacts = [item.eager_pass_1, item.eager_pass_2]
        if item.jit_eager is not None:
            compacts.append(item.jit_eager)
        if item.jit_compiled is not None:
            compacts.append(item.jit_compiled)
        for compact in compacts:
            if any(_compacts_share_storage(compact, prior) for prior in seen_compacts):
                _fail(
                    "live_determinism_execution_replay",
                    "one compact output was replayed across evidence slots",
                )
            seen_compacts.append(compact)
    if tuple(by_key) != expected_order:
        raise ValueError(
            "live determinism executions must use canonical selected 2x2 order"
        )

    issued: dict[tuple[int, str, str], M6WaymaxLiveDeterminismRow] = {}
    for key in expected_order:
        item = by_key[key]
        member = selection.members[item.selection_position]
        qualification = item.qualification
        if (
            qualification.cohort_index != member.cohort_index
            or qualification.qualification_binding_sha256
            != member.qualification_binding_sha256
            or qualification.source_binding_sha256 != member.source_binding_sha256
            or qualification.primary_entry_sha256 != member.primary_entry_sha256
            or qualification.scenario_id != member.scenario_id
            or qualification.target_index != member.target_index
            or qualification.target_agent_id != member.target_agent_id
            or qualification.target_slot != member.target_slot
        ):
            _fail(
                "live_determinism_selection_mismatch",
                "execution qualification differs from its selected position",
            )
        eager_1 = _validated_live_compact_sha256(item.eager_pass_1)
        eager_2 = _validated_live_compact_sha256(item.eager_pass_2)
        if eager_1 != eager_2:
            _fail(
                "live_determinism_eager_mismatch",
                "independent eager compact outputs disagree",
            )
        jit_eager_sha256 = None
        jit_compiled_sha256 = None
        if item.selection_position == 0:
            assert item.jit_eager is not None and item.jit_compiled is not None
            jit_eager_sha256 = _validated_live_compact_sha256(item.jit_eager)
            jit_compiled_sha256 = _validated_live_compact_sha256(
                item.jit_compiled
            )
            if not (
                eager_1 == jit_eager_sha256 == jit_compiled_sha256
            ):
                _fail(
                    "live_determinism_jit_mismatch",
                    "position-zero eager and JIT outputs disagree",
                )
        issued[key] = M6WaymaxLiveDeterminismRow(
            selection_position=item.selection_position,
            bundle=item.bundle,
            condition=item.condition,
            cohort_index=member.cohort_index,
            qualification_binding_sha256=member.qualification_binding_sha256,
            status="passed",
            eager_pass_1_sha256=eager_1,
            eager_pass_2_sha256=eager_2,
            jit_eager_sha256=jit_eager_sha256,
            jit_compiled_sha256=jit_compiled_sha256,
            _issuance_capability=_LIVE_DETERMINISM_ISSUER,
        )
    rows = tuple(
        issued.get(key)
        or M6WaymaxLiveDeterminismRow(
            selection_position=key[0],
            bundle=key[1],
            condition=key[2],
            cohort_index=None,
            qualification_binding_sha256=None,
            status="not_applicable",
            eager_pass_1_sha256=None,
            eager_pass_2_sha256=None,
            jit_eager_sha256=None,
            jit_compiled_sha256=None,
            _issuance_capability=_LIVE_DETERMINISM_ISSUER,
        )
        for key in _no_execution_determinism_keys()
    )
    table = M6WaymaxLiveDeterminismTable(
        selected_member_count=len(selection.members),
        selection_binding_sha256=selection_binding,
        primary_domain_sha256=primary_domain.domain_sha256,
        rows=rows,
        _issuance_capability=_LIVE_DETERMINISM_ISSUER,
    )
    table.revalidate(selection=selection, primary_domain=primary_domain)
    return table


def validate_m6_waymax_live_determinism_table(
    value: Any,
    *,
    selection: M6WaymaxSelection,
    primary_domain: M6WaymaxPrimaryDomain,
) -> M6WaymaxLiveDeterminismTable:
    """Reject mappings/replays and validate factory-issued live evidence."""

    if not isinstance(value, M6WaymaxLiveDeterminismTable):
        raise TypeError(
            "live determinism evidence must be a factory-issued "
            "M6WaymaxLiveDeterminismTable"
        )
    value.revalidate(selection=selection, primary_domain=primary_domain)
    return value


@dataclass(frozen=True, slots=True)
class M6WaymaxSceneScalar:
    """Safe sealed scalar used to reconstruct the fixed 128-row store table.

    ``to_store_dict`` deliberately excludes native scenario identifiers, target
    identifiers, source-state hashes, plan/view hashes, and other data-derived
    execution detail.  The opaque cohort index and frozen qualification binding are
    sufficient to enforce complete cross-bundle/cross-measure pairing.
    """

    selection_position: int
    cohort_index: int | None
    qualification_binding_sha256: str | None
    primary_domain_sha256: str
    selection_binding_sha256: str
    selection_supported: bool
    selection_member_count: int
    identity_configuration_fingerprint: str
    primary_b2_configuration_fingerprint: str
    bundle: str
    metric_name: str
    metric_version: str
    value_unit: str
    value: float | None
    responded: bool | None
    responder_latency_s: float | None
    source_pairing_complete: bool
    status: M6WaymaxScalarStatus
    scalar_binding_sha256: str | None = field(default=None, repr=False)

    STORE_FIELDS = frozenset(
        {
            "selection_position",
            "cohort_index",
            "qualification_binding_sha256",
            "primary_domain_sha256",
            "selection_binding_sha256",
            "selection_supported",
            "selection_member_count",
            "identity_configuration_fingerprint",
            "primary_b2_configuration_fingerprint",
            "bundle",
            "metric_name",
            "metric_version",
            "value_unit",
            "value",
            "responded",
            "responder_latency_s",
            "source_pairing_complete",
            "status",
        }
    )

    def __post_init__(self) -> None:
        position = _strict_int(
            self.selection_position,
            "selection_position",
        )
        if position >= M6_WAYMAX_MAX_SCENES:
            raise ValueError("selection_position must lie in [0, 15]")
        object.__setattr__(self, "selection_position", position)
        for name in (
            "primary_domain_sha256",
            "selection_binding_sha256",
            "identity_configuration_fingerprint",
            "primary_b2_configuration_fingerprint",
        ):
            _sha256(getattr(self, name), name)
        if (
            self.identity_configuration_fingerprint
            != _IDENTITY_CONFIGURATION_FINGERPRINT
            or self.primary_b2_configuration_fingerprint
            != _PRIMARY_B2_CONFIGURATION_FINGERPRINT
        ):
            raise ValueError(
                "safe scalar requires exact identity and primary b=2 fingerprints"
            )
        if type(self.selection_supported) is not bool:
            raise TypeError("selection_supported must be a bool")
        member_count = _strict_int(
            self.selection_member_count,
            "selection_member_count",
        )
        if member_count > M6_WAYMAX_MAX_SCENES:
            raise ValueError("selection_member_count cannot exceed 16")
        if self.selection_supported:
            if member_count < M6_WAYMAX_MIN_SUPPORTED_N:
                raise ValueError("supported selection requires at least eight members")
        elif member_count != 0:
            raise ValueError("unsupported selection has no executable members")
        object.__setattr__(self, "selection_member_count", member_count)
        if self.bundle not in M6_WAYMAX_BUNDLES:
            raise ValueError("scalar bundle is not registered")
        if _METRICS_BY_NAME.get(self.metric_name) != (
            self.metric_version,
            self.value_unit,
        ):
            raise ValueError("scalar metric identity/version/unit is not registered")
        if type(self.source_pairing_complete) is not bool:
            raise TypeError("source_pairing_complete must be a bool")
        if self.status not in ("selected", "not_selected"):
            raise ValueError("scalar status must be selected or not_selected")

        if self.status == "not_selected":
            if self.source_pairing_complete:
                raise ValueError("not-selected rows cannot claim source pairing")
            if any(
                value is not None
                for value in (
                    self.cohort_index,
                    self.qualification_binding_sha256,
                    self.value,
                    self.responded,
                    self.responder_latency_s,
                )
            ):
                raise ValueError("not-selected scalar payload must be strict NA")
        else:
            cohort_index = _strict_int(self.cohort_index, "cohort_index")
            if cohort_index >= 128:
                raise ValueError("cohort_index must lie in the frozen 0..127 domain")
            if (
                not self.selection_supported
                or self.selection_position >= self.selection_member_count
            ):
                raise ValueError(
                    "selected outcome requires a supported selection member"
                )
            object.__setattr__(self, "cohort_index", cohort_index)
            _sha256(
                self.qualification_binding_sha256,
                "qualification_binding_sha256",
            )
            object.__setattr__(self, "value", _finite_float(self.value, "value"))
            if self.metric_name == "response_timeliness_s":
                if type(self.responded) is not bool:
                    raise ValueError("selected timeliness scalar requires responded")
                if self.responded:
                    latency = _finite_float(
                        self.responder_latency_s,
                        "responder_latency_s",
                    )
                    if latency < 0.0:
                        raise ValueError(
                            "responder_latency_s must be non-negative"
                        )
                    object.__setattr__(
                        self,
                        "responder_latency_s",
                        latency,
                    )
                elif self.responder_latency_s is not None:
                    raise ValueError(
                        "censored timeliness scalar cannot have responder latency"
                    )
            elif self.responded is not None or self.responder_latency_s is not None:
                raise ValueError("response fields belong only to timeliness")

        expected_hash = hashlib.sha256(
            _SAFE_SCALAR_DOMAIN
            + b"\x00"
            + self.canonical_json.encode("utf-8")
        ).hexdigest()
        if (
            self.scalar_binding_sha256 is not None
            and self.scalar_binding_sha256 != expected_hash
        ):
            raise ValueError("scalar_binding_sha256 does not bind the safe scalar")
        object.__setattr__(self, "scalar_binding_sha256", expected_hash)

    def to_store_dict(self) -> dict[str, Any]:
        """Return only the explicitly safe sealed-store projection."""

        return {
            "bundle": self.bundle,
            "cohort_index": self.cohort_index,
            "metric_name": self.metric_name,
            "metric_version": self.metric_version,
            "identity_configuration_fingerprint": (
                self.identity_configuration_fingerprint
            ),
            "primary_b2_configuration_fingerprint": (
                self.primary_b2_configuration_fingerprint
            ),
            "primary_domain_sha256": self.primary_domain_sha256,
            "qualification_binding_sha256": (
                self.qualification_binding_sha256
            ),
            "responded": self.responded,
            "responder_latency_s": self.responder_latency_s,
            "selection_position": self.selection_position,
            "selection_binding_sha256": self.selection_binding_sha256,
            "selection_member_count": self.selection_member_count,
            "selection_supported": self.selection_supported,
            "source_pairing_complete": self.source_pairing_complete,
            "status": self.status,
            "value": self.value,
            "value_unit": self.value_unit,
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.to_store_dict())

    @classmethod
    def from_store_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "M6WaymaxSceneScalar":
        if not isinstance(payload, Mapping):
            raise TypeError("safe scalar payload must be a mapping")
        if set(payload) != cls.STORE_FIELDS:
            raise ValueError("safe scalar store fields do not match the exact schema")
        return cls(
            selection_position=payload["selection_position"],
            cohort_index=payload["cohort_index"],
            qualification_binding_sha256=payload[
                "qualification_binding_sha256"
            ],
            primary_domain_sha256=payload["primary_domain_sha256"],
            selection_binding_sha256=payload["selection_binding_sha256"],
            selection_supported=payload["selection_supported"],
            selection_member_count=payload["selection_member_count"],
            identity_configuration_fingerprint=payload[
                "identity_configuration_fingerprint"
            ],
            primary_b2_configuration_fingerprint=payload[
                "primary_b2_configuration_fingerprint"
            ],
            bundle=payload["bundle"],
            metric_name=payload["metric_name"],
            metric_version=payload["metric_version"],
            value_unit=payload["value_unit"],
            value=payload["value"],
            responded=payload["responded"],
            responder_latency_s=payload["responder_latency_s"],
            source_pairing_complete=payload["source_pairing_complete"],
            status=payload["status"],
        )

    def revalidate(self) -> None:
        expected_hash = hashlib.sha256(
            _SAFE_SCALAR_DOMAIN
            + b"\x00"
            + self.canonical_json.encode("utf-8")
        ).hexdigest()
        if expected_hash != self.scalar_binding_sha256:
            _fail("scene_scalar_mutated", "safe scene scalar binding failed")


def _scalar_table_binding_sha256(
    *,
    domain: bytes,
    rows: Sequence[M6WaymaxSceneScalar],
    selection_binding_sha256: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(b"\x00")
    digest.update(bytes.fromhex(selection_binding_sha256))
    digest.update(struct.pack(">I", len(rows)))
    for row in rows:
        digest.update(bytes.fromhex(row.scalar_binding_sha256))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M6WaymaxIssuedScalarTable(Sequence[M6WaymaxSceneScalar]):
    """Pre-seal live evidence issued only from validated pair views."""

    rows: tuple[M6WaymaxSceneScalar, ...]
    selection_binding_sha256: str
    primary_domain_sha256: str
    table_binding_sha256: str | None = field(default=None, repr=False)
    _issued_original_binding_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _ISSUED_SCALAR_TABLE_ISSUER:
            raise TypeError("M6WaymaxIssuedScalarTable is builder-issued only")
        rows, _, _ = _normalize_safe_scalar_table(self.rows)
        _sha256(self.selection_binding_sha256, "selection_binding_sha256")
        _sha256(self.primary_domain_sha256, "primary_domain_sha256")
        if any(
            row.selection_binding_sha256 != self.selection_binding_sha256
            or row.primary_domain_sha256 != self.primary_domain_sha256
            for row in rows
        ):
            raise ValueError(
                "issued scalar rows differ from table selection/domain binding"
            )
        object.__setattr__(self, "rows", rows)
        expected = _scalar_table_binding_sha256(
            domain=_ISSUED_SCALAR_TABLE_DOMAIN,
            rows=rows,
            selection_binding_sha256=self.selection_binding_sha256,
        )
        if (
            self.table_binding_sha256 is not None
            and self.table_binding_sha256 != expected
        ):
            raise ValueError("table_binding_sha256 does not bind issued rows")
        object.__setattr__(self, "table_binding_sha256", expected)
        object.__setattr__(
            self,
            "_issued_original_binding_sha256",
            expected,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int | slice,
    ) -> M6WaymaxSceneScalar | tuple[M6WaymaxSceneScalar, ...]:
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)

    def revalidate(self, *, selection: M6WaymaxSelection) -> None:
        selection_binding = _validate_selection(selection)
        rows, _, _ = _normalize_safe_scalar_table(self.rows)
        _cross_bind_scalar_rows_to_selection(rows, selection)
        expected = _scalar_table_binding_sha256(
            domain=_ISSUED_SCALAR_TABLE_DOMAIN,
            rows=rows,
            selection_binding_sha256=selection_binding,
        )
        if (
            selection_binding != self.selection_binding_sha256
            or selection.primary_domain_sha256 != self.primary_domain_sha256
            or expected != self.table_binding_sha256
            or expected != self._issued_original_binding_sha256
        ):
            _fail(
                "issued_scalar_table_mutated",
                "issued scalar evidence failed selection/table revalidation",
            )


@dataclass(frozen=True, slots=True)
class M6WaymaxParsedScalarTable(Sequence[M6WaymaxSceneScalar]):
    """Post-seal structural rows; permanently non-promotable."""

    rows: tuple[M6WaymaxSceneScalar, ...]
    parsed_binding_sha256: str | None = field(default=None, repr=False)
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _PARSED_SCALAR_TABLE_ISSUER:
            raise TypeError(
                "M6WaymaxParsedScalarTable is parser-issued only"
            )
        rows, _, _ = _normalize_safe_scalar_table(self.rows)
        object.__setattr__(self, "rows", rows)
        expected = _scalar_table_binding_sha256(
            domain=_PARSED_SCALAR_TABLE_DOMAIN,
            rows=rows,
            selection_binding_sha256=rows[0].selection_binding_sha256,
        )
        if (
            self.parsed_binding_sha256 is not None
            and self.parsed_binding_sha256 != expected
        ):
            raise ValueError("parsed_binding_sha256 does not bind stored rows")
        object.__setattr__(self, "parsed_binding_sha256", expected)

    @property
    def promotable(self) -> bool:
        return False

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int | slice,
    ) -> M6WaymaxSceneScalar | tuple[M6WaymaxSceneScalar, ...]:
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)

    def revalidate(self) -> None:
        rows, _, _ = _normalize_safe_scalar_table(self.rows)
        expected = _scalar_table_binding_sha256(
            domain=_PARSED_SCALAR_TABLE_DOMAIN,
            rows=rows,
            selection_binding_sha256=rows[0].selection_binding_sha256,
        )
        if expected != self.parsed_binding_sha256:
            _fail(
                "parsed_scalar_table_mutated",
                "post-seal structural table failed revalidation",
            )


@dataclass(frozen=True, slots=True)
class M6WaymaxStoredReconstruction:
    """Non-promotable post-seal verification receipt."""

    pair_n: int
    status: M6WaymaxCellStatus
    selection_binding_sha256: str
    reconstruction_sha256: str
    promotable: bool = False
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _STORED_RECONSTRUCTION_ISSUER:
            raise TypeError(
                "M6WaymaxStoredReconstruction is verifier-issued only"
            )
        pair_n = _strict_int(self.pair_n, "pair_n")
        if pair_n > M6_WAYMAX_MAX_SCENES:
            raise ValueError("stored reconstruction N cannot exceed 16")
        expected_status: M6WaymaxCellStatus = (
            "unsupported"
            if pair_n < M6_WAYMAX_MIN_SUPPORTED_N
            else (
                "insufficient_n"
                if pair_n < M6_WAYMAX_MIN_DESCRIPTIVE_N
                else "descriptive"
            )
        )
        if self.status != expected_status or self.promotable is not False:
            raise ValueError(
                "stored reconstruction status/promotability is not exact"
            )
        _sha256(self.selection_binding_sha256, "selection_binding_sha256")
        _sha256(self.reconstruction_sha256, "reconstruction_sha256")


@dataclass(frozen=True, slots=True)
class M6WaymaxVerifiedStoredSelection:
    """Manifest-verified, post-seal selection projection; never live authority."""

    manifest_sha256: str
    selection_binding_sha256: str
    primary_domain_sha256: str
    supported: bool
    members: tuple[tuple[int, int, str], ...]
    receipt_sha256: str | None = field(default=None, repr=False)
    promotable: bool = False
    _issued_original_receipt_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _VERIFIED_STORED_SELECTION_ISSUER:
            raise TypeError(
                "M6WaymaxVerifiedStoredSelection is verifier-issued only"
            )
        for name in (
            "manifest_sha256",
            "selection_binding_sha256",
            "primary_domain_sha256",
        ):
            _sha256(getattr(self, name), name)
        if type(self.supported) is not bool or self.promotable is not False:
            raise ValueError(
                "stored selection support/promotability must be exact"
            )
        normalized: list[tuple[int, int, str]] = []
        for member in self.members:
            if not isinstance(member, tuple) or len(member) != 3:
                raise TypeError(
                    "stored selection members must be (position, cohort, binding)"
                )
            position = _strict_int(member[0], "selection_position")
            cohort = _strict_int(member[1], "cohort_index")
            if cohort >= 128:
                raise ValueError("stored selection cohort must lie in 0..127")
            binding = _sha256(
                member[2],
                "qualification_binding_sha256",
            )
            normalized.append((position, cohort, binding))
        members = tuple(normalized)
        if tuple(item[0] for item in members) != tuple(range(len(members))):
            raise ValueError("stored selection positions must be one exact prefix")
        if len({item[1] for item in members}) != len(members):
            raise ValueError("stored selection cohorts must be unique")
        if len({item[2] for item in members}) != len(members):
            raise ValueError("stored qualification bindings must be unique")
        if self.supported:
            if not 8 <= len(members) <= 16:
                raise ValueError("supported stored selection requires 8..16 members")
        elif members:
            raise ValueError("unsupported stored selection has no members")
        object.__setattr__(self, "members", members)
        expected = self._binding_sha256()
        if self.receipt_sha256 is not None and self.receipt_sha256 != expected:
            raise ValueError("receipt_sha256 does not bind stored selection")
        object.__setattr__(self, "receipt_sha256", expected)
        object.__setattr__(
            self,
            "_issued_original_receipt_sha256",
            expected,
        )

    def _binding_sha256(self) -> str:
        payload = {
            "manifest_sha256": self.manifest_sha256,
            "members": [list(member) for member in self.members],
            "primary_domain_sha256": self.primary_domain_sha256,
            "promotable": False,
            "selection_binding_sha256": self.selection_binding_sha256,
            "supported": self.supported,
        }
        return hashlib.sha256(
            _VERIFIED_STORED_SELECTION_DOMAIN
            + b"\x00"
            + _canonical_json(payload).encode("utf-8")
        ).hexdigest()

    def revalidate(self) -> None:
        if (
            self._binding_sha256() != self.receipt_sha256
            or self.receipt_sha256
            != self._issued_original_receipt_sha256
            or self.promotable is not False
        ):
            _fail(
                "stored_selection_receipt_mutated",
                "verified stored-selection receipt failed revalidation",
            )


def _target_accelerations(
    view: M6WaymaxTwentyTransitionPairView,
) -> tuple[np.ndarray, np.ndarray]:
    baseline_speed = np.hypot(
        view.baseline_target_vx,
        view.baseline_target_vy,
    )
    treatment_speed = np.hypot(
        view.treatment_target_vx,
        view.treatment_target_vy,
    )
    baseline = np.diff(baseline_speed) / view.dt_s
    treatment = np.diff(treatment_speed) / view.dt_s
    if not bool(np.all(np.isfinite(baseline))) or not bool(
        np.all(np.isfinite(treatment))
    ):
        _fail("metric_nonfinite", "target acceleration is non-finite")
    return baseline, treatment


def _measure_result(
    view: M6WaymaxTwentyTransitionPairView,
    *,
    metric_name: str,
    value: float,
    responded: bool | None = None,
    responder_latency_s: float | None = None,
) -> M6WaymaxPairedMeasureResult:
    version, unit = _METRICS_BY_NAME[metric_name]
    return M6WaymaxPairedMeasureResult(
        selection_position=view.selection_position,
        cohort_index=view.cohort_index,
        scenario_id=view.scenario_id,
        bundle=view.bundle,
        metric_name=metric_name,
        metric_version=version,
        value_unit=unit,
        value=value,
        target_agent_id=view.target_agent_id,
        qualification_binding_sha256=view.qualification_binding_sha256,
        intervention_configuration_fingerprint=(
            view.intervention_configuration_fingerprint
        ),
        view_binding_sha256=view.view_binding_sha256,
        responded=responded,
        responder_latency_s=responder_latency_s,
    )


def compute_m6_waymax_paired_measures(
    view: M6WaymaxTwentyTransitionPairView,
) -> tuple[M6WaymaxPairedMeasureResult, ...]:
    """Recompute the exact four preregistered measures over one 20-step view."""

    if not isinstance(view, M6WaymaxTwentyTransitionPairView):
        raise TypeError("view must be an M6WaymaxTwentyTransitionPairView")
    view.revalidate()
    baseline_acceleration, treatment_acceleration = _target_accelerations(view)
    impulse = math.fsum(
        float(component)
        for component in (
            np.maximum(
                0.0,
                baseline_acceleration - treatment_acceleration,
            )
            * view.dt_s
        )
    )

    delta_acceleration = treatment_acceleration - baseline_acceleration
    response_end: int | None = None
    run_start: int | None = None
    for transition in range(1, M6_WAYMAX_TRANSITIONS):
        if (
            float(delta_acceleration[transition])
            <= M6_RESPONSE_ACCELERATION_THRESHOLD_MPS2
        ):
            if run_start is None:
                run_start = transition
            duration_micros = (
                int(view.timestamps_micros[transition + 1])
                - int(view.timestamps_micros[run_start])
            )
            if duration_micros >= int(
                round(M6_RESPONSE_PERSISTENCE_S * 1_000_000.0)
            ):
                response_end = transition
                break
        else:
            run_start = None
    window_s = (
        int(view.timestamps_micros[-1]) - int(view.timestamps_micros[0])
    ) * 1e-6
    responded = response_end is not None
    if responded:
        assert response_end is not None
        event_time_s = (
            int(view.timestamps_micros[response_end + 1])
            - int(view.timestamps_micros[0])
        ) * 1e-6
        timeliness = window_s - min(event_time_s, window_s)
    else:
        event_time_s = None
        timeliness = 0.0

    current_vx = float(view.baseline_target_vx[0])
    current_vy = float(view.baseline_target_vy[0])
    current_speed = math.hypot(current_vx, current_vy)
    if current_speed > 1e-12:
        gap_hx, gap_hy = current_vx / current_speed, current_vy / current_speed
    else:
        current_heading = float(view.baseline_target_heading[0])
        gap_hx, gap_hy = math.cos(current_heading), math.sin(current_heading)
    half_length = 0.5 * (view.target_length_m + view.ego_length_m)
    baseline_gaps = (
        (view.baseline_ego_x[1:] - view.baseline_target_x[1:]) * gap_hx
        + (view.baseline_ego_y[1:] - view.baseline_target_y[1:]) * gap_hy
        - half_length
    )
    treatment_gaps = (
        (view.treatment_ego_x[1:] - view.treatment_target_x[1:]) * gap_hx
        + (view.treatment_ego_y[1:] - view.treatment_target_y[1:]) * gap_hy
        - half_length
    )
    gap_change = float(np.min(treatment_gaps)) - float(np.min(baseline_gaps))

    heading = float(view.baseline_target_heading[0])
    progress_hx, progress_hy = math.cos(heading), math.sin(heading)
    origin_x = float(view.baseline_target_x[0])
    origin_y = float(view.baseline_target_y[0])
    baseline_progress = (
        (float(view.baseline_target_x[-1]) - origin_x) * progress_hx
        + (float(view.baseline_target_y[-1]) - origin_y) * progress_hy
    )
    treatment_progress = (
        (float(view.treatment_target_x[-1]) - origin_x) * progress_hx
        + (float(view.treatment_target_y[-1]) - origin_y) * progress_hy
    )
    progress_loss = baseline_progress - treatment_progress

    return (
        _measure_result(
            view,
            metric_name="additional_target_braking_impulse_mps",
            value=impulse,
        ),
        _measure_result(
            view,
            metric_name="response_timeliness_s",
            value=timeliness,
            responded=responded,
            responder_latency_s=event_time_s,
        ),
        _measure_result(
            view,
            metric_name="minimum_longitudinal_bumper_gap_change_m",
            value=gap_change,
        ),
        _measure_result(
            view,
            metric_name="target_progress_loss_m",
            value=progress_loss,
        ),
    )


@dataclass(frozen=True, slots=True)
class M6WaymaxReweightingBand:
    """Pointwise fixed-cohort reweighting sensitivity band."""

    level: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        level = _finite_float(self.level, "level")
        lower = _finite_float(self.lower, "lower")
        upper = _finite_float(self.upper, "upper")
        if level != M6_WAYMAX_POINTWISE_LEVEL:
            raise ValueError("Waymax pointwise level must equal exactly 0.95")
        if lower > upper:
            raise ValueError("band lower cannot exceed upper")
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class M6WaymaxResamplingKey:
    """Canonical deterministic entropy identity for one secondary Waymax cell."""

    canonical_json: str
    sha256: str
    digest_words: tuple[int, ...]
    pair_n: int

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_json, str):
            raise TypeError("canonical_json must be a string")
        _sha256(self.sha256, "sha256")
        if (
            hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()
            != self.sha256
        ):
            raise ValueError("resampling sha256 does not match canonical_json")
        words = tuple(self.digest_words)
        if (
            len(words) != 8
            or any(
                isinstance(word, (bool, np.bool_))
                or not isinstance(word, Integral)
                or not 0 <= int(word) <= np.iinfo(np.uint32).max
                for word in words
            )
        ):
            raise ValueError("digest_words must contain eight uint32 values")
        if tuple(struct.unpack(">8I", bytes.fromhex(self.sha256))) != tuple(
            int(word) for word in words
        ):
            raise ValueError("digest_words do not match sha256")
        pair_n = _strict_int(self.pair_n, "pair_n", minimum=1)
        if not M6_WAYMAX_MIN_DESCRIPTIVE_N <= pair_n <= M6_WAYMAX_MAX_SCENES:
            raise ValueError("resampling requires N in [10, 16]")
        object.__setattr__(self, "digest_words", tuple(int(word) for word in words))
        object.__setattr__(self, "pair_n", pair_n)
        self._validate_payload()

    def _validate_payload(self) -> None:
        try:
            payload = json.loads(self.canonical_json)
        except json.JSONDecodeError as exc:
            raise ValueError("resampling key must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("resampling key payload must be an object")
        expected_fields = {
            "base_seed",
            "bundle",
            "intervention_configuration_fingerprint",
            "metric_name",
            "metric_version",
            "paired_n",
            "policy_access_role",
            "policy_name",
            "resamples",
            "statistics_schema_version",
        }
        if set(payload) != expected_fields:
            raise ValueError("resampling key fields do not match the exact schema")
        if _canonical_json(payload) != self.canonical_json:
            raise ValueError("resampling key is not canonical JSON")
        bundle = payload["bundle"]
        metric_name = payload["metric_name"]
        if bundle not in M6_WAYMAX_BUNDLES:
            raise ValueError("resampling key bundle is not registered")
        metric_identity = _METRICS_BY_NAME.get(metric_name)
        if (
            not isinstance(metric_name, str)
            or metric_identity is None
            or payload["metric_version"] != metric_identity[0]
        ):
            raise ValueError("resampling key metric identity/version drifted")
        expected_policy, expected_access = _BUNDLE_POLICY_ACCESS[bundle]
        if (
            payload["policy_name"] != expected_policy
            or payload["policy_access_role"] != expected_access
        ):
            raise ValueError("resampling key policy/access role drifted")
        if (
            type(payload["base_seed"]) is not int
            or payload["base_seed"] != M6_WAYMAX_BASE_SEED
            or type(payload["resamples"]) is not int
            or payload["resamples"] != M6_WAYMAX_RESAMPLES
            or type(payload["paired_n"]) is not int
            or payload["paired_n"] != self.pair_n
            or payload["statistics_schema_version"]
            != M6_WAYMAX_STATISTICS_SCHEMA_VERSION
            or payload["intervention_configuration_fingerprint"]
            != _PRIMARY_B2_CONFIGURATION_FINGERPRINT
        ):
            raise ValueError("resampling key constants or types drifted")

    def revalidate(self) -> None:
        try:
            self._validate_payload()
        except (TypeError, ValueError):
            _fail("resampling_key_mutated", "resampling key is not canonical JSON")
        digest = hashlib.sha256(self.canonical_json.encode("utf-8")).digest()
        expected_words = struct.unpack(">8I", digest)
        if digest.hex() != self.sha256 or tuple(self.digest_words) != expected_words:
            _fail("resampling_key_mutated", "resampling key digest changed")


def _make_resampling_key(
    *,
    bundle: str,
    metric_name: str,
    metric_version: str,
    intervention_configuration_fingerprint: str,
    pair_n: int,
) -> M6WaymaxResamplingKey:
    payload = {
        "base_seed": M6_WAYMAX_BASE_SEED,
        "bundle": bundle,
        "intervention_configuration_fingerprint": (
            intervention_configuration_fingerprint
        ),
        "metric_name": metric_name,
        "metric_version": metric_version,
        "paired_n": pair_n,
        "policy_access_role": _BUNDLE_POLICY_ACCESS[bundle][1],
        "policy_name": _BUNDLE_POLICY_ACCESS[bundle][0],
        "resamples": M6_WAYMAX_RESAMPLES,
        "statistics_schema_version": M6_WAYMAX_STATISTICS_SCHEMA_VERSION,
    }
    canonical_json = _canonical_json(payload)
    digest = hashlib.sha256(canonical_json.encode("utf-8")).digest()
    return M6WaymaxResamplingKey(
        canonical_json=canonical_json,
        sha256=digest.hex(),
        digest_words=struct.unpack(">8I", digest),
        pair_n=pair_n,
    )


def _resampled_means(
    values: np.ndarray,
    key: M6WaymaxResamplingKey,
) -> np.ndarray:
    key.revalidate()
    if (
        values.dtype != np.dtype(np.float64)
        or values.shape != (key.pair_n,)
        or not bool(np.all(np.isfinite(values)))
    ):
        raise ValueError("resampling values must be finite float64[N]")
    rng = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence(
                [M6_WAYMAX_BASE_SEED, *key.digest_words]
            )
        )
    )
    draws = rng.integers(
        0,
        key.pair_n,
        size=(M6_WAYMAX_RESAMPLES, key.pair_n),
        dtype=np.int64,
    )
    means = np.mean(values[draws], axis=1, dtype=np.float64)
    if not bool(np.all(np.isfinite(means))):
        _fail("resampling_nonfinite", "Waymax resampled means are non-finite")
    return means


@dataclass(frozen=True, slots=True)
class M6WaymaxCellResult:
    """One of the exact eight bounded-Waymax secondary cells."""

    bundle: str
    metric_name: str
    metric_version: str
    value_unit: str
    pair_n: int
    cohort_indices: tuple[int, ...]
    scene_scalar_sha256s: tuple[str, ...]
    status: M6WaymaxCellStatus
    arithmetic_mean: float | None
    median: float | None
    pointwise_band: M6WaymaxReweightingBand | None
    responder_n: int | None
    censor_n: int | None
    resampling_key: M6WaymaxResamplingKey | None
    directional_language_allowed: bool = False
    cell_binding_sha256: str | None = field(default=None, repr=False)
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _CELL_RESULT_ISSUER:
            raise TypeError("M6WaymaxCellResult is analyzer-issued only")
        if self.bundle not in M6_WAYMAX_BUNDLES:
            raise ValueError("cell bundle is not registered")
        if _METRICS_BY_NAME.get(self.metric_name) != (
            self.metric_version,
            self.value_unit,
        ):
            raise ValueError("cell metric identity/version/unit is not registered")
        pair_n = _strict_int(self.pair_n, "pair_n")
        if pair_n > M6_WAYMAX_MAX_SCENES:
            raise ValueError("Waymax cell N cannot exceed 16")
        indices = tuple(
            _strict_int(index, "cohort_index") for index in self.cohort_indices
        )
        if len(indices) != pair_n or indices != tuple(sorted(set(indices))):
            raise ValueError("cohort_indices must be complete, unique, and sorted")
        result_hashes = tuple(self.scene_scalar_sha256s)
        if len(result_hashes) != pair_n:
            raise ValueError("scene_scalar_sha256s must have one hash per scene")
        for result_hash in result_hashes:
            _sha256(result_hash, "scene_result_sha256")
        expected_status: M6WaymaxCellStatus
        if pair_n < M6_WAYMAX_MIN_SUPPORTED_N:
            expected_status = "unsupported"
        elif pair_n < M6_WAYMAX_MIN_DESCRIPTIVE_N:
            expected_status = "insufficient_n"
        else:
            expected_status = "descriptive"
        if self.status != expected_status:
            raise ValueError("cell status does not match N")
        if (
            type(self.directional_language_allowed) is not bool
            or self.directional_language_allowed
        ):
            raise ValueError("Waymax secondary cells never allow directional language")
        if expected_status == "descriptive":
            for name in ("arithmetic_mean", "median"):
                object.__setattr__(
                    self,
                    name,
                    _finite_float(getattr(self, name), name),
                )
            if not isinstance(self.pointwise_band, M6WaymaxReweightingBand):
                raise ValueError("descriptive cell requires a pointwise band")
            if not isinstance(self.resampling_key, M6WaymaxResamplingKey):
                raise ValueError("descriptive cell requires a resampling key")
            self.resampling_key.revalidate()
            if self.resampling_key.pair_n != pair_n:
                raise ValueError("resampling key N differs from cell N")
        elif any(
            value is not None
            for value in (
                self.arithmetic_mean,
                self.median,
                self.pointwise_band,
                self.resampling_key,
            )
        ):
            raise ValueError("N < 10 must suppress point and band values")
        if self.metric_name == "response_timeliness_s":
            responder_n = _strict_int(self.responder_n, "responder_n")
            censor_n = _strict_int(self.censor_n, "censor_n")
            if responder_n + censor_n != pair_n:
                raise ValueError("responder and censor counts must sum to N")
            object.__setattr__(self, "responder_n", responder_n)
            object.__setattr__(self, "censor_n", censor_n)
        elif self.responder_n is not None or self.censor_n is not None:
            raise ValueError("response counts belong only to timeliness")
        object.__setattr__(self, "pair_n", pair_n)
        object.__setattr__(self, "cohort_indices", indices)
        object.__setattr__(self, "scene_scalar_sha256s", result_hashes)
        expected_hash = self._binding_sha256()
        if (
            self.cell_binding_sha256 is not None
            and self.cell_binding_sha256 != expected_hash
        ):
            raise ValueError("cell_binding_sha256 does not bind the cell")
        object.__setattr__(self, "cell_binding_sha256", expected_hash)

    def _payload(self) -> dict[str, Any]:
        return {
            "arithmetic_mean": self.arithmetic_mean,
            "bundle": self.bundle,
            "censor_n": self.censor_n,
            "cohort_indices": list(self.cohort_indices),
            "directional_language_allowed": self.directional_language_allowed,
            "interpretation": M6_WAYMAX_REWEIGHTING_INTERPRETATION,
            "median": self.median,
            "metric_name": self.metric_name,
            "metric_version": self.metric_version,
            "pair_n": self.pair_n,
            "pointwise_band": (
                None
                if self.pointwise_band is None
                else {
                    "level": self.pointwise_band.level,
                    "lower": self.pointwise_band.lower,
                    "upper": self.pointwise_band.upper,
                }
            ),
            "resampling_key": (
                None
                if self.resampling_key is None
                else {
                    "canonical_json": self.resampling_key.canonical_json,
                    "digest_words": list(self.resampling_key.digest_words),
                    "pair_n": self.resampling_key.pair_n,
                    "sha256": self.resampling_key.sha256,
                }
            ),
            "responder_n": self.responder_n,
            "scene_scalar_sha256s": list(self.scene_scalar_sha256s),
            "schema_version": M6_WAYMAX_STATISTICS_SCHEMA_VERSION,
            "status": self.status,
            "value_unit": self.value_unit,
        }

    def _binding_sha256(self) -> str:
        return hashlib.sha256(
            _CELL_RESULT_DOMAIN
            + b"\x00"
            + _canonical_json(self._payload()).encode("utf-8")
        ).hexdigest()

    def revalidate(self) -> None:
        if self.resampling_key is not None:
            self.resampling_key.revalidate()
        if self._binding_sha256() != self.cell_binding_sha256:
            _fail("cell_result_mutated", "Waymax cell result binding failed")


@dataclass(frozen=True, slots=True)
class M6WaymaxMatrixResult:
    """Exact two-bundle by four-measure secondary matrix."""

    pair_n: int
    cohort_indices: tuple[int, ...]
    intervention_configuration_fingerprint: str
    scene_scalars: tuple[M6WaymaxSceneScalar, ...]
    cells: tuple[M6WaymaxCellResult, ...]
    matrix_binding_sha256: str | None = field(default=None, repr=False)
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _MATRIX_RESULT_ISSUER:
            raise TypeError("M6WaymaxMatrixResult is analyzer-issued only")
        pair_n = _strict_int(self.pair_n, "pair_n")
        if pair_n > M6_WAYMAX_MAX_SCENES:
            raise ValueError("Waymax matrix N cannot exceed 16")
        indices = tuple(self.cohort_indices)
        if len(indices) != pair_n or indices != tuple(sorted(set(indices))):
            raise ValueError("matrix cohort_indices are incomplete or unordered")
        _sha256(
            self.intervention_configuration_fingerprint,
            "intervention_configuration_fingerprint",
        )
        if (
            self.intervention_configuration_fingerprint
            != _PRIMARY_B2_CONFIGURATION_FINGERPRINT
        ):
            raise ValueError(
                "matrix requires the exact registered primary b=2 fingerprint"
            )
        scalars = tuple(self.scene_scalars)
        if len(scalars) != (
            M6_WAYMAX_MAX_SCENES * M6_WAYMAX_CELL_COUNT
        ):
            raise ValueError("matrix needs the exact fixed 128-row scalar table")
        for scalar in scalars:
            if not isinstance(scalar, M6WaymaxSceneScalar):
                raise TypeError("scene_scalars must contain safe scene scalars")
            scalar.revalidate()
        expected_scalar_keys = tuple(
            (position, bundle, metric_name)
            for position in range(M6_WAYMAX_MAX_SCENES)
            for bundle in M6_WAYMAX_BUNDLES
            for metric_name, _, _ in _METRICS
        )
        actual_scalar_keys = tuple(
            (
                scalar.selection_position,
                scalar.bundle,
                scalar.metric_name,
            )
            for scalar in scalars
        )
        if actual_scalar_keys != expected_scalar_keys:
            raise ValueError("matrix scalar table is not in exact canonical order")
        (
            normalized_scalars,
            selected_positions,
            normalized_indices,
        ) = _normalize_safe_scalar_table(scalars)
        if (
            tuple(row.scalar_binding_sha256 for row in normalized_scalars)
            != tuple(row.scalar_binding_sha256 for row in scalars)
            or len(selected_positions) != pair_n
            or normalized_indices != indices
        ):
            raise ValueError(
                "matrix N/cohort does not match the complete safe scalar table"
            )
        selected_scalars = tuple(
            scalar for scalar in scalars if scalar.status == "selected"
        )
        if len(selected_scalars) != pair_n * M6_WAYMAX_CELL_COUNT:
            raise ValueError("matrix scalar table selected count differs from N")
        if {
            int(scalar.cohort_index) for scalar in selected_scalars
        } != set(indices):
            raise ValueError("matrix scalar table cohort differs from matrix cohort")
        cells = tuple(self.cells)
        expected_identities = tuple(
            (bundle, metric_name, metric_version)
            for bundle in M6_WAYMAX_BUNDLES
            for metric_name, metric_version, _ in _METRICS
        )
        actual_identities = tuple(
            (cell.bundle, cell.metric_name, cell.metric_version) for cell in cells
        )
        if (
            len(cells) != M6_WAYMAX_CELL_COUNT
            or actual_identities != expected_identities
        ):
            raise ValueError("matrix cells are not the exact registered eight")
        for cell in cells:
            cell.revalidate()
            if cell.pair_n != pair_n or cell.cohort_indices != indices:
                raise ValueError("matrix cells do not share one complete cohort")
            cell_scalars = sorted(
                (
                    scalar
                    for scalar in selected_scalars
                    if scalar.bundle == cell.bundle
                    and scalar.metric_name == cell.metric_name
                ),
                key=lambda scalar: int(scalar.cohort_index),
            )
            if cell.scene_scalar_sha256s != tuple(
                scalar.scalar_binding_sha256 for scalar in cell_scalars
            ):
                raise ValueError("matrix cell is not derived from its safe scalars")
        expected_cells = _analyze_safe_scalar_cells(
            scalar_table=scalars,
            selected_positions=selected_positions,
            cohort_indices=indices,
            intervention_configuration_fingerprint=(
                self.intervention_configuration_fingerprint
            ),
        )
        for supplied, expected in zip(cells, expected_cells, strict=True):
            supplied_key = supplied.resampling_key
            expected_key = expected.resampling_key
            if (
                supplied._payload() != expected._payload()
                or (
                    None
                    if supplied_key is None
                    else supplied_key.canonical_json
                )
                != (
                    None
                    if expected_key is None
                    else expected_key.canonical_json
                )
            ):
                raise ValueError(
                    "matrix cell statistics were not reconstructed from 128 scalars"
                )
        object.__setattr__(self, "pair_n", pair_n)
        object.__setattr__(self, "cohort_indices", indices)
        object.__setattr__(self, "scene_scalars", scalars)
        object.__setattr__(self, "cells", cells)
        expected_hash = self._binding_sha256()
        if (
            self.matrix_binding_sha256 is not None
            and self.matrix_binding_sha256 != expected_hash
        ):
            raise ValueError("matrix_binding_sha256 does not bind the matrix")
        object.__setattr__(self, "matrix_binding_sha256", expected_hash)

    @property
    def status(self) -> M6WaymaxCellStatus:
        return self.cells[0].status

    def _binding_sha256(self) -> str:
        payload = {
            "cell_sha256s": [cell.cell_binding_sha256 for cell in self.cells],
            "cohort_indices": list(self.cohort_indices),
            "intervention_configuration_fingerprint": (
                self.intervention_configuration_fingerprint
            ),
            "pair_n": self.pair_n,
            "scene_scalar_sha256s": [
                scalar.scalar_binding_sha256 for scalar in self.scene_scalars
            ],
            "schema_version": M6_WAYMAX_STATISTICS_SCHEMA_VERSION,
        }
        return hashlib.sha256(
            _MATRIX_RESULT_DOMAIN
            + b"\x00"
            + _canonical_json(payload).encode("utf-8")
        ).hexdigest()

    def revalidate(self) -> None:
        try:
            self.__post_init__(_MATRIX_RESULT_ISSUER)
        except (TypeError, ValueError) as exc:
            _fail(
                "matrix_result_mutated",
                f"Waymax matrix reconstruction failed: {exc}",
            )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return math.fsum((ordered[midpoint - 1], ordered[midpoint])) / 2.0


def _normalize_safe_scalar_table(
    scene_scalars: Sequence[M6WaymaxSceneScalar],
) -> tuple[
    tuple[M6WaymaxSceneScalar, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Validate the fixed grid and return canonical rows/positions/cohort."""

    if isinstance(scene_scalars, (str, bytes)) or not isinstance(
        scene_scalars,
        Sequence,
    ):
        raise TypeError("scene_scalars must be a sequence")
    rows = tuple(scene_scalars)
    expected_row_count = M6_WAYMAX_MAX_SCENES * M6_WAYMAX_CELL_COUNT
    if len(rows) != expected_row_count:
        raise ValueError("scene_scalars must contain exactly 128 rows")
    if any(not isinstance(row, M6WaymaxSceneScalar) for row in rows):
        raise TypeError("scene_scalars must contain M6WaymaxSceneScalar values")
    for row in rows:
        row.revalidate()

    by_key: dict[tuple[int, str, str], M6WaymaxSceneScalar] = {}
    for row in rows:
        key = (row.selection_position, row.bundle, row.metric_name)
        if key in by_key:
            _fail("duplicate_scalar", "safe scalar table contains a duplicate key")
        by_key[key] = row
    expected_keys = {
        (position, bundle, metric_name)
        for position in range(M6_WAYMAX_MAX_SCENES)
        for bundle in M6_WAYMAX_BUNDLES
        for metric_name, _, _ in _METRICS
    }
    if set(by_key) != expected_keys:
        _fail("scalar_grid", "safe scalar table is not the exact 16x2x4 grid")
    canonical_rows = tuple(
        by_key[(position, bundle, metric_name)]
        for position in range(M6_WAYMAX_MAX_SCENES)
        for bundle in M6_WAYMAX_BUNDLES
        for metric_name, _, _ in _METRICS
    )

    provenance = {
        (
            row.primary_domain_sha256,
            row.selection_binding_sha256,
            row.selection_supported,
            row.selection_member_count,
            row.identity_configuration_fingerprint,
            row.primary_b2_configuration_fingerprint,
        )
        for row in canonical_rows
    }
    if len(provenance) != 1:
        _fail(
            "scalar_provenance",
            "safe scalar rows mix selection/domain/intervention provenance",
        )
    (
        _,
        _,
        selection_supported,
        selection_member_count,
        _,
        _,
    ) = next(iter(provenance))

    selected_positions: list[int] = []
    selected_identities: list[tuple[int, str]] = []
    for position in range(M6_WAYMAX_MAX_SCENES):
        position_rows = canonical_rows[
            position * M6_WAYMAX_CELL_COUNT :
            (position + 1) * M6_WAYMAX_CELL_COUNT
        ]
        statuses = {row.status for row in position_rows}
        if len(statuses) != 1:
            _fail("selection_pairing", "position mixes selected and NA scalar rows")
        status = next(iter(statuses))
        if status == "selected":
            identities = {
                (
                    int(row.cohort_index),
                    str(row.qualification_binding_sha256),
                )
                for row in position_rows
            }
            if len(identities) != 1:
                _fail(
                    "qualification_pairing",
                    "position rows disagree on cohort/qualification identity",
                )
            if not all(row.source_pairing_complete for row in position_rows):
                _fail(
                    "source_pairing",
                    "selected scalar table contains an incomplete source pair",
                )
            selected_positions.append(position)
            selected_identities.append(next(iter(identities)))
    if selected_positions != list(range(len(selected_positions))):
        _fail("selection_positions", "selected positions must be one exact prefix")
    if selection_supported:
        if (
            len(selected_positions) != selection_member_count
            or len(selected_positions) < M6_WAYMAX_MIN_SUPPORTED_N
        ):
            _fail(
                "selection_completeness",
                "supported selection requires outcomes for every selected member",
            )
    elif selected_positions or selection_member_count != 0:
        _fail(
            "selection_unsupported",
            "unsupported selection must retain an all-NA outcome table",
        )

    cohort_values = [identity[0] for identity in selected_identities]
    qualification_values = [identity[1] for identity in selected_identities]
    if len(set(cohort_values)) != len(cohort_values):
        _fail("cohort_pairing", "cohort_index is reused across selection positions")
    if len(set(qualification_values)) != len(qualification_values):
        _fail(
            "qualification_pairing",
            "qualification binding is reused across selection positions",
        )
    return (
        canonical_rows,
        tuple(selected_positions),
        tuple(sorted(cohort_values)),
    )


def _cross_bind_scalar_rows_to_selection(
    rows: Sequence[M6WaymaxSceneScalar],
    selection: M6WaymaxSelection,
) -> None:
    """Bind every fixed-grid position to the canonical selection ledger."""

    selection_binding = _validate_selection(selection)
    canonical_rows, selected_positions, _ = _normalize_safe_scalar_table(rows)
    expected_positions = (
        tuple(range(len(selection.members))) if selection.supported else ()
    )
    if selected_positions != expected_positions:
        _fail(
            "selection_cross_binding",
            "scalar outcomes differ from canonical selected positions",
        )
    for row in canonical_rows:
        if (
            row.selection_binding_sha256 != selection_binding
            or row.primary_domain_sha256 != selection.primary_domain_sha256
            or row.selection_supported != selection.supported
            or row.selection_member_count != len(selection.members)
        ):
            _fail(
                "selection_cross_binding",
                "scalar global selection/domain provenance is not canonical",
            )
        if row.selection_position < len(selection.members):
            member = selection.members[row.selection_position]
            if (
                row.status != "selected"
                or row.cohort_index != member.cohort_index
                or row.qualification_binding_sha256
                != member.qualification_binding_sha256
            ):
                _fail(
                    "selection_cross_binding",
                    "scalar position does not match its qualification-ledger member",
                )
        elif row.status != "not_selected":
            _fail(
                "selection_cross_binding",
                "unselected scalar position retained an outcome",
            )


def _analyze_safe_scalar_cells(
    *,
    scalar_table: tuple[M6WaymaxSceneScalar, ...],
    selected_positions: tuple[int, ...],
    cohort_indices: tuple[int, ...],
    intervention_configuration_fingerprint: str,
) -> tuple[M6WaymaxCellResult, ...]:
    pair_n = len(selected_positions)
    cells: list[M6WaymaxCellResult] = []
    rows_by_identity: dict[
        tuple[str, str],
        list[M6WaymaxSceneScalar],
    ] = {
        (bundle, metric_name): []
        for bundle in M6_WAYMAX_BUNDLES
        for metric_name, _, _ in _METRICS
    }
    for row in scalar_table:
        if row.status == "selected":
            rows_by_identity[(row.bundle, row.metric_name)].append(row)

    for bundle in M6_WAYMAX_BUNDLES:
        for metric_name, metric_version, unit in _METRICS:
            rows = sorted(
                rows_by_identity[(bundle, metric_name)],
                key=lambda row: int(row.cohort_index),
            )
            if tuple(row.cohort_index for row in rows) != cohort_indices:
                _fail(
                    "effect_pairing",
                    "cell effects are not complete in ascending cohort order",
                )
            values = np.asarray(
                [float(row.value) for row in rows],
                dtype=np.float64,
            )
            if not bool(np.all(np.isfinite(values))):
                _fail("effect_nonfinite", "cell contains non-finite scene effects")
            if pair_n < M6_WAYMAX_MIN_SUPPORTED_N:
                status: M6WaymaxCellStatus = "unsupported"
            elif pair_n < M6_WAYMAX_MIN_DESCRIPTIVE_N:
                status = "insufficient_n"
            else:
                status = "descriptive"
            if status == "descriptive":
                arithmetic_mean: float | None = (
                    math.fsum(float(value) for value in values) / pair_n
                )
                median: float | None = _median(values)
                resampling_key: M6WaymaxResamplingKey | None = (
                    _make_resampling_key(
                        bundle=bundle,
                        metric_name=metric_name,
                        metric_version=metric_version,
                        intervention_configuration_fingerprint=(
                            intervention_configuration_fingerprint
                        ),
                        pair_n=pair_n,
                    )
                )
                sampled_means = _resampled_means(values, resampling_key)
                quantiles = np.quantile(
                    sampled_means,
                    [0.025, 0.975],
                    method="linear",
                )
                pointwise_band: M6WaymaxReweightingBand | None = (
                    M6WaymaxReweightingBand(
                        level=M6_WAYMAX_POINTWISE_LEVEL,
                        lower=float(quantiles[0]),
                        upper=float(quantiles[1]),
                    )
                )
            else:
                arithmetic_mean = None
                median = None
                resampling_key = None
                pointwise_band = None
            if metric_name == "response_timeliness_s":
                responder_n = sum(row.responded is True for row in rows)
                censor_n = pair_n - responder_n
            else:
                responder_n = None
                censor_n = None
            cells.append(
                M6WaymaxCellResult(
                    bundle=bundle,
                    metric_name=metric_name,
                    metric_version=metric_version,
                    value_unit=unit,
                    pair_n=pair_n,
                    cohort_indices=cohort_indices,
                    scene_scalar_sha256s=tuple(
                        row.scalar_binding_sha256 for row in rows
                    ),
                    status=status,
                    arithmetic_mean=arithmetic_mean,
                    median=median,
                    pointwise_band=pointwise_band,
                    responder_n=responder_n,
                    censor_n=censor_n,
                    resampling_key=resampling_key,
                    _issuance_capability=_CELL_RESULT_ISSUER,
                )
            )
    return tuple(cells)


def analyze_m6_waymax_cells(
    scene_scalars: M6WaymaxIssuedScalarTable,
    *,
    selection: M6WaymaxSelection,
    intervention_configuration_fingerprint: str,
) -> M6WaymaxMatrixResult:
    """Derive live/promotable evidence only from a builder-issued table."""

    _sha256(
        intervention_configuration_fingerprint,
        "intervention_configuration_fingerprint",
    )
    if (
        intervention_configuration_fingerprint
        != _PRIMARY_B2_CONFIGURATION_FINGERPRINT
    ):
        raise ValueError(
            "Waymax matrix requires the exact registered primary b=2 fingerprint"
        )
    if not isinstance(scene_scalars, M6WaymaxIssuedScalarTable):
        raise TypeError(
            "live analysis requires M6WaymaxIssuedScalarTable evidence"
        )
    scene_scalars.revalidate(selection=selection)
    (
        canonical_rows,
        selected_positions,
        cohort_indices,
    ) = _normalize_safe_scalar_table(scene_scalars.rows)
    cells = _analyze_safe_scalar_cells(
        scalar_table=canonical_rows,
        selected_positions=selected_positions,
        cohort_indices=cohort_indices,
        intervention_configuration_fingerprint=(
            intervention_configuration_fingerprint
        ),
    )
    result = M6WaymaxMatrixResult(
        pair_n=len(selected_positions),
        cohort_indices=cohort_indices,
        intervention_configuration_fingerprint=(
            intervention_configuration_fingerprint
        ),
        scene_scalars=canonical_rows,
        cells=cells,
        _issuance_capability=_MATRIX_RESULT_ISSUER,
    )
    result.revalidate()
    return result


def build_m6_waymax_scene_scalar_table(
    views: Sequence[M6WaymaxTwentyTransitionPairView],
    *,
    selection: M6WaymaxSelection,
) -> M6WaymaxIssuedScalarTable:
    """Recompute selected views and fill the remaining fixed grid with strict NA."""

    if isinstance(views, (str, bytes)) or not isinstance(views, Sequence):
        raise TypeError("views must be a sequence")
    selection_binding_sha256 = _validate_selection(selection)
    normalized = tuple(views)
    if not selection.supported:
        if normalized:
            _fail(
                "selection_unsupported",
                "unsupported selection cannot contain outcome views",
            )
    elif len(normalized) != len(selection.members) * len(M6_WAYMAX_BUNDLES):
        _fail(
            "selection_completeness",
            "supported selection requires both bundles for every exact member",
        )
    if any(
        not isinstance(view, M6WaymaxTwentyTransitionPairView)
        for view in normalized
    ):
        raise TypeError("views must contain exact 20-transition pair views")
    for view in normalized:
        view.revalidate()
    by_position_bundle: dict[
        tuple[int, str],
        M6WaymaxTwentyTransitionPairView,
    ] = {}
    for view in normalized:
        key = (view.selection_position, view.bundle)
        if key in by_position_bundle:
            _fail("duplicate_pair", "duplicate selection-position/bundle view")
        by_position_bundle[key] = view
    selected_positions = tuple(sorted({view.selection_position for view in normalized}))
    if selected_positions != tuple(range(len(selected_positions))):
        _fail("selection_positions", "selected views must form one exact prefix")
    expected_keys = {
        (position, bundle)
        for position in selected_positions
        for bundle in M6_WAYMAX_BUNDLES
    }
    if set(by_position_bundle) != expected_keys:
        _fail("incomplete_pairing", "each selected position requires both bundles")
    expected_positions = (
        tuple(range(len(selection.members))) if selection.supported else ()
    )
    if selected_positions != expected_positions:
        _fail(
            "selection_completeness",
            "outcome positions differ from the complete typed selection",
        )

    scenario_ids: set[str] = set()
    cohort_indices: set[int] = set()
    qualification_hashes: set[str] = set()
    intervention_fingerprints: set[str] = set()
    domain_hashes: set[str] = set()
    selected_rows: dict[tuple[int, str, str], M6WaymaxSceneScalar] = {}
    for position in selected_positions:
        member = selection.members[position]
        left, right = tuple(
            by_position_bundle[(position, bundle)]
            for bundle in M6_WAYMAX_BUNDLES
        )
        shared_fields = (
            "selection_position",
            "cohort_index",
            "scenario_id",
            "target_index",
            "target_agent_id",
            "target_slot",
            "ego_index",
            "ego_agent_id",
            "source_state_sha256",
            "qualification_binding_sha256",
            "primary_domain_sha256",
            "selection_binding_sha256",
            "selection_member_count",
            "baseline_plan_fingerprint",
            "treatment_plan_fingerprint",
            "baseline_configuration_fingerprint",
            "intervention_configuration_fingerprint",
            "baseline_perturbation_identity",
            "treatment_perturbation_identity",
            "target_length_m",
            "ego_length_m",
        )
        if any(getattr(left, name) != getattr(right, name) for name in shared_fields):
            _fail(
                "cross_bundle_pairing",
                "same-position bundles disagree on source/plan/target identity",
            )
        if (
            left.cohort_index != member.cohort_index
            or left.qualification_binding_sha256
            != member.qualification_binding_sha256
            or left.primary_domain_sha256
            != selection.primary_domain_sha256
            or left.selection_binding_sha256
            != selection_binding_sha256
            or left.selection_member_count != len(selection.members)
        ):
            _fail(
                "selection_member",
                "pair views do not match the exact qualification selection ledger",
            )
        exact_cross_bundle_arrays = (
            "timestamps_micros",
            "target_valid",
            "ego_valid",
            "baseline_ego_x",
            "baseline_ego_y",
            "baseline_ego_heading",
            "baseline_ego_vx",
            "baseline_ego_vy",
            "treatment_ego_x",
            "treatment_ego_y",
            "treatment_ego_heading",
            "treatment_ego_vx",
            "treatment_ego_vy",
        )
        if any(
            not np.array_equal(
                np.asarray(getattr(left, name)),
                np.asarray(getattr(right, name)),
            )
            for name in exact_cross_bundle_arrays
        ):
            _fail(
                "cross_bundle_realization",
                "bundles differ in timestamps/validity/dimensions/ego realization",
            )
        if left.scenario_id in scenario_ids:
            _fail("scenario_relabel", "scenario_id reused across selected positions")
        if left.cohort_index in cohort_indices:
            _fail("cohort_pairing", "cohort_index reused across selected positions")
        if left.qualification_binding_sha256 in qualification_hashes:
            _fail(
                "qualification_pairing",
                "qualification binding reused across selected positions",
            )
        scenario_ids.add(left.scenario_id)
        cohort_indices.add(left.cohort_index)
        qualification_hashes.add(left.qualification_binding_sha256)
        intervention_fingerprints.add(
            left.intervention_configuration_fingerprint
        )
        domain_hashes.add(left.primary_domain_sha256)
        for view in (left, right):
            for measure in compute_m6_waymax_paired_measures(view):
                selected_rows[
                    (position, view.bundle, measure.metric_name)
                ] = M6WaymaxSceneScalar(
                    selection_position=position,
                    cohort_index=measure.cohort_index,
                    qualification_binding_sha256=(
                        measure.qualification_binding_sha256
                    ),
                    primary_domain_sha256=selection.primary_domain_sha256,
                    selection_binding_sha256=selection_binding_sha256,
                    selection_supported=True,
                    selection_member_count=len(selection.members),
                    identity_configuration_fingerprint=(
                        _IDENTITY_CONFIGURATION_FINGERPRINT
                    ),
                    primary_b2_configuration_fingerprint=(
                        _PRIMARY_B2_CONFIGURATION_FINGERPRINT
                    ),
                    bundle=view.bundle,
                    metric_name=measure.metric_name,
                    metric_version=measure.metric_version,
                    value_unit=measure.value_unit,
                    value=measure.value,
                    responded=measure.responded,
                    responder_latency_s=measure.responder_latency_s,
                    source_pairing_complete=True,
                    status="selected",
                )
    if selection.supported:
        if intervention_fingerprints != {
            _PRIMARY_B2_CONFIGURATION_FINGERPRINT
        }:
            _fail("intervention_drift", "selected views are not exact primary b=2")
        if domain_hashes != {selection.primary_domain_sha256}:
            _fail(
                "primary_domain_drift",
                "selected views differ from the selection primary domain",
            )

    table: list[M6WaymaxSceneScalar] = []
    for position in range(M6_WAYMAX_MAX_SCENES):
        for bundle in M6_WAYMAX_BUNDLES:
            for metric_name, metric_version, unit in _METRICS:
                selected = selected_rows.get((position, bundle, metric_name))
                if selected is not None:
                    table.append(selected)
                else:
                    table.append(
                        M6WaymaxSceneScalar(
                            selection_position=position,
                            cohort_index=None,
                            qualification_binding_sha256=None,
                            primary_domain_sha256=(
                                selection.primary_domain_sha256
                            ),
                            selection_binding_sha256=(
                                selection_binding_sha256
                            ),
                            selection_supported=selection.supported,
                            selection_member_count=len(selection.members),
                            identity_configuration_fingerprint=(
                                _IDENTITY_CONFIGURATION_FINGERPRINT
                            ),
                            primary_b2_configuration_fingerprint=(
                                _PRIMARY_B2_CONFIGURATION_FINGERPRINT
                            ),
                            bundle=bundle,
                            metric_name=metric_name,
                            metric_version=metric_version,
                            value_unit=unit,
                            value=None,
                            responded=None,
                            responder_latency_s=None,
                            source_pairing_complete=False,
                            status="not_selected",
                        )
                    )
    issued = M6WaymaxIssuedScalarTable(
        rows=tuple(table),
        selection_binding_sha256=selection_binding_sha256,
        primary_domain_sha256=selection.primary_domain_sha256,
        _issuance_capability=_ISSUED_SCALAR_TABLE_ISSUER,
    )
    issued.revalidate(selection=selection)
    return issued


def analyze_m6_waymax_matrix(
    views: Sequence[M6WaymaxTwentyTransitionPairView],
    *,
    selection: M6WaymaxSelection,
) -> M6WaymaxMatrixResult:
    """Convenience path from private validated views to the safe sealed analyzer."""

    normalized = tuple(views)
    table = build_m6_waymax_scene_scalar_table(
        normalized,
        selection=selection,
    )
    return analyze_m6_waymax_cells(
        table,
        selection=selection,
        intervention_configuration_fingerprint=(
            _PRIMARY_B2_CONFIGURATION_FINGERPRINT
        ),
    )


def parse_m6_waymax_scene_scalar_table(
    rows: Sequence[M6WaymaxSceneScalar | Mapping[str, Any]],
) -> M6WaymaxParsedScalarTable:
    """Parse sealed structural rows without granting live-evidence authority."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("stored scalar rows must be a sequence")
    parsed: list[M6WaymaxSceneScalar] = []
    for row in rows:
        if isinstance(row, M6WaymaxSceneScalar):
            scalar = row
        elif isinstance(row, Mapping):
            scalar = M6WaymaxSceneScalar.from_store_dict(row)
        else:
            raise TypeError(
                "stored scalar rows must be mappings or M6WaymaxSceneScalar"
            )
        scalar.revalidate()
        parsed.append(scalar)
    return M6WaymaxParsedScalarTable(
        rows=tuple(parsed),
        _issuance_capability=_PARSED_SCALAR_TABLE_ISSUER,
    )


def verify_m6_waymax_stored_selection(
    parsed_table: M6WaymaxParsedScalarTable,
    *,
    manifest_sha256: str,
    selection_binding_sha256: str,
    primary_domain_sha256: str,
    supported: bool,
    members: Sequence[tuple[int, int, str]],
) -> M6WaymaxVerifiedStoredSelection:
    """Cross-bind parsed rows to selection fields from a verified manifest."""

    if not isinstance(parsed_table, M6WaymaxParsedScalarTable):
        raise TypeError(
            "stored selection verification requires a parsed scalar table"
        )
    parsed_table.revalidate()
    _sha256(manifest_sha256, "manifest_sha256")
    _sha256(selection_binding_sha256, "selection_binding_sha256")
    _sha256(primary_domain_sha256, "primary_domain_sha256")
    rows, selected_positions, _ = _normalize_safe_scalar_table(
        parsed_table.rows
    )
    actual_members: list[tuple[int, int, str]] = []
    for position in selected_positions:
        row = rows[position * M6_WAYMAX_CELL_COUNT]
        actual_members.append(
            (
                position,
                int(row.cohort_index),
                str(row.qualification_binding_sha256),
            )
        )
    supplied_members = tuple(members)
    if (
        tuple(actual_members) != supplied_members
        or any(
            row.selection_binding_sha256 != selection_binding_sha256
            or row.primary_domain_sha256 != primary_domain_sha256
            or row.selection_supported is not supported
            for row in rows
        )
    ):
        _fail(
            "stored_selection_cross_binding",
            "parsed rows differ from manifest-verified selection fields",
        )
    return M6WaymaxVerifiedStoredSelection(
        manifest_sha256=manifest_sha256,
        selection_binding_sha256=selection_binding_sha256,
        primary_domain_sha256=primary_domain_sha256,
        supported=supported,
        members=supplied_members,
        _issuance_capability=_VERIFIED_STORED_SELECTION_ISSUER,
    )


def reconstruct_m6_waymax_stored_cells(
    parsed_table: M6WaymaxParsedScalarTable,
    *,
    selection: M6WaymaxSelection | None = None,
    verified_selection_binding_sha256: str | None = None,
    stored_selection: M6WaymaxVerifiedStoredSelection | None = None,
    intervention_configuration_fingerprint: str,
) -> M6WaymaxStoredReconstruction:
    """Verify post-seal rows while preserving their non-promotable status."""

    if not isinstance(parsed_table, M6WaymaxParsedScalarTable):
        raise TypeError(
            "post-seal reconstruction requires M6WaymaxParsedScalarTable"
        )
    if (
        intervention_configuration_fingerprint
        != _PRIMARY_B2_CONFIGURATION_FINGERPRINT
    ):
        raise ValueError(
            "stored reconstruction requires exact primary b=2 fingerprint"
        )
    parsed_table.revalidate()
    rows, selected_positions, cohort_indices = _normalize_safe_scalar_table(
        parsed_table.rows
    )
    if (selection is None) == (stored_selection is None):
        raise ValueError(
            "provide exactly one live selection or verified stored selection"
        )
    if (
        stored_selection is not None
        and verified_selection_binding_sha256 is not None
    ):
        raise ValueError(
            "verified_selection_binding_sha256 applies only to live selection"
        )
    stored_selection_receipt_sha256: str | None = None
    if selection is not None:
        selection_binding = _validate_selection(selection)
        if verified_selection_binding_sha256 != selection_binding:
            _fail(
                "selection_verification",
                "verified selection binding does not match canonical selection",
            )
        _cross_bind_scalar_rows_to_selection(rows, selection)
    else:
        assert stored_selection is not None
        stored_selection.revalidate()
        selection_binding = stored_selection.selection_binding_sha256
        stored_selection_receipt_sha256 = stored_selection.receipt_sha256
        actual_members = tuple(
            (
                position,
                int(
                    rows[
                        position * M6_WAYMAX_CELL_COUNT
                    ].cohort_index
                ),
                str(
                    rows[
                        position * M6_WAYMAX_CELL_COUNT
                    ].qualification_binding_sha256
                ),
            )
            for position in selected_positions
        )
        if (
            actual_members != stored_selection.members
            or any(
                row.selection_binding_sha256 != selection_binding
                or row.primary_domain_sha256
                != stored_selection.primary_domain_sha256
                or row.selection_supported is not stored_selection.supported
                or row.selection_member_count != len(stored_selection.members)
                for row in rows
            )
        ):
            _fail(
                "stored_selection_cross_binding",
                "stored selection receipt differs from parsed scalar rows",
            )
    cells = _analyze_safe_scalar_cells(
        scalar_table=rows,
        selected_positions=selected_positions,
        cohort_indices=cohort_indices,
        intervention_configuration_fingerprint=(
            intervention_configuration_fingerprint
        ),
    )
    digest = hashlib.sha256()
    digest.update(_STORED_RECONSTRUCTION_DOMAIN)
    digest.update(b"\x00")
    digest.update(bytes.fromhex(parsed_table.parsed_binding_sha256))
    digest.update(bytes.fromhex(selection_binding))
    if stored_selection_receipt_sha256 is not None:
        digest.update(bytes.fromhex(stored_selection_receipt_sha256))
    for cell in cells:
        cell.revalidate()
        digest.update(bytes.fromhex(cell.cell_binding_sha256))
    pair_n = len(selected_positions)
    status: M6WaymaxCellStatus = (
        "unsupported"
        if pair_n < M6_WAYMAX_MIN_SUPPORTED_N
        else (
            "insufficient_n"
            if pair_n < M6_WAYMAX_MIN_DESCRIPTIVE_N
            else "descriptive"
        )
    )
    return M6WaymaxStoredReconstruction(
        pair_n=pair_n,
        status=status,
        selection_binding_sha256=selection_binding,
        reconstruction_sha256=digest.hexdigest(),
        _issuance_capability=_STORED_RECONSTRUCTION_ISSUER,
    )


def m6_waymax_measure_contract() -> Mapping[str, Any]:
    """Return the small immutable registered bridge identity."""

    return MappingProxyType(
        {
            "bundles": M6_WAYMAX_BUNDLES,
            "cell_count": M6_WAYMAX_CELL_COUNT,
            "identity_configuration_fingerprint": (
                _IDENTITY_CONFIGURATION_FINGERPRINT
            ),
            "measure_schema_version": M6_WAYMAX_MEASURE_SCHEMA_VERSION,
            "metrics": _METRICS,
            "minimum_descriptive_n": M6_WAYMAX_MIN_DESCRIPTIVE_N,
            "minimum_supported_n": M6_WAYMAX_MIN_SUPPORTED_N,
            "resamples": M6_WAYMAX_RESAMPLES,
            "policy_access_roles": tuple(
                (
                    bundle,
                    *_BUNDLE_POLICY_ACCESS[bundle],
                )
                for bundle in M6_WAYMAX_BUNDLES
            ),
            "primary_b2_configuration_fingerprint": (
                _PRIMARY_B2_CONFIGURATION_FINGERPRINT
            ),
            "safe_scalar_fields": tuple(
                sorted(M6WaymaxSceneScalar.STORE_FIELDS)
            ),
            "live_scalar_table_type": "M6WaymaxIssuedScalarTable",
            "parsed_scalar_table_promotable": False,
            "determinism_row_count": M6_WAYMAX_DETERMINISM_ROW_COUNT,
            "determinism_conditions": (
                M6_WAYMAX_DETERMINISM_CONDITIONS
            ),
            "live_determinism_issuance_available": True,
            "live_determinism_execution_type": (
                "M6WaymaxLiveDeterminismExecution"
            ),
            "live_determinism_table_type": "M6WaymaxLiveDeterminismTable",
            "no_execution_determinism_table_type": (
                "M6WaymaxNoExecutionDeterminismTable"
            ),
            "no_execution_determinism_promotable": False,
            "verified_stored_selection_type": (
                "M6WaymaxVerifiedStoredSelection"
            ),
            "verified_stored_selection_promotable": False,
            "post_seal_reconstruction_type": (
                "M6WaymaxStoredReconstruction"
            ),
            "selection_required_for_outcomes": True,
            "statistics_schema_version": M6_WAYMAX_STATISTICS_SCHEMA_VERSION,
            "store_scalar_row_count": (
                M6_WAYMAX_MAX_SCENES * M6_WAYMAX_CELL_COUNT
            ),
            "transition_count": M6_WAYMAX_TRANSITIONS,
        }
    )


__all__ = [
    "M6_WAYMAX_BASE_SEED",
    "M6_WAYMAX_CELL_COUNT",
    "M6_WAYMAX_DETERMINISM_CONDITIONS",
    "M6_WAYMAX_DETERMINISM_ROW_COUNT",
    "M6_WAYMAX_MEASURE_SCHEMA_VERSION",
    "M6_WAYMAX_MIN_DESCRIPTIVE_N",
    "M6_WAYMAX_MIN_SUPPORTED_N",
    "M6_WAYMAX_POINTWISE_LEVEL",
    "M6_WAYMAX_RESAMPLES",
    "M6_WAYMAX_STATISTICS_SCHEMA_VERSION",
    "M6WaymaxCellResult",
    "M6WaymaxMatrixResult",
    "M6WaymaxMeasureError",
    "M6WaymaxNoExecutionDeterminismRow",
    "M6WaymaxNoExecutionDeterminismTable",
    "M6WaymaxIssuedScalarTable",
    "M6WaymaxLiveDeterminismExecution",
    "M6WaymaxLiveDeterminismRow",
    "M6WaymaxLiveDeterminismTable",
    "M6WaymaxPairedMeasureResult",
    "M6WaymaxParsedScalarTable",
    "M6WaymaxReweightingBand",
    "M6WaymaxResamplingKey",
    "M6WaymaxSceneScalar",
    "M6WaymaxStoredReconstruction",
    "M6WaymaxTwentyTransitionPairView",
    "M6WaymaxVerifiedStoredSelection",
    "analyze_m6_waymax_cells",
    "analyze_m6_waymax_matrix",
    "build_m6_waymax_scene_scalar_table",
    "build_m6_waymax_data_free_determinism_table",
    "build_m6_waymax_live_determinism_table",
    "build_m6_waymax_unsupported_determinism_table",
    "build_m6_waymax_twenty_transition_pair_view",
    "compute_m6_waymax_paired_measures",
    "m6_waymax_measure_contract",
    "parse_m6_waymax_scene_scalar_table",
    "reconstruct_m6_waymax_stored_cells",
    "validate_m6_waymax_live_determinism_table",
    "validate_m6_waymax_no_execution_determinism_table",
    "verify_m6_waymax_stored_selection",
]
