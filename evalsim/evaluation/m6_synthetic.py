"""Fixed, independently audited data-free acceptance oracles for EvalSim M6.

The ten aligned replicas execute through the public, pinned M6 evaluator.  Two
additional analytic roles execute through an exact default :class:`RolloutEngine`:

* ``no_conflict`` intentionally lies outside primary follower eligibility and retains
  that real rejection in a separate typed analytic scope; and
* ``overreactive_sentinel`` is a test-only history-only responder, never a registered
  production evaluator policy.

Acceptance truth is recomputed directly from immutable raw arrays.  Production M6
metric results are retained where the real primary eligibility contract permits a
``CounterfactualPair`` and are cross-checked against the independent formulas.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import struct
from types import MappingProxyType
from typing import Any

import numpy as np

from evalsim.metrics import m6 as production_m6_metrics
from evalsim.contracts import (
    Agent,
    AgentType,
    CounterfactualPair,
    EgoTrajectoryPlan,
    HistoryOnlyPolicyContext,
    HistoryOnlyPolicyObservation,
    HistoryOnlySimulatorPolicy,
    InterventionEligibility,
    PairedMetricResult,
    PolicyMetadata,
    PolicyStep,
    PrivilegedSimulatorPolicy,
    Scenario,
)
from evalsim.contracts.counterfactual import (
    RolloutSnapshot,
    ScenarioSnapshot,
)
from evalsim.metrics.m6 import (
    m6_primary_paired_metrics,
    m6_secondary_paired_metrics,
)
from evalsim.perturb.m6 import (
    PRIMARY_BRAKE_MAGNITUDE_MPS2,
    compile_identity_plan,
    compile_longitudinal_brake_pulse_plan,
    evaluate_primary_brake_eligibility,
    validate_registered_ego_plan,
)
from evalsim.rollout import (
    DYNAMICS_NAME,
    DYNAMICS_VERSION,
    ROLLOUT_ENGINE_NAME,
    ROLLOUT_ENGINE_VERSION,
    DynamicsLimits,
    PolicyExecutionTrace,
    RolloutEngine,
)
from evalsim.simulators import IDMPolicy

from .m6 import (
    M6_NUMPY_SEED,
    M6EvaluationCase,
    M6EvaluationResult,
    M6PairedSceneResult,
    run_m6_numpy_evaluation,
)

M6_SYNTHETIC_ACCEPTANCE_VERSION = "2.0.0"
M6_SYNTHETIC_ANALYTIC_SCOPE_VERSION = "1.0.0"
M6_SYNTHETIC_CASE_COUNT = 10
M6_SYNTHETIC_CURRENT_INDEX = 3
M6_SYNTHETIC_FRAME_COUNT = 48
M6_SYNTHETIC_DT_S = 0.1
M6_SYNTHETIC_SPEED_MPS = 10.0
M6_SYNTHETIC_FOLLOWER_OFFSET_M = -15.0
M6_SYNTHETIC_NO_CONFLICT_LATERAL_M = 10.0
M6_SYNTHETIC_EARLY_RESPONSE_WINDOW_S = 1.0
M6_SYNTHETIC_MIN_SENTINEL_EXTRA_PROGRESS_LOSS_M = 1.0
M6_SYNTHETIC_MIN_SENTINEL_EXTRA_HARD_BRAKING_S = 0.2

_SYNTHETIC_SOURCE = "synthetic"
_SYNTHETIC_SOURCE_VERSION = "m6-data-free-acceptance-2.0.0"
_SYNTHETIC_TIME_UNIT = "seconds"
_SYNTHETIC_SCENARIO_PREFIX = "m6-data-free-aligned-"
_ALIGNED_GEOMETRY_ID = "aligned_straight_follower_v1"
_NO_CONFLICT_GEOMETRY_ID = "no_conflict_straight_parallel_v1"
_NO_CONFLICT_SCENARIO_ID = "m6-data-free-no-conflict"
_NO_CONFLICT_FIXTURE_ID = "no_conflict_fixed_v1"
_SENTINEL_FIXTURE_ID = "aligned_replica_00_v1"
_NO_CONFLICT_ORACLE = "no_conflict"
_SENTINEL_ORACLE = "overreactive_sentinel"
_SENTINEL_POLICY_NAME = "m6_test_overreactive_sentinel"
_SENTINEL_POLICY_VERSION = "1.0.0"
_SENTINEL_BRAKING_MPS2 = -8.0
_SENTINEL_TRIGGER_SPEED_DEFICIT_MPS = 1e-12
_INDEPENDENT_RESPONSE_ACCELERATION_THRESHOLD_MPS2 = -0.5
_INDEPENDENT_RESPONSE_PERSISTENCE_S = 0.2
_INDEPENDENT_HARD_BRAKING_THRESHOLD_MPS2 = -4.0
_WORLD_STATE_FIELDS = ("valid", "x", "y", "heading", "vx", "vy")
_FLOAT_STATE_FIELDS = ("x", "y", "heading", "vx", "vy")
_PLAN_STATE_FIELDS = ("valid", "x", "y", "heading", "vx", "vy")
_TRACE_FLOAT_FIELDS = (
    "longitudinal_acceleration",
    "yaw_rate",
    "override_x",
    "override_y",
    "override_heading",
    "override_vx",
    "override_vy",
)
_TRACE_BOOL_FIELDS = (
    "override_mask",
    "override_valid",
    "effective_control_mask",
    "lifecycle_birth_mask",
)
_ROLLOUT_METADATA_KEYS = frozenset(
    {
        "agent_control_modes",
        "controlled_agent_ids",
        "dynamics",
        "ego_control",
        "engine",
        "policy",
        "rollout_start_index",
        "scenario_source",
        "scenario_source_fingerprint",
    }
)
_DYNAMICS_KEYS = frozenset(
    {
        "clamp_counts",
        "integration",
        "limits",
        "name",
        "version",
    }
)
_CLAMP_KEYS = frozenset(
    {
        "acceleration",
        "deceleration",
        "reverse_prevented",
        "speed",
        "yaw_rate",
    }
)
_DEFAULT_LIMITS = DynamicsLimits().to_dict()
_EXPECTED_ENGINE_METADATA = {
    "name": ROLLOUT_ENGINE_NAME,
    "version": ROLLOUT_ENGINE_VERSION,
}
_EXPECTED_DYNAMICS_CONFIGURATION = {
    "integration": "midpoint_heading_trapezoidal_speed",
    "limits": _DEFAULT_LIMITS,
    "name": DYNAMICS_NAME,
    "version": DYNAMICS_VERSION,
}
_SOURCE_FINGERPRINT_DOMAIN = b"evalsim-m6-synthetic-source-v2"


class M6SyntheticAcceptanceError(RuntimeError):
    """One fixed data-free M6 acceptance gate failed closed."""


def _assert_live_production_semantics() -> None:
    """Pin both independent and production semantics to preregistered literals."""

    if (
        _INDEPENDENT_RESPONSE_ACCELERATION_THRESHOLD_MPS2 != -0.5
        or _INDEPENDENT_RESPONSE_PERSISTENCE_S != 0.2
        or _INDEPENDENT_HARD_BRAKING_THRESHOLD_MPS2 != -4.0
        or production_m6_metrics.M6_RESPONSE_ACCELERATION_THRESHOLD_MPS2
        != -0.5
        or production_m6_metrics.M6_RESPONSE_PERSISTENCE_S != 0.2
        or production_m6_metrics.M6_HARD_BRAKING_THRESHOLD_MPS2 != -4.0
    ):
        raise M6SyntheticAcceptanceError(
            "live production or independent metric semantics drifted from "
            "the exact M6 preregistration literals"
        )


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _plain_json(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def _hash_parts(domain: bytes, parts: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(b"\x00")
    for part in parts:
        digest.update(_u64(len(part)))
        digest.update(part)
    return digest.hexdigest()


def _source_fingerprint(source: ScenarioSnapshot) -> str:
    """Hash the exact contract-native source without production helper reuse."""

    source.revalidate()
    parts: list[bytes] = [
        source.scenario_id.encode("utf-8"),
        np.asarray(source.timestamps, dtype="<f8").tobytes(order="C"),
        _u64(source.ego_index),
        _canonical_json(source.metadata),
    ]
    for agent in source.agents:
        parts.extend(
            (
                _canonical_json(
                    {
                        "id": int(agent.id),
                        "length": float(agent.length),
                        "type": agent.type.value,
                        "width": float(agent.width),
                    }
                ),
                np.asarray(agent.valid, dtype=np.uint8).tobytes(order="C"),
                *(
                    np.asarray(
                        getattr(agent, name),
                        dtype="<f8",
                    ).tobytes(order="C")
                    for name in _FLOAT_STATE_FIELDS
                ),
            )
        )
    for feature in source.map:
        parts.extend(
            (
                feature.type.value.encode("ascii"),
                np.asarray(feature.xy, dtype="<f8").tobytes(order="C"),
            )
        )
    return _hash_parts(_SOURCE_FINGERPRINT_DOMAIN, parts)


def _agent(
    agent_id: int,
    *,
    x: np.ndarray,
    y: np.ndarray,
    speed_mps: float,
) -> Agent:
    count = len(x)
    return Agent(
        id=agent_id,
        type=AgentType.VEHICLE,
        valid=np.ones(count, dtype=bool),
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        heading=np.zeros(count, dtype=np.float64),
        vx=np.full(count, speed_mps, dtype=np.float64),
        vy=np.zeros(count, dtype=np.float64),
        length=4.5,
        width=2.0,
    )


def _straight_scenario(
    scenario_id: str,
    *,
    geometry_id: str,
    follower_lateral_m: float,
    replica_index: int,
    replica_count: int,
) -> Scenario:
    timestamps = (
        np.arange(M6_SYNTHETIC_FRAME_COUNT, dtype=np.float64)
        * M6_SYNTHETIC_DT_S
    )
    ego_x = timestamps * M6_SYNTHETIC_SPEED_MPS
    return Scenario(
        scenario_id=scenario_id,
        timestamps=timestamps,
        agents=[
            _agent(
                100,
                x=ego_x,
                y=np.zeros(M6_SYNTHETIC_FRAME_COUNT, dtype=np.float64),
                speed_mps=M6_SYNTHETIC_SPEED_MPS,
            ),
            _agent(
                200,
                x=ego_x + M6_SYNTHETIC_FOLLOWER_OFFSET_M,
                y=np.full(
                    M6_SYNTHETIC_FRAME_COUNT,
                    follower_lateral_m,
                    dtype=np.float64,
                ),
                speed_mps=M6_SYNTHETIC_SPEED_MPS,
            ),
        ],
        ego_index=0,
        metadata={
            "analytic_geometry_id": geometry_id,
            "analytic_replica_count": replica_count,
            "analytic_replica_index": replica_index,
            "analytic_replica_role": "deterministic_replica",
            "analytic_unique_geometry_count": 1,
            "current_index": M6_SYNTHETIC_CURRENT_INDEX,
            "source": _SYNTHETIC_SOURCE,
            "source_time_unit": _SYNTHETIC_TIME_UNIT,
            "source_version": _SYNTHETIC_SOURCE_VERSION,
        },
    )


def _aligned_scenario(replica_index: int) -> Scenario:
    if (
        isinstance(replica_index, (bool, np.bool_))
        or not isinstance(replica_index, (int, np.integer))
        or not 0 <= int(replica_index) < M6_SYNTHETIC_CASE_COUNT
    ):
        raise ValueError("replica_index must identify one of the ten replicas")
    index = int(replica_index)
    return _straight_scenario(
        f"{_SYNTHETIC_SCENARIO_PREFIX}{index:02d}",
        geometry_id=_ALIGNED_GEOMETRY_ID,
        follower_lateral_m=0.0,
        replica_index=index,
        replica_count=M6_SYNTHETIC_CASE_COUNT,
    )


def _no_conflict_scenario() -> Scenario:
    return _straight_scenario(
        _NO_CONFLICT_SCENARIO_ID,
        geometry_id=_NO_CONFLICT_GEOMETRY_ID,
        follower_lateral_m=M6_SYNTHETIC_NO_CONFLICT_LATERAL_M,
        replica_index=0,
        replica_count=1,
    )


def synthetic_m6_cases() -> tuple[M6EvaluationCase, ...]:
    """Return ten IDs over one deliberately replicated analytic geometry."""

    return tuple(
        M6EvaluationCase(
            cohort_index=index,
            scenario=_aligned_scenario(index),
        )
        for index in range(M6_SYNTHETIC_CASE_COUNT)
    )


def synthetic_m6_source_evidence() -> Mapping[str, Any]:
    """Return compact local metadata that does not imply ten unique geometries."""

    return MappingProxyType(
        {
            "acceptance_version": M6_SYNTHETIC_ACCEPTANCE_VERSION,
            "case_count": M6_SYNTHETIC_CASE_COUNT,
            "current_index": M6_SYNTHETIC_CURRENT_INDEX,
            "deterministic_replica_count": M6_SYNTHETIC_CASE_COUNT,
            "dt_seconds": M6_SYNTHETIC_DT_S,
            "frame_count": M6_SYNTHETIC_FRAME_COUNT,
            "geometry_id": _ALIGNED_GEOMETRY_ID,
            "ordered_scenario_ids": tuple(
                case.scenario.scenario_id for case in synthetic_m6_cases()
            ),
            "replica_semantics": (
                "ten deterministic IDs over one unique analytic geometry"
            ),
            "seed": M6_NUMPY_SEED,
            "source": _SYNTHETIC_SOURCE,
            "source_version": _SYNTHETIC_SOURCE_VERSION,
            "unique_analytic_geometry_count": 1,
        }
    )


def _snapshot_source(
    value: Scenario | ScenarioSnapshot,
) -> ScenarioSnapshot:
    if isinstance(value, Scenario):
        return ScenarioSnapshot.from_scenario(value)
    if isinstance(value, ScenarioSnapshot):
        value.revalidate()
        return ScenarioSnapshot.from_scenario(value.to_scenario())
    raise TypeError("source_snapshot must be a Scenario or ScenarioSnapshot")


def _expected_scope_source(fixture_id: str) -> Scenario:
    if fixture_id == _NO_CONFLICT_FIXTURE_ID:
        return _no_conflict_scenario()
    if fixture_id == _SENTINEL_FIXTURE_ID:
        return _aligned_scenario(0)
    raise ValueError("fixture_id is not a registered analytic fixture")


@dataclass(frozen=True, slots=True)
class M6AnalyticOracleScope:
    """Immutable source scope distinct from primary eligibility semantics."""

    fixture_id: str
    scenario_id: str
    source_snapshot: Scenario | ScenarioSnapshot = field(
        repr=False,
        compare=False,
    )
    source_fingerprint: str | None
    primary_eligibility: InterventionEligibility
    current_index: int
    stop_index: int
    target_index: int
    schema_version: str = M6_SYNTHETIC_ANALYTIC_SCOPE_VERSION

    def __post_init__(self) -> None:
        snapshot = _snapshot_source(self.source_snapshot)
        object.__setattr__(self, "source_snapshot", snapshot)
        computed = _source_fingerprint(snapshot)
        if self.source_fingerprint is None:
            object.__setattr__(self, "source_fingerprint", computed)
        eligibility = InterventionEligibility.from_dict(
            self.primary_eligibility.to_dict()
        )
        object.__setattr__(self, "primary_eligibility", eligibility)
        self.revalidate()

    @classmethod
    def from_source(
        cls,
        fixture_id: str,
        source: Scenario,
        *,
        target_index: int,
    ) -> "M6AnalyticOracleScope":
        if not isinstance(source, Scenario):
            raise TypeError("source must be a Scenario")
        current = source.metadata.get("current_index")
        if (
            isinstance(current, (bool, np.bool_))
            or not isinstance(current, (int, np.integer))
        ):
            raise ValueError("source current_index must be an integer")
        current_index = int(current)
        return cls(
            fixture_id=fixture_id,
            scenario_id=source.scenario_id,
            source_snapshot=source,
            source_fingerprint=None,
            primary_eligibility=evaluate_primary_brake_eligibility(source),
            current_index=current_index,
            stop_index=current_index + 40,
            target_index=target_index,
        )

    def revalidate(self) -> None:
        _assert_live_production_semantics()
        if self.schema_version != M6_SYNTHETIC_ANALYTIC_SCOPE_VERSION:
            raise ValueError("analytic scope schema_version drifted")
        if self.fixture_id not in {
            _NO_CONFLICT_FIXTURE_ID,
            _SENTINEL_FIXTURE_ID,
        }:
            raise ValueError("analytic scope fixture_id drifted")
        if not isinstance(self.scenario_id, str) or not self.scenario_id:
            raise ValueError("analytic scope scenario_id must be non-empty")
        if not isinstance(self.source_snapshot, ScenarioSnapshot):
            raise TypeError("analytic scope must retain a ScenarioSnapshot")
        self.source_snapshot.revalidate()
        computed = _source_fingerprint(self.source_snapshot)
        if self.source_fingerprint != computed:
            raise ValueError("analytic scope source fingerprint drifted")
        expected = ScenarioSnapshot.from_scenario(
            _expected_scope_source(self.fixture_id)
        )
        if (
            self.scenario_id != self.source_snapshot.scenario_id
            or self.scenario_id != expected.scenario_id
            or computed != _source_fingerprint(expected)
        ):
            raise ValueError(
                "analytic scope source identity or fixed geometry drifted"
            )
        for name, value in (
            ("current_index", self.current_index),
            ("stop_index", self.stop_index),
            ("target_index", self.target_index),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
            ):
                raise ValueError(f"analytic scope {name} must be nonnegative")
        source_current = self.source_snapshot.metadata.get("current_index")
        if (
            self.current_index != source_current
            or self.stop_index != self.current_index + 40
            or self.stop_index >= self.source_snapshot.num_steps
            or self.target_index >= self.source_snapshot.num_agents
            or self.target_index == self.source_snapshot.ego_index
        ):
            raise ValueError("analytic scope target or 40-transition window drifted")
        target = self.source_snapshot.agents[self.target_index]
        if (
            target.type != AgentType.VEHICLE
            or not bool(
                np.all(
                    target.valid[self.current_index : self.stop_index + 1]
                )
            )
        ):
            raise ValueError(
                "analytic scope target must be a stable world vehicle"
            )
        if not isinstance(self.primary_eligibility, InterventionEligibility):
            raise TypeError(
                "analytic scope must retain typed primary eligibility"
            )
        recomputed = evaluate_primary_brake_eligibility(
            self.source_snapshot.to_scenario()
        )
        if recomputed.to_dict() != self.primary_eligibility.to_dict():
            raise ValueError(
                "analytic scope primary eligibility drifted from source"
            )
        if self.fixture_id == _NO_CONFLICT_FIXTURE_ID:
            if (
                recomputed.eligible
                or recomputed.reason != "no_stable_aligned_follower"
                or recomputed.target_index is not None
                or self.target_index != 1
            ):
                raise ValueError(
                    "no-conflict scope must retain its genuine primary rejection"
                )
        elif (
            not recomputed.eligible
            or recomputed.target_index != self.target_index
            or self.target_index != 1
        ):
            raise ValueError(
                "sentinel scope must retain real aligned primary eligibility"
            )


@dataclass(frozen=True, slots=True)
class _SentinelState:
    agent_count: int
    initial_ego_speed_mps: float
    nominal_state: Any = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _OverreactiveSentinelPolicy(HistoryOnlySimulatorPolicy):
    """Test-only responder that adds hard braking after realized ego slowdown."""

    nominal_policy: IDMPolicy = field(default_factory=IDMPolicy)

    def __post_init__(self) -> None:
        if type(self.nominal_policy) is not IDMPolicy or (
            self.nominal_policy.metadata().to_dict()
            != IDMPolicy().metadata().to_dict()
        ):
            raise TypeError("sentinel requires the exact default IDMPolicy")

    def initialize(
        self,
        context: HistoryOnlyPolicyContext,
        seed: int,
    ) -> _SentinelState:
        if not isinstance(context, HistoryOnlyPolicyContext):
            raise TypeError("sentinel requires HistoryOnlyPolicyContext")
        current = context.frames[-1]
        initial_ego_speed = float(
            math.hypot(
                float(current.vx[context.ego_index]),
                float(current.vy[context.ego_index]),
            )
        )
        return _SentinelState(
            agent_count=len(context.agent_ids),
            initial_ego_speed_mps=initial_ego_speed,
            nominal_state=self.nominal_policy.initialize(context, seed),
        )

    def step(
        self,
        state: _SentinelState,
        observation: HistoryOnlyPolicyObservation,
    ) -> PolicyStep:
        if not isinstance(state, _SentinelState):
            raise TypeError("sentinel received incompatible policy state")
        if state.agent_count != observation.frame.num_agents:
            raise ValueError("sentinel state does not match observation")
        nominal = self.nominal_policy.step(
            state.nominal_state,
            observation,
        )
        acceleration = np.array(
            nominal.longitudinal_acceleration,
            dtype=np.float64,
            copy=True,
        )
        ego_speed = float(
            math.hypot(
                float(observation.frame.vx[observation.ego_index]),
                float(observation.frame.vy[observation.ego_index]),
            )
        )
        if (
            ego_speed
            < state.initial_ego_speed_mps
            - _SENTINEL_TRIGGER_SPEED_DEFICIT_MPS
        ):
            for index, agent_type in enumerate(observation.agent_types):
                if (
                    index != observation.ego_index
                    and observation.frame.valid[index]
                    and agent_type == AgentType.VEHICLE
                ):
                    acceleration[index] = min(
                        float(acceleration[index]),
                        _SENTINEL_BRAKING_MPS2,
                    )
        return PolicyStep(
            next_state=_SentinelState(
                agent_count=state.agent_count,
                initial_ego_speed_mps=state.initial_ego_speed_mps,
                nominal_state=nominal.next_state,
            ),
            longitudinal_acceleration=acceleration,
            yaw_rate=nominal.yaw_rate,
        )

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name=_SENTINEL_POLICY_NAME,
            version=_SENTINEL_POLICY_VERSION,
            deterministic=True,
            required_features=(
                "current_agent_states",
                "agent_dimensions",
                "agent_types",
                "realized_ego_speed",
            ),
            supported_agent_types=(AgentType.VEHICLE,),
            params={
                "nominal_policy": self.nominal_policy.metadata().to_dict(),
                "sentinel_braking_mps2": _SENTINEL_BRAKING_MPS2,
                "trigger_speed_deficit_mps": (
                    _SENTINEL_TRIGGER_SPEED_DEFICIT_MPS
                ),
            },
            known_limitations=(
                "Test-only exaggerated responder; not an M6 simulator role.",
                "Acts only after a realized ego speed deficit is observable.",
            ),
            fallback_policy="constant_velocity",
        )


@dataclass(frozen=True, slots=True)
class M6IndependentMeasures:
    """Raw-array measures used as acceptance truth, independent of M6 metrics."""

    additional_target_braking_impulse_mps: float
    response_timeliness_s: float
    response_responded: bool
    response_censored: bool
    response_event_time_s: float | None
    response_restricted_latency_s: float
    response_window_s: float
    response_start_transition: int | None
    response_end_transition: int | None
    minimum_longitudinal_bumper_gap_change_m: float
    baseline_minimum_longitudinal_bumper_gap_m: float
    intervention_minimum_longitudinal_bumper_gap_m: float
    target_progress_loss_m: float
    baseline_target_progress_m: float
    intervention_target_progress_m: float
    target_world_displacement_mean_m: float
    target_speed_reduction_max_mps: float
    additional_absolute_jerk_integral_mps2: float
    jerk_derivative_interval_count: int
    additional_hard_braking_exposure_s: float
    baseline_hard_braking_exposure_s: float
    intervention_hard_braking_exposure_s: float
    world_tensor_equal: bool
    structurally_nonreactive: bool
    first_world_divergence_frame: int | None

    def __post_init__(self) -> None:
        numeric = (
            "additional_target_braking_impulse_mps",
            "response_timeliness_s",
            "response_restricted_latency_s",
            "response_window_s",
            "minimum_longitudinal_bumper_gap_change_m",
            "baseline_minimum_longitudinal_bumper_gap_m",
            "intervention_minimum_longitudinal_bumper_gap_m",
            "target_progress_loss_m",
            "baseline_target_progress_m",
            "intervention_target_progress_m",
            "target_world_displacement_mean_m",
            "target_speed_reduction_max_mps",
            "additional_absolute_jerk_integral_mps2",
            "additional_hard_braking_exposure_s",
            "baseline_hard_braking_exposure_s",
            "intervention_hard_braking_exposure_s",
        )
        for name in numeric:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"independent measure {name} must be finite")
            object.__setattr__(self, name, value)
        if self.response_event_time_s is not None:
            event = float(self.response_event_time_s)
            if not math.isfinite(event):
                raise ValueError("response_event_time_s must be finite or None")
            object.__setattr__(self, "response_event_time_s", event)
        for name in (
            "response_responded",
            "response_censored",
            "world_tensor_equal",
            "structurally_nonreactive",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if self.response_censored == self.response_responded:
            raise ValueError("response censor and responder flags contradict")
        for name in (
            "response_start_transition",
            "response_end_transition",
            "first_world_divergence_frame",
            "jerk_derivative_interval_count",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
            ):
                raise ValueError(f"{name} must be nonnegative or None")
            if value is not None:
                object.__setattr__(self, name, int(value))
        if self.jerk_derivative_interval_count is None:
            raise ValueError(
                "jerk_derivative_interval_count must be nonnegative"
            )


def _target_accelerations(
    baseline: RolloutSnapshot,
    intervention: RolloutSnapshot,
    *,
    target_index: int,
    current: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.asarray(
        baseline.timestamps[current : stop + 1],
        dtype=np.float64,
    )
    dt = np.diff(timestamps)
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0.0):
        raise M6SyntheticAcceptanceError(
            "independent oracle requires positive finite timestamps"
        )
    baseline_target = baseline.agents[target_index]
    intervention_target = intervention.agents[target_index]
    baseline_speed = np.hypot(
        baseline_target.vx[current : stop + 1],
        baseline_target.vy[current : stop + 1],
    )
    intervention_speed = np.hypot(
        intervention_target.vx[current : stop + 1],
        intervention_target.vy[current : stop + 1],
    )
    baseline_acceleration = np.diff(baseline_speed) / dt
    intervention_acceleration = np.diff(intervention_speed) / dt
    if not np.all(np.isfinite(baseline_acceleration)) or not np.all(
        np.isfinite(intervention_acceleration)
    ):
        raise M6SyntheticAcceptanceError(
            "independent oracle acceleration is non-finite"
        )
    return baseline_acceleration, intervention_acceleration, dt


def _raw_world_equality(
    source: ScenarioSnapshot,
    baseline: RolloutSnapshot,
    intervention: RolloutSnapshot,
    *,
    current: int,
    stop: int,
) -> tuple[bool, int | None]:
    equal = np.array_equal(baseline.timestamps, intervention.timestamps)
    first_divergence: int | None = None
    for index, (left, right) in enumerate(
        zip(baseline.agents, intervention.agents, strict=True)
    ):
        if index == source.ego_index:
            continue
        contract_equal = (
            left.id == right.id
            and left.type == right.type
            and left.length == right.length
            and left.width == right.width
        )
        equal = bool(equal and contract_equal)
        for frame in range(current + 1, stop + 1):
            frame_equal = contract_equal and all(
                np.array_equal(
                    getattr(left, name)[frame : frame + 1],
                    getattr(right, name)[frame : frame + 1],
                )
                for name in _WORLD_STATE_FIELDS
            )
            if not frame_equal:
                equal = False
                if first_divergence is None:
                    first_divergence = frame
        for name in _WORLD_STATE_FIELDS:
            if not np.array_equal(
                getattr(left, name)[current + 1 : stop + 1],
                getattr(right, name)[current + 1 : stop + 1],
            ):
                equal = False
    return equal, first_divergence


def _independent_measures(
    source: ScenarioSnapshot,
    baseline: RolloutSnapshot,
    intervention: RolloutSnapshot,
    *,
    current: int,
    stop: int,
    target_index: int,
) -> M6IndependentMeasures:
    """Compute every accepted paired scalar directly from immutable arrays."""

    _assert_live_production_semantics()
    source.revalidate()
    baseline.revalidate()
    intervention.revalidate()
    baseline_acc, intervention_acc, dt = _target_accelerations(
        baseline,
        intervention,
        target_index=target_index,
        current=current,
        stop=stop,
    )
    impulse = math.fsum(
        float(max(0.0, left - right) * duration)
        for left, right, duration in zip(
            baseline_acc,
            intervention_acc,
            dt,
            strict=True,
        )
    )

    delta = intervention_acc - baseline_acc
    response_start: int | None = None
    response_end: int | None = None
    run_start: int | None = None
    for transition in range(1, len(delta)):
        if (
            float(delta[transition])
            <= _INDEPENDENT_RESPONSE_ACCELERATION_THRESHOLD_MPS2
        ):
            if run_start is None:
                run_start = transition
            duration = math.fsum(
                float(value)
                for value in dt[run_start : transition + 1]
            )
            if duration >= _INDEPENDENT_RESPONSE_PERSISTENCE_S:
                response_start = run_start
                response_end = transition
                break
        else:
            run_start = None
    timestamps = baseline.timestamps[current : stop + 1]
    window_s = float(timestamps[-1] - timestamps[0])
    responded = response_end is not None
    if responded:
        event_time: float | None = float(
            timestamps[response_end + 1] - timestamps[0]
        )
        restricted_latency = min(event_time, window_s)
    else:
        event_time = None
        restricted_latency = window_s
    timeliness = window_s - restricted_latency

    source_target = source.agents[target_index]
    vx = float(source_target.vx[current])
    vy = float(source_target.vy[current])
    speed = math.hypot(vx, vy)
    if speed > 1e-12:
        gap_hx, gap_hy = vx / speed, vy / speed
    else:
        gap_hx = math.cos(float(source_target.heading[current]))
        gap_hy = math.sin(float(source_target.heading[current]))
    baseline_target = baseline.agents[target_index]
    intervention_target = intervention.agents[target_index]
    baseline_ego = baseline.agents[source.ego_index]
    intervention_ego = intervention.agents[source.ego_index]
    half_length = 0.5 * (
        float(source.ego.length) + float(source_target.length)
    )
    baseline_gaps = tuple(
        (
            (float(baseline_ego.x[frame]) - float(baseline_target.x[frame]))
            * gap_hx
            + (
                float(baseline_ego.y[frame])
                - float(baseline_target.y[frame])
            )
            * gap_hy
            - half_length
        )
        for frame in range(current + 1, stop + 1)
    )
    intervention_gaps = tuple(
        (
            (
                float(intervention_ego.x[frame])
                - float(intervention_target.x[frame])
            )
            * gap_hx
            + (
                float(intervention_ego.y[frame])
                - float(intervention_target.y[frame])
            )
            * gap_hy
            - half_length
        )
        for frame in range(current + 1, stop + 1)
    )
    baseline_minimum_gap = min(baseline_gaps)
    intervention_minimum_gap = min(intervention_gaps)
    gap_change = intervention_minimum_gap - baseline_minimum_gap

    heading = float(source_target.heading[current])
    progress_hx, progress_hy = math.cos(heading), math.sin(heading)
    baseline_progress = (
        (
            float(baseline_target.x[stop])
            - float(source_target.x[current])
        )
        * progress_hx
        + (
            float(baseline_target.y[stop])
            - float(source_target.y[current])
        )
        * progress_hy
    )
    intervention_progress = (
        (
            float(intervention_target.x[stop])
            - float(source_target.x[current])
        )
        * progress_hx
        + (
            float(intervention_target.y[stop])
            - float(source_target.y[current])
        )
        * progress_hy
    )
    progress_loss = baseline_progress - intervention_progress

    future = slice(current + 1, stop + 1)
    distances = np.hypot(
        intervention_target.x[future] - baseline_target.x[future],
        intervention_target.y[future] - baseline_target.y[future],
    )
    displacement_mean = math.fsum(
        float(value) for value in distances
    ) / len(distances)
    baseline_speed = np.hypot(
        baseline_target.vx[future],
        baseline_target.vy[future],
    )
    intervention_speed = np.hypot(
        intervention_target.vx[future],
        intervention_target.vy[future],
    )
    speed_reduction = float(np.max(baseline_speed - intervention_speed))

    midpoint_dt = (dt[:-1] + dt[1:]) / 2.0
    baseline_jerk = np.diff(baseline_acc) / midpoint_dt
    intervention_jerk = np.diff(intervention_acc) / midpoint_dt
    jerk_integral = math.fsum(
        float(max(0.0, abs(right) - abs(left)) * duration)
        for left, right, duration in zip(
            baseline_jerk,
            intervention_jerk,
            midpoint_dt,
            strict=True,
        )
    )
    baseline_hard = math.fsum(
        float(duration)
        for acceleration, duration in zip(
            baseline_acc,
            dt,
            strict=True,
        )
        if (
            float(acceleration)
            <= _INDEPENDENT_HARD_BRAKING_THRESHOLD_MPS2
        )
    )
    intervention_hard = math.fsum(
        float(duration)
        for acceleration, duration in zip(
            intervention_acc,
            dt,
            strict=True,
        )
        if (
            float(acceleration)
            <= _INDEPENDENT_HARD_BRAKING_THRESHOLD_MPS2
        )
    )
    hard_exposure = intervention_hard - baseline_hard
    world_equal, first_divergence = _raw_world_equality(
        source,
        baseline,
        intervention,
        current=current,
        stop=stop,
    )
    nonreactive = bool(
        world_equal
        and impulse == 0.0
        and timeliness == 0.0
        and progress_loss == 0.0
    )
    return M6IndependentMeasures(
        additional_target_braking_impulse_mps=impulse,
        response_timeliness_s=timeliness,
        response_responded=responded,
        response_censored=not responded,
        response_event_time_s=event_time,
        response_restricted_latency_s=restricted_latency,
        response_window_s=window_s,
        response_start_transition=response_start,
        response_end_transition=response_end,
        minimum_longitudinal_bumper_gap_change_m=gap_change,
        baseline_minimum_longitudinal_bumper_gap_m=baseline_minimum_gap,
        intervention_minimum_longitudinal_bumper_gap_m=(
            intervention_minimum_gap
        ),
        target_progress_loss_m=progress_loss,
        baseline_target_progress_m=baseline_progress,
        intervention_target_progress_m=intervention_progress,
        target_world_displacement_mean_m=displacement_mean,
        target_speed_reduction_max_mps=speed_reduction,
        additional_absolute_jerk_integral_mps2=jerk_integral,
        jerk_derivative_interval_count=len(midpoint_dt),
        additional_hard_braking_exposure_s=hard_exposure,
        baseline_hard_braking_exposure_s=baseline_hard,
        intervention_hard_braking_exposure_s=intervention_hard,
        world_tensor_equal=world_equal,
        structurally_nonreactive=nonreactive,
        first_world_divergence_frame=first_divergence,
    )


_RAW_METRIC_FIELDS = MappingProxyType(
    {
        "additional_absolute_jerk_integral_mps2": (
            "additional_absolute_jerk_integral_mps2"
        ),
        "additional_hard_braking_exposure_s": (
            "additional_hard_braking_exposure_s"
        ),
        "additional_target_braking_impulse_mps": (
            "additional_target_braking_impulse_mps"
        ),
        "minimum_longitudinal_bumper_gap_change_m": (
            "minimum_longitudinal_bumper_gap_change_m"
        ),
        "response_timeliness_s": "response_timeliness_s",
        "target_progress_loss_m": "target_progress_loss_m",
        "target_speed_reduction_max_mps": "target_speed_reduction_max_mps",
        "target_world_displacement_mean_m": (
            "target_world_displacement_mean_m"
        ),
    }
)


def _metric_results(
    pair: CounterfactualPair,
) -> tuple[PairedMetricResult, ...]:
    return tuple(
        metric.compute(pair)
        for metric in (
            *m6_primary_paired_metrics(),
            *m6_secondary_paired_metrics(),
        )
    )


def _metric_results_equal(
    left: tuple[PairedMetricResult, ...],
    right: tuple[PairedMetricResult, ...],
) -> bool:
    return tuple(item.to_dict() for item in left) == tuple(
        item.to_dict() for item in right
    )


def _crosscheck_metrics(
    measures: M6IndependentMeasures,
    results: tuple[PairedMetricResult, ...],
    *,
    target_index: int,
    transition_count: int,
) -> None:
    if len(results) != len(_RAW_METRIC_FIELDS):
        raise M6SyntheticAcceptanceError(
            "production metric family is incomplete"
        )
    observed = {result.metric_name: result for result in results}
    if set(observed) != set(_RAW_METRIC_FIELDS):
        raise M6SyntheticAcceptanceError(
            "production metric identities differ from raw oracle registry"
        )
    for metric_name, field_name in _RAW_METRIC_FIELDS.items():
        if observed[metric_name].value != getattr(measures, field_name):
            raise M6SyntheticAcceptanceError(
                f"production metric {metric_name} disagrees with raw formula"
            )
    base = {
        "target_index": target_index,
        "transition_count": transition_count,
    }
    expected_details = {
        "additional_absolute_jerk_integral_mps2": {
            **base,
            "derivative_interval_count": (
                measures.jerk_derivative_interval_count
            ),
        },
        "additional_hard_braking_exposure_s": {
            **base,
            "baseline_exposure_s": (
                measures.baseline_hard_braking_exposure_s
            ),
            "intervention_exposure_s": (
                measures.intervention_hard_braking_exposure_s
            ),
            "inclusive_acceleration_threshold_mps2": (
                _INDEPENDENT_HARD_BRAKING_THRESHOLD_MPS2
            ),
        },
        "additional_target_braking_impulse_mps": dict(base),
        "minimum_longitudinal_bumper_gap_change_m": {
            **base,
            "baseline_minimum_m": (
                measures.baseline_minimum_longitudinal_bumper_gap_m
            ),
            "intervention_minimum_m": (
                measures.intervention_minimum_longitudinal_bumper_gap_m
            ),
        },
        "response_timeliness_s": {
            **base,
            "acceleration_threshold_mps2": (
                _INDEPENDENT_RESPONSE_ACCELERATION_THRESHOLD_MPS2
            ),
            "censored": measures.response_censored,
            "event_time_s": measures.response_event_time_s,
            "persistence_s": _INDEPENDENT_RESPONSE_PERSISTENCE_S,
            "responded": measures.response_responded,
            "response_end_transition": measures.response_end_transition,
            "response_start_transition": (
                measures.response_start_transition
            ),
            "restricted_latency_s": (
                measures.response_restricted_latency_s
            ),
            "search_start_transition": 1,
            "window_s": measures.response_window_s,
        },
        "target_progress_loss_m": {
            **base,
            "baseline_progress_m": measures.baseline_target_progress_m,
            "intervention_progress_m": (
                measures.intervention_target_progress_m
            ),
        },
        "target_speed_reduction_max_mps": dict(base),
        "target_world_displacement_mean_m": dict(base),
    }
    for metric_name, expected in expected_details.items():
        if _plain_json(observed[metric_name].details) != expected:
            raise M6SyntheticAcceptanceError(
                f"production metric {metric_name} semantic details "
                "disagree with independent preregistration"
            )


def _type_identity(value: type[Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _expected_policy_for_oracle(
    oracle_name: str,
) -> tuple[type[Any], HistoryOnlySimulatorPolicy]:
    if oracle_name == _NO_CONFLICT_ORACLE:
        return IDMPolicy, IDMPolicy()
    if oracle_name == _SENTINEL_ORACLE:
        return _OverreactiveSentinelPolicy, _OverreactiveSentinelPolicy()
    raise ValueError("oracle_name is not registered")


def _validate_exact_engine(engine: RolloutEngine) -> None:
    if (
        type(engine) is not RolloutEngine
        or engine.name != ROLLOUT_ENGINE_NAME
        or engine.version != ROLLOUT_ENGINE_VERSION
        or engine.dynamics_limits.to_dict() != _DEFAULT_LIMITS
    ):
        raise TypeError(
            "analytic oracle requires the exact default RolloutEngine"
        )


def _validate_rollout_metadata(
    rollout: RolloutSnapshot,
    scope: M6AnalyticOracleScope,
    *,
    policy_metadata: Mapping[str, Any],
    perturbation_identity: str,
) -> None:
    metadata = rollout.metadata
    if set(metadata) != _ROLLOUT_METADATA_KEYS:
        raise ValueError("analytic rollout metadata fields drifted")
    if _plain_json(metadata["engine"]) != _EXPECTED_ENGINE_METADATA:
        raise ValueError("analytic rollout engine metadata drifted")
    dynamics = metadata["dynamics"]
    if not isinstance(dynamics, Mapping) or set(dynamics) != _DYNAMICS_KEYS:
        raise ValueError("analytic rollout dynamics metadata fields drifted")
    configuration = {
        key: _plain_json(value)
        for key, value in dynamics.items()
        if key != "clamp_counts"
    }
    if configuration != _EXPECTED_DYNAMICS_CONFIGURATION:
        raise ValueError("analytic rollout dynamics configuration drifted")
    clamps = dynamics["clamp_counts"]
    if (
        not isinstance(clamps, Mapping)
        or set(clamps) != _CLAMP_KEYS
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            for value in clamps.values()
        )
    ):
        raise ValueError("analytic rollout clamp counts are invalid")
    if _plain_json(metadata["policy"]) != _plain_json(policy_metadata):
        raise ValueError("analytic rollout full policy metadata drifted")
    source = scope.source_snapshot
    expected_ids = [
        int(agent.id)
        for index, agent in enumerate(source.agents)
        if index != source.ego_index
    ]
    if (
        metadata["ego_control"] != "typed_ego_plan"
        or metadata["rollout_start_index"] != scope.current_index
        or _plain_json(metadata["controlled_agent_ids"]) != expected_ids
        or metadata["scenario_source"] != source.metadata.get("source")
        or metadata["scenario_source_fingerprint"]
        != source.metadata.get("source_fingerprint")
        or rollout.perturbation != perturbation_identity
    ):
        raise ValueError("analytic rollout execution provenance drifted")
    modes = _plain_json(metadata["agent_control_modes"])
    if not isinstance(modes, dict) or set(modes) != {
        str(agent.id) for agent in source.agents
    }:
        raise ValueError("analytic rollout component modes drifted")
    policy_name = policy_metadata["name"]
    for index, agent in enumerate(source.agents):
        expected_mode = (
            "typed_ego_plan" if index == source.ego_index else policy_name
        )
        if modes[str(agent.id)] != expected_mode:
            raise ValueError("analytic rollout component mode drifted")


def _validate_one_analytic_rollout(
    scope: M6AnalyticOracleScope,
    rollout: RolloutSnapshot,
    trace: PolicyExecutionTrace,
    plan: EgoTrajectoryPlan,
    *,
    policy_metadata: Mapping[str, Any],
) -> None:
    source = scope.source_snapshot
    rollout.revalidate()
    trace.validate_for_rollout(rollout)
    plan.revalidate()
    if (
        rollout.scenario_id != scope.scenario_id
        or rollout.sim_name != policy_metadata["name"]
        or rollout.sim_version != policy_metadata["version"]
        or rollout.seed != M6_NUMPY_SEED
        or rollout.num_steps != scope.stop_index + 1
        or rollout.num_agents != source.num_agents
        or not np.array_equal(
            rollout.timestamps,
            source.timestamps[: scope.stop_index + 1],
        )
    ):
        raise ValueError("analytic rollout identity, seed, or horizon drifted")
    if (
        trace.policy_access_role != "history_only"
        or trace.policy_name != policy_metadata["name"]
        or trace.policy_version != policy_metadata["version"]
        or trace.start_index != scope.current_index
        or trace.stop_index != scope.stop_index
        or trace.ego_index != source.ego_index
        or trace.perturbation_identity != plan.perturbation_identity
    ):
        raise ValueError("analytic trace access or execution identity drifted")
    if np.any(trace.override_mask):
        raise ValueError("history-only analytic trace cannot use overrides")
    _validate_rollout_metadata(
        rollout,
        scope,
        policy_metadata=policy_metadata,
        perturbation_identity=plan.perturbation_identity,
    )
    for index, (source_agent, rollout_agent) in enumerate(
        zip(source.agents, rollout.agents, strict=True)
    ):
        if (
            source_agent.id != rollout_agent.id
            or source_agent.type != rollout_agent.type
            or source_agent.length != rollout_agent.length
            or source_agent.width != rollout_agent.width
            or not np.array_equal(
                source_agent.valid[: scope.stop_index + 1],
                rollout_agent.valid,
            )
        ):
            raise ValueError("analytic rollout agent contract drifted")
        for name in _FLOAT_STATE_FIELDS:
            if not np.array_equal(
                getattr(source_agent, name)[: scope.current_index + 1],
                getattr(rollout_agent, name)[: scope.current_index + 1],
            ):
                raise ValueError("analytic rollout observed history drifted")
        if index == source.ego_index:
            window = slice(scope.current_index, scope.stop_index + 1)
            for name in _PLAN_STATE_FIELDS:
                if not np.array_equal(
                    getattr(rollout_agent, name)[window],
                    getattr(plan, name),
                ):
                    raise ValueError(
                        "analytic rollout ego does not equal serialized plan"
                    )


def _trace_action_tensors_equal(
    left: PolicyExecutionTrace,
    right: PolicyExecutionTrace,
) -> bool:
    left.revalidate()
    right.revalidate()
    if (
        left.policy_access_role != right.policy_access_role
        or left.start_index != right.start_index
        or left.stop_index != right.stop_index
        or left.ego_index != right.ego_index
        or left.perturbation_identity != right.perturbation_identity
        or not np.array_equal(left.timestamps, right.timestamps)
    ):
        return False
    left_modes = tuple(
        "<policy>" if mode == left.policy_name else mode
        for mode in left.control_modes
    )
    right_modes = tuple(
        "<policy>" if mode == right.policy_name else mode
        for mode in right.control_modes
    )
    if left_modes != right_modes:
        return False
    return all(
        np.array_equal(getattr(left, name), getattr(right, name))
        for name in (*_TRACE_FLOAT_FIELDS, *_TRACE_BOOL_FIELDS)
    )


def _rollout_tensors_equal(
    left: RolloutSnapshot,
    right: RolloutSnapshot,
) -> bool:
    left.revalidate()
    right.revalidate()
    if (
        left.scenario_id != right.scenario_id
        or left.seed != right.seed
        or not np.array_equal(left.timestamps, right.timestamps)
        or len(left.agents) != len(right.agents)
    ):
        return False
    for first, second in zip(left.agents, right.agents, strict=True):
        if (
            first.id != second.id
            or first.type != second.type
            or first.length != second.length
            or first.width != second.width
            or any(
                not np.array_equal(
                    getattr(first, name),
                    getattr(second, name),
                )
                for name in _WORLD_STATE_FIELDS
            )
        ):
            return False
    return True


@dataclass(frozen=True, slots=True)
class M6SyntheticOracleResult:
    """One traced analytic execution with raw-array acceptance evidence."""

    oracle_name: str
    scope: M6AnalyticOracleScope
    baseline_rollout: RolloutSnapshot = field(repr=False, compare=False)
    intervention_rollout: RolloutSnapshot = field(repr=False, compare=False)
    baseline_plan: EgoTrajectoryPlan = field(repr=False, compare=False)
    intervention_plan: EgoTrajectoryPlan = field(repr=False, compare=False)
    baseline_trace: PolicyExecutionTrace = field(repr=False, compare=False)
    intervention_trace: PolicyExecutionTrace = field(
        repr=False,
        compare=False,
    )
    policy_type_identity: str
    policy_metadata_json: str
    policy_access_role: str
    independent_measures: M6IndependentMeasures
    production_metric_results: tuple[PairedMetricResult, ...]
    test_only: bool
    nominal_identity_rollout: RolloutSnapshot | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    nominal_identity_trace: PolicyExecutionTrace | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.scope, M6AnalyticOracleScope):
            raise TypeError("oracle scope must be M6AnalyticOracleScope")
        for name in ("baseline_rollout", "intervention_rollout"):
            value = getattr(self, name)
            if not isinstance(value, RolloutSnapshot):
                raise TypeError(f"{name} must be a RolloutSnapshot")
        for name in ("baseline_plan", "intervention_plan"):
            value = getattr(self, name)
            if not isinstance(value, EgoTrajectoryPlan):
                raise TypeError(f"{name} must be an EgoTrajectoryPlan")
            object.__setattr__(
                self,
                name,
                EgoTrajectoryPlan.deserialize(bytes(value.serialize())),
            )
        object.__setattr__(
            self,
            "production_metric_results",
            tuple(self.production_metric_results),
        )
        self.revalidate()

    def _primary_pair(self) -> CounterfactualPair:
        if not self.scope.primary_eligibility.eligible:
            raise ValueError(
                "rejected analytic scope cannot masquerade as a primary pair"
            )
        return CounterfactualPair(
            scenario=self.scope.source_snapshot,
            baseline=self.baseline_rollout,
            intervention=self.intervention_rollout,
            baseline_plan=self.baseline_plan,
            intervention_plan=self.intervention_plan,
            eligibility=self.scope.primary_eligibility,
            intervention_identity=self.intervention_plan.perturbation_identity,
        )

    def revalidate(self) -> None:
        _assert_live_production_semantics()
        self.scope.revalidate()
        expected_type, expected_policy = _expected_policy_for_oracle(
            self.oracle_name
        )
        expected_metadata = expected_policy.metadata().to_dict()
        expected_json = json.dumps(
            expected_metadata,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        if (
            self.oracle_name not in {_NO_CONFLICT_ORACLE, _SENTINEL_ORACLE}
            or self.policy_type_identity != _type_identity(expected_type)
            or self.policy_metadata_json != expected_json
            or self.policy_access_role != "history_only"
            or type(self.test_only) is not bool
            or self.test_only != (self.oracle_name == _SENTINEL_ORACLE)
        ):
            raise ValueError("analytic oracle policy/type/access identity drifted")
        expected_fixture = (
            _NO_CONFLICT_FIXTURE_ID
            if self.oracle_name == _NO_CONFLICT_ORACLE
            else _SENTINEL_FIXTURE_ID
        )
        if self.scope.fixture_id != expected_fixture:
            raise ValueError("analytic oracle source scope drifted")
        validate_registered_ego_plan(
            self.scope.source_snapshot.to_scenario(),
            self.baseline_plan,
        )
        validate_registered_ego_plan(
            self.scope.source_snapshot.to_scenario(),
            self.intervention_plan,
        )
        expected_identity = compile_identity_plan(
            self.scope.source_snapshot.to_scenario()
        )
        expected_intervention = compile_longitudinal_brake_pulse_plan(
            self.scope.source_snapshot.to_scenario(),
            PRIMARY_BRAKE_MAGNITUDE_MPS2,
        )
        if (
            self.baseline_plan.serialize() != expected_identity.serialize()
            or self.intervention_plan.serialize()
            != expected_intervention.serialize()
        ):
            raise ValueError("analytic oracle serialized plans drifted")
        _validate_one_analytic_rollout(
            self.scope,
            self.baseline_rollout,
            self.baseline_trace,
            self.baseline_plan,
            policy_metadata=expected_metadata,
        )
        _validate_one_analytic_rollout(
            self.scope,
            self.intervention_rollout,
            self.intervention_trace,
            self.intervention_plan,
            policy_metadata=expected_metadata,
        )
        for index, (baseline_agent, intervention_agent) in enumerate(
            zip(
                self.baseline_rollout.agents,
                self.intervention_rollout.agents,
                strict=True,
            )
        ):
            if index == self.scope.source_snapshot.ego_index:
                continue
            for name in _WORLD_STATE_FIELDS:
                if not np.array_equal(
                    getattr(baseline_agent, name)[
                        : self.scope.current_index + 2
                    ],
                    getattr(intervention_agent, name)[
                        : self.scope.current_index + 2
                    ],
                ):
                    raise ValueError(
                        "analytic world response occurred before synchronous t+2"
                    )
        recomputed = _independent_measures(
            self.scope.source_snapshot,
            self.baseline_rollout,
            self.intervention_rollout,
            current=self.scope.current_index,
            stop=self.scope.stop_index,
            target_index=self.scope.target_index,
        )
        if recomputed != self.independent_measures:
            raise ValueError("retained independent raw measures drifted")
        if self.scope.primary_eligibility.eligible:
            pair = self._primary_pair()
            expected_results = _metric_results(pair)
            if not _metric_results_equal(
                self.production_metric_results,
                expected_results,
            ):
                raise ValueError("retained production metric results drifted")
            _crosscheck_metrics(
                recomputed,
                self.production_metric_results,
                target_index=self.scope.target_index,
                transition_count=(
                    self.scope.stop_index - self.scope.current_index
                ),
            )
        elif self.production_metric_results:
            raise ValueError(
                "rejected analytic scope cannot retain primary metric results"
            )
        if self.oracle_name == _NO_CONFLICT_ORACLE:
            if (
                self.nominal_identity_rollout is not None
                or self.nominal_identity_trace is not None
                or not recomputed.world_tensor_equal
                or not recomputed.structurally_nonreactive
                or recomputed.response_responded
            ):
                raise ValueError("no-conflict raw nonreaction gate failed")
        else:
            if (
                not isinstance(
                    self.nominal_identity_rollout,
                    RolloutSnapshot,
                )
                or not isinstance(
                    self.nominal_identity_trace,
                    PolicyExecutionTrace,
                )
            ):
                raise ValueError(
                    "sentinel must retain canonical nominal-IDM sham evidence"
                )
            nominal_metadata = IDMPolicy().metadata().to_dict()
            _validate_one_analytic_rollout(
                self.scope,
                self.nominal_identity_rollout,
                self.nominal_identity_trace,
                self.baseline_plan,
                policy_metadata=nominal_metadata,
            )
            if (
                not _rollout_tensors_equal(
                    self.baseline_rollout,
                    self.nominal_identity_rollout,
                )
                or not _trace_action_tensors_equal(
                    self.baseline_trace,
                    self.nominal_identity_trace,
                )
            ):
                raise ValueError(
                    "sentinel sham differs from canonical nominal-IDM sham"
                )
            if (
                recomputed.structurally_nonreactive
                or not recomputed.response_responded
                or recomputed.first_world_divergence_frame
                != self.scope.current_index + 2
            ):
                raise ValueError("test-only sentinel raw response gate failed")

    def metric(self, metric_name: str) -> PairedMetricResult:
        self.revalidate()
        for result in self.production_metric_results:
            if result.metric_name == metric_name:
                return result
        raise KeyError(metric_name)


def _snapshot_rollout(value: Any) -> RolloutSnapshot:
    if isinstance(value, RolloutSnapshot):
        value.revalidate()
        return RolloutSnapshot.from_rollout(
            _rollout_snapshot_to_rollout(value)
        )
    return RolloutSnapshot.from_rollout(value)


def _rollout_snapshot_to_rollout(value: RolloutSnapshot) -> Any:
    """Create a detached mutable Rollout only for defensive resnapshotting."""

    from evalsim.contracts import Rollout

    value.revalidate()
    return Rollout(
        scenario_id=value.scenario_id,
        sim_name=value.sim_name,
        sim_version=value.sim_version,
        seed=value.seed,
        timestamps=np.array(value.timestamps, copy=True),
        agents=[
            Agent(
                id=agent.id,
                type=agent.type,
                valid=np.array(agent.valid, copy=True),
                x=np.array(agent.x, copy=True),
                y=np.array(agent.y, copy=True),
                heading=np.array(agent.heading, copy=True),
                vx=np.array(agent.vx, copy=True),
                vy=np.array(agent.vy, copy=True),
                length=agent.length,
                width=agent.width,
            )
            for agent in value.agents
        ],
        perturbation=value.perturbation,
        metadata=_plain_json(value.metadata),
    )


def _execute_analytic_oracle(
    scope: M6AnalyticOracleScope,
    policy: HistoryOnlySimulatorPolicy,
    *,
    engine: RolloutEngine | None = None,
) -> M6SyntheticOracleResult:
    """Private exact-type seam for the two bounded analytic executions."""

    _assert_live_production_semantics()
    if not isinstance(scope, M6AnalyticOracleScope):
        raise TypeError("scope must be M6AnalyticOracleScope")
    scope.revalidate()
    oracle_name = (
        _NO_CONFLICT_ORACLE
        if scope.fixture_id == _NO_CONFLICT_FIXTURE_ID
        else _SENTINEL_ORACLE
    )
    expected_type, expected_policy = _expected_policy_for_oracle(oracle_name)
    if (
        type(policy) is not expected_type
        or not isinstance(policy, HistoryOnlySimulatorPolicy)
        or isinstance(policy, PrivilegedSimulatorPolicy)
        or policy.metadata().to_dict()
        != expected_policy.metadata().to_dict()
    ):
        raise TypeError(
            "analytic oracle policy must have the exact registered history-only type"
        )
    exact_engine = RolloutEngine() if engine is None else engine
    _validate_exact_engine(exact_engine)
    source = scope.source_snapshot.to_scenario()
    identity = compile_identity_plan(source)
    intervention = compile_longitudinal_brake_pulse_plan(
        source,
        PRIMARY_BRAKE_MAGNITUDE_MPS2,
    )
    baseline = exact_engine.run_with_trace(
        source,
        policy,
        seed=M6_NUMPY_SEED,
        ego_plan=identity,
    )
    treatment = exact_engine.run_with_trace(
        source,
        policy,
        seed=M6_NUMPY_SEED,
        ego_plan=intervention,
    )
    baseline_snapshot = _snapshot_rollout(baseline.rollout)
    treatment_snapshot = _snapshot_rollout(treatment.rollout)
    measures = _independent_measures(
        scope.source_snapshot,
        baseline_snapshot,
        treatment_snapshot,
        current=scope.current_index,
        stop=scope.stop_index,
        target_index=scope.target_index,
    )
    production_results: tuple[PairedMetricResult, ...] = ()
    if scope.primary_eligibility.eligible:
        pair = CounterfactualPair(
            scenario=scope.source_snapshot,
            baseline=baseline_snapshot,
            intervention=treatment_snapshot,
            baseline_plan=identity,
            intervention_plan=intervention,
            eligibility=scope.primary_eligibility,
            intervention_identity=intervention.perturbation_identity,
        )
        production_results = _metric_results(pair)
    nominal_rollout: RolloutSnapshot | None = None
    nominal_trace: PolicyExecutionTrace | None = None
    if oracle_name == _SENTINEL_ORACLE:
        nominal = exact_engine.run_with_trace(
            source,
            IDMPolicy(),
            seed=M6_NUMPY_SEED,
            ego_plan=identity,
        )
        nominal_rollout = _snapshot_rollout(nominal.rollout)
        nominal_trace = nominal.trace
    metadata = expected_policy.metadata().to_dict()
    return M6SyntheticOracleResult(
        oracle_name=oracle_name,
        scope=scope,
        baseline_rollout=baseline_snapshot,
        intervention_rollout=treatment_snapshot,
        baseline_plan=identity,
        intervention_plan=intervention,
        baseline_trace=baseline.trace,
        intervention_trace=treatment.trace,
        policy_type_identity=_type_identity(expected_type),
        policy_metadata_json=json.dumps(
            metadata,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        policy_access_role="history_only",
        independent_measures=measures,
        production_metric_results=production_results,
        test_only=oracle_name == _SENTINEL_ORACLE,
        nominal_identity_rollout=nominal_rollout,
        nominal_identity_trace=nominal_trace,
    )


def _build_no_conflict_oracle() -> M6SyntheticOracleResult:
    source = _no_conflict_scenario()
    scope = M6AnalyticOracleScope.from_source(
        _NO_CONFLICT_FIXTURE_ID,
        source,
        target_index=1,
    )
    return _execute_analytic_oracle(scope, IDMPolicy())


def _build_sentinel_oracle() -> M6SyntheticOracleResult:
    source = _aligned_scenario(0)
    scope = M6AnalyticOracleScope.from_source(
        _SENTINEL_FIXTURE_ID,
        source,
        target_index=1,
    )
    return _execute_analytic_oracle(
        scope,
        _OverreactiveSentinelPolicy(),
    )


def _metric_from_scene(
    scene: M6PairedSceneResult,
    metric_name: str,
) -> PairedMetricResult:
    for result in (
        *scene.primary_metric_results,
        *scene.secondary_metric_results,
    ):
        if result.metric_name == metric_name:
            return result
    raise M6SyntheticAcceptanceError(
        f"scene is missing required metric {metric_name!r}"
    )


def _scene_lookup(
    results: tuple[M6PairedSceneResult, ...],
) -> dict[tuple[int, str], M6PairedSceneResult]:
    lookup = {
        (scene.cohort_index, scene.policy_name): scene for scene in results
    }
    if len(lookup) != len(results):
        raise M6SyntheticAcceptanceError(
            "synthetic scene results contain duplicate case/policy keys"
        )
    return lookup


def _independent_scene_measures(
    scene: M6PairedSceneResult,
) -> M6IndependentMeasures:
    pair = scene.pair
    return _independent_measures(
        pair.scenario,
        pair.baseline,
        pair.intervention,
        current=pair.eligibility.current_index,
        stop=pair.eligibility.stop_index,
        target_index=int(pair.eligibility.target_index),
    )


def _raw_sham_matches_legacy(scene: M6PairedSceneResult) -> bool:
    source = scene.pair.scenario
    sham = scene.pair.baseline
    legacy = scene.legacy_rollout
    stop = scene.pair.eligibility.stop_index
    if (
        not np.array_equal(sham.timestamps, legacy.timestamps[: stop + 1])
        or len(sham.agents) != len(legacy.agents)
    ):
        return False
    for left, right in zip(sham.agents, legacy.agents, strict=True):
        if (
            left.id != right.id
            or left.type != right.type
            or left.length != right.length
            or left.width != right.width
            or any(
                not np.array_equal(
                    getattr(left, name),
                    getattr(right, name)[: stop + 1],
                )
                for name in _WORLD_STATE_FIELDS
            )
        ):
            return False
    sham_trace = scene.baseline_trace
    legacy_trace = scene.legacy_trace
    if (
        sham_trace.policy_name != legacy_trace.policy_name
        or sham_trace.policy_version != legacy_trace.policy_version
        or sham_trace.policy_access_role != legacy_trace.policy_access_role
        or sham_trace.start_index != legacy_trace.start_index
        or sham_trace.stop_index > legacy_trace.stop_index
        or sham_trace.ego_index != source.ego_index
    ):
        return False
    count = sham_trace.transition_count
    if not np.array_equal(
        sham_trace.timestamps,
        legacy_trace.timestamps[: count + 1],
    ):
        return False
    for index, (sham_mode, legacy_mode) in enumerate(
        zip(
            sham_trace.control_modes,
            legacy_trace.control_modes,
            strict=True,
        )
    ):
        if index == source.ego_index:
            if (
                sham_mode != "typed_ego_plan"
                or legacy_mode != "logged_ego"
            ):
                return False
        elif sham_mode != legacy_mode:
            return False
    return all(
        np.array_equal(
            getattr(sham_trace, name),
            getattr(legacy_trace, name)[:count],
        )
        for name in (*_TRACE_FLOAT_FIELDS, *_TRACE_BOOL_FIELDS)
    )


def _early_braking_impulse(
    scene: M6PairedSceneResult,
    *,
    duration_s: float,
) -> float:
    pair = scene.pair
    current = pair.eligibility.current_index
    stop = pair.eligibility.stop_index
    target = int(pair.eligibility.target_index)
    baseline_acc, intervention_acc, dt = _target_accelerations(
        pair.baseline,
        pair.intervention,
        target_index=target,
        current=current,
        stop=stop,
    )
    cutoff = float(pair.baseline.timestamps[current]) + duration_s
    components: list[float] = []
    for offset, transition_dt in enumerate(dt):
        frame = current + offset
        start = float(pair.baseline.timestamps[frame])
        end = float(pair.baseline.timestamps[frame + 1])
        overlap = max(0.0, min(end, cutoff) - start)
        if overlap > 0.0:
            components.append(
                max(
                    0.0,
                    float(
                        baseline_acc[offset] - intervention_acc[offset]
                    ),
                )
                * overlap
            )
    return math.fsum(components)


def _assert_exact_fixture_cohort(result: M6EvaluationResult) -> None:
    ledger = result.eligibility_ledger
    if (
        ledger.input_n != M6_SYNTHETIC_CASE_COUNT
        or ledger.eligible_n != M6_SYNTHETIC_CASE_COUNT
        or ledger.eligible_indices != tuple(range(M6_SYNTHETIC_CASE_COUNT))
    ):
        raise M6SyntheticAcceptanceError(
            "fixed synthetic cohort must contain ten eligible replicas"
        )
    for index, entry in enumerate(ledger.entries):
        entry.revalidate()
        expected = ScenarioSnapshot.from_scenario(_aligned_scenario(index))
        if (
            _source_fingerprint(entry.source_snapshot)
            != _source_fingerprint(expected)
            or entry.target_index != 1
            or entry.eligibility.current_index != M6_SYNTHETIC_CURRENT_INDEX
            or entry.eligibility.stop_index
            != M6_SYNTHETIC_CURRENT_INDEX + 40
        ):
            raise M6SyntheticAcceptanceError(
                "synthetic replica source/target/window drifted"
            )
        metadata = entry.source_snapshot.metadata
        if (
            metadata.get("analytic_unique_geometry_count") != 1
            or metadata.get("analytic_replica_count")
            != M6_SYNTHETIC_CASE_COUNT
            or metadata.get("analytic_replica_index") != index
            or metadata.get("analytic_geometry_id") != _ALIGNED_GEOMETRY_ID
        ):
            raise M6SyntheticAcceptanceError(
                "synthetic deterministic-replica labeling drifted"
            )


def _assert_canonical_matrix(result: M6EvaluationResult) -> None:
    if (
        len(result.primary_scene_results) != M6_SYNTHETIC_CASE_COUNT * 3
        or len(result.secondary_scene_results) != M6_SYNTHETIC_CASE_COUNT * 3
        or len(result.secondary_plan_ledger) != M6_SYNTHETIC_CASE_COUNT
        or not all(entry.feasible for entry in result.secondary_plan_ledger)
    ):
        raise M6SyntheticAcceptanceError(
            "synthetic b=2/b=4 policy matrix is incomplete"
        )
    for scene in (
        *result.primary_scene_results,
        *result.secondary_scene_results,
    ):
        scene.revalidate()
        raw = _independent_scene_measures(scene)
        retained = (
            *scene.primary_metric_results,
            *scene.secondary_metric_results,
        )
        _crosscheck_metrics(
            raw,
            retained,
            target_index=int(scene.pair.eligibility.target_index),
            transition_count=(
                scene.pair.eligibility.stop_index
                - scene.pair.eligibility.current_index
            ),
        )
        if not _raw_sham_matches_legacy(scene):
            raise M6SyntheticAcceptanceError(
                "identity sham differs from legacy under raw comparison"
            )
        if scene.policy_name in {"log_replay", "constant_velocity"} and (
            not raw.world_tensor_equal
            or not raw.structurally_nonreactive
        ):
            raise M6SyntheticAcceptanceError(
                f"{scene.policy_name} reacted under raw b="
                f"{scene.intervention_magnitude_mps2:g} comparison"
            )

    primary = _scene_lookup(result.primary_scene_results)
    secondary = _scene_lookup(result.secondary_scene_results)
    for index in range(M6_SYNTHETIC_CASE_COUNT):
        b2 = primary[(index, "idm")]
        b4 = secondary[(index, "idm")]
        b2_raw = _independent_scene_measures(b2)
        if (
            not b2_raw.response_responded
            or b2_raw.additional_target_braking_impulse_mps <= 0.0
            or b2_raw.first_world_divergence_frame
            != b2.pair.eligibility.current_index + 2
            or b2_raw.response_start_transition is None
            or b2_raw.response_start_transition < 1
        ):
            raise M6SyntheticAcceptanceError(
                "nominal IDM raw aligned-response/t+2 gate failed"
            )
        b2_plan = b2.pair.intervention_plan
        b4_plan = b4.pair.intervention_plan
        if any(
            not np.array_equal(
                getattr(b2_plan, name)[0:1],
                getattr(b4_plan, name)[0:1],
            )
            for name in _PLAN_STATE_FIELDS
        ):
            raise M6SyntheticAcceptanceError(
                "nested doses do not share exact current ego state"
            )
        b2_speed = np.hypot(b2_plan.vx, b2_plan.vy)
        b4_speed = np.hypot(b4_plan.vx, b4_plan.vy)
        b2_progress = b2_plan.x - b2_plan.x[0]
        b4_progress = b4_plan.x - b4_plan.x[0]
        if np.any(b4_speed[1:] > b2_speed[1:]) or np.any(
            b4_progress[1:] > b2_progress[1:]
        ):
            raise M6SyntheticAcceptanceError(
                "b=4 ego plan is not nested below b=2"
            )
        b2_early = _early_braking_impulse(
            b2,
            duration_s=M6_SYNTHETIC_EARLY_RESPONSE_WINDOW_S,
        )
        b4_early = _early_braking_impulse(
            b4,
            duration_s=M6_SYNTHETIC_EARLY_RESPONSE_WINDOW_S,
        )
        if b2_early <= 0.0 or b4_early < b2_early:
            raise M6SyntheticAcceptanceError(
                "independent early IDM dose-response gate failed"
            )


def _assert_no_conflict(oracle: M6SyntheticOracleResult) -> None:
    oracle.revalidate()
    scope = oracle.scope
    if (
        scope.primary_eligibility.eligible
        or scope.primary_eligibility.reason
        != "no_stable_aligned_follower"
        or not oracle.independent_measures.world_tensor_equal
        or not oracle.independent_measures.structurally_nonreactive
        or oracle.independent_measures.response_responded
        or oracle.production_metric_results
    ):
        raise M6SyntheticAcceptanceError(
            "typed no-conflict rejection/raw nonreaction gate failed"
        )
    expected = ScenarioSnapshot.from_scenario(_no_conflict_scenario())
    if _source_fingerprint(scope.source_snapshot) != _source_fingerprint(
        expected
    ):
        raise M6SyntheticAcceptanceError(
            "no-conflict exact fixed source geometry drifted"
        )


def _assert_sentinel(
    oracle: M6SyntheticOracleResult,
    nominal_scene: M6PairedSceneResult,
) -> None:
    oracle.revalidate()
    nominal_scene.revalidate()
    nominal_raw = _independent_scene_measures(nominal_scene)
    raw = oracle.independent_measures
    if (
        not raw.response_responded
        or raw.structurally_nonreactive
        or raw.first_world_divergence_frame != oracle.scope.current_index + 2
        or raw.target_progress_loss_m - nominal_raw.target_progress_loss_m
        < M6_SYNTHETIC_MIN_SENTINEL_EXTRA_PROGRESS_LOSS_M
        or raw.additional_hard_braking_exposure_s
        - nominal_raw.additional_hard_braking_exposure_s
        < M6_SYNTHETIC_MIN_SENTINEL_EXTRA_HARD_BRAKING_S
    ):
        raise M6SyntheticAcceptanceError(
            "test-only sentinel raw response/cost separation gate failed"
        )


@dataclass(frozen=True, slots=True)
class M6SyntheticAcceptanceResult:
    """Complete immutable data-free M6 evidence with independent gate truth."""

    evaluation: M6EvaluationResult = field(repr=False, compare=False)
    no_conflict: M6SyntheticOracleResult
    overreactive_sentinel: M6SyntheticOracleResult

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation, M6EvaluationResult):
            raise TypeError("evaluation must be an M6EvaluationResult")
        if not isinstance(self.no_conflict, M6SyntheticOracleResult):
            raise TypeError("no_conflict must be M6SyntheticOracleResult")
        if not isinstance(
            self.overreactive_sentinel,
            M6SyntheticOracleResult,
        ):
            raise TypeError(
                "overreactive_sentinel must be M6SyntheticOracleResult"
            )
        self.revalidate()

    def revalidate(self) -> None:
        _assert_live_production_semantics()
        self.evaluation.revalidate()
        _assert_exact_fixture_cohort(self.evaluation)
        _assert_canonical_matrix(self.evaluation)
        _assert_no_conflict(self.no_conflict)
        nominal = _scene_lookup(self.evaluation.primary_scene_results)[
            (0, "idm")
        ]
        _assert_sentinel(self.overreactive_sentinel, nominal)

    @property
    def case_count(self) -> int:
        self.revalidate()
        return self.evaluation.pair_n

    def to_local_dict(self) -> dict[str, Any]:
        self.revalidate()
        nominal = _scene_lookup(self.evaluation.primary_scene_results)[
            (0, "idm")
        ]
        nominal_raw = _independent_scene_measures(nominal)
        sentinel = self.overreactive_sentinel.independent_measures
        return {
            "schema_version": M6_SYNTHETIC_ACCEPTANCE_VERSION,
            "case_count": M6_SYNTHETIC_CASE_COUNT,
            "deterministic_replica_count": M6_SYNTHETIC_CASE_COUNT,
            "unique_analytic_geometry_count": 1,
            "all_replicas_eligible": True,
            "identity_sham_passed": True,
            "nonreactive_policy_dose_gate_count": (
                M6_SYNTHETIC_CASE_COUNT * 2 * 2
            ),
            "idm_response_replica_count": M6_SYNTHETIC_CASE_COUNT,
            "synchronous_response_floor_frames": 2,
            "nested_dose_replica_count": M6_SYNTHETIC_CASE_COUNT,
            "no_conflict_primary_eligibility": "rejected",
            "no_conflict_primary_rejection_reason": (
                self.no_conflict.scope.primary_eligibility.reason
            ),
            "no_conflict_exactly_nonreactive": True,
            "overreactive_sentinel_test_only": True,
            "sentinel_extra_progress_loss_m": (
                sentinel.target_progress_loss_m
                - nominal_raw.target_progress_loss_m
            ),
            "sentinel_extra_hard_braking_exposure_s": (
                sentinel.additional_hard_braking_exposure_s
                - nominal_raw.additional_hard_braking_exposure_s
            ),
        }


def run_m6_synthetic_acceptance() -> M6SyntheticAcceptanceResult:
    """Run and verify the exact fixed data-free M6 acceptance suite."""

    evaluation = run_m6_numpy_evaluation(
        synthetic_m6_cases(),
        include_local_secondary=True,
    )
    return M6SyntheticAcceptanceResult(
        evaluation=evaluation,
        no_conflict=_build_no_conflict_oracle(),
        overreactive_sentinel=_build_sentinel_oracle(),
    )


__all__ = [
    "M6_SYNTHETIC_ACCEPTANCE_VERSION",
    "M6_SYNTHETIC_ANALYTIC_SCOPE_VERSION",
    "M6_SYNTHETIC_CASE_COUNT",
    "M6_SYNTHETIC_CURRENT_INDEX",
    "M6_SYNTHETIC_DT_S",
    "M6_SYNTHETIC_EARLY_RESPONSE_WINDOW_S",
    "M6_SYNTHETIC_FRAME_COUNT",
    "M6_SYNTHETIC_MIN_SENTINEL_EXTRA_HARD_BRAKING_S",
    "M6_SYNTHETIC_MIN_SENTINEL_EXTRA_PROGRESS_LOSS_M",
    "M6AnalyticOracleScope",
    "M6IndependentMeasures",
    "M6SyntheticAcceptanceError",
    "M6SyntheticAcceptanceResult",
    "M6SyntheticOracleResult",
    "run_m6_synthetic_acceptance",
    "synthetic_m6_cases",
    "synthetic_m6_source_evidence",
]
