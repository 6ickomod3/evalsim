"""Source-neutral NumPy orchestration for EvalSim M6.

This module is intentionally bounded to the contract-native M6 execution matrix.  It
does not import WOMD, Waymax, result stores, or official-run lifecycle code.  Source
adapters must hand it the already-frozen cohort in canonical opaque-index order.

The orchestration is deliberately two phase:

1. defensively snapshot and validate every source case, then compute the complete
   source-only eligibility ledger; and
2. only when at least ten cases are eligible, compile every primary plan before
   executing complete baseline/treatment pairs.

Rejected cases never reach a policy.  A plan, rollout, pair, metric, or matrix defect
fails the complete evaluation instead of dropping or replacing a scene.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
from numbers import Integral
from types import MappingProxyType
from typing import Any, Callable, Iterable, MutableMapping, Protocol, Sequence, runtime_checkable

import numpy as np

from evalsim.contracts import (
    Agent,
    CounterfactualPair,
    EgoTrajectoryPlan,
    HistoryOnlySimulatorPolicy,
    InterventionEligibility,
    PairedMetricResult,
    PrivilegedSimulatorPolicy,
    Rollout,
    Scenario,
    SimulatorPolicy,
    evaluate_paired_metric,
)
from evalsim.contracts.counterfactual import RolloutSnapshot, ScenarioSnapshot
from evalsim.metrics.m6 import (
    M6_PRIMARY_PAIRED_METRIC_SPECS,
    M6_SECONDARY_PAIRED_METRIC_SPECS,
    is_exactly_nonreactive,
    m6_primary_paired_metrics,
    m6_secondary_paired_metrics,
    world_trajectory_tensor_equal,
)
from evalsim.perturb.m6 import (
    M6_ANALYSIS_TRANSITIONS,
    PRIMARY_BRAKE_MAGNITUDE_MPS2,
    PRIMARY_ELIGIBILITY_REASONS,
    SECONDARY_BRAKE_MAGNITUDE_MPS2,
    InterventionCompilationError,
    compile_identity_plan,
    compile_longitudinal_brake_pulse_plan,
    evaluate_primary_brake_eligibility,
)
from evalsim.rollout import (
    DYNAMICS_NAME,
    DYNAMICS_VERSION,
    ROLLOUT_ENGINE_NAME,
    ROLLOUT_ENGINE_VERSION,
    DynamicsLimits,
    PolicyExecutionTrace,
    RolloutEngine,
    TracedRollout,
    policy_trace_prefix_equal,
)
from evalsim.simulators import (
    CONSTANT_VELOCITY_VERSION,
    IDM_VERSION,
    LOG_REPLAY_VERSION,
    ConstantVelocityPolicy,
    IDMPolicy,
    LogReplayPolicy,
)
from evalsim.stats.m6 import (
    M6_MAX_PRIMARY_PAIR_N,
    M6_PRIMARY_METRICS,
    M6_PRIMARY_POLICY_ROLES,
    M6_RESPONSE_TIMELINESS_METRIC,
    M6PrimaryCellInput,
    M6PrimaryMatrixResult,
    M6SceneEffect,
    analyze_m6_primary_matrix,
    m6_primary_cell_specs,
)

M6_NUMPY_SEED = 0
M6_MINIMUM_PRIMARY_PAIR_N = 10
M6_NUMPY_POLICY_ORDER = tuple(
    role.policy_name for role in M6_PRIMARY_POLICY_ROLES
)
M6_NUMPY_POLICY_ACCESS_ROLES = tuple(
    role.access_role for role in M6_PRIMARY_POLICY_ROLES
)

_POLICY_ACCESS_BY_NAME = {
    role.policy_name: role.access_role for role in M6_PRIMARY_POLICY_ROLES
}
_CANONICAL_POLICY_TYPES = (
    LogReplayPolicy,
    ConstantVelocityPolicy,
    IDMPolicy,
)
_CANONICAL_POLICY_VERSIONS = (
    LOG_REPLAY_VERSION,
    CONSTANT_VELOCITY_VERSION,
    IDM_VERSION,
)
_CANONICAL_POLICY_METADATA_JSON = tuple(
    json.dumps(
        policy.metadata().to_dict(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    for policy in (
        LogReplayPolicy(),
        ConstantVelocityPolicy(),
        IDMPolicy(),
    )
)
_DEFAULT_DYNAMICS_LIMITS = MappingProxyType(DynamicsLimits().to_dict())
_CANONICAL_ENGINE_METADATA = MappingProxyType(
    {
        "name": ROLLOUT_ENGINE_NAME,
        "version": ROLLOUT_ENGINE_VERSION,
    }
)
_CANONICAL_DYNAMICS_CONFIGURATION = MappingProxyType(
    {
        "name": DYNAMICS_NAME,
        "version": DYNAMICS_VERSION,
        "integration": "midpoint_heading_trapezoidal_speed",
        "limits": _DEFAULT_DYNAMICS_LIMITS,
    }
)
_PRIMARY_METRIC_IDENTITIES = tuple(
    (spec.name, spec.version) for spec in M6_PRIMARY_PAIRED_METRIC_SPECS
)
_SECONDARY_METRIC_IDENTITIES = tuple(
    (spec.name, spec.version) for spec in M6_SECONDARY_PAIRED_METRIC_SPECS
)
_AGENT_TENSOR_FIELDS = ("valid", "x", "y", "heading", "vx", "vy")
_EXACT_NONREACTIVE_POLICIES = frozenset(
    {"log_replay", "constant_velocity"}
)


def _canonical_policy_metadata(policy_name: str) -> dict[str, object]:
    try:
        index = M6_NUMPY_POLICY_ORDER.index(policy_name)
    except ValueError as exc:
        raise ValueError("policy is not in the canonical M6 order") from exc
    return json.loads(_CANONICAL_POLICY_METADATA_JSON[index])


class M6EvaluationError(RuntimeError):
    """A source, orchestration, complete-pair, or matrix gate failed closed."""


class M6PrimaryOutcomeBlocked(M6EvaluationError):
    """The source-only eligible cohort is too small to execute primary outcomes."""

    def __init__(self, ledger: "M6EligibilityLedger") -> None:
        self.ledger = ledger
        super().__init__(
            "m6_primary_outcome_blocked: "
            f"eligible_n={ledger.eligible_n} is below "
            f"{M6_MINIMUM_PRIMARY_PAIR_N}"
        )


def _cohort_index(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("cohort_index must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError("cohort_index must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class M6EvaluationCase:
    """One already-selected source case identified only by its opaque index."""

    cohort_index: int
    scenario: Scenario = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cohort_index", _cohort_index(self.cohort_index))
        if not isinstance(self.scenario, Scenario):
            raise TypeError("scenario must be a Scenario")


@dataclass(frozen=True, slots=True)
class M6EligibilityLedgerEntry:
    """One source-only eligibility disposition in canonical cohort order."""

    cohort_index: int
    eligibility: InterventionEligibility
    source_snapshot: Scenario | ScenarioSnapshot = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "cohort_index", _cohort_index(self.cohort_index))
        if not isinstance(self.eligibility, InterventionEligibility):
            raise TypeError("eligibility must be InterventionEligibility")
        # Round-trip through the exact contract schema to detach even this small
        # object from any future caller-owned extension.
        object.__setattr__(
            self,
            "eligibility",
            InterventionEligibility.from_dict(self.eligibility.to_dict()),
        )
        if isinstance(self.source_snapshot, Scenario):
            source_snapshot = ScenarioSnapshot.from_scenario(
                self.source_snapshot
            )
        elif isinstance(self.source_snapshot, ScenarioSnapshot):
            self.source_snapshot.revalidate()
            source_snapshot = ScenarioSnapshot.from_scenario(
                self.source_snapshot.to_scenario()
            )
        else:
            raise TypeError(
                "source_snapshot must be a Scenario or ScenarioSnapshot"
            )
        object.__setattr__(self, "source_snapshot", source_snapshot)
        self.revalidate()

    def revalidate(self) -> None:
        if _cohort_index(self.cohort_index) != self.cohort_index:
            raise ValueError("eligibility ledger cohort_index drifted")
        if not isinstance(self.source_snapshot, ScenarioSnapshot):
            raise TypeError(
                "eligibility ledger source evidence must remain a ScenarioSnapshot"
            )
        self.source_snapshot.revalidate()
        if not isinstance(self.eligibility, InterventionEligibility):
            raise TypeError(
                "eligibility ledger disposition must remain typed"
            )
        reconstructed = InterventionEligibility.from_dict(
            self.eligibility.to_dict()
        )
        recomputed = evaluate_primary_brake_eligibility(
            self.source_snapshot.to_scenario()
        )
        if reconstructed.to_dict() != recomputed.to_dict():
            raise ValueError(
                "eligibility disposition drifted from its retained source snapshot"
            )

    @property
    def eligible(self) -> bool:
        return self.eligibility.eligible

    @property
    def reason(self) -> str | None:
        return self.eligibility.reason

    @property
    def target_index(self) -> int | None:
        return self.eligibility.target_index


@dataclass(frozen=True, slots=True)
class M6EligibilityLedger:
    """The complete source-only ledger for every evaluator input."""

    entries: tuple[M6EligibilityLedgerEntry, ...]

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if not entries or any(
            not isinstance(entry, M6EligibilityLedgerEntry) for entry in entries
        ):
            raise ValueError("eligibility ledger requires typed entries")
        for entry in entries:
            entry.revalidate()
        indices = tuple(entry.cohort_index for entry in entries)
        if any(left >= right for left, right in zip(indices, indices[1:])):
            raise ValueError(
                "eligibility ledger indices must be unique and strictly increasing"
            )
        object.__setattr__(self, "entries", entries)

    @property
    def input_n(self) -> int:
        return len(self.entries)

    @property
    def eligible_n(self) -> int:
        return sum(entry.eligible for entry in self.entries)

    @property
    def eligible_indices(self) -> tuple[int, ...]:
        return tuple(
            entry.cohort_index for entry in self.entries if entry.eligible
        )

    @property
    def rejection_counts(self) -> tuple[tuple[str, int], ...]:
        counts = Counter(
            entry.reason for entry in self.entries if not entry.eligible
        )
        return tuple(
            (reason, int(counts.get(reason, 0)))
            for reason in PRIMARY_ELIGIBILITY_REASONS
        )

    def entry_for(self, cohort_index: int) -> M6EligibilityLedgerEntry:
        normalized = _cohort_index(cohort_index)
        for entry in self.entries:
            if entry.cohort_index == normalized:
                return entry
        raise KeyError(normalized)


@dataclass(frozen=True, slots=True)
class M6SecondaryPlanEntry:
    """Outcome-blind b=4 feasibility on one primary-eligible case."""

    cohort_index: int
    feasible: bool
    reason: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cohort_index", _cohort_index(self.cohort_index))
        if type(self.feasible) is not bool:
            raise TypeError("feasible must be a bool")
        if self.feasible:
            if self.reason is not None:
                raise ValueError("a feasible secondary plan cannot have a reason")
        elif self.reason != "secondary_ego_plan_infeasible":
            raise ValueError(
                "an infeasible M6 severity plan requires the registered reason"
            )


@dataclass(frozen=True, slots=True)
class M6PairedSceneResult:
    """One complete policy × intervention pair with all scene-level measures."""

    cohort_index: int
    policy_name: str
    policy_access_role: str
    pair: CounterfactualPair = field(repr=False, compare=False)
    legacy_rollout: Rollout | RolloutSnapshot = field(
        repr=False,
        compare=False,
    )
    legacy_trace: PolicyExecutionTrace = field(repr=False, compare=False)
    baseline_trace: PolicyExecutionTrace = field(repr=False, compare=False)
    intervention_trace: PolicyExecutionTrace = field(repr=False, compare=False)
    primary_metric_results: tuple[PairedMetricResult, ...]
    secondary_metric_results: tuple[PairedMetricResult, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cohort_index", _cohort_index(self.cohort_index))
        object.__setattr__(
            self,
            "legacy_rollout",
            _defensive_rollout_snapshot(self.legacy_rollout),
        )
        object.__setattr__(
            self,
            "primary_metric_results",
            tuple(self.primary_metric_results),
        )
        object.__setattr__(
            self,
            "secondary_metric_results",
            tuple(self.secondary_metric_results),
        )
        self._validate_semantics()

    def _validate_semantics(self) -> None:
        if _cohort_index(self.cohort_index) != self.cohort_index:
            raise ValueError("scene result cohort_index drifted")
        if type(self.primary_metric_results) is not tuple or type(
            self.secondary_metric_results
        ) is not tuple:
            raise TypeError("scene metric collections must remain tuples")
        expected_access = _POLICY_ACCESS_BY_NAME.get(self.policy_name)
        if expected_access is None or self.policy_access_role != expected_access:
            raise ValueError("scene result policy/access role is not registered")
        if not isinstance(self.pair, CounterfactualPair):
            raise TypeError("pair must be a CounterfactualPair")
        self.pair.revalidate()
        if not isinstance(self.legacy_rollout, RolloutSnapshot):
            raise TypeError(
                "scene legacy evidence must remain a RolloutSnapshot"
            )
        self.legacy_rollout.revalidate()
        expected_policy = _canonical_policy_metadata(self.policy_name)
        expected_version = expected_policy["version"]
        if (
            self.pair.baseline.sim_name != self.policy_name
            or self.pair.baseline.sim_version != expected_version
            or self.pair.intervention.sim_name != self.policy_name
            or self.pair.intervention.sim_version != expected_version
        ):
            raise ValueError("scene result policy identity drifted from its pair")
        _validate_numpy_pair_metadata(self.pair, expected_policy)
        traces = (
            self.legacy_trace,
            self.baseline_trace,
            self.intervention_trace,
        )
        if any(not isinstance(trace, PolicyExecutionTrace) for trace in traces):
            raise TypeError("scene traces must be PolicyExecutionTrace values")
        for trace in traces:
            trace.revalidate()
            if (
                trace.policy_name != self.policy_name
                or trace.policy_version != expected_version
                or trace.policy_access_role != self.policy_access_role
            ):
                raise ValueError("scene trace policy/access identity drifted")
        self.legacy_trace.validate_for_rollout(self.legacy_rollout)
        self.baseline_trace.validate_for_rollout(self.pair.baseline)
        self.intervention_trace.validate_for_rollout(self.pair.intervention)
        if (
            self.legacy_trace.perturbation_identity is not None
            or self.baseline_trace.perturbation_identity
            != self.pair.baseline.perturbation
            or self.intervention_trace.perturbation_identity
            != self.pair.intervention.perturbation
            or not policy_trace_prefix_equal(
                self.baseline_trace,
                self.legacy_trace,
            )
        ):
            raise ValueError("scene trace plan provenance or sham gate drifted")
        assert_m6_sham_matches_legacy_prefix(
            self.pair.scenario.to_scenario(),
            self.legacy_rollout,
            self.pair.baseline,
            self.pair.baseline_plan,
            legacy_trace=self.legacy_trace,
            sham_trace=self.baseline_trace,
        )

        primary = self.primary_metric_results
        if any(not isinstance(result, PairedMetricResult) for result in primary):
            raise TypeError("scene primary metrics must be PairedMetricResult values")
        if tuple(
            (result.metric_name, result.metric_version) for result in primary
        ) != _PRIMARY_METRIC_IDENTITIES:
            raise ValueError(
                "scene result must contain the four primary metrics in exact order"
            )
        secondary = self.secondary_metric_results
        if any(not isinstance(result, PairedMetricResult) for result in secondary):
            raise TypeError("scene secondary metrics must be PairedMetricResult values")
        if secondary and tuple(
            (result.metric_name, result.metric_version) for result in secondary
        ) != _SECONDARY_METRIC_IDENTITIES:
            raise ValueError(
                "secondary scene metrics must use the exact registered order"
            )
        for result in (*primary, *secondary):
            if (
                result.scenario_id != self.pair.scenario.scenario_id
                or result.intervention_identity
                != self.pair.intervention_identity
            ):
                raise ValueError("scene metric identity drifted from its pair")
        expected_primary = _recompute_metric_results(
            self.pair,
            m6_primary_paired_metrics(),
        )
        if not _metric_results_exact(primary, expected_primary):
            raise ValueError(
                "scene primary metric values/details drifted from its exact pair"
            )
        if secondary:
            expected_secondary = _recompute_metric_results(
                self.pair,
                m6_secondary_paired_metrics(),
            )
            if not _metric_results_exact(secondary, expected_secondary):
                raise ValueError(
                    "scene secondary metric values/details drifted from its exact pair"
                )
        if (
            self.policy_name in _EXACT_NONREACTIVE_POLICIES
            and not is_exactly_nonreactive(self.pair)
        ):
            raise ValueError(
                "registered nonreactive policy produced a world response"
            )

    def revalidate(self) -> None:
        """Recompute every scene-level semantic after construction."""

        self._validate_semantics()

    @property
    def intervention_magnitude_mps2(self) -> float:
        self.revalidate()
        return float(self.pair.intervention_plan.spec.dose)

    @property
    def world_tensor_equal(self) -> bool:
        self.revalidate()
        return world_trajectory_tensor_equal(self.pair)

    @property
    def exactly_nonreactive(self) -> bool:
        self.revalidate()
        return is_exactly_nonreactive(self.pair)

    def primary_metric(self, metric_name: str) -> PairedMetricResult:
        self.revalidate()
        for result in self.primary_metric_results:
            if result.metric_name == metric_name:
                return result
        raise KeyError(metric_name)


@dataclass(frozen=True, slots=True)
class M6EvaluationResult:
    """Complete primary matrix plus explicitly separate optional local evidence."""

    eligibility_ledger: M6EligibilityLedger
    primary_scene_results: tuple[M6PairedSceneResult, ...]
    primary_cell_inputs: tuple[M6PrimaryCellInput, ...]
    primary_matrix: M6PrimaryMatrixResult
    secondary_plan_ledger: tuple[M6SecondaryPlanEntry, ...] = ()
    secondary_scene_results: tuple[M6PairedSceneResult, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.eligibility_ledger, M6EligibilityLedger):
            raise TypeError("eligibility_ledger must be M6EligibilityLedger")
        if self.eligibility_ledger.eligible_n < M6_MINIMUM_PRIMARY_PAIR_N:
            raise ValueError("an outcome result requires at least ten eligible pairs")
        primary = tuple(self.primary_scene_results)
        expected_order = tuple(
            (cohort_index, role.policy_name)
            for cohort_index in self.eligibility_ledger.eligible_indices
            for role in M6_PRIMARY_POLICY_ROLES
        )
        if tuple(
            (result.cohort_index, result.policy_name) for result in primary
        ) != expected_order:
            raise ValueError(
                "primary scene results are not in canonical case/policy order"
            )
        if any(
            not isinstance(result, M6PairedSceneResult)
            or result.intervention_magnitude_mps2
            != PRIMARY_BRAKE_MAGNITUDE_MPS2
            for result in primary
        ):
            raise ValueError("primary scene results must all use the b=2 plan")
        primary_fingerprints = {
            result.pair.intervention_plan.configuration_fingerprint
            for result in primary
        }
        if primary_fingerprints != {
            self.primary_matrix.intervention_config_fingerprint
        }:
            raise ValueError(
                "primary scene plans and matrix intervention identity drifted"
            )

        cells = tuple(self.primary_cell_inputs)
        if len(cells) != len(M6_PRIMARY_POLICY_ROLES) * len(
            M6_PRIMARY_METRICS
        ) or any(not isinstance(cell, M6PrimaryCellInput) for cell in cells):
            raise ValueError("primary_cell_inputs must contain the exact 12 cells")
        if tuple(cell.spec.identity for cell in cells) != tuple(
            row.spec.identity for row in self.primary_matrix.rows
        ):
            raise ValueError("primary cells and matrix row order drifted")
        if any(
            cell.cohort_indices != self.eligibility_ledger.eligible_indices
            for cell in cells
        ):
            raise ValueError("primary cells do not retain the complete eligible cohort")
        if self.primary_matrix.pair_n != self.eligibility_ledger.eligible_n:
            raise ValueError("matrix pair N drifted from the eligibility ledger")
        expected_effects: dict[tuple[str, str], list[M6SceneEffect]] = {
            (role.policy_name, metric.metric_name): []
            for role in M6_PRIMARY_POLICY_ROLES
            for metric in M6_PRIMARY_METRICS
        }
        for result in primary:
            for metric in result.primary_metric_results:
                expected_effects[
                    (result.policy_name, metric.metric_name)
                ].append(_scene_effect(result.cohort_index, metric))
        for cell in cells:
            if cell.scene_effects != tuple(
                expected_effects[(cell.spec.policy_name, cell.spec.metric_name)]
            ):
                raise ValueError(
                    "primary cell scene effects drifted from complete scene results"
                )

        secondary_ledger = tuple(self.secondary_plan_ledger)
        secondary = tuple(self.secondary_scene_results)
        if secondary_ledger:
            if any(
                len(result.secondary_metric_results)
                != len(_SECONDARY_METRIC_IDENTITIES)
                for result in primary
            ):
                raise ValueError(
                    "local secondary mode requires complete b=2 diagnostics"
                )
            if tuple(
                entry.cohort_index for entry in secondary_ledger
            ) != self.eligibility_ledger.eligible_indices:
                raise ValueError(
                    "secondary plan ledger must cover every primary-eligible case"
                )
            feasible = tuple(
                entry.cohort_index for entry in secondary_ledger if entry.feasible
            )
            expected_secondary = tuple(
                (cohort_index, role.policy_name)
                for cohort_index in feasible
                for role in M6_PRIMARY_POLICY_ROLES
            )
            if tuple(
                (result.cohort_index, result.policy_name) for result in secondary
            ) != expected_secondary:
                raise ValueError(
                    "secondary scene results are not the complete frozen subset"
                )
            if any(
                result.intervention_magnitude_mps2
                != SECONDARY_BRAKE_MAGNITUDE_MPS2
                for result in secondary
            ):
                raise ValueError("secondary scene results must all use the b=4 plan")
            if any(
                len(result.secondary_metric_results)
                != len(_SECONDARY_METRIC_IDENTITIES)
                for result in secondary
            ):
                raise ValueError(
                    "local b=4 results require complete secondary diagnostics"
                )
        elif secondary:
            raise ValueError(
                "secondary scene results require a source-only plan ledger"
            )
        elif any(result.secondary_metric_results for result in primary):
            raise ValueError(
                "b=2 secondary diagnostics require the separate secondary mode"
            )

        object.__setattr__(self, "primary_scene_results", primary)
        object.__setattr__(self, "primary_cell_inputs", cells)
        object.__setattr__(self, "secondary_plan_ledger", secondary_ledger)
        object.__setattr__(self, "secondary_scene_results", secondary)
        _validate_evaluation_result_integrity(self)

    def revalidate(self) -> None:
        """Independently reconstruct eligibility, plans, metrics, and statistics."""

        _validate_evaluation_result_integrity(self)

    @property
    def pair_n(self) -> int:
        self.revalidate()
        return self.eligibility_ledger.eligible_n


@runtime_checkable
class _M6TypedPlanExecutor(Protocol):
    """Private test seam; the public evaluator never accepts an executor."""

    def execute(
        self,
        scenario: Scenario,
        policy: SimulatorPolicy,
        *,
        seed: int,
        ego_plan: EgoTrajectoryPlan,
    ) -> TracedRollout:
        """Execute one exact typed plan with its immutable policy sidecar."""

    def execute_legacy(
        self,
        scenario: Scenario,
        policy: SimulatorPolicy,
        *,
        seed: int,
    ) -> TracedRollout:
        """Execute the frozen no-plan path for the sham equality gate."""


@dataclass(frozen=True, slots=True)
class _NumpyM6TypedPlanExecutor:
    """Pinned contract-native executor used by the public NumPy evaluator."""

    engine: RolloutEngine = field(default_factory=RolloutEngine)

    def __post_init__(self) -> None:
        if (
            type(self.engine) is not RolloutEngine
            or self.engine.name != ROLLOUT_ENGINE_NAME
            or self.engine.version != ROLLOUT_ENGINE_VERSION
            or self.engine.dynamics_limits.to_dict() != _DEFAULT_DYNAMICS_LIMITS
        ):
            raise TypeError(
                "M6 NumPy execution requires the exact default RolloutEngine"
            )

    def execute(
        self,
        scenario: Scenario,
        policy: SimulatorPolicy,
        *,
        seed: int,
        ego_plan: EgoTrajectoryPlan,
    ) -> TracedRollout:
        return self.engine.run_with_trace(
            scenario,
            policy,
            seed=seed,
            ego_plan=ego_plan,
        )

    def execute_legacy(
        self,
        scenario: Scenario,
        policy: SimulatorPolicy,
        *,
        seed: int,
    ) -> TracedRollout:
        return self.engine.run_with_trace(
            scenario,
            policy,
            seed=seed,
        )


@dataclass(frozen=True, slots=True)
class _ValidatedCase:
    cohort_index: int
    source_snapshot: ScenarioSnapshot
    caller_scenario: Scenario = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _PreparedCase:
    cohort_index: int
    source_snapshot: ScenarioSnapshot
    eligibility: InterventionEligibility
    identity_plan: EgoTrajectoryPlan
    primary_plan: EgoTrajectoryPlan
    secondary_plan: EgoTrajectoryPlan | None


def canonical_m6_policies() -> tuple[SimulatorPolicy, ...]:
    """Return fresh policies in exact pre-registered M6 execution order."""

    return (
        LogReplayPolicy(),
        ConstantVelocityPolicy(),
        IDMPolicy(),
    )


def evaluate_m6_source_eligibility(
    cases: Iterable[M6EvaluationCase],
) -> M6EligibilityLedger:
    """Validate all sources, then classify every input without running a policy."""

    validated = _validated_cases(cases)
    return _eligibility_ledger(validated)


def run_m6_numpy_evaluation(
    cases: Iterable[M6EvaluationCase],
    *,
    include_local_secondary: bool = False,
) -> M6EvaluationResult:
    """Run the exact pinned source-neutral M6 NumPy primary matrix.

    The primary execution order is case-major, then
    ``log_replay, constant_velocity, idm``; within each policy, identity executes
    immediately before b=2 treatment.  When requested, b=4 executes afterward on its
    separately frozen feasible subset.  It never enters the primary statistics.

    Policy instances, their complete metadata/configuration, the executor, rollout
    engine identity/version, dynamics identity/version, and default dynamics limits
    are all fixed here.  Alternate executors are intentionally not part of this
    public evidence path.
    """

    return _run_m6_numpy_evaluation_with_executor(
        cases,
        executor=_NumpyM6TypedPlanExecutor(),
        include_local_secondary=include_local_secondary,
    )


def _run_m6_numpy_evaluation_with_phase_observer(
    cases: Iterable[M6EvaluationCase],
    phase_observer: Callable[[str, int], None],
    clock_ns: Callable[[], int],
) -> M6EvaluationResult:
    """Run the fixed evaluator while reporting actual phase nanoseconds."""

    if not callable(phase_observer) or not callable(clock_ns):
        raise TypeError("phase_observer and clock_ns must be callable")
    return _run_m6_numpy_evaluation_with_executor(
        cases,
        executor=_NumpyM6TypedPlanExecutor(),
        include_local_secondary=True,
        phase_observer=phase_observer,
        clock_ns=clock_ns,
    )


def _observe_m6_phase(
    name: str,
    operation: Callable[[], Any],
    *,
    phase_observer: Callable[[str, int], None] | None,
    clock_ns: Callable[[], int] | None,
) -> Any:
    if phase_observer is None:
        return operation()
    assert clock_ns is not None
    start = clock_ns()
    result = operation()
    stop = clock_ns()
    if type(start) is not int or type(stop) is not int or stop <= start:
        raise M6EvaluationError("m6_phase_clock_did_not_advance")
    phase_observer(name, stop - start)
    return result


@dataclass(frozen=True, slots=True)
class _M6PolicyExecutionProducts:
    legacy: TracedRollout
    baseline: TracedRollout
    intervention: TracedRollout
    primary_pair: CounterfactualPair
    secondary_rollout: TracedRollout | None
    secondary_pair: CounterfactualPair | None


def _execute_prepared_policy_rollouts(
    execution: _M6TypedPlanExecutor,
    case: _PreparedCase,
    *,
    policy_name: str,
    policy: SimulatorPolicy,
) -> _M6PolicyExecutionProducts:
    """Execute only the ordered legacy/identity/b=2[/b=4] policy conditions."""

    legacy = _execute_legacy(
        execution,
        case,
        policy,
        policy_name=policy_name,
    )
    baseline = _execute_condition(
        execution,
        case,
        policy,
        policy_name=policy_name,
        condition="identity",
        plan=case.identity_plan,
    )
    _validate_sham_gate(
        case,
        policy_name=policy_name,
        legacy=legacy,
        baseline=baseline,
    )
    intervention = _execute_condition(
        execution,
        case,
        policy,
        policy_name=policy_name,
        condition="primary_b2",
        plan=case.primary_plan,
    )
    primary_pair = _build_pair(
        case,
        baseline=baseline.rollout,
        intervention=intervention.rollout,
        intervention_plan=case.primary_plan,
        policy_name=policy_name,
        condition="primary_b2",
    )
    _assert_registered_nonreactivity(
        primary_pair,
        cohort_index=case.cohort_index,
        policy_name=policy_name,
        condition="primary_b2",
    )
    secondary_rollout = None
    secondary_pair = None
    if case.secondary_plan is not None:
        secondary_rollout = _execute_condition(
            execution,
            case,
            policy,
            policy_name=policy_name,
            condition="secondary_b4",
            plan=case.secondary_plan,
        )
        secondary_pair = _build_pair(
            case,
            baseline=baseline.rollout,
            intervention=secondary_rollout.rollout,
            intervention_plan=case.secondary_plan,
            policy_name=policy_name,
            condition="secondary_b4",
        )
        _assert_registered_nonreactivity(
            secondary_pair,
            cohort_index=case.cohort_index,
            policy_name=policy_name,
            condition="secondary_b4",
        )
    return _M6PolicyExecutionProducts(
        legacy=legacy,
        baseline=baseline,
        intervention=intervention,
        primary_pair=primary_pair,
        secondary_rollout=secondary_rollout,
        secondary_pair=secondary_pair,
    )


def _analyze_prepared_policy(
    products: _M6PolicyExecutionProducts,
    case: _PreparedCase,
    *,
    policy_name: str,
    access_role: str,
    primary_metrics: Sequence[object],
    secondary_metrics: Sequence[object],
) -> tuple[M6PairedSceneResult, M6PairedSceneResult | None]:
    """Compute the paired measures and issue typed scene results."""

    primary_scene = M6PairedSceneResult(
        cohort_index=case.cohort_index,
        policy_name=policy_name,
        policy_access_role=access_role,
        pair=products.primary_pair,
        legacy_rollout=products.legacy.rollout,
        legacy_trace=products.legacy.trace,
        baseline_trace=products.baseline.trace,
        intervention_trace=products.intervention.trace,
        primary_metric_results=_evaluate_metrics(
            products.primary_pair,
            primary_metrics,
            cohort_index=case.cohort_index,
            policy_name=policy_name,
            family="primary",
        ),
        secondary_metric_results=_evaluate_metrics(
            products.primary_pair,
            secondary_metrics,
            cohort_index=case.cohort_index,
            policy_name=policy_name,
            family="local_secondary",
        ),
    )
    secondary_scene = None
    if products.secondary_pair is not None:
        if products.secondary_rollout is None:
            raise M6EvaluationError("m6_secondary_execution_product_incomplete")
        secondary_scene = M6PairedSceneResult(
            cohort_index=case.cohort_index,
            policy_name=policy_name,
            policy_access_role=access_role,
            pair=products.secondary_pair,
            legacy_rollout=products.legacy.rollout,
            legacy_trace=products.legacy.trace,
            baseline_trace=products.baseline.trace,
            intervention_trace=products.secondary_rollout.trace,
            primary_metric_results=_evaluate_metrics(
                products.secondary_pair,
                primary_metrics,
                cohort_index=case.cohort_index,
                policy_name=policy_name,
                family="secondary_b4_primary_measures",
            ),
            secondary_metric_results=_evaluate_metrics(
                products.secondary_pair,
                secondary_metrics,
                cohort_index=case.cohort_index,
                policy_name=policy_name,
                family="secondary_b4_local_measures",
            ),
        )
    return primary_scene, secondary_scene
def _run_m6_numpy_evaluation_with_executor(


    cases: Iterable[M6EvaluationCase],
    *,
    executor: _M6TypedPlanExecutor,
    include_local_secondary: bool = False,
    phase_observer: Callable[[str, int], None] | None = None,
    clock_ns: Callable[[], int] | None = None,
) -> M6EvaluationResult:
    """Private failure-injection seam used only by evaluator tests."""

    if type(include_local_secondary) is not bool:
        raise TypeError("include_local_secondary must be a bool")
    if not isinstance(executor, _M6TypedPlanExecutor):
        raise TypeError("executor must implement the private M6 execution protocol")
    if (phase_observer is None) != (clock_ns is None) or (
        phase_observer is not None
        and (not callable(phase_observer) or not callable(clock_ns))
    ):
        raise TypeError("phase observer and clock must be supplied together")
    validated = _validated_cases(cases)
    ledger = _eligibility_ledger(validated)
    if ledger.eligible_n < M6_MINIMUM_PRIMARY_PAIR_N:
        _assert_caller_sources_unchanged(validated)
        raise M6PrimaryOutcomeBlocked(ledger)

    policy_entries = _validated_policies(canonical_m6_policies())
    execution = executor

    prepared, secondary_ledger = _observe_m6_phase(
        "numpy_rollouts",
        lambda: _prepare_plans(
            validated,
            ledger,
            include_local_secondary=include_local_secondary,
        ),
        phase_observer=phase_observer,
        clock_ns=clock_ns,
    )
    primary_metrics = m6_primary_paired_metrics()
    secondary_metrics = (
        m6_secondary_paired_metrics() if include_local_secondary else ()
    )
    primary_scene_results: list[M6PairedSceneResult] = []
    secondary_scene_results: list[M6PairedSceneResult] = []
    effects: dict[tuple[str, str], list[M6SceneEffect]] = {
        (role.policy_name, metric.metric_name): []
        for role in M6_PRIMARY_POLICY_ROLES
        for metric in M6_PRIMARY_METRICS
    }

    for case in prepared:
        for policy_name, access_role, policy in policy_entries:
            products = _observe_m6_phase(
                "numpy_rollouts",
                lambda: _execute_prepared_policy_rollouts(
                    execution,
                    case,
                    policy_name=policy_name,
                    policy=policy,
                ),
                phase_observer=phase_observer,
                clock_ns=clock_ns,
            )
            scene, secondary_scene = _observe_m6_phase(
                "paired_metrics",
                lambda: _analyze_prepared_policy(
                    products,
                    case,
                    policy_name=policy_name,
                    access_role=access_role,
                    primary_metrics=primary_metrics,
                    secondary_metrics=secondary_metrics,
                ),
                phase_observer=phase_observer,
                clock_ns=clock_ns,
            )
            primary_scene_results.append(scene)
            for result in scene.primary_metric_results:
                effects[(policy_name, result.metric_name)].append(
                    _scene_effect(case.cohort_index, result)
                )
            if secondary_scene is not None:
                secondary_scene_results.append(secondary_scene)

    statistics_start = (
        clock_ns() if phase_observer is not None and clock_ns is not None else None
    )
    primary_fingerprint = prepared[0].primary_plan.configuration_fingerprint
    if any(
        case.primary_plan.configuration_fingerprint != primary_fingerprint
        for case in prepared[1:]
    ):
        raise M6EvaluationError(
            "m6_primary_plan_configuration_fingerprint_drift"
        )
    cell_specs = m6_primary_cell_specs(primary_fingerprint)
    cell_inputs = tuple(
        M6PrimaryCellInput(
            spec=spec,
            scene_effects=tuple(effects[(spec.policy_name, spec.metric_name)]),
            source_pairing_complete=True,
        )
        for spec in cell_specs
    )
    try:
        matrix = analyze_m6_primary_matrix(cell_inputs)
    except Exception as exc:
        raise M6EvaluationError("m6_primary_matrix_analysis_failed") from exc
    if statistics_start is not None:
        assert phase_observer is not None and clock_ns is not None
        statistics_stop = clock_ns()
        if type(statistics_stop) is not int or statistics_stop <= statistics_start:
            raise M6EvaluationError("m6_phase_clock_did_not_advance")
        phase_observer("statistics", statistics_stop - statistics_start)

    result = M6EvaluationResult(
        eligibility_ledger=ledger,
        primary_scene_results=tuple(primary_scene_results),
        primary_cell_inputs=cell_inputs,
        primary_matrix=matrix,
        secondary_plan_ledger=secondary_ledger,
        secondary_scene_results=tuple(secondary_scene_results),
    )
    _assert_caller_sources_unchanged(validated)
    return result


def _run_m6_numpy_pilot_execution(
    cases: Iterable[M6EvaluationCase],
    ledger: M6EligibilityLedger,
    selected_cohort_indices: Sequence[int],
    selection_binding_sha256: str,
    *,
    clock_ns: Callable[[], int],
) -> tuple[tuple[int, ...], str]:
    """Execute the exact policy/plan/metric block without returning outcomes.

    This narrow primitive exists so the outcome-suppressed pilot and full evaluator
    share one execution implementation. Its tuple return is private to the pilot
    adapter and contains only ceil-millisecond scene durations plus a digest binding
    the ordered selection to the complete source ledger.
    """

    if not callable(clock_ns):
        raise TypeError("clock_ns must be callable")
    validated = _validated_cases(cases)
    if (
        len(validated) != M6_MAX_PRIMARY_PAIR_N
        or tuple(case.cohort_index for case in validated)
        != tuple(range(M6_MAX_PRIMARY_PAIR_N))
    ):
        raise M6EvaluationError(
            "m6_pilot_requires_complete_128_case_cohort"
        )
    if not isinstance(ledger, M6EligibilityLedger):
        raise TypeError("ledger must be an M6EligibilityLedger")
    supplied = M6EligibilityLedger(
        tuple(
            M6EligibilityLedgerEntry(
                cohort_index=entry.cohort_index,
                eligibility=entry.eligibility,
                source_snapshot=entry.source_snapshot,
            )
            for entry in ledger.entries
        )
    )
    reconstructed = _eligibility_ledger(validated)
    if (
        supplied.input_n != M6_MAX_PRIMARY_PAIR_N
        or tuple(entry.cohort_index for entry in supplied.entries)
        != tuple(range(M6_MAX_PRIMARY_PAIR_N))
        or tuple(entry.cohort_index for entry in supplied.entries)
        != tuple(entry.cohort_index for entry in reconstructed.entries)
        or any(
            not _eligibility_exact(left.eligibility, right.eligibility)
            or not _scenario_snapshots_exact(
                left.source_snapshot,
                right.source_snapshot,
            )
            for left, right in zip(
                supplied.entries,
                reconstructed.entries,
                strict=True,
            )
        )
    ):
        raise M6EvaluationError(
            "m6_pilot_source_ledger_drifted"
        )
    if isinstance(selected_cohort_indices, (str, bytes, dict)):
        raise TypeError("selected_cohort_indices must be an ordered sequence")
    try:
        raw_indices = tuple(selected_cohort_indices)
    except TypeError as exc:
        raise TypeError(
            "selected_cohort_indices must be an ordered sequence"
        ) from exc
    if not 1 <= len(raw_indices) <= 8:
        raise M6EvaluationError("m6_pilot_requires_between_1_and_8_scenes")
    normalized: list[int] = []
    for value in raw_indices:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError("selected cohort indices must be integers")
        normalized.append(_cohort_index(value))
    selected = tuple(normalized)
    if len(set(selected)) != len(selected):
        raise M6EvaluationError("m6_pilot_selection_indices_not_unique")
    eligible = set(reconstructed.eligible_indices)
    if any(index not in eligible for index in selected):
        raise M6EvaluationError("m6_pilot_selection_contains_ineligible_case")
    if (
        not isinstance(selection_binding_sha256, str)
        or len(selection_binding_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in selection_binding_sha256
        )
    ):
        raise ValueError("selection_binding_sha256 must be lowercase SHA-256")

    selected_set = set(selected)
    selected_validated = tuple(
        case for case in validated if case.cohort_index in selected_set
    )
    selected_ledger = M6EligibilityLedger(
        tuple(
            entry
            for entry in reconstructed.entries
            if entry.cohort_index in selected_set
        )
    )
    plan_timing_ns: dict[int, int] = {}
    prepared, _secondary_ledger = _prepare_plans(
        selected_validated,
        selected_ledger,
        include_local_secondary=True,
        plan_timing_ns=plan_timing_ns,
        clock_ns=clock_ns,
    )
    prepared_by_index = {case.cohort_index: case for case in prepared}
    if set(prepared_by_index) != selected_set:
        raise M6EvaluationError("m6_pilot_prepared_selection_drifted")
    policy_entries = _validated_policies(canonical_m6_policies())
    execution = _NumpyM6TypedPlanExecutor()
    primary_metrics = m6_primary_paired_metrics()
    secondary_metrics = m6_secondary_paired_metrics()
    durations: list[int] = []
    for cohort_index in selected:
        before = clock_ns()
        if isinstance(before, bool) or not isinstance(before, int):
            raise M6EvaluationError("m6_pilot_clock_must_return_integer_ns")
        case = prepared_by_index[cohort_index]
        for policy_name, access_role, policy in policy_entries:
            products = _execute_prepared_policy_rollouts(
                execution,
                case,
                policy_name=policy_name,
                policy=policy,
            )
            primary_scene, secondary_scene = _analyze_prepared_policy(
                products,
                case,
                policy_name=policy_name,
                access_role=access_role,
                primary_metrics=primary_metrics,
                secondary_metrics=secondary_metrics,
            )
            primary_scene.revalidate()
            if secondary_scene is not None:
                secondary_scene.revalidate()
            del primary_scene, secondary_scene
        after = clock_ns()
        if (
            isinstance(after, bool)
            or not isinstance(after, int)
            or after <= before
        ):
            raise M6EvaluationError(
                "m6_pilot_clock_did_not_advance"
            )
        plan_duration_ns = plan_timing_ns.get(cohort_index)
        if type(plan_duration_ns) is not int or plan_duration_ns <= 0:
            raise M6EvaluationError(
                "m6_pilot_plan_timing_missing"
            )
        duration_ns = plan_duration_ns + after - before
        durations.append((duration_ns + 999_999) // 1_000_000)
    _assert_caller_sources_unchanged(validated)

    digest = hashlib.sha256(b"evalsim-m6-numpy-pilot-selection-v1\0")
    digest.update(bytes.fromhex(selection_binding_sha256))
    for entry in reconstructed.entries:
        digest.update(entry.cohort_index.to_bytes(4, "big"))
        disposition = json.dumps(
            entry.eligibility.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        digest.update(len(disposition).to_bytes(8, "big"))
        digest.update(disposition)
        digest.update(
            bytes.fromhex(entry.source_snapshot._integrity_fingerprint)
        )
    digest.update(len(selected).to_bytes(4, "big"))
    for cohort_index in selected:
        digest.update(cohort_index.to_bytes(4, "big"))
    return tuple(durations), digest.hexdigest()


def assert_m6_sham_matches_legacy_prefix(
    scenario: Scenario,
    legacy: Rollout | RolloutSnapshot,
    sham: Rollout | RolloutSnapshot,
    identity_plan: EgoTrajectoryPlan,
    *,
    legacy_trace: PolicyExecutionTrace,
    sham_trace: PolicyExecutionTrace,
) -> None:
    """Fail unless the sham trajectory tensors exactly equal the legacy prefix.

    Serialization and provenance are intentionally excluded: the typed sham carries a
    registered perturbation identity while the frozen legacy path carries ``None``.
    """

    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    if not isinstance(legacy, (Rollout, RolloutSnapshot)) or not isinstance(
        sham,
        (Rollout, RolloutSnapshot),
    ):
        raise TypeError(
            "legacy and sham must be Rollout or RolloutSnapshot values"
        )
    if not isinstance(legacy_trace, PolicyExecutionTrace) or not isinstance(
        sham_trace,
        PolicyExecutionTrace,
    ):
        raise TypeError("legacy_trace and sham_trace must be typed traces")
    legacy_trace.validate_for_rollout(legacy)
    sham_trace.validate_for_rollout(sham)
    if not policy_trace_prefix_equal(sham_trace, legacy_trace):
        raise M6EvaluationError("m6_sham_legacy_policy_trace_mismatch")
    if not isinstance(identity_plan, EgoTrajectoryPlan):
        raise TypeError("identity_plan must be an EgoTrajectoryPlan")
    identity_plan.revalidate()
    if (
        identity_plan.spec.family != "identity"
        or identity_plan.spec.version != "v1"
    ):
        raise ValueError("identity_plan must be the registered identity/v1 plan")
    current = int(scenario.metadata.get("current_index", 0))
    stop = current + M6_ANALYSIS_TRANSITIONS
    if (
        legacy.perturbation is not None
        or sham.perturbation != identity_plan.perturbation_identity
        or sham.num_steps != stop + 1
        or legacy.num_steps < sham.num_steps
        or legacy.scenario_id != sham.scenario_id
        or legacy.sim_name != sham.sim_name
        or legacy.sim_version != sham.sim_version
        or legacy.seed != M6_NUMPY_SEED
        or sham.seed != M6_NUMPY_SEED
    ):
        raise M6EvaluationError("m6_sham_legacy_identity_mismatch")
    _assert_sham_metadata_exact(
        scenario,
        legacy,
        sham,
        legacy_trace=legacy_trace,
        sham_trace=sham_trace,
    )
    if not np.array_equal(
        sham.timestamps,
        legacy.timestamps[: stop + 1],
    ):
        raise M6EvaluationError("m6_sham_legacy_timestamp_mismatch")
    if len(legacy.agents) != len(sham.agents):
        raise M6EvaluationError("m6_sham_legacy_agent_count_mismatch")
    for legacy_agent, sham_agent in zip(
        legacy.agents,
        sham.agents,
        strict=True,
    ):
        if (
            legacy_agent.id != sham_agent.id
            or legacy_agent.type != sham_agent.type
            or legacy_agent.length != sham_agent.length
            or legacy_agent.width != sham_agent.width
        ):
            raise M6EvaluationError("m6_sham_legacy_agent_contract_mismatch")
        for field_name in _AGENT_TENSOR_FIELDS:
            if not np.array_equal(
                getattr(sham_agent, field_name),
                getattr(legacy_agent, field_name)[: stop + 1],
            ):
                raise M6EvaluationError(
                    "m6_sham_legacy_agent_tensor_mismatch: "
                    f"agent_id={sham_agent.id}, field={field_name}"
                )


def _assert_sham_metadata_exact(
    scenario: Scenario,
    legacy: Rollout | RolloutSnapshot,
    sham: Rollout | RolloutSnapshot,
    *,
    legacy_trace: PolicyExecutionTrace,
    sham_trace: PolicyExecutionTrace,
) -> None:
    legacy_metadata = _plain_json(legacy.metadata)
    sham_metadata = _plain_json(sham.metadata)
    if not isinstance(legacy_metadata, dict) or not isinstance(
        sham_metadata,
        dict,
    ) or set(legacy_metadata) != set(sham_metadata):
        raise M6EvaluationError("m6_sham_legacy_metadata_fields_mismatch")
    expected_keys = {
        "engine",
        "dynamics",
        "policy",
        "ego_control",
        "rollout_start_index",
        "controlled_agent_ids",
        "agent_control_modes",
        "scenario_source",
        "scenario_source_fingerprint",
    }
    if set(legacy_metadata) != expected_keys:
        raise M6EvaluationError("m6_sham_legacy_metadata_fields_mismatch")

    for key in (
        "engine",
        "policy",
        "rollout_start_index",
        "controlled_agent_ids",
        "scenario_source",
        "scenario_source_fingerprint",
    ):
        if legacy_metadata[key] != sham_metadata[key]:
            raise M6EvaluationError(
                f"m6_sham_legacy_metadata_mismatch: field={key}"
            )
    try:
        expected_policy = _canonical_policy_metadata(legacy.sim_name)
    except ValueError:
        expected_policy = None
    if (
        expected_policy is None
        or legacy_metadata["policy"] != expected_policy
        or legacy_metadata["engine"] != _CANONICAL_ENGINE_METADATA
    ):
        raise M6EvaluationError(
            "m6_sham_legacy_canonical_execution_metadata_drifted"
        )
    if (
        legacy_metadata["ego_control"] != "logged"
        or sham_metadata["ego_control"] != "typed_ego_plan"
    ):
        raise M6EvaluationError("m6_sham_legacy_ego_control_mismatch")

    legacy_modes = legacy_metadata["agent_control_modes"]
    sham_modes = sham_metadata["agent_control_modes"]
    if not isinstance(legacy_modes, dict) or not isinstance(sham_modes, dict) or (
        set(legacy_modes) != set(sham_modes)
    ):
        raise M6EvaluationError("m6_sham_legacy_control_mode_fields_mismatch")
    ego_id = str(scenario.agents[scenario.ego_index].id)
    for agent_id in legacy_modes:
        if agent_id == ego_id:
            if (
                legacy_modes[agent_id] != "logged_ego"
                or sham_modes[agent_id] != "typed_ego_plan"
            ):
                raise M6EvaluationError(
                    "m6_sham_legacy_ego_control_mode_mismatch"
                )
        elif legacy_modes[agent_id] != sham_modes[agent_id]:
            raise M6EvaluationError(
                "m6_sham_legacy_world_control_mode_mismatch"
            )

    legacy_dynamics = legacy_metadata["dynamics"]
    sham_dynamics = sham_metadata["dynamics"]
    if not isinstance(legacy_dynamics, dict) or not isinstance(
        sham_dynamics,
        dict,
    ) or set(legacy_dynamics) != set(sham_dynamics):
        raise M6EvaluationError("m6_sham_legacy_dynamics_fields_mismatch")
    if set(legacy_dynamics) != {
        "name",
        "version",
        "integration",
        "limits",
        "clamp_counts",
    }:
        raise M6EvaluationError("m6_sham_legacy_dynamics_fields_mismatch")
    for key, expected in _CANONICAL_DYNAMICS_CONFIGURATION.items():
        if (
            legacy_dynamics[key] != expected
            or sham_dynamics[key] != expected
        ):
            raise M6EvaluationError(
                f"m6_sham_legacy_dynamics_mismatch: field={key}"
            )
    legacy_clamps = legacy_dynamics["clamp_counts"]
    sham_clamps = sham_dynamics["clamp_counts"]
    expected_clamp_keys = {
        "acceleration",
        "deceleration",
        "speed",
        "yaw_rate",
        "reverse_prevented",
    }
    if (
        not isinstance(legacy_clamps, dict)
        or not isinstance(sham_clamps, dict)
        or set(legacy_clamps) != expected_clamp_keys
        or set(sham_clamps) != expected_clamp_keys
        or any(
            type(legacy_clamps[key]) is not int
            or type(sham_clamps[key]) is not int
            or legacy_clamps[key] < 0
            or sham_clamps[key] < 0
            for key in expected_clamp_keys
        )
    ):
        raise M6EvaluationError(
            "m6_sham_legacy_clamp_counts_invalid"
        )
    if legacy.num_steps == sham.num_steps:
        if legacy_clamps != sham_clamps:
            raise M6EvaluationError(
                "m6_sham_legacy_equal_horizon_clamp_counts_mismatch"
            )
        return
    legacy_prefix_clamps = legacy_trace.replayed_clamp_counts_for_prefix(
        legacy,
        sham_trace.transition_count,
    )
    sham_replayed_clamps = sham_trace.replayed_clamp_counts_for_prefix(
        sham,
        sham_trace.transition_count,
    )
    if (
        legacy_prefix_clamps != sham_replayed_clamps
        or sham_replayed_clamps != sham_clamps
    ):
        raise M6EvaluationError(
            "m6_sham_legacy_prefix_clamp_counts_mismatch"
        )


def _validated_cases(
    raw_cases: Iterable[M6EvaluationCase],
) -> tuple[_ValidatedCase, ...]:
    if isinstance(raw_cases, (str, bytes, dict)):
        raise TypeError("cases must be an iterable of M6EvaluationCase values")
    try:
        cases = tuple(raw_cases)
    except TypeError as exc:
        raise TypeError(
            "cases must be an iterable of M6EvaluationCase values"
        ) from exc
    if not cases or any(not isinstance(case, M6EvaluationCase) for case in cases):
        raise M6EvaluationError("m6_cohort_requires_typed_cases")
    if len(cases) > M6_MAX_PRIMARY_PAIR_N:
        raise M6EvaluationError(
            f"m6_cohort_exceeds_maximum_n_{M6_MAX_PRIMARY_PAIR_N}"
        )
    indices = tuple(case.cohort_index for case in cases)
    if any(left >= right for left, right in zip(indices, indices[1:])):
        raise M6EvaluationError(
            "m6_cohort_indices_must_be_unique_and_strictly_increasing"
        )

    validated: list[_ValidatedCase] = []
    scenario_ids: set[str] = set()
    for case in cases:
        try:
            snapshot = ScenarioSnapshot.from_scenario(case.scenario)
            snapshot.revalidate()
            scenario = snapshot.to_scenario()
            _validate_full_source_contract(scenario)
        except Exception as exc:
            raise M6EvaluationError(
                "m6_source_contract_invalid: "
                f"cohort_index={case.cohort_index}"
            ) from exc
        if scenario.scenario_id in scenario_ids:
            raise M6EvaluationError("m6_source_scenario_id_is_not_unique")
        scenario_ids.add(scenario.scenario_id)
        validated.append(
            _ValidatedCase(
                cohort_index=case.cohort_index,
                source_snapshot=snapshot,
                caller_scenario=case.scenario,
            )
        )
    result = tuple(validated)
    _assert_caller_sources_unchanged(result)
    return result


def _validate_full_source_contract(scenario: Scenario) -> None:
    """Validate adapter/source integrity before any ordinary eligibility reason."""

    if scenario.num_steps < 1 or scenario.num_agents < 1:
        raise ValueError("scenario requires at least one frame and one agent")
    raw_current = scenario.metadata.get("current_index", 0)
    if (
        isinstance(raw_current, (bool, np.bool_))
        or not isinstance(raw_current, Integral)
        or not 0 <= int(raw_current) < scenario.num_steps
    ):
        raise ValueError("current_index must be an in-range integer")
    if any(not bool(np.any(agent.valid)) for agent in scenario.agents):
        raise ValueError("every contract agent must be valid at least once")
    # ScenarioSnapshot already enforces finite, increasing timestamps; unique integer
    # identities; finite bounded headings/state tensors; positive dimensions; finite
    # map coordinates; and recursively JSON-safe metadata.  Repeat the two array-shape
    # invariants most vulnerable to post-construction caller mutation.
    for agent in scenario.agents:
        if any(
            np.asarray(getattr(agent, name)).shape != (scenario.num_steps,)
            for name in _AGENT_TENSOR_FIELDS
        ):
            raise ValueError("agent tensors must match the scenario horizon")
        valid = np.asarray(agent.valid, dtype=bool)
        for name in ("x", "y", "heading", "vx", "vy"):
            values = np.asarray(getattr(agent, name))
            if not np.all(np.isfinite(values[valid])):
                raise ValueError(
                    f"valid agent payload {name} must be finite"
                )


def _eligibility_ledger(
    cases: tuple[_ValidatedCase, ...],
) -> M6EligibilityLedger:
    entries: list[M6EligibilityLedgerEntry] = []
    for case in cases:
        try:
            case.source_snapshot.revalidate()
            eligibility = evaluate_primary_brake_eligibility(
                case.source_snapshot.to_scenario()
            )
        except Exception as exc:
            raise M6EvaluationError(
                "m6_source_eligibility_evaluation_failed: "
                f"cohort_index={case.cohort_index}"
            ) from exc
        entries.append(
            M6EligibilityLedgerEntry(
                cohort_index=case.cohort_index,
                eligibility=eligibility,
                source_snapshot=case.source_snapshot,
            )
        )
    return M6EligibilityLedger(tuple(entries))


def _validated_policies(
    raw_policies: Sequence[SimulatorPolicy],
) -> tuple[tuple[str, str, SimulatorPolicy], ...]:
    if isinstance(raw_policies, (str, bytes)) or not isinstance(
        raw_policies,
        Sequence,
    ):
        raise TypeError("policies must be a sequence")
    policies = tuple(raw_policies)
    if len(policies) != len(_CANONICAL_POLICY_TYPES):
        raise M6EvaluationError("m6_requires_exactly_three_numpy_policies")
    entries: list[tuple[str, str, SimulatorPolicy]] = []
    for index, (policy, expected_type, expected_version, expected_role) in enumerate(
        zip(
            policies,
            _CANONICAL_POLICY_TYPES,
            _CANONICAL_POLICY_VERSIONS,
            M6_PRIMARY_POLICY_ROLES,
            strict=True,
        )
    ):
        if type(policy) is not expected_type:
            raise M6EvaluationError(
                "m6_policy_type_or_order_drifted: "
                f"position={index}"
            )
        metadata = policy.metadata()
        try:
            metadata_json = json.dumps(
                metadata.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        except Exception as exc:
            raise M6EvaluationError(
                f"m6_policy_metadata_invalid: policy={expected_role.policy_name}"
            ) from exc
        if (
            metadata.name != expected_role.policy_name
            or metadata.version != expected_version
            or metadata.deterministic is not True
            or metadata_json != _CANONICAL_POLICY_METADATA_JSON[index]
        ):
            raise M6EvaluationError(
                f"m6_policy_metadata_drifted: policy={expected_role.policy_name}"
            )
        history_only = isinstance(policy, HistoryOnlySimulatorPolicy)
        privileged = isinstance(policy, PrivilegedSimulatorPolicy)
        expected_history_only = expected_role.access_role == "history_only"
        if (
            history_only == privileged
            or history_only != expected_history_only
        ):
            raise M6EvaluationError(
                f"m6_policy_access_role_drifted: policy={metadata.name}"
            )
        entries.append((metadata.name, expected_role.access_role, policy))
    return tuple(entries)


def _prepare_plans(
    cases: tuple[_ValidatedCase, ...],
    ledger: M6EligibilityLedger,
    *,
    include_local_secondary: bool,
    plan_timing_ns: MutableMapping[int, int] | None = None,
    clock_ns: Callable[[], int] | None = None,
) -> tuple[tuple[_PreparedCase, ...], tuple[M6SecondaryPlanEntry, ...]]:
    if (plan_timing_ns is None) != (clock_ns is None):
        raise TypeError("plan timing accumulator and clock must be supplied together")
    if plan_timing_ns is not None and (
        not isinstance(plan_timing_ns, MutableMapping) or plan_timing_ns
    ):
        raise TypeError("plan_timing_ns must be an empty mutable mapping")
    if clock_ns is not None and not callable(clock_ns):
        raise TypeError("clock_ns must be callable")

    def start_plan_timing() -> int | None:
        if clock_ns is None:
            return None
        started = clock_ns()
        if type(started) is not int:
            raise M6EvaluationError("m6_plan_clock_must_return_integer_ns")
        return started

    def finish_plan_timing(cohort_index: int, started: int | None) -> None:
        if clock_ns is None:
            return
        assert started is not None and plan_timing_ns is not None
        stopped = clock_ns()
        if type(stopped) is not int or stopped <= started:
            raise M6EvaluationError("m6_plan_clock_did_not_advance")
        plan_timing_ns[cohort_index] = (
            plan_timing_ns.get(cohort_index, 0) + stopped - started
        )

    by_index = {case.cohort_index: case for case in cases}
    prepared_parts: list[
        tuple[
            _ValidatedCase,
            InterventionEligibility,
            EgoTrajectoryPlan,
            EgoTrajectoryPlan,
        ]
    ] = []
    for entry in ledger.entries:
        if not entry.eligible:
            continue
        plan_started = start_plan_timing()
        case = by_index[entry.cohort_index]
        try:
            case.source_snapshot.revalidate()
            source = case.source_snapshot.to_scenario()
            identity = compile_identity_plan(source)
            primary = compile_longitudinal_brake_pulse_plan(
                source,
                PRIMARY_BRAKE_MAGNITUDE_MPS2,
            )
        except Exception as exc:
            finish_plan_timing(case.cohort_index, plan_started)
            plan_started = None
            raise M6EvaluationError(
                "m6_accepted_case_primary_plan_compilation_failed: "
                f"cohort_index={case.cohort_index}"
            ) from exc
        finish_plan_timing(case.cohort_index, plan_started)
        prepared_parts.append(
            (case, entry.eligibility, identity, primary)
        )
    if len(
        {
            identity.configuration_fingerprint
            for _, _, identity, _ in prepared_parts
        }
    ) != 1 or len(
        {
            primary.configuration_fingerprint
            for _, _, _, primary in prepared_parts
        }
    ) != 1:
        raise M6EvaluationError(
            "m6_primary_plan_configuration_fingerprint_drift"
        )

    secondary_by_index: dict[int, EgoTrajectoryPlan | None] = {
        case.cohort_index: None for case, _, _, _ in prepared_parts
    }
    secondary_entries: list[M6SecondaryPlanEntry] = []
    if include_local_secondary:
        # Freeze the complete b=4 subset before the first policy outcome.
        for case, _, _, _ in prepared_parts:
            plan_started = start_plan_timing()
            try:
                secondary = compile_longitudinal_brake_pulse_plan(
                    case.source_snapshot.to_scenario(),
                    SECONDARY_BRAKE_MAGNITUDE_MPS2,
                )
            except InterventionCompilationError as exc:
                finish_plan_timing(case.cohort_index, plan_started)
                plan_started = None
                if exc.code != "secondary_ego_plan_infeasible":
                    raise M6EvaluationError(
                        "m6_secondary_plan_unexpected_compilation_failure: "
                        f"cohort_index={case.cohort_index}"
                    ) from exc
                secondary_entries.append(
                    M6SecondaryPlanEntry(
                        cohort_index=case.cohort_index,
                        feasible=False,
                        reason=exc.code,
                    )
                )
            except Exception as exc:
                finish_plan_timing(case.cohort_index, plan_started)
                plan_started = None
                raise M6EvaluationError(
                    "m6_secondary_plan_compilation_failed: "
                    f"cohort_index={case.cohort_index}"
                ) from exc
            else:
                finish_plan_timing(case.cohort_index, plan_started)
                secondary_by_index[case.cohort_index] = secondary
                secondary_entries.append(
                    M6SecondaryPlanEntry(
                        cohort_index=case.cohort_index,
                        feasible=True,
                        reason=None,
                    )
                )
        secondary_fingerprints = {
            plan.configuration_fingerprint
            for plan in secondary_by_index.values()
            if plan is not None
        }
        if len(secondary_fingerprints) > 1:
            raise M6EvaluationError(
                "m6_secondary_plan_configuration_fingerprint_drift"
            )

    prepared = tuple(
        _PreparedCase(
            cohort_index=case.cohort_index,
            source_snapshot=case.source_snapshot,
            eligibility=eligibility,
            identity_plan=identity,
            primary_plan=primary,
            secondary_plan=secondary_by_index[case.cohort_index],
        )
        for case, eligibility, identity, primary in prepared_parts
    )
    return prepared, tuple(secondary_entries)


def _execute_condition(
    executor: _M6TypedPlanExecutor,
    case: _PreparedCase,
    policy: SimulatorPolicy,
    *,
    policy_name: str,
    condition: str,
    plan: EgoTrajectoryPlan,
) -> TracedRollout:
    case.source_snapshot.revalidate()
    source = case.source_snapshot.to_scenario()
    source_before = ScenarioSnapshot.from_scenario(source)
    call_plan = EgoTrajectoryPlan.deserialize(bytes(plan.serialize()))
    try:
        rollout = executor.execute(
            source,
            policy,
            seed=M6_NUMPY_SEED,
            ego_plan=call_plan,
        )
    except Exception as exc:
        raise M6EvaluationError(
            "m6_pair_execution_failed: "
            f"cohort_index={case.cohort_index}, "
            f"policy={policy_name}, condition={condition}"
        ) from exc
    finally:
        _assert_fresh_execution_inputs_unchanged(
            source_before,
            source,
            original_plan=plan,
            call_plan=call_plan,
            cohort_index=case.cohort_index,
            policy_name=policy_name,
            condition=condition,
        )
    if not isinstance(rollout, TracedRollout):
        raise M6EvaluationError(
            "m6_executor_returned_non_traced_rollout: "
            f"cohort_index={case.cohort_index}, "
            f"policy={policy_name}, condition={condition}"
        )
    try:
        rollout.revalidate()
    except Exception as exc:
        raise M6EvaluationError(
            "m6_executor_trace_binding_invalid: "
            f"cohort_index={case.cohort_index}, "
            f"policy={policy_name}, condition={condition}"
        ) from exc
    if (
        rollout.trace.policy_name != policy_name
        or rollout.trace.policy_access_role
        != _POLICY_ACCESS_BY_NAME[policy_name]
        or rollout.trace.perturbation_identity
        != plan.perturbation_identity
    ):
        raise M6EvaluationError(
            "m6_executor_condition_identity_drifted: "
            f"cohort_index={case.cohort_index}, "
            f"policy={policy_name}, condition={condition}"
        )
    return rollout


def _execute_legacy(
    executor: _M6TypedPlanExecutor,
    case: _PreparedCase,
    policy: SimulatorPolicy,
    *,
    policy_name: str,
) -> TracedRollout:
    case.source_snapshot.revalidate()
    source = case.source_snapshot.to_scenario()
    source_before = ScenarioSnapshot.from_scenario(source)
    try:
        traced = executor.execute_legacy(
            source,
            policy,
            seed=M6_NUMPY_SEED,
        )
    except Exception as exc:
        raise M6EvaluationError(
            "m6_legacy_sham_reference_execution_failed: "
            f"cohort_index={case.cohort_index}, policy={policy_name}"
        ) from exc
    finally:
        _assert_fresh_execution_inputs_unchanged(
            source_before,
            source,
            original_plan=None,
            call_plan=None,
            cohort_index=case.cohort_index,
            policy_name=policy_name,
            condition="legacy",
        )
    if not isinstance(traced, TracedRollout):
        raise M6EvaluationError(
            "m6_executor_returned_non_traced_legacy: "
            f"cohort_index={case.cohort_index}, policy={policy_name}"
        )
    try:
        traced.revalidate()
    except Exception as exc:
        raise M6EvaluationError(
            "m6_legacy_trace_binding_invalid: "
            f"cohort_index={case.cohort_index}, policy={policy_name}"
        ) from exc
    if (
        traced.trace.policy_name != policy_name
        or traced.trace.policy_access_role
        != _POLICY_ACCESS_BY_NAME[policy_name]
        or traced.trace.perturbation_identity is not None
    ):
        raise M6EvaluationError(
            "m6_legacy_reference_identity_drifted: "
            f"cohort_index={case.cohort_index}, policy={policy_name}"
        )
    return traced


def _validate_sham_gate(
    case: _PreparedCase,
    *,
    policy_name: str,
    legacy: TracedRollout,
    baseline: TracedRollout,
) -> None:
    try:
        assert_m6_sham_matches_legacy_prefix(
            case.source_snapshot.to_scenario(),
            legacy.rollout,
            baseline.rollout,
            case.identity_plan,
            legacy_trace=legacy.trace,
            sham_trace=baseline.trace,
        )
        trace_equal = policy_trace_prefix_equal(
            baseline.trace,
            legacy.trace,
        )
    except Exception as exc:
        raise M6EvaluationError(
            "m6_sham_legacy_gate_failed: "
            f"cohort_index={case.cohort_index}, policy={policy_name}"
        ) from exc
    if not trace_equal:
        raise M6EvaluationError(
            "m6_sham_legacy_policy_trace_mismatch: "
            f"cohort_index={case.cohort_index}, policy={policy_name}"
        )


def _build_pair(
    case: _PreparedCase,
    *,
    baseline: Rollout,
    intervention: Rollout,
    intervention_plan: EgoTrajectoryPlan,
    policy_name: str,
    condition: str,
) -> CounterfactualPair:
    try:
        return CounterfactualPair(
            scenario=case.source_snapshot.to_scenario(),
            baseline=baseline,
            intervention=intervention,
            baseline_plan=case.identity_plan,
            intervention_plan=intervention_plan,
            eligibility=case.eligibility,
            intervention_identity=intervention_plan.perturbation_identity,
        )
    except Exception as exc:
        raise M6EvaluationError(
            "m6_complete_pair_validation_failed: "
            f"cohort_index={case.cohort_index}, "
            f"policy={policy_name}, condition={condition}"
        ) from exc


def _evaluate_metrics(
    pair: CounterfactualPair,
    metrics: Sequence[object],
    *,
    cohort_index: int,
    policy_name: str,
    family: str,
) -> tuple[PairedMetricResult, ...]:
    results: list[PairedMetricResult] = []
    for metric in metrics:
        try:
            results.append(evaluate_paired_metric(metric, pair))  # type: ignore[arg-type]
        except Exception as exc:
            metric_name = getattr(getattr(metric, "spec", None), "name", "unknown")
            raise M6EvaluationError(
                "m6_paired_metric_failed: "
                f"cohort_index={cohort_index}, policy={policy_name}, "
                f"family={family}, metric={metric_name}"
            ) from exc
    return tuple(results)


def _plain_json(value: object) -> object:
    """Return a detached exact JSON-shaped value for semantic comparison."""

    from collections.abc import Mapping as MappingABC

    if isinstance(value, MappingABC):
        return {
            str(key): _plain_json(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _defensive_rollout_snapshot(
    value: Rollout | RolloutSnapshot,
) -> RolloutSnapshot:
    if isinstance(value, Rollout):
        return RolloutSnapshot.from_rollout(value)
    if not isinstance(value, RolloutSnapshot):
        raise TypeError(
            "legacy_rollout must be a Rollout or RolloutSnapshot"
        )
    value.revalidate()
    metadata = _plain_json(value.metadata)
    if not isinstance(metadata, dict):
        raise ValueError("legacy rollout metadata must remain a mapping")
    detached = Rollout(
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
        metadata=metadata,
    )
    return RolloutSnapshot.from_rollout(detached)


def _metric_results_exact(
    actual: Sequence[PairedMetricResult],
    expected: Sequence[PairedMetricResult],
) -> bool:
    return tuple(_plain_json(result.to_dict()) for result in actual) == tuple(
        _plain_json(result.to_dict()) for result in expected
    )


def _recompute_metric_results(
    pair: CounterfactualPair,
    metrics: Sequence[object],
) -> tuple[PairedMetricResult, ...]:
    pair.revalidate()
    return tuple(
        evaluate_paired_metric(metric, pair)  # type: ignore[arg-type]
        for metric in metrics
    )


def _validate_numpy_rollout_metadata(
    rollout: RolloutSnapshot,
    expected_policy: dict[str, object],
) -> None:
    metadata = rollout.metadata
    expected_keys = {
        "engine",
        "dynamics",
        "policy",
        "ego_control",
        "rollout_start_index",
        "controlled_agent_ids",
        "agent_control_modes",
        "scenario_source",
        "scenario_source_fingerprint",
    }
    if set(metadata) != expected_keys:
        raise ValueError("M6 NumPy rollout metadata fields drifted")
    if _plain_json(metadata["engine"]) != _CANONICAL_ENGINE_METADATA:
        raise ValueError("M6 NumPy rollout engine identity/version drifted")
    if _plain_json(metadata["policy"]) != expected_policy:
        raise ValueError("M6 NumPy rollout policy metadata/configuration drifted")
    dynamics = metadata["dynamics"]
    if not isinstance(dynamics, dict) and not hasattr(dynamics, "items"):
        raise ValueError("M6 NumPy rollout dynamics metadata is invalid")
    dynamics_plain = _plain_json(dynamics)
    if not isinstance(dynamics_plain, dict) or set(dynamics_plain) != {
        "name",
        "version",
        "integration",
        "limits",
        "clamp_counts",
    }:
        raise ValueError("M6 NumPy rollout dynamics metadata fields drifted")
    for key, expected in _CANONICAL_DYNAMICS_CONFIGURATION.items():
        if dynamics_plain[key] != expected:
            raise ValueError(
                f"M6 NumPy rollout dynamics {key} drifted"
            )
    if metadata["ego_control"] != "typed_ego_plan":
        raise ValueError("M6 pair rollout must use typed_ego_plan")


def _validate_numpy_pair_metadata(
    pair: CounterfactualPair,
    expected_policy: dict[str, object],
) -> None:
    _validate_numpy_rollout_metadata(pair.baseline, expected_policy)
    _validate_numpy_rollout_metadata(pair.intervention, expected_policy)


def _assert_registered_nonreactivity(
    pair: CounterfactualPair,
    *,
    cohort_index: int,
    policy_name: str,
    condition: str,
) -> None:
    if (
        policy_name in _EXACT_NONREACTIVE_POLICIES
        and not is_exactly_nonreactive(pair)
    ):
        raise M6EvaluationError(
            "m6_registered_nonreactivity_gate_failed: "
            f"cohort_index={cohort_index}, policy={policy_name}, "
            f"condition={condition}"
        )


def _scenario_snapshots_exact(
    left: ScenarioSnapshot,
    right: ScenarioSnapshot,
) -> bool:
    left.revalidate()
    right.revalidate()
    return left._integrity_fingerprint == right._integrity_fingerprint


def _assert_fresh_execution_inputs_unchanged(
    source_before: ScenarioSnapshot,
    source_after_value: Scenario,
    *,
    original_plan: EgoTrajectoryPlan | None,
    call_plan: EgoTrajectoryPlan | None,
    cohort_index: int,
    policy_name: str,
    condition: str,
) -> None:
    try:
        source_after = ScenarioSnapshot.from_scenario(source_after_value)
        unchanged = _scenario_snapshots_exact(source_before, source_after)
        if original_plan is not None:
            original_plan.revalidate()
        if call_plan is not None:
            call_plan.revalidate()
        plan_unchanged = (
            original_plan is None
            and call_plan is None
        ) or (
            original_plan is not None
            and call_plan is not None
            and call_plan.serialize() == original_plan.serialize()
        )
    except Exception as exc:
        raise M6EvaluationError(
            "m6_executor_mutated_defensive_input: "
            f"cohort_index={cohort_index}, policy={policy_name}, "
            f"condition={condition}"
        ) from exc
    if not unchanged or not plan_unchanged:
        raise M6EvaluationError(
            "m6_executor_mutated_defensive_input: "
            f"cohort_index={cohort_index}, policy={policy_name}, "
            f"condition={condition}"
        )


def _assert_caller_sources_unchanged(
    cases: Sequence[_ValidatedCase],
) -> None:
    for case in cases:
        try:
            current = ScenarioSnapshot.from_scenario(case.caller_scenario)
            unchanged = _scenario_snapshots_exact(
                case.source_snapshot,
                current,
            )
        except Exception as exc:
            raise M6EvaluationError(
                "m6_caller_source_mutated_during_evaluation: "
                f"cohort_index={case.cohort_index}"
            ) from exc
        if not unchanged:
            raise M6EvaluationError(
                "m6_caller_source_mutated_during_evaluation: "
                f"cohort_index={case.cohort_index}"
            )


def _eligibility_exact(
    left: InterventionEligibility,
    right: InterventionEligibility,
) -> bool:
    return left.to_dict() == right.to_dict()


@lru_cache(maxsize=8)
def _reanalyzed_matrix_signature(
    cells: tuple[M6PrimaryCellInput, ...],
) -> tuple[str, str]:
    """Cache only an immutable exact signature after a full trusted reanalysis."""

    matrix = analyze_m6_primary_matrix(cells)
    return (
        matrix.intervention_config_fingerprint,
        json.dumps(
            matrix.to_local_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def _validate_evaluation_result_integrity(
    result: M6EvaluationResult,
) -> None:
    ledger = result.eligibility_ledger
    if not isinstance(ledger, M6EligibilityLedger):
        raise TypeError("eligibility_ledger must be M6EligibilityLedger")
    # Reconstruct every entry so post-construction field bypasses are caught.
    reconstructed_ledger = M6EligibilityLedger(
        tuple(
            M6EligibilityLedgerEntry(
                cohort_index=entry.cohort_index,
                eligibility=entry.eligibility,
                source_snapshot=entry.source_snapshot,
            )
            for entry in tuple(ledger.entries)
        )
    )
    if reconstructed_ledger.eligible_n < M6_MINIMUM_PRIMARY_PAIR_N:
        raise ValueError("an outcome result requires at least ten eligible pairs")

    primary = tuple(result.primary_scene_results)
    expected_order = tuple(
        (cohort_index, role.policy_name)
        for cohort_index in reconstructed_ledger.eligible_indices
        for role in M6_PRIMARY_POLICY_ROLES
    )
    if tuple(
        (scene.cohort_index, scene.policy_name) for scene in primary
    ) != expected_order:
        raise ValueError("primary scene results are not in canonical order")
    for scene in primary:
        if not isinstance(scene, M6PairedSceneResult):
            raise TypeError("primary scene result is not typed")
        scene.revalidate()
        if scene.pair.intervention_plan.spec.dose != (
            PRIMARY_BRAKE_MAGNITUDE_MPS2
        ):
            raise ValueError("primary scene result is not the exact b=2 plan")

    primary_by_index = {
        cohort_index: tuple(
            scene for scene in primary if scene.cohort_index == cohort_index
        )
        for cohort_index in reconstructed_ledger.eligible_indices
    }
    for cohort_index, scenes in primary_by_index.items():
        entry = reconstructed_ledger.entry_for(cohort_index)
        entry.revalidate()
        reference_source = entry.source_snapshot
        reference_source.revalidate()
        recomputed_eligibility = evaluate_primary_brake_eligibility(
            reference_source.to_scenario()
        )
        if not _eligibility_exact(entry.eligibility, recomputed_eligibility):
            raise ValueError(
                "eligibility ledger drifted from independent source recomputation"
            )
        source = reference_source.to_scenario()
        expected_identity = compile_identity_plan(source)
        expected_b2 = compile_longitudinal_brake_pulse_plan(
            source,
            PRIMARY_BRAKE_MAGNITUDE_MPS2,
        )
        for scene in scenes:
            if not _scenario_snapshots_exact(
                scene.pair.scenario,
                reference_source,
            ):
                raise ValueError(
                    "cross-policy source snapshots drifted within one cohort case"
                )
            if not _eligibility_exact(
                scene.pair.eligibility,
                entry.eligibility,
            ):
                raise ValueError(
                    "scene pair eligibility drifted from its exact ledger entry"
                )
            if scene.pair.baseline_plan.serialize() != (
                expected_identity.serialize()
            ):
                raise ValueError(
                    "cross-policy identity plan/audit serialization drifted"
                )
            if scene.pair.intervention_plan.serialize() != expected_b2.serialize():
                raise ValueError(
                    "cross-policy b=2 plan/audit serialization drifted"
                )

    secondary_ledger = tuple(
        M6SecondaryPlanEntry(
            cohort_index=entry.cohort_index,
            feasible=entry.feasible,
            reason=entry.reason,
        )
        for entry in tuple(result.secondary_plan_ledger)
    )
    secondary = tuple(result.secondary_scene_results)
    if secondary_ledger:
        if tuple(entry.cohort_index for entry in secondary_ledger) != (
            reconstructed_ledger.eligible_indices
        ):
            raise ValueError(
                "secondary plan ledger must cover the complete eligible cohort"
            )
        if any(
            not isinstance(entry, M6SecondaryPlanEntry)
            for entry in secondary_ledger
        ):
            raise TypeError("secondary plan ledger entries must be typed")
        expected_secondary_order = tuple(
            (entry.cohort_index, role.policy_name)
            for entry in secondary_ledger
            if entry.feasible
            for role in M6_PRIMARY_POLICY_ROLES
        )
        if tuple(
            (scene.cohort_index, scene.policy_name) for scene in secondary
        ) != expected_secondary_order:
            raise ValueError("secondary scene results are not the exact frozen subset")
        for primary_scene in primary:
            if len(primary_scene.secondary_metric_results) != len(
                _SECONDARY_METRIC_IDENTITIES
            ):
                raise ValueError(
                    "local secondary mode requires complete b=2 diagnostics"
                )
        for entry in secondary_ledger:
            source_snapshot = reconstructed_ledger.entry_for(
                entry.cohort_index
            ).source_snapshot
            try:
                expected_b4 = compile_longitudinal_brake_pulse_plan(
                    source_snapshot.to_scenario(),
                    SECONDARY_BRAKE_MAGNITUDE_MPS2,
                )
            except InterventionCompilationError as exc:
                if (
                    exc.code != "secondary_ego_plan_infeasible"
                    or entry.feasible
                    or entry.reason != exc.code
                ):
                    raise ValueError(
                        "secondary feasibility ledger drifted from recompilation"
                    ) from exc
                continue
            if not entry.feasible or entry.reason is not None:
                raise ValueError(
                    "secondary feasibility ledger drifted from recompilation"
                )
            cohort_secondary = tuple(
                scene
                for scene in secondary
                if scene.cohort_index == entry.cohort_index
            )
            if len(cohort_secondary) != len(M6_PRIMARY_POLICY_ROLES):
                raise ValueError("feasible b=4 case requires all three policy pairs")
            expected_identity = compile_identity_plan(
                source_snapshot.to_scenario()
            )
            ledger_entry = reconstructed_ledger.entry_for(entry.cohort_index)
            for scene in cohort_secondary:
                scene.revalidate()
                if not _scenario_snapshots_exact(
                    scene.pair.scenario,
                    source_snapshot,
                ):
                    raise ValueError(
                        "b=4 cross-policy source snapshots drifted"
                    )
                if not _eligibility_exact(
                    scene.pair.eligibility,
                    ledger_entry.eligibility,
                ):
                    raise ValueError(
                        "b=4 pair eligibility drifted from the ledger"
                    )
                if scene.pair.baseline_plan.serialize() != (
                    expected_identity.serialize()
                ) or scene.pair.intervention_plan.serialize() != (
                    expected_b4.serialize()
                ):
                    raise ValueError(
                        "cross-policy b=4 plan/audit serialization drifted"
                    )
                if len(scene.secondary_metric_results) != len(
                    _SECONDARY_METRIC_IDENTITIES
                ):
                    raise ValueError(
                        "b=4 result requires complete secondary diagnostics"
                    )
    elif secondary:
        raise ValueError("secondary results require a source-only plan ledger")
    elif any(scene.secondary_metric_results for scene in primary):
        raise ValueError(
            "b=2 secondary diagnostics require the separate local mode"
        )

    fingerprints = {
        scene.pair.intervention_plan.configuration_fingerprint
        for scene in primary
    }
    if len(fingerprints) != 1:
        raise ValueError("primary b=2 configuration fingerprints drifted")
    primary_fingerprint = next(iter(fingerprints))
    recomputed_effects: dict[tuple[str, str], list[M6SceneEffect]] = {
        (role.policy_name, metric.metric_name): []
        for role in M6_PRIMARY_POLICY_ROLES
        for metric in M6_PRIMARY_METRICS
    }
    for scene in primary:
        fresh_metrics = _recompute_metric_results(
            scene.pair,
            m6_primary_paired_metrics(),
        )
        for metric in fresh_metrics:
            recomputed_effects[(scene.policy_name, metric.metric_name)].append(
                _scene_effect(scene.cohort_index, metric)
            )
    expected_cells = tuple(
        M6PrimaryCellInput(
            spec=spec,
            scene_effects=tuple(
                recomputed_effects[(spec.policy_name, spec.metric_name)]
            ),
            source_pairing_complete=True,
        )
        for spec in m6_primary_cell_specs(primary_fingerprint)
    )
    actual_cells = tuple(result.primary_cell_inputs)
    if len(actual_cells) != len(expected_cells) or any(
        not isinstance(cell, M6PrimaryCellInput)
        for cell in actual_cells
    ):
        raise ValueError("primary_cell_inputs must contain the exact 12 cells")
    for actual, expected in zip(actual_cells, expected_cells, strict=True):
        if (
            actual.spec != expected.spec
            or actual.source_pairing_complete is not True
            or actual.scene_effects != expected.scene_effects
        ):
            raise ValueError(
                "primary cell inputs drifted from independently recomputed metrics"
            )
    expected_fingerprint, expected_matrix_json = _reanalyzed_matrix_signature(
        expected_cells
    )
    actual_matrix = result.primary_matrix
    if not isinstance(actual_matrix, M6PrimaryMatrixResult) or (
        actual_matrix.intervention_config_fingerprint
        != expected_fingerprint
    ) or json.dumps(
        actual_matrix.to_local_dict(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ) != expected_matrix_json:
        raise ValueError(
            "primary matrix drifted from independent exact 12-cell reanalysis"
        )


def _scene_effect(
    cohort_index: int,
    result: PairedMetricResult,
) -> M6SceneEffect:
    if result.metric_name != M6_RESPONSE_TIMELINESS_METRIC:
        return M6SceneEffect(
            cohort_index=cohort_index,
            value=result.value,
        )
    responded = result.details.get("responded")
    censored = result.details.get("censored")
    event_time = result.details.get("event_time_s")
    if type(responded) is not bool or type(censored) is not bool or (
        censored == responded
    ):
        raise M6EvaluationError(
            "m6_timeliness_responder_censor_accounting_invalid: "
            f"cohort_index={cohort_index}"
        )
    if responded:
        if (
            isinstance(event_time, (bool, np.bool_))
            or not isinstance(event_time, (int, float, np.integer, np.floating))
            or not np.isfinite(float(event_time))
            or float(event_time) < 0.0
        ):
            raise M6EvaluationError(
                "m6_timeliness_responder_latency_invalid: "
                f"cohort_index={cohort_index}"
            )
        latency: float | None = float(event_time)
    else:
        if event_time is not None:
            raise M6EvaluationError(
                "m6_censored_timeliness_must_not_have_event_time: "
                f"cohort_index={cohort_index}"
            )
        latency = None
    return M6SceneEffect(
        cohort_index=cohort_index,
        value=result.value,
        responded=responded,
        responder_latency_s=latency,
    )


__all__ = [
    "M6_MINIMUM_PRIMARY_PAIR_N",
    "M6_NUMPY_POLICY_ACCESS_ROLES",
    "M6_NUMPY_POLICY_ORDER",
    "M6_NUMPY_SEED",
    "M6EligibilityLedger",
    "M6EligibilityLedgerEntry",
    "M6EvaluationCase",
    "M6EvaluationError",
    "M6EvaluationResult",
    "M6PairedSceneResult",
    "M6PrimaryOutcomeBlocked",
    "M6SecondaryPlanEntry",
    "assert_m6_sham_matches_legacy_prefix",
    "canonical_m6_policies",
    "evaluate_m6_source_eligibility",
    "run_m6_numpy_evaluation",
]
