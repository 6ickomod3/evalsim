"""Source-neutral M5 evaluation runner and fixed synthetic acceptance adapter.

The generic runner sees only :class:`EvaluationCase`, :class:`Scenario`, and
:class:`Rollout` contracts.  It deliberately never reads ``reference_payload``;
source-specific executors may use that opaque value behind the typed executor seam.
No Waymax, JAX, TensorFlow, dataset, or result-store import is needed here.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from numbers import Integral
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from evalsim.contracts import (
    Metric,
    MetricEligibility,
    MetricResult,
    PolicyMetadata,
    Rollout,
    Scenario,
    SimulatorPolicy,
)
from evalsim.metrics.m5 import M5_METRIC_SPECS, m5_metrics
from evalsim.rollout import RolloutEngine
from evalsim.simulators import (
    ConstantVelocityPolicy,
    IDMPolicy,
    LogReplayPolicy,
)
from evalsim.slices.m5 import (
    M5_SLICE_SPECS,
    SliceResult,
    evaluate_m5_slices,
)
from evalsim.sources.synthetic import SyntheticSource
from evalsim.stats.m5 import (
    M5_POLICY_CONTRASTS,
    PairedCellResult,
    PairedCellSpec,
    ScenarioScalar,
    analyze_paired_cell,
)


M5_SYNTHETIC_CASE_COUNT = 5
M5_SYNTHETIC_SEED = 20_260_728
M5_SYNTHETIC_NUM_STEPS = 91
M5_SYNTHETIC_CURRENT_INDEX = 10
M5_SYNTHETIC_SPLIT = "m5_data_free"
M5_SYNTHETIC_ADAPTER_VERSION = "m5-synthetic-adapter-1.0.0"
M5_SYNTHETIC_ORDER_VERSION = "m5-synthetic-order-1"

M5_POLICY_NAMES = (
    "constant_velocity",
    "idm",
    "log_replay",
)
M5_ERROR_ORACLE_METRICS = (
    "acceleration_error_mps2",
    "jerk_error_mps3",
    "position_error_m",
    "speed_error_mps",
    "yaw_rate_error_radps",
)

_METRIC_ROW_FIELDS = frozenset(
    {
        "cohort_index",
        "details_json",
        "distribution",
        "eligible_components",
        "execution_name",
        "execution_role",
        "invalid_reason",
        "metric_name",
        "metric_version",
        "seed",
        "total_components",
        "valid",
        "value",
    }
)
_SLICE_ROW_FIELDS = frozenset(
    {
        "cohort_index",
        "eligible",
        "member",
        "reason",
        "slice_name",
        "slice_version",
    }
)


class M5EvaluationError(RuntimeError):
    """A source-neutral M5 orchestration contract failed closed."""


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One canonical cohort member plus executor-private source payload."""

    cohort_index: int
    scenario: Scenario = field(repr=False, compare=False)
    reference_payload: Any = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        index = _nonnegative_integer(self.cohort_index, "cohort_index")
        if not isinstance(self.scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        object.__setattr__(self, "cohort_index", index)


@runtime_checkable
class CohortAdapter(Protocol):
    """Adapt a source into the canonical, already-selected evaluation order."""

    def cases(self) -> Iterable[EvaluationCase]:
        """Return each accepted case exactly once in cohort-index order."""


@runtime_checkable
class PolicyExecutor(Protocol):
    """Execute one policy while keeping source-specific state behind this seam."""

    def execute(
        self,
        case: EvaluationCase,
        policy: SimulatorPolicy,
        seed: int,
    ) -> Rollout:
        """Return the source-neutral rollout for one case and policy."""


@dataclass(frozen=True, slots=True)
class NumpyPolicyExecutor:
    """Execute EvalSim policies through the contract-owned NumPy engine."""

    engine: RolloutEngine = field(default_factory=RolloutEngine)

    def __post_init__(self) -> None:
        if not isinstance(self.engine, RolloutEngine):
            raise TypeError("engine must be a RolloutEngine")

    def execute(
        self,
        case: EvaluationCase,
        policy: SimulatorPolicy,
        seed: int,
    ) -> Rollout:
        if not isinstance(case, EvaluationCase):
            raise TypeError("case must be an EvaluationCase")
        if not isinstance(policy, SimulatorPolicy):
            raise TypeError("policy must be a SimulatorPolicy")
        return self.engine.run(case.scenario, policy, seed=seed)


@dataclass(frozen=True, slots=True)
class SyntheticM5CohortAdapter:
    """The exact five-family, 91-frame data-free M5 fixture."""

    def cases(self) -> tuple[EvaluationCase, ...]:
        source = _synthetic_source()
        scenarios = source.generate(M5_SYNTHETIC_CASE_COUNT)
        cases: list[EvaluationCase] = []
        for cohort_index, scenario in enumerate(scenarios):
            scenario.metadata["current_index"] = M5_SYNTHETIC_CURRENT_INDEX
            cases.append(
                EvaluationCase(
                    cohort_index=cohort_index,
                    scenario=scenario,
                )
            )
        return tuple(cases)


@dataclass(frozen=True, slots=True)
class ScorecardCellInput:
    """Exact paired scenario scalars for one registered scorecard cell."""

    spec: PairedCellSpec
    values_a: tuple[ScenarioScalar, ...] = field(repr=False)
    values_b: tuple[ScenarioScalar, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, PairedCellSpec):
            raise TypeError("spec must be a PairedCellSpec")
        values_a = tuple(self.values_a)
        values_b = tuple(self.values_b)
        if any(not isinstance(value, ScenarioScalar) for value in values_a):
            raise TypeError("values_a must contain ScenarioScalar values")
        if any(not isinstance(value, ScenarioScalar) for value in values_b):
            raise TypeError("values_b must contain ScenarioScalar values")
        indices_a = tuple(value.cohort_index for value in values_a)
        indices_b = tuple(value.cohort_index for value in values_b)
        if (
            indices_a != tuple(sorted(set(indices_a)))
            or indices_b != tuple(sorted(set(indices_b)))
            or indices_a != indices_b
        ):
            raise ValueError(
                "scorecard inputs must retain one identical canonical index set"
            )
        object.__setattr__(self, "values_a", values_a)
        object.__setattr__(self, "values_b", values_b)


@dataclass(frozen=True, slots=True)
class LogReplayZeroOracle:
    """Aggregate-safe evidence for one exact-log error-metric check."""

    cohort_index: int
    metric_name: str
    metric_version: str
    checked_components: int
    scalar_value: float
    maximum_absolute_component: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cohort_index",
            _nonnegative_integer(self.cohort_index, "cohort_index"),
        )
        if self.metric_name not in M5_ERROR_ORACLE_METRICS:
            raise ValueError("metric_name is not an M5 exact-log zero oracle")
        spec = M5_METRIC_SPECS[self.metric_name]
        if self.metric_version != spec.version:
            raise ValueError("metric_version differs from the M5 registry")
        components = _nonnegative_integer(
            self.checked_components,
            "checked_components",
        )
        if components < 1:
            raise ValueError("a zero oracle requires a checked component")
        if (
            type(self.scalar_value) is not float
            or type(self.maximum_absolute_component) is not float
            or self.scalar_value != 0.0
            or self.maximum_absolute_component != 0.0
        ):
            raise ValueError("log-replay oracle evidence must be exact zero")
        object.__setattr__(self, "checked_components", components)


@dataclass(frozen=True, slots=True)
class M5EvaluationResult:
    """Complete in-memory M5 rows and paired scorecard inputs."""

    cases: tuple[EvaluationCase, ...] = field(repr=False)
    metric_rows: tuple[Mapping[str, Any], ...] = field(repr=False)
    slice_rows: tuple[Mapping[str, Any], ...] = field(repr=False)
    scorecard_inputs: tuple[ScorecardCellInput, ...] = field(repr=False)
    scorecard_results: tuple[PairedCellResult, ...] = field(repr=False)
    zero_oracles: tuple[LogReplayZeroOracle, ...] = field(repr=False)

    def __post_init__(self) -> None:
        cases = tuple(self.cases)
        metric_rows = tuple(self.metric_rows)
        slice_rows = tuple(self.slice_rows)
        inputs = tuple(self.scorecard_inputs)
        results = tuple(self.scorecard_results)
        oracles = tuple(self.zero_oracles)
        if any(not isinstance(case, EvaluationCase) for case in cases):
            raise TypeError("cases must contain EvaluationCase values")
        if len(metric_rows) != len(cases) * len(M5_POLICY_NAMES) * len(
            M5_METRIC_SPECS
        ):
            raise ValueError("metric row count does not match the M5 case matrix")
        if len(slice_rows) != len(cases) * len(M5_SLICE_SPECS):
            raise ValueError("slice row count does not match the M5 case matrix")
        expected_cells = (
            len(M5_METRIC_SPECS)
            * len(M5_SLICE_SPECS)
            * len(M5_POLICY_CONTRASTS)
        )
        if len(inputs) != expected_cells or len(results) != expected_cells:
            raise ValueError("scorecard cell count is not the fixed M5 matrix")
        if any(not isinstance(row, Mapping) for row in metric_rows):
            raise TypeError("metric_rows must contain mappings")
        if any(set(row) != _METRIC_ROW_FIELDS for row in metric_rows):
            raise ValueError("metric row fields differ from the M5 store schema")
        if any(not isinstance(row, Mapping) for row in slice_rows):
            raise TypeError("slice_rows must contain mappings")
        if any(set(row) != _SLICE_ROW_FIELDS for row in slice_rows):
            raise ValueError("slice row fields differ from the M5 store schema")
        if any(not isinstance(item, ScorecardCellInput) for item in inputs):
            raise TypeError("scorecard_inputs have the wrong type")
        if any(not isinstance(item, PairedCellResult) for item in results):
            raise TypeError("scorecard_results have the wrong type")
        if tuple(item.spec for item in inputs) != tuple(
            item.spec for item in results
        ):
            raise ValueError("scorecard inputs and results have different cells")
        if any(not isinstance(item, LogReplayZeroOracle) for item in oracles):
            raise TypeError("zero_oracles have the wrong type")
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "metric_rows", metric_rows)
        object.__setattr__(self, "slice_rows", slice_rows)
        object.__setattr__(self, "scorecard_inputs", inputs)
        object.__setattr__(self, "scorecard_results", results)
        object.__setattr__(self, "zero_oracles", oracles)

    @property
    def case_count(self) -> int:
        return len(self.cases)


def canonical_m5_policies() -> tuple[SimulatorPolicy, ...]:
    """Return fresh instances of the three frozen EvalSim comparison policies."""

    return (
        ConstantVelocityPolicy(),
        IDMPolicy(),
        LogReplayPolicy(),
    )


def synthetic_source_evidence() -> Mapping[str, Any]:
    """Return the fixed synthetic config and canonical case order for provenance."""

    source = _synthetic_source()
    scenarios = source.generate(M5_SYNTHETIC_CASE_COUNT)
    return MappingProxyType(
        {
            "adapter_version": M5_SYNTHETIC_ADAPTER_VERSION,
            "case_count": M5_SYNTHETIC_CASE_COUNT,
            "current_index": M5_SYNTHETIC_CURRENT_INDEX,
            "dt_seconds": 0.1,
            "num_steps": M5_SYNTHETIC_NUM_STEPS,
            "order_version": M5_SYNTHETIC_ORDER_VERSION,
            "ordered_scenario_ids": tuple(
                scenario.scenario_id for scenario in scenarios
            ),
            "seed": M5_SYNTHETIC_SEED,
            "source_fingerprint": source.fingerprint,
            "split": M5_SYNTHETIC_SPLIT,
        }
    )


def run_m5_evaluation(
    adapter: CohortAdapter,
    executor: PolicyExecutor,
    *,
    policies: Sequence[SimulatorPolicy] | None = None,
    seed: int = 0,
) -> M5EvaluationResult:
    """Evaluate one already-selected cohort without inspecting source payloads."""

    if not isinstance(adapter, CohortAdapter):
        raise TypeError("adapter must implement CohortAdapter")
    if not isinstance(executor, PolicyExecutor):
        raise TypeError("executor must implement PolicyExecutor")
    normalized_seed = _nonnegative_integer(seed, "seed")
    if normalized_seed > 2**32 - 1:
        raise ValueError("seed must fit uint32")
    cases = _validated_cases(adapter.cases())
    policy_entries = _validated_policies(
        canonical_m5_policies() if policies is None else policies
    )
    metrics = _validated_metrics(m5_metrics())

    eligibility: dict[tuple[int, str], MetricEligibility] = {}
    slices: dict[tuple[int, str], SliceResult] = {}
    slice_rows: list[Mapping[str, Any]] = []
    for case in cases:
        for metric in metrics:
            decision = metric.eligibility(case.scenario)
            _validate_eligibility(metric, decision)
            eligibility[(case.cohort_index, metric.spec.name)] = decision
        observed_slices = evaluate_m5_slices(case.scenario)
        _validate_slice_results(case, observed_slices)
        for result in observed_slices:
            slices[(case.cohort_index, result.slice_name)] = result
            slice_rows.append(
                _row(
                    {
                        "cohort_index": case.cohort_index,
                        "eligible": result.eligible,
                        "member": result.member,
                        "reason": result.reason,
                        "slice_name": result.slice_name,
                        "slice_version": result.slice_version,
                    }
                )
            )

    metric_rows: list[Mapping[str, Any]] = []
    results_by_key: dict[tuple[int, str, str], MetricResult] = {}
    zero_oracles: list[LogReplayZeroOracle] = []
    for case in cases:
        pairing: dict[str, tuple[tuple[Any, ...], ...]] = {}
        for policy_name, policy_version, policy in policy_entries:
            rollout = executor.execute(case, policy, normalized_seed)
            _validate_rollout_identity(
                case,
                rollout,
                policy_name=policy_name,
                policy_version=policy_version,
                seed=normalized_seed,
            )
            signatures: list[tuple[Any, ...]] = []
            for metric in metrics:
                result = metric.compute(case.scenario, rollout)
                decision = eligibility[
                    (case.cohort_index, metric.spec.name)
                ]
                _validate_metric_result(case, metric, decision, result)
                signatures.append(_pairing_signature(result))
                results_by_key[
                    (case.cohort_index, policy_name, metric.spec.name)
                ] = result
                metric_rows.append(
                    _metric_store_row(
                        case.cohort_index,
                        policy_name,
                        normalized_seed,
                        result,
                    )
                )
                if (
                    policy_name == "log_replay"
                    and metric.spec.name in M5_ERROR_ORACLE_METRICS
                ):
                    zero_oracles.append(
                        _log_replay_zero_oracle(
                            case.cohort_index,
                            result,
                        )
                    )
            pairing[policy_name] = tuple(signatures)
        if len(set(pairing.values())) != 1:
            raise M5EvaluationError(
                "policy metric eligibility or component pairing drifted"
            )

    scorecard_inputs, scorecard_results = _scorecards(
        cases,
        slices,
        results_by_key,
    )
    metric_rows.sort(
        key=lambda row: (
            row["cohort_index"],
            row["execution_name"],
            row["seed"],
            row["metric_name"],
            row["metric_version"],
        )
    )
    slice_rows.sort(
        key=lambda row: (
            row["cohort_index"],
            row["slice_name"],
            row["slice_version"],
        )
    )
    zero_oracles.sort(
        key=lambda item: (item.cohort_index, item.metric_name)
    )
    return M5EvaluationResult(
        cases=cases,
        metric_rows=tuple(metric_rows),
        slice_rows=tuple(slice_rows),
        scorecard_inputs=scorecard_inputs,
        scorecard_results=scorecard_results,
        zero_oracles=tuple(zero_oracles),
    )


def run_synthetic_m5_evaluation(
    *,
    executor: PolicyExecutor | None = None,
) -> M5EvaluationResult:
    """Run the exact data-free five-scenario M5 acceptance fixture."""

    result = run_m5_evaluation(
        SyntheticM5CohortAdapter(),
        NumpyPolicyExecutor() if executor is None else executor,
    )
    if (
        result.case_count != M5_SYNTHETIC_CASE_COUNT
        or len(result.metric_rows) != 195
        or len(result.slice_rows) != 40
        or len(result.scorecard_inputs) != 312
        or len(result.scorecard_results) != 312
        or len(result.zero_oracles)
        != M5_SYNTHETIC_CASE_COUNT * len(M5_ERROR_ORACLE_METRICS)
    ):
        raise M5EvaluationError(
            "the fixed synthetic M5 acceptance matrix is incomplete"
        )
    return result


def _validated_cases(
    raw_cases: Iterable[EvaluationCase],
) -> tuple[EvaluationCase, ...]:
    if isinstance(raw_cases, (str, bytes, Mapping)):
        raise TypeError("adapter cases must be an iterable of EvaluationCase")
    cases = tuple(raw_cases)
    if not cases or any(not isinstance(case, EvaluationCase) for case in cases):
        raise M5EvaluationError(
            "the evaluation cohort must contain typed cases"
        )
    indices = tuple(case.cohort_index for case in cases)
    if indices != tuple(range(len(cases))):
        raise M5EvaluationError(
            "evaluation cases must retain contiguous canonical cohort order"
        )
    scenario_ids = tuple(case.scenario.scenario_id for case in cases)
    if (
        any(
            not isinstance(scenario_id, str) or not scenario_id
            for scenario_id in scenario_ids
        )
        or len(set(scenario_ids)) != len(scenario_ids)
    ):
        raise M5EvaluationError(
            "evaluation cases must have unique nonempty scenario identities"
        )
    return cases


def _synthetic_source() -> SyntheticSource:
    return SyntheticSource(
        seed=M5_SYNTHETIC_SEED,
        num_steps=M5_SYNTHETIC_NUM_STEPS,
        dt=0.1,
        split=M5_SYNTHETIC_SPLIT,
    )


def _validated_policies(
    raw_policies: Sequence[SimulatorPolicy],
) -> tuple[tuple[str, str, SimulatorPolicy], ...]:
    if isinstance(raw_policies, (str, bytes)) or not isinstance(
        raw_policies,
        Sequence,
    ):
        raise TypeError("policies must be a sequence")
    entries: list[tuple[str, str, SimulatorPolicy]] = []
    for policy in raw_policies:
        if not isinstance(policy, SimulatorPolicy):
            raise TypeError("policies must contain SimulatorPolicy values")
        metadata = policy.metadata()
        if not isinstance(metadata, PolicyMetadata):
            raise TypeError("policy metadata must be PolicyMetadata")
        if metadata.deterministic is not True:
            raise M5EvaluationError("M5 policies must be deterministic")
        entries.append((metadata.name, metadata.version, policy))
    entries.sort(key=lambda item: item[0])
    if tuple(item[0] for item in entries) != M5_POLICY_NAMES:
        raise M5EvaluationError(
            "M5 requires exactly constant_velocity, idm, and log_replay"
        )
    return tuple(entries)


def _validated_metrics(
    metrics: Sequence[Metric],
) -> tuple[Metric, ...]:
    normalized = tuple(metrics)
    if any(not isinstance(metric, Metric) for metric in normalized):
        raise TypeError("M5 registry must contain Metric values")
    observed = tuple(
        (metric.spec.name, metric.spec.version) for metric in normalized
    )
    expected = tuple(
        (spec.name, spec.version) for spec in M5_METRIC_SPECS.values()
    )
    if observed != expected:
        raise M5EvaluationError(
            "M5 metric registry order or identity drifted"
        )
    return normalized


def _validate_eligibility(
    metric: Metric,
    decision: MetricEligibility,
) -> None:
    if not isinstance(decision, MetricEligibility):
        raise TypeError("metric eligibility must be MetricEligibility")
    if (
        not decision.eligible
        and decision.reason not in metric.spec.invalid_reason_codes
    ):
        raise M5EvaluationError(
            "metric eligibility used an unregistered source reason"
        )


def _validate_slice_results(
    case: EvaluationCase,
    results: Sequence[SliceResult],
) -> None:
    expected = tuple((spec.name, spec.version) for spec in M5_SLICE_SPECS)
    observed = tuple(
        (result.slice_name, result.slice_version) for result in results
    )
    if (
        len(results) != len(M5_SLICE_SPECS)
        or any(not isinstance(result, SliceResult) for result in results)
        or observed != expected
        or any(result.scenario_id != case.scenario.scenario_id for result in results)
    ):
        raise M5EvaluationError(
            "slice evaluation lost canonical identity or coverage"
        )


def _validate_rollout_identity(
    case: EvaluationCase,
    rollout: Rollout,
    *,
    policy_name: str,
    policy_version: str,
    seed: int,
) -> None:
    scenario = case.scenario
    if not isinstance(rollout, Rollout):
        raise TypeError("policy executor must return a Rollout")
    if (
        rollout.scenario_id != scenario.scenario_id
        or rollout.sim_name != policy_name
        or rollout.sim_version != policy_version
        or rollout.seed != seed
        or rollout.perturbation is not None
        or rollout.num_steps != scenario.num_steps
        or rollout.num_agents != scenario.num_agents
        or not np.array_equal(rollout.timestamps, scenario.timestamps)
    ):
        raise M5EvaluationError(
            "policy rollout identity or provenance does not match its case"
        )
    for source, candidate in zip(
        scenario.agents,
        rollout.agents,
        strict=True,
    ):
        if (
            source.id != candidate.id
            or source.type != candidate.type
            or source.length != candidate.length
            or source.width != candidate.width
            or not np.array_equal(source.valid, candidate.valid)
        ):
            raise M5EvaluationError(
                "policy rollout agent identity or validity drifted"
            )


def _validate_metric_result(
    case: EvaluationCase,
    metric: Metric,
    decision: MetricEligibility,
    result: MetricResult,
) -> None:
    if not isinstance(result, MetricResult):
        raise TypeError("metric compute must return MetricResult")
    if (
        result.metric_name != metric.spec.name
        or result.metric_version != metric.spec.version
        or result.scenario_id != case.scenario.scenario_id
    ):
        raise M5EvaluationError(
            "metric result identity differs from its case or registry"
        )
    if result.valid != decision.eligible or (
        not result.valid and result.invalid_reason != decision.reason
    ):
        raise M5EvaluationError(
            "metric result contradicts source-only eligibility"
        )
    if (
        not result.valid
        and result.invalid_reason not in metric.spec.invalid_reason_codes
    ):
        raise M5EvaluationError(
            "metric result used an unregistered missingness reason"
        )


def _pairing_signature(result: MetricResult) -> tuple[Any, ...]:
    return (
        result.metric_name,
        result.metric_version,
        result.valid,
        result.invalid_reason,
        result.eligible_components,
        result.total_components,
    )


def _metric_store_row(
    cohort_index: int,
    policy_name: str,
    seed: int,
    result: MetricResult,
) -> Mapping[str, Any]:
    details_json = json.dumps(
        _thaw_json(result.details),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return _row(
        {
            "cohort_index": cohort_index,
            "details_json": details_json,
            "distribution": tuple(result.distribution),
            "eligible_components": result.eligible_components,
            "execution_name": policy_name,
            "execution_role": "policy",
            "invalid_reason": result.invalid_reason,
            "metric_name": result.metric_name,
            "metric_version": result.metric_version,
            "seed": seed,
            "total_components": result.total_components,
            "valid": result.valid,
            "value": result.value,
        }
    )


def _log_replay_zero_oracle(
    cohort_index: int,
    result: MetricResult,
) -> LogReplayZeroOracle:
    if (
        not result.valid
        or result.value != 0.0
        or not result.distribution
        or any(component != 0.0 for component in result.distribution)
    ):
        raise M5EvaluationError(
            "log replay failed an exact error-metric zero oracle"
        )
    maximum = max(abs(component) for component in result.distribution)
    return LogReplayZeroOracle(
        cohort_index=cohort_index,
        metric_name=result.metric_name,
        metric_version=result.metric_version,
        checked_components=result.eligible_components,
        scalar_value=float(result.value),
        maximum_absolute_component=float(maximum),
    )


def _scorecards(
    cases: tuple[EvaluationCase, ...],
    slices: Mapping[tuple[int, str], SliceResult],
    metric_results: Mapping[tuple[int, str, str], MetricResult],
) -> tuple[tuple[ScorecardCellInput, ...], tuple[PairedCellResult, ...]]:
    member_indices = {
        spec.name: tuple(
            case.cohort_index
            for case in cases
            if (
                slices[(case.cohort_index, spec.name)].eligible
                and slices[(case.cohort_index, spec.name)].member
            )
        )
        for spec in M5_SLICE_SPECS
    }
    inputs: list[ScorecardCellInput] = []
    results: list[PairedCellResult] = []
    for metric_name, metric_spec in M5_METRIC_SPECS.items():
        for slice_spec in M5_SLICE_SPECS:
            indices = member_indices[slice_spec.name]
            for contrast in M5_POLICY_CONTRASTS:
                spec = PairedCellSpec(
                    metric_name=metric_name,
                    metric_version=metric_spec.version,
                    slice_name=slice_spec.name,
                    slice_version=slice_spec.version,
                    contrast=contrast,
                )
                values_a = tuple(
                    _scenario_scalar(
                        cohort_index,
                        metric_results[
                            (
                                cohort_index,
                                contrast.policy_a,
                                metric_name,
                            )
                        ],
                    )
                    for cohort_index in indices
                )
                values_b = tuple(
                    _scenario_scalar(
                        cohort_index,
                        metric_results[
                            (
                                cohort_index,
                                contrast.policy_b,
                                metric_name,
                            )
                        ],
                    )
                    for cohort_index in indices
                )
                cell_input = ScorecardCellInput(
                    spec=spec,
                    values_a=values_a,
                    values_b=values_b,
                )
                inputs.append(cell_input)
                results.append(
                    analyze_paired_cell(spec, values_a, values_b)
                )
    return tuple(inputs), tuple(results)


def _scenario_scalar(
    cohort_index: int,
    result: MetricResult,
) -> ScenarioScalar:
    if result.valid:
        return ScenarioScalar(
            cohort_index=cohort_index,
            value=result.value,
            eligible_components=result.eligible_components,
            total_components=result.total_components,
        )
    return ScenarioScalar.missing(
        cohort_index,
        str(result.invalid_reason),
        total_components=result.total_components,
    )


def _row(payload: dict[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(payload)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


__all__ = [
    "M5_ERROR_ORACLE_METRICS",
    "M5_POLICY_NAMES",
    "M5_SYNTHETIC_ADAPTER_VERSION",
    "M5_SYNTHETIC_CASE_COUNT",
    "M5_SYNTHETIC_CURRENT_INDEX",
    "M5_SYNTHETIC_NUM_STEPS",
    "M5_SYNTHETIC_ORDER_VERSION",
    "M5_SYNTHETIC_SEED",
    "M5_SYNTHETIC_SPLIT",
    "CohortAdapter",
    "EvaluationCase",
    "LogReplayZeroOracle",
    "M5EvaluationError",
    "M5EvaluationResult",
    "NumpyPolicyExecutor",
    "PolicyExecutor",
    "ScorecardCellInput",
    "SyntheticM5CohortAdapter",
    "canonical_m5_policies",
    "run_m5_evaluation",
    "run_synthetic_m5_evaluation",
    "synthetic_source_evidence",
]
