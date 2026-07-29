"""Source-neutral M5 evaluation runner and fixed synthetic acceptance adapter.

The generic runner sees only :class:`EvaluationCase`, :class:`Scenario`, and
:class:`Rollout` contracts.  It deliberately never reads ``reference_payload``;
source-specific executors may use that opaque value behind the typed executor seam.
No Waymax, JAX, TensorFlow, dataset, or result-store import is needed here.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from numbers import Integral
import re
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

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
    CONSTANT_VELOCITY_VERSION,
    IDM_VERSION,
    LOG_REPLAY_VERSION,
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
_M5_POLICY_VERSIONS = MappingProxyType(
    {
        "constant_velocity": CONSTANT_VELOCITY_VERSION,
        "idm": IDM_VERSION,
        "log_replay": LOG_REPLAY_VERSION,
    }
)
M5_ERROR_ORACLE_METRICS = (
    "acceleration_error_mps2",
    "jerk_error_mps3",
    "position_error_m",
    "speed_error_mps",
    "yaw_rate_error_radps",
)

ExecutionRole = Literal["policy", "reference"]

_EXECUTION_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_EXECUTION_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+@-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CASE_METRIC_DIGEST_DOMAIN = "evalsim-m5-case-metric-pass-v1"
_COHORT_METRIC_DIGEST_DOMAIN = "evalsim-m5-cohort-metric-pass-v1"
_STATISTICS_DIGEST_DOMAIN = "evalsim-m5-statistics-pass-v1"
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


@dataclass(frozen=True, slots=True, order=True)
class ExecutionSpec:
    """Typed, source-neutral identity for one deterministic M5 execution."""

    name: str
    version: str
    role: ExecutionRole
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or _EXECUTION_NAME_PATTERN.fullmatch(self.name) is None
        ):
            raise ValueError(
                "execution name must be a non-empty lower_snake_case string"
            )
        if (
            not isinstance(self.version, str)
            or _EXECUTION_VERSION_PATTERN.fullmatch(self.version) is None
        ):
            raise ValueError(
                "execution version must be a bounded identifier-safe string"
            )
        if self.role not in ("policy", "reference"):
            raise ValueError("execution role must be 'policy' or 'reference'")
        normalized_seed = _nonnegative_integer(self.seed, "seed")
        if normalized_seed > 2**32 - 1:
            raise ValueError("seed must fit uint32")
        if self.role == "policy" and self.name not in M5_POLICY_NAMES:
            raise ValueError(
                "policy execution name must be registered in M5_POLICY_NAMES"
            )
        if (
            self.role == "policy"
            and self.version != _M5_POLICY_VERSIONS[self.name]
        ):
            raise ValueError(
                "policy execution version differs from the frozen M5 registry"
            )
        if self.role == "reference" and self.name in M5_POLICY_NAMES:
            raise ValueError(
                "a reference execution cannot reuse a registered policy name"
            )
        object.__setattr__(self, "seed", normalized_seed)


@dataclass(frozen=True, slots=True)
class ExecutionRollout:
    """One typed execution identity paired with its transient rollout."""

    spec: ExecutionSpec
    rollout: Rollout = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ExecutionSpec):
            raise TypeError("spec must be an ExecutionSpec")
        if not isinstance(self.rollout, Rollout):
            raise TypeError("rollout must be a Rollout")


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
class M5CaseEvaluation:
    """One transient case result suitable for immediate result-store writing.

    The object deliberately retains neither the source :class:`EvaluationCase` nor
    any :class:`Rollout`.  Callers may write ``metric_rows`` and then add this object
    to :class:`M5ScorecardAccumulator`, which keeps only compact scenario scalars,
    slice facts, and digests.
    """

    cohort_index: int
    execution_specs: tuple[ExecutionSpec, ...]
    metric_rows: tuple[Mapping[str, Any], ...] = field(repr=False)
    slice_rows: tuple[Mapping[str, Any], ...] = field(repr=False)
    policy_scalars: Mapping[tuple[str, str], ScenarioScalar] = field(
        repr=False
    )
    zero_oracles: tuple[LogReplayZeroOracle, ...] = field(repr=False)
    metric_pass_1_sha256: str = field(repr=False)
    metric_pass_2_sha256: str = field(repr=False)

    def __post_init__(self) -> None:
        cohort_index = _nonnegative_integer(
            self.cohort_index,
            "cohort_index",
        )
        specs = _validated_execution_specs(self.execution_specs)
        metric_rows = tuple(self.metric_rows)
        slice_rows = tuple(self.slice_rows)
        zero_oracles = tuple(self.zero_oracles)
        if any(not isinstance(row, Mapping) for row in metric_rows):
            raise TypeError("metric_rows must contain mappings")
        if any(set(row) != _METRIC_ROW_FIELDS for row in metric_rows):
            raise ValueError("metric row fields differ from the M5 store schema")
        expected_metric_keys = tuple(
            sorted(
                (
                    cohort_index,
                    spec.name,
                    spec.role,
                    spec.seed,
                    metric_spec.name,
                    metric_spec.version,
                )
                for spec in specs
                for metric_spec in M5_METRIC_SPECS.values()
            )
        )
        observed_metric_keys = tuple(
            (
                row["cohort_index"],
                row["execution_name"],
                row["execution_role"],
                row["seed"],
                row["metric_name"],
                row["metric_version"],
            )
            for row in metric_rows
        )
        if observed_metric_keys != expected_metric_keys:
            raise ValueError(
                "metric rows lost canonical execution, metric, or cohort order"
            )

        if any(not isinstance(row, Mapping) for row in slice_rows):
            raise TypeError("slice_rows must contain mappings")
        if any(set(row) != _SLICE_ROW_FIELDS for row in slice_rows):
            raise ValueError("slice row fields differ from the M5 store schema")
        expected_slice_keys = tuple(
            (
                cohort_index,
                spec.name,
                spec.version,
            )
            for spec in M5_SLICE_SPECS
        )
        observed_slice_keys = tuple(
            (
                row["cohort_index"],
                row["slice_name"],
                row["slice_version"],
            )
            for row in slice_rows
        )
        if observed_slice_keys != expected_slice_keys:
            raise ValueError("slice rows lost canonical identity or order")

        if not isinstance(self.policy_scalars, Mapping):
            raise TypeError("policy_scalars must be a mapping")
        policy_scalars = dict(self.policy_scalars)
        expected_scalar_keys = {
            (policy_name, metric_name)
            for policy_name in M5_POLICY_NAMES
            for metric_name in M5_METRIC_SPECS
        }
        if set(policy_scalars) != expected_scalar_keys:
            raise ValueError(
                "policy_scalars must contain exactly the M5 policy-metric matrix"
            )
        if any(
            not isinstance(value, ScenarioScalar)
            or value.cohort_index != cohort_index
            for value in policy_scalars.values()
        ):
            raise ValueError(
                "policy_scalars must retain typed values for this cohort index"
            )
        for row in metric_rows:
            if row["execution_role"] != "policy":
                continue
            scalar_key = (
                str(row["execution_name"]),
                str(row["metric_name"]),
            )
            if policy_scalars[scalar_key] != _scenario_scalar_from_row(row):
                raise ValueError(
                    "policy scalar differs from its canonical metric row"
                )

        if any(
            not isinstance(oracle, LogReplayZeroOracle)
            for oracle in zero_oracles
        ):
            raise TypeError(
                "zero_oracles must contain LogReplayZeroOracle values"
            )
        expected_oracle_keys = tuple(
            (cohort_index, metric_name)
            for metric_name in M5_ERROR_ORACLE_METRICS
        )
        observed_oracle_keys = tuple(
            (oracle.cohort_index, oracle.metric_name)
            for oracle in zero_oracles
        )
        if observed_oracle_keys != expected_oracle_keys:
            raise ValueError(
                "zero_oracles must contain every canonical log-replay oracle"
            )

        metric_rows = tuple(_row(dict(row)) for row in metric_rows)
        slice_rows = tuple(_row(dict(row)) for row in slice_rows)
        expected_digest = _canonical_digest(
            _CASE_METRIC_DIGEST_DOMAIN,
            metric_rows,
        )
        pass_1 = _validated_sha256(
            self.metric_pass_1_sha256,
            "metric_pass_1_sha256",
        )
        pass_2 = _validated_sha256(
            self.metric_pass_2_sha256,
            "metric_pass_2_sha256",
        )
        if pass_1 != expected_digest or pass_2 != expected_digest:
            raise ValueError(
                "metric pass digests do not match the retained canonical rows"
            )

        object.__setattr__(self, "cohort_index", cohort_index)
        object.__setattr__(self, "execution_specs", specs)
        object.__setattr__(self, "metric_rows", metric_rows)
        object.__setattr__(self, "slice_rows", slice_rows)
        object.__setattr__(
            self,
            "policy_scalars",
            MappingProxyType(dict(sorted(policy_scalars.items()))),
        )
        object.__setattr__(self, "zero_oracles", zero_oracles)
        object.__setattr__(self, "metric_pass_1_sha256", pass_1)
        object.__setattr__(self, "metric_pass_2_sha256", pass_2)


@dataclass(frozen=True, slots=True)
class M5ScorecardSummary:
    """Deterministic aggregate produced from compact streaming case facts."""

    case_count: int
    slice_rows: tuple[Mapping[str, Any], ...] = field(repr=False)
    scorecard_inputs: tuple[ScorecardCellInput, ...] = field(repr=False)
    scorecard_results: tuple[PairedCellResult, ...] = field(repr=False)
    metric_pass_1_sha256: str = field(repr=False)
    metric_pass_2_sha256: str = field(repr=False)
    statistics_pass_1_sha256: str = field(repr=False)
    statistics_pass_2_sha256: str = field(repr=False)

    def __post_init__(self) -> None:
        case_count = _positive_integer(self.case_count, "case_count")
        slice_rows = tuple(self.slice_rows)
        inputs = tuple(self.scorecard_inputs)
        results = tuple(self.scorecard_results)
        if len(slice_rows) != case_count * len(M5_SLICE_SPECS):
            raise ValueError(
                "slice row count does not match the finalized M5 cohort"
            )
        if any(
            not isinstance(row, Mapping)
            or set(row) != _SLICE_ROW_FIELDS
            for row in slice_rows
        ):
            raise ValueError("slice_rows differ from the M5 store schema")
        expected_slice_keys = tuple(
            (
                cohort_index,
                slice_spec.name,
                slice_spec.version,
            )
            for cohort_index in range(case_count)
            for slice_spec in M5_SLICE_SPECS
        )
        observed_slice_keys = tuple(
            (
                row["cohort_index"],
                row["slice_name"],
                row["slice_version"],
            )
            for row in slice_rows
        )
        if observed_slice_keys != expected_slice_keys:
            raise ValueError("final slice rows are not in canonical cohort order")
        expected_cells = (
            len(M5_METRIC_SPECS)
            * len(M5_SLICE_SPECS)
            * len(M5_POLICY_CONTRASTS)
        )
        if (
            len(inputs) != expected_cells
            or len(results) != expected_cells
            or any(not isinstance(item, ScorecardCellInput) for item in inputs)
            or any(not isinstance(item, PairedCellResult) for item in results)
            or tuple(item.spec for item in inputs)
            != tuple(item.spec for item in results)
        ):
            raise ValueError("scorecards differ from the fixed M5 cell matrix")
        digests = {
            name: _validated_sha256(getattr(self, name), name)
            for name in (
                "metric_pass_1_sha256",
                "metric_pass_2_sha256",
                "statistics_pass_1_sha256",
                "statistics_pass_2_sha256",
            )
        }
        if (
            digests["metric_pass_1_sha256"]
            != digests["metric_pass_2_sha256"]
            or digests["statistics_pass_1_sha256"]
            != digests["statistics_pass_2_sha256"]
        ):
            raise ValueError("M5 deterministic pass digests differ")
        object.__setattr__(self, "case_count", case_count)
        object.__setattr__(
            self,
            "slice_rows",
            tuple(_row(dict(row)) for row in slice_rows),
        )
        object.__setattr__(self, "scorecard_inputs", inputs)
        object.__setattr__(self, "scorecard_results", results)
        for name, digest in digests.items():
            object.__setattr__(self, name, digest)


class M5ScorecardAccumulator:
    """Retain only compact per-case facts needed for the paired scorecard."""

    __slots__ = (
        "_case_digests_1",
        "_case_digests_2",
        "_execution_specs",
        "_finalized",
        "_policy_scalars",
        "_slice_rows",
    )

    def __init__(self) -> None:
        self._case_digests_1: dict[int, str] = {}
        self._case_digests_2: dict[int, str] = {}
        self._execution_specs: tuple[ExecutionSpec, ...] | None = None
        self._finalized = False
        self._policy_scalars: dict[
            tuple[int, str, str],
            ScenarioScalar,
        ] = {}
        self._slice_rows: dict[tuple[int, str], Mapping[str, Any]] = {}

    def __repr__(self) -> str:
        return (
            "M5ScorecardAccumulator("
            f"case_count={self.case_count}, finalized={self._finalized})"
        )

    @property
    def case_count(self) -> int:
        return len(self._case_digests_1)

    @property
    def retained_scalar_count(self) -> int:
        return len(self._policy_scalars)

    @property
    def retained_slice_count(self) -> int:
        return len(self._slice_rows)

    def add_case(self, case: M5CaseEvaluation) -> None:
        """Consume one case without retaining distributions or source objects."""

        if self._finalized:
            raise M5EvaluationError("a finalized M5 accumulator is immutable")
        if not isinstance(case, M5CaseEvaluation):
            raise TypeError("case must be an M5CaseEvaluation")
        cohort_index = case.cohort_index
        if cohort_index in self._case_digests_1:
            raise M5EvaluationError(
                f"duplicate M5 cohort_index {cohort_index}"
            )
        if self._execution_specs is None:
            self._execution_specs = case.execution_specs
        elif case.execution_specs != self._execution_specs:
            raise M5EvaluationError(
                "execution specifications drifted across M5 cases"
            )

        for row in case.slice_rows:
            key = (cohort_index, str(row["slice_name"]))
            if key in self._slice_rows:
                raise M5EvaluationError("duplicate M5 slice fact")
            self._slice_rows[key] = row
        for (policy_name, metric_name), scalar in case.policy_scalars.items():
            key = (cohort_index, policy_name, metric_name)
            if key in self._policy_scalars:
                raise M5EvaluationError("duplicate M5 policy scalar")
            self._policy_scalars[key] = scalar
        self._case_digests_1[cohort_index] = case.metric_pass_1_sha256
        self._case_digests_2[cohort_index] = case.metric_pass_2_sha256

    def finalize(self, *, expected_case_count: int) -> M5ScorecardSummary:
        """Validate exact index coverage and run two deterministic stats passes."""

        if self._finalized:
            raise M5EvaluationError("a finalized M5 accumulator is immutable")
        expected_count = _positive_integer(
            expected_case_count,
            "expected_case_count",
        )
        expected_indices = tuple(range(expected_count))
        observed_indices = tuple(sorted(self._case_digests_1))
        if observed_indices != expected_indices:
            raise M5EvaluationError(
                "M5 cohort indices are missing, out of range, or non-contiguous"
            )
        if tuple(sorted(self._case_digests_2)) != expected_indices:
            raise M5EvaluationError("M5 metric pass index coverage drifted")
        if self._execution_specs is None:
            raise M5EvaluationError("cannot finalize an empty M5 accumulator")

        slice_rows = tuple(
            self._slice_rows[(cohort_index, spec.name)]
            for cohort_index in expected_indices
            for spec in M5_SLICE_SPECS
        )
        inputs = _scorecard_inputs_from_scalars(
            expected_indices,
            self._slice_rows,
            self._policy_scalars,
        )
        results_1 = tuple(
            analyze_paired_cell(item.spec, item.values_a, item.values_b)
            for item in inputs
        )
        results_2 = tuple(
            analyze_paired_cell(item.spec, item.values_a, item.values_b)
            for item in inputs
        )
        statistics_pass_1 = _canonical_digest(
            _STATISTICS_DIGEST_DOMAIN,
            tuple(result.to_dict() for result in results_1),
        )
        statistics_pass_2 = _canonical_digest(
            _STATISTICS_DIGEST_DOMAIN,
            tuple(result.to_dict() for result in results_2),
        )
        if results_1 != results_2 or statistics_pass_1 != statistics_pass_2:
            raise M5EvaluationError(
                "M5 paired statistics changed across deterministic passes"
            )

        metric_pass_1 = _cohort_metric_digest(self._case_digests_1)
        metric_pass_2 = _cohort_metric_digest(self._case_digests_2)
        if metric_pass_1 != metric_pass_2:
            raise M5EvaluationError(
                "M5 metric rows changed across deterministic passes"
            )
        summary = M5ScorecardSummary(
            case_count=expected_count,
            slice_rows=slice_rows,
            scorecard_inputs=inputs,
            scorecard_results=results_1,
            metric_pass_1_sha256=metric_pass_1,
            metric_pass_2_sha256=metric_pass_2,
            statistics_pass_1_sha256=statistics_pass_1,
            statistics_pass_2_sha256=statistics_pass_2,
        )
        self._finalized = True
        return summary


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


def evaluate_m5_case(
    case: EvaluationCase,
    executions: Sequence[ExecutionRollout],
) -> M5CaseEvaluation:
    """Evaluate one case twice and return only store-ready and compact facts.

    Input order is intentionally irrelevant.  Executions are canonicalized by
    name, while their identities, roles, rollout provenance, metric pairing, and
    two-pass determinism are validated before any result is returned.
    """

    return _evaluate_m5_case(
        case,
        executions,
        metrics=_validated_metrics(m5_metrics()),
    )


def metric_store_row(
    cohort_index: int,
    execution: ExecutionSpec,
    result: MetricResult,
) -> Mapping[str, Any]:
    """Return one immutable result-store row with an explicit execution role."""

    normalized_index = _nonnegative_integer(cohort_index, "cohort_index")
    if not isinstance(execution, ExecutionSpec):
        raise TypeError("execution must be an ExecutionSpec")
    if not isinstance(result, MetricResult):
        raise TypeError("result must be a MetricResult")
    details_json = json.dumps(
        _thaw_json(result.details),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return _row(
        {
            "cohort_index": normalized_index,
            "details_json": details_json,
            "distribution": tuple(result.distribution),
            "eligible_components": result.eligible_components,
            "execution_name": execution.name,
            "execution_role": execution.role,
            "invalid_reason": result.invalid_reason,
            "metric_name": result.metric_name,
            "metric_version": result.metric_version,
            "seed": execution.seed,
            "total_components": result.total_components,
            "valid": result.valid,
            "value": result.value,
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
    accumulator = M5ScorecardAccumulator()
    metric_rows: list[Mapping[str, Any]] = []
    zero_oracles: list[LogReplayZeroOracle] = []
    for case in cases:
        executions: list[ExecutionRollout] = []
        for policy_name, policy_version, policy in policy_entries:
            rollout = executor.execute(case, policy, normalized_seed)
            executions.append(
                ExecutionRollout(
                    spec=ExecutionSpec(
                        name=policy_name,
                        version=policy_version,
                        role="policy",
                        seed=normalized_seed,
                    ),
                    rollout=rollout,
                )
            )
        case_result = _evaluate_m5_case(
            case,
            tuple(executions),
            metrics=metrics,
        )
        metric_rows.extend(case_result.metric_rows)
        zero_oracles.extend(case_result.zero_oracles)
        accumulator.add_case(case_result)

    summary = accumulator.finalize(expected_case_count=len(cases))
    zero_oracles.sort(
        key=lambda item: (item.cohort_index, item.metric_name)
    )
    return M5EvaluationResult(
        cases=cases,
        metric_rows=tuple(metric_rows),
        slice_rows=summary.slice_rows,
        scorecard_inputs=summary.scorecard_inputs,
        scorecard_results=summary.scorecard_results,
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


@dataclass(frozen=True, slots=True)
class _MetricPass:
    metric_rows: tuple[Mapping[str, Any], ...] = field(repr=False)
    policy_scalars: Mapping[tuple[str, str], ScenarioScalar] = field(
        repr=False
    )
    results: Mapping[tuple[str, str], MetricResult] = field(repr=False)
    zero_oracles: tuple[LogReplayZeroOracle, ...] = field(repr=False)


def _evaluate_m5_case(
    case: EvaluationCase,
    executions: Sequence[ExecutionRollout],
    *,
    metrics: Sequence[Metric],
) -> M5CaseEvaluation:
    if not isinstance(case, EvaluationCase):
        raise TypeError("case must be an EvaluationCase")
    normalized_executions = _validated_execution_rollouts(executions)
    normalized_metrics = _validated_metrics(metrics)
    for execution in normalized_executions:
        _validate_rollout_identity(
            case,
            execution.rollout,
            execution_name=execution.spec.name,
            execution_version=execution.spec.version,
            seed=execution.spec.seed,
        )

    eligibility_1, slices_1 = _evaluate_source_facts(
        case,
        normalized_metrics,
    )
    metric_pass_1 = _evaluate_metric_pass(
        case,
        normalized_executions,
        normalized_metrics,
        eligibility_1,
    )
    eligibility_2, slices_2 = _evaluate_source_facts(
        case,
        normalized_metrics,
    )
    if eligibility_1 != eligibility_2 or slices_1 != slices_2:
        raise M5EvaluationError(
            "M5 source eligibility or slices changed across deterministic passes"
        )
    metric_pass_2 = _evaluate_metric_pass(
        case,
        normalized_executions,
        normalized_metrics,
        eligibility_2,
    )
    digest_1 = _canonical_digest(
        _CASE_METRIC_DIGEST_DOMAIN,
        metric_pass_1.metric_rows,
    )
    digest_2 = _canonical_digest(
        _CASE_METRIC_DIGEST_DOMAIN,
        metric_pass_2.metric_rows,
    )
    if (
        metric_pass_1.metric_rows != metric_pass_2.metric_rows
        or metric_pass_1.policy_scalars != metric_pass_2.policy_scalars
        or metric_pass_1.zero_oracles != metric_pass_2.zero_oracles
        or digest_1 != digest_2
    ):
        raise M5EvaluationError(
            "M5 metric results changed across deterministic passes"
        )

    slice_rows = tuple(
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
        for result in slices_1
    )
    return M5CaseEvaluation(
        cohort_index=case.cohort_index,
        execution_specs=tuple(
            execution.spec for execution in normalized_executions
        ),
        metric_rows=metric_pass_1.metric_rows,
        slice_rows=slice_rows,
        policy_scalars=metric_pass_1.policy_scalars,
        zero_oracles=metric_pass_1.zero_oracles,
        metric_pass_1_sha256=digest_1,
        metric_pass_2_sha256=digest_2,
    )


def _evaluate_source_facts(
    case: EvaluationCase,
    metrics: Sequence[Metric],
) -> tuple[tuple[MetricEligibility, ...], tuple[SliceResult, ...]]:
    eligibility: list[MetricEligibility] = []
    for metric in metrics:
        decision = metric.eligibility(case.scenario)
        _validate_eligibility(metric, decision)
        eligibility.append(decision)
    slices = tuple(evaluate_m5_slices(case.scenario))
    _validate_slice_results(case, slices)
    return tuple(eligibility), slices


def _evaluate_metric_pass(
    case: EvaluationCase,
    executions: Sequence[ExecutionRollout],
    metrics: Sequence[Metric],
    eligibility: Sequence[MetricEligibility],
) -> _MetricPass:
    metric_rows: list[Mapping[str, Any]] = []
    policy_scalars: dict[tuple[str, str], ScenarioScalar] = {}
    results: dict[tuple[str, str], MetricResult] = {}
    zero_oracles: list[LogReplayZeroOracle] = []
    pairing: dict[str, tuple[tuple[Any, ...], ...]] = {}
    for execution in executions:
        signatures: list[tuple[Any, ...]] = []
        for metric, decision in zip(metrics, eligibility, strict=True):
            result = metric.compute(case.scenario, execution.rollout)
            _validate_metric_result(case, metric, decision, result)
            signatures.append(_pairing_signature(result))
            results[(execution.spec.name, metric.spec.name)] = result
            metric_rows.append(
                metric_store_row(
                    case.cohort_index,
                    execution.spec,
                    result,
                )
            )
            if execution.spec.role == "policy":
                policy_scalars[(execution.spec.name, metric.spec.name)] = (
                    _scenario_scalar(case.cohort_index, result)
                )
            if (
                execution.spec.name == "log_replay"
                and metric.spec.name in M5_ERROR_ORACLE_METRICS
            ):
                zero_oracles.append(
                    _log_replay_zero_oracle(case.cohort_index, result)
                )
        pairing[execution.spec.name] = tuple(signatures)
    if len(set(pairing.values())) != 1:
        raise M5EvaluationError(
            "execution metric eligibility or component pairing drifted"
        )

    references = tuple(
        execution
        for execution in executions
        if execution.spec.role == "reference"
    )
    if references:
        reference_name = references[0].spec.name
        for metric in metrics:
            reference_result = results[(reference_name, metric.spec.name)]
            log_replay_result = results[("log_replay", metric.spec.name)]
            if _metric_result_signature(
                reference_result
            ) != _metric_result_signature(log_replay_result):
                raise M5EvaluationError(
                    "reference execution differs from exact log replay"
                )

    metric_rows.sort(key=_metric_row_key)
    zero_oracles.sort(key=lambda item: item.metric_name)
    return _MetricPass(
        metric_rows=tuple(metric_rows),
        policy_scalars=MappingProxyType(dict(sorted(policy_scalars.items()))),
        results=MappingProxyType(dict(sorted(results.items()))),
        zero_oracles=tuple(zero_oracles),
    )


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
    if any(
        version != _M5_POLICY_VERSIONS[name]
        for name, version, _ in entries
    ):
        raise M5EvaluationError(
            "M5 policy version differs from the frozen registry"
        )
    return tuple(entries)


def _validated_execution_specs(
    raw_specs: Sequence[ExecutionSpec],
) -> tuple[ExecutionSpec, ...]:
    if isinstance(raw_specs, (str, bytes)) or not isinstance(
        raw_specs,
        Sequence,
    ):
        raise TypeError("execution_specs must be a sequence")
    specs = tuple(raw_specs)
    if any(not isinstance(spec, ExecutionSpec) for spec in specs):
        raise TypeError("execution_specs must contain ExecutionSpec values")
    specs = tuple(sorted(specs, key=lambda spec: spec.name))
    names = tuple(spec.name for spec in specs)
    if len(set(names)) != len(names):
        raise M5EvaluationError("M5 execution names must be unique")
    policy_names = tuple(
        spec.name for spec in specs if spec.role == "policy"
    )
    references = tuple(spec for spec in specs if spec.role == "reference")
    if policy_names != M5_POLICY_NAMES:
        raise M5EvaluationError(
            "M5 requires exactly constant_velocity, idm, and log_replay policies"
        )
    if len(references) > 1:
        raise M5EvaluationError(
            "M5 permits at most one exact-log reference execution"
        )
    return specs


def _validated_execution_rollouts(
    raw_executions: Sequence[ExecutionRollout],
) -> tuple[ExecutionRollout, ...]:
    if isinstance(raw_executions, (str, bytes)) or not isinstance(
        raw_executions,
        Sequence,
    ):
        raise TypeError("executions must be a sequence")
    executions = tuple(raw_executions)
    if any(
        not isinstance(execution, ExecutionRollout)
        for execution in executions
    ):
        raise TypeError("executions must contain ExecutionRollout values")
    executions = tuple(
        sorted(executions, key=lambda execution: execution.spec.name)
    )
    specs = _validated_execution_specs(
        tuple(execution.spec for execution in executions)
    )
    if tuple(execution.spec for execution in executions) != specs:
        raise M5EvaluationError("M5 execution order could not be canonicalized")
    return executions


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
    execution_name: str,
    execution_version: str,
    seed: int,
) -> None:
    scenario = case.scenario
    if not isinstance(rollout, Rollout):
        raise TypeError("policy executor must return a Rollout")
    if (
        rollout.scenario_id != scenario.scenario_id
        or rollout.sim_name != execution_name
        or rollout.sim_version != execution_version
        or rollout.seed != seed
        or rollout.perturbation is not None
        or rollout.num_steps != scenario.num_steps
        or rollout.num_agents != scenario.num_agents
        or not np.array_equal(rollout.timestamps, scenario.timestamps)
    ):
        raise M5EvaluationError(
            "execution rollout identity or provenance does not match its case"
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
                "execution rollout agent identity or validity drifted"
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


def _metric_result_signature(result: MetricResult) -> tuple[Any, ...]:
    return (
        result.metric_name,
        result.metric_version,
        result.scenario_id,
        result.value,
        result.distribution,
        result.valid,
        result.invalid_reason,
        result.eligible_components,
        result.total_components,
        json.dumps(
            _thaw_json(result.details),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
    )


def _metric_row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["cohort_index"],
        row["execution_name"],
        row["seed"],
        row["metric_name"],
        row["metric_version"],
    )


def _cohort_metric_digest(case_digests: Mapping[int, str]) -> str:
    payload = tuple(
        {
            "cohort_index": cohort_index,
            "sha256": _validated_sha256(
                case_digests[cohort_index],
                "case metric digest",
            ),
        }
        for cohort_index in sorted(case_digests)
    )
    return _canonical_digest(
        _COHORT_METRIC_DIGEST_DOMAIN,
        payload,
    )


def _canonical_digest(domain: str, payload: Any) -> str:
    canonical_json = json.dumps(
        _thaw_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        domain.encode("ascii") + b"\0" + canonical_json
    ).hexdigest()


def _validated_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _scorecard_inputs_from_scalars(
    cohort_indices: Sequence[int],
    slice_rows: Mapping[tuple[int, str], Mapping[str, Any]],
    policy_scalars: Mapping[tuple[int, str, str], ScenarioScalar],
) -> tuple[ScorecardCellInput, ...]:
    indices = tuple(cohort_indices)
    member_indices = {
        spec.name: tuple(
            cohort_index
            for cohort_index in indices
            if (
                slice_rows[(cohort_index, spec.name)]["eligible"]
                and slice_rows[(cohort_index, spec.name)]["member"]
            )
        )
        for spec in M5_SLICE_SPECS
    }
    inputs: list[ScorecardCellInput] = []
    for metric_name, metric_spec in M5_METRIC_SPECS.items():
        for slice_spec in M5_SLICE_SPECS:
            selected_indices = member_indices[slice_spec.name]
            for contrast in M5_POLICY_CONTRASTS:
                spec = PairedCellSpec(
                    metric_name=metric_name,
                    metric_version=metric_spec.version,
                    slice_name=slice_spec.name,
                    slice_version=slice_spec.version,
                    contrast=contrast,
                )
                inputs.append(
                    ScorecardCellInput(
                        spec=spec,
                        values_a=tuple(
                            policy_scalars[
                                (
                                    cohort_index,
                                    contrast.policy_a,
                                    metric_name,
                                )
                            ]
                            for cohort_index in selected_indices
                        ),
                        values_b=tuple(
                            policy_scalars[
                                (
                                    cohort_index,
                                    contrast.policy_b,
                                    metric_name,
                                )
                            ]
                            for cohort_index in selected_indices
                        ),
                    )
                )
    return tuple(inputs)


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


def _scenario_scalar_from_row(
    row: Mapping[str, Any],
) -> ScenarioScalar:
    cohort_index = _nonnegative_integer(
        row["cohort_index"],
        "cohort_index",
    )
    if row["valid"] is True:
        return ScenarioScalar(
            cohort_index=cohort_index,
            value=row["value"],
            eligible_components=row["eligible_components"],
            total_components=row["total_components"],
        )
    if row["valid"] is not False:
        raise ValueError("metric row valid must be a boolean")
    return ScenarioScalar.missing(
        cohort_index,
        str(row["invalid_reason"]),
        total_components=row["total_components"],
    )


def _row(payload: dict[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(payload)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw_json(item) for item in value]
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_integer(value: Any, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result < 1:
        raise ValueError(f"{name} must be positive")
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
    "ExecutionRole",
    "ExecutionRollout",
    "ExecutionSpec",
    "LogReplayZeroOracle",
    "M5CaseEvaluation",
    "M5EvaluationError",
    "M5EvaluationResult",
    "M5ScorecardAccumulator",
    "M5ScorecardSummary",
    "NumpyPolicyExecutor",
    "PolicyExecutor",
    "ScorecardCellInput",
    "SyntheticM5CohortAdapter",
    "canonical_m5_policies",
    "evaluate_m5_case",
    "metric_store_row",
    "run_m5_evaluation",
    "run_synthetic_m5_evaluation",
    "synthetic_source_evidence",
]
