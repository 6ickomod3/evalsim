"""Data-free contract tests for the source-neutral streaming M5 foundation."""
from __future__ import annotations

import copy
from dataclasses import replace
import gc
import weakref

import pytest

from evalsim.contracts import Metric, MetricEligibility, MetricResult, Rollout
import evalsim.evaluation.m5 as runner_module
from evalsim.evaluation.m5 import (
    EvaluationCase,
    ExecutionRollout,
    ExecutionSpec,
    M5EvaluationError,
    M5ScorecardAccumulator,
    NumpyPolicyExecutor,
    SyntheticM5CohortAdapter,
    canonical_m5_policies,
    evaluate_m5_case,
)
from evalsim.metrics.m5 import M5_METRIC_SPECS, m5_metrics
import evalsim.stats.m5 as stats_module


_REFERENCE_NAME = "waymax_exact_log_state_dynamics"
_REFERENCE_VERSION = "0.1.0"


def _policy_executions(case: EvaluationCase) -> tuple[ExecutionRollout, ...]:
    executor = NumpyPolicyExecutor()
    executions = []
    for policy in canonical_m5_policies():
        metadata = policy.metadata()
        executions.append(
            ExecutionRollout(
                spec=ExecutionSpec(
                    metadata.name,
                    metadata.version,
                    "policy",
                    seed=0,
                ),
                rollout=executor.execute(case, policy, 0),
            )
        )
    return tuple(executions)


def _reference_execution(
    source: ExecutionRollout,
) -> ExecutionRollout:
    rollout = copy.deepcopy(source.rollout)
    rollout.sim_name = _REFERENCE_NAME
    rollout.sim_version = _REFERENCE_VERSION
    return ExecutionRollout(
        spec=ExecutionSpec(
            _REFERENCE_NAME,
            _REFERENCE_VERSION,
            "reference",
            seed=0,
        ),
        rollout=rollout,
    )


def test_per_case_rows_are_role_aware_and_reference_is_not_contrasted() -> None:
    case = SyntheticM5CohortAdapter().cases()[0]
    policy_executions = _policy_executions(case)
    log_replay = next(
        execution
        for execution in policy_executions
        if execution.spec.name == "log_replay"
    )
    result = evaluate_m5_case(
        case,
        (*reversed(policy_executions), _reference_execution(log_replay)),
    )

    assert len(result.metric_rows) == 4 * len(M5_METRIC_SPECS) == 52
    assert len(result.slice_rows) == 8
    assert len(result.policy_scalars) == 3 * len(M5_METRIC_SPECS) == 39
    assert len(result.zero_oracles) == 5
    assert result.metric_pass_1_sha256 == result.metric_pass_2_sha256
    assert tuple(row["execution_name"] for row in result.metric_rows) == tuple(
        sorted(row["execution_name"] for row in result.metric_rows)
    )
    reference_rows = tuple(
        row
        for row in result.metric_rows
        if row["execution_name"] == _REFERENCE_NAME
    )
    assert len(reference_rows) == len(M5_METRIC_SPECS)
    assert all(row["execution_role"] == "reference" for row in reference_rows)
    assert all(
        _REFERENCE_NAME not in scalar_key
        for scalar_key in result.policy_scalars
    )

    accumulator = M5ScorecardAccumulator()
    accumulator.add_case(result)
    summary = accumulator.finalize(expected_case_count=1)
    assert all(
        _REFERENCE_NAME
        not in (
            cell.spec.contrast.policy_a,
            cell.spec.contrast.policy_b,
        )
        for cell in summary.scorecard_results
    )


def test_reference_must_equal_log_replay_for_every_metric() -> None:
    case = SyntheticM5CohortAdapter().cases()[0]
    policy_executions = _policy_executions(case)
    idm = next(
        execution
        for execution in policy_executions
        if execution.spec.name == "idm"
    )

    with pytest.raises(M5EvaluationError, match="reference.*log replay"):
        evaluate_m5_case(
            case,
            (*policy_executions, _reference_execution(idm)),
        )


class _PassDriftMetric(Metric):
    def __init__(self, delegate: Metric) -> None:
        self.delegate = delegate
        self.spec = delegate.spec
        self.compute_calls = 0

    def eligibility(self, scenario) -> MetricEligibility:
        return self.delegate.eligibility(scenario)

    def compute(self, scenario, rollout) -> MetricResult:
        result = self.delegate.compute(scenario, rollout)
        pass_index = self.compute_calls // 3
        self.compute_calls += 1
        details = dict(result.details)
        details["test_pass_index"] = pass_index
        return replace(result, details=details)


def test_metric_pass_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = SyntheticM5CohortAdapter().cases()[0]
    wrapper = _PassDriftMetric(
        next(
            metric
            for metric in m5_metrics()
            if metric.spec.name == "oriented_box_overlap_rate"
        )
    )
    metrics = tuple(
        wrapper if metric.spec.name == wrapper.spec.name else metric
        for metric in m5_metrics()
    )
    monkeypatch.setattr(runner_module, "m5_metrics", lambda: metrics)

    with pytest.raises(M5EvaluationError, match="deterministic passes"):
        evaluate_m5_case(case, _policy_executions(case))


def test_statistics_pass_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zero_metrics = tuple(
        _ZeroMetric(spec) for spec in M5_METRIC_SPECS.values()
    )
    monkeypatch.setattr(runner_module, "m5_metrics", lambda: zero_metrics)
    scenario = SyntheticM5CohortAdapter().cases()[0].scenario
    scenario.scenario_id = "mock-statistics-drift"
    case_result = evaluate_m5_case(
        EvaluationCase(0, scenario),
        _mock_executions(scenario),
    )
    real_analyze = runner_module.analyze_paired_cell
    calls = 0

    def drifting_analyze(spec, values_a, values_b):
        nonlocal calls
        result = real_analyze(spec, values_a, values_b)
        calls += 1
        if calls > 312:
            return replace(result, status=f"{result.status}_drift")
        return result

    monkeypatch.setattr(
        runner_module,
        "analyze_paired_cell",
        drifting_analyze,
    )
    accumulator = M5ScorecardAccumulator()
    accumulator.add_case(case_result)
    with pytest.raises(M5EvaluationError, match="statistics.*deterministic"):
        accumulator.finalize(expected_case_count=1)


class _ZeroMetric(Metric):
    def __init__(self, spec) -> None:
        self.spec = spec

    def eligibility(self, scenario) -> MetricEligibility:
        del scenario
        return MetricEligibility.accepted()

    def compute(self, scenario, rollout) -> MetricResult:
        del rollout
        return MetricResult(
            metric_name=self.spec.name,
            metric_version=self.spec.version,
            scenario_id=scenario.scenario_id,
            value=0.0,
            distribution=(0.0,),
            valid=True,
            invalid_reason=None,
            eligible_components=1,
            total_components=1,
            details={"mock_streaming": True},
        )


class _EphemeralPayload:
    pass


def _mock_executions(scenario) -> tuple[ExecutionRollout, ...]:
    specs = (
        ExecutionSpec("constant_velocity", "0.1.0", "policy"),
        ExecutionSpec("idm", "0.1.0", "policy"),
        ExecutionSpec("log_replay", "0.1.0", "policy"),
        ExecutionSpec(
            _REFERENCE_NAME,
            _REFERENCE_VERSION,
            "reference",
        ),
    )
    return tuple(
        ExecutionRollout(
            spec=spec,
            rollout=Rollout(
                scenario_id=scenario.scenario_id,
                sim_name=spec.name,
                sim_version=spec.version,
                seed=spec.seed,
                timestamps=scenario.timestamps,
                agents=scenario.agents,
            ),
        )
        for spec in specs
    )


def _add_mock_case(
    accumulator: M5ScorecardAccumulator,
    scenario_template,
    cohort_index: int,
) -> None:
    scenario = copy.deepcopy(scenario_template)
    scenario.scenario_id = f"mock-stream-{cohort_index:03d}"
    scenario_reference = weakref.ref(scenario)
    payload = _EphemeralPayload()
    payload_reference = weakref.ref(payload)
    case = EvaluationCase(
        cohort_index,
        scenario,
        reference_payload=payload,
    )
    result = evaluate_m5_case(case, _mock_executions(scenario))
    accumulator.add_case(result)
    del payload, case, result, scenario
    gc.collect()
    assert payload_reference() is None
    assert scenario_reference() is None


def test_128_case_streaming_is_compact_order_invariant_and_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zero_metrics = tuple(
        _ZeroMetric(spec) for spec in M5_METRIC_SPECS.values()
    )
    monkeypatch.setattr(runner_module, "m5_metrics", lambda: zero_metrics)
    monkeypatch.setattr(stats_module, "M5_PRIMARY_RESAMPLES", 16)
    monkeypatch.setattr(stats_module, "M5_OTHER_RESAMPLES", 16)
    scenario = SyntheticM5CohortAdapter().cases()[0].scenario

    reversed_accumulator = M5ScorecardAccumulator()
    for cohort_index in reversed(range(128)):
        _add_mock_case(reversed_accumulator, scenario, cohort_index)
    assert reversed_accumulator.case_count == 128
    assert reversed_accumulator.retained_scalar_count == 128 * 3 * 13
    assert reversed_accumulator.retained_slice_count == 128 * 8
    assert "metric_rows" not in repr(reversed_accumulator)
    reversed_summary = reversed_accumulator.finalize(
        expected_case_count=128
    )

    forward_accumulator = M5ScorecardAccumulator()
    for cohort_index in range(128):
        _add_mock_case(forward_accumulator, scenario, cohort_index)
    forward_summary = forward_accumulator.finalize(expected_case_count=128)

    assert len(reversed_summary.slice_rows) == 1_024
    assert len(reversed_summary.scorecard_results) == 312
    assert reversed_summary.metric_pass_1_sha256 == (
        reversed_summary.metric_pass_2_sha256
    )
    assert reversed_summary.statistics_pass_1_sha256 == (
        reversed_summary.statistics_pass_2_sha256
    )
    assert reversed_summary.metric_pass_1_sha256 == (
        forward_summary.metric_pass_1_sha256
    )
    assert reversed_summary.statistics_pass_1_sha256 == (
        forward_summary.statistics_pass_1_sha256
    )
    assert reversed_summary.scorecard_results == (
        forward_summary.scorecard_results
    )


def test_accumulator_rejects_duplicate_missing_and_out_of_range_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zero_metrics = tuple(
        _ZeroMetric(spec) for spec in M5_METRIC_SPECS.values()
    )
    monkeypatch.setattr(runner_module, "m5_metrics", lambda: zero_metrics)
    scenario = SyntheticM5CohortAdapter().cases()[0].scenario
    scenario.scenario_id = "mock-index-contract"
    result = evaluate_m5_case(
        EvaluationCase(0, scenario),
        _mock_executions(scenario),
    )

    duplicate = M5ScorecardAccumulator()
    duplicate.add_case(result)
    with pytest.raises(M5EvaluationError, match="duplicate"):
        duplicate.add_case(result)

    missing = M5ScorecardAccumulator()
    missing.add_case(result)
    with pytest.raises(M5EvaluationError, match="missing|non-contiguous"):
        missing.finalize(expected_case_count=2)

    scenario.scenario_id = "mock-index-out-of-range"
    out_of_range_result = evaluate_m5_case(
        EvaluationCase(1, scenario),
        _mock_executions(scenario),
    )
    out_of_range = M5ScorecardAccumulator()
    out_of_range.add_case(out_of_range_result)
    with pytest.raises(M5EvaluationError, match="out of range|non-contiguous"):
        out_of_range.finalize(expected_case_count=1)

    scenario.scenario_id = "mock-execution-drift"
    policy_only_result = evaluate_m5_case(
        EvaluationCase(1, scenario),
        _mock_executions(scenario)[:3],
    )
    execution_drift = M5ScorecardAccumulator()
    execution_drift.add_case(result)
    with pytest.raises(M5EvaluationError, match="specifications drifted"):
        execution_drift.add_case(policy_only_result)


def test_case_contract_rejects_row_order_index_and_execution_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zero_metrics = tuple(
        _ZeroMetric(spec) for spec in M5_METRIC_SPECS.values()
    )
    monkeypatch.setattr(runner_module, "m5_metrics", lambda: zero_metrics)
    scenario = SyntheticM5CohortAdapter().cases()[0].scenario
    scenario.scenario_id = "mock-case-contract"
    case = EvaluationCase(0, scenario)
    executions = _mock_executions(scenario)
    result = evaluate_m5_case(case, tuple(reversed(executions)))

    with pytest.raises(ValueError, match="canonical.*order"):
        replace(result, metric_rows=tuple(reversed(result.metric_rows)))
    with pytest.raises(ValueError, match="cohort|canonical"):
        replace(result, cohort_index=1)
    with pytest.raises(M5EvaluationError, match="unique"):
        evaluate_m5_case(case, (*executions, executions[0]))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("name", "not-valid"),
        ("version", "not valid"),
        ("version", "9.9.9"),
        ("role", "baseline"),
        ("seed", True),
        ("seed", -1),
        ("seed", 2**32),
    ),
)
def test_execution_spec_rejects_invalid_identity(
    field: str,
    value: object,
) -> None:
    kwargs = {
        "name": "constant_velocity",
        "version": "0.1.0",
        "role": "policy",
        "seed": 0,
    }
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError)):
        ExecutionSpec(**kwargs)
