"""Data-free tests for the source-neutral M5 evaluation runner."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from pathlib import Path
import subprocess
import sys

import pytest

from evalsim.contracts import (
    Metric,
    MetricEligibility,
    MetricResult,
    Rollout,
    SimulatorPolicy,
)
import evalsim.evaluation.m5 as runner_module
from evalsim.evaluation.m5 import (
    M5_ERROR_ORACLE_METRICS,
    M5_POLICY_NAMES,
    M5_SYNTHETIC_CASE_COUNT,
    M5_SYNTHETIC_CURRENT_INDEX,
    M5_SYNTHETIC_NUM_STEPS,
    CohortAdapter,
    EvaluationCase,
    M5EvaluationError,
    NumpyPolicyExecutor,
    PolicyExecutor,
    SyntheticM5CohortAdapter,
    canonical_m5_policies,
    run_m5_evaluation,
    run_synthetic_m5_evaluation,
    synthetic_source_evidence,
)
from evalsim.metrics.m5 import M5_METRIC_SPECS, PositionErrorMetric, m5_metrics
from evalsim.results import (
    ExpectedRowCounts,
    M5ResultStore,
    scorecard_row_from_result,
)
from evalsim.slices.m5 import M5_SLICE_SPECS
from evalsim.sources.synthetic import SCENARIO_KINDS
from evalsim.stats.m5 import M5_POLICY_CONTRASTS


@pytest.fixture(scope="module")
def synthetic_result():
    return run_synthetic_m5_evaluation()


def test_fixed_synthetic_matrix_and_store_row_domains(synthetic_result) -> None:
    result = synthetic_result
    assert result.case_count == M5_SYNTHETIC_CASE_COUNT == 5
    assert len(result.metric_rows) == 195
    assert len(result.slice_rows) == 40
    assert len(result.scorecard_inputs) == 312
    assert len(result.scorecard_results) == 312
    assert len(result.zero_oracles) == 25

    expected_metric_keys = {
        (cohort_index, policy_name, 0, metric_name, spec.version)
        for cohort_index in range(5)
        for policy_name in M5_POLICY_NAMES
        for metric_name, spec in M5_METRIC_SPECS.items()
    }
    observed_metric_keys = {
        (
            row["cohort_index"],
            row["execution_name"],
            row["seed"],
            row["metric_name"],
            row["metric_version"],
        )
        for row in result.metric_rows
    }
    assert observed_metric_keys == expected_metric_keys
    assert all(row["execution_role"] == "policy" for row in result.metric_rows)
    assert all("scenario_id" not in row for row in result.metric_rows)

    expected_slice_keys = {
        (cohort_index, spec.name, spec.version)
        for cohort_index in range(5)
        for spec in M5_SLICE_SPECS
    }
    observed_slice_keys = {
        (
            row["cohort_index"],
            row["slice_name"],
            row["slice_version"],
        )
        for row in result.slice_rows
    }
    assert observed_slice_keys == expected_slice_keys
    assert all("scenario_id" not in row for row in result.slice_rows)

    expected_cells = {
        (
            metric_name,
            slice_spec.name,
            contrast.policy_a,
            contrast.policy_b,
        )
        for metric_name in M5_METRIC_SPECS
        for slice_spec in M5_SLICE_SPECS
        for contrast in M5_POLICY_CONTRASTS
    }
    observed_cells = {
        (
            item.spec.metric_name,
            item.spec.slice_name,
            item.spec.contrast.policy_a,
            item.spec.contrast.policy_b,
        )
        for item in result.scorecard_results
    }
    assert observed_cells == expected_cells
    assert all(item.source_pairing_complete for item in result.scorecard_results)
    assert len(
        [scorecard_row_from_result(item) for item in result.scorecard_results]
    ) == 312


def test_synthetic_adapter_is_exact_and_repr_hides_local_cases(
    synthetic_result,
) -> None:
    cases = synthetic_result.cases
    assert tuple(case.cohort_index for case in cases) == tuple(range(5))
    assert tuple(
        case.scenario.metadata["scenario_kind"] for case in cases
    ) == tuple(kind.value for kind in SCENARIO_KINDS)
    assert all(
        case.scenario.num_steps == M5_SYNTHETIC_NUM_STEPS
        and case.scenario.metadata["current_index"]
        == M5_SYNTHETIC_CURRENT_INDEX
        for case in cases
    )
    assert len({case.scenario.scenario_id for case in cases}) == 5
    representation = repr(synthetic_result) + "".join(
        repr(case) for case in cases
    )
    assert representation == "M5EvaluationResult()" + "".join(
        f"EvaluationCase(cohort_index={index})" for index in range(5)
    )
    assert all(
        case.scenario.scenario_id not in representation for case in cases
    )
    evidence = synthetic_source_evidence()
    assert evidence["case_count"] == 5
    assert evidence["num_steps"] == 91
    assert evidence["current_index"] == 10
    assert evidence["split"] == "m5_data_free"
    assert evidence["order_version"] == "m5-synthetic-order-1"
    assert evidence["ordered_scenario_ids"] == tuple(
        case.scenario.scenario_id for case in cases
    )


def test_log_replay_exact_zero_oracles_retain_every_component(
    synthetic_result,
) -> None:
    expected_keys = {
        (cohort_index, metric_name)
        for cohort_index in range(5)
        for metric_name in M5_ERROR_ORACLE_METRICS
    }
    assert {
        (item.cohort_index, item.metric_name)
        for item in synthetic_result.zero_oracles
    } == expected_keys
    assert all(
        item.scalar_value == 0.0
        and item.maximum_absolute_component == 0.0
        and item.checked_components > 0
        for item in synthetic_result.zero_oracles
    )

    rows = [
        row
        for row in synthetic_result.metric_rows
        if (
            row["execution_name"] == "log_replay"
            and row["metric_name"] in M5_ERROR_ORACLE_METRICS
        )
    ]
    assert len(rows) == 25
    assert all(
        row["valid"] is True
        and row["value"] == 0.0
        and row["distribution"]
        and all(component == 0.0 for component in row["distribution"])
        for row in rows
    )


def test_runner_rows_write_through_the_fixed_store_seams(
    synthetic_result,
    tmp_path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = M5ResultStore.create(
        project,
        "runner-integration",
        expected_rows=ExpectedRowCounts(195, 40, 312, 0),
        data_free=True,
    )
    metric_record = store.write_metric_results_part(
        synthetic_result.metric_rows
    )
    slice_record = store.write_slice_membership(
        synthetic_result.slice_rows
    )
    scorecard_record = store.write_scorecards(
        tuple(
            scorecard_row_from_result(result)
            for result in synthetic_result.scorecard_results
        )
    )
    parity_record = store.write_waymax_parity_summary(())
    report = store.write_human_readable_scorecard()

    assert (
        metric_record.rows,
        slice_record.rows,
        scorecard_record.rows,
        parity_record.rows,
    ) == (195, 40, 312, 0)
    assert report.path == "scorecard.md"


class _OpaquePayload:
    def __repr__(self) -> str:  # pragma: no cover - must remain untouched
        raise AssertionError("runner inspected reference payload repr")

    def __iter__(self):  # pragma: no cover - must remain untouched
        raise AssertionError("runner iterated reference payload")

    def __eq__(self, other: object) -> bool:  # pragma: no cover
        del other
        raise AssertionError("runner compared reference payload")


@dataclass(frozen=True)
class _StaticAdapter:
    retained_cases: tuple[EvaluationCase, ...]

    def cases(self) -> tuple[EvaluationCase, ...]:
        return self.retained_cases


def test_generic_runner_is_payload_opaque_and_order_deterministic(
    synthetic_result,
) -> None:
    opaque_cases = tuple(
        EvaluationCase(
            case.cohort_index,
            case.scenario,
            reference_payload=_OpaquePayload(),
        )
        for case in SyntheticM5CohortAdapter().cases()
    )
    adapter = _StaticAdapter(opaque_cases)
    assert isinstance(adapter, CohortAdapter)
    executor = NumpyPolicyExecutor()
    assert isinstance(executor, PolicyExecutor)
    repeated = run_m5_evaluation(
        adapter,
        executor,
        policies=tuple(reversed(canonical_m5_policies())),
    )

    assert repeated.metric_rows == synthetic_result.metric_rows
    assert repeated.slice_rows == synthetic_result.slice_rows
    assert repeated.scorecard_results == synthetic_result.scorecard_results
    assert repeated.zero_oracles == synthetic_result.zero_oracles
    assert "reference_payload" not in inspect.getsource(run_m5_evaluation)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda cases: (
            cases[0],
            EvaluationCase(0, cases[1].scenario),
            *cases[2:],
        ),
        lambda cases: (
            cases[0],
            EvaluationCase(
                1,
                _scenario_with_id(
                    cases[1].scenario,
                    cases[0].scenario.scenario_id,
                ),
            ),
            *cases[2:],
        ),
    ),
)
def test_cohort_identity_drift_fails_before_execution(mutate) -> None:
    cases = SyntheticM5CohortAdapter().cases()
    adapter = _StaticAdapter(tuple(mutate(cases)))

    class NeverExecutor:
        def execute(self, case, policy, seed):
            del case, policy, seed
            pytest.fail("invalid cohort must fail before policy execution")

    with pytest.raises(M5EvaluationError, match="cohort|identit"):
        run_m5_evaluation(adapter, NeverExecutor())


def _scenario_with_id(scenario, scenario_id: str):
    import copy

    copied = copy.deepcopy(scenario)
    copied.scenario_id = scenario_id
    return copied


def test_policy_rollout_identity_drift_fails_closed() -> None:
    class WrongIdentityExecutor:
        def __init__(self) -> None:
            self.delegate = NumpyPolicyExecutor()

        def execute(
            self,
            case: EvaluationCase,
            policy: SimulatorPolicy,
            seed: int,
        ) -> Rollout:
            rollout = self.delegate.execute(case, policy, seed)
            rollout.scenario_id = "wrong-scenario"
            return rollout

    with pytest.raises(M5EvaluationError, match="identity or provenance"):
        run_m5_evaluation(
            SyntheticM5CohortAdapter(),
            WrongIdentityExecutor(),
        )


class _MetricWrapper(Metric):
    spec = PositionErrorMetric.spec

    def __init__(self, *, eligibility_drift: bool = False) -> None:
        self.delegate = PositionErrorMetric()
        self.eligibility_drift = eligibility_drift

    def eligibility(self, scenario) -> MetricEligibility:
        if self.eligibility_drift:
            return MetricEligibility.rejected("no_non_ego_agent")
        return self.delegate.eligibility(scenario)

    def compute(self, scenario, rollout) -> MetricResult:
        result = self.delegate.compute(scenario, rollout)
        if self.eligibility_drift or rollout.sim_name != "idm":
            return result
        distribution = result.distribution[:-1]
        return MetricResult(
            metric_name=result.metric_name,
            metric_version=result.metric_version,
            scenario_id=result.scenario_id,
            value=math.fsum(distribution) / len(distribution),
            distribution=distribution,
            valid=True,
            invalid_reason=None,
            eligible_components=len(distribution),
            total_components=result.total_components,
            details=result.details,
        )


def _metrics_with(wrapper: Metric) -> tuple[Metric, ...]:
    return tuple(
        wrapper if metric.spec.name == wrapper.spec.name else metric
        for metric in m5_metrics()
    )


def test_metric_eligibility_contradiction_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _MetricWrapper(eligibility_drift=True)
    monkeypatch.setattr(
        runner_module,
        "m5_metrics",
        lambda: _metrics_with(wrapper),
    )
    with pytest.raises(M5EvaluationError, match="eligibility"):
        run_m5_evaluation(
            SyntheticM5CohortAdapter(),
            NumpyPolicyExecutor(),
        )


def test_policy_dependent_metric_pairing_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _MetricWrapper()
    monkeypatch.setattr(
        runner_module,
        "m5_metrics",
        lambda: _metrics_with(wrapper),
    )
    with pytest.raises(M5EvaluationError, match="pairing"):
        run_m5_evaluation(
            SyntheticM5CohortAdapter(),
            NumpyPolicyExecutor(),
        )


def test_rows_are_immutable_and_optional_runtimes_remain_lazy(
    synthetic_result,
) -> None:
    with pytest.raises(TypeError):
        synthetic_result.metric_rows[0]["value"] = 7.0  # type: ignore[index]
    with pytest.raises(TypeError):
        synthetic_result.slice_rows[0]["member"] = False  # type: ignore[index]
    code = """
import sys

class BlockOptionalRuntime:
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname.split(".", 1)[0] in {"jax", "tensorflow", "waymax"}:
            raise AssertionError("optional runtime import attempted")
        return None

sys.meta_path.insert(0, BlockOptionalRuntime())
from evalsim.evaluation.m5 import run_synthetic_m5_evaluation
result = run_synthetic_m5_evaluation()
assert len(result.metric_rows) == 195
assert len(result.slice_rows) == 40
assert len(result.scorecard_results) == 312
assert not any(
    name.split(".", 1)[0] in {"jax", "tensorflow", "waymax"}
    for name in sys.modules
)
"""
    completed = subprocess.run(
        (sys.executable, "-c", code),
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
