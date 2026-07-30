"""Tests for the outcome-suppressed M6 NumPy pilot."""
from __future__ import annotations

import dataclasses
from dataclasses import fields
import json

import pytest

import evalsim.evaluation.m6 as m6_evaluation
from evalsim.evaluation.m6 import (
    M6EligibilityLedger,
    M6EvaluationCase,
    M6EvaluationError,
    evaluate_m6_source_eligibility,
)
from evalsim.evaluation.m6_pilot import (
    M6NumpyPilotObservation,
    m6_numpy_pilot_selected_cohort_indices_sha256,
    run_m6_numpy_pilot,
)
from tests.test_m6_evaluation import _straight_scenario


@pytest.fixture(scope="module")
def complete_source() -> tuple[
    tuple[M6EvaluationCase, ...],
    M6EligibilityLedger,
]:
    cases = tuple(
        M6EvaluationCase(
            cohort_index=index,
            scenario=_straight_scenario(index),
        )
        for index in range(128)
    )
    return cases, evaluate_m6_source_eligibility(cases)


def _clock(values: list[int]):
    iterator = iter(values)
    return lambda: next(iterator)


def test_pilot_executes_caller_order_and_returns_timing_only(
    complete_source,
) -> None:
    cases, ledger = complete_source
    observation = run_m6_numpy_pilot(
        cases,
        ledger,
        (7, 3),
        selection_binding_sha256="a" * 64,
        clock_ns=_clock(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                1_000_009,
                2_000_000,
                4_500_001,
            ]
        ),
    )
    assert observation.scene_count == 2
    assert observation.scene_durations_ms == (2, 3)
    assert observation.total_execution_ms == 5
    assert observation.max_scene_ms == 3
    assert observation.to_summary_fields() == {
        "max_scene_ms": 3,
        "numpy_ms": 5,
        "pilot_scene_n": 2,
        "selected_cohort_indices_sha256": (
            m6_numpy_pilot_selected_cohort_indices_sha256((7, 3))
        ),
    }
    observation.revalidate()
    names = {item.name for item in fields(observation)}
    assert names == {
        "scene_count",
        "scene_durations_ms",
        "total_execution_ms",
        "max_scene_ms",
        "selected_cohort_indices_sha256",
        "source_selection_binding_sha256",
        "execution_binding_sha256",
        "schema_version",
        "_issued_binding_sha256",
    }
    assert not any(
        fragment in names
        for fragment in {
            "cohort_index",
            "rollout",
            "trace",
            "metric",
            "responded",
            "value",
            "sign",
        }
    )
    encoded = json.dumps(observation.to_summary_fields(), sort_keys=True)
    assert "ordered_cohort_indices" not in encoded
    assert '"cohort_index"' not in encoded
    assert "value" not in encoded
    assert "respond" not in encoded
    with pytest.raises(TypeError, match="runner-issued"):
        dataclasses.replace(observation, total_execution_ms=6)

    original_durations = observation.scene_durations_ms
    original_total = observation.total_execution_ms
    original_max = observation.max_scene_ms
    original_binding = observation._issued_binding_sha256
    object.__setattr__(observation, "scene_durations_ms", (4, 5))
    object.__setattr__(observation, "total_execution_ms", 9)
    object.__setattr__(observation, "max_scene_ms", 5)
    object.__setattr__(
        observation,
        "_issued_binding_sha256",
        observation._public_fields_sha256(),
    )
    try:
        with pytest.raises(ValueError, match="integrity binding"):
            observation.revalidate()
    finally:
        object.__setattr__(
            observation,
            "scene_durations_ms",
            original_durations,
        )
        object.__setattr__(observation, "total_execution_ms", original_total)
        object.__setattr__(observation, "max_scene_ms", original_max)
        object.__setattr__(
            observation,
            "_issued_binding_sha256",
            original_binding,
        )
    observation.revalidate()


def test_pilot_selection_binding_preserves_caller_order(
    complete_source,
) -> None:
    cases, ledger = complete_source
    first = run_m6_numpy_pilot(
        cases,
        ledger,
        (1, 2),
        selection_binding_sha256="b" * 64,
        clock_ns=_clock(list(range(12))),
    )
    second = run_m6_numpy_pilot(
        cases,
        ledger,
        (2, 1),
        selection_binding_sha256="b" * 64,
        clock_ns=_clock(list(range(12))),
    )
    assert first.scene_durations_ms == second.scene_durations_ms == (1, 1)
    assert first.source_selection_binding_sha256 == "b" * 64
    assert second.source_selection_binding_sha256 == "b" * 64
    assert first.execution_binding_sha256 != second.execution_binding_sha256
    assert first.selected_cohort_indices_sha256 == (
        m6_numpy_pilot_selected_cohort_indices_sha256((1, 2))
    )
    assert second.selected_cohort_indices_sha256 == (
        m6_numpy_pilot_selected_cohort_indices_sha256((2, 1))
    )
    assert (
        first.selected_cohort_indices_sha256
        != second.selected_cohort_indices_sha256
    )


def test_pilot_uses_exact_caller_scene_and_builtin_policy_order(
    monkeypatch: pytest.MonkeyPatch,
    complete_source,
) -> None:
    cases, ledger = complete_source
    observed: list[tuple[str, int, str]] = []
    clock_values = iter(
        (
            0,
            1_500_000,
            2_000_000,
            3_500_000,
            4_000_000,
            5_500_000,
            6_000_000,
            7_500_000,
            8_000_000,
            8_500_000,
            9_000_000,
            9_500_000,
        )
    )
    clock_calls: list[int] = []

    def clock_ns() -> int:
        value = next(clock_values)
        clock_calls.append(value)
        return value

    class _ValidatedScene:
        def revalidate(self) -> None:
            return None

    def execute_rollouts(execution, case, **kwargs):
        del execution
        if not observed:
            assert len(clock_calls) == 9
        observed.append(
            ("rollouts", case.cohort_index, kwargs["policy_name"])
        )
        return object()

    def analyze(products, case, **kwargs):
        del products
        observed.append(
            ("metrics", case.cohort_index, kwargs["policy_name"])
        )
        return _ValidatedScene(), _ValidatedScene()

    monkeypatch.setattr(
        m6_evaluation,
        "_execute_prepared_policy_rollouts",
        execute_rollouts,
    )
    monkeypatch.setattr(m6_evaluation, "_analyze_prepared_policy", analyze)
    result = run_m6_numpy_pilot(
        cases,
        ledger,
        (7, 3),
        selection_binding_sha256="9" * 64,
        clock_ns=clock_ns,
    )
    assert result.scene_count == 2
    assert result.scene_durations_ms == (4, 4)
    assert observed == [
        (stage, cohort_index, policy_name)
        for cohort_index in (7, 3)
        for policy_name in (
            "log_replay",
            "constant_velocity",
            "idm",
        )
        for stage in ("rollouts", "metrics")
    ]
    assert not hasattr(m6_evaluation, "_execute_prepared_policy")


@pytest.mark.parametrize(
    "selection",
    (
        (),
        tuple(range(9)),
        (1, 1),
    ),
)
def test_pilot_rejects_unbounded_or_duplicate_selection(
    complete_source,
    selection,
) -> None:
    cases, ledger = complete_source
    with pytest.raises(M6EvaluationError):
        run_m6_numpy_pilot(
            cases,
            ledger,
            selection,
            selection_binding_sha256="c" * 64,
            clock_ns=lambda: 1,
        )


def test_pilot_rejects_stale_full_source_ledger() -> None:
    cases = tuple(
        M6EvaluationCase(
            cohort_index=index,
            scenario=_straight_scenario(index),
        )
        for index in range(128)
    )
    ledger = evaluate_m6_source_eligibility(cases)
    cases[0].scenario.agents[0].x[0] += 1.0
    with pytest.raises(M6EvaluationError, match="source_ledger_drifted"):
        run_m6_numpy_pilot(
            cases,
            ledger,
            (0,),
            selection_binding_sha256="d" * 64,
            clock_ns=lambda: 1,
        )


def test_pilot_rejects_primary_ineligible_selection() -> None:
    scenarios = [_straight_scenario(index) for index in range(128)]
    scenarios[0].agents[0].vx[:] = 0.0
    scenarios[0].agents[0].vy[:] = 0.0
    cases = tuple(
        M6EvaluationCase(cohort_index=index, scenario=scenario)
        for index, scenario in enumerate(scenarios)
    )
    ledger = evaluate_m6_source_eligibility(cases)
    assert ledger.entries[0].reason == "ego_speed_below_5_mps"
    with pytest.raises(
        M6EvaluationError,
        match="selection_contains_ineligible_case",
    ):
        run_m6_numpy_pilot(
            cases,
            ledger,
            (0,),
            selection_binding_sha256="e" * 64,
            clock_ns=lambda: 1,
        )


def test_pilot_requires_advancing_integer_clock(complete_source) -> None:
    cases, ledger = complete_source
    with pytest.raises(M6EvaluationError, match="clock_did_not_advance"):
        run_m6_numpy_pilot(
            cases,
            ledger,
            (0,),
            selection_binding_sha256="e" * 64,
            clock_ns=_clock([0, 1, 2, 3, 100, 100]),
        )


def test_pilot_observation_cannot_be_caller_minted() -> None:
    with pytest.raises(TypeError, match="runner-issued"):
        M6NumpyPilotObservation(
            scene_count=1,
            scene_durations_ms=(1,),
            total_execution_ms=1,
            max_scene_ms=1,
            selected_cohort_indices_sha256=(
                m6_numpy_pilot_selected_cohort_indices_sha256((0,))
            ),
            source_selection_binding_sha256="f" * 64,
            execution_binding_sha256="e" * 64,
        )
