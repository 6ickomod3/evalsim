"""Data-free tests for the pure M6 official execution/row adapter."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

import evalsim.evaluation.m6_official as official
from evalsim.evaluation.m6 import M6EvaluationCase, run_m6_numpy_evaluation
from evalsim.perturb.m6 import (
    SECONDARY_BRAKE_MAGNITUDE_MPS2,
    compile_longitudinal_brake_pulse_plan,
)
from evalsim.sources.m5_m4_reuse import ReloadedM4Member
from evalsim.sources.waymax_loader import WaymaxRecord
from evalsim.stats.m6 import M6_PRIMARY_METRICS, M6_PRIMARY_POLICY_ROLES
from tests.test_m6_evaluation import _straight_scenario


@pytest.fixture(scope="module")
def complete_result():
    cases = tuple(
        M6EvaluationCase(
            cohort_index=index,
            scenario=_straight_scenario(index),
        )
        for index in range(10)
    )
    return run_m6_numpy_evaluation(cases, include_local_secondary=True)


def _record(index: int, state: object) -> WaymaxRecord:
    return WaymaxRecord(
        scenario_id=f"native-private-{index}",
        state=state,
        audit={"private": np.asarray([index], dtype=np.int64)},
        shard_suffix=f"{index % 10:05d}",
        record_ordinal=index,
        shard_sha256=f"{index:064x}",
        dataset_config_fingerprint=f"{index + 1:064x}",
    )


def test_case_adapter_and_collector_detach_native_source_objects() -> None:
    collector = official.M6OfficialCaseCollector()
    retained_inputs: list[tuple[Any, Any, Any]] = []
    for index in reversed(range(128)):
        scenario = _straight_scenario(index)
        state = object()
        record = _record(index, state)
        member = ReloadedM4Member(
            cohort_index=index,
            scenario=scenario,
            record=record,
        )
        retained_inputs.append((member, record, state))
        collector(member)
        scenario.agents[0].x[0] = -999.0

    assert collector.count == 128
    assert not hasattr(collector, "__dict__")
    assert set(collector.__slots__) == {"_snapshots"}
    assert all(
        type(value).__name__ == "ScenarioSnapshot"
        for value in collector._snapshots.values()
    )
    for member, record, state in retained_inputs:
        assert all(value is not member for value in collector._snapshots.values())
        assert all(value is not record for value in collector._snapshots.values())
        assert all(value is not state for value in collector._snapshots.values())

    cases = collector.cases
    assert tuple(case.cohort_index for case in cases) == tuple(range(128))
    assert all(case.scenario.agents[0].x[0] == 0.0 for case in cases)
    first = cases[0]
    first.scenario.agents[0].x[0] = 123.0
    assert collector.cases[0].scenario.agents[0].x[0] == 0.0


def test_collector_rejects_duplicate_and_incomplete_members() -> None:
    collector = official.M6OfficialCaseCollector()
    member = ReloadedM4Member(
        cohort_index=0,
        scenario=_straight_scenario(0),
        record=_record(0, object()),
    )
    collector(member)
    with pytest.raises(
        official.M6OfficialAdapterError,
        match="duplicate_cohort_member",
    ):
        collector(member)
    with pytest.raises(
        official.M6OfficialAdapterError,
        match="cohort_incomplete",
    ):
        _ = collector.cases


def test_eligibility_rows_are_independently_reconstructed(complete_result) -> None:
    rows = official.m6_eligibility_rows(
        complete_result.eligibility_ledger,
        mode="data_free",
        secondary_plan_ledger=complete_result.secondary_plan_ledger,
    )
    expected: list[dict[str, object]] = []
    for entry in complete_result.eligibility_ledger.entries:
        feasible: bool | None = None
        if entry.eligible:
            plan = compile_longitudinal_brake_pulse_plan(
                entry.source_snapshot.to_scenario(),
                SECONDARY_BRAKE_MAGNITUDE_MPS2,
            )
            plan.revalidate()
            feasible = True
        expected.append(
            {
                "cohort_index": entry.cohort_index,
                "primary_eligible": entry.eligible,
                "rejection_reason": entry.reason,
                "secondary_b4_feasible": feasible,
            }
        )
    assert [dict(row) for row in rows] == expected

    bad_ledger = list(complete_result.secondary_plan_ledger)
    bad_ledger[0] = type(bad_ledger[0])(
        cohort_index=bad_ledger[0].cohort_index,
        feasible=False,
        reason="secondary_ego_plan_infeasible",
    )
    with pytest.raises(
        official.M6OfficialAdapterError,
        match="secondary_plan_ledger_drifted",
    ):
        official.m6_eligibility_rows(
            complete_result.eligibility_ledger,
            mode="data_free",
            secondary_plan_ledger=bad_ledger,
        )


def _expected_scene_rows(scenes) -> list[dict[str, object]]:
    units = {metric.metric_name: metric.value_unit for metric in M6_PRIMARY_METRICS}
    expected: list[dict[str, object]] = []
    for scene in scenes:
        for result in scene.primary_metric_results:
            if result.metric_name == "response_timeliness_s":
                responded = result.details["responded"]
                latency = result.details["event_time_s"] if responded else None
            else:
                responded = None
                latency = None
            expected.append(
                {
                    "cohort_index": scene.cohort_index,
                    "policy_name": scene.policy_name,
                    "policy_access_role": scene.policy_access_role,
                    "metric_name": result.metric_name,
                    "metric_version": result.metric_version,
                    "unit": units[result.metric_name],
                    "value": result.value,
                    "responded": responded,
                    "responder_latency_s": latency,
                    "source_pairing_complete": True,
                    "intervention_config_fingerprint": (
                        scene.pair.intervention_plan.configuration_fingerprint
                    ),
                }
            )
    return expected


def test_numpy_result_rows_match_independent_projection(complete_result) -> None:
    primary = official.m6_primary_scene_scalar_rows(complete_result)
    secondary = official.m6_secondary_scene_scalar_rows(complete_result)
    assert [dict(row) for row in primary] == _expected_scene_rows(
        complete_result.primary_scene_results
    )
    assert [dict(row) for row in secondary] == _expected_scene_rows(
        complete_result.secondary_scene_results
    )
    assert len(primary) == 10 * 3 * 4
    assert len(secondary) == 10 * 3 * 4

    encoded = json.dumps(
        [dict(row) for row in (*primary, *secondary)],
        allow_nan=False,
        sort_keys=True,
    )
    for scene in (
        *complete_result.primary_scene_results,
        *complete_result.secondary_scene_results,
    ):
        assert scene.pair.scenario.scenario_id not in encoded
        for agent in scene.pair.scenario.agents:
            assert f'"agent_id": {agent.id}' not in encoded
    assert "scenario_id" not in encoded
    assert "target_agent_id" not in encoded
    assert "target_index" not in encoded


def test_negative_observation_domain_and_hashes_are_exact(complete_result) -> None:
    rows = official.m6_negative_timing_observation_rows(complete_result)
    feasible = tuple(
        entry.cohort_index
        for entry in complete_result.secondary_plan_ledger
        if entry.feasible
    )
    expected_keys: list[tuple[str, int, str | None]] = []
    policy_names = tuple(role.policy_name for role in M6_PRIMARY_POLICY_ROLES)
    gate_policies = (
        ("log_replay_world_tensor_equality", ("log_replay",)),
        ("constant_velocity_world_tensor_equality", ("constant_velocity",)),
        ("sham_legacy_equality", policy_names),
        ("synchronous_response_floor", policy_names),
        ("primary_plan_feasibility", (None,)),
        ("nested_dose_monotonicity", (None,)),
    )
    for gate_name, policies in gate_policies:
        indices = (
            feasible
            if gate_name == "nested_dose_monotonicity"
            else complete_result.eligibility_ledger.eligible_indices
        )
        expected_keys.extend(
            (gate_name, cohort_index, policy_name)
            for cohort_index in indices
            for policy_name in policies
        )
    assert [
        (row["gate_name"], row["cohort_index"], row["policy_name"])
        for row in rows
    ] == expected_keys
    assert len(rows) == 10 * 9 + 10
    assert all(row["assessed_n"] == 1 for row in rows)
    assert all(row["violation_n"] == 0 for row in rows)
    assert all(
        len(row["observation_sha256"]) == 64
        and set(row["observation_sha256"]) <= set("0123456789abcdef")
        for row in rows
    )
    assert rows == official.m6_negative_timing_observation_rows(complete_result)
    encoded = json.dumps([dict(row) for row in rows], sort_keys=True)
    assert "scenario_id" not in encoded
    assert "agent_id" not in encoded


def _eligibility_row() -> dict[str, object]:
    return {
        "cohort_index": 0,
        "primary_eligible": True,
        "rejection_reason": None,
        "secondary_b4_feasible": True,
    }


def _scene_row(value: float = 0.0) -> dict[str, object]:
    return {
        "cohort_index": 0,
        "policy_name": "log_replay",
        "policy_access_role": "privileged",
        "metric_name": "additional_target_braking_impulse_mps",
        "metric_version": "1.0.0",
        "unit": "m/s",
        "value": value,
        "responded": None,
        "responder_latency_s": None,
        "source_pairing_complete": True,
        "intervention_config_fingerprint": "a" * 64,
    }


def _negative_row() -> dict[str, object]:
    return {
        "gate_name": "primary_plan_feasibility",
        "cohort_index": 0,
        "policy_name": None,
        "assessed_n": 1,
        "violation_n": 0,
        "observation_sha256": "b" * 64,
    }




def test_official_numpy_orchestrator_uses_two_fresh_complete_passes(
    monkeypatch: pytest.MonkeyPatch,
    complete_result,
) -> None:
    cases = tuple(
        M6EvaluationCase(index, _straight_scenario(index))
        for index in range(128)
    )
    original_scenario_ids = {id(case.scenario) for case in cases}
    calls: list[tuple[M6EvaluationCase, ...]] = []
    results = [complete_result, complete_result]

    def fake_evaluate(received, observer, clock_ns):
        materialized = tuple(received)
        assert callable(clock_ns)
        for index, phase in enumerate(official._NUMPY_EXECUTION_PHASES, start=1):
            observer(phase, index * 1_000_000)
        assert tuple(case.cohort_index for case in materialized) == tuple(range(128))
        assert not original_scenario_ids.intersection(
            id(case.scenario) for case in materialized
        )
        calls.append(materialized)
        return results[len(calls) - 1]

    monkeypatch.setattr(
        official,
        "_run_m6_numpy_evaluation_with_phase_observer",
        fake_evaluate,
    )
    monkeypatch.setattr(
        official,
        "m6_eligibility_rows",
        lambda *_args, **_kwargs: (_eligibility_row(),),
    )
    monkeypatch.setattr(
        official,
        "m6_primary_scene_scalar_rows",
        lambda *_args, **_kwargs: (_scene_row(),),
    )
    monkeypatch.setattr(
        official,
        "m6_secondary_scene_scalar_rows",
        lambda *_args, **_kwargs: (_scene_row(),),
    )
    monkeypatch.setattr(
        official,
        "m6_negative_timing_observation_rows",
        lambda *_args, **_kwargs: (_negative_row(),),
    )

    wall_ticks = iter((0, 20_000_000))
    monkeypatch.setattr(
        official.time, "monotonic_ns", lambda: next(wall_ticks)
    )

    rows = official.run_m6_official_numpy(cases)
    assert len(calls) == 2
    assert all(
        left.scenario is not right.scenario
        for left, right in zip(calls[0], calls[1], strict=True)
    )
    assert rows.typed_result is complete_result
    assert dict(rows.primary_scene_scalar_rows[0]) == _scene_row()
    assert dict(rows.primary_repeat_scene_scalar_rows[0]) == _scene_row()
    assert dict(rows.phase_durations_ms) == {
        "numpy_rollouts": 2,
        "paired_metrics": 4,
        "statistics": 6,
        "verification": 8,
    }
    assert not hasattr(rows, "primary_result")
    assert not hasattr(rows, "repeat_result")


def test_official_numpy_orchestrator_rejects_repeat_drift(
    monkeypatch: pytest.MonkeyPatch,
    complete_result,
) -> None:
    cases = tuple(
        M6EvaluationCase(index, _straight_scenario(index))
        for index in range(128)
    )
    def fake_evaluate(_received, observer, _clock):
        for phase in official._NUMPY_EXECUTION_PHASES:
            observer(phase, 1)
        return complete_result

    monkeypatch.setattr(
        official,
        "_run_m6_numpy_evaluation_with_phase_observer",
        fake_evaluate,
    )
    monkeypatch.setattr(
        official,
        "m6_eligibility_rows",
        lambda *_args, **_kwargs: (_eligibility_row(),),
    )
    calls = 0

    def drifting_primary(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return (_scene_row(float(calls - 1)),)

    monkeypatch.setattr(official, "m6_primary_scene_scalar_rows", drifting_primary)
    monkeypatch.setattr(
        official,
        "m6_secondary_scene_scalar_rows",
        lambda *_args, **_kwargs: (_scene_row(),),
    )
    monkeypatch.setattr(
        official,
        "m6_negative_timing_observation_rows",
        lambda *_args, **_kwargs: (_negative_row(),),
    )
    with pytest.raises(
        official.M6OfficialAdapterError,
        match="numpy_repeat_drifted",
    ):
        official.run_m6_official_numpy(cases)
