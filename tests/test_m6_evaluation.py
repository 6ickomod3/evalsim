"""Synthetic end-to-end tests for the source-neutral M6 evaluator."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pytest

import evalsim.evaluation as evaluation_api
import evalsim.evaluation.m6 as m6_evaluation
from evalsim.contracts import (
    Agent,
    AgentType,
    CounterfactualPair,
    EgoTrajectoryPlan,
    InterventionEligibility,
    PairedMetricResult,
    Rollout,
    Scenario,
    SimulatorPolicy,
)
from evalsim.contracts.counterfactual import RolloutSnapshot
from evalsim.evaluation.m6 import (
    M6_NUMPY_POLICY_ACCESS_ROLES,
    M6_NUMPY_POLICY_ORDER,
    M6EligibilityLedger,
    M6EligibilityLedgerEntry,
    M6EvaluationCase,
    M6EvaluationError,
    M6PairedSceneResult,
    M6PrimaryOutcomeBlocked,
    assert_m6_sham_matches_legacy_prefix,
    canonical_m6_policies,
    evaluate_m6_source_eligibility,
    run_m6_numpy_evaluation,
)
from evalsim.metrics.m6 import (
    AdditionalTargetBrakingImpulseMetric,
    ResponseTimelinessMetric,
    is_exactly_nonreactive,
    world_trajectory_tensor_equal,
)
from evalsim.perturb.m6 import (
    PRIMARY_BRAKE_MAGNITUDE_MPS2,
    SECONDARY_BRAKE_MAGNITUDE_MPS2,
    compile_identity_plan,
    compile_longitudinal_brake_pulse_plan,
)
from evalsim.rollout import (
    RolloutEngine,
    TracedRollout,
    policy_trace_prefix_equal,
)
from evalsim.stats.m6 import (
    M6_PRIMARY_METRICS,
    M6_PRIMARY_POLICY_ROLES,
    M6PrimaryCellInput,
    M6SceneEffect,
)

_AGENT_FIELDS = ("valid", "x", "y", "heading", "vx", "vy")


def _agent(
    agent_id: int,
    *,
    x: np.ndarray,
    y: np.ndarray,
    speed_mps: float = 10.0,
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
    case_number: int,
    *,
    frame_count: int = 48,
    current_index: int = 3,
    follower_lateral_m: float = 0.0,
    follower_offset_m: float = -15.0,
) -> Scenario:
    timestamps = np.arange(frame_count, dtype=np.float64) * 0.1
    ego_x = timestamps * 10.0
    ego = _agent(
        100,
        x=ego_x,
        y=np.zeros(frame_count, dtype=np.float64),
    )
    follower = _agent(
        200,
        x=ego_x + follower_offset_m,
        y=np.full(frame_count, follower_lateral_m, dtype=np.float64),
    )
    return Scenario(
        scenario_id=f"m6-evaluation-synthetic-{case_number:03d}",
        timestamps=timestamps,
        agents=[ego, follower],
        ego_index=0,
        metadata={
            "source": "synthetic",
            "source_version": "m6-evaluation-test-v1",
            "source_time_unit": "seconds",
            "current_index": current_index,
        },
    )


def _cases(count: int) -> tuple[M6EvaluationCase, ...]:
    # Deliberately non-contiguous: cohort indices are opaque but must retain exact
    # ascending canonical order.
    return tuple(
        M6EvaluationCase(
            cohort_index=7 + 3 * index,
            scenario=_straight_scenario(index),
        )
        for index in range(count)
    )


@pytest.fixture(scope="module")
def complete_result() -> Any:
    return run_m6_numpy_evaluation(
        _cases(10),
        include_local_secondary=True,
    )


def _metric_signature(result: Any) -> tuple[Any, ...]:
    return tuple(
        (
            scene.cohort_index,
            scene.policy_name,
            scene.pair.baseline_plan.plan_fingerprint,
            scene.pair.intervention_plan.plan_fingerprint,
            tuple(metric.to_dict() for metric in scene.primary_metric_results),
            tuple(metric.to_dict() for metric in scene.secondary_metric_results),
        )
        for scene in result
    )


def test_canonical_policies_and_complete_matrix_order(
    complete_result: Any,
) -> None:
    policies = canonical_m6_policies()
    assert tuple(policy.metadata().name for policy in policies) == (
        "log_replay",
        "constant_velocity",
        "idm",
    )
    assert M6_NUMPY_POLICY_ORDER == tuple(
        role.policy_name for role in M6_PRIMARY_POLICY_ROLES
    )
    assert M6_NUMPY_POLICY_ACCESS_ROLES == (
        "privileged",
        "history_only",
        "history_only",
    )

    expected_rows = tuple(
        (
            role.policy_name,
            role.access_role,
            metric.metric_name,
            metric.metric_version,
        )
        for role in M6_PRIMARY_POLICY_ROLES
        for metric in M6_PRIMARY_METRICS
    )
    assert tuple(
        row.spec.identity for row in complete_result.primary_matrix.rows
    ) == expected_rows
    assert tuple(
        (cell.spec.identity, cell.cohort_indices)
        for cell in complete_result.primary_cell_inputs
    ) == tuple(
        (identity, complete_result.eligibility_ledger.eligible_indices)
        for identity in expected_rows
    )
    assert complete_result.primary_matrix.pair_n == 10


def test_result_rejects_scene_scalar_to_cell_drift(
    complete_result: Any,
) -> None:
    first = complete_result.primary_cell_inputs[0]
    effects = list(first.scene_effects)
    effects[0] = M6SceneEffect(
        cohort_index=effects[0].cohort_index,
        value=effects[0].value + 1.0,
    )
    tampered_cell = M6PrimaryCellInput(
        spec=first.spec,
        scene_effects=tuple(effects),
    )
    with pytest.raises(ValueError, match="scene effects drifted"):
        replace(
            complete_result,
            primary_cell_inputs=(
                tampered_cell,
                *complete_result.primary_cell_inputs[1:],
            ),
        )


def test_nonreactive_policies_and_idm_response_timing(
    complete_result: Any,
) -> None:
    first_index = complete_result.eligibility_ledger.eligible_indices[0]
    by_policy = {
        scene.policy_name: scene
        for scene in complete_result.primary_scene_results
        if scene.cohort_index == first_index
    }
    for policy_name in ("log_replay", "constant_velocity"):
        scene = by_policy[policy_name]
        current = scene.pair.eligibility.current_index
        stop = scene.pair.eligibility.stop_index
        for agent_index, (baseline_agent, treatment_agent) in enumerate(
            zip(
                scene.pair.baseline.agents,
                scene.pair.intervention.agents,
                strict=True,
            )
        ):
            if agent_index == scene.pair.scenario.ego_index:
                continue
            for name in _AGENT_FIELDS:
                np.testing.assert_array_equal(
                    getattr(baseline_agent, name)[current + 1 : stop + 1],
                    getattr(treatment_agent, name)[current + 1 : stop + 1],
                )
        assert scene.world_tensor_equal
        assert scene.exactly_nonreactive
        assert (
            scene.primary_metric(
                "additional_target_braking_impulse_mps"
            ).value
            == 0.0
        )
        assert scene.primary_metric("response_timeliness_s").details[
            "responded"
        ] is False
        assert (
            scene.primary_metric("target_progress_loss_m").value == 0.0
        )

    idm = by_policy["idm"]
    assert not idm.world_tensor_equal
    assert not idm.exactly_nonreactive
    assert (
        idm.primary_metric("additional_target_braking_impulse_mps").value
        > 0.0
    )
    timeliness = idm.primary_metric("response_timeliness_s")
    assert timeliness.details["responded"] is True
    assert timeliness.details["censored"] is False
    assert timeliness.details["response_start_transition"] >= 1

    current = idm.pair.eligibility.current_index
    target = idm.pair.eligibility.target_index
    assert target is not None
    baseline = idm.pair.baseline.agents[target]
    treatment = idm.pair.intervention.agents[target]
    first_divergence = next(
        frame
        for frame in range(current + 1, idm.pair.eligibility.stop_index + 1)
        if any(
            not np.array_equal(
                getattr(baseline, name)[frame : frame + 1],
                getattr(treatment, name)[frame : frame + 1],
            )
            for name in _AGENT_FIELDS
        )
    )
    assert first_divergence == current + 2


def test_scene_results_preserve_responder_censor_details(
    complete_result: Any,
) -> None:
    for scene in complete_result.primary_scene_results:
        timeliness = scene.primary_metric("response_timeliness_s")
        assert set(timeliness.details) >= {
            "responded",
            "censored",
            "event_time_s",
            "restricted_latency_s",
            "response_start_transition",
            "response_end_transition",
        }
    timeliness_cells = tuple(
        cell
        for cell in complete_result.primary_cell_inputs
        if cell.spec.metric_name == "response_timeliness_s"
    )
    assert len(timeliness_cells) == 3
    for cell in timeliness_cells:
        assert all(effect.responded is not None for effect in cell.scene_effects)
        assert all(
            (effect.responder_latency_s is not None) == effect.responded
            for effect in cell.scene_effects
        )


def test_secondary_b4_is_nested_monotone_and_separate(
    complete_result: Any,
) -> None:
    assert len(complete_result.secondary_plan_ledger) == 10
    assert all(entry.feasible for entry in complete_result.secondary_plan_ledger)
    assert len(complete_result.secondary_scene_results) == 30

    primary = complete_result.primary_scene_results[0]
    secondary = complete_result.secondary_scene_results[0]
    assert primary.intervention_magnitude_mps2 == PRIMARY_BRAKE_MAGNITUDE_MPS2
    assert (
        secondary.intervention_magnitude_mps2
        == SECONDARY_BRAKE_MAGNITUDE_MPS2
    )
    b2 = primary.pair.intervention_plan
    b4 = secondary.pair.intervention_plan
    b2_speed = np.hypot(b2.vx, b2.vy)
    b4_speed = np.hypot(b4.vx, b4.vy)
    assert np.all(b4_speed[1:] <= b2_speed[1:])
    assert np.all(b4.x[1:] <= b2.x[1:])
    assert b4.x[0] == b2.x[0]
    assert b4.y[0] == b2.y[0]
    assert len(secondary.secondary_metric_results) == 4
    assert all(
        cell.spec.intervention_config_fingerprint
        == primary.pair.intervention_plan.configuration_fingerprint
        for cell in complete_result.primary_cell_inputs
    )


def test_numpy_evaluation_is_deterministic(complete_result: Any) -> None:
    repeated = run_m6_numpy_evaluation(
        _cases(10),
        include_local_secondary=True,
    )
    assert repeated.eligibility_ledger == complete_result.eligibility_ledger
    assert repeated.primary_matrix.to_local_dict() == (
        complete_result.primary_matrix.to_local_dict()
    )
    assert _metric_signature(repeated.primary_scene_results) == (
        _metric_signature(complete_result.primary_scene_results)
    )
    assert _metric_signature(repeated.secondary_scene_results) == (
        _metric_signature(complete_result.secondary_scene_results)
    )
    for left, right in zip(
        complete_result.primary_scene_results,
        repeated.primary_scene_results,
        strict=True,
    ):
        for left_agent, right_agent in zip(
            left.pair.intervention.agents,
            right.pair.intervention.agents,
            strict=True,
        ):
            for name in _AGENT_FIELDS:
                assert np.array_equal(
                    getattr(left_agent, name),
                    getattr(right_agent, name),
                )


def test_identity_sham_exactly_matches_legacy_numeric_prefix() -> None:
    scenario = _straight_scenario(500)
    identity = compile_identity_plan(scenario)
    engine = RolloutEngine()
    for policy in canonical_m6_policies():
        legacy = engine.run_with_trace(scenario, policy, seed=0)
        sham = engine.run_with_trace(
            scenario,
            policy,
            seed=0,
            ego_plan=identity,
        )
        assert_m6_sham_matches_legacy_prefix(
            scenario,
            legacy.rollout,
            sham.rollout,
            identity,
            legacy_trace=legacy.trace,
            sham_trace=sham.trace,
        )
        stop = sham.rollout.num_steps
        np.testing.assert_array_equal(
            sham.rollout.timestamps,
            legacy.rollout.timestamps[:stop],
        )
        for sham_agent, legacy_agent in zip(
            sham.rollout.agents,
            legacy.rollout.agents,
            strict=True,
        ):
            assert (
                sham_agent.id,
                sham_agent.type,
                sham_agent.length,
                sham_agent.width,
            ) == (
                legacy_agent.id,
                legacy_agent.type,
                legacy_agent.length,
                legacy_agent.width,
            )
            for name in _AGENT_FIELDS:
                np.testing.assert_array_equal(
                    getattr(sham_agent, name),
                    getattr(legacy_agent, name)[:stop],
                )
        assert policy_trace_prefix_equal(sham.trace, legacy.trace)


def test_idm_no_conflict_fixture_is_exactly_nonreactive() -> None:
    scenario = _straight_scenario(600, follower_lateral_m=10.0)
    current = int(scenario.metadata["current_index"])
    eligibility = InterventionEligibility.accepted(
        (current, current + 40),
        target_index=1,
    )
    identity = compile_identity_plan(scenario)
    brake = compile_longitudinal_brake_pulse_plan(
        scenario,
        PRIMARY_BRAKE_MAGNITUDE_MPS2,
    )
    policy = canonical_m6_policies()[2]
    engine = RolloutEngine()
    baseline = engine.run(scenario, policy, seed=0, ego_plan=identity)
    treatment = engine.run(scenario, policy, seed=0, ego_plan=brake)
    pair = CounterfactualPair(
        scenario=scenario,
        baseline=baseline,
        intervention=treatment,
        baseline_plan=identity,
        intervention_plan=brake,
        eligibility=eligibility,
        intervention_identity=brake.perturbation_identity,
    )
    for baseline_agent, treatment_agent in zip(
        pair.baseline.agents[1:],
        pair.intervention.agents[1:],
        strict=True,
    ):
        for name in _AGENT_FIELDS:
            np.testing.assert_array_equal(
                getattr(baseline_agent, name)[current + 1 : current + 41],
                getattr(treatment_agent, name)[current + 1 : current + 41],
            )
    assert world_trajectory_tensor_equal(pair)
    assert is_exactly_nonreactive(pair)
    assert AdditionalTargetBrakingImpulseMetric().compute(pair).value == 0.0
    response = ResponseTimelinessMetric().compute(pair)
    assert response.value == 0.0
    assert response.details["responded"] is False


@dataclass
class _CountingExecutor:
    engine: RolloutEngine = field(default_factory=RolloutEngine)
    calls: list[tuple[str, str, float]] = field(default_factory=list)
    fail_on_call: int | None = None

    def execute(
        self,
        scenario: Scenario,
        policy: SimulatorPolicy,
        *,
        seed: int,
        ego_plan: EgoTrajectoryPlan,
    ) -> TracedRollout:
        self.calls.append(
            (
                scenario.scenario_id,
                policy.metadata().name,
                ego_plan.spec.dose,
            )
        )
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("injected deterministic execution failure")
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
        self.calls.append(
            (
                scenario.scenario_id,
                policy.metadata().name,
                -1.0,
            )
        )
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("injected deterministic execution failure")
        return self.engine.run_with_trace(
            scenario,
            policy,
            seed=seed,
        )


def test_every_input_gets_ledger_entry_and_rejected_case_never_executes() -> None:
    eligible = list(_cases(10))
    rejected = M6EvaluationCase(
        cohort_index=eligible[-1].cohort_index + 3,
        scenario=_straight_scenario(700, frame_count=20, current_index=3),
    )
    executor = _CountingExecutor()
    result = m6_evaluation._run_m6_numpy_evaluation_with_executor(
        (*eligible, rejected),
        executor=executor,
    )
    assert result.eligibility_ledger.input_n == 11
    assert result.eligibility_ledger.eligible_n == 10
    assert result.eligibility_ledger.entries[-1].reason == (
        "insufficient_future_horizon"
    )
    assert len(executor.calls) == 10 * 3 * 3
    assert all(
        scenario_id != rejected.scenario.scenario_id
        for scenario_id, _, _ in executor.calls
    )


def test_n_below_ten_blocks_all_policy_outcomes() -> None:
    executor = _CountingExecutor()
    with pytest.raises(M6PrimaryOutcomeBlocked) as raised:
        m6_evaluation._run_m6_numpy_evaluation_with_executor(
            _cases(9),
            executor=executor,
        )
    assert raised.value.ledger.eligible_n == 9
    assert executor.calls == []


def test_policy_type_and_order_drift_blocks_before_execution() -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        run_m6_numpy_evaluation(
            _cases(10),
            policies=canonical_m6_policies(),  # type: ignore[call-arg]
        )
    assert not hasattr(evaluation_api, "M6TypedPlanExecutor")
    assert not hasattr(evaluation_api, "NumpyM6TypedPlanExecutor")


def test_complete_pair_failure_returns_no_partial_result_and_stable_context() -> None:
    messages = []
    for _ in range(2):
        executor = _CountingExecutor(fail_on_call=3)
        with pytest.raises(M6EvaluationError) as raised:
            m6_evaluation._run_m6_numpy_evaluation_with_executor(
                _cases(10),
                executor=executor,
            )
        messages.append(str(raised.value))
        assert len(executor.calls) == 3
    assert messages[0] == messages[1]
    assert "condition=primary_b2" in messages[0]


def test_full_source_contract_defect_precedes_ordinary_eligibility() -> None:
    malformed = _straight_scenario(800, frame_count=20, current_index=3)
    malformed.agents[0].x[0] = np.nan
    case = M6EvaluationCase(cohort_index=91, scenario=malformed)
    with pytest.raises(M6EvaluationError, match="m6_source_contract_invalid"):
        evaluate_m6_source_eligibility((case,))


def test_opaque_index_order_and_uniqueness_fail_closed() -> None:
    left = M6EvaluationCase(5, _straight_scenario(900))
    right = M6EvaluationCase(8, _straight_scenario(901))
    ledger = evaluate_m6_source_eligibility((left, right))
    assert isinstance(ledger, M6EligibilityLedger)
    assert ledger.eligible_indices == (5, 8)

    with pytest.raises(M6EvaluationError, match="strictly_increasing"):
        evaluate_m6_source_eligibility((right, left))
    duplicate = M6EvaluationCase(5, _straight_scenario(902))
    with pytest.raises(M6EvaluationError, match="strictly_increasing"):
        evaluate_m6_source_eligibility((left, duplicate))


def test_public_evaluator_pins_default_engine_and_has_no_injection_surface() -> None:
    with pytest.raises(TypeError, match="exact default RolloutEngine"):
        m6_evaluation._NumpyM6TypedPlanExecutor(
            engine=RolloutEngine(name="unregistered_engine"),
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        run_m6_numpy_evaluation(
            _cases(10),
            executor=_CountingExecutor(),  # type: ignore[call-arg]
        )


def test_full_canonical_policy_metadata_configuration_is_bound() -> None:
    scenario = _straight_scenario(1000)
    current = int(scenario.metadata["current_index"])
    eligibility = InterventionEligibility.accepted(
        (current, current + 40),
        target_index=1,
    )
    identity = compile_identity_plan(scenario)
    brake = compile_longitudinal_brake_pulse_plan(
        scenario,
        PRIMARY_BRAKE_MAGNITUDE_MPS2,
    )
    policy = canonical_m6_policies()[1]
    engine = RolloutEngine()
    baseline = engine.run(scenario, policy, ego_plan=identity)
    intervention = engine.run(scenario, policy, ego_plan=brake)
    baseline.metadata["policy"]["params"] = {"unregistered": 1}
    intervention.metadata["policy"]["params"] = {"unregistered": 1}
    pair = CounterfactualPair(
        scenario=scenario,
        baseline=baseline,
        intervention=intervention,
        baseline_plan=identity,
        intervention_plan=brake,
        eligibility=eligibility,
        intervention_identity=brake.perturbation_identity,
    )
    with pytest.raises(ValueError, match="metadata/configuration"):
        m6_evaluation._validate_numpy_pair_metadata(
            pair,
            m6_evaluation._canonical_policy_metadata("constant_velocity"),
        )


class _SourceMutatingExecutor(_CountingExecutor):
    def execute(
        self,
        scenario: Scenario,
        policy: SimulatorPolicy,
        *,
        seed: int,
        ego_plan: EgoTrajectoryPlan,
    ) -> TracedRollout:
        traced = self.engine.run_with_trace(
            scenario,
            policy,
            seed=seed,
            ego_plan=ego_plan,
        )
        scenario.agents[0].x[0] += 1.0
        return traced


def test_private_executor_receives_fresh_source_and_mutation_is_detected() -> None:
    with pytest.raises(
        M6EvaluationError,
        match="m6_executor_mutated_defensive_input",
    ):
        m6_evaluation._run_m6_numpy_evaluation_with_executor(
            _cases(10),
            executor=_SourceMutatingExecutor(),
        )


@pytest.mark.parametrize(
    "magnitude",
    [PRIMARY_BRAKE_MAGNITUDE_MPS2, SECONDARY_BRAKE_MAGNITUDE_MPS2],
    ids=["primary_b2", "secondary_b4"],
)
def test_registered_nonreactivity_gate_hard_fails_reactive_pair(
    magnitude: float,
) -> None:
    scenario = _straight_scenario(1100)
    current = int(scenario.metadata["current_index"])
    eligibility = InterventionEligibility.accepted(
        (current, current + 40),
        target_index=1,
    )
    identity = compile_identity_plan(scenario)
    brake = compile_longitudinal_brake_pulse_plan(scenario, magnitude)
    idm = canonical_m6_policies()[2]
    engine = RolloutEngine()
    pair = CounterfactualPair(
        scenario=scenario,
        baseline=engine.run(scenario, idm, ego_plan=identity),
        intervention=engine.run(scenario, idm, ego_plan=brake),
        baseline_plan=identity,
        intervention_plan=brake,
        eligibility=eligibility,
        intervention_identity=brake.perturbation_identity,
    )
    assert not is_exactly_nonreactive(pair)
    with pytest.raises(
        M6EvaluationError,
        match="registered_nonreactivity_gate_failed",
    ):
        m6_evaluation._assert_registered_nonreactivity(
            pair,
            cohort_index=1,
            policy_name="constant_velocity",
            condition="adversarial_repro",
        )


@pytest.mark.parametrize("tamper_details", [False, True])
def test_scene_postconstruction_recomputes_metric_value_and_details(
    complete_result: Any,
    tamper_details: bool,
) -> None:
    scene: M6PairedSceneResult = complete_result.primary_scene_results[0]
    original = scene.primary_metric_results
    first: PairedMetricResult = original[0]
    tampered = (
        replace(first, details={**dict(first.details), "forged": True})
        if tamper_details
        else replace(first, value=first.value + 1.0)
    )
    object.__setattr__(
        scene,
        "primary_metric_results",
        (tampered, *original[1:]),
    )
    try:
        with pytest.raises(ValueError, match="values/details"):
            scene.revalidate()
    finally:
        object.__setattr__(scene, "primary_metric_results", original)
    scene.revalidate()


def test_result_postconstruction_reanalyzes_exact_full_matrix(
    complete_result: Any,
) -> None:
    matrix = complete_result.primary_matrix
    first = matrix.rows[0]
    forged_row = replace(
        first,
        arithmetic_mean=first.arithmetic_mean + 1.0,
    )
    forged_matrix = replace(
        matrix,
        rows=(forged_row, *matrix.rows[1:]),
    )
    object.__setattr__(complete_result, "primary_matrix", forged_matrix)
    try:
        with pytest.raises(ValueError, match="exact 12-cell reanalysis"):
            complete_result.revalidate()
    finally:
        object.__setattr__(complete_result, "primary_matrix", matrix)
    complete_result.revalidate()


def test_result_binds_ledger_to_independently_recomputed_pair_eligibility(
    complete_result: Any,
) -> None:
    ledger = complete_result.eligibility_ledger
    first = ledger.entries[0]
    original_eligibility = first.eligibility
    forged_eligibility = InterventionEligibility.accepted(
        first.eligibility.analysis_window,
        target_index=0,
    )
    object.__setattr__(first, "eligibility", forged_eligibility)
    try:
        with pytest.raises(
            ValueError,
            match="source snapshot|recomputation|ledger entry",
        ):
            complete_result.revalidate()
    finally:
        object.__setattr__(first, "eligibility", original_eligibility)
    complete_result.revalidate()


def test_result_binds_exact_source_snapshot_across_policies(
    complete_result: Any,
) -> None:
    scenes = complete_result.primary_scene_results
    target = scenes[1]
    donor = scenes[4]
    forged = replace(
        donor,
        cohort_index=target.cohort_index,
    )
    original = scenes
    object.__setattr__(
        complete_result,
        "primary_scene_results",
        (scenes[0], forged, *scenes[2:]),
    )
    try:
        with pytest.raises(ValueError, match="source snapshots"):
            complete_result.revalidate()
    finally:
        object.__setattr__(
            complete_result,
            "primary_scene_results",
            original,
        )
    complete_result.revalidate()


def test_scene_trace_rebinds_to_exact_pair_rollout_snapshot(
    complete_result: Any,
) -> None:
    scene = complete_result.primary_scene_results[0]
    donor = complete_result.primary_scene_results[3]
    original = scene.baseline_trace
    object.__setattr__(scene, "baseline_trace", donor.baseline_trace)
    try:
        with pytest.raises(ValueError, match="rollout_fingerprint"):
            scene.revalidate()
    finally:
        object.__setattr__(scene, "baseline_trace", original)
    scene.revalidate()


def test_sham_gate_rejects_every_unregistered_metadata_difference() -> None:
    scenario = _straight_scenario(1200)
    identity = compile_identity_plan(scenario)
    engine = RolloutEngine()
    policy = canonical_m6_policies()[1]
    legacy = engine.run_with_trace(scenario, policy)
    sham = engine.run_with_trace(scenario, policy, ego_plan=identity)
    sham.rollout.metadata["scenario_source_fingerprint"] = "forged"
    with pytest.raises(
        (M6EvaluationError, ValueError),
        match="rollout_fingerprint|metadata_mismatch",
    ):
        assert_m6_sham_matches_legacy_prefix(
            scenario,
            legacy.rollout,
            sham.rollout,
            identity,
            legacy_trace=legacy.trace,
            sham_trace=sham.trace,
        )


def test_scene_rejects_forged_legacy_rollout_fingerprint_and_controls(
    complete_result: Any,
) -> None:
    scene = complete_result.primary_scene_results[1]
    original = scene.legacy_trace
    yaw_rate = np.array(original.yaw_rate, copy=True)
    yaw_rate[0, 1] += 0.5
    forged = replace(
        original,
        yaw_rate=yaw_rate,
        # The digest is deliberately made to look correctly bound. Full semantic
        # replay against the retained immutable legacy rollout must still reject.
        rollout_fingerprint=scene.legacy_rollout._integrity_fingerprint,
    )
    object.__setattr__(scene, "legacy_trace", forged)
    try:
        with pytest.raises(ValueError, match="kinematic controls"):
            scene.revalidate()
    finally:
        object.__setattr__(scene, "legacy_trace", original)
    scene.revalidate()


def test_rejected_ledger_reason_cannot_be_swapped_between_sources() -> None:
    short = M6EvaluationCase(
        cohort_index=1,
        scenario=_straight_scenario(
            1300,
            frame_count=20,
            current_index=3,
        ),
    )
    slow_scenario = _straight_scenario(1301)
    slow_scenario.agents[0].vx[:] = 0.0
    slow_scenario.agents[0].vy[:] = 0.0
    slow = M6EvaluationCase(cohort_index=2, scenario=slow_scenario)
    ledger = evaluate_m6_source_eligibility((short, slow))
    assert tuple(entry.reason for entry in ledger.entries) == (
        "insufficient_future_horizon",
        "ego_speed_below_5_mps",
    )
    first, second = ledger.entries
    with pytest.raises(ValueError, match="retained source snapshot"):
        M6EligibilityLedgerEntry(
            cohort_index=first.cohort_index,
            eligibility=second.eligibility,
            source_snapshot=first.source_snapshot,
        )

    original = first.eligibility
    object.__setattr__(first, "eligibility", second.eligibility)
    try:
        with pytest.raises(ValueError, match="retained source snapshot"):
            first.revalidate()
    finally:
        object.__setattr__(first, "eligibility", original)
    first.revalidate()


def test_equal_horizon_sham_rejects_plus_999_clamp_counts() -> None:
    scenario = _straight_scenario(
        1400,
        frame_count=44,
        current_index=3,
    )
    identity = compile_identity_plan(scenario)
    policy = canonical_m6_policies()[1]
    engine = RolloutEngine()
    legacy = engine.run_with_trace(scenario, policy)
    sham = engine.run_with_trace(scenario, policy, ego_plan=identity)
    assert legacy.rollout.num_steps == sham.rollout.num_steps

    tampered_legacy = copy.deepcopy(legacy.rollout)
    tampered_legacy.metadata["dynamics"]["clamp_counts"]["speed"] += 999
    forged_trace = replace(
        legacy.trace,
        rollout_fingerprint=(
            RolloutSnapshot.from_rollout(
                tampered_legacy
            )._integrity_fingerprint
        ),
    )
    with pytest.raises(ValueError, match="clamp counts"):
        assert_m6_sham_matches_legacy_prefix(
            scenario,
            tampered_legacy,
            sham.rollout,
            identity,
            legacy_trace=forged_trace,
            sham_trace=sham.trace,
        )
    with pytest.raises(
        M6EvaluationError,
        match="equal_horizon_clamp_counts_mismatch",
    ):
        m6_evaluation._assert_sham_metadata_exact(
            scenario,
            tampered_legacy,
            sham.rollout,
            legacy_trace=legacy.trace,
            sham_trace=sham.trace,
        )
