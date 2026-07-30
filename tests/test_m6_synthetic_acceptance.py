"""Adversarial tests for the independently audited M6 synthetic acceptance."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from evalsim.contracts import (
    HistoryOnlyPolicyContext,
    HistoryOnlyPolicyObservation,
    HistoryOnlySimulatorPolicy,
    InterventionEligibility,
    PairedMetricResult,
    PolicyMetadata,
    PolicyStep,
    PrivilegedPolicyContext,
    PrivilegedSimulatorPolicy,
)
from evalsim.evaluation import m6_synthetic as synthetic
from evalsim.evaluation.m6_synthetic import (
    M6_SYNTHETIC_ACCEPTANCE_VERSION,
    M6_SYNTHETIC_CASE_COUNT,
    M6_SYNTHETIC_CURRENT_INDEX,
    M6_SYNTHETIC_DT_S,
    M6_SYNTHETIC_FRAME_COUNT,
    M6_SYNTHETIC_MIN_SENTINEL_EXTRA_HARD_BRAKING_S,
    M6_SYNTHETIC_MIN_SENTINEL_EXTRA_PROGRESS_LOSS_M,
    M6AnalyticOracleScope,
    M6IndependentMeasures,
    M6SyntheticAcceptanceResult,
    M6SyntheticOracleResult,
    run_m6_synthetic_acceptance,
    synthetic_m6_cases,
    synthetic_m6_source_evidence,
)
from evalsim.metrics import m6 as production_metrics
from evalsim.rollout import DynamicsLimits, RolloutEngine
from evalsim.simulators import IDMPolicy


@pytest.fixture(scope="module")
def acceptance() -> M6SyntheticAcceptanceResult:
    return run_m6_synthetic_acceptance()


@pytest.fixture(scope="module")
def no_conflict() -> M6SyntheticOracleResult:
    return synthetic._build_no_conflict_oracle()


@pytest.fixture(scope="module")
def sentinel() -> M6SyntheticOracleResult:
    return synthetic._build_sentinel_oracle()


class _HistoryWrapper(HistoryOnlySimulatorPolicy):
    def __init__(self) -> None:
        self.delegate = IDMPolicy()

    def initialize(
        self,
        context: HistoryOnlyPolicyContext,
        seed: int,
    ) -> Any:
        return self.delegate.initialize(context, seed)

    def step(
        self,
        state: Any,
        observation: HistoryOnlyPolicyObservation,
    ) -> PolicyStep:
        return self.delegate.step(state, observation)

    def metadata(self) -> PolicyMetadata:
        return self.delegate.metadata()


class _PrivilegedWrapper(PrivilegedSimulatorPolicy):
    def initialize(
        self,
        context: PrivilegedPolicyContext,
        seed: int,
    ) -> Any:
        raise AssertionError("exact-type rejection must precede initialization")

    def step(
        self,
        state: Any,
        observation: HistoryOnlyPolicyObservation,
    ) -> PolicyStep:
        raise AssertionError("exact-type rejection must precede execution")

    def metadata(self) -> PolicyMetadata:
        return IDMPolicy().metadata()


class _DualWrapper(
    HistoryOnlySimulatorPolicy,
    PrivilegedSimulatorPolicy,
):
    def initialize(self, context: Any, seed: int) -> Any:
        raise AssertionError("exact-type rejection must precede initialization")

    def step(
        self,
        state: Any,
        observation: HistoryOnlyPolicyObservation,
    ) -> PolicyStep:
        raise AssertionError("exact-type rejection must precede execution")

    def metadata(self) -> PolicyMetadata:
        return IDMPolicy().metadata()


class _CustomEngine(RolloutEngine):
    pass


def test_fixed_source_is_ten_ids_over_one_labeled_geometry() -> None:
    left = synthetic_m6_cases()
    right = synthetic_m6_cases()
    assert len(left) == len(right) == M6_SYNTHETIC_CASE_COUNT
    assert tuple(case.cohort_index for case in left) == tuple(range(10))
    assert len({case.scenario.scenario_id for case in left}) == 10
    for index, case in enumerate(left):
        scenario = case.scenario
        assert scenario.num_steps == M6_SYNTHETIC_FRAME_COUNT
        assert scenario.num_agents == 2
        assert scenario.metadata["analytic_geometry_id"] == (
            "aligned_straight_follower_v1"
        )
        assert scenario.metadata["analytic_unique_geometry_count"] == 1
        assert scenario.metadata["analytic_replica_count"] == 10
        assert scenario.metadata["analytic_replica_index"] == index
        assert scenario.metadata["analytic_replica_role"] == (
            "deterministic_replica"
        )
        assert scenario.metadata["current_index"] == (
            M6_SYNTHETIC_CURRENT_INDEX
        )
        np.testing.assert_array_equal(
            scenario.timestamps,
            np.arange(M6_SYNTHETIC_FRAME_COUNT, dtype=np.float64)
            * M6_SYNTHETIC_DT_S,
        )
        np.testing.assert_array_equal(
            scenario.agents[1].x - scenario.agents[0].x,
            np.full(M6_SYNTHETIC_FRAME_COUNT, -15.0),
        )
        np.testing.assert_array_equal(
            scenario.agents[0].x,
            right[index].scenario.agents[0].x,
        )
    left[0].scenario.agents[0].x[0] = 12345.0
    assert right[0].scenario.agents[0].x[0] == 0.0

    evidence = synthetic_m6_source_evidence()
    assert evidence["case_count"] == 10
    assert evidence["deterministic_replica_count"] == 10
    assert evidence["unique_analytic_geometry_count"] == 1
    assert evidence["replica_semantics"] == (
        "ten deterministic IDs over one unique analytic geometry"
    )
    with pytest.raises(TypeError):
        evidence["case_count"] = 11  # type: ignore[index]


def test_no_conflict_retains_real_rejection_in_separate_scope(
    no_conflict: M6SyntheticOracleResult,
) -> None:
    no_conflict.revalidate()
    scope = no_conflict.scope
    assert isinstance(scope, M6AnalyticOracleScope)
    assert scope.primary_eligibility.eligible is False
    assert scope.primary_eligibility.reason == "no_stable_aligned_follower"
    assert scope.primary_eligibility.target_index is None
    assert scope.primary_eligibility.analysis_window == (3, 43)
    assert scope.target_index == 1
    assert scope.current_index == 3
    assert scope.stop_index == 43
    assert len(scope.source_fingerprint or "") == 64
    assert no_conflict.production_metric_results == ()
    raw = no_conflict.independent_measures
    assert raw.world_tensor_equal
    assert raw.structurally_nonreactive
    assert raw.response_responded is False
    assert raw.additional_target_braking_impulse_mps == 0.0
    assert raw.target_progress_loss_m == 0.0
    assert raw.additional_hard_braking_exposure_s == 0.0
    # Relational gap changes because ego moved; it is deliberately excluded from
    # structural world-response truth.
    assert raw.minimum_longitudinal_bumper_gap_change_m == -7.0


def test_analytic_oracles_bind_exact_engine_policy_plans_and_traces(
    no_conflict: M6SyntheticOracleResult,
    sentinel: M6SyntheticOracleResult,
) -> None:
    for oracle in (no_conflict, sentinel):
        oracle.revalidate()
        assert oracle.policy_access_role == "history_only"
        assert oracle.baseline_rollout.seed == 0
        assert oracle.intervention_rollout.seed == 0
        assert oracle.baseline_trace.policy_access_role == "history_only"
        assert oracle.intervention_trace.policy_access_role == "history_only"
        assert oracle.baseline_trace.perturbation_identity == (
            oracle.baseline_plan.perturbation_identity
        )
        assert oracle.intervention_trace.perturbation_identity == (
            oracle.intervention_plan.perturbation_identity
        )
        oracle.baseline_trace.validate_for_rollout(
            oracle.baseline_rollout
        )
        oracle.intervention_trace.validate_for_rollout(
            oracle.intervention_rollout
        )
        assert oracle.baseline_rollout.metadata["engine"]["name"] == (
            "numpy_rollout_engine"
        )
        assert oracle.baseline_rollout.metadata["dynamics"]["limits"] == (
            DynamicsLimits().to_dict()
        )
        assert oracle.baseline_rollout.scenario_id == oracle.scope.scenario_id
        assert oracle.baseline_plan.spec.family == "identity"
        assert oracle.intervention_plan.spec.family == (
            "longitudinal_brake_pulse"
        )

    assert no_conflict.policy_type_identity.endswith(".IDMPolicy")
    assert sentinel.policy_type_identity.endswith(
        "._OverreactiveSentinelPolicy"
    )
    assert sentinel.nominal_identity_rollout is not None
    assert sentinel.nominal_identity_trace is not None
    assert synthetic._rollout_tensors_equal(
        sentinel.baseline_rollout,
        sentinel.nominal_identity_rollout,
    )
    assert synthetic._trace_action_tensors_equal(
        sentinel.baseline_trace,
        sentinel.nominal_identity_trace,
    )
    metrics = {
        result.metric_name: result
        for result in sentinel.production_metric_results
    }
    response_details = metrics["response_timeliness_s"].details
    assert response_details["acceleration_threshold_mps2"] == -0.5
    assert response_details["persistence_s"] == 0.2
    assert response_details["search_start_transition"] == 1
    assert metrics["additional_hard_braking_exposure_s"].details[
        "inclusive_acceleration_threshold_mps2"
    ] == -4.0


def test_false_primary_eligibility_attack_is_rejected(
    no_conflict: M6SyntheticOracleResult,
) -> None:
    scope = no_conflict.scope
    forged = InterventionEligibility.accepted(
        (scope.current_index, scope.stop_index),
        target_index=scope.target_index,
    )
    with pytest.raises(ValueError, match="eligibility drifted"):
        replace(scope, primary_eligibility=forged)


@pytest.mark.parametrize(
    "policy",
    [_HistoryWrapper(), _PrivilegedWrapper(), _DualWrapper()],
    ids=["history-wrapper", "privileged-wrapper", "dual-capability"],
)
def test_private_execution_seam_rejects_nonexact_or_privileged_policy(
    no_conflict: M6SyntheticOracleResult,
    policy: Any,
) -> None:
    with pytest.raises(TypeError, match="exact registered history-only type"):
        synthetic._execute_analytic_oracle(
            no_conflict.scope,
            policy,
        )


def test_private_execution_seam_rejects_custom_engine_and_limits(
    no_conflict: M6SyntheticOracleResult,
) -> None:
    with pytest.raises(TypeError, match="exact default RolloutEngine"):
        synthetic._execute_analytic_oracle(
            no_conflict.scope,
            IDMPolicy(),
            engine=_CustomEngine(),
        )
    with pytest.raises(TypeError, match="exact default RolloutEngine"):
        synthetic._execute_analytic_oracle(
            no_conflict.scope,
            IDMPolicy(),
            engine=RolloutEngine(
                dynamics_limits=DynamicsLimits(
                    max_deceleration_mps2=7.0,
                )
            ),
        )


def test_scope_rejects_source_swap_id_and_rehashed_geometry_drift(
    no_conflict: M6SyntheticOracleResult,
) -> None:
    scope = no_conflict.scope
    with pytest.raises(ValueError, match="source identity"):
        replace(scope, scenario_id="m6-data-free-no-conflict-swapped")

    with pytest.raises(ValueError, match="source identity"):
        M6AnalyticOracleScope.from_source(
            scope.fixture_id,
            synthetic._aligned_scenario(0),
            target_index=1,
        )

    drifted = synthetic._no_conflict_scenario()
    drifted.agents[1].y[:] = 9.5
    with pytest.raises(ValueError, match="source identity"):
        M6AnalyticOracleScope.from_source(
            scope.fixture_id,
            drifted,
            target_index=1,
        )

    id_drift = synthetic._no_conflict_scenario()
    id_drift.scenario_id = "m6-data-free-no-conflict-other"
    with pytest.raises(ValueError, match="source identity"):
        M6AnalyticOracleScope.from_source(
            scope.fixture_id,
            id_drift,
            target_index=1,
        )


def test_sentinel_sham_drift_and_nominal_sham_swap_fail_closed(
    sentinel: M6SyntheticOracleResult,
) -> None:
    with pytest.raises(ValueError):
        replace(
            sentinel,
            baseline_rollout=sentinel.intervention_rollout,
            baseline_trace=sentinel.intervention_trace,
        )
    with pytest.raises(ValueError):
        replace(
            sentinel,
            nominal_identity_rollout=sentinel.intervention_rollout,
            nominal_identity_trace=sentinel.intervention_trace,
        )


def test_trace_swap_and_access_role_drift_fail_closed(
    sentinel: M6SyntheticOracleResult,
) -> None:
    with pytest.raises(ValueError):
        replace(
            sentinel,
            baseline_trace=sentinel.intervention_trace,
        )
    privileged_trace = replace(
        sentinel.baseline_trace,
        policy_access_role="privileged",
    )
    with pytest.raises(ValueError, match="access"):
        replace(
            sentinel,
            baseline_trace=privileged_trace,
        )


def test_independent_measures_and_retained_metrics_reject_mutation(
    sentinel: M6SyntheticOracleResult,
) -> None:
    raw = sentinel.independent_measures
    with pytest.raises(ValueError, match="independent raw measures drifted"):
        replace(
            sentinel,
            independent_measures=replace(
                raw,
                target_progress_loss_m=raw.target_progress_loss_m + 1.0,
            ),
        )

    original = sentinel.production_metric_results[0]
    tampered = PairedMetricResult(
        metric_name=original.metric_name,
        metric_version=original.metric_version,
        scenario_id=original.scenario_id,
        intervention_identity=original.intervention_identity,
        value=original.value + 1.0,
        details=original.details,
    )
    with pytest.raises(ValueError, match="production metric results drifted"):
        replace(
            sentinel,
            production_metric_results=(
                tampered,
                *sentinel.production_metric_results[1:],
            ),
        )


def test_production_comparator_mutants_do_not_define_oracle_truth(
    monkeypatch: pytest.MonkeyPatch,
    no_conflict: M6SyntheticOracleResult,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("production comparator must not define oracle truth")

    monkeypatch.setattr(
        production_metrics,
        "is_exactly_nonreactive",
        forbidden,
    )
    monkeypatch.setattr(
        production_metrics,
        "world_trajectory_tensor_equal",
        forbidden,
    )
    no_conflict.revalidate()
    assert no_conflict.independent_measures.structurally_nonreactive


def test_production_only_prereg_threshold_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    no_conflict: M6SyntheticOracleResult,
) -> None:
    attacks = (
        ("M6_RESPONSE_ACCELERATION_THRESHOLD_MPS2", -0.75),
        ("M6_RESPONSE_PERSISTENCE_S", 0.3),
        ("M6_HARD_BRAKING_THRESHOLD_MPS2", -5.0),
    )
    for name, drifted in attacks:
        with monkeypatch.context() as attack:
            attack.setattr(production_metrics, name, drifted)
            with pytest.raises(
                synthetic.M6SyntheticAcceptanceError,
                match="preregistration literals",
            ):
                no_conflict.revalidate()


def test_coordinated_production_and_independent_threshold_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    no_conflict: M6SyntheticOracleResult,
) -> None:
    attacks = (
        (
            "M6_RESPONSE_ACCELERATION_THRESHOLD_MPS2",
            "_INDEPENDENT_RESPONSE_ACCELERATION_THRESHOLD_MPS2",
            -0.75,
        ),
        (
            "M6_RESPONSE_PERSISTENCE_S",
            "_INDEPENDENT_RESPONSE_PERSISTENCE_S",
            0.3,
        ),
        (
            "M6_HARD_BRAKING_THRESHOLD_MPS2",
            "_INDEPENDENT_HARD_BRAKING_THRESHOLD_MPS2",
            -5.0,
        ),
    )
    for production_name, independent_name, drifted in attacks:
        with monkeypatch.context() as attack:
            attack.setattr(production_metrics, production_name, drifted)
            attack.setattr(synthetic, independent_name, drifted)
            with pytest.raises(
                synthetic.M6SyntheticAcceptanceError,
                match="preregistration literals",
            ):
                no_conflict.revalidate()


def test_production_metric_mutant_is_caught_by_retention_or_raw_crosscheck(
    monkeypatch: pytest.MonkeyPatch,
    sentinel: M6SyntheticOracleResult,
) -> None:
    original_compute = (
        production_metrics.AdditionalTargetBrakingImpulseMetric.compute
    )

    def mutant(
        self: Any,
        pair: Any,
    ) -> PairedMetricResult:
        result = original_compute(self, pair)
        return PairedMetricResult(
            metric_name=result.metric_name,
            metric_version=result.metric_version,
            scenario_id=result.scenario_id,
            intervention_identity=result.intervention_identity,
            value=result.value + 0.25,
            details=result.details,
        )

    monkeypatch.setattr(
        production_metrics.AdditionalTargetBrakingImpulseMetric,
        "compute",
        mutant,
    )
    with pytest.raises(
        (ValueError, synthetic.M6SyntheticAcceptanceError),
        match="metric",
    ):
        sentinel.revalidate()


def test_sentinel_raw_cost_separation_is_pre_registered(
    sentinel: M6SyntheticOracleResult,
    acceptance: M6SyntheticAcceptanceResult,
) -> None:
    raw = sentinel.independent_measures
    assert isinstance(raw, M6IndependentMeasures)
    assert raw.response_responded
    assert raw.response_start_transition == 1
    assert raw.first_world_divergence_frame == 5
    nominal_scene = next(
        scene
        for scene in acceptance.evaluation.primary_scene_results
        if scene.cohort_index == 0 and scene.policy_name == "idm"
    )
    nominal_raw = synthetic._independent_scene_measures(nominal_scene)
    assert (
        raw.target_progress_loss_m - nominal_raw.target_progress_loss_m
        >= M6_SYNTHETIC_MIN_SENTINEL_EXTRA_PROGRESS_LOSS_M
    )
    assert (
        raw.additional_hard_braking_exposure_s
        - nominal_raw.additional_hard_braking_exposure_s
        >= M6_SYNTHETIC_MIN_SENTINEL_EXTRA_HARD_BRAKING_S
    )


def test_complete_acceptance_revalidates_independent_gates(
    acceptance: M6SyntheticAcceptanceResult,
) -> None:
    acceptance.revalidate()
    local = acceptance.to_local_dict()
    assert local["schema_version"] == M6_SYNTHETIC_ACCEPTANCE_VERSION
    assert local["case_count"] == 10
    assert local["deterministic_replica_count"] == 10
    assert local["unique_analytic_geometry_count"] == 1
    assert local["all_replicas_eligible"] is True
    assert local["identity_sham_passed"] is True
    assert local["nonreactive_policy_dose_gate_count"] == 40
    assert local["idm_response_replica_count"] == 10
    assert local["synchronous_response_floor_frames"] == 2
    assert local["nested_dose_replica_count"] == 10
    assert local["no_conflict_primary_eligibility"] == "rejected"
    assert local["no_conflict_primary_rejection_reason"] == (
        "no_stable_aligned_follower"
    )
    assert local["no_conflict_exactly_nonreactive"] is True
    assert local["overreactive_sentinel_test_only"] is True


def test_retained_sources_rollouts_plans_and_traces_are_immutable(
    sentinel: M6SyntheticOracleResult,
) -> None:
    assert sentinel.scope.source_snapshot.timestamps.flags.writeable is False
    assert sentinel.baseline_rollout.agents[1].x.flags.writeable is False
    assert sentinel.intervention_rollout.agents[1].vx.flags.writeable is False
    assert sentinel.baseline_plan.x.flags.writeable is False
    assert (
        sentinel.baseline_trace.longitudinal_acceleration.flags.writeable
        is False
    )
    with pytest.raises(ValueError):
        sentinel.intervention_rollout.agents[1].vx[0] = 0.0
