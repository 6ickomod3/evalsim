"""Independent analytic tests for the accepted M6 paired measures."""
from __future__ import annotations

import math

import numpy as np
import pytest

from evalsim.contracts.counterfactual import (
    CounterfactualPair,
    InterventionEligibility,
    evaluate_paired_metric,
)
from evalsim.contracts.rollout import Rollout
from evalsim.contracts.scenario import Agent, Scenario
from evalsim.contracts.simulator import PolicyMetadata
from evalsim.contracts.types import AgentType
from evalsim.metrics import (
    M6_PRIMARY_PAIRED_METRIC_SPECS,
    M6_SECONDARY_PAIRED_METRIC_SPECS,
    AdditionalAbsoluteJerkIntegralMetric,
    AdditionalHardBrakingExposureMetric,
    AdditionalTargetBrakingImpulseMetric,
    MinimumLongitudinalBumperGapChangeMetric,
    ResponseTimelinessMetric,
    TargetProgressLossMetric,
    TargetSpeedReductionMaxMetric,
    TargetWorldDisplacementMeanMetric,
    is_exactly_nonreactive,
    m6_paired_metrics,
    world_trajectory_tensor_equal,
)
from evalsim.perturb.m6 import (
    compile_identity_plan,
    compile_longitudinal_brake_pulse_plan,
)

_TRANSITIONS = 40


def _velocities_from_acceleration(
    initial_speed: float,
    acceleration: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    speeds = np.empty(len(dt) + 1, dtype=float)
    speeds[0] = initial_speed
    for transition, duration in enumerate(dt):
        speeds[transition + 1] = (
            speeds[transition] + acceleration[transition] * duration
        )
    assert np.all(speeds >= 0.0)
    return speeds


def _positions_from_speed(
    initial_position: float,
    speed: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    positions = np.empty(len(speed), dtype=float)
    positions[0] = initial_position
    for transition, duration in enumerate(dt):
        positions[transition + 1] = positions[transition] + (
            0.5
            * (speed[transition] + speed[transition + 1])
            * duration
        )
    return positions


def _agent(
    *,
    identifier: int,
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray | None = None,
    heading: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    length: float = 4.0,
    width: float = 2.0,
) -> Agent:
    frames = len(x)
    return Agent(
        id=identifier,
        type=AgentType.VEHICLE,
        valid=np.ones(frames, dtype=bool) if valid is None else valid,
        x=x,
        y=y,
        heading=(
            np.zeros(frames, dtype=float) if heading is None else heading
        ),
        vx=vx,
        vy=np.zeros(frames, dtype=float) if vy is None else vy,
        length=length,
        width=width,
    )


def _copy_agent(agent: Agent, **changes: np.ndarray) -> Agent:
    values = {
        "valid": np.array(agent.valid, copy=True),
        "x": np.array(agent.x, copy=True),
        "y": np.array(agent.y, copy=True),
        "heading": np.array(agent.heading, copy=True),
        "vx": np.array(agent.vx, copy=True),
        "vy": np.array(agent.vy, copy=True),
    }
    values.update(changes)
    return Agent(
        id=agent.id,
        type=agent.type,
        length=agent.length,
        width=agent.width,
        **values,
    )


def _rollout_metadata() -> dict[str, object]:
    return {
        "engine": {"name": "rollout_engine", "version": "m6-test"},
        "policy": PolicyMetadata(
            name="idm",
            version="1.0.0",
            deterministic=True,
            supported_agent_types=(AgentType.VEHICLE,),
        ).to_dict(),
        "dynamics": {
            "name": "bicycle",
            "limits": {"maximum_speed_mps": 60.0},
            "clamp_counts": {"acceleration": 0},
        },
        "ego_control": "typed_ego_plan",
        "rollout_start_index": 0,
        "controlled_agent_ids": [20, 30],
        "agent_control_modes": {
            "10": "typed_ego_plan",
            "20": "idm",
            "30": "idm",
        },
        "scenario_source": "synthetic",
        "scenario_source_fingerprint": "a" * 64,
    }


def _pair(
    *,
    dt: np.ndarray | None = None,
    baseline_target_acceleration: np.ndarray | None = None,
    intervention_target_acceleration: np.ndarray | None = None,
    baseline_target_x: np.ndarray | None = None,
    intervention_target_x: np.ndarray | None = None,
    baseline_target_y: np.ndarray | None = None,
    intervention_target_y: np.ndarray | None = None,
    target_heading: np.ndarray | None = None,
    target_valid: np.ndarray | None = None,
    initial_target_speed: float = 20.0,
    world_change: tuple[int, str] | None = None,
) -> CounterfactualPair:
    if dt is None:
        dt = np.full(_TRANSITIONS, 0.25, dtype=float)
    dt = np.asarray(dt, dtype=float)
    assert dt.shape == (_TRANSITIONS,)
    timestamps = np.concatenate(([0.0], np.cumsum(dt)))
    frames = len(timestamps)
    if baseline_target_acceleration is None:
        baseline_target_acceleration = np.zeros(_TRANSITIONS, dtype=float)
    if intervention_target_acceleration is None:
        intervention_target_acceleration = np.zeros(_TRANSITIONS, dtype=float)
    baseline_target_vx = _velocities_from_acceleration(
        initial_target_speed,
        np.asarray(baseline_target_acceleration, dtype=float),
        dt,
    )
    intervention_target_vx = _velocities_from_acceleration(
        initial_target_speed,
        np.asarray(intervention_target_acceleration, dtype=float),
        dt,
    )
    if baseline_target_x is None:
        baseline_target_x = _positions_from_speed(
            10.0,
            baseline_target_vx,
            dt,
        )
    if intervention_target_x is None:
        intervention_target_x = _positions_from_speed(
            10.0,
            intervention_target_vx,
            dt,
        )
    if baseline_target_y is None:
        baseline_target_y = np.zeros(frames, dtype=float)
    if intervention_target_y is None:
        intervention_target_y = np.zeros(frames, dtype=float)

    ego_speed = np.full(frames, 24.0, dtype=float)
    baseline_ego_x = 35.0 + 24.0 * timestamps
    ego_y = np.zeros(frames, dtype=float)

    source_ego = _agent(
        identifier=10,
        x=np.asarray(baseline_ego_x, dtype=float),
        y=ego_y,
        vx=ego_speed,
        length=5.0,
    )
    source_target = _agent(
        identifier=20,
        x=np.asarray(baseline_target_x, dtype=float),
        y=np.asarray(baseline_target_y, dtype=float),
        vx=baseline_target_vx,
        heading=target_heading,
        valid=target_valid,
    )
    other_x = -20.0 + 15.0 * timestamps
    other = _agent(
        identifier=30,
        x=other_x,
        y=np.full(frames, 7.0, dtype=float),
        vx=np.full(frames, 15.0, dtype=float),
    )
    source = Scenario(
        scenario_id="synthetic-m6-metric",
        timestamps=timestamps,
        agents=[source_ego, source_target, other],
        ego_index=0,
        metadata={
            "current_index": 0,
            "source": "synthetic",
            "source_fingerprint": "a" * 64,
        },
    )
    baseline_plan = compile_identity_plan(source)
    intervention_plan = compile_longitudinal_brake_pulse_plan(source, 2.0)
    baseline = Rollout(
        scenario_id=source.scenario_id,
        sim_name="idm",
        sim_version="1.0.0",
        seed=0,
        timestamps=timestamps,
        agents=[_copy_agent(agent) for agent in source.agents],
        perturbation=baseline_plan.perturbation_identity,
        metadata=_rollout_metadata(),
    )
    treatment_target = _copy_agent(
        source_target,
        x=np.asarray(intervention_target_x, dtype=float),
        y=np.asarray(intervention_target_y, dtype=float),
        vx=intervention_target_vx,
    )
    treatment_agents = [
        _copy_agent(
            source_ego,
            valid=np.array(intervention_plan.valid, copy=True),
            x=np.array(intervention_plan.x, copy=True),
            y=np.array(intervention_plan.y, copy=True),
            heading=np.array(intervention_plan.heading, copy=True),
            vx=np.array(intervention_plan.vx, copy=True),
            vy=np.array(intervention_plan.vy, copy=True),
        ),
        treatment_target,
        _copy_agent(other),
    ]
    if world_change is not None:
        agent_index, field = world_change
        values = np.array(
            getattr(treatment_agents[agent_index], field),
            copy=True,
        )
        values[7] += 0.125
        setattr(treatment_agents[agent_index], field, values)
    treatment = Rollout(
        scenario_id=source.scenario_id,
        sim_name="idm",
        sim_version="1.0.0",
        seed=0,
        timestamps=timestamps,
        agents=treatment_agents,
        perturbation=intervention_plan.perturbation_identity,
        metadata=_rollout_metadata(),
    )
    return CounterfactualPair(
        scenario=source,
        baseline=baseline,
        intervention=treatment,
        baseline_plan=baseline_plan,
        intervention_plan=intervention_plan,
        eligibility=InterventionEligibility.accepted((0, _TRANSITIONS), 1),
        intervention_identity=intervention_plan.perturbation_identity,
    )


def test_registered_metric_families_are_exact_and_ordered() -> None:
    assert [spec.metric_id for spec in M6_PRIMARY_PAIRED_METRIC_SPECS] == [
        "additional_target_braking_impulse_mps@1.0.0",
        "response_timeliness_s@1.0.0",
        "minimum_longitudinal_bumper_gap_change_m@1.0.0",
        "target_progress_loss_m@1.0.0",
    ]
    assert [spec.metric_id for spec in M6_SECONDARY_PAIRED_METRIC_SPECS] == [
        "target_world_displacement_mean_m@1.0.0",
        "target_speed_reduction_max_mps@1.0.0",
        "additional_absolute_jerk_integral_mps2@1.0.0",
        "additional_hard_braking_exposure_s@1.0.0",
    ]
    assert [metric.spec for metric in m6_paired_metrics()] == [
        *M6_PRIMARY_PAIRED_METRIC_SPECS,
        *M6_SECONDARY_PAIRED_METRIC_SPECS,
    ]
    assert all(spec.direction == "neutral" for spec in (
        *M6_PRIMARY_PAIRED_METRIC_SPECS,
        *M6_SECONDARY_PAIRED_METRIC_SPECS,
    ))


def test_primary_metrics_match_independent_irregular_dt_formulas() -> None:
    dt = np.array(
        [0.25, 0.5, 0.125, 0.375, *([0.25] * 36)],
        dtype=float,
    )
    baseline_acceleration = np.zeros(_TRANSITIONS, dtype=float)
    intervention_acceleration = np.zeros(_TRANSITIONS, dtype=float)
    baseline_acceleration[:4] = [1.0, -1.0, 0.5, -5.0]
    intervention_acceleration[:4] = [1.0, -4.0, 0.5, -4.0]
    pair = _pair(
        dt=dt,
        baseline_target_acceleration=baseline_acceleration,
        intervention_target_acceleration=intervention_acceleration,
    )

    # Independent interval formula: max(0, a_base - a_treatment) * dt.
    expected_impulse = math.fsum(
        max(0.0, float(left - right)) * float(duration)
        for left, right, duration in zip(
            baseline_acceleration,
            intervention_acceleration,
            dt,
            strict=True,
        )
    )
    impulse = evaluate_paired_metric(
        AdditionalTargetBrakingImpulseMetric(),
        pair,
    )
    assert impulse.value == pytest.approx(expected_impulse)

    timing = evaluate_paired_metric(ResponseTimelinessMetric(), pair)
    expected_event_time = float(dt[0] + dt[1])
    expected_window = float(math.fsum(float(value) for value in dt))
    assert timing.details["responded"] is True
    assert timing.details["censored"] is False
    assert timing.details["response_start_transition"] == 1
    assert timing.details["response_end_transition"] == 1
    assert timing.details["event_time_s"] == pytest.approx(expected_event_time)
    assert timing.value == pytest.approx(expected_window - expected_event_time)

    target = pair.scenario.agents[1]
    speed = math.hypot(float(target.vx[0]), float(target.vy[0]))
    hx = float(target.vx[0]) / speed
    hy = float(target.vy[0]) / speed
    half_length = 0.5 * (
        pair.scenario.ego.length + pair.scenario.agents[1].length
    )

    def independent_min_gap(rollout: object) -> float:
        ego = rollout.agents[0]
        follower = rollout.agents[1]
        return min(
            (
                (float(ego.x[frame]) - float(follower.x[frame])) * hx
                + (float(ego.y[frame]) - float(follower.y[frame])) * hy
                - half_length
            )
            for frame in range(1, _TRANSITIONS + 1)
        )

    expected_gap_change = independent_min_gap(
        pair.intervention
    ) - independent_min_gap(pair.baseline)
    gap = evaluate_paired_metric(
        MinimumLongitudinalBumperGapChangeMetric(),
        pair,
    )
    assert gap.value == pytest.approx(expected_gap_change)

    heading = float(target.heading[0])
    heading_x = math.cos(heading)
    heading_y = math.sin(heading)

    def independent_progress(rollout: object) -> float:
        follower = rollout.agents[1]
        return (
            (float(follower.x[-1]) - float(target.x[0])) * heading_x
            + (float(follower.y[-1]) - float(target.y[0])) * heading_y
        )

    expected_progress_loss = independent_progress(
        pair.baseline
    ) - independent_progress(pair.intervention)
    progress = evaluate_paired_metric(TargetProgressLossMetric(), pair)
    assert progress.value == pytest.approx(expected_progress_loss)


def test_secondary_metrics_match_independent_irregular_dt_formulas() -> None:
    dt = np.array(
        [0.25, 0.5, 0.125, 0.375, *([0.25] * 36)],
        dtype=float,
    )
    baseline_acceleration = np.zeros(_TRANSITIONS, dtype=float)
    intervention_acceleration = np.zeros(_TRANSITIONS, dtype=float)
    baseline_acceleration[:4] = [1.0, -1.0, 0.5, -5.0]
    intervention_acceleration[:4] = [1.0, -4.0, 0.5, -4.0]
    pair = _pair(
        dt=dt,
        baseline_target_acceleration=baseline_acceleration,
        intervention_target_acceleration=intervention_acceleration,
    )
    baseline = pair.baseline.agents[1]
    treatment = pair.intervention.agents[1]

    expected_displacement = math.fsum(
        math.hypot(
            float(treatment.x[frame] - baseline.x[frame]),
            float(treatment.y[frame] - baseline.y[frame]),
        )
        for frame in range(1, _TRANSITIONS + 1)
    ) / _TRANSITIONS
    assert TargetWorldDisplacementMeanMetric().compute(
        pair
    ).value == pytest.approx(expected_displacement)

    expected_max_speed_reduction = max(
        math.hypot(float(baseline.vx[frame]), float(baseline.vy[frame]))
        - math.hypot(float(treatment.vx[frame]), float(treatment.vy[frame]))
        for frame in range(1, _TRANSITIONS + 1)
    )
    assert TargetSpeedReductionMaxMetric().compute(
        pair
    ).value == pytest.approx(expected_max_speed_reduction)

    expected_jerk_integral = 0.0
    for transition in range(1, _TRANSITIONS):
        derivative_dt = 0.5 * (
            float(dt[transition - 1]) + float(dt[transition])
        )
        baseline_jerk = (
            float(
                baseline_acceleration[transition]
                - baseline_acceleration[transition - 1]
            )
            / derivative_dt
        )
        treatment_jerk = (
            float(
                intervention_acceleration[transition]
                - intervention_acceleration[transition - 1]
            )
            / derivative_dt
        )
        expected_jerk_integral += max(
            0.0,
            abs(treatment_jerk) - abs(baseline_jerk),
        ) * derivative_dt
    jerk = AdditionalAbsoluteJerkIntegralMetric().compute(pair)
    assert jerk.value == pytest.approx(expected_jerk_integral)
    assert jerk.details["derivative_interval_count"] == 39

    expected_baseline_exposure = math.fsum(
        float(duration)
        for acceleration, duration in zip(
            baseline_acceleration,
            dt,
            strict=True,
        )
        if acceleration <= -4.0
    )
    expected_treatment_exposure = math.fsum(
        float(duration)
        for acceleration, duration in zip(
            intervention_acceleration,
            dt,
            strict=True,
        )
        if acceleration <= -4.0
    )
    hard_braking = AdditionalHardBrakingExposureMetric().compute(pair)
    assert hard_braking.value == pytest.approx(
        expected_treatment_exposure - expected_baseline_exposure
    )
    assert hard_braking.details["baseline_exposure_s"] == pytest.approx(
        expected_baseline_exposure
    )
    assert hard_braking.details["intervention_exposure_s"] == pytest.approx(
        expected_treatment_exposure
    )


def test_response_obeys_t_plus_two_floor() -> None:
    baseline = np.zeros(_TRANSITIONS, dtype=float)
    second_transition = np.zeros(_TRANSITIONS, dtype=float)
    second_transition[1] = -0.5
    detected = ResponseTimelinessMetric().compute(
        _pair(
            baseline_target_acceleration=baseline,
            intervention_target_acceleration=second_transition,
        )
    )
    assert detected.details["responded"] is True
    assert detected.details["response_start_transition"] == 1
    assert detected.details["response_end_transition"] == 1
    assert detected.details["event_time_s"] == pytest.approx(0.5)


def test_response_persistence_is_contiguous_and_inclusive_at_boundaries() -> None:
    dt = np.full(_TRANSITIONS, 0.125, dtype=float)
    baseline = np.zeros(_TRANSITIONS, dtype=float)
    treatment = np.zeros(_TRANSITIONS, dtype=float)
    treatment[1:3] = -0.5
    result = ResponseTimelinessMetric().compute(
        _pair(
            dt=dt,
            baseline_target_acceleration=baseline,
            intervention_target_acceleration=treatment,
        )
    )
    assert result.details["response_start_transition"] == 1
    assert result.details["response_end_transition"] == 2
    assert result.details["event_time_s"] == pytest.approx(0.375)

    interrupted = np.zeros(_TRANSITIONS, dtype=float)
    interrupted[1] = -0.5
    interrupted[3] = -0.5
    no_event = ResponseTimelinessMetric().compute(
        _pair(
            dt=dt,
            baseline_target_acceleration=baseline,
            intervention_target_acceleration=interrupted,
        )
    )
    assert no_event.value == 0.0
    assert no_event.details["responded"] is False


def test_response_event_exactly_at_window_end_is_not_censored() -> None:
    baseline = np.zeros(_TRANSITIONS, dtype=float)
    treatment = np.zeros(_TRANSITIONS, dtype=float)
    treatment[-1] = -0.5
    result = ResponseTimelinessMetric().compute(
        _pair(
            baseline_target_acceleration=baseline,
            intervention_target_acceleration=treatment,
        )
    )
    assert result.value == 0.0
    assert result.details["responded"] is True
    assert result.details["censored"] is False
    assert result.details["event_time_s"] == result.details["window_s"]
    assert result.details["restricted_latency_s"] == result.details["window_s"]


def test_nonresponse_is_right_censored_at_w() -> None:
    result = ResponseTimelinessMetric().compute(_pair())
    assert result.value == 0.0
    assert result.details["responded"] is False
    assert result.details["censored"] is True
    assert result.details["event_time_s"] is None
    assert result.details["restricted_latency_s"] == result.details["window_s"]
    assert result.details["response_start_transition"] is None
    assert result.details["response_end_transition"] is None


def test_relational_gap_change_never_classifies_world_response() -> None:
    pair = _pair()
    gap = MinimumLongitudinalBumperGapChangeMetric().compute(pair)
    assert gap.value < 0.0
    assert world_trajectory_tensor_equal(pair) is True
    assert is_exactly_nonreactive(pair) is True


@pytest.mark.parametrize(
    ("agent_index", "field"),
    [
        (1, "x"),
        (1, "y"),
        (1, "heading"),
        (1, "vx"),
        (1, "vy"),
        (2, "x"),
        (2, "y"),
        (2, "heading"),
        (2, "vx"),
        (2, "vy"),
    ],
)
def test_world_tensor_comparator_detects_any_world_state_response(
    agent_index: int,
    field: str,
) -> None:
    pair = _pair(world_change=(agent_index, field))
    assert world_trajectory_tensor_equal(pair) is False
    assert is_exactly_nonreactive(pair) is False


def test_gap_zero_speed_direction_uses_current_heading_fallback() -> None:
    frames = _TRANSITIONS + 1
    acceleration = np.zeros(_TRANSITIONS, dtype=float)
    acceleration[0] = 4.0
    heading = np.full(frames, np.pi / 2.0, dtype=float)
    baseline_y = np.zeros(frames, dtype=float)
    treatment_y = np.array(baseline_y, copy=True)
    treatment_y[2:] = 3.0
    pair = _pair(
        baseline_target_acceleration=acceleration,
        intervention_target_acceleration=acceleration,
        baseline_target_y=baseline_y,
        intervention_target_y=treatment_y,
        target_heading=heading,
        initial_target_speed=0.0,
    )
    assert MinimumLongitudinalBumperGapChangeMetric().compute(
        pair
    ).value == pytest.approx(-3.0)


def test_gap_uses_velocity_direction_but_progress_uses_heading() -> None:
    frames = _TRANSITIONS + 1
    heading = np.full(frames, np.pi / 2.0, dtype=float)
    baseline_target_x = np.full(frames, 10.0, dtype=float)
    treatment_target_x = np.array(baseline_target_x, copy=True)
    baseline_target_y = np.arange(frames, dtype=float)
    treatment_target_y = np.array(baseline_target_y, copy=True)
    treatment_target_y[-1] -= 4.0
    pair = _pair(
        baseline_target_x=baseline_target_x,
        intervention_target_x=treatment_target_x,
        baseline_target_y=baseline_target_y,
        intervention_target_y=treatment_target_y,
        target_heading=heading,
    )
    no_lateral_world_change = _pair(
        baseline_target_x=baseline_target_x,
        intervention_target_x=treatment_target_x,
        baseline_target_y=baseline_target_y,
        intervention_target_y=baseline_target_y,
        target_heading=heading,
    )
    # Current target velocity is +x, so the frozen gap axis ignores the -4 m y
    # displacement even though the current heading points +y.
    assert MinimumLongitudinalBumperGapChangeMetric().compute(pair).value == (
        MinimumLongitudinalBumperGapChangeMetric()
        .compute(no_lateral_world_change)
        .value
    )
    # Progress is explicitly projected along current heading (+y).
    assert TargetProgressLossMetric().compute(pair).value == pytest.approx(4.0)


@pytest.mark.parametrize(
    "metric",
    [
        AdditionalTargetBrakingImpulseMetric(),
        ResponseTimelinessMetric(),
        MinimumLongitudinalBumperGapChangeMetric(),
        TargetProgressLossMetric(),
        TargetWorldDisplacementMeanMetric(),
        TargetSpeedReductionMaxMetric(),
        AdditionalAbsoluteJerkIntegralMetric(),
        AdditionalHardBrakingExposureMetric(),
    ],
)
def test_every_metric_rejects_target_missingness(metric: object) -> None:
    valid = np.ones(_TRANSITIONS + 1, dtype=bool)
    valid[12] = False
    pair = _pair(target_valid=valid)
    with pytest.raises(ValueError, match="source-valid throughout"):
        metric.compute(pair)


def test_metric_and_world_comparator_reject_snapshot_tampering() -> None:
    pair = _pair()
    altered = np.array(pair.intervention.agents[1].vx, copy=True)
    altered[5] -= 1.0
    object.__setattr__(pair.intervention.agents[1], "vx", altered)
    with pytest.raises(ValueError, match="snapshot was mutated"):
        AdditionalTargetBrakingImpulseMetric().compute(pair)
    with pytest.raises(ValueError, match="snapshot was mutated"):
        world_trajectory_tensor_equal(pair)
