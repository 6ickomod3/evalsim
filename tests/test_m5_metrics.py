"""Analytic oracles and adversarial boundaries for the M5 metric system."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from evalsim import (
    Agent,
    AgentType,
    MapPolyline,
    MapType,
    Rollout,
    Scenario,
    rollout_from_parquet,
    rollout_to_parquet,
    scenario_from_parquet,
    scenario_to_parquet,
)
from evalsim.metrics.m5 import (
    M5_KINEMATIC_METRIC_VERSION,
    AccelerationErrorMetric,
    ConstantVelocityTTCCapMetric,
    JerkErrorMetric,
    KinematicContinuityResidualMetric,
    LaneCenterDistanceMetric,
    LaneHeadingDisagreementMetric,
    LifecycleReentryPerAgentMetric,
    MinimumCenterDistanceMetric,
    OrientedBoxOverlapRateMetric,
    PositionErrorMetric,
    SpeedErrorMetric,
    WaymaxKinematicInfeasibilityRateMetric,
    YawRateErrorMetric,
    constant_velocity_disc_ttc,
    kinematic_infeasibility_components,
    kinematic_infeasibility_flags,
    m5_metrics,
    oriented_box_overlap_components,
    position_divergence_components,
)
from evalsim.metrics.registry import MetricRegistry


def _series(value, count: int, *, dtype=float) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        return np.full(count, array.item(), dtype=dtype)
    assert array.shape == (count,)
    return np.array(array, dtype=dtype, copy=True)


def _agent(
    agent_id: int,
    timestamps: np.ndarray,
    *,
    agent_type: AgentType = AgentType.VEHICLE,
    valid=True,
    x=0.0,
    y=0.0,
    heading=0.0,
    vx=0.0,
    vy=0.0,
    length: float = 2.0,
    width: float = 2.0,
) -> Agent:
    count = len(timestamps)
    return Agent(
        id=agent_id,
        type=agent_type,
        valid=_series(valid, count, dtype=bool),
        x=_series(x, count),
        y=_series(y, count),
        heading=_series(heading, count),
        vx=_series(vx, count),
        vy=_series(vy, count),
        length=length,
        width=width,
    )


def _pair(
    *,
    timestamps: np.ndarray | None = None,
    current_index: int = 0,
    source_agents: list[Agent] | None = None,
    candidate_agents: list[Agent] | None = None,
    map_features: list[MapPolyline] | None = None,
) -> tuple[Scenario, Rollout]:
    if timestamps is None:
        timestamps = np.arange(5, dtype=float) * 0.1
    if source_agents is None:
        source_agents = [
            _agent(0, timestamps, x=-10.0),
            _agent(
                1,
                timestamps,
                x=np.arange(len(timestamps), dtype=float),
                vx=10.0,
            ),
        ]
    if candidate_agents is None:
        candidate_agents = copy.deepcopy(source_agents)
    scenario = Scenario(
        scenario_id="m5-unit-scenario",
        timestamps=np.array(timestamps, copy=True),
        agents=source_agents,
        map=list(map_features or []),
        ego_index=0,
        metadata={"source": "unit", "current_index": current_index},
    )
    rollout = Rollout(
        scenario_id=scenario.scenario_id,
        sim_name="candidate",
        sim_version="1.0.0",
        seed=0,
        timestamps=np.array(timestamps, copy=True),
        agents=candidate_agents,
    )
    return scenario, rollout


def _result(metric, scenario: Scenario, rollout: Rollout):
    return MetricRegistry([metric]).evaluate(scenario, rollout)[0]


@pytest.mark.parametrize(
    "metric_type",
    [
        PositionErrorMetric,
        SpeedErrorMetric,
        AccelerationErrorMetric,
        JerkErrorMetric,
        YawRateErrorMetric,
    ],
)
def test_exact_log_replay_error_metrics_are_zero(metric_type) -> None:
    timestamps = np.arange(6, dtype=float) * 0.1
    t = timestamps
    world = _agent(
        1,
        timestamps,
        x=2.0 * t,
        y=-0.5 * t,
        heading=np.arctan2(-0.5, 2.0),
        vx=2.0,
        vy=-0.5,
    )
    scenario, rollout = _pair(
        timestamps=timestamps,
        current_index=1,
        source_agents=[_agent(0, timestamps, x=-20.0), world],
    )
    result = _result(metric_type(), scenario, rollout)
    assert result.valid
    assert result.value == pytest.approx(0.0, abs=0.0)
    assert set(result.distribution) == {0.0}


def test_position_parity_view_is_float32_and_uses_joint_validity() -> None:
    timestamps = np.arange(3, dtype=float) * 0.1
    valid = np.array([True, False, True])
    source = _agent(1, timestamps, valid=valid, x=[0.0, 100.0, 2.0])
    candidate = _agent(
        1,
        timestamps,
        valid=valid,
        x=[3.0, -999.0, 5.0],
        y=[4.0, -999.0, 4.0],
    )
    scenario, rollout = _pair(
        timestamps=timestamps,
        source_agents=[_agent(0, timestamps, x=-20.0), source],
        candidate_agents=[_agent(0, timestamps, x=-20.0), candidate],
    )
    values, mask = position_divergence_components(scenario, rollout)
    assert values.dtype == np.float32
    np.testing.assert_array_equal(mask[:, 1], valid)
    assert values[0, 1] == np.float32(5.0)
    assert values[2, 1] == np.float32(5.0)


def test_nonuniform_derivatives_have_known_values() -> None:
    timestamps = np.array([0.0, 0.5, 1.5, 3.0])
    source = _agent(1, timestamps, vx=0.0, vy=0.0)
    candidate = _agent(
        1,
        timestamps,
        vx=np.array([0.0, 1.0, 5.0, 14.0]),
        vy=0.0,
    )
    scenario, rollout = _pair(
        timestamps=timestamps,
        source_agents=[_agent(0, timestamps, x=-20.0), source],
        candidate_agents=[_agent(0, timestamps, x=-20.0), candidate],
    )
    acceleration = _result(AccelerationErrorMetric(), scenario, rollout)
    np.testing.assert_allclose(acceleration.distribution, [2.0, 4.0, 6.0])
    assert acceleration.value == pytest.approx(4.0)

    jerk = _result(JerkErrorMetric(), scenario, rollout)
    # Interval accelerations are 2, 4, 6; midpoint spacings are .75 and 1.25.
    np.testing.assert_allclose(jerk.distribution, [2.0 / 0.75, 2.0 / 1.25])


def test_wrapped_yaw_rate_uses_short_angle() -> None:
    timestamps = np.array([0.0, 0.5])
    source = _agent(
        1,
        timestamps,
        heading=[math.pi - 0.1, -math.pi + 0.1],
    )
    candidate = _agent(
        1,
        timestamps,
        heading=[math.pi - 0.1, -math.pi + 0.2],
    )
    scenario, rollout = _pair(
        timestamps=timestamps,
        source_agents=[_agent(0, timestamps, x=-20.0), source],
        candidate_agents=[_agent(0, timestamps, x=-20.0), candidate],
    )
    result = _result(YawRateErrorMetric(), scenario, rollout)
    assert result.distribution == pytest.approx((0.2,))


def test_derivatives_never_bridge_a_validity_gap_or_reentry() -> None:
    timestamps = np.arange(5, dtype=float) * 0.1
    valid = np.array([True, True, False, True, True])
    source = _agent(1, timestamps, valid=valid, vx=0.0)
    candidate = _agent(1, timestamps, valid=valid, vx=[0, 1, 999, 10, 12])
    scenario, rollout = _pair(
        timestamps=timestamps,
        source_agents=[_agent(0, timestamps, x=-20.0), source],
        candidate_agents=[_agent(0, timestamps, x=-20.0), candidate],
    )
    acceleration = _result(AccelerationErrorMetric(), scenario, rollout)
    np.testing.assert_allclose(acceleration.distribution, [10.0, 20.0])
    jerk = _result(JerkErrorMetric(), scenario, rollout)
    assert not jerk.valid
    assert jerk.invalid_reason == "no_contiguous_valid_window"


def test_nonzero_current_index_windows_and_component_counts_are_exact() -> None:
    timestamps = np.arange(6, dtype=float) * 0.1
    valid = np.array([True, True, True, True, False, True])
    source = _agent(
        1,
        timestamps,
        valid=valid,
        x=np.zeros(6),
        y=5.0,
        vx=np.zeros(6),
        vy=0.0,
    )
    candidate = _agent(
        1,
        timestamps,
        valid=valid,
        x=[1000.0, -1000.0, 500.0, 1.0, 999.0, 3.0],
        y=5.0,
        vx=[999.0, 0.0, 1.0, 3.0, 999.0, 5.0],
        vy=0.0,
    )
    lane = MapPolyline(
        type=MapType.LANE,
        xy=np.array([[-100.0, 0.0], [100.0, 0.0]]),
    )
    scenario, rollout = _pair(
        timestamps=timestamps,
        current_index=2,
        source_agents=[_agent(0, timestamps, x=-20.0), source],
        candidate_agents=[
            _agent(0, timestamps, x=-20.0),
            candidate,
        ],
        map_features=[lane],
    )

    position = _result(PositionErrorMetric(), scenario, rollout)
    assert position.distribution == (1.0, 3.0)
    assert position.value == 2.0
    assert position.eligible_components == 2
    assert position.total_components == 3

    acceleration = _result(AccelerationErrorMetric(), scenario, rollout)
    assert acceleration.distribution == pytest.approx((20.0,))
    assert acceleration.eligible_components == 1
    assert acceleration.total_components == 3

    jerk = _result(JerkErrorMetric(), scenario, rollout)
    assert jerk.distribution == pytest.approx((100.0,))
    assert jerk.eligible_components == 1
    assert jerk.total_components == 3

    interaction = _result(MinimumCenterDistanceMetric(), scenario, rollout)
    assert interaction.eligible_components == 2
    assert interaction.total_components == 3
    assert interaction.details["evaluated_pair_count"] == 2

    lane_distance = _result(LaneCenterDistanceMetric(), scenario, rollout)
    assert lane_distance.distribution == (5.0, 5.0)
    assert lane_distance.eligible_components == 2
    assert lane_distance.total_components == 3

    lifecycle = _result(LifecycleReentryPerAgentMetric(), scenario, rollout)
    assert lifecycle.distribution == (1.0,)
    assert lifecycle.eligible_components == 1
    assert lifecycle.total_components == 1


def _overlap_pair(
    separation: float,
    *,
    target_valid: bool = True,
    invalid_padding: bool = False,
) -> tuple[Scenario, Rollout]:
    timestamps = np.array([0.0, 0.1])
    ego = _agent(0, timestamps, x=0.0, length=2.0, width=2.0)
    target = _agent(
        1,
        timestamps,
        valid=[True, target_valid],
        x=[separation, separation],
        length=2.0,
        width=2.0,
    )
    agents = [ego, target]
    if invalid_padding:
        agents.append(
            _agent(
                2,
                timestamps,
                valid=False,
                x=separation,
                length=20.0,
                width=20.0,
            )
        )
    return _pair(timestamps=timestamps, source_agents=agents)


def test_oriented_boxes_distinguish_separation_touch_and_penetration() -> None:
    separated_value = np.nextafter(
        np.float32(2.0),
        np.float32(np.inf),
        dtype=np.float32,
    )
    assert separated_value.view(np.uint32) == np.float32(2.0).view(np.uint32) + 1
    separated = _overlap_pair(float(separated_value))
    touching = _overlap_pair(2.0)
    penetrating = _overlap_pair(
        float(np.nextafter(np.float32(2.0), np.float32(0.0)))
    )
    assert _result(OrientedBoxOverlapRateMetric(), *separated).value == 0.0
    assert _result(OrientedBoxOverlapRateMetric(), *touching).value == 0.0
    assert _result(OrientedBoxOverlapRateMetric(), *penetrating).value == 1.0


def test_overlap_masks_invalid_target_and_invalid_counterpart() -> None:
    scenario, rollout = _overlap_pair(
        0.0,
        target_valid=False,
        invalid_padding=True,
    )
    flags, valid = oriented_box_overlap_components(scenario, rollout)
    assert not valid[1, 1]
    assert not valid[1, 2]
    assert not flags[1, 1]
    assert not flags[1, 2]
    result = _result(OrientedBoxOverlapRateMetric(), scenario, rollout)
    assert not result.valid
    assert result.invalid_reason == "no_eligible_target_frame"


def test_invalid_overlapping_counterpart_cannot_trigger_valid_target() -> None:
    timestamps = np.array([0.0, 0.1])
    ego = _agent(0, timestamps, x=-20.0)
    target = _agent(1, timestamps, x=0.0)
    invalid_counterpart = _agent(
        2,
        timestamps,
        valid=False,
        x=0.0,
        length=20.0,
        width=20.0,
    )
    scenario, rollout = _pair(
        timestamps=timestamps,
        source_agents=[ego, target, invalid_counterpart],
    )
    flags, valid = oriented_box_overlap_components(scenario, rollout)
    assert valid[1, 1]
    assert not valid[1, 2]
    assert not flags[1, 1]
    result = _result(OrientedBoxOverlapRateMetric(), scenario, rollout)
    assert result.valid
    assert result.distribution == (0.0,)


def _kinematic_pair(
    *,
    old_speed: np.float32,
    new_speed: np.float32,
    old_heading: np.float32 = np.float32(0.0),
    new_heading: np.float32 = np.float32(0.0),
    new_velocity_heading: np.float32 = np.float32(0.0),
) -> tuple[Scenario, Rollout]:
    timestamps = np.array([0.0, 0.1])
    source = _agent(1, timestamps, vx=[old_speed, new_speed])
    candidate = _agent(
        1,
        timestamps,
        vx=[
            old_speed,
            np.float32(
                new_speed
                * np.cos(new_velocity_heading, dtype=np.float32)
            ),
        ],
        vy=[
            np.float32(0.0),
            np.float32(
                new_speed
                * np.sin(new_velocity_heading, dtype=np.float32)
            ),
        ],
        heading=[old_heading, new_heading],
    )
    return _pair(
        timestamps=timestamps,
        source_agents=[_agent(0, timestamps, x=-20.0), source],
        candidate_agents=[_agent(0, timestamps, x=-20.0), candidate],
    )


def test_kinematic_speed_threshold_uses_exact_strict_branches() -> None:
    exact = _kinematic_pair(
        old_speed=np.float32(0.6),
        new_speed=np.float32(0.6),
        new_heading=np.float32(0.03),
    )
    flags, valid, _, curvature = kinematic_infeasibility_components(*exact)
    assert valid[0, 1]
    assert curvature[0, 1] > np.float32(0.301)
    assert flags[0, 1]

    above = _kinematic_pair(
        old_speed=np.float32(0.6),
        new_speed=np.nextafter(
            np.float32(0.6),
            np.float32(np.inf),
            dtype=np.float32,
        ),
        new_heading=np.float32(1.0),
    )
    flags, _, _, curvature = kinematic_infeasibility_components(*above)
    # Above 0.6, velocity yaw replaces the deliberately conflicting candidate yaw.
    assert curvature[0, 1] == np.float32(0.0)
    assert not flags[0, 1]

    below = _kinematic_pair(
        old_speed=np.nextafter(
            np.float32(0.6),
            np.float32(-np.inf),
            dtype=np.float32,
        ),
        new_speed=np.float32(0.6),
        new_heading=np.float32(1.0),
    )
    flags, _, _, curvature = kinematic_infeasibility_components(*below)
    assert curvature[0, 1] == np.float32(0.0)
    assert not flags[0, 1]


def test_kinematic_acceleration_exact_and_nextafter_threshold() -> None:
    threshold = np.float32(10.401)
    center_speed = np.multiply(threshold, np.float32(0.1), dtype=np.float32)
    cases = [
        np.nextafter(
            center_speed,
            np.float32(-np.inf),
            dtype=np.float32,
        ),
        center_speed,
        np.nextafter(
            center_speed,
            np.float32(np.inf),
            dtype=np.float32,
        ),
    ]
    observed: list[tuple[np.float32, bool]] = []
    for new_speed in cases:
        flags, _, accel, _ = kinematic_infeasibility_components(
            *_kinematic_pair(
                old_speed=np.float32(0.0),
                new_speed=new_speed,
            )
        )
        observed.append((accel[0, 1], bool(flags[0, 1])))
    assert observed[0][0] < threshold and not observed[0][1]
    assert observed[1] == (threshold, False)
    assert observed[2][0] == np.nextafter(
        threshold,
        np.float32(np.inf),
        dtype=np.float32,
    )
    assert observed[2][1]


@pytest.mark.parametrize("threshold", [np.float32(10.401), np.float32(0.301)])
def test_kinematic_exact_and_nextafter_thresholds_are_strict(threshold) -> None:
    below = np.nextafter(
        threshold,
        np.float32(-np.inf),
        dtype=np.float32,
    )
    above = np.nextafter(
        threshold,
        np.float32(np.inf),
        dtype=np.float32,
    )
    values = np.array([below, threshold, above], dtype=np.float32)
    zeros = np.zeros(3, dtype=np.float32)
    if threshold == np.float32(10.401):
        flags = kinematic_infeasibility_flags(values, zeros)
    else:
        flags = kinematic_infeasibility_flags(zeros, values)
    np.testing.assert_array_equal(flags, [False, False, True])


def test_kinematic_metric_uses_fixed_waymax_step_on_nonuniform_timestamps() -> None:
    def case(timestamps: np.ndarray) -> tuple[Scenario, Rollout]:
        return _pair(
            timestamps=timestamps,
            source_agents=[
                _agent(0, timestamps, x=-20.0),
                _agent(1, timestamps, vx=[0.0, 2.0, 2.0]),
            ],
        )

    nominal = case(np.array([0.0, 0.1, 0.2]))
    nonuniform = case(np.array([0.0, 0.100001, 0.200003]))
    nominal_components = kinematic_infeasibility_components(*nominal)
    nonuniform_components = kinematic_infeasibility_components(*nonuniform)
    for nominal_component, nonuniform_component in zip(
        nominal_components,
        nonuniform_components,
        strict=True,
    ):
        np.testing.assert_array_equal(nominal_component, nonuniform_component)

    metric = WaymaxKinematicInfeasibilityRateMetric()
    nominal_result = _result(metric, *nominal)
    nonuniform_result = _result(metric, *nonuniform)
    assert metric.spec.version == M5_KINEMATIC_METRIC_VERSION == "1.0.1"
    assert (
        "fixed_0.1_s_inverse_dynamics_timebase"
        in metric.spec.required_fields
    )
    assert any(
        "not physical-time-normalized" in failure_mode
        for failure_mode in metric.spec.known_failure_modes
    )
    assert nominal_result.value == nonuniform_result.value == 0.5
    assert nominal_result.distribution == nonuniform_result.distribution


@pytest.mark.parametrize(
    ("relative_position", "relative_velocity", "radius", "expected"),
    [
        ([0.5, 0.0], [1.0, 0.0], 1.0, 0.0),
        ([3.0, 0.0], [-1.0, 0.0], 1.0, 2.0),
        ([3.0, 0.0], [1.0, 0.0], 1.0, 5.0),
        ([3.0, 0.0], [0.0, 0.0], 1.0, 5.0),
        ([20.0, 0.0], [-1.0, 0.0], 1.0, 5.0),
        ([3.0, 0.0], [0.0, 1.0], 1.0, 5.0),
    ],
)
def test_constant_velocity_disc_ttc_oracles(
    relative_position,
    relative_velocity,
    radius,
    expected,
) -> None:
    assert constant_velocity_disc_ttc(
        np.array(relative_position),
        np.array(relative_velocity),
        radius,
    ) == pytest.approx(expected)


def test_interaction_metrics_use_frame_minima_and_pair_counts() -> None:
    timestamps = np.array([0.0, 0.1, 0.2])
    source_agents = [
        _agent(0, timestamps, x=0.0),
        _agent(1, timestamps, x=[5.0, 3.0, 4.0], vx=-1.0),
        _agent(2, timestamps, x=10.0, vx=0.0),
    ]
    scenario, rollout = _pair(
        timestamps=timestamps,
        source_agents=source_agents,
    )
    distance = _result(MinimumCenterDistanceMetric(), scenario, rollout)
    assert distance.distribution == pytest.approx((3.0, 4.0))
    assert distance.value == pytest.approx(3.0)
    assert distance.details["evaluated_pair_count"] == 6

    ttc = _result(ConstantVelocityTTCCapMetric(), scenario, rollout)
    expected_ttc = (3.0 - math.sqrt(8.0), 4.0 - math.sqrt(8.0))
    assert ttc.distribution == pytest.approx(expected_ttc)
    assert ttc.value == pytest.approx(expected_ttc[0])
    assert ttc.details["evaluated_pair_count"] == 6


def test_overlap_and_interaction_metrics_are_rigid_and_order_invariant() -> None:
    timestamps = np.array([0.0, 0.1, 0.2])
    agents = [
        _agent(0, timestamps, x=0.0, y=0.0, vx=1.0),
        _agent(1, timestamps, x=1.0, y=0.0, vx=1.0),
        _agent(2, timestamps, x=10.0, y=2.0, vx=-1.0),
    ]
    baseline = _pair(timestamps=timestamps, source_agents=agents)

    angle = 0.8
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    translation = np.array([31.0, -17.0])
    transformed_agents = copy.deepcopy(agents)
    for agent in transformed_agents:
        xy = np.column_stack((agent.x, agent.y)) @ rotation.T + translation
        velocity = np.column_stack((agent.vx, agent.vy)) @ rotation.T
        agent.x, agent.y = xy[:, 0], xy[:, 1]
        agent.vx, agent.vy = velocity[:, 0], velocity[:, 1]
        agent.heading = np.array(
            [
                math.atan2(
                    math.sin(value + angle),
                    math.cos(value + angle),
                )
                for value in agent.heading
            ]
        )
    transformed = _pair(
        timestamps=timestamps,
        source_agents=[
            transformed_agents[0],
            transformed_agents[2],
            transformed_agents[1],
        ],
    )

    for metric in (
        OrientedBoxOverlapRateMetric(),
        MinimumCenterDistanceMetric(),
        ConstantVelocityTTCCapMetric(),
    ):
        original_result = _result(metric, *baseline)
        transformed_result = _result(metric, *transformed)
        assert transformed_result.value == pytest.approx(original_result.value)
        assert transformed_result.eligible_components == (
            original_result.eligible_components
        )


def _lane_pair(
    *,
    lanes: list[np.ndarray] | None = None,
    target_xy: tuple[float, float] = (5.0, 2.0),
    target_heading: float = math.pi / 2.0,
) -> tuple[Scenario, Rollout]:
    timestamps = np.array([0.0, 0.1])
    world = _agent(
        1,
        timestamps,
        x=target_xy[0],
        y=target_xy[1],
        heading=target_heading,
    )
    features = [
        MapPolyline(type=MapType.LANE, xy=lane)
        for lane in (
            [np.array([[0.0, 0.0], [10.0, 0.0]])]
            if lanes is None
            else lanes
        )
    ]
    return _pair(
        timestamps=timestamps,
        source_agents=[_agent(0, timestamps, x=-20.0), world],
        map_features=features,
    )


def test_lane_projection_heading_and_canonical_tie_breaking() -> None:
    scenario, rollout = _lane_pair()
    distance = _result(LaneCenterDistanceMetric(), scenario, rollout)
    heading = _result(LaneHeadingDisagreementMetric(), scenario, rollout)
    assert distance.value == pytest.approx(2.0)
    assert heading.value == pytest.approx(math.pi / 2.0)

    tied = _lane_pair(
        lanes=[
            np.array([[0.0, 1.0], [10.0, 1.0]]),
            np.array([[10.0, -1.0], [0.0, -1.0]]),
        ],
        target_xy=(5.0, 0.0),
        target_heading=0.0,
    )
    assert _result(LaneHeadingDisagreementMetric(), *tied).value == 0.0

    tied_within_polyline = _lane_pair(
        lanes=[
            np.array([[-1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]]),
        ],
        target_xy=(0.0, 1.0),
        target_heading=0.0,
    )
    assert (
        _result(
            LaneHeadingDisagreementMetric(),
            *tied_within_polyline,
        ).value
        == 0.0
    )


def test_lane_metrics_are_rigid_transform_invariant() -> None:
    scenario, rollout = _lane_pair(
        target_xy=(3.0, 2.0),
        target_heading=0.2,
    )
    baseline_distance = _result(
        LaneCenterDistanceMetric(),
        scenario,
        rollout,
    ).value
    baseline_heading = _result(
        LaneHeadingDisagreementMetric(),
        scenario,
        rollout,
    ).value

    angle = 0.7
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    translation = np.array([100.0, -40.0])

    def transform_pair():
        transformed_scenario = copy.deepcopy(scenario)
        transformed_rollout = copy.deepcopy(rollout)
        for container in (transformed_scenario, transformed_rollout):
            for agent in container.agents:
                xy = np.column_stack((agent.x, agent.y)) @ rotation.T + translation
                velocity = np.column_stack((agent.vx, agent.vy)) @ rotation.T
                agent.x, agent.y = xy[:, 0], xy[:, 1]
                agent.vx, agent.vy = velocity[:, 0], velocity[:, 1]
                agent.heading = np.array(
                    [
                        math.atan2(
                            math.sin(value + angle),
                            math.cos(value + angle),
                        )
                        for value in agent.heading
                    ]
                )
        for feature in transformed_scenario.map:
            feature.xy = feature.xy @ rotation.T + translation
        return transformed_scenario, transformed_rollout

    transformed = transform_pair()
    assert _result(LaneCenterDistanceMetric(), *transformed).value == pytest.approx(
        baseline_distance
    )
    assert _result(
        LaneHeadingDisagreementMetric(),
        *transformed,
    ).value == pytest.approx(baseline_heading)


@pytest.mark.parametrize(
    "metric_type",
    [LaneCenterDistanceMetric, LaneHeadingDisagreementMetric],
)
def test_lane_metrics_report_no_supported_lane(metric_type) -> None:
    scenario, rollout = _pair()
    result = _result(metric_type(), scenario, rollout)
    assert not result.valid
    assert result.invalid_reason == "no_supported_lane"


def test_continuity_residual_has_known_trapezoidal_error() -> None:
    timestamps = np.array([0.0, 1.0])
    source = _agent(1, timestamps, x=[0.0, 1.0], vx=[0.0, 0.0])
    scenario, rollout = _pair(
        timestamps=timestamps,
        source_agents=[_agent(0, timestamps, x=-20.0), source],
    )
    result = _result(KinematicContinuityResidualMetric(), scenario, rollout)
    assert result.value == pytest.approx(1.0)


def test_lifecycle_reentry_counts_only_after_first_post_current_validity() -> None:
    timestamps = np.arange(7, dtype=float) * 0.1
    valid_reentry = [True, False, True, False, True, True, True]
    valid_birth_then_reentry = [False, False, True, False, True, False, True]
    scenario, rollout = _pair(
        timestamps=timestamps,
        source_agents=[
            _agent(0, timestamps, x=-20.0),
            _agent(1, timestamps, valid=valid_reentry),
            _agent(2, timestamps, valid=valid_birth_then_reentry),
        ],
    )
    result = _result(LifecycleReentryPerAgentMetric(), scenario, rollout)
    assert result.distribution == (2.0, 2.0)
    assert result.value == 2.0


def test_agent_order_and_never_valid_padding_do_not_change_position_mean() -> None:
    timestamps = np.array([0.0, 0.1, 0.2])
    ego = _agent(0, timestamps, x=-20.0)
    first = _agent(1, timestamps, x=0.0)
    second = _agent(2, timestamps, x=10.0)
    padding = _agent(3, timestamps, valid=False, x=999.0)
    first_candidate = copy.deepcopy(first)
    first_candidate.x += 1.0
    second_candidate = copy.deepcopy(second)
    second_candidate.x += 3.0
    original = _pair(
        timestamps=timestamps,
        source_agents=[ego, first, second],
        candidate_agents=[copy.deepcopy(ego), first_candidate, second_candidate],
    )
    reordered = _pair(
        timestamps=timestamps,
        source_agents=[copy.deepcopy(ego), second, padding, first],
        candidate_agents=[
            copy.deepcopy(ego),
            second_candidate,
            copy.deepcopy(padding),
            first_candidate,
        ],
    )
    assert _result(PositionErrorMetric(), *original).value == pytest.approx(2.0)
    assert _result(PositionErrorMetric(), *reordered).value == pytest.approx(2.0)


def test_invalid_gap_numeric_poison_is_never_interpreted() -> None:
    timestamps = np.arange(5, dtype=float) * 0.1
    valid = np.array([True, True, False, True, True])
    clean_source = _agent(1, timestamps, valid=valid, x=[0, 1, 0, 3, 4], vx=1.0)
    clean_candidate = _agent(
        1,
        timestamps,
        valid=valid,
        x=[0, 1.1, 0, 3.2, 4.3],
        vx=[1.0, 1.1, 0.0, 1.2, 1.3],
    )
    poisoned_source = copy.deepcopy(clean_source)
    poisoned_candidate = copy.deepcopy(clean_candidate)
    for agent, poison in (
        (poisoned_source, np.nan),
        (poisoned_candidate, np.inf),
    ):
        for field in ("x", "y", "heading", "vx", "vy"):
            getattr(agent, field)[2] = poison
    clean = _pair(
        timestamps=timestamps,
        source_agents=[_agent(0, timestamps, x=-20.0), clean_source],
        candidate_agents=[
            _agent(0, timestamps, x=-20.0),
            clean_candidate,
        ],
    )
    poisoned = _pair(
        timestamps=timestamps,
        source_agents=[
            _agent(0, timestamps, x=-20.0),
            poisoned_source,
        ],
        candidate_agents=[
            _agent(0, timestamps, x=-20.0),
            poisoned_candidate,
        ],
    )
    for metric in (
        PositionErrorMetric(),
        SpeedErrorMetric(),
        AccelerationErrorMetric(),
        JerkErrorMetric(),
        YawRateErrorMetric(),
        KinematicContinuityResidualMetric(),
    ):
        with np.errstate(all="ignore"):
            clean_result = _result(metric, *clean)
            poisoned_result = _result(metric, *poisoned)
        assert poisoned_result.value == clean_result.value
        assert poisoned_result.distribution == clean_result.distribution


def test_never_valid_padding_numeric_poison_does_not_change_metric_values() -> None:
    timestamps = np.arange(4, dtype=float) * 0.1
    ego = _agent(0, timestamps, x=-20.0)
    world = _agent(1, timestamps, x=np.arange(4), vx=10.0)
    candidate = copy.deepcopy(world)
    candidate.x += 0.5
    baseline = _pair(
        timestamps=timestamps,
        source_agents=[ego, world],
        candidate_agents=[copy.deepcopy(ego), candidate],
    )
    padding_source = _agent(99, timestamps, valid=False)
    padding_candidate = copy.deepcopy(padding_source)
    for agent, poison in (
        (padding_source, np.nan),
        (padding_candidate, -np.inf),
    ):
        for field in ("x", "y", "heading", "vx", "vy"):
            getattr(agent, field)[:] = poison
    padded = _pair(
        timestamps=timestamps,
        source_agents=[copy.deepcopy(ego), copy.deepcopy(world), padding_source],
        candidate_agents=[
            copy.deepcopy(ego),
            copy.deepcopy(candidate),
            padding_candidate,
        ],
    )
    for metric in (
        PositionErrorMetric(),
        SpeedErrorMetric(),
        OrientedBoxOverlapRateMetric(),
        WaymaxKinematicInfeasibilityRateMetric(),
        AccelerationErrorMetric(),
        JerkErrorMetric(),
        YawRateErrorMetric(),
        KinematicContinuityResidualMetric(),
        LifecycleReentryPerAgentMetric(),
    ):
        with np.errstate(all="ignore"):
            baseline_result = _result(metric, *baseline)
            padded_result = _result(metric, *padded)
        assert padded_result.value == baseline_result.value


def test_all_metrics_survive_contract_parquet_roundtrip(tmp_path) -> None:
    timestamps = np.arange(6, dtype=float) * 0.1
    lane = MapPolyline(
        type=MapType.LANE,
        xy=np.array([[-20.0, 0.0], [50.0, 0.0]]),
    )
    source_agents = [
        _agent(0, timestamps, x=-10.0),
        _agent(1, timestamps, x=np.arange(6), vx=10.0),
        _agent(
            2,
            timestamps,
            agent_type=AgentType.PEDESTRIAN,
            x=20.0,
            vx=0.0,
            length=0.8,
            width=0.6,
        ),
    ]
    scenario, rollout = _pair(
        timestamps=timestamps,
        source_agents=source_agents,
        map_features=[lane],
    )
    scenario_path = tmp_path / "scenario.parquet"
    rollout_path = tmp_path / "rollout.parquet"
    scenario_to_parquet(scenario, scenario_path)
    rollout_to_parquet(rollout, rollout_path)
    restored = (
        scenario_from_parquet(scenario_path),
        rollout_from_parquet(rollout_path),
    )
    registry = MetricRegistry(m5_metrics())
    original_results = registry.evaluate(scenario, rollout)
    restored_results = registry.evaluate(*restored)
    assert original_results == restored_results
    assert len(original_results) == 13
    assert all("composite" not in result.metric_name for result in original_results)


def test_metric_registry_order_is_independent_of_input_order() -> None:
    scenario, rollout = _lane_pair()
    forward = MetricRegistry(m5_metrics()).evaluate(scenario, rollout)
    reverse = MetricRegistry(reversed(m5_metrics())).evaluate(scenario, rollout)
    assert forward == reverse
    assert [result.metric_name for result in forward] == sorted(
        result.metric_name for result in forward
    )


def test_policy_dependent_validity_is_a_contract_failure() -> None:
    scenario, rollout = _pair()
    rollout.agents[1].valid[-1] = False
    with pytest.raises(ValueError, match="validity_mismatch"):
        PositionErrorMetric().compute(scenario, rollout)


def test_eligibility_accepts_only_the_source_scenario(
    monkeypatch,
) -> None:
    scenario, rollout = _pair()
    metric = PositionErrorMetric()

    def forbidden_outcome(*_args, **_kwargs):
        raise AssertionError("eligibility interpreted a metric outcome")

    monkeypatch.setattr(metric, "_outcome", forbidden_outcome)
    eligibility = metric.eligibility(scenario)
    assert eligibility.eligible
    with pytest.raises(TypeError):
        metric.eligibility(scenario, rollout)
