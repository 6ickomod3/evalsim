"""Independent analytic and source-only tests for the frozen M6 interventions."""
from __future__ import annotations

import math

import numpy as np
import pytest

from evalsim.contracts import Agent, AgentType, Scenario
from evalsim.perturb.m6 import (
    BRAKE_PULSE_DURATION_S,
    M6_ANALYSIS_TRANSITIONS,
    PRIMARY_BRAKE_MAGNITUDE_MPS2,
    PRIMARY_ELIGIBILITY_REASONS,
    SECONDARY_BRAKE_MAGNITUDE_MPS2,
    InterventionCompilationError,
    audit_ego_plan_feasibility,
    compile_identity_plan,
    compile_longitudinal_brake_pulse_plan,
    evaluate_primary_brake_eligibility,
    identity_spec,
    longitudinal_brake_pulse_spec,
    zero_dose_reconstruction_matches,
)


def _wrap_for_oracle(value: np.ndarray | float) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    wrapped = (raw + np.pi) % (2.0 * np.pi) - np.pi
    return np.where(
        np.isclose(wrapped, -np.pi, rtol=0.0, atol=1e-15)
        & (raw > 0.0),
        np.pi,
        wrapped,
    )


def _agent(
    agent_id: int,
    *,
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    valid: np.ndarray | None = None,
    agent_type: AgentType = AgentType.VEHICLE,
    length: float = 4.5,
    width: float = 2.0,
) -> Agent:
    frame_count = len(x)
    return Agent(
        id=agent_id,
        type=agent_type,
        valid=(
            np.ones(frame_count, dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        ),
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        heading=np.asarray(heading, dtype=np.float64),
        vx=np.asarray(vx, dtype=np.float64),
        vy=np.asarray(vy, dtype=np.float64),
        length=length,
        width=width,
    )


def _straight_scenario(
    *,
    timestamps: np.ndarray | None = None,
    current_index: int = 0,
    ego_speed: float = 10.0,
    follower_offset_m: float = -15.0,
    follower_lateral_m: float = 0.0,
    include_follower: bool = True,
) -> Scenario:
    if timestamps is None:
        timestamps = np.arange(41, dtype=np.float64) * 0.1
    timestamps = np.asarray(timestamps, dtype=np.float64)
    elapsed = timestamps - timestamps[0]
    ego_x = ego_speed * elapsed
    zeros = np.zeros_like(elapsed)
    ego = _agent(
        100,
        x=ego_x,
        y=zeros,
        heading=zeros,
        vx=np.full_like(elapsed, ego_speed),
        vy=zeros,
    )
    agents = [ego]
    if include_follower:
        agents.append(
            _agent(
                200,
                x=ego_x + follower_offset_m,
                y=np.full_like(elapsed, follower_lateral_m),
                heading=zeros,
                vx=np.full_like(elapsed, ego_speed),
                vy=zeros,
            )
        )
    return Scenario(
        scenario_id="m6-test-scene",
        timestamps=timestamps,
        agents=agents,
        ego_index=0,
        metadata={"source": "synthetic", "current_index": current_index},
    )


def _manual_brake_progress(
    timestamps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    magnitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Test oracle written directly from the preregistered equations."""

    elapsed = timestamps - timestamps[0]
    deficit = magnitude * np.minimum(
        np.maximum(elapsed, 0.0),
        BRAKE_PULSE_DURATION_S,
    )
    segment = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    progress = np.zeros(len(timestamps), dtype=np.float64)
    for interval in range(len(timestamps) - 1):
        dt = elapsed[interval + 1] - elapsed[interval]
        lost = min(
            segment[interval],
            0.5 * (deficit[interval] + deficit[interval + 1]) * dt,
        )
        progress[interval + 1] = (
            progress[interval] + segment[interval] - lost
        )
    return progress, deficit


def _manual_polyline_state(
    *,
    arc_knots: np.ndarray,
    progress: float,
    x: np.ndarray,
    y: np.ndarray,
    raw_heading: np.ndarray,
    unwrapped_heading: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    final_speed: float,
) -> tuple[float, float, float, float, float]:
    exact = np.flatnonzero(arc_knots == progress)
    if exact.size:
        index = int(exact[0])
        raw_x = float(x[index])
        raw_y = float(y[index])
        selected_heading = float(raw_heading[index])
        raw_vx = float(vx[index])
        raw_vy = float(vy[index])
    else:
        left = 0
        while not (
            arc_knots[left] <= progress < arc_knots[left + 1]
        ):
            left += 1
        alpha = (
            (progress - arc_knots[left])
            / (arc_knots[left + 1] - arc_knots[left])
        )
        raw_x = float(x[left] + alpha * (x[left + 1] - x[left]))
        raw_y = float(y[left] + alpha * (y[left + 1] - y[left]))
        selected_heading = float(
            _wrap_for_oracle(
                unwrapped_heading[left]
                + alpha
                * (
                    unwrapped_heading[left + 1]
                    - unwrapped_heading[left]
                )
            )
        )
        raw_vx = float(vx[left] + alpha * (vx[left + 1] - vx[left]))
        raw_vy = float(vy[left] + alpha * (vy[left + 1] - vy[left]))
    norm = math.hypot(raw_vx, raw_vy)
    if norm <= 1e-12:
        direction_x = math.cos(selected_heading)
        direction_y = math.sin(selected_heading)
    else:
        direction_x = raw_vx / norm
        direction_y = raw_vy / norm
    return (
        raw_x,
        raw_y,
        selected_heading,
        final_speed * direction_x,
        final_speed * direction_y,
    )


def test_registered_specs_are_stable_and_zero_is_not_a_treatment() -> None:
    sham = identity_spec()
    primary = longitudinal_brake_pulse_spec(
        PRIMARY_BRAKE_MAGNITUDE_MPS2
    )
    severe = longitudinal_brake_pulse_spec(
        SECONDARY_BRAKE_MAGNITUDE_MPS2
    )

    assert sham.intervention_id == "identity/v1"
    assert primary.intervention_id == "longitudinal_brake_pulse/v1"
    assert severe.intervention_id == "longitudinal_brake_pulse/v1"
    assert primary.dose == 2.0
    assert severe.dose == 4.0
    assert primary.configuration_fingerprint != severe.configuration_fingerprint
    with pytest.raises(ValueError, match="exactly 2.0 or 4.0"):
        longitudinal_brake_pulse_spec(0.0)


def test_identity_is_exact_but_carries_typed_sham_provenance() -> None:
    scenario = _straight_scenario()
    ego = scenario.ego
    ego.x[0] = -0.0
    plan = compile_identity_plan(scenario)

    assert plan.perturbation_identity.startswith("identity/v1@sha256:")
    np.testing.assert_array_equal(plan.timestamps, scenario.timestamps)
    np.testing.assert_array_equal(plan.valid, ego.valid)
    np.testing.assert_array_equal(plan.x, ego.x)
    np.testing.assert_array_equal(plan.y, ego.y)
    np.testing.assert_array_equal(plan.heading, ego.heading)
    np.testing.assert_array_equal(plan.vx, ego.vx)
    np.testing.assert_array_equal(plan.vy, ego.vy)
    np.testing.assert_array_equal(
        plan.applied,
        np.r_[False, np.ones(M6_ANALYSIS_TRANSITIONS, dtype=bool)],
    )
    assert plan.feasibility.passed
    assert audit_ego_plan_feasibility(scenario, plan).passed
    assert zero_dose_reconstruction_matches(scenario)


def test_general_interpolator_returns_raw_bits_at_all_knots_and_closed_end() -> None:
    """Independently pin the exact-knot branch, including signed branch bits."""

    import evalsim.perturb.m6 as module

    frame = np.arange(41, dtype=np.float64)
    x = np.array(frame, copy=True)
    x[0] = -0.0
    y = np.zeros(41, dtype=np.float64)
    y[0] = -0.0
    heading = _wrap_for_oracle(3.0 + 0.03 * frame)
    heading[8] = np.pi
    heading[9] = -np.pi
    vx = 7.0 + 0.01 * frame
    vy = np.zeros(41, dtype=np.float64)
    vy[::2] = -0.0
    scenario = Scenario(
        scenario_id="raw-exact-knots",
        timestamps=frame * 0.1,
        agents=[
            _agent(
                100,
                x=x,
                y=y,
                heading=heading,
                vx=vx,
                vy=vy,
            )
        ],
        ego_index=0,
        metadata={"source": "synthetic", "current_index": 0},
    )
    source = module._source_window(scenario, stop_index=40)

    # Construct source arc knots independently rather than calling the production
    # arc helper under test.
    arc_knots = np.zeros(41, dtype=np.float64)
    for index in range(40):
        arc_knots[index + 1] = arc_knots[index] + math.hypot(
            float(source.x[index + 1] - source.x[index]),
            float(source.y[index + 1] - source.y[index]),
        )
    unwrapped = np.unwrap(np.asarray(source.heading), discont=np.pi)
    raw_fields = ("x", "y", "heading", "vx", "vy")
    for index, progress in enumerate(arc_knots):
        actual = np.asarray(
            module._interpolate_source(
                source,
                arc_knots,
                unwrapped,
                float(progress),
            ),
            dtype="<f8",
        )
        expected = np.asarray(
            [getattr(source, name)[index] for name in raw_fields],
            dtype="<f8",
        )
        np.testing.assert_array_equal(
            actual.view("<u8"),
            expected.view("<u8"),
        )

    # The last knot is explicitly closed, not an extrapolation or previous segment.
    final = np.asarray(
        module._interpolate_source(
            source,
            arc_knots,
            unwrapped,
            float(arc_knots[-1]),
        ),
        dtype="<f8",
    )
    final_expected = np.asarray(
        [getattr(source, name)[-1] for name in raw_fields],
        dtype="<f8",
    )
    np.testing.assert_array_equal(
        final.view("<u8"),
        final_expected.view("<u8"),
    )

    # The b=0 special case is a separate raw-copy path and must also preserve bits.
    zero = module._brake_arrays(source, arc_knots, 0.0)
    for name in raw_fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(zero, name), dtype="<f8").view("<u8"),
            np.asarray(getattr(source, name), dtype="<f8").view("<u8"),
        )
    assert zero_dose_reconstruction_matches(scenario)


def test_straight_brake_matches_closed_form_and_carries_deficit() -> None:
    scenario = _straight_scenario()
    plan = compile_longitudinal_brake_pulse_plan(scenario, 2.0)
    elapsed = scenario.timestamps - scenario.timestamps[0]
    deficit = 2.0 * np.minimum(elapsed, 1.0)
    loss = np.where(
        elapsed <= 1.0,
        elapsed**2,
        2.0 * (elapsed - 0.5),
    )
    expected_x = 10.0 * elapsed - loss
    expected_speed = 10.0 - deficit

    np.testing.assert_allclose(plan.x, expected_x, rtol=0.0, atol=3e-14)
    np.testing.assert_array_equal(plan.y, np.zeros(41))
    np.testing.assert_allclose(plan.vx, expected_speed, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(plan.vy, np.zeros(41))
    assert plan.x[0] == scenario.ego.x[0]
    assert plan.vx[0] == scenario.ego.vx[0]
    # The pulse stops ramping at one second, but the 2 m/s deficit persists.
    np.testing.assert_allclose(plan.vx[10:], 8.0, rtol=0.0, atol=1e-15)
    assert plan.feasibility.passed


def test_irregular_timestamps_use_actual_interval_overlap() -> None:
    intervals = np.resize(
        np.array([0.08, 0.17, 0.06, 0.11], dtype=np.float64),
        40,
    )
    timestamps = np.r_[0.0, np.cumsum(intervals)]
    scenario = _straight_scenario(timestamps=timestamps)
    plan = compile_longitudinal_brake_pulse_plan(scenario, 2.0)
    progress, deficit = _manual_brake_progress(
        timestamps,
        scenario.ego.x,
        scenario.ego.y,
        2.0,
    )

    np.testing.assert_allclose(plan.x, progress, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(plan.vx, 10.0 - deficit, rtol=0.0, atol=1e-15)
    crossing = int(np.flatnonzero(timestamps > 1.0)[0])
    assert timestamps[crossing - 1] < 1.0 < timestamps[crossing]
    assert plan.feasibility.passed


def test_curved_path_interpolation_matches_independent_oracle() -> None:
    timestamps = np.arange(41, dtype=np.float64) * 0.1
    radius = 50.0
    speed = 10.0
    unwrapped_heading = (speed / radius) * timestamps
    x = radius * np.sin(unwrapped_heading)
    y = radius * (1.0 - np.cos(unwrapped_heading))
    heading = _wrap_for_oracle(unwrapped_heading)
    vx = speed * np.cos(unwrapped_heading)
    vy = speed * np.sin(unwrapped_heading)
    ego = _agent(
        100,
        x=x,
        y=y,
        heading=heading,
        vx=vx,
        vy=vy,
    )
    follower = _agent(
        200,
        x=x - 15.0 * np.cos(unwrapped_heading),
        y=y - 15.0 * np.sin(unwrapped_heading),
        heading=heading,
        vx=vx,
        vy=vy,
    )
    scenario = Scenario(
        scenario_id="curved",
        timestamps=timestamps,
        agents=[ego, follower],
        ego_index=0,
        metadata={"source": "synthetic", "current_index": 0},
    )
    plan = compile_longitudinal_brake_pulse_plan(scenario, 2.0)
    progress, deficit = _manual_brake_progress(timestamps, x, y, 2.0)
    arc_knots = np.r_[
        0.0,
        np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)),
    ]
    frame = 17
    expected = _manual_polyline_state(
        arc_knots=arc_knots,
        progress=float(progress[frame]),
        x=x,
        y=y,
        raw_heading=heading,
        unwrapped_heading=unwrapped_heading,
        vx=vx,
        vy=vy,
        final_speed=speed - deficit[frame],
    )

    np.testing.assert_allclose(
        np.array(
            [
                plan.x[frame],
                plan.y[frame],
                plan.heading[frame],
                plan.vx[frame],
                plan.vy[frame],
            ]
        ),
        expected,
        rtol=0.0,
        atol=2e-14,
    )
    assert plan.feasibility.passed


def test_branch_crossing_heading_interpolates_through_pi_not_zero() -> None:
    timestamps = np.arange(41, dtype=np.float64) * 0.1
    unwrapped_heading = 3.0 + 0.03 * np.arange(41, dtype=np.float64)
    heading = _wrap_for_oracle(unwrapped_heading)
    speed = 10.0
    vx = speed * np.cos(unwrapped_heading)
    vy = speed * np.sin(unwrapped_heading)
    x = np.zeros(41, dtype=np.float64)
    y = np.zeros(41, dtype=np.float64)
    for frame in range(40):
        x[frame + 1] = x[frame] + 0.05 * (vx[frame] + vx[frame + 1])
        y[frame + 1] = y[frame] + 0.05 * (vy[frame] + vy[frame + 1])
    ego = _agent(
        100,
        x=x,
        y=y,
        heading=heading,
        vx=vx,
        vy=vy,
    )
    scenario = Scenario(
        scenario_id="heading-branch",
        timestamps=timestamps,
        agents=[ego],
        ego_index=0,
        metadata={"source": "synthetic", "current_index": 0},
    )
    plan = compile_longitudinal_brake_pulse_plan(scenario, 2.0)

    continuous = np.unwrap(np.asarray(plan.heading))
    assert np.all(np.diff(continuous) >= -1e-12)
    branch_frames = np.flatnonzero(
        (np.asarray(plan.heading[:-1]) > 3.0)
        & (np.asarray(plan.heading[1:]) < -3.0)
    )
    assert branch_frames.size == 1
    assert np.all(np.abs(plan.heading) > 2.2)


def test_zero_velocity_interpolation_boundary_uses_heading_direction() -> None:
    timestamps = np.arange(41, dtype=np.float64) * 0.1
    source_speed = 0.4 * np.arange(41, dtype=np.float64)
    segment = 0.5 * (source_speed[:-1] + source_speed[1:]) * 0.1
    # The first source segment is shorter than the first interval's 0.01 m
    # treatment loss, so q is exactly the source-segment bits and progress stays
    # on source knot zero.
    segment[0] = 0.005
    x = np.r_[0.0, np.cumsum(segment)]
    zeros = np.zeros(41, dtype=np.float64)
    ego = _agent(
        100,
        x=x,
        y=zeros,
        heading=zeros,
        vx=source_speed,
        vy=zeros,
    )
    scenario = Scenario(
        scenario_id="stopped-boundary",
        timestamps=timestamps,
        agents=[ego],
        ego_index=0,
        metadata={"source": "synthetic", "current_index": 0},
    )
    plan = compile_longitudinal_brake_pulse_plan(scenario, 2.0)

    # At frame 1 the treatment progress is exactly source knot zero, whose source
    # velocity is zero.  The final 0.2 m/s vector therefore uses heading.
    assert plan.x[1] == x[0]
    assert plan.y[1] == 0.0
    assert plan.vx[1] == pytest.approx(0.2)
    assert plan.vy[1] == 0.0
    assert plan.feasibility.passed


def test_path_end_is_closed_and_braking_stays_inside_template() -> None:
    scenario = _straight_scenario()
    identity = compile_identity_plan(scenario)
    brake = compile_longitudinal_brake_pulse_plan(scenario, 2.0)

    assert identity.x[-1] == scenario.ego.x[-1]
    assert identity.heading[-1] == scenario.ego.heading[-1]
    assert brake.x[-1] < scenario.ego.x[-1]
    assert brake.x[-1] >= scenario.ego.x[0]


def test_nested_severity_never_increases_speed_or_progress() -> None:
    scenario = _straight_scenario()
    primary = compile_longitudinal_brake_pulse_plan(scenario, 2.0)
    severe = compile_longitudinal_brake_pulse_plan(scenario, 4.0)
    primary_speed = np.hypot(primary.vx, primary.vy)
    severe_speed = np.hypot(severe.vx, severe.vy)

    np.testing.assert_array_equal(
        np.array(
            [
                primary.x[0],
                primary.y[0],
                primary.heading[0],
                primary.vx[0],
                primary.vy[0],
            ]
        ),
        np.array(
            [
                severe.x[0],
                severe.y[0],
                severe.heading[0],
                severe.vx[0],
                severe.vy[0],
            ]
        ),
    )
    assert np.all(severe_speed[1:] <= primary_speed[1:])
    assert np.all(severe.x[1:] <= primary.x[1:])


@pytest.mark.parametrize(
    ("failed_check", "mutate"),
    [
        (
            "finite",
            lambda scene: scene.ego.vx.__setitem__(7, np.nan),
        ),
        (
            "speed_bounds",
            lambda scene: scene.ego.vx.__setitem__(slice(None), 60.1),
        ),
        (
            "acceleration_bounds",
            lambda scene: scene.ego.vx.__setitem__(1, 11.0),
        ),
        (
            "yaw_rate_bounds",
            lambda scene: scene.ego.heading.__setitem__(1, 0.2),
        ),
        (
            "displacement_upper_bound",
            lambda scene: scene.ego.x.__setitem__(
                slice(1, None),
                scene.ego.x[1:] + 0.06,
            ),
        ),
        (
            "distance_residual",
            lambda scene: scene.ego.x.__setitem__(
                slice(1, None),
                scene.ego.x[1:] - 0.11,
            ),
        ),
        (
            "heading_velocity_alignment",
            lambda scene: scene.ego.heading.__setitem__(
                slice(None),
                0.6,
            ),
        ),
    ],
)
def test_feasibility_audit_rejects_each_frozen_bound(
    failed_check: str,
    mutate,
) -> None:
    scenario = _straight_scenario()
    mutate(scenario)
    with pytest.raises(
        InterventionCompilationError,
        match="ego_plan_infeasible",
    ) as caught:
        compile_identity_plan(scenario)

    audit = caught.value.feasibility
    assert audit is not None
    assert not audit.passed
    assert not audit.checks[failed_check]
    assert audit.failure_reason == f"{failed_check}_failed"


def test_acceleration_and_speed_closed_bounds_are_inclusive() -> None:
    timestamps = np.arange(41, dtype=np.float64) * 0.25
    speed = np.full(41, 9.0, dtype=np.float64)
    speed[0] = 10.0
    speed[1] = 11.0  # +4 m/s², then -8 m/s²: both exact closed bounds.
    x = np.zeros(41, dtype=np.float64)
    for frame in range(40):
        x[frame + 1] = (
            x[frame] + 0.5 * (speed[frame] + speed[frame + 1]) * 0.25
        )
    zeros = np.zeros(41, dtype=np.float64)
    scenario = Scenario(
        scenario_id="closed-accel-bounds",
        timestamps=timestamps,
        agents=[
            _agent(
                100,
                x=x,
                y=zeros,
                heading=zeros,
                vx=speed,
                vy=zeros,
            )
        ],
        ego_index=0,
        metadata={"source": "synthetic", "current_index": 0},
    )
    plan = compile_identity_plan(scenario)
    assert plan.feasibility.checks["acceleration_bounds"]
    assert plan.feasibility.checks["speed_bounds"]


def test_yaw_rate_and_heading_disagreement_closed_bounds_are_inclusive() -> None:
    timestamps = np.arange(41, dtype=np.float64) * 0.25
    unwrapped = 0.25 * np.arange(41, dtype=np.float64)
    heading = _wrap_for_oracle(unwrapped)
    speed = np.full(41, 10.0, dtype=np.float64)
    vx = speed * np.cos(unwrapped)
    vy = speed * np.sin(unwrapped)
    x = np.zeros(41, dtype=np.float64)
    y = np.zeros(41, dtype=np.float64)
    for frame in range(40):
        x[frame + 1] = x[frame] + 0.125 * (vx[frame] + vx[frame + 1])
        y[frame + 1] = y[frame] + 0.125 * (vy[frame] + vy[frame + 1])
    scenario = Scenario(
        scenario_id="closed-yaw-bound",
        timestamps=timestamps,
        agents=[
            _agent(
                100,
                x=x,
                y=y,
                heading=heading,
                vx=vx,
                vy=vy,
            )
        ],
        ego_index=0,
        metadata={"source": "synthetic", "current_index": 0},
    )
    plan = compile_identity_plan(scenario)
    assert plan.feasibility.checks["yaw_rate_bounds"]
    assert plan.feasibility.checks["heading_velocity_alignment"]

    aligned_boundary = _straight_scenario()
    aligned_boundary.ego.heading[:] = np.pi / 6.0
    boundary_plan = compile_identity_plan(aligned_boundary)
    assert boundary_plan.feasibility.checks["heading_velocity_alignment"]


def test_reaudit_detects_source_identity_mismatch() -> None:
    source = _straight_scenario()
    plan = compile_identity_plan(source)
    other = _straight_scenario()
    other.ego.x += 1.0

    audit = audit_ego_plan_feasibility(other, plan)
    assert not audit.passed
    assert not audit.checks["source_identity"]
    assert audit.failure_reason == "source_identity_failed"


def test_degenerate_path_is_registered_before_compilation() -> None:
    scenario = _straight_scenario()
    scenario.ego.x[12] = scenario.ego.x[11]
    with pytest.raises(
        InterventionCompilationError,
        match="source_ego_path_degenerate",
    ) as caught:
        compile_longitudinal_brake_pulse_plan(scenario, 2.0)
    assert caught.value.code == "source_ego_path_degenerate"


def test_frozen_horizon_rejects_short_or_arbitrary_stop() -> None:
    short = _straight_scenario(timestamps=np.arange(40) * 0.1)
    with pytest.raises(
        InterventionCompilationError,
        match="insufficient_future_horizon",
    ):
        compile_identity_plan(short)

    scenario = _straight_scenario()
    with pytest.raises(ValueError, match=r"current_index \+ 40"):
        compile_identity_plan(scenario, stop_index=20)


def test_compiler_slices_from_nonzero_current_index() -> None:
    timestamps = np.arange(46, dtype=np.float64) * 0.1
    scenario = _straight_scenario(
        timestamps=timestamps,
        current_index=5,
    )
    plan = compile_longitudinal_brake_pulse_plan(scenario, 2.0)

    np.testing.assert_array_equal(plan.timestamps, timestamps[5:46])
    assert plan.x[0] == scenario.ego.x[5]
    assert plan.vx[0] == scenario.ego.vx[5]
    assert plan.frame_count == 41


def test_direct_compilation_rejects_invalid_ego_window() -> None:
    scenario = _straight_scenario()
    scenario.ego.valid[30] = False
    with pytest.raises(
        InterventionCompilationError,
        match="ego_invalid_in_window",
    ):
        compile_identity_plan(scenario)
    with pytest.raises(
        InterventionCompilationError,
        match="ego_invalid_in_window",
    ):
        compile_longitudinal_brake_pulse_plan(scenario, 2.0)


def test_primary_eligibility_accepts_and_freezes_target() -> None:
    eligibility = evaluate_primary_brake_eligibility(_straight_scenario())
    assert eligibility.eligible
    assert eligibility.reason is None
    assert eligibility.analysis_window == (0, 40)
    assert eligibility.target_index == 1


def test_target_tie_breaks_by_integer_agent_id_before_contract_index() -> None:
    scenario = _straight_scenario(include_follower=False)
    zeros = np.zeros(41, dtype=np.float64)
    for agent_id, lateral in ((300, 2.5), (150, -2.5)):
        scenario.agents.append(
            _agent(
                agent_id,
                x=scenario.ego.x - 15.0,
                y=np.full(41, lateral, dtype=np.float64),
                heading=zeros,
                vx=np.full(41, 10.0, dtype=np.float64),
                vy=zeros,
            )
        )

    eligibility = evaluate_primary_brake_eligibility(scenario)
    assert eligibility.eligible
    assert scenario.agents[eligibility.target_index].id == 150
    assert eligibility.target_index == 2


def test_nearest_leader_exact_gap_tie_breaks_by_integer_agent_id() -> None:
    scenario = _straight_scenario()
    zeros = np.zeros(41, dtype=np.float64)
    scenario.agents.append(
        _agent(
            50,  # Lower than ego ID 100, so this exact-gap tie wins.
            x=np.array(scenario.ego.x, copy=True),
            y=zeros,
            heading=zeros,
            vx=np.full(41, 10.0, dtype=np.float64),
            vy=zeros,
            agent_type=AgentType.PEDESTRIAN,
        )
    )

    eligibility = evaluate_primary_brake_eligibility(scenario)
    assert not eligibility.eligible
    assert eligibility.reason == "no_stable_aligned_follower"


def test_combined_rejections_follow_registered_priority() -> None:
    invalid_first = _straight_scenario()
    invalid_first.ego.valid[20] = False
    invalid_first.ego.vx[0] = 4.0
    invalid_first.ego.x[12] = invalid_first.ego.x[11]
    invalid_first.agents[1].y[:] = 3.0
    assert (
        evaluate_primary_brake_eligibility(invalid_first).reason
        == "ego_invalid_in_window"
    )

    speed_first = _straight_scenario()
    speed_first.ego.vx[0] = 4.0
    speed_first.ego.x[12] = speed_first.ego.x[11]
    speed_first.agents[1].y[:] = 3.0
    assert (
        evaluate_primary_brake_eligibility(speed_first).reason
        == "ego_speed_below_5_mps"
    )

    path_first = _straight_scenario()
    path_first.ego.x[12] = path_first.ego.x[11]
    path_first.ego.heading[:] = 0.6
    path_first.agents[1].x[:] = path_first.ego.x - 1.0
    assert (
        evaluate_primary_brake_eligibility(path_first).reason
        == "source_ego_path_degenerate"
    )

    plan_first = _straight_scenario()
    plan_first.ego.heading[:] = 0.6
    plan_first.agents[1].x[:] = plan_first.ego.x - 1.0
    assert (
        evaluate_primary_brake_eligibility(plan_first).reason
        == "primary_ego_plan_infeasible"
    )


@pytest.mark.parametrize(
    ("reason", "mutate"),
    [
        (
            "insufficient_future_horizon",
            lambda scene: setattr(
                scene,
                "timestamps",
                np.asarray(scene.timestamps[:40]),
            ),
        ),
        (
            "ego_invalid_in_window",
            lambda scene: scene.ego.valid.__setitem__(20, False),
        ),
        (
            "ego_speed_below_5_mps",
            lambda scene: (
                scene.ego.vx.__setitem__(0, 4.99),
                scene.ego.x.__setitem__(slice(None), np.arange(41) * 0.499),
            ),
        ),
        (
            "source_ego_path_degenerate",
            lambda scene: scene.ego.x.__setitem__(12, scene.ego.x[11]),
        ),
        (
            "primary_ego_plan_infeasible",
            lambda scene: scene.ego.heading.__setitem__(
                slice(None),
                np.pi / 2.0,
            ),
        ),
        (
            "no_stable_aligned_follower",
            lambda scene: scene.agents[1].y.__setitem__(
                slice(None),
                3.0,
            ),
        ),
        (
            "current_ego_follower_overlap",
            lambda scene: scene.agents[1].x.__setitem__(
                slice(None),
                scene.ego.x - 1.0,
            ),
        ),
    ],
)
def test_registered_eligibility_rejection_order(
    reason: str,
    mutate,
) -> None:
    scenario = _straight_scenario()
    mutate(scenario)
    # The horizon fixture needs all agent arrays trimmed with its timestamps.
    if reason == "insufficient_future_horizon":
        for agent in scenario.agents:
            for name in ("valid", "x", "y", "heading", "vx", "vy"):
                setattr(agent, name, np.asarray(getattr(agent, name)[:40]))
    eligibility = evaluate_primary_brake_eligibility(scenario)
    assert not eligibility.eligible
    assert eligibility.reason == reason
    assert reason in PRIMARY_ELIGIBILITY_REASONS


def test_zero_reconstruction_reason_has_frozen_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evalsim.perturb.m6 as module

    monkeypatch.setattr(
        module,
        "_zero_dose_reconstruction_matches",
        lambda source, knots: False,
    )
    eligibility = evaluate_primary_brake_eligibility(_straight_scenario())
    assert not eligibility.eligible
    assert eligibility.reason == "zero_dose_reconstruction_mismatch"


def test_ego_must_be_followers_nearest_forward_leader() -> None:
    scenario = _straight_scenario()
    zeros = np.zeros(41, dtype=np.float64)
    nearer_nonvehicle = _agent(
        150,
        x=scenario.ego.x - 5.0,
        y=zeros,
        heading=zeros,
        vx=np.full(41, 10.0),
        vy=zeros,
        agent_type=AgentType.PEDESTRIAN,
    )
    scenario.agents.append(nearer_nonvehicle)

    eligibility = evaluate_primary_brake_eligibility(scenario)
    assert not eligibility.eligible
    assert eligibility.reason == "no_stable_aligned_follower"


def test_exact_box_edge_touch_is_not_positive_overlap() -> None:
    # Equal 4.5 m boxes touch at one edge when their centers differ by 4.5 m.
    # The gap fails Stage B, but strict SAT must not relabel it as overlap.
    scenario = _straight_scenario(follower_offset_m=-4.5)
    eligibility = evaluate_primary_brake_eligibility(scenario)
    assert not eligibility.eligible
    assert eligibility.reason == "no_stable_aligned_follower"


def test_overlap_eligibility_rejects_m5_semantic_version_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evalsim.metrics.m5 as m5

    monkeypatch.setattr(m5, "M5_METRIC_VERSION", "1.0.1")
    scenario = _straight_scenario(follower_offset_m=-1.0)
    with pytest.raises(
        RuntimeError,
        match="frozen M5 overlap definition 1.0.0",
    ):
        evaluate_primary_brake_eligibility(scenario)


def test_plan_snapshot_does_not_alias_mutable_source() -> None:
    scenario = _straight_scenario()
    plan = compile_longitudinal_brake_pulse_plan(scenario, 2.0)
    before = np.array(plan.x, copy=True)
    scenario.ego.x[:] = -999.0
    np.testing.assert_array_equal(plan.x, before)
