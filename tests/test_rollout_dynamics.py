"""Independent oracles for M2 point-mass integration and feasibility clamps."""
from __future__ import annotations

import numpy as np
import pytest

from evalsim import AgentFrame
from evalsim.rollout import DynamicsLimits, integrate_point_mass


def _frame(
    *,
    x: float = 0.0,
    y: float = 0.0,
    heading: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    valid: bool = True,
) -> AgentFrame:
    return AgentFrame(
        valid=np.array([valid]),
        x=np.array([x]),
        y=np.array([y]),
        heading=np.array([heading]),
        vx=np.array([vx]),
        vy=np.array([vy]),
    )


def test_zero_control_is_exact_constant_world_velocity() -> None:
    frame = _frame(heading=np.arctan2(4.0, 3.0), vx=3.0, vy=4.0)
    result = integrate_point_mass(
        frame,
        np.array([0.0]),
        np.array([0.0]),
        0.4,
        limits=DynamicsLimits(),
    )

    assert result.frame.x[0] == pytest.approx(1.2, abs=1e-12)
    assert result.frame.y[0] == pytest.approx(1.6, abs=1e-12)
    assert result.frame.vx[0] == pytest.approx(3.0, abs=1e-12)
    assert result.frame.vy[0] == pytest.approx(4.0, abs=1e-12)
    assert result.clamp_counts.to_dict() == {
        "acceleration": 0,
        "deceleration": 0,
        "speed": 0,
        "yaw_rate": 0,
        "reverse_prevented": 0,
    }


def test_stationary_zero_control_preserves_position_and_heading() -> None:
    frame = _frame(x=4.0, y=-2.0, heading=1.2)
    result = integrate_point_mass(
        frame,
        np.zeros(1),
        np.zeros(1),
        3.0,
        limits=DynamicsLimits(),
    )
    assert result.frame.x[0] == 4.0
    assert result.frame.y[0] == -2.0
    assert result.frame.heading[0] == pytest.approx(1.2)
    assert result.frame.speed()[0] == 0.0


def test_constant_acceleration_matches_analytic_point_mass_solution() -> None:
    frame = _frame(vx=5.0)
    dt = 0.25
    acceleration = 2.0
    result = integrate_point_mass(
        frame,
        np.array([acceleration]),
        np.zeros(1),
        dt,
        limits=DynamicsLimits(),
    )

    expected_x = 5.0 * dt + 0.5 * acceleration * dt**2
    expected_vx = 5.0 + acceleration * dt
    assert result.frame.x[0] == pytest.approx(expected_x, abs=1e-12)
    assert result.frame.vx[0] == pytest.approx(expected_vx, abs=1e-12)
    assert result.frame.y[0] == pytest.approx(0.0, abs=1e-12)


def test_yaw_uses_midpoint_heading_and_wraps_output() -> None:
    frame = _frame(heading=np.pi - 0.1, vx=-2.0)
    yaw_rate = 0.5
    dt = 0.5
    result = integrate_point_mass(
        frame,
        np.zeros(1),
        np.array([yaw_rate]),
        dt,
        limits=DynamicsLimits(),
    )

    motion_heading = np.pi
    expected_midpoint = motion_heading + 0.5 * yaw_rate * dt
    assert result.frame.x[0] == pytest.approx(
        2.0 * dt * np.cos(expected_midpoint),
        abs=1e-12,
    )
    assert result.frame.y[0] == pytest.approx(
        2.0 * dt * np.sin(expected_midpoint),
        abs=1e-12,
    )
    assert -np.pi <= result.frame.heading[0] <= np.pi
    assert result.frame.heading[0] == pytest.approx(-np.pi + 0.25)


def test_feasibility_clamps_are_applied_and_audited() -> None:
    limits = DynamicsLimits(
        max_acceleration_mps2=2.0,
        max_deceleration_mps2=3.0,
        max_speed_mps=1.0,
        max_yaw_rate_radps=0.2,
    )
    result = integrate_point_mass(
        _frame(),
        np.array([100.0]),
        np.array([4.0]),
        1.0,
        limits=limits,
    )

    assert result.frame.speed()[0] == pytest.approx(1.0)
    # Accelerate for 0.5 s to the 1 m/s cap, then cruise for 0.5 s.
    assert np.hypot(result.frame.x[0], result.frame.y[0]) == pytest.approx(
        0.75
    )
    assert result.frame.heading[0] == pytest.approx(0.2)
    assert result.clamp_counts.acceleration == 1
    assert result.clamp_counts.speed == 1
    assert result.clamp_counts.yaw_rate == 1


def test_braking_cannot_reverse_an_agent() -> None:
    limits = DynamicsLimits(max_deceleration_mps2=4.0)
    result = integrate_point_mass(
        _frame(vx=1.0),
        np.array([-100.0]),
        np.zeros(1),
        1.0,
        limits=limits,
    )

    assert result.frame.speed()[0] == 0.0
    # v=1 m/s under the bounded -4 m/s² command stops after 0.25 s.
    assert result.frame.x[0] == pytest.approx(0.125)
    assert result.clamp_counts.deceleration == 1
    assert result.clamp_counts.reverse_prevented == 1


def test_unselected_agent_is_held_without_false_clamps() -> None:
    result = integrate_point_mass(
        _frame(x=3.0, vx=100.0),
        np.array([1e9]),
        np.array([1e9]),
        0.1,
        limits=DynamicsLimits(),
        update_mask=np.array([False]),
    )
    assert result.frame.x[0] == 3.0
    assert result.frame.vx[0] == 100.0
    assert sum(result.clamp_counts.to_dict().values()) == 0


@pytest.mark.parametrize(
    ("acceleration", "yaw_rate", "dt"),
    [
        (np.array([np.nan]), np.zeros(1), 0.1),
        (np.zeros(1), np.array([np.inf]), 0.1),
        (np.zeros(1), np.zeros(1), 0.0),
    ],
)
def test_nonfinite_or_nonpositive_dynamics_inputs_reject(
    acceleration: np.ndarray,
    yaw_rate: np.ndarray,
    dt: float,
) -> None:
    with pytest.raises(ValueError):
        integrate_point_mass(
            _frame(),
            acceleration,
            yaw_rate,
            dt,
            limits=DynamicsLimits(),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_acceleration_mps2": 0.0},
        {"max_deceleration_mps2": -1.0},
        {"max_speed_mps": np.inf},
        {"max_yaw_rate_radps": np.nan},
    ],
)
def test_invalid_dynamics_limits_reject(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        DynamicsLimits(**kwargs)
