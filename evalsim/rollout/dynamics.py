"""Shared NumPy point-mass dynamics with explicit feasibility limits."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from evalsim.contracts import AgentFrame

DYNAMICS_NAME = "point_mass"
DYNAMICS_VERSION = "0.1.0"


def wrap_heading(value: np.ndarray | float) -> np.ndarray | float:
    """Wrap radians to the closed contract range ``[-pi, pi]``."""

    wrapped = (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi
    # Preserve +pi for positive inputs that land exactly on the branch cut.
    wrapped = np.where(
        np.isclose(wrapped, -np.pi, rtol=0.0, atol=1e-15)
        & (np.asarray(value) > 0.0),
        np.pi,
        wrapped,
    )
    if np.ndim(value) == 0:
        return float(wrapped)
    return wrapped


@dataclass(frozen=True, slots=True)
class DynamicsLimits:
    """Engine-owned bounds applied to kinematic policy controls."""

    max_acceleration_mps2: float = 4.0
    max_deceleration_mps2: float = 8.0
    max_speed_mps: float = 60.0
    max_yaw_rate_radps: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "max_acceleration_mps2",
            "max_deceleration_mps2",
            "max_speed_mps",
            "max_yaw_rate_radps",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ClampCounts:
    """How often each feasibility guard changed a proposed transition."""

    acceleration: int = 0
    deceleration: int = 0
    speed: int = 0
    yaw_rate: int = 0
    reverse_prevented: int = 0

    def __add__(self, other: "ClampCounts") -> "ClampCounts":
        if not isinstance(other, ClampCounts):
            return NotImplemented
        return ClampCounts(
            acceleration=self.acceleration + other.acceleration,
            deceleration=self.deceleration + other.deceleration,
            speed=self.speed + other.speed,
            yaw_rate=self.yaw_rate + other.yaw_rate,
            reverse_prevented=self.reverse_prevented + other.reverse_prevented,
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DynamicsResult:
    frame: AgentFrame
    clamp_counts: ClampCounts


def integrate_point_mass(
    frame: AgentFrame,
    longitudinal_acceleration: np.ndarray,
    yaw_rate: np.ndarray,
    dt: float,
    *,
    limits: DynamicsLimits,
    update_mask: np.ndarray | None = None,
) -> DynamicsResult:
    """Advance selected agents using midpoint heading and trapezoidal speed.

    Controls are clamped before integration. Braking can reach zero speed but never
    reverse an agent. Unselected agents are held exactly. Absolute-state policies bypass
    this function in the rollout engine.
    """

    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    acceleration = np.asarray(longitudinal_acceleration, dtype=float)
    yaw = np.asarray(yaw_rate, dtype=float)
    if acceleration.ndim != 1 or yaw.ndim != 1:
        raise ValueError("controls must be 1-D arrays")
    if acceleration.shape != (frame.num_agents,) or yaw.shape != (
        frame.num_agents,
    ):
        raise ValueError("controls must match AgentFrame agent count")
    if not np.all(np.isfinite(acceleration)) or not np.all(np.isfinite(yaw)):
        raise ValueError("controls must be finite")

    if update_mask is None:
        selected = np.asarray(frame.valid, dtype=bool)
    else:
        selected = np.asarray(update_mask, dtype=bool)
        if selected.shape != (frame.num_agents,):
            raise ValueError("update_mask must match AgentFrame agent count")

    x = np.array(frame.x, copy=True)
    y = np.array(frame.y, copy=True)
    heading = np.array(frame.heading, copy=True)
    vx = np.array(frame.vx, copy=True)
    vy = np.array(frame.vy, copy=True)

    acceleration_clamps = 0
    deceleration_clamps = 0
    speed_clamps = 0
    yaw_clamps = 0
    reverse_prevented = 0
    epsilon = 1e-12

    for index in np.flatnonzero(selected):
        raw_acceleration = float(acceleration[index])
        bounded_acceleration = float(
            np.clip(
                raw_acceleration,
                -limits.max_deceleration_mps2,
                limits.max_acceleration_mps2,
            )
        )
        acceleration_clamps += int(
            raw_acceleration > limits.max_acceleration_mps2
        )
        deceleration_clamps += int(
            raw_acceleration < -limits.max_deceleration_mps2
        )

        raw_yaw = float(yaw[index])
        bounded_yaw = float(
            np.clip(
                raw_yaw,
                -limits.max_yaw_rate_radps,
                limits.max_yaw_rate_radps,
            )
        )
        yaw_clamps += int(raw_yaw != bounded_yaw)

        raw_speed = float(np.hypot(vx[index], vy[index]))
        current_speed = min(raw_speed, limits.max_speed_mps)
        speed_clamps += int(raw_speed > limits.max_speed_mps)

        if raw_speed > epsilon:
            motion_heading = float(np.arctan2(vy[index], vx[index]))
        else:
            motion_heading = float(heading[index])

        proposed_speed = current_speed + bounded_acceleration * dt
        next_speed = float(np.clip(proposed_speed, 0.0, limits.max_speed_mps))
        motion_duration = dt
        if proposed_speed < 0.0:
            reverse_prevented += 1
            motion_duration = current_speed / -bounded_acceleration
        if proposed_speed > limits.max_speed_mps:
            speed_clamps += 1

        if bounded_acceleration < 0.0 and proposed_speed < 0.0:
            # Stop inside the interval, then remain stationary. Extending the
            # trapezoid to the full dt would over-travel after reaching zero speed.
            travel = (
                current_speed * motion_duration
                + 0.5 * bounded_acceleration * motion_duration**2
            )
        elif (
            bounded_acceleration > 0.0
            and proposed_speed > limits.max_speed_mps
        ):
            # Reach the cap, then cruise at the cap for the remainder.
            time_to_speed_limit = (
                limits.max_speed_mps - current_speed
            ) / bounded_acceleration
            travel = (
                current_speed * time_to_speed_limit
                + 0.5
                * bounded_acceleration
                * time_to_speed_limit**2
                + limits.max_speed_mps * (dt - time_to_speed_limit)
            )
        else:
            travel = 0.5 * (current_speed + next_speed) * dt

        heading_delta = bounded_yaw * dt
        midpoint_heading = (
            motion_heading + 0.5 * bounded_yaw * motion_duration
        )
        if (
            raw_speed <= epsilon
            and next_speed <= epsilon
            and bounded_yaw == 0.0
        ):
            # Preserve a stationary actor's recorded orientation bit-for-bit.
            next_heading = float(heading[index])
        else:
            next_heading = float(
                wrap_heading(motion_heading + heading_delta)
            )
        x[index] += travel * np.cos(midpoint_heading)
        y[index] += travel * np.sin(midpoint_heading)
        heading[index] = next_heading
        vx[index] = next_speed * np.cos(next_heading)
        vy[index] = next_speed * np.sin(next_heading)

    return DynamicsResult(
        frame=AgentFrame(
            valid=frame.valid,
            x=x,
            y=y,
            heading=heading,
            vx=vx,
            vy=vy,
        ),
        clamp_counts=ClampCounts(
            acceleration=acceleration_clamps,
            deceleration=deceleration_clamps,
            speed=speed_clamps,
            yaw_rate=yaw_clamps,
            reverse_prevented=reverse_prevented,
        ),
    )


__all__ = [
    "DYNAMICS_NAME",
    "DYNAMICS_VERSION",
    "ClampCounts",
    "DynamicsLimits",
    "DynamicsResult",
    "integrate_point_mass",
    "wrap_heading",
]
