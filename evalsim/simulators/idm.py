"""Longitudinal Intelligent Driver Model (IDM) baseline."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from evalsim.contracts import (
    AgentType,
    PolicyMetadata,
    PolicyObservation,
    PolicyStep,
    Scenario,
    SimulatorPolicy,
)

IDM_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class IDMParameters:
    """Canonical IDM parameters plus deterministic 2-D leader gating."""

    desired_speed_mps: float = 13.9
    minimum_gap_m: float = 2.0
    safe_headway_s: float = 1.5
    max_acceleration_mps2: float = 1.5
    comfortable_deceleration_mps2: float = 2.0
    acceleration_exponent: float = 4.0
    lateral_tolerance_m: float = 2.75
    max_heading_difference_rad: float = np.pi / 6.0
    minimum_numerical_gap_m: float = 0.1

    def __post_init__(self) -> None:
        positive = (
            "desired_speed_mps",
            "max_acceleration_mps2",
            "comfortable_deceleration_mps2",
            "acceleration_exponent",
            "minimum_numerical_gap_m",
        )
        non_negative = (
            "minimum_gap_m",
            "safe_headway_s",
            "lateral_tolerance_m",
        )
        for name in (*positive, *non_negative, "max_heading_difference_rad"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in positive:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in non_negative:
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.max_heading_difference_rad < np.pi / 2.0:
            raise ValueError(
                "max_heading_difference_rad must be in [0, pi/2)"
            )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _IDMState:
    agent_count: int


@dataclass(frozen=True, slots=True)
class IDMPolicy(SimulatorPolicy):
    """Apply IDM to vehicles; use constant velocity for nonvehicle agents."""

    params: IDMParameters = IDMParameters()

    def __post_init__(self) -> None:
        if not isinstance(self.params, IDMParameters):
            raise TypeError("IDMPolicy.params must be IDMParameters")

    def initialize(self, scenario: Scenario, seed: int) -> _IDMState:
        # No logged future or mutable run state is retained.
        return _IDMState(agent_count=scenario.num_agents)

    @staticmethod
    def _motion_direction(
        vx: float,
        vy: float,
        heading: float,
    ) -> tuple[np.ndarray, float]:
        speed = float(np.hypot(vx, vy))
        if speed > 1e-12:
            return np.array([vx / speed, vy / speed]), speed
        return np.array([np.cos(heading), np.sin(heading)]), 0.0

    def _nearest_leader(
        self,
        follower_index: int,
        observation: PolicyObservation,
    ) -> tuple[float, float] | None:
        frame = observation.frame
        follower_direction, _ = self._motion_direction(
            float(frame.vx[follower_index]),
            float(frame.vy[follower_index]),
            float(frame.heading[follower_index]),
        )
        follower_position = np.array(
            [frame.x[follower_index], frame.y[follower_index]]
        )
        alignment_threshold = float(
            np.cos(self.params.max_heading_difference_rad)
        )
        candidates: list[tuple[float, int, float]] = []

        for candidate_index in range(frame.num_agents):
            if (
                candidate_index == follower_index
                or not frame.valid[candidate_index]
            ):
                continue

            candidate_direction, candidate_speed = self._motion_direction(
                float(frame.vx[candidate_index]),
                float(frame.vy[candidate_index]),
                float(frame.heading[candidate_index]),
            )
            if (
                float(np.dot(follower_direction, candidate_direction))
                < alignment_threshold
            ):
                continue

            relative = np.array(
                [
                    frame.x[candidate_index] - follower_position[0],
                    frame.y[candidate_index] - follower_position[1],
                ]
            )
            longitudinal = float(np.dot(relative, follower_direction))
            if longitudinal <= 0.0:
                continue
            lateral = abs(
                float(
                    follower_direction[0] * relative[1]
                    - follower_direction[1] * relative[0]
                )
            )
            if lateral > self.params.lateral_tolerance_m:
                continue

            bumper_gap = longitudinal - 0.5 * (
                observation.lengths[follower_index]
                + observation.lengths[candidate_index]
            )
            leader_speed = max(
                0.0,
                float(
                    frame.vx[candidate_index] * follower_direction[0]
                    + frame.vy[candidate_index] * follower_direction[1]
                ),
            )
            candidates.append(
                (
                    float(bumper_gap),
                    int(observation.agent_ids[candidate_index]),
                    leader_speed,
                )
            )

        if not candidates:
            return None
        bumper_gap, _, leader_speed = min(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        return bumper_gap, leader_speed

    def _acceleration(
        self,
        follower_speed: float,
        leader: tuple[float, float] | None,
    ) -> float:
        params = self.params
        # NumPy arithmetic lets us saturate extreme-but-finite contract inputs
        # instead of overflowing before the engine can apply its speed/accel clamps.
        max_term = (
            np.finfo(float).max / 8.0
        ) / max(params.max_acceleration_mps2, 1.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            free_road_term = np.power(
                np.float64(follower_speed)
                / np.float64(params.desired_speed_mps),
                np.float64(params.acceleration_exponent),
            )
        free_road_term = float(
            np.clip(
                np.nan_to_num(
                    free_road_term,
                    nan=max_term,
                    posinf=max_term,
                    neginf=0.0,
                ),
                0.0,
                max_term,
            )
        )
        interaction_term = 0.0
        if leader is not None:
            bumper_gap, leader_speed = leader
            closing_speed = follower_speed - leader_speed
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                dynamic_component = (
                    np.float64(follower_speed)
                    * np.float64(params.safe_headway_s)
                    + np.float64(follower_speed)
                    * np.float64(closing_speed)
                    / (
                        2.0
                        * np.sqrt(
                            np.float64(params.max_acceleration_mps2)
                            * np.float64(
                                params.comfortable_deceleration_mps2
                            )
                        )
                    )
                )
            dynamic_component = float(
                np.nan_to_num(
                    dynamic_component,
                    nan=max_term,
                    posinf=max_term,
                    neginf=-max_term,
                )
            )
            dynamic_gap = min(
                max_term,
                params.minimum_gap_m + max(0.0, dynamic_component),
            )
            safe_gap = max(bumper_gap, params.minimum_numerical_gap_m)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                interaction_term = np.square(
                    np.float64(dynamic_gap) / np.float64(safe_gap)
                )
            interaction_term = float(
                np.clip(
                    np.nan_to_num(
                        interaction_term,
                        nan=max_term,
                        posinf=max_term,
                        neginf=0.0,
                    ),
                    0.0,
                    max_term,
                )
            )
        acceleration = params.max_acceleration_mps2 * (
            1.0 - free_road_term - interaction_term
        )
        if not np.isfinite(acceleration):
            raise RuntimeError("IDM produced non-finite acceleration")
        return float(acceleration)

    def step(
        self,
        state: _IDMState,
        observation: PolicyObservation,
    ) -> PolicyStep:
        if not isinstance(state, _IDMState):
            raise TypeError("IDMPolicy received incompatible policy state")
        if state.agent_count != observation.frame.num_agents:
            raise ValueError("IDMPolicy state does not match observation")

        count = observation.frame.num_agents
        acceleration = np.zeros(count)
        for index in range(count):
            if (
                index == observation.ego_index
                or not observation.frame.valid[index]
                or not observation.next_valid[index]
                or observation.agent_types[index] != AgentType.VEHICLE
            ):
                continue
            speed = float(
                np.hypot(
                    observation.frame.vx[index],
                    observation.frame.vy[index],
                )
            )
            acceleration[index] = self._acceleration(
                speed,
                self._nearest_leader(index, observation),
            )

        return PolicyStep(
            next_state=state,
            longitudinal_acceleration=acceleration,
            yaw_rate=np.zeros(count),
        )

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name="idm",
            version=IDM_VERSION,
            deterministic=True,
            required_features=(
                "current_agent_states",
                "agent_dimensions",
                "agent_types",
                "logged_ego_state",
            ),
            supported_agent_types=(AgentType.VEHICLE,),
            params=self.params.to_dict(),
            known_limitations=(
                "Longitudinal-only: yaw control is zero.",
                "Leader selection uses local geometric gating rather than lane topology.",
                "Can intentionally depart curved lanes and ignore crossing traffic.",
                "Motion is subject to rollout-engine feasibility limits.",
            ),
            fallback_policy="constant_velocity",
        )


__all__ = ["IDM_VERSION", "IDMParameters", "IDMPolicy"]
