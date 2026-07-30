"""World-frame constant-velocity baseline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evalsim.contracts import (
    AgentType,
    HistoryOnlyPolicyContext,
    HistoryOnlyPolicyObservation,
    HistoryOnlySimulatorPolicy,
    PolicyMetadata,
    PolicyStep,
)

CONSTANT_VELOCITY_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class _ConstantVelocityState:
    agent_count: int


@dataclass(frozen=True, slots=True)
class ConstantVelocityPolicy(HistoryOnlySimulatorPolicy):
    """Extrapolate each world agent's active-segment velocity without interaction."""

    def initialize(
        self,
        context: HistoryOnlyPolicyContext,
        seed: int,
    ) -> _ConstantVelocityState:
        # Intentionally retain no trajectory data: future logged world states cannot
        # influence constant-velocity controls.
        if not isinstance(context, HistoryOnlyPolicyContext):
            raise TypeError(
                "ConstantVelocityPolicy requires HistoryOnlyPolicyContext"
            )
        return _ConstantVelocityState(agent_count=len(context.agent_ids))

    def step(
        self,
        state: _ConstantVelocityState,
        observation: HistoryOnlyPolicyObservation,
    ) -> PolicyStep:
        if not isinstance(state, _ConstantVelocityState):
            raise TypeError(
                "ConstantVelocityPolicy received incompatible policy state"
            )
        if state.agent_count != observation.frame.num_agents:
            raise ValueError(
                "ConstantVelocityPolicy state does not match observation"
            )
        count = observation.frame.num_agents
        return PolicyStep(
            next_state=state,
            longitudinal_acceleration=np.zeros(count),
            yaw_rate=np.zeros(count),
        )

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name="constant_velocity",
            version=CONSTANT_VELOCITY_VERSION,
            deterministic=True,
            required_features=("current_position", "current_velocity"),
            supported_agent_types=tuple(AgentType),
            params={},
            known_limitations=(
                "Does not react to other agents.",
                "Does not follow curved lanes or map geometry.",
                "Motion is subject to rollout-engine feasibility limits.",
            ),
        )


__all__ = ["CONSTANT_VELOCITY_VERSION", "ConstantVelocityPolicy"]
