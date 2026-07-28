"""Nonreactive baseline that exactly replays recorded world-agent states."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evalsim.contracts import (
    AgentFrame,
    AgentType,
    PolicyMetadata,
    PolicyObservation,
    PolicyStep,
    Scenario,
    SimulatorPolicy,
)

LOG_REPLAY_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class _ReplayState:
    reference: Scenario


@dataclass(frozen=True, slots=True)
class LogReplayPolicy(SimulatorPolicy):
    """Replay every recorded non-ego state, deliberately ignoring interaction."""

    def initialize(self, scenario: Scenario, seed: int) -> _ReplayState:
        # RolloutEngine supplies a defensive scenario copy dedicated to this run.
        return _ReplayState(reference=scenario)

    def step(
        self,
        state: _ReplayState,
        observation: PolicyObservation,
    ) -> PolicyStep:
        if not isinstance(state, _ReplayState):
            raise TypeError("LogReplayPolicy received incompatible policy state")
        if state.reference.num_agents != observation.frame.num_agents:
            raise ValueError("LogReplayPolicy state does not match observation")

        count = observation.frame.num_agents
        override_mask = np.ones(count, dtype=bool)
        override_mask[observation.ego_index] = False
        return PolicyStep(
            next_state=state,
            longitudinal_acceleration=np.zeros(count),
            yaw_rate=np.zeros(count),
            override=AgentFrame.from_scenario(
                state.reference,
                observation.next_index,
            ),
            override_mask=override_mask,
        )

    def metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name="log_replay",
            version=LOG_REPLAY_VERSION,
            deterministic=True,
            required_features=("recorded_future_states",),
            supported_agent_types=tuple(AgentType),
            params={},
            known_limitations=(
                "Replays the recorded future and cannot react to ego changes.",
                "Not a generative or causal traffic model.",
            ),
        )


__all__ = ["LOG_REPLAY_VERSION", "LogReplayPolicy"]
