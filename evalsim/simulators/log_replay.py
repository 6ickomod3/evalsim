"""Nonreactive baseline that exactly replays recorded world-agent states."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evalsim.contracts import (
    AgentType,
    HistoryOnlyPolicyObservation,
    PolicyMetadata,
    PolicyStep,
    PrivilegedPolicyContext,
    PrivilegedSimulatorPolicy,
)

LOG_REPLAY_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class _ReplayState:
    reference: PrivilegedPolicyContext


@dataclass(frozen=True, slots=True)
class LogReplayPolicy(PrivilegedSimulatorPolicy):
    """Replay every recorded non-ego state, deliberately ignoring interaction."""

    def initialize(
        self,
        context: PrivilegedPolicyContext,
        seed: int,
    ) -> _ReplayState:
        if not isinstance(context, PrivilegedPolicyContext):
            raise TypeError("LogReplayPolicy requires PrivilegedPolicyContext")
        return _ReplayState(reference=context)

    def step(
        self,
        state: _ReplayState,
        observation: HistoryOnlyPolicyObservation,
    ) -> PolicyStep:
        if not isinstance(state, _ReplayState):
            raise TypeError("LogReplayPolicy received incompatible policy state")
        if len(state.reference.agent_ids) != observation.frame.num_agents:
            raise ValueError("LogReplayPolicy state does not match observation")

        count = observation.frame.num_agents
        override_mask = np.ones(count, dtype=bool)
        override_mask[observation.ego_index] = False
        return PolicyStep(
            next_state=state,
            longitudinal_acceleration=np.zeros(count),
            yaw_rate=np.zeros(count),
            override=state.reference.frames[observation.next_index],
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
