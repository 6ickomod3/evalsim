"""Seam B: the ``SimulatorPolicy`` interface (draft §7).

Every simulator — log-replay, constant-velocity, IDM, and the intentionally corrupted
variants — implements this interface. The rollout engine and all evaluation logic depend
only on this contract, never on a concrete simulator. A new policy is added by
implementing an adapter, not by modifying the pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .scenario import Scenario
from .types import AgentType


@dataclass
class PolicyMetadata:
    """Self-description of a simulator policy (draft §7)."""

    name: str
    version: str
    deterministic: bool
    required_features: list[str] = field(default_factory=list)
    supported_agent_types: list[AgentType] = field(default_factory=lambda: [AgentType.VEHICLE])
    params: dict = field(default_factory=dict)
    known_limitations: list[str] = field(default_factory=list)


class SimulatorPolicy(ABC):
    """Abstract traffic-agent simulator.

    ``initialize`` sets up per-scenario state; ``step`` advances one timestep given the
    current state and an observation; ``metadata`` describes the policy for provenance.
    """

    @abstractmethod
    def initialize(self, scenario: Scenario, seed: int) -> Any:
        """Return an opaque per-rollout state for ``scenario`` under ``seed``."""

    @abstractmethod
    def step(self, state: Any, observation: Any) -> Any:
        """Advance one timestep; return the next state (and/or actions)."""

    @abstractmethod
    def metadata(self) -> PolicyMetadata:
        """Return this policy's :class:`PolicyMetadata`."""
