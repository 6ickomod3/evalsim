"""Seam A (output side): the ``Rollout`` contract.

A ``Rollout`` is what a simulator produces for a scenario: the per-agent simulated
states over the rollout horizon, plus provenance (which simulator, version, seed, and
which counterfactual perturbation, if any) needed for reproducibility and caching.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scenario import Agent


@dataclass
class Rollout:
    """Simulated agent states for one scenario under one simulator + condition."""

    scenario_id: str
    sim_name: str
    sim_version: str
    seed: int
    timestamps: np.ndarray  # float [T]
    agents: list[Agent]
    perturbation: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=float)
        if self.timestamps.ndim != 1:
            raise ValueError("Rollout.timestamps must be 1-D")
        T = self.timestamps.shape[0]
        for a in self.agents:
            if a.num_steps != T:
                raise ValueError(
                    f"Agent {a.id} has {a.num_steps} steps but rollout horizon is {T}"
                )

    @property
    def num_steps(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def num_agents(self) -> int:
        return len(self.agents)
