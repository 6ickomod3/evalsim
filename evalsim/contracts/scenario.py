"""Seam A: the ``Scenario`` contract.

A ``Scenario`` is the substrate-agnostic representation of a driving scene. Both the
synthetic generator (Mac) and the Waymax/WOMD loader (Cloud) produce this exact type,
and everything downstream consumes only this contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import AgentType, MapType

# Per-agent time-series fields, all shaped [T].
AGENT_SERIES_FIELDS = ("valid", "x", "y", "heading", "vx", "vy")


@dataclass
class Agent:
    """A single traffic agent's trajectory over ``T`` timesteps.

    All series fields (``valid``, ``x``, ``y``, ``heading``, ``vx``, ``vy``) are 1-D
    arrays of length ``T``. ``valid[t]`` marks whether the agent's state at step ``t`` is
    observed/meaningful.
    """

    id: int
    type: AgentType
    valid: np.ndarray  # bool  [T]
    x: np.ndarray      # float [T]
    y: np.ndarray      # float [T]
    heading: np.ndarray  # float [T], radians
    vx: np.ndarray     # float [T]
    vy: np.ndarray     # float [T]
    length: float = 4.5
    width: float = 2.0

    def __post_init__(self) -> None:
        self.type = AgentType(self.type)
        self.valid = np.asarray(self.valid, dtype=bool)
        for name in ("x", "y", "heading", "vx", "vy"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        lengths = {getattr(self, n).shape for n in AGENT_SERIES_FIELDS}
        if len(lengths) != 1:
            raise ValueError(
                f"Agent {self.id}: series fields must share one shape, got {lengths}"
            )
        if getattr(self, "valid").ndim != 1:
            raise ValueError(f"Agent {self.id}: series fields must be 1-D")

    @property
    def num_steps(self) -> int:
        return int(self.valid.shape[0])

    def speed(self) -> np.ndarray:
        """Scalar speed per step, [T]."""
        return np.hypot(self.vx, self.vy)


@dataclass
class MapPolyline:
    """A polyline map feature (lane centerline, road edge, crosswalk, ...)."""

    type: MapType
    xy: np.ndarray  # float [P, 2]

    def __post_init__(self) -> None:
        self.type = MapType(self.type)
        self.xy = np.asarray(self.xy, dtype=float)
        if self.xy.ndim != 2 or self.xy.shape[1] != 2:
            raise ValueError(f"MapPolyline.xy must be [P, 2], got {self.xy.shape}")


@dataclass
class Scenario:
    """A complete driving scenario: agents, map, ego reference, and metadata."""

    scenario_id: str
    timestamps: np.ndarray  # float [T], seconds
    agents: list[Agent]
    map: list[MapPolyline] = field(default_factory=list)
    ego_index: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=float)
        if self.timestamps.ndim != 1:
            raise ValueError("Scenario.timestamps must be 1-D")
        T = self.timestamps.shape[0]
        for a in self.agents:
            if a.num_steps != T:
                raise ValueError(
                    f"Agent {a.id} has {a.num_steps} steps but scenario has {T}"
                )
        if self.agents and not (0 <= self.ego_index < len(self.agents)):
            raise ValueError(
                f"ego_index {self.ego_index} out of range for {len(self.agents)} agents"
            )
        self.metadata.setdefault("source", "unknown")

    @property
    def num_steps(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def ego(self) -> Agent:
        return self.agents[self.ego_index]
