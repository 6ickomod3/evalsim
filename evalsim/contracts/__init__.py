"""EvalSim data contracts — the frozen seams every layer depends on."""
from .types import AgentType, MapType
from .scenario import Agent, MapPolyline, Scenario
from .rollout import Rollout
from .simulator import PolicyMetadata, SimulatorPolicy
from .metric import Metric, MetricResult, MetricSpec
from .manifest import RunManifest
from .serialization import (
    scenario_to_parquet,
    scenario_from_parquet,
    rollout_to_parquet,
    rollout_from_parquet,
)

__all__ = [
    "AgentType",
    "MapType",
    "Agent",
    "MapPolyline",
    "Scenario",
    "Rollout",
    "PolicyMetadata",
    "SimulatorPolicy",
    "Metric",
    "MetricResult",
    "MetricSpec",
    "RunManifest",
    "scenario_to_parquet",
    "scenario_from_parquet",
    "rollout_to_parquet",
    "rollout_from_parquet",
]
