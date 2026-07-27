"""EvalSim — closed-loop simulation-evaluation platform.

Public API re-exports the core data contracts; deeper layers (sources, simulators,
rollout engine, metrics, slices, stats, perturbations, stress tests, reporting) are added
in later milestones.
"""
from .contracts import (
    Agent,
    AgentType,
    MapPolyline,
    MapType,
    Metric,
    MetricResult,
    MetricSpec,
    PolicyMetadata,
    Rollout,
    RunManifest,
    Scenario,
    SimulatorPolicy,
    rollout_from_parquet,
    rollout_to_parquet,
    scenario_from_parquet,
    scenario_to_parquet,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentType",
    "MapPolyline",
    "MapType",
    "Metric",
    "MetricResult",
    "MetricSpec",
    "PolicyMetadata",
    "Rollout",
    "RunManifest",
    "Scenario",
    "SimulatorPolicy",
    "rollout_from_parquet",
    "rollout_to_parquet",
    "scenario_from_parquet",
    "scenario_to_parquet",
]
