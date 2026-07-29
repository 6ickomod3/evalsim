"""EvalSim — closed-loop simulation-evaluation platform.

The top-level API re-exports the stable data contracts. Implemented deeper layers use
their focused namespaces (for example, ``evalsim.sources``, ``evalsim.simulators``, and
``evalsim.rollout``) so the contract surface remains explicit.
"""
from .contracts import (
    Agent,
    AgentFrame,
    AgentType,
    MapPolyline,
    MapType,
    Metric,
    MetricEligibility,
    MetricResult,
    MetricSpec,
    PolicyMetadata,
    PolicyObservation,
    PolicyStep,
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
    "AgentFrame",
    "AgentType",
    "MapPolyline",
    "MapType",
    "Metric",
    "MetricEligibility",
    "MetricResult",
    "MetricSpec",
    "PolicyMetadata",
    "PolicyObservation",
    "PolicyStep",
    "Rollout",
    "RunManifest",
    "Scenario",
    "SimulatorPolicy",
    "rollout_from_parquet",
    "rollout_to_parquet",
    "scenario_from_parquet",
    "scenario_to_parquet",
]
