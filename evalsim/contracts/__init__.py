"""EvalSim data contracts — the frozen seams every layer depends on."""
from .types import AgentType, MapType
from .scenario import Agent, MapPolyline, Scenario
from .rollout import Rollout
from .simulator import (
    AgentFrame,
    HistoryOnlyPolicyContext,
    HistoryOnlyPolicyObservation,
    HistoryOnlySimulatorPolicy,
    PolicyMetadata,
    PolicyMapFeature,
    PolicyObservation,
    PolicyStep,
    PrivilegedPolicyContext,
    PrivilegedSimulatorPolicy,
    SimulatorPolicy,
)
from .metric import Metric, MetricEligibility, MetricResult, MetricSpec
from .counterfactual import (
    CounterfactualPair,
    EgoInterventionSpec,
    EgoTrajectoryPlan,
    FeasibilityAudit,
    InterventionEligibility,
    PairedMetric,
    PairedMetricResult,
    evaluate_paired_metric,
)
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
    "AgentFrame",
    "PolicyMapFeature",
    "HistoryOnlyPolicyContext",
    "PrivilegedPolicyContext",
    "HistoryOnlyPolicyObservation",
    "PolicyMetadata",
    "PolicyObservation",
    "PolicyStep",
    "SimulatorPolicy",
    "HistoryOnlySimulatorPolicy",
    "PrivilegedSimulatorPolicy",
    "Metric",
    "MetricEligibility",
    "MetricResult",
    "MetricSpec",
    "CounterfactualPair",
    "EgoInterventionSpec",
    "EgoTrajectoryPlan",
    "FeasibilityAudit",
    "InterventionEligibility",
    "PairedMetric",
    "PairedMetricResult",
    "evaluate_paired_metric",
    "RunManifest",
    "scenario_to_parquet",
    "scenario_from_parquet",
    "rollout_to_parquet",
    "rollout_from_parquet",
]
