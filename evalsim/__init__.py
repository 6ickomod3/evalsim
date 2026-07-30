"""EvalSim — closed-loop simulation-evaluation platform.

The top-level API re-exports the stable data contracts. Implemented deeper layers use
their focused namespaces (for example, ``evalsim.sources``, ``evalsim.simulators``, and
``evalsim.rollout``) so the contract surface remains explicit.
"""
import importlib
from typing import Any


__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentFrame",
    "AgentType",
    "CounterfactualPair",
    "EgoInterventionSpec",
    "EgoTrajectoryPlan",
    "FeasibilityAudit",
    "HistoryOnlyPolicyContext",
    "HistoryOnlyPolicyObservation",
    "HistoryOnlySimulatorPolicy",
    "MapPolyline",
    "MapType",
    "Metric",
    "MetricEligibility",
    "MetricResult",
    "MetricSpec",
    "InterventionEligibility",
    "PairedMetric",
    "PairedMetricResult",
    "PolicyMetadata",
    "PolicyObservation",
    "PolicyStep",
    "PrivilegedPolicyContext",
    "PrivilegedSimulatorPolicy",
    "Rollout",
    "RunManifest",
    "Scenario",
    "SimulatorPolicy",
    "evaluate_paired_metric",
    "rollout_from_parquet",
    "rollout_to_parquet",
    "scenario_from_parquet",
    "scenario_to_parquet",
]

_PUBLIC_CONTRACT_NAMES = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """Load the stable contract API only when a caller requests it."""

    if name not in _PUBLIC_CONTRACT_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    contracts = importlib.import_module(".contracts", __name__)
    value = getattr(contracts, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _PUBLIC_CONTRACT_NAMES)
