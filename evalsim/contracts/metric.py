"""Metric contract (draft §8).

Metrics are registered through a common contract so the platform never collapses to a
single composite score: component values, distributions, and their aggregation stay
visible. Each metric declares its unit of analysis, direction, and input requirements.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .scenario import Scenario
from .rollout import Rollout


@dataclass
class MetricSpec:
    """Declarative description of a metric."""

    name: str
    version: str
    # "frame" | "agent" | "scenario"
    unit_of_analysis: str = "scenario"
    higher_is_better: bool = True
    # How per-scenario values combine into a summary: "mean" | "median" | "rate" | ...
    aggregation: str = "mean"
    required_fields: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    deterministic: bool = True
    expected_failure_modes: list[str] = field(default_factory=list)


@dataclass
class MetricResult:
    """Per-scenario metric output. ``value`` is the headline scalar; ``distribution``
    optionally carries the underlying samples (metrics should be read as distributions,
    not just means)."""

    metric_name: str
    metric_version: str
    scenario_id: str
    value: float
    distribution: list[float] | None = None
    valid: bool = True
    details: dict = field(default_factory=dict)


class Metric(ABC):
    """Abstract metric. Concrete metrics set ``spec`` and implement the three methods."""

    spec: MetricSpec

    @abstractmethod
    def validate_inputs(self, scenario: Scenario, rollout: Rollout) -> bool:
        """Return whether this metric can be computed for the given inputs."""

    @abstractmethod
    def compute(self, scenario: Scenario, rollout: Rollout) -> MetricResult:
        """Compute the per-scenario metric result."""

    @abstractmethod
    def aggregate(self, per_scenario_values: list[float]) -> dict[str, Any]:
        """Aggregate per-scenario values into a summary (e.g. mean/median + count)."""
