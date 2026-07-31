"""M7 metric governance cards.

A metric card is the release-candidate governance artifact for one evaluator: its
intended use and statistical unit (from the frozen :class:`MetricSpec`), its known
failure modes, and -- when a detection matrix is supplied -- the defect families it was
empirically shown to detect versus the families it is blind to. Cards make evaluator
blind spots an explicit, reviewable property rather than an accident.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from evalsim.contracts.metric import Metric

from .detection import DetectionCell


@dataclass(frozen=True)
class MetricCard:
    """Governance summary for one evaluator, optionally annotated with detection."""

    metric_name: str
    version: str
    value_unit: str
    direction: str
    aggregation: str
    agent_scope: str
    unit_of_analysis: str
    evaluation_window: str
    eligibility: str
    known_failure_modes: tuple[str, ...]
    detected_families: tuple[str, ...]
    blind_spot_families: tuple[str, ...]


def build_metric_card(
    metric: Metric, matrix: Iterable[DetectionCell] = ()
) -> MetricCard:
    """Build a metric card from the spec, partitioning any matching detection cells."""
    spec = metric.spec
    detected: set[str] = set()
    blind: set[str] = set()
    for item in matrix:
        if item.metric_name != spec.name:
            continue
        (detected if item.detected else blind).add(item.defect_family)
    return MetricCard(
        metric_name=spec.name,
        version=spec.version,
        value_unit=spec.value_unit,
        direction=spec.direction,
        aggregation=spec.aggregation,
        agent_scope=spec.agent_scope,
        unit_of_analysis=spec.unit_of_analysis,
        evaluation_window=spec.evaluation_window,
        eligibility=spec.eligibility,
        known_failure_modes=tuple(spec.known_failure_modes),
        detected_families=tuple(sorted(detected)),
        blind_spot_families=tuple(sorted(blind)),
    )


def build_metric_cards(
    metrics: Iterable[Metric], matrix: Iterable[DetectionCell] = ()
) -> tuple[MetricCard, ...]:
    """Build governance cards for every metric (matrix reused across all cards)."""
    cells = tuple(matrix)
    return tuple(build_metric_card(metric, cells) for metric in metrics)


__all__ = ["MetricCard", "build_metric_card", "build_metric_cards"]
