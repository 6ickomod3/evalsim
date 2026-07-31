"""M7 detection matrix: measure which M5 evaluator detects which injected defect.

For each (defect family x metric) pair this computes the clean-baseline metric value and
a severity curve (mean metric value across cases as injected severity rises), then flags
whether the metric *detects* the family (curve rises above the clean baseline) and whether
the curve is monotone. Cells where a kinematically-relevant metric fails to rise are the
evaluator blind spots the M7 red-team exists to expose. All values are conditional on the
supplied cases and defect generators -- not population or realism claims.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from evalsim.contracts.metric import Metric
from evalsim.contracts.rollout import Rollout
from evalsim.contracts.scenario import Scenario
from evalsim.metrics.registry import MetricRegistry

from .defects import Defect

_DETECT_TOL = 1e-6


@dataclass(frozen=True)
class DetectionCase:
    """One clean (scenario, defect-free rollout) case to inject defects into."""

    scenario: Scenario
    clean_rollout: Rollout


@dataclass(frozen=True)
class DetectionCell:
    """Detection evidence for one (defect family, metric) pair over a severity grid."""

    defect_family: str
    metric_name: str
    clean_value: float
    severity_values: tuple[tuple[float, float], ...]
    detected: bool
    monotone: bool


def _metric_value(metric: Metric, scenario: Scenario, rollout: Rollout) -> float | None:
    result = MetricRegistry([metric]).evaluate(scenario, rollout)[0]
    return float(result.value) if result.valid else None


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else math.nan


def _clean_mean(metric: Metric, cases: Sequence[DetectionCase]) -> float:
    vals = [
        v
        for case in cases
        if (v := _metric_value(metric, case.scenario, case.clean_rollout)) is not None
    ]
    return _mean(vals)


def _severity_mean(
    defect: Defect,
    metric: Metric,
    cases: Sequence[DetectionCase],
    severity: float,
    *,
    seed: int,
) -> float:
    vals: list[float] = []
    for case in cases:
        corrupted, _ = defect.apply(
            case.scenario, case.clean_rollout, severity, seed=seed
        )
        value = _metric_value(metric, case.scenario, corrupted)
        if value is not None:
            vals.append(value)
    return _mean(vals)


def detection_matrix(
    defects: Iterable[Defect],
    metrics: Iterable[Metric],
    cases: Sequence[DetectionCase],
    severities: Sequence[float],
    *,
    seed: int,
    detect_tol: float = _DETECT_TOL,
) -> tuple[DetectionCell, ...]:
    """Compute a detection cell for every (defect family, metric) pair.

    ``detected`` is True when the maximum severity-curve value rises more than
    ``detect_tol`` above the clean baseline; ``monotone`` is True when the curve is
    non-decreasing in severity within ``detect_tol``.
    """

    if not cases:
        raise ValueError("detection_matrix requires at least one case")
    ordered_severities = tuple(float(s) for s in severities)
    metric_list = list(metrics)
    cells: list[DetectionCell] = []
    for defect in defects:
        family = defect.spec.family
        for metric in metric_list:
            clean = _clean_mean(metric, cases)
            curve = tuple(
                (sev, _severity_mean(defect, metric, cases, sev, seed=seed))
                for sev in ordered_severities
            )
            values = [v for _, v in curve if not math.isnan(v)]
            peak = max(values) if values else math.nan
            detected = bool(
                values
                and not math.isnan(clean)
                and (peak - clean) > detect_tol
            )
            monotone = all(
                curve[i][1] <= curve[i + 1][1] + detect_tol
                for i in range(len(curve) - 1)
                if not (math.isnan(curve[i][1]) or math.isnan(curve[i + 1][1]))
            )
            cells.append(
                DetectionCell(
                    defect_family=family,
                    metric_name=metric.spec.name,
                    clean_value=clean,
                    severity_values=curve,
                    detected=detected,
                    monotone=monotone,
                )
            )
    return tuple(cells)


def cell(
    matrix: Iterable[DetectionCell], defect_family: str, metric_name: str
) -> DetectionCell:
    """Return the single matrix cell for the given (defect family, metric)."""
    for item in matrix:
        if item.defect_family == defect_family and item.metric_name == metric_name:
            return item
    raise KeyError(f"no detection cell for ({defect_family}, {metric_name})")


def blind_spots(matrix: Iterable[DetectionCell]) -> tuple[DetectionCell, ...]:
    """Cells where the metric did not detect the defect (candidate blind spots)."""
    return tuple(item for item in matrix if not item.detected)


__all__ = [
    "DetectionCase",
    "DetectionCell",
    "blind_spots",
    "cell",
    "detection_matrix",
]
