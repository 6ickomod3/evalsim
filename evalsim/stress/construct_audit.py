"""Outcome-aware M7 construct audit over three transparent analytic cases.

This module is deliberately small and data-free. It verifies the intended field of
view of three selected M5 evaluators against four controlled corruptions; it is not a
calibrated, held-out, population-level, or real-scene metric validation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from evalsim.contracts import Agent, AgentType, Rollout, Scenario
from evalsim.metrics.m5 import (
    OrientedBoxOverlapRateMetric,
    PositionErrorMetric,
    WaymaxKinematicInfeasibilityRateMetric,
)
from evalsim.metrics.registry import MetricRegistry

from .defects import Defect, DefectManifest, construct_audit_defect_registry

_DOSES = (0.25, 0.50, 0.75, 1.00)
_TOLERANCE = 1e-6
_SEED = 0
_CASE_COUNT = 3
_COMPONENT_COUNT = 16

_DEFECT_VERSIONS = (
    ("frozen_agent", "v2"),
    ("kinematic_spike", "v1"),
    ("overlap", "v1"),
    ("teleportation", "v1"),
)
_METRIC_VERSIONS = (
    ("oriented_box_overlap_rate", "1.0.0"),
    ("position_error_m", "1.0.0"),
    ("waymax_kinematic_infeasibility_rate", "1.0.1"),
)
_EXPECTED_CELL_KEYS = tuple(
    (family, metric_name)
    for family, _ in _DEFECT_VERSIONS
    for metric_name, _ in _METRIC_VERSIONS
)
_EXPECTED_RESPONSES = frozenset(
    {
        ("frozen_agent", "position_error_m"),
        ("frozen_agent", "waymax_kinematic_infeasibility_rate"),
        ("kinematic_spike", "waymax_kinematic_infeasibility_rate"),
        ("overlap", "oriented_box_overlap_rate"),
        ("overlap", "position_error_m"),
        ("teleportation", "position_error_m"),
    }
)


class ConstructAuditError(RuntimeError):
    """A frozen construct-audit identity, completeness, or outcome gate failed."""


def _classify_curve(
    clean_mean: float,
    dose_means: tuple[tuple[float, float], ...],
) -> tuple[bool, bool]:
    curve = tuple(mean for _, mean in dose_means)
    deltas = tuple(mean - clean_mean for mean in curve)
    no_adverse_delta = all(delta >= -_TOLERANCE for delta in deltas)
    responds = no_adverse_delta and all(delta > _TOLERANCE for delta in deltas)
    monotone_sensitive = responds and all(
        later >= earlier - _TOLERANCE
        for earlier, later in zip(curve, curve[1:])
    ) and (max(curve) - min(curve) > _TOLERANCE)
    return responds, monotone_sensitive


@dataclass(frozen=True, slots=True)
class ConstructAuditCase:
    """One exact analytic scenario and its defensive clean log-copy rollout."""

    scenario: Scenario
    clean_rollout: Rollout


@dataclass(frozen=True, slots=True)
class ConstructAuditCell:
    """Immutable four-dose evidence for one defect-family/metric pair."""

    defect_family: str
    defect_version: str
    metric_name: str
    metric_version: str
    clean_mean: float
    dose_means: tuple[tuple[float, float], ...]
    responds: bool
    monotone_sensitive: bool

    def __post_init__(self) -> None:
        if (self.defect_family, self.defect_version) not in _DEFECT_VERSIONS:
            raise ValueError("unexpected construct-audit defect identity")
        if (self.metric_name, self.metric_version) not in _METRIC_VERSIONS:
            raise ValueError("unexpected construct-audit metric identity")
        clean_mean = float(self.clean_mean)
        if not math.isfinite(clean_mean):
            raise ValueError("clean_mean must be finite")
        object.__setattr__(self, "clean_mean", clean_mean)

        normalized: list[tuple[float, float]] = []
        for item in self.dose_means:
            if len(item) != 2:
                raise ValueError("each dose mean must be a (dose, mean) pair")
            dose, mean = float(item[0]), float(item[1])
            if not (math.isfinite(dose) and math.isfinite(mean)):
                raise ValueError("dose means must be finite")
            normalized.append((dose, mean))
        dose_means = tuple(normalized)
        if tuple(dose for dose, _ in dose_means) != _DOSES:
            raise ValueError("construct-audit doses do not match the frozen grid")
        object.__setattr__(self, "dose_means", dose_means)
        if type(self.responds) is not bool or type(self.monotone_sensitive) is not bool:
            raise ValueError("construct-audit classifications must be booleans")
        expected = _classify_curve(clean_mean, dose_means)
        if (self.responds, self.monotone_sensitive) != expected:
            raise ConstructAuditError(
                "construct-audit classifications are inconsistent with the curve"
            )


@dataclass(frozen=True, slots=True)
class ConstructAuditResult:
    """Immutable complete twelve-cell result and frozen-matrix verdict."""

    cells: tuple[ConstructAuditCell, ...]
    response_count: int
    nonresponse_count: int
    matrix_matched: bool

    def __post_init__(self) -> None:
        cells = tuple(self.cells)
        object.__setattr__(self, "cells", cells)
        keys = tuple((cell.defect_family, cell.metric_name) for cell in cells)
        if keys != _EXPECTED_CELL_KEYS:
            raise ConstructAuditError(
                "construct audit must contain exactly twelve canonically ordered cells"
            )
        if len(set(keys)) != len(keys):
            raise ConstructAuditError("construct audit contains duplicate cells")
        responses = sum(cell.responds for cell in cells)
        nonresponses = sum(_is_exact_nonresponse(cell) for cell in cells)
        if self.response_count != responses or self.nonresponse_count != nonresponses:
            raise ConstructAuditError("construct-audit response counts are inconsistent")
        matched = _matches_expected_matrix(cells)
        if type(self.matrix_matched) is not bool or self.matrix_matched != matched:
            raise ConstructAuditError("construct-audit matrix verdict is inconsistent")


def _copy_agent(agent: Agent) -> Agent:
    return Agent(
        id=agent.id,
        type=agent.type,
        valid=np.array(agent.valid, copy=True),
        x=np.array(agent.x, copy=True),
        y=np.array(agent.y, copy=True),
        heading=np.array(agent.heading, copy=True),
        vx=np.array(agent.vx, copy=True),
        vy=np.array(agent.vy, copy=True),
        length=agent.length,
        width=agent.width,
    )


def _analytic_agent(
    agent_id: int,
    timestamps: np.ndarray,
    *,
    x: np.ndarray,
    y: float,
    vx: float,
) -> Agent:
    count = len(timestamps)
    return Agent(
        id=agent_id,
        type=AgentType.VEHICLE,
        valid=np.ones(count, dtype=bool),
        x=np.array(x, dtype=np.float64, copy=True),
        y=np.full(count, y, dtype=np.float64),
        heading=np.zeros(count, dtype=np.float64),
        vx=np.full(count, vx, dtype=np.float64),
        vy=np.zeros(count, dtype=np.float64),
        length=2.0,
        width=2.0,
    )


def construct_audit_cases() -> tuple[ConstructAuditCase, ...]:
    """Build the exact three contract-only cases frozen by the accepted amendment."""

    cases: list[ConstructAuditCase] = []
    for case_index in range(_CASE_COUNT):
        timestamps = np.arange(6, dtype=np.float64) * 0.1
        agents = [
            _analytic_agent(
                0,
                timestamps,
                x=np.full(6, -100.0 - case_index, dtype=np.float64),
                y=0.0,
                vx=0.0,
            )
        ]
        for world_index in range(4):
            velocity = 2.0 + world_index
            agents.append(
                _analytic_agent(
                    10 + world_index,
                    timestamps,
                    x=velocity * timestamps + case_index,
                    y=50.0 * world_index,
                    vx=velocity,
                )
            )
        scenario = Scenario(
            scenario_id=f"m7-construct-{case_index}",
            timestamps=np.array(timestamps, copy=True),
            agents=agents,
            map=[],
            ego_index=0,
            metadata={
                "source": "m7_construct",
                "current_index": 1,
                "case_index": case_index,
            },
        )
        clean_rollout = Rollout(
            scenario_id=scenario.scenario_id,
            sim_name="m7_construct_clean",
            sim_version="1.0.0",
            seed=0,
            timestamps=np.array(scenario.timestamps, copy=True),
            agents=[_copy_agent(agent) for agent in scenario.agents],
            perturbation=None,
            metadata={},
        )
        cases.append(
            ConstructAuditCase(scenario=scenario, clean_rollout=clean_rollout)
        )
    return tuple(cases)


def _metric_registry() -> MetricRegistry:
    registry = MetricRegistry(
        [
            PositionErrorMetric(),
            OrientedBoxOverlapRateMetric(),
            WaymaxKinematicInfeasibilityRateMetric(),
        ]
    )
    identities = tuple((metric.spec.name, metric.spec.version) for metric in registry)
    if identities != _METRIC_VERSIONS:
        raise ConstructAuditError("selected metric identity/version set changed")
    return registry


def _validated_values(
    registry: MetricRegistry,
    case: ConstructAuditCase,
    rollout: Rollout,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for result in registry.evaluate(case.scenario, rollout):
        if (result.metric_name, result.metric_version) not in _METRIC_VERSIONS:
            raise ConstructAuditError("evaluated metric identity/version changed")
        if (
            not result.valid
            or result.value is None
            or result.eligible_components != _COMPONENT_COUNT
            or result.total_components != _COMPONENT_COUNT
        ):
            raise ConstructAuditError(
                f"{result.metric_name} did not produce a valid 16/16 result "
                f"for {case.scenario.scenario_id}"
            )
        values[result.metric_name] = float(result.value)
    if tuple(sorted(values)) != tuple(name for name, _ in _METRIC_VERSIONS):
        raise ConstructAuditError("metric evaluation was incomplete or duplicated")
    return values


def _expected_ordinals(family: str, dose: float) -> tuple[int, ...]:
    if family == "overlap":
        count = math.ceil(dose * 3)
        return tuple(range(1, count + 1))
    return tuple(range(math.ceil(dose * 4)))


def _validate_manifest(
    defect: Defect,
    manifest: DefectManifest,
    *,
    dose: float,
    case: ConstructAuditCase,
) -> None:
    expected = _expected_ordinals(defect.spec.family, dose)
    if (
        manifest.family != defect.spec.family
        or manifest.version != defect.spec.version
        or manifest.severity != dose
        or manifest.seed != _SEED
        or manifest.total_world_agent_count != 4
        or manifest.affected_agent_ordinals != expected
    ):
        raise ConstructAuditError(
            f"defect manifest drift for {defect.spec.family} on "
            f"{case.scenario.scenario_id} at dose {dose}"
        )
    if not manifest.affected_agent_ordinals:
        raise ConstructAuditError("a positive construct-audit dose affected no agents")


def _validate_forced_overlap(
    case: ConstructAuditCase,
    rollout: Rollout,
    manifest: DefectManifest,
) -> None:
    """Prove the overlap corruption geometrically before any metric is evaluated."""

    world_positions = tuple(
        position
        for position in range(len(rollout.agents))
        if position != case.scenario.ego_index
    )
    reference = rollout.agents[world_positions[0]]
    future = int(case.scenario.metadata["current_index"]) + 1
    for ordinal in manifest.affected_agent_ordinals:
        affected = rollout.agents[world_positions[ordinal]]
        for field in ("x", "y", "heading"):
            if not np.array_equal(
                getattr(affected, field)[future:],
                getattr(reference, field)[future:],
            ):
                raise ConstructAuditError(
                    "forced-overlap fields do not match the reference trajectory"
                )


def _mean(values: list[float]) -> float:
    if len(values) != _CASE_COUNT or not all(math.isfinite(value) for value in values):
        raise ConstructAuditError("construct-audit means require three finite values")
    return math.fsum(values) / _CASE_COUNT


def _is_exact_nonresponse(cell: ConstructAuditCell) -> bool:
    return all(
        abs(mean - cell.clean_mean) <= _TOLERANCE
        for _, mean in cell.dose_means
    )


def _matches_expected_matrix(cells: tuple[ConstructAuditCell, ...]) -> bool:
    for cell in cells:
        key = (cell.defect_family, cell.metric_name)
        if key in _EXPECTED_RESPONSES:
            if not (cell.responds and cell.monotone_sensitive):
                return False
        elif not _is_exact_nonresponse(cell):
            return False
    return True


def run_construct_audit() -> ConstructAuditResult:
    """Run the frozen analytic audit and return all twelve immutable cells.

    Structural drift, missing or partial metric results, identity/version mismatches,
    and incorrect defect manifests raise :class:`ConstructAuditError` instead of being
    converted into apparent non-responses.
    """

    cases = construct_audit_cases()
    metrics = _metric_registry()
    defects = construct_audit_defect_registry()
    defect_identities = tuple(
        (defect.spec.family, defect.spec.version) for defect in defects
    )
    if defect_identities != _DEFECT_VERSIONS:
        raise ConstructAuditError("construct-audit defect identity/version set changed")

    clean_by_metric: dict[str, list[float]] = {
        name: [] for name, _ in _METRIC_VERSIONS
    }
    for case in cases:
        for name, value in _validated_values(
            metrics, case, case.clean_rollout
        ).items():
            clean_by_metric[name].append(value)
    clean_means = {name: _mean(values) for name, values in clean_by_metric.items()}

    cells: list[ConstructAuditCell] = []
    for defect in defects:
        values_by_dose: dict[float, dict[str, list[float]]] = {
            dose: {name: [] for name, _ in _METRIC_VERSIONS} for dose in _DOSES
        }
        previous_ordinals: list[tuple[int, ...]] = [() for _ in cases]
        for dose in _DOSES:
            for case_index, case in enumerate(cases):
                corrupted, manifest = defect.apply(
                    case.scenario,
                    case.clean_rollout,
                    dose,
                    seed=_SEED,
                )
                _validate_manifest(defect, manifest, dose=dose, case=case)
                if defect.spec.family == "overlap":
                    _validate_forced_overlap(case, corrupted, manifest)
                if not set(previous_ordinals[case_index]).issubset(
                    manifest.affected_agent_ordinals
                ):
                    raise ConstructAuditError("affected-agent manifests are not nested")
                previous_ordinals[case_index] = manifest.affected_agent_ordinals
                for name, value in _validated_values(metrics, case, corrupted).items():
                    values_by_dose[dose][name].append(value)

        for metric_name, metric_version in _METRIC_VERSIONS:
            clean_mean = clean_means[metric_name]
            dose_means = tuple(
                (dose, _mean(values_by_dose[dose][metric_name])) for dose in _DOSES
            )
            responds, monotone_sensitive = _classify_curve(clean_mean, dose_means)
            cells.append(
                ConstructAuditCell(
                    defect_family=defect.spec.family,
                    defect_version=defect.spec.version,
                    metric_name=metric_name,
                    metric_version=metric_version,
                    clean_mean=clean_mean,
                    dose_means=dose_means,
                    responds=responds,
                    monotone_sensitive=monotone_sensitive,
                )
            )

    frozen_cells = tuple(cells)
    return ConstructAuditResult(
        cells=frozen_cells,
        response_count=sum(cell.responds for cell in frozen_cells),
        nonresponse_count=sum(_is_exact_nonresponse(cell) for cell in frozen_cells),
        matrix_matched=_matches_expected_matrix(frozen_cells),
    )


__all__ = [
    "ConstructAuditCase",
    "ConstructAuditCell",
    "ConstructAuditError",
    "ConstructAuditResult",
    "construct_audit_cases",
    "run_construct_audit",
]
