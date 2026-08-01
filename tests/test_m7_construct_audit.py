"""Exact oracles for the outcome-aware, data-free M7 construct audit."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from evalsim import AgentType
from evalsim.metrics.m5 import (
    OrientedBoxOverlapRateMetric,
    PositionErrorMetric,
    WaymaxKinematicInfeasibilityRateMetric,
)
from evalsim.metrics.registry import MetricRegistry
from evalsim.stress.construct_audit import (
    ConstructAuditCell,
    ConstructAuditError,
    ConstructAuditResult,
    construct_audit_cases,
    run_construct_audit,
)
from evalsim.stress.defects import construct_audit_defect_registry

DOSES = (0.25, 0.50, 0.75, 1.00)
EXPECTED_RESPONSES = {
    ("frozen_agent", "position_error_m"),
    ("frozen_agent", "waymax_kinematic_infeasibility_rate"),
    ("kinematic_spike", "waymax_kinematic_infeasibility_rate"),
    ("overlap", "oriented_box_overlap_rate"),
    ("overlap", "position_error_m"),
    ("teleportation", "position_error_m"),
}


def _selected_metrics() -> MetricRegistry:
    return MetricRegistry(
        [
            PositionErrorMetric(),
            OrientedBoxOverlapRateMetric(),
            WaymaxKinematicInfeasibilityRateMetric(),
        ]
    )


def test_construct_audit_cases_match_exact_formulas_and_contract() -> None:
    cases = construct_audit_cases()
    assert len(cases) == 3
    for index, case in enumerate(cases):
        scenario = case.scenario
        clean = case.clean_rollout
        timestamps = np.arange(6, dtype=np.float64) * 0.1
        assert scenario.scenario_id == f"m7-construct-{index}"
        assert scenario.timestamps.dtype == np.float64
        assert np.array_equal(scenario.timestamps, timestamps)
        assert scenario.ego_index == 0
        assert scenario.map == []
        assert scenario.metadata == {
            "source": "m7_construct",
            "current_index": 1,
            "case_index": index,
        }
        assert len(scenario.agents) == 5

        ego = scenario.agents[0]
        assert ego.id == 0 and ego.type == AgentType.VEHICLE
        assert np.all(ego.valid)
        assert np.array_equal(ego.x, np.full(6, -100.0 - index))
        for field in ("y", "heading", "vx", "vy"):
            assert np.array_equal(getattr(ego, field), np.zeros(6))
        assert ego.length == 2.0 and ego.width == 2.0

        for world_index, agent in enumerate(scenario.agents[1:]):
            velocity = 2.0 + world_index
            assert agent.id == 10 + world_index
            assert agent.type == AgentType.VEHICLE
            assert np.all(agent.valid)
            assert np.array_equal(agent.x, velocity * timestamps + index)
            assert np.array_equal(agent.y, np.full(6, 50.0 * world_index))
            assert np.array_equal(agent.vx, np.full(6, velocity))
            for field in ("heading", "vy"):
                assert np.array_equal(getattr(agent, field), np.zeros(6))
            assert agent.length == 2.0 and agent.width == 2.0

        assert clean.scenario_id == scenario.scenario_id
        assert clean.sim_name == "m7_construct_clean"
        assert clean.sim_version == "1.0.0"
        assert clean.seed == 0
        assert clean.perturbation is None
        assert clean.metadata == {}
        assert np.array_equal(clean.timestamps, scenario.timestamps)
        assert not np.shares_memory(clean.timestamps, scenario.timestamps)
        for source, copied in zip(scenario.agents, clean.agents, strict=True):
            assert source.id == copied.id
            assert source.type == copied.type
            assert source.length == copied.length
            assert source.width == copied.width
            for field in ("valid", "x", "y", "heading", "vx", "vy"):
                assert np.array_equal(getattr(source, field), getattr(copied, field))
                assert not np.shares_memory(
                    getattr(source, field), getattr(copied, field)
                )


def test_every_clean_and_injected_metric_result_is_complete_16_of_16() -> None:
    metrics = _selected_metrics()
    for case in construct_audit_cases():
        clean_results = metrics.evaluate(case.scenario, case.clean_rollout)
        assert len(clean_results) == 3
        for result in clean_results:
            assert result.valid is True
            assert result.eligible_components == result.total_components == 16

        for defect in construct_audit_defect_registry():
            for dose in DOSES:
                corrupted, manifest = defect.apply(
                    case.scenario, case.clean_rollout, dose, seed=0
                )
                assert manifest.affected_agent_count > 0
                for result in metrics.evaluate(case.scenario, corrupted):
                    assert result.valid is True
                    assert result.eligible_components == result.total_components == 16


def test_forced_overlap_has_an_independent_geometry_oracle_before_metric() -> None:
    metric = OrientedBoxOverlapRateMetric()
    registry = MetricRegistry([metric])
    defect = construct_audit_defect_registry().get("overlap")
    for case in construct_audit_cases():
        clean = registry.evaluate(case.scenario, case.clean_rollout)[0]
        assert clean.valid is True and clean.value == pytest.approx(0.0, abs=1e-12)
        future = int(case.scenario.metadata["current_index"]) + 1
        world_positions = tuple(
            position
            for position in range(len(case.clean_rollout.agents))
            if position != case.scenario.ego_index
        )
        for dose in DOSES:
            corrupted, manifest = defect.apply(
                case.scenario, case.clean_rollout, dose, seed=0
            )
            reference = corrupted.agents[world_positions[0]]
            for ordinal in manifest.affected_agent_ordinals:
                affected = corrupted.agents[world_positions[ordinal]]
                for field in ("x", "y", "heading"):
                    assert np.array_equal(
                        getattr(affected, field)[future:],
                        getattr(reference, field)[future:],
                    )
            injected = registry.evaluate(case.scenario, corrupted)[0]
            assert injected.valid is True
            assert injected.value is not None and injected.value > float(clean.value) + 1e-6


def test_construct_audit_returns_complete_canonical_immutable_matrix() -> None:
    result = run_construct_audit()
    assert isinstance(result, ConstructAuditResult)
    assert len(result.cells) == 12
    expected_order = tuple(
        (family, metric)
        for family in (
            "frozen_agent",
            "kinematic_spike",
            "overlap",
            "teleportation",
        )
        for metric in (
            "oriented_box_overlap_rate",
            "position_error_m",
            "waymax_kinematic_infeasibility_rate",
        )
    )
    assert tuple(
        (cell.defect_family, cell.metric_name) for cell in result.cells
    ) == expected_order
    assert len(set(expected_order)) == 12
    assert all(isinstance(cell, ConstructAuditCell) for cell in result.cells)
    assert all(tuple(dose for dose, _ in cell.dose_means) == DOSES for cell in result.cells)
    assert all(cell.clean_mean == pytest.approx(0.0, abs=1e-12) for cell in result.cells)

    with pytest.raises(FrozenInstanceError):
        result.response_count = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.cells[0].responds = False  # type: ignore[misc]


def test_construct_audit_matches_exact_six_by_six_expected_matrix() -> None:
    result = run_construct_audit()
    responding = {
        (cell.defect_family, cell.metric_name)
        for cell in result.cells
        if cell.responds
    }
    assert responding == EXPECTED_RESPONSES
    assert result.response_count == 6
    assert result.nonresponse_count == 6
    assert result.matrix_matched is True

    for cell in result.cells:
        key = (cell.defect_family, cell.metric_name)
        means = tuple(mean for _, mean in cell.dose_means)
        if key in EXPECTED_RESPONSES:
            assert all(mean > cell.clean_mean + 1e-6 for mean in means)
            assert all(later >= earlier - 1e-6 for earlier, later in zip(means, means[1:]))
            assert max(means) - min(means) > 1e-6
            assert cell.monotone_sensitive is True
        else:
            assert all(abs(mean - cell.clean_mean) <= 1e-6 for mean in means)
            assert cell.monotone_sensitive is False


@pytest.mark.parametrize(
    "curve_kind",
    ("flat", "adverse"),
)
def test_curve_data_cannot_be_mislabeled_or_match_the_matrix(curve_kind: str) -> None:
    result = run_construct_audit()
    index = next(
        index
        for index, cell in enumerate(result.cells)
        if (cell.defect_family, cell.metric_name)
        == ("frozen_agent", "position_error_m")
    )
    original = result.cells[index]
    if curve_kind == "flat":
        dose_means = tuple((dose, original.clean_mean) for dose in DOSES)
        nonresponse_count = 7
    else:
        dose_means = (
            (0.25, original.clean_mean - 1.0),
            (0.50, original.clean_mean + 1.0),
            (0.75, original.clean_mean + 2.0),
            (1.00, original.clean_mean + 3.0),
        )
        nonresponse_count = 6

    with pytest.raises(ConstructAuditError, match="inconsistent with the curve"):
        replace(
            original,
            dose_means=dose_means,
            responds=True,
            monotone_sensitive=True,
        )

    corrected = replace(
        original,
        dose_means=dose_means,
        responds=False,
        monotone_sensitive=False,
    )
    cells = result.cells[:index] + (corrected,) + result.cells[index + 1 :]
    adversarial_result = ConstructAuditResult(
        cells=cells,
        response_count=5,
        nonresponse_count=nonresponse_count,
        matrix_matched=False,
    )
    assert adversarial_result.matrix_matched is False
    with pytest.raises(ConstructAuditError, match="verdict is inconsistent"):
        replace(adversarial_result, matrix_matched=True)


def test_construct_audit_is_pure_and_deterministic() -> None:
    first = run_construct_audit()
    second = run_construct_audit()
    assert first == second

    cases = construct_audit_cases()
    cases[0].scenario.agents[1].x[0] = 999.0
    fresh = construct_audit_cases()
    assert fresh[0].scenario.agents[1].x[0] == 0.0
    assert run_construct_audit() == first
