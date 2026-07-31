"""M7 metric governance: per-metric cards derived from the spec + detection matrix."""
from __future__ import annotations

from evalsim.metrics.m5 import PositionErrorMetric, m5_metrics
from evalsim.stress.detection import DetectionCell
from evalsim.stress.metric_cards import (
    MetricCard,
    build_metric_card,
    build_metric_cards,
)


def test_card_derives_core_fields_from_spec() -> None:
    card = build_metric_card(PositionErrorMetric())
    assert isinstance(card, MetricCard)
    assert card.metric_name == "position_error_m"
    assert card.direction == "lower"
    assert card.aggregation == "mean"
    assert card.agent_scope == "world"
    assert card.detected_families == ()
    assert card.blind_spot_families == ()


def test_card_partitions_detection_matrix_into_sensitivity_and_blind_spots() -> None:
    name = "waymax_kinematic_infeasibility_rate"
    matrix = (
        DetectionCell(
            defect_family="kinematic_spike", metric_name=name, clean_value=0.0,
            severity_values=(), detected=True, monotone=True,
        ),
        DetectionCell(
            defect_family="frozen_agent", metric_name=name, clean_value=0.0,
            severity_values=(), detected=False, monotone=True,
        ),
        DetectionCell(
            defect_family="teleportation", metric_name=name, clean_value=0.0,
            severity_values=(), detected=False, monotone=True,
        ),
        DetectionCell(
            defect_family="overlap", metric_name="oriented_box_overlap_rate",
            clean_value=0.0, severity_values=(), detected=True, monotone=True,
        ),
    )

    from evalsim.metrics.m5 import WaymaxKinematicInfeasibilityRateMetric

    card = build_metric_card(WaymaxKinematicInfeasibilityRateMetric(), matrix)
    assert card.detected_families == ("kinematic_spike",)
    assert card.blind_spot_families == ("frozen_agent", "teleportation")
    # a cell for a different metric must not leak into this card
    assert "overlap" not in card.detected_families


def test_all_m5_metrics_have_complete_cards() -> None:
    cards = build_metric_cards(m5_metrics())
    assert len(cards) == len(m5_metrics())
    for card in cards:
        assert card.metric_name
        assert card.value_unit
        assert card.direction in {"lower", "higher", "neutral"}
        assert isinstance(card.known_failure_modes, tuple)
