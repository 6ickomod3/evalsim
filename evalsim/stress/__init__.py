"""M7 evaluator red-team: severity-controlled defect generators + detection.

Typed, seeded, severity-0-identity corruptions of simulated rollouts used to stress-test
the M5 evaluators for detection sensitivity, blind spots, and false positives. Datasets
and generated defect corpora stay local per ``AGENTS.md``.
"""
from .defects import (
    Defect,
    DefectManifest,
    DefectRegistry,
    DefectRegistryError,
    DefectSpec,
    FrozenAgentDefect,
    OverlapDefect,
    TeleportationDefect,
    default_defect_registry,
)
from .detection import (
    DetectionCase,
    DetectionCell,
    blind_spots,
    cell,
    detection_matrix,
)
from .metric_cards import MetricCard, build_metric_card, build_metric_cards

__all__ = [
    "Defect",
    "DefectManifest",
    "DefectRegistry",
    "DefectRegistryError",
    "DefectSpec",
    "FrozenAgentDefect",
    "OverlapDefect",
    "TeleportationDefect",
    "default_defect_registry",
    "DetectionCase",
    "DetectionCell",
    "blind_spots",
    "cell",
    "detection_matrix",
    "MetricCard",
    "build_metric_card",
    "build_metric_cards",
]
