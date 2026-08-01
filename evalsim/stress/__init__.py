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
    FrozenAgentDefectV2,
    KinematicSpikeDefect,
    OverlapDefect,
    TeleportationDefect,
    construct_audit_defect_registry,
    default_defect_registry,
)
from .construct_audit import (
    ConstructAuditCase,
    ConstructAuditCell,
    ConstructAuditError,
    ConstructAuditResult,
    construct_audit_cases,
    run_construct_audit,
)
from .detection import (
    DetectionCase,
    DetectionCell,
    blind_spots,
    cell,
    detection_matrix,
)
from .invariance import (
    AgentPermutationProbe,
    InvarianceProbe,
    InvarianceResult,
    RolloutOnlyTranslationProbe,
    RolloutOnlyVelocityImpulseProbe,
    TranslationProbe,
    check_invariance,
    invariance_matrix,
)
from .metric_cards import MetricCard, build_metric_card, build_metric_cards

__all__ = [
    "Defect",
    "DefectManifest",
    "DefectRegistry",
    "DefectRegistryError",
    "DefectSpec",
    "FrozenAgentDefect",
    "FrozenAgentDefectV2",
    "KinematicSpikeDefect",
    "OverlapDefect",
    "TeleportationDefect",
    "construct_audit_defect_registry",
    "default_defect_registry",
    "ConstructAuditCase",
    "ConstructAuditCell",
    "ConstructAuditError",
    "ConstructAuditResult",
    "construct_audit_cases",
    "run_construct_audit",
    "DetectionCase",
    "DetectionCell",
    "blind_spots",
    "cell",
    "detection_matrix",
    "MetricCard",
    "build_metric_card",
    "build_metric_cards",
    "AgentPermutationProbe",
    "InvarianceProbe",
    "InvarianceResult",
    "RolloutOnlyTranslationProbe",
    "RolloutOnlyVelocityImpulseProbe",
    "TranslationProbe",
    "check_invariance",
    "invariance_matrix",
]
