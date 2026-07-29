"""Validated contracts for per-scenario metrics.

Metrics expose their semantics through an immutable :class:`MetricSpec`, classify
source-only eligibility explicitly, and retain every eligible component in a
:class:`MetricResult`.  Cross-scenario aggregation intentionally does not belong to
this interface; the statistics layer owns that operation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType
from typing import Any

from .rollout import Rollout
from .scenario import Scenario

_METRIC_NAME = re.compile(r"[a-z][a-z0-9_]*")
_REASON_CODE = re.compile(r"[a-z][a-z0-9_]*")
_SEMANTIC_VERSION = re.compile(
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
_UNITS_OF_ANALYSIS = frozenset({"frame", "agent", "scenario"})
_DIRECTIONS = frozenset({"higher", "lower", "neutral"})
_AGGREGATIONS = frozenset(
    {"mean", "median", "rate", "minimum", "maximum", "sum"}
)
_AGENT_SCOPES = frozenset({"world", "ego", "all"})


def _nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _metric_name(value: Any, *, name: str = "name") -> str:
    value = _nonempty_text(value, name=name)
    if _METRIC_NAME.fullmatch(value) is None:
        raise ValueError(
            f"{name} must use lowercase snake_case and start with a letter"
        )
    return value


def _semantic_version(value: Any) -> str:
    value = _nonempty_text(value, name="version")
    if _SEMANTIC_VERSION.fullmatch(value) is None:
        raise ValueError("version must be a semantic version")
    return value


def _string_tuple(
    value: Any,
    *,
    name: str,
    pattern: re.Pattern[str] | None = None,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    normalized: list[str] = []
    for item in value:
        item = _nonempty_text(item, name=f"{name} item")
        if pattern is not None and pattern.fullmatch(item) is None:
            raise ValueError(f"{name} contains invalid value {item!r}")
        normalized.append(item)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(normalized)


def _freeze_json(value: Any, *, path: str, active: set[int]) -> Any:
    """Copy and recursively freeze an exact JSON-compatible value."""

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} contains a reference cycle")
        active.add(identity)
        try:
            frozen: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(
                        f"{path} keys must be non-empty strings"
                    )
                frozen[key] = _freeze_json(
                    item,
                    path=f"{path}.{key}",
                    active=active,
                )
            return MappingProxyType(frozen)
        finally:
            active.remove(identity)

    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} contains a reference cycle")
        active.add(identity)
        try:
            return tuple(
                _freeze_json(
                    item,
                    path=f"{path}[{index}]",
                    active=active,
                )
                for index, item in enumerate(value)
            )
        finally:
            active.remove(identity)

    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} floats must be finite")
        return value
    raise ValueError(
        f"{path} must contain only JSON-compatible values, got "
        f"{type(value).__name__}"
    )


def _component_tuple(value: Any) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("distribution must be a sequence of finite numbers")
    components: list[float] = []
    for component in value:
        if isinstance(component, bool):
            raise ValueError("distribution components must be finite numbers")
        try:
            numeric = float(component)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "distribution components must be finite numbers"
            ) from exc
        if not math.isfinite(numeric):
            raise ValueError("distribution components must be finite numbers")
        components.append(numeric)
    return tuple(components)


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Immutable semantic declaration for one active metric version."""

    name: str
    version: str
    value_unit: str = "unitless"
    unit_of_analysis: str = "scenario"
    direction: str = "neutral"
    aggregation: str = "mean"
    agent_scope: str = "all"
    evaluation_window: str = "simulated_future"
    eligibility: str = "valid contract inputs"
    invalid_reason_codes: tuple[str, ...] = ()
    required_fields: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    deterministic: bool = True
    known_failure_modes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _metric_name(self.name))
        object.__setattr__(self, "version", _semantic_version(self.version))
        object.__setattr__(
            self,
            "value_unit",
            _nonempty_text(self.value_unit, name="value_unit"),
        )
        object.__setattr__(
            self,
            "eligibility",
            _nonempty_text(self.eligibility, name="eligibility"),
        )
        object.__setattr__(
            self,
            "evaluation_window",
            _nonempty_text(
                self.evaluation_window,
                name="evaluation_window",
            ),
        )
        for name, allowed in (
            ("unit_of_analysis", _UNITS_OF_ANALYSIS),
            ("direction", _DIRECTIONS),
            ("aggregation", _AGGREGATIONS),
            ("agent_scope", _AGENT_SCOPES),
        ):
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(
                    f"{name} must be one of {sorted(allowed)}, got {value!r}"
                )
        object.__setattr__(
            self,
            "invalid_reason_codes",
            _string_tuple(
                self.invalid_reason_codes,
                name="invalid_reason_codes",
                pattern=_REASON_CODE,
            ),
        )
        object.__setattr__(
            self,
            "required_fields",
            _string_tuple(self.required_fields, name="required_fields"),
        )
        object.__setattr__(
            self,
            "depends_on",
            _string_tuple(
                self.depends_on,
                name="depends_on",
                pattern=_METRIC_NAME,
            ),
        )
        object.__setattr__(
            self,
            "known_failure_modes",
            _string_tuple(
                self.known_failure_modes,
                name="known_failure_modes",
            ),
        )
        if type(self.deterministic) is not bool:
            raise ValueError("deterministic must be a boolean")
        if self.name in self.depends_on:
            raise ValueError("a metric cannot depend on itself")

    @property
    def metric_id(self) -> str:
        return f"{self.name}@{self.version}"


@dataclass(frozen=True, slots=True)
class MetricEligibility:
    """Source-only eligibility decision made before interpreting an outcome."""

    eligible: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.eligible) is not bool:
            raise ValueError("eligible must be a boolean")
        if self.eligible:
            if self.reason is not None:
                raise ValueError(
                    "an eligible metric input cannot have an invalid reason"
                )
            return
        reason = _nonempty_text(self.reason, name="reason")
        if _REASON_CODE.fullmatch(reason) is None:
            raise ValueError(
                "reason must use lowercase snake_case and start with a letter"
            )
        object.__setattr__(self, "reason", reason)

    @classmethod
    def accepted(cls) -> "MetricEligibility":
        return cls(eligible=True)

    @classmethod
    def rejected(cls, reason: str) -> "MetricEligibility":
        return cls(eligible=False, reason=reason)


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Validated per-scenario scalar plus its complete eligible distribution."""

    metric_name: str
    metric_version: str
    scenario_id: str
    value: float | None
    distribution: tuple[float, ...] = ()
    valid: bool = True
    invalid_reason: str | None = None
    eligible_components: int = 1
    total_components: int = 1
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metric_name",
            _metric_name(self.metric_name, name="metric_name"),
        )
        object.__setattr__(
            self,
            "metric_version",
            _semantic_version(self.metric_version),
        )
        object.__setattr__(
            self,
            "scenario_id",
            _nonempty_text(self.scenario_id, name="scenario_id"),
        )
        if type(self.valid) is not bool:
            raise ValueError("valid must be a boolean")

        eligible = _nonnegative_integer(
            self.eligible_components,
            name="eligible_components",
        )
        total = _nonnegative_integer(
            self.total_components,
            name="total_components",
        )
        if eligible > total:
            raise ValueError(
                "eligible_components cannot exceed total_components"
            )
        distribution = _component_tuple(self.distribution)
        if len(distribution) != eligible:
            raise ValueError(
                "distribution length must equal eligible_components"
            )
        object.__setattr__(self, "eligible_components", eligible)
        object.__setattr__(self, "total_components", total)
        object.__setattr__(self, "distribution", distribution)

        if self.valid:
            if self.value is None or isinstance(self.value, bool):
                raise ValueError(
                    "a valid metric result requires a finite scalar value"
                )
            try:
                value = float(self.value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "a valid metric result requires a finite scalar value"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    "a valid metric result requires a finite scalar value"
                )
            if self.invalid_reason is not None:
                raise ValueError(
                    "a valid metric result cannot have an invalid reason"
                )
            if eligible < 1:
                raise ValueError(
                    "a valid metric result requires an eligible component"
                )
            object.__setattr__(self, "value", value)
        else:
            if self.value is not None:
                raise ValueError("an invalid metric result must have value None")
            reason = _nonempty_text(
                self.invalid_reason,
                name="invalid_reason",
            )
            if _REASON_CODE.fullmatch(reason) is None:
                raise ValueError(
                    "invalid_reason must use lowercase snake_case"
                )
            if eligible != 0 or distribution:
                raise ValueError(
                    "an invalid metric result cannot retain eligible components"
                )
            object.__setattr__(self, "invalid_reason", reason)

        if not isinstance(self.details, Mapping):
            raise ValueError("details must be a mapping")
        object.__setattr__(
            self,
            "details",
            _freeze_json(self.details, path="details", active=set()),
        )


class Metric(ABC):
    """Per-scenario metric interface with no cross-scenario aggregation seam."""

    spec: MetricSpec

    @abstractmethod
    def eligibility(
        self,
        scenario: Scenario,
    ) -> MetricEligibility:
        """Return eligibility from source scenario fields only."""

    @abstractmethod
    def compute(self, scenario: Scenario, rollout: Rollout) -> MetricResult:
        """Compute one per-scenario result consistent with :meth:`eligibility`."""


__all__ = [
    "Metric",
    "MetricEligibility",
    "MetricResult",
    "MetricSpec",
]
