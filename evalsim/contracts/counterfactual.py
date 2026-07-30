"""Source-neutral contracts for M6 paired counterfactual evaluation.

The contracts in this module deliberately know nothing about WOMD or Waymax.  They
provide immutable intervention/plan identities, defensive snapshots of the mutable
core :class:`Scenario` and :class:`Rollout` contracts, and the one-scene paired metric
surface used by the M6 evaluation layer.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .metric import MetricSpec
from .rollout import Rollout
from .scenario import Agent, MapPolyline, Scenario
from .types import AgentType, MapType

M6_ANALYSIS_TRANSITIONS = 40
M6_PLAN_FRAME_COUNT = M6_ANALYSIS_TRANSITIONS + 1
M6_INTERVENTION_SCHEMA_VERSION = "1.0.0"
M6_FEASIBILITY_SCHEMA_VERSION = "1.0.0"
M6_ELIGIBILITY_SCHEMA_VERSION = "1.0.0"
M6_PLAN_SCHEMA_VERSION = "1.0.0"
M6_PAIRED_METRIC_RESULT_SCHEMA_VERSION = "1.0.0"
M6_PRIMARY_ELIGIBILITY_REASONS = (
    "insufficient_future_horizon",
    "ego_invalid_in_window",
    "ego_speed_below_5_mps",
    "source_ego_path_degenerate",
    "zero_dose_reconstruction_mismatch",
    "primary_ego_plan_infeasible",
    "no_stable_aligned_follower",
    "current_ego_follower_overlap",
)

CONFIGURATION_DOMAIN = b"evalsim-ego-intervention-config-v1"
PLAN_DOMAIN = b"evalsim-ego-trajectory-plan-v1"
PLAN_ENVELOPE_DOMAIN = b"evalsim-ego-trajectory-plan-envelope-v1"
PLAN_AUDIT_DOMAIN = b"evalsim-ego-trajectory-plan-audit-v1"

_FAMILY = re.compile(r"[a-z][a-z0-9_]*")
_VERSION = re.compile(r"v(?:0|[1-9][0-9]*)")
_SEMANTIC_VERSION = re.compile(
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
_REASON = re.compile(r"[a-z][a-z0-9_]*")
_CHECK = re.compile(r"[a-z][a-z0-9_]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PERTURBATION_IDENTITY = re.compile(
    r"(?P<family>[a-z][a-z0-9_]*)/"
    r"(?P<version>v(?:0|[1-9][0-9]*))"
    r"@sha256:(?P<sha256>[0-9a-f]{64})"
)
_REALIZATION_TYPES = frozenset({"logged_future_privileged"})
_PLAN_FLOAT_FIELDS = ("timestamps", "x", "y", "heading", "vx", "vy")
_AGENT_FLOAT_FIELDS = ("x", "y", "heading", "vx", "vy")


def _nonempty_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _strict_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


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
                    raise ValueError(f"{path} keys must be non-empty strings")
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

    if isinstance(value, np.generic):
        value = value.item()
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


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _frozen_json_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return _freeze_json(value, path=name, active=set())


def canonical_configuration_json(value: Mapping[str, Any]) -> str:
    """Encode one configuration mapping using the frozen canonical JSON rule."""

    frozen = _frozen_json_mapping(value, name="configuration")
    return json.dumps(
        _thaw_json(frozen),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _validate_domain(domain: bytes | str) -> bytes:
    if isinstance(domain, str):
        try:
            domain_bytes = domain.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("canonical domain must be ASCII") from exc
    elif isinstance(domain, bytes):
        domain_bytes = bytes(domain)
        try:
            domain_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("canonical domain must be ASCII") from exc
    else:
        raise TypeError("canonical domain must be bytes or str")
    if not domain_bytes or b"\x00" in domain_bytes:
        raise ValueError("canonical domain must be non-empty and contain no NUL")
    return domain_bytes


def canonical_configuration_bytes(
    value: Mapping[str, Any],
    *,
    domain: bytes | str = CONFIGURATION_DOMAIN,
) -> bytes:
    """Return ``ASCII(domain) || NUL || canonical JSON``."""

    return (
        _validate_domain(domain)
        + b"\x00"
        + canonical_configuration_json(value).encode("utf-8")
    )


def _expect_exact_keys(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a mapping")
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{name} fields do not match schema; missing={missing}, extra={extra}"
        )


@dataclass(frozen=True, slots=True)
class EgoInterventionSpec:
    """Immutable configuration and canonical identity for one ego intervention."""

    family: str
    version: str
    dose: float
    duration_s: float
    parameters: Mapping[str, Any] = field(default_factory=dict)
    access_class: str = "logged_future_privileged"
    configuration_fingerprint: str | None = None

    def __post_init__(self) -> None:
        family = _nonempty_text(self.family, name="family")
        if _FAMILY.fullmatch(family) is None:
            raise ValueError("family must use lowercase snake_case")
        version = _nonempty_text(self.version, name="version")
        if _VERSION.fullmatch(version) is None:
            raise ValueError("version must have the form vN")
        dose = _finite_float(self.dose, name="dose", minimum=0.0)
        duration_s = _finite_float(
            self.duration_s,
            name="duration_s",
            minimum=0.0,
        )
        access_class = _nonempty_text(
            self.access_class,
            name="access_class",
        )
        if access_class not in _REALIZATION_TYPES:
            raise ValueError(
                f"access_class must be one of {sorted(_REALIZATION_TYPES)}"
            )
        parameters = _frozen_json_mapping(
            self.parameters,
            name="parameters",
        )

        object.__setattr__(self, "family", family)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "dose", dose)
        object.__setattr__(self, "duration_s", duration_s)
        object.__setattr__(self, "access_class", access_class)
        object.__setattr__(self, "parameters", parameters)

        expected = hashlib.sha256(self.canonical_bytes).hexdigest()
        supplied = self.configuration_fingerprint
        if supplied is not None and supplied != expected:
            raise ValueError(
                "configuration_fingerprint does not match canonical configuration"
            )
        object.__setattr__(self, "configuration_fingerprint", expected)

    @property
    def intervention_id(self) -> str:
        return f"{self.family}/{self.version}"

    @property
    def canonical_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "access_class": self.access_class,
                "dose": self.dose,
                "duration_s": self.duration_s,
                "family": self.family,
                "parameters": _thaw_json(self.parameters),
                "schema_version": M6_INTERVENTION_SCHEMA_VERSION,
                "version": self.version,
            }
        )

    @property
    def canonical_json(self) -> str:
        return canonical_configuration_json(self.canonical_payload)

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_configuration_bytes(self.canonical_payload)

    def revalidate(self) -> None:
        if hashlib.sha256(self.canonical_bytes).hexdigest() != (
            self.configuration_fingerprint
        ):
            raise ValueError("ego intervention configuration was mutated")

    def to_dict(self) -> dict[str, Any]:
        return {
            **_thaw_json(self.canonical_payload),
            "configuration_fingerprint": self.configuration_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EgoInterventionSpec":
        expected = frozenset(
            {
                "schema_version",
                "family",
                "version",
                "dose",
                "duration_s",
                "parameters",
                "access_class",
                "configuration_fingerprint",
            }
        )
        _expect_exact_keys(payload, expected, name="EgoInterventionSpec")
        if payload["schema_version"] != M6_INTERVENTION_SCHEMA_VERSION:
            raise ValueError("unsupported ego intervention schema_version")
        return cls(
            family=payload["family"],
            version=payload["version"],
            dose=payload["dose"],
            duration_s=payload["duration_s"],
            parameters=payload["parameters"],
            access_class=payload["access_class"],
            configuration_fingerprint=payload["configuration_fingerprint"],
        )


@dataclass(frozen=True, slots=True)
class FeasibilityAudit:
    """Typed result of the frozen plan-feasibility checks."""

    passed: bool
    checks: Mapping[str, bool]
    failure_reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise ValueError("passed must be a boolean")
        if not isinstance(self.checks, Mapping) or not self.checks:
            raise ValueError("checks must be a non-empty mapping")
        normalized_checks: dict[str, bool] = {}
        for key, value in self.checks.items():
            if not isinstance(key, str) or _CHECK.fullmatch(key) is None:
                raise ValueError("feasibility check names must be lowercase snake_case")
            if type(value) is not bool:
                raise ValueError("feasibility check values must be booleans")
            normalized_checks[key] = value
        checks = MappingProxyType(normalized_checks)

        if self.passed:
            if self.failure_reason is not None:
                raise ValueError("a passed audit cannot have a failure_reason")
            if not all(checks.values()):
                raise ValueError("a passed audit requires every check to pass")
        else:
            reason = _nonempty_text(
                self.failure_reason,
                name="failure_reason",
            )
            if _REASON.fullmatch(reason) is None:
                raise ValueError(
                    "failure_reason must use lowercase snake_case"
                )
            if all(checks.values()):
                raise ValueError("a failed audit requires at least one failed check")
            object.__setattr__(self, "failure_reason", reason)

        object.__setattr__(self, "checks", checks)
        object.__setattr__(
            self,
            "details",
            _frozen_json_mapping(self.details, name="details"),
        )

    @classmethod
    def accepted(
        cls,
        checks: Mapping[str, bool],
        *,
        details: Mapping[str, Any] | None = None,
    ) -> "FeasibilityAudit":
        return cls(
            passed=True,
            checks=checks,
            details={} if details is None else details,
        )

    @classmethod
    def rejected(
        cls,
        reason: str,
        checks: Mapping[str, bool],
        *,
        details: Mapping[str, Any] | None = None,
    ) -> "FeasibilityAudit":
        return cls(
            passed=False,
            checks=checks,
            failure_reason=reason,
            details={} if details is None else details,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": M6_FEASIBILITY_SCHEMA_VERSION,
            "passed": self.passed,
            "checks": dict(self.checks),
            "failure_reason": self.failure_reason,
            "details": _thaw_json(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeasibilityAudit":
        expected = frozenset(
            {
                "schema_version",
                "passed",
                "checks",
                "failure_reason",
                "details",
            }
        )
        _expect_exact_keys(payload, expected, name="FeasibilityAudit")
        if payload["schema_version"] != M6_FEASIBILITY_SCHEMA_VERSION:
            raise ValueError("unsupported feasibility schema_version")
        return cls(
            passed=payload["passed"],
            checks=payload["checks"],
            failure_reason=payload["failure_reason"],
            details=payload["details"],
        )


@dataclass(frozen=True, slots=True)
class InterventionEligibility:
    """Outcome-blind source eligibility and the frozen analysis target."""

    eligible: bool
    reason: str | None
    analysis_window: tuple[int, int]
    target_index: int | None

    def __post_init__(self) -> None:
        if type(self.eligible) is not bool:
            raise ValueError("eligible must be a boolean")
        if (
            isinstance(self.analysis_window, (str, bytes))
            or not isinstance(self.analysis_window, Sequence)
            or len(self.analysis_window) != 2
        ):
            raise ValueError("analysis_window must be a (current, stop) pair")
        current = _strict_int(
            self.analysis_window[0],
            name="analysis_window current index",
            minimum=0,
        )
        stop = _strict_int(
            self.analysis_window[1],
            name="analysis_window stop index",
            minimum=0,
        )
        if stop - current != M6_ANALYSIS_TRANSITIONS:
            raise ValueError(
                "analysis_window must span exactly 40 post-current transitions"
            )
        object.__setattr__(self, "analysis_window", (current, stop))

        target_index = self.target_index
        if target_index is not None:
            target_index = _strict_int(
                target_index,
                name="target_index",
                minimum=0,
            )
            object.__setattr__(self, "target_index", target_index)

        if self.eligible:
            if self.reason is not None:
                raise ValueError("eligible input cannot have a rejection reason")
            if target_index is None:
                raise ValueError("eligible input requires a frozen target_index")
        else:
            reason = _nonempty_text(self.reason, name="reason")
            if _REASON.fullmatch(reason) is None:
                raise ValueError("reason must use lowercase snake_case")
            if reason not in M6_PRIMARY_ELIGIBILITY_REASONS:
                raise ValueError(
                    "reason must be one of the registered M6 eligibility reasons"
                )
            object.__setattr__(self, "reason", reason)

    @property
    def current_index(self) -> int:
        return self.analysis_window[0]

    @property
    def stop_index(self) -> int:
        return self.analysis_window[1]

    @classmethod
    def accepted(
        cls,
        analysis_window: tuple[int, int],
        target_index: int,
    ) -> "InterventionEligibility":
        return cls(
            eligible=True,
            reason=None,
            analysis_window=analysis_window,
            target_index=target_index,
        )

    @classmethod
    def rejected(
        cls,
        reason: str,
        analysis_window: tuple[int, int],
        target_index: int | None = None,
    ) -> "InterventionEligibility":
        return cls(
            eligible=False,
            reason=reason,
            analysis_window=analysis_window,
            target_index=target_index,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": M6_ELIGIBILITY_SCHEMA_VERSION,
            "eligible": self.eligible,
            "reason": self.reason,
            "analysis_window": list(self.analysis_window),
            "target_index": self.target_index,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "InterventionEligibility":
        expected = frozenset(
            {
                "schema_version",
                "eligible",
                "reason",
                "analysis_window",
                "target_index",
            }
        )
        _expect_exact_keys(payload, expected, name="InterventionEligibility")
        if payload["schema_version"] != M6_ELIGIBILITY_SCHEMA_VERSION:
            raise ValueError("unsupported intervention eligibility schema_version")
        return cls(
            eligible=payload["eligible"],
            reason=payload["reason"],
            analysis_window=payload["analysis_window"],
            target_index=payload["target_index"],
        )


def _readonly_vector(
    value: Any,
    *,
    dtype: np.dtype[Any] | type[Any],
    name: str,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {array.shape}")
    immutable = np.frombuffer(
        array.tobytes(order="C"),
        dtype=array.dtype,
    )
    immutable.setflags(write=False)
    return immutable


def _readonly_matrix(
    value: Any,
    *,
    dtype: np.dtype[Any] | type[Any],
    name: str,
    columns: int,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(
            f"{name} must have shape [N, {columns}], got {array.shape}"
        )
    immutable = np.frombuffer(
        array.tobytes(order="C"),
        dtype=array.dtype,
    ).reshape(array.shape)
    immutable.setflags(write=False)
    return immutable


def _u64(value: int) -> bytes:
    if not 0 <= value <= (1 << 64) - 1:
        raise ValueError("canonical length does not fit unsigned 64-bit")
    return struct.pack(">Q", value)


def canonical_perturbation_identity(
    spec: EgoInterventionSpec,
    plan_fingerprint: str,
) -> str:
    """Return the only accepted free-text projection of a typed ego plan."""

    if not isinstance(spec, EgoInterventionSpec):
        raise TypeError("spec must be an EgoInterventionSpec")
    spec.revalidate()
    if not isinstance(plan_fingerprint, str) or (
        _SHA256.fullmatch(plan_fingerprint) is None
    ):
        raise ValueError("plan_fingerprint must be a lowercase SHA-256 digest")
    return f"{spec.intervention_id}@sha256:{plan_fingerprint}"


@dataclass(frozen=True, slots=True)
class EgoTrajectoryPlan:
    """Exact 41-frame ego trajectory consumed by every paired condition."""

    spec: EgoInterventionSpec
    timestamps: np.ndarray
    valid: np.ndarray
    applied: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    realization_type: str
    feasibility: FeasibilityAudit
    plan_fingerprint: str | None = None
    audit_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, EgoInterventionSpec):
            raise TypeError("spec must be an EgoInterventionSpec")
        self.spec.revalidate()
        realization_type = _nonempty_text(
            self.realization_type,
            name="realization_type",
        )
        if realization_type not in _REALIZATION_TYPES:
            raise ValueError(
                f"realization_type must be one of {sorted(_REALIZATION_TYPES)}"
            )
        if realization_type != self.spec.access_class:
            raise ValueError(
                "realization_type must equal intervention access_class"
            )
        object.__setattr__(self, "realization_type", realization_type)

        object.__setattr__(
            self,
            "valid",
            _readonly_vector(self.valid, dtype=np.bool_, name="valid"),
        )
        object.__setattr__(
            self,
            "applied",
            _readonly_vector(self.applied, dtype=np.bool_, name="applied"),
        )
        for name in _PLAN_FLOAT_FIELDS:
            object.__setattr__(
                self,
                name,
                _readonly_vector(
                    getattr(self, name),
                    dtype=np.dtype("<f8"),
                    name=name,
                ),
            )

        arrays = (
            self.valid,
            self.applied,
            *(getattr(self, name) for name in _PLAN_FLOAT_FIELDS),
        )
        if any(array.shape != (M6_PLAN_FRAME_COUNT,) for array in arrays):
            raise ValueError(
                "every ego plan array must have exactly 41 frames "
                "(current plus 40 transitions)"
            )
        if not all(
            np.all(np.isfinite(getattr(self, name)))
            for name in _PLAN_FLOAT_FIELDS
        ):
            raise ValueError("ego plan timestamps and states must be finite")
        if np.any(np.diff(self.timestamps) <= 0.0):
            raise ValueError("ego plan timestamps must be strictly increasing")
        if np.any(self.heading < -np.pi) or np.any(self.heading > np.pi):
            raise ValueError("ego plan headings must lie in [-pi, pi]")
        if not bool(np.all(self.valid)):
            raise ValueError("an executable M6 v1 ego plan must be valid at all frames")
        if bool(self.applied[0]) or not bool(np.all(self.applied[1:])):
            raise ValueError(
                "applied must be false at current and true for all 40 future frames"
            )
        if not isinstance(self.feasibility, FeasibilityAudit):
            raise TypeError("feasibility must be a FeasibilityAudit")
        if not self.feasibility.passed:
            raise ValueError("an executable ego plan requires a passed feasibility audit")

        expected = hashlib.sha256(self.canonical_bytes).hexdigest()
        supplied = self.plan_fingerprint
        if supplied is not None and supplied != expected:
            raise ValueError("plan_fingerprint does not match canonical plan bytes")
        object.__setattr__(self, "plan_fingerprint", expected)
        expected_audit = hashlib.sha256(self.audit_canonical_bytes).hexdigest()
        supplied_audit = self.audit_fingerprint
        if supplied_audit is not None and supplied_audit != expected_audit:
            raise ValueError(
                "audit_fingerprint does not match the plan feasibility envelope"
            )
        object.__setattr__(self, "audit_fingerprint", expected_audit)

    @property
    def frame_count(self) -> int:
        return M6_PLAN_FRAME_COUNT

    @property
    def configuration_fingerprint(self) -> str:
        return self.spec.configuration_fingerprint

    @property
    def canonical_bytes(self) -> bytes:
        config = self.spec.configuration_fingerprint.encode("ascii")
        realization = self.realization_type.encode("utf-8")
        parts = [
            PLAN_DOMAIN,
            b"\x00",
            _u64(len(config)),
            config,
            _u64(len(realization)),
            realization,
            _u64(self.frame_count),
            self.valid.astype(np.uint8, copy=False).tobytes(order="C"),
            self.applied.astype(np.uint8, copy=False).tobytes(order="C"),
        ]
        parts.extend(
            np.asarray(getattr(self, name), dtype="<f8").tobytes(order="C")
            for name in _PLAN_FLOAT_FIELDS
        )
        return b"".join(parts)

    @property
    def audit_canonical_bytes(self) -> bytes:
        """Bind the accepted plan identity to its complete feasibility evidence."""

        plan_fingerprint = self.plan_fingerprint.encode("ascii")
        feasibility = canonical_configuration_json(
            self.feasibility.to_dict()
        ).encode("utf-8")
        return b"".join(
            (
                PLAN_AUDIT_DOMAIN,
                b"\x00",
                _u64(len(plan_fingerprint)),
                plan_fingerprint,
                _u64(len(feasibility)),
                feasibility,
            )
        )

    @property
    def perturbation_identity(self) -> str:
        return canonical_perturbation_identity(
            self.spec,
            self.plan_fingerprint,
        )

    def revalidate(self) -> None:
        self.spec.revalidate()
        if not self.feasibility.passed:
            raise ValueError("ego plan feasibility audit no longer passes")
        if hashlib.sha256(self.canonical_bytes).hexdigest() != self.plan_fingerprint:
            raise ValueError("ego trajectory plan was mutated")
        if (
            hashlib.sha256(self.audit_canonical_bytes).hexdigest()
            != self.audit_fingerprint
        ):
            raise ValueError("ego trajectory plan feasibility envelope was mutated")
        if self.perturbation_identity != canonical_perturbation_identity(
            self.spec,
            self.plan_fingerprint,
        ):
            raise ValueError("ego trajectory perturbation identity drifted")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": M6_PLAN_SCHEMA_VERSION,
            "spec": self.spec.to_dict(),
            "realization_type": self.realization_type,
            "frame_count": self.frame_count,
            "valid_u8_hex": self.valid.astype(
                np.uint8,
                copy=False,
            ).tobytes(order="C").hex(),
            "applied_u8_hex": self.applied.astype(
                np.uint8,
                copy=False,
            ).tobytes(order="C").hex(),
            "feasibility": self.feasibility.to_dict(),
            "plan_fingerprint": self.plan_fingerprint,
            "audit_fingerprint": self.audit_fingerprint,
            "perturbation_identity": self.perturbation_identity,
        }
        for name in _PLAN_FLOAT_FIELDS:
            result[f"{name}_f64le_hex"] = np.asarray(
                getattr(self, name),
                dtype="<f8",
            ).tobytes(order="C").hex()
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EgoTrajectoryPlan":
        expected = frozenset(
            {
                "schema_version",
                "spec",
                "realization_type",
                "frame_count",
                "valid_u8_hex",
                "applied_u8_hex",
                "timestamps_f64le_hex",
                "x_f64le_hex",
                "y_f64le_hex",
                "heading_f64le_hex",
                "vx_f64le_hex",
                "vy_f64le_hex",
                "feasibility",
                "plan_fingerprint",
                "audit_fingerprint",
                "perturbation_identity",
            }
        )
        _expect_exact_keys(payload, expected, name="EgoTrajectoryPlan")
        if payload["schema_version"] != M6_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported ego trajectory plan schema_version")
        if payload["frame_count"] != M6_PLAN_FRAME_COUNT:
            raise ValueError("serialized ego plan frame_count must equal 41")

        def decode(name: str, *, dtype: str, size: int) -> np.ndarray:
            value = payload[name]
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a hexadecimal string")
            try:
                raw = bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError(f"{name} must be canonical hexadecimal") from exc
            if value != raw.hex():
                raise ValueError(f"{name} must use lowercase canonical hexadecimal")
            if len(raw) != size:
                raise ValueError(f"{name} has the wrong encoded byte length")
            return np.frombuffer(raw, dtype=dtype).copy()

        masks_size = M6_PLAN_FRAME_COUNT
        float_size = M6_PLAN_FRAME_COUNT * np.dtype("<f8").itemsize
        valid = decode(
            "valid_u8_hex",
            dtype="u1",
            size=masks_size,
        )
        applied = decode(
            "applied_u8_hex",
            dtype="u1",
            size=masks_size,
        )
        if np.any(valid > 1) or np.any(applied > 1):
            raise ValueError("serialized plan masks must contain only byte 0 or 1")
        plan = cls(
            spec=EgoInterventionSpec.from_dict(payload["spec"]),
            timestamps=decode(
                "timestamps_f64le_hex",
                dtype="<f8",
                size=float_size,
            ),
            valid=valid,
            applied=applied,
            x=decode("x_f64le_hex", dtype="<f8", size=float_size),
            y=decode("y_f64le_hex", dtype="<f8", size=float_size),
            heading=decode(
                "heading_f64le_hex",
                dtype="<f8",
                size=float_size,
            ),
            vx=decode("vx_f64le_hex", dtype="<f8", size=float_size),
            vy=decode("vy_f64le_hex", dtype="<f8", size=float_size),
            realization_type=payload["realization_type"],
            feasibility=FeasibilityAudit.from_dict(payload["feasibility"]),
            plan_fingerprint=payload["plan_fingerprint"],
            audit_fingerprint=payload["audit_fingerprint"],
        )
        if payload["perturbation_identity"] != plan.perturbation_identity:
            raise ValueError(
                "serialized perturbation_identity does not match the ego plan"
            )
        return plan

    def to_json(self) -> str:
        return canonical_configuration_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "EgoTrajectoryPlan":
        if not isinstance(text, str):
            raise TypeError("serialized ego plan must be a string")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("serialized ego plan is not valid JSON") from exc
        recoded = canonical_configuration_json(payload)
        if recoded != text:
            raise ValueError("serialized ego plan JSON is not canonical")
        return cls.from_dict(payload)

    def serialize(self) -> bytes:
        return PLAN_ENVELOPE_DOMAIN + b"\x00" + self.to_json().encode("utf-8")

    @classmethod
    def deserialize(cls, payload: bytes) -> "EgoTrajectoryPlan":
        if not isinstance(payload, bytes):
            raise TypeError("serialized ego plan must be bytes")
        prefix = PLAN_ENVELOPE_DOMAIN + b"\x00"
        if not payload.startswith(prefix):
            raise ValueError("serialized ego plan has the wrong domain")
        try:
            text = payload[len(prefix) :].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("serialized ego plan is not UTF-8") from exc
        return cls.from_json(text)


def _snapshot_digest(parts: Sequence[bytes], *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(b"\x00")
    for part in parts:
        digest.update(_u64(len(part)))
        digest.update(part)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_configuration_json(value).encode("utf-8")


@dataclass(frozen=True, slots=True)
class AgentTrajectorySnapshot:
    """Immutable, bytes-backed snapshot of one mutable core ``Agent``."""

    id: int
    type: AgentType
    valid: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    length: float
    width: float
    _integrity_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "id",
            _strict_int(self.id, name="agent id"),
        )
        object.__setattr__(self, "type", AgentType(self.type))
        object.__setattr__(
            self,
            "valid",
            _readonly_vector(self.valid, dtype=np.bool_, name="agent.valid"),
        )
        for name in _AGENT_FLOAT_FIELDS:
            object.__setattr__(
                self,
                name,
                _readonly_vector(
                    getattr(self, name),
                    dtype=np.dtype("<f8"),
                    name=f"agent.{name}",
                ),
            )
        shapes = {
            self.valid.shape,
            *(getattr(self, name).shape for name in _AGENT_FLOAT_FIELDS),
        }
        if len(shapes) != 1:
            raise ValueError("snapshot agent arrays must share one shape")
        if not all(
            np.all(np.isfinite(getattr(self, name)))
            for name in _AGENT_FLOAT_FIELDS
        ):
            raise ValueError("snapshot agent states must be finite")
        if np.any(self.heading < -np.pi) or np.any(self.heading > np.pi):
            raise ValueError("snapshot agent headings must lie in [-pi, pi]")
        length = _finite_float(self.length, name="agent length", minimum=0.0)
        width = _finite_float(self.width, name="agent width", minimum=0.0)
        if length == 0.0 or width == 0.0:
            raise ValueError("snapshot agent dimensions must be positive")
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "width", width)
        object.__setattr__(
            self,
            "_integrity_fingerprint",
            self._compute_integrity_fingerprint(),
        )

    @classmethod
    def from_agent(cls, agent: Agent) -> "AgentTrajectorySnapshot":
        if not isinstance(agent, Agent):
            raise TypeError("agent must be an Agent")
        return cls(
            id=agent.id,
            type=agent.type,
            valid=agent.valid,
            x=agent.x,
            y=agent.y,
            heading=agent.heading,
            vx=agent.vx,
            vy=agent.vy,
            length=agent.length,
            width=agent.width,
        )

    @property
    def num_steps(self) -> int:
        return int(self.valid.shape[0])

    def speed(self) -> np.ndarray:
        return np.hypot(self.vx, self.vy)

    def _compute_integrity_fingerprint(self) -> str:
        scalar = _json_bytes(
            {
                "id": self.id,
                "length": self.length,
                "type": self.type.value,
                "width": self.width,
            }
        )
        return _snapshot_digest(
            (
                scalar,
                self.valid.astype(np.uint8, copy=False).tobytes(order="C"),
                *(
                    np.asarray(
                        getattr(self, name),
                        dtype="<f8",
                    ).tobytes(order="C")
                    for name in _AGENT_FLOAT_FIELDS
                ),
            ),
            domain=b"evalsim-agent-trajectory-snapshot-v1",
        )

    def revalidate(self) -> None:
        if self._compute_integrity_fingerprint() != self._integrity_fingerprint:
            raise ValueError("agent trajectory snapshot was mutated")


@dataclass(frozen=True, slots=True)
class MapPolylineSnapshot:
    """Immutable snapshot of a static map polyline."""

    type: MapType
    xy: np.ndarray
    _integrity_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", MapType(self.type))
        object.__setattr__(
            self,
            "xy",
            _readonly_matrix(
                self.xy,
                dtype=np.dtype("<f8"),
                name="map polyline xy",
                columns=2,
            ),
        )
        if not np.all(np.isfinite(self.xy)):
            raise ValueError("map polyline coordinates must be finite")
        object.__setattr__(
            self,
            "_integrity_fingerprint",
            self._compute_integrity_fingerprint(),
        )

    @classmethod
    def from_polyline(
        cls,
        polyline: MapPolyline,
    ) -> "MapPolylineSnapshot":
        if not isinstance(polyline, MapPolyline):
            raise TypeError("polyline must be a MapPolyline")
        return cls(type=polyline.type, xy=polyline.xy)

    def _compute_integrity_fingerprint(self) -> str:
        return _snapshot_digest(
            (
                self.type.value.encode("ascii"),
                _u64(self.xy.shape[0]),
                np.asarray(self.xy, dtype="<f8").tobytes(order="C"),
            ),
            domain=b"evalsim-map-polyline-snapshot-v1",
        )

    def revalidate(self) -> None:
        if self._compute_integrity_fingerprint() != self._integrity_fingerprint:
            raise ValueError("map polyline snapshot was mutated")


@dataclass(frozen=True, slots=True)
class ScenarioSnapshot:
    """Immutable defensive snapshot of a source scenario."""

    scenario_id: str
    timestamps: np.ndarray
    agents: tuple[AgentTrajectorySnapshot, ...]
    map: tuple[MapPolylineSnapshot, ...]
    ego_index: int
    metadata: Mapping[str, Any]
    _integrity_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scenario_id",
            _nonempty_text(self.scenario_id, name="scenario_id"),
        )
        object.__setattr__(
            self,
            "timestamps",
            _readonly_vector(
                self.timestamps,
                dtype=np.dtype("<f8"),
                name="scenario timestamps",
            ),
        )
        if (
            not np.all(np.isfinite(self.timestamps))
            or np.any(np.diff(self.timestamps) <= 0.0)
        ):
            raise ValueError("scenario timestamps must be finite and increasing")
        agents = tuple(self.agents)
        if not agents or not all(
            isinstance(agent, AgentTrajectorySnapshot) for agent in agents
        ):
            raise ValueError("scenario snapshot requires immutable agent snapshots")
        for agent in agents:
            agent.revalidate()
        if any(agent.num_steps != len(self.timestamps) for agent in agents):
            raise ValueError("scenario snapshot agent horizons must match timestamps")
        if len({agent.id for agent in agents}) != len(agents):
            raise ValueError("scenario snapshot agent IDs must be unique")
        object.__setattr__(self, "agents", agents)
        polylines = tuple(self.map)
        if not all(
            isinstance(polyline, MapPolylineSnapshot)
            for polyline in polylines
        ):
            raise ValueError("scenario map must contain immutable snapshots")
        for polyline in polylines:
            polyline.revalidate()
        object.__setattr__(self, "map", polylines)
        ego_index = _strict_int(
            self.ego_index,
            name="ego_index",
            minimum=0,
        )
        if ego_index >= len(agents):
            raise ValueError("ego_index is outside scenario agent order")
        object.__setattr__(self, "ego_index", ego_index)
        object.__setattr__(
            self,
            "metadata",
            _frozen_json_mapping(self.metadata, name="scenario metadata"),
        )
        object.__setattr__(
            self,
            "_integrity_fingerprint",
            self._compute_integrity_fingerprint(),
        )

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "ScenarioSnapshot":
        if not isinstance(scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        return cls(
            scenario_id=scenario.scenario_id,
            timestamps=scenario.timestamps,
            agents=tuple(
                AgentTrajectorySnapshot.from_agent(agent)
                for agent in scenario.agents
            ),
            map=tuple(
                MapPolylineSnapshot.from_polyline(polyline)
                for polyline in scenario.map
            ),
            ego_index=scenario.ego_index,
            metadata=scenario.metadata,
        )

    @property
    def num_steps(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def ego(self) -> AgentTrajectorySnapshot:
        return self.agents[self.ego_index]

    def to_scenario(self) -> Scenario:
        """Return a detached mutable source for registered-plan recompilation."""

        self.revalidate()
        return Scenario(
            scenario_id=self.scenario_id,
            timestamps=np.array(self.timestamps, copy=True),
            agents=[
                Agent(
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
                for agent in self.agents
            ],
            map=[
                MapPolyline(
                    type=polyline.type,
                    xy=np.array(polyline.xy, copy=True),
                )
                for polyline in self.map
            ],
            ego_index=self.ego_index,
            metadata=_thaw_json(self.metadata),
        )

    def _compute_integrity_fingerprint(self) -> str:
        return _snapshot_digest(
            (
                self.scenario_id.encode("utf-8"),
                np.asarray(self.timestamps, dtype="<f8").tobytes(order="C"),
                _u64(self.ego_index),
                _json_bytes(self.metadata),
                *(agent._integrity_fingerprint.encode("ascii") for agent in self.agents),
                *(
                    polyline._integrity_fingerprint.encode("ascii")
                    for polyline in self.map
                ),
            ),
            domain=b"evalsim-scenario-snapshot-v1",
        )

    def revalidate(self) -> None:
        for agent in self.agents:
            agent.revalidate()
        for polyline in self.map:
            polyline.revalidate()
        if self._compute_integrity_fingerprint() != self._integrity_fingerprint:
            raise ValueError("scenario snapshot was mutated")


@dataclass(frozen=True, slots=True)
class RolloutSnapshot:
    """Immutable defensive snapshot of a simulator rollout."""

    scenario_id: str
    sim_name: str
    sim_version: str
    seed: int
    timestamps: np.ndarray
    agents: tuple[AgentTrajectorySnapshot, ...]
    perturbation: str | None
    metadata: Mapping[str, Any]
    _integrity_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("scenario_id", "sim_name", "sim_version"):
            object.__setattr__(
                self,
                name,
                _nonempty_text(getattr(self, name), name=name),
            )
        object.__setattr__(self, "seed", _strict_int(self.seed, name="seed"))
        object.__setattr__(
            self,
            "timestamps",
            _readonly_vector(
                self.timestamps,
                dtype=np.dtype("<f8"),
                name="rollout timestamps",
            ),
        )
        if (
            not np.all(np.isfinite(self.timestamps))
            or np.any(np.diff(self.timestamps) <= 0.0)
        ):
            raise ValueError("rollout timestamps must be finite and increasing")
        agents = tuple(self.agents)
        if not agents or not all(
            isinstance(agent, AgentTrajectorySnapshot) for agent in agents
        ):
            raise ValueError("rollout snapshot requires immutable agent snapshots")
        for agent in agents:
            agent.revalidate()
        if any(agent.num_steps != len(self.timestamps) for agent in agents):
            raise ValueError("rollout snapshot agent horizons must match timestamps")
        if len({agent.id for agent in agents}) != len(agents):
            raise ValueError("rollout snapshot agent IDs must be unique")
        object.__setattr__(self, "agents", agents)
        if self.perturbation is not None:
            perturbation = _nonempty_text(
                self.perturbation,
                name="perturbation",
            )
            object.__setattr__(self, "perturbation", perturbation)
        object.__setattr__(
            self,
            "metadata",
            _frozen_json_mapping(self.metadata, name="rollout metadata"),
        )
        object.__setattr__(
            self,
            "_integrity_fingerprint",
            self._compute_integrity_fingerprint(),
        )

    @classmethod
    def from_rollout(cls, rollout: Rollout) -> "RolloutSnapshot":
        if not isinstance(rollout, Rollout):
            raise TypeError("rollout must be a Rollout")
        return cls(
            scenario_id=rollout.scenario_id,
            sim_name=rollout.sim_name,
            sim_version=rollout.sim_version,
            seed=rollout.seed,
            timestamps=rollout.timestamps,
            agents=tuple(
                AgentTrajectorySnapshot.from_agent(agent)
                for agent in rollout.agents
            ),
            perturbation=rollout.perturbation,
            metadata=rollout.metadata,
        )

    @property
    def num_steps(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    def _compute_integrity_fingerprint(self) -> str:
        return _snapshot_digest(
            (
                _json_bytes(
                    {
                        "perturbation": self.perturbation,
                        "scenario_id": self.scenario_id,
                        "seed": self.seed,
                        "sim_name": self.sim_name,
                        "sim_version": self.sim_version,
                    }
                ),
                np.asarray(self.timestamps, dtype="<f8").tobytes(order="C"),
                _json_bytes(self.metadata),
                *(agent._integrity_fingerprint.encode("ascii") for agent in self.agents),
            ),
            domain=b"evalsim-rollout-snapshot-v1",
        )

    def revalidate(self) -> None:
        for agent in self.agents:
            agent.revalidate()
        if self._compute_integrity_fingerprint() != self._integrity_fingerprint:
            raise ValueError("rollout snapshot was mutated")


def _is_canonical_perturbation_identity(value: str) -> bool:
    return _PERTURBATION_IDENTITY.fullmatch(value) is not None


def _same_agent_contract(
    source: AgentTrajectorySnapshot,
    left: AgentTrajectorySnapshot,
    right: AgentTrajectorySnapshot,
    *,
    stop_index: int,
) -> bool:
    return (
        source.id == left.id == right.id
        and source.type == left.type == right.type
        and source.length == left.length == right.length
        and source.width == left.width == right.width
        and np.array_equal(source.valid[: stop_index + 1], left.valid)
        and np.array_equal(source.valid[: stop_index + 1], right.valid)
    )


def _required_metadata_value(
    metadata: Mapping[str, Any],
    key: str,
    *,
    side: str,
) -> Any:
    if key not in metadata:
        raise ValueError(f"{side} rollout metadata is missing {key!r}")
    return metadata[key]


def _paired_rollout_configuration(
    metadata: Mapping[str, Any],
    *,
    side: str,
) -> Mapping[str, Any]:
    """Extract the registered pair-invariant metadata projection.

    ``dynamics.clamp_counts`` is an outcome diagnostic and may legitimately differ
    after the intervention.  It is required for provenance but excluded from the
    configuration projection.
    """

    engine = _required_metadata_value(metadata, "engine", side=side)
    policy = _required_metadata_value(metadata, "policy", side=side)
    dynamics = _required_metadata_value(metadata, "dynamics", side=side)
    if not isinstance(engine, Mapping):
        raise ValueError(f"{side} rollout engine metadata must be a mapping")
    if not isinstance(policy, Mapping):
        raise ValueError(f"{side} rollout policy metadata must be a mapping")
    if not isinstance(dynamics, Mapping):
        raise ValueError(f"{side} rollout dynamics metadata must be a mapping")
    if "clamp_counts" not in dynamics or not isinstance(
        dynamics["clamp_counts"],
        Mapping,
    ):
        raise ValueError(
            f"{side} rollout dynamics metadata requires clamp_counts"
        )
    dynamics_config = {
        key: _thaw_json(value)
        for key, value in dynamics.items()
        if key != "clamp_counts"
    }
    return MappingProxyType(
        {
            "engine": _thaw_json(engine),
            "policy": _thaw_json(policy),
            "dynamics": dynamics_config,
            "ego_control": _required_metadata_value(
                metadata,
                "ego_control",
                side=side,
            ),
            "rollout_start_index": _required_metadata_value(
                metadata,
                "rollout_start_index",
                side=side,
            ),
            "controlled_agent_ids": _thaw_json(
                _required_metadata_value(
                    metadata,
                    "controlled_agent_ids",
                    side=side,
                )
            ),
            "agent_control_modes": _thaw_json(
                _required_metadata_value(
                    metadata,
                    "agent_control_modes",
                    side=side,
                )
            ),
            "scenario_source": _required_metadata_value(
                metadata,
                "scenario_source",
                side=side,
            ),
            "scenario_source_fingerprint": _required_metadata_value(
                metadata,
                "scenario_source_fingerprint",
                side=side,
            ),
        }
    )


def _expected_agent_control_modes(
    scenario: ScenarioSnapshot,
    policy: Mapping[str, Any],
) -> dict[str, str]:
    """Derive the exact typed-run component-mode mask from source and policy."""

    required = frozenset(
        {
            "name",
            "version",
            "deterministic",
            "required_features",
            "supported_agent_types",
            "params",
            "known_limitations",
            "fallback_policy",
        }
    )
    if set(policy) != required:
        raise ValueError(
            "pair policy metadata does not match the complete policy schema"
        )
    policy_name = _nonempty_text(policy["name"], name="policy name")
    supported_raw = policy["supported_agent_types"]
    if (
        isinstance(supported_raw, (str, bytes))
        or not isinstance(supported_raw, Sequence)
    ):
        raise ValueError("policy supported_agent_types must be a sequence")
    try:
        supported = tuple(AgentType(value) for value in supported_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "policy supported_agent_types contains an unknown type"
        ) from exc
    if len(set(supported)) != len(supported):
        raise ValueError("policy supported_agent_types must be unique")
    fallback = policy["fallback_policy"]
    if fallback is not None:
        fallback = _nonempty_text(fallback, name="policy fallback_policy")

    expected: dict[str, str] = {}
    for index, agent in enumerate(scenario.agents):
        if index == scenario.ego_index:
            mode = "typed_ego_plan"
        elif agent.type in supported:
            mode = policy_name
        elif fallback is not None:
            mode = fallback
        else:
            raise ValueError(
                "pair policy metadata leaves a world agent without a control mode"
            )
        expected[str(agent.id)] = mode
    return expected


@dataclass(frozen=True, slots=True)
class CounterfactualPair:
    """One complete, immutable baseline/treatment scene pair."""

    scenario: Scenario | ScenarioSnapshot
    baseline: Rollout | RolloutSnapshot
    intervention: Rollout | RolloutSnapshot
    baseline_plan: EgoTrajectoryPlan
    intervention_plan: EgoTrajectoryPlan
    eligibility: InterventionEligibility
    intervention_identity: str
    _integrity_fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        scenario = (
            self.scenario
            if isinstance(self.scenario, ScenarioSnapshot)
            else ScenarioSnapshot.from_scenario(self.scenario)
        )
        baseline = (
            self.baseline
            if isinstance(self.baseline, RolloutSnapshot)
            else RolloutSnapshot.from_rollout(self.baseline)
        )
        intervention = (
            self.intervention
            if isinstance(self.intervention, RolloutSnapshot)
            else RolloutSnapshot.from_rollout(self.intervention)
        )
        if not isinstance(self.baseline_plan, EgoTrajectoryPlan):
            raise TypeError("baseline_plan must be an EgoTrajectoryPlan")
        if not isinstance(self.intervention_plan, EgoTrajectoryPlan):
            raise TypeError("intervention_plan must be an EgoTrajectoryPlan")
        # Canonical round trips create detached immutable plan snapshots and verify
        # the complete plan envelope, including feasibility evidence.
        baseline_plan = EgoTrajectoryPlan.deserialize(
            bytes(self.baseline_plan.serialize())
        )
        intervention_plan = EgoTrajectoryPlan.deserialize(
            bytes(self.intervention_plan.serialize())
        )
        object.__setattr__(self, "scenario", scenario)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "intervention", intervention)
        object.__setattr__(self, "baseline_plan", baseline_plan)
        object.__setattr__(self, "intervention_plan", intervention_plan)
        scenario.revalidate()
        baseline.revalidate()
        intervention.revalidate()
        baseline_plan.revalidate()
        intervention_plan.revalidate()

        if not isinstance(self.eligibility, InterventionEligibility):
            raise TypeError("eligibility must be InterventionEligibility")
        if not self.eligibility.eligible:
            raise ValueError("a CounterfactualPair requires accepted eligibility")
        identity = _nonempty_text(
            self.intervention_identity,
            name="intervention_identity",
        )
        if not _is_canonical_perturbation_identity(identity):
            raise ValueError(
                "intervention_identity must be derived from a typed ego plan"
            )
        object.__setattr__(self, "intervention_identity", identity)
        self._validate_semantics()
        object.__setattr__(
            self,
            "_integrity_fingerprint",
            self._compute_integrity_fingerprint(),
        )

    @property
    def source_scenario(self) -> ScenarioSnapshot:
        return self.scenario

    @property
    def baseline_rollout(self) -> RolloutSnapshot:
        return self.baseline

    @property
    def intervention_rollout(self) -> RolloutSnapshot:
        return self.intervention

    def _validate_semantics(self) -> None:
        scenario = self.scenario
        baseline = self.baseline
        intervention = self.intervention
        baseline_plan = self.baseline_plan
        intervention_plan = self.intervention_plan
        current = self.eligibility.current_index
        stop = self.eligibility.stop_index
        target = self.eligibility.target_index

        source_current = scenario.metadata.get("current_index", 0)
        if (
            isinstance(source_current, bool)
            or not isinstance(source_current, int)
            or source_current != current
        ):
            raise ValueError(
                "eligibility current index does not match source scenario"
            )
        if scenario.scenario_id not in {
            baseline.scenario_id,
            intervention.scenario_id,
        } or baseline.scenario_id != intervention.scenario_id:
            raise ValueError("pair scenario identities do not match")
        if stop >= scenario.num_steps:
            raise ValueError("pair analysis window exceeds source scenario")
        if baseline.num_steps != stop + 1 or intervention.num_steps != stop + 1:
            raise ValueError(
                "typed M6 pair rollouts must end exactly at analysis stop_index"
            )
        expected_timestamps = scenario.timestamps[: stop + 1]
        if not np.array_equal(baseline.timestamps, expected_timestamps) or (
            not np.array_equal(intervention.timestamps, expected_timestamps)
        ):
            raise ValueError("pair timestamps must equal the exact source prefix")
        if baseline.seed != 0 or intervention.seed != 0:
            raise ValueError("M6 paired rollouts require seed 0")
        if (
            baseline.sim_name != intervention.sim_name
            or baseline.sim_version != intervention.sim_version
        ):
            raise ValueError("pair simulator policy/configuration identity drifted")
        baseline_config = _paired_rollout_configuration(
            baseline.metadata,
            side="baseline",
        )
        intervention_config = _paired_rollout_configuration(
            intervention.metadata,
            side="intervention",
        )
        if baseline_config != intervention_config:
            raise ValueError(
                "pair engine, policy, dynamics, or control configuration drifted"
            )
        if (
            baseline_config["policy"].get("name") != baseline.sim_name
            or baseline_config["policy"].get("version")
            != baseline.sim_version
        ):
            raise ValueError(
                "pair policy metadata does not match rollout simulator identity"
            )
        if baseline_config["ego_control"] != "typed_ego_plan":
            raise ValueError("M6 paired rollouts require typed_ego_plan ego control")
        if baseline_config["rollout_start_index"] != current:
            raise ValueError(
                "pair rollout_start_index must equal the eligibility current index"
            )
        expected_world_ids = [
            agent.id
            for index, agent in enumerate(scenario.agents)
            if index != scenario.ego_index
        ]
        if baseline_config["controlled_agent_ids"] != expected_world_ids:
            raise ValueError(
                "pair controlled_agent_ids do not match source world-agent order"
            )
        expected_modes = _expected_agent_control_modes(
            scenario,
            baseline_config["policy"],
        )
        if baseline_config["agent_control_modes"] != expected_modes:
            raise ValueError(
                "pair agent_control_modes do not match the exact source/policy mask"
            )
        if baseline_config["scenario_source"] != scenario.metadata.get(
            "source",
            "unknown",
        ) or baseline_config["scenario_source_fingerprint"] != (
            scenario.metadata.get("source_fingerprint")
        ):
            raise ValueError("pair rollout source provenance drifted")
        if baseline.num_agents != scenario.num_agents or (
            intervention.num_agents != scenario.num_agents
        ):
            raise ValueError("pair agent counts do not match source scenario")
        if target is None or target == scenario.ego_index or (
            target >= scenario.num_agents
        ):
            raise ValueError("frozen target_index must be an in-range world agent")
        for agent_index, (
            source_agent,
            base_agent,
            treatment_agent,
        ) in enumerate(
            zip(
                scenario.agents,
                baseline.agents,
                intervention.agents,
                strict=True,
            )
        ):
            if not _same_agent_contract(
                source_agent,
                base_agent,
                treatment_agent,
                stop_index=stop,
            ):
                raise ValueError(
                    "pair identity, dimensions, type, or lifecycle mask drifted"
                )
            for name in _AGENT_FLOAT_FIELDS:
                source_history = getattr(source_agent, name)[: current + 1]
                if not np.array_equal(
                    getattr(base_agent, name)[: current + 1],
                    source_history,
                ) or not np.array_equal(
                    getattr(treatment_agent, name)[: current + 1],
                    source_history,
                ):
                    raise ValueError("pair logged history/current state drifted")
            if agent_index != scenario.ego_index:
                for name in ("valid", *_AGENT_FLOAT_FIELDS):
                    if not np.array_equal(
                        getattr(base_agent, name)[: current + 2],
                        getattr(treatment_agent, name)[: current + 2],
                    ):
                        raise ValueError(
                            "world-agent treatment response cannot precede t+2"
                        )

        if (
            baseline_plan.spec.family != "identity"
            or baseline_plan.spec.version != "v1"
        ):
            raise ValueError("baseline must use the identity/v1 sham plan")
        if (
            intervention_plan.spec.family != "longitudinal_brake_pulse"
            or intervention_plan.spec.version != "v1"
            or intervention_plan.spec.dose not in (2.0, 4.0)
        ):
            raise ValueError(
                "intervention must use a registered M6 v1 brake-pulse plan"
            )

        # Recompile on a detached copy of the exact source snapshot. This binds both
        # plan tensors and feasibility provenance to the registered compiler.
        from evalsim.perturb.m6 import validate_registered_ego_plan

        source = scenario.to_scenario()
        validate_registered_ego_plan(source, baseline_plan)
        validate_registered_ego_plan(source, intervention_plan)

        if baseline.perturbation != baseline_plan.perturbation_identity:
            raise ValueError(
                "baseline rollout perturbation does not match its exact sham plan"
            )
        if self.intervention_identity != intervention_plan.perturbation_identity:
            raise ValueError(
                "pair intervention identity does not match its exact plan"
            )
        if intervention.perturbation != intervention_plan.perturbation_identity:
            raise ValueError(
                "intervention rollout perturbation does not match its exact plan"
            )

        plan_window = slice(current, stop + 1)
        if not np.array_equal(
            baseline_plan.timestamps,
            scenario.timestamps[plan_window],
        ) or not np.array_equal(
            intervention_plan.timestamps,
            scenario.timestamps[plan_window],
        ):
            raise ValueError("pair plan timestamps drifted from the source window")
        baseline_ego = baseline.agents[scenario.ego_index]
        intervention_ego = intervention.agents[scenario.ego_index]
        for name in ("valid", *_AGENT_FLOAT_FIELDS):
            if not np.array_equal(
                getattr(baseline_ego, name)[plan_window],
                getattr(baseline_plan, name),
            ):
                raise ValueError(
                    "baseline rollout ego tensors do not match the exact sham plan"
                )
            if not np.array_equal(
                getattr(intervention_ego, name)[plan_window],
                getattr(intervention_plan, name),
            ):
                raise ValueError(
                    "intervention rollout ego tensors do not match the exact plan"
                )

    def _compute_integrity_fingerprint(self) -> str:
        return _snapshot_digest(
            (
                self.scenario._integrity_fingerprint.encode("ascii"),
                self.baseline._integrity_fingerprint.encode("ascii"),
                self.intervention._integrity_fingerprint.encode("ascii"),
                self.baseline_plan.serialize(),
                self.intervention_plan.serialize(),
                canonical_configuration_json(
                    self.eligibility.to_dict()
                ).encode("utf-8"),
                self.intervention_identity.encode("ascii"),
            ),
            domain=b"evalsim-counterfactual-pair-v1",
        )

    def revalidate(self) -> None:
        """Revalidate every immutable input immediately before a metric pass."""

        self.scenario.revalidate()
        self.baseline.revalidate()
        self.intervention.revalidate()
        self.baseline_plan.revalidate()
        self.intervention_plan.revalidate()
        self._validate_semantics()
        if self._compute_integrity_fingerprint() != self._integrity_fingerprint:
            raise ValueError("counterfactual pair was mutated")


@dataclass(frozen=True, slots=True)
class PairedMetricResult:
    """One finite, one-scene scalar produced from a complete pair."""

    metric_name: str
    metric_version: str
    scenario_id: str
    intervention_identity: str
    value: float
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        metric_name = _nonempty_text(self.metric_name, name="metric_name")
        if _FAMILY.fullmatch(metric_name) is None:
            raise ValueError("metric_name must use lowercase snake_case")
        metric_version = _nonempty_text(
            self.metric_version,
            name="metric_version",
        )
        if _SEMANTIC_VERSION.fullmatch(metric_version) is None:
            raise ValueError("metric_version must be a semantic version")
        scenario_id = _nonempty_text(self.scenario_id, name="scenario_id")
        identity = _nonempty_text(
            self.intervention_identity,
            name="intervention_identity",
        )
        if not _is_canonical_perturbation_identity(identity):
            raise ValueError(
                "intervention_identity must be derived from a typed ego plan"
            )
        value = _finite_float(self.value, name="value")
        object.__setattr__(self, "metric_name", metric_name)
        object.__setattr__(self, "metric_version", metric_version)
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "intervention_identity", identity)
        object.__setattr__(self, "value", value)
        object.__setattr__(
            self,
            "details",
            _frozen_json_mapping(self.details, name="details"),
        )

    @property
    def metric_id(self) -> str:
        return f"{self.metric_name}@{self.metric_version}"

    def validate_for(
        self,
        pair: CounterfactualPair,
        spec: MetricSpec,
    ) -> None:
        if not isinstance(pair, CounterfactualPair):
            raise TypeError("pair must be a CounterfactualPair")
        if not isinstance(spec, MetricSpec):
            raise TypeError("spec must be a MetricSpec")
        if self.metric_name != spec.name or self.metric_version != spec.version:
            raise ValueError("paired metric result does not match metric spec")
        if self.scenario_id != pair.scenario.scenario_id:
            raise ValueError("paired metric result scenario identity drifted")
        if self.intervention_identity != pair.intervention_identity:
            raise ValueError("paired metric result intervention identity drifted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": M6_PAIRED_METRIC_RESULT_SCHEMA_VERSION,
            "metric_name": self.metric_name,
            "metric_version": self.metric_version,
            "scenario_id": self.scenario_id,
            "intervention_identity": self.intervention_identity,
            "value": self.value,
            "details": _thaw_json(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairedMetricResult":
        expected = frozenset(
            {
                "schema_version",
                "metric_name",
                "metric_version",
                "scenario_id",
                "intervention_identity",
                "value",
                "details",
            }
        )
        _expect_exact_keys(payload, expected, name="PairedMetricResult")
        if payload["schema_version"] != M6_PAIRED_METRIC_RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported paired metric result schema_version")
        return cls(
            metric_name=payload["metric_name"],
            metric_version=payload["metric_version"],
            scenario_id=payload["scenario_id"],
            intervention_identity=payload["intervention_identity"],
            value=payload["value"],
            details=payload["details"],
        )


@runtime_checkable
class PairedMetric(Protocol):
    """Structural protocol for one-scene paired metrics."""

    spec: MetricSpec

    def compute(self, pair: CounterfactualPair) -> PairedMetricResult:
        """Reduce a revalidated complete pair to one scene scalar."""


def evaluate_paired_metric(
    metric: PairedMetric,
    pair: CounterfactualPair,
) -> PairedMetricResult:
    """Run one paired metric with integrity checks on both sides of the pass."""

    if not isinstance(pair, CounterfactualPair):
        raise TypeError("pair must be a CounterfactualPair")
    if not isinstance(metric, PairedMetric):
        raise TypeError("metric must implement the PairedMetric protocol")
    if not isinstance(metric.spec, MetricSpec):
        raise TypeError("paired metric spec must be a MetricSpec")
    pair.revalidate()
    try:
        result = metric.compute(pair)
    except Exception:
        # Preserve the original metric exception only when the metric did not corrupt
        # its supposedly immutable inputs.
        pair.revalidate()
        raise
    pair.revalidate()
    if not isinstance(result, PairedMetricResult):
        raise TypeError("paired metric must return PairedMetricResult")
    result.validate_for(pair, metric.spec)
    return result


__all__ = [
    "AgentTrajectorySnapshot",
    "CONFIGURATION_DOMAIN",
    "CounterfactualPair",
    "EgoInterventionSpec",
    "EgoTrajectoryPlan",
    "FeasibilityAudit",
    "InterventionEligibility",
    "M6_ANALYSIS_TRANSITIONS",
    "M6_PLAN_FRAME_COUNT",
    "M6_PRIMARY_ELIGIBILITY_REASONS",
    "MapPolylineSnapshot",
    "PLAN_DOMAIN",
    "PairedMetric",
    "PairedMetricResult",
    "RolloutSnapshot",
    "ScenarioSnapshot",
    "canonical_configuration_bytes",
    "canonical_configuration_json",
    "canonical_perturbation_identity",
    "evaluate_paired_metric",
]
