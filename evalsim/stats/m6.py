"""Deterministic paired finite-cohort statistics for EvalSim M6.

M6 reduces each complete baseline/treatment scenario pair to one signed effect.
This module summarizes those effects over the fixed eligible cohort and applies the
exact deterministic scenario-reweighting procedure frozen in the M6
pre-registration.  The percentile bands are fixed-cohort reweighting sensitivity
summaries.  They are not confidence intervals, hypothesis tests, p-values, or
population-uncertainty estimates.

The public M6 family is deliberately closed: four registered paired metrics crossed
with three registered NumPy policy/access roles.  Matrix analysis fails closed if a
cell is missing, duplicated, added, asymmetrically paired, or evaluated over a
different cohort.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import re
import struct
from typing import Iterable, Literal

import numpy as np


M6_STATISTICS_SCHEMA_VERSION = "m6-paired-statistics-1.0.0"
M6_BASE_SEED = 20260729
M6_PRIMARY_RESAMPLES = 100_000
M6_PRIMARY_FAMILY_SIZE = 12
M6_POINTWISE_REWEIGHTING_LEVEL = 0.95
M6_ADJUSTED_REWEIGHTING_LEVEL = 1.0 - 0.05 / M6_PRIMARY_FAMILY_SIZE
M6_MAX_PRIMARY_PAIR_N = 128
M6_MAX_SAMPLED_SCALARS_PER_CHUNK = 1_000_000
M6_MAX_PRIMARY_DRAW_MATRIX_BYTES = (
    M6_PRIMARY_RESAMPLES
    * M6_MAX_PRIMARY_PAIR_N
    * np.dtype(np.int64).itemsize
)
M6_REWEIGHTING_INTERPRETATION = "fixed_cohort_reweighting_sensitivity"

M6_RESPONSE_TIMELINESS_METRIC = "response_timeliness_s"

_SEMANTIC_VERSION = re.compile(
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
_NAME = re.compile(r"[a-z][a-z0-9_]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")

M6PrimaryStatus = Literal[
    "event_sparse",
    "small_n",
    "descriptive",
    "direction_supported",
]
M6EffectSign = Literal["positive", "negative"]
M6ResponderStatus = Literal["responder_sparse", "descriptive"]


def _finite_float(value: Real, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_integer(value: int, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True, slots=True)
class M6PrimaryPolicyRole:
    """One frozen NumPy policy and its audited initialization capability."""

    policy_name: str
    access_role: str

    def __post_init__(self) -> None:
        for field_name in ("policy_name", "access_role"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or _NAME.fullmatch(value) is None:
                raise ValueError(f"{field_name} must use lowercase snake_case")


M6_LOG_REPLAY_ROLE = M6PrimaryPolicyRole("log_replay", "privileged")
M6_CONSTANT_VELOCITY_ROLE = M6PrimaryPolicyRole(
    "constant_velocity",
    "history_only",
)
M6_IDM_ROLE = M6PrimaryPolicyRole("idm", "history_only")
M6_PRIMARY_POLICY_ROLES = (
    M6_LOG_REPLAY_ROLE,
    M6_CONSTANT_VELOCITY_ROLE,
    M6_IDM_ROLE,
)


@dataclass(frozen=True, slots=True)
class M6PrimaryMetricIdentity:
    """Frozen identity, unit, and numeric-event rule for one primary metric."""

    metric_name: str
    metric_version: str
    value_unit: str
    nonzero_threshold: float
    use_absolute_threshold: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.metric_name, str)
            or _NAME.fullmatch(self.metric_name) is None
        ):
            raise ValueError("metric_name must use lowercase snake_case")
        if (
            not isinstance(self.metric_version, str)
            or _SEMANTIC_VERSION.fullmatch(self.metric_version) is None
        ):
            raise ValueError("metric_version must be a semantic version")
        if not isinstance(self.value_unit, str) or not self.value_unit:
            raise ValueError("value_unit must be a non-empty string")
        threshold = _finite_float(
            self.nonzero_threshold,
            "nonzero_threshold",
        )
        if threshold < 0.0:
            raise ValueError("nonzero_threshold must be non-negative")
        if type(self.use_absolute_threshold) is not bool:
            raise TypeError("use_absolute_threshold must be a bool")
        object.__setattr__(self, "nonzero_threshold", threshold)

    def is_thresholded_nonzero(self, value: float) -> bool:
        """Apply the exact metric-specific registered numeric-event rule."""

        candidate = abs(value) if self.use_absolute_threshold else value
        return candidate > self.nonzero_threshold


M6_PRIMARY_METRICS = (
    M6PrimaryMetricIdentity(
        "additional_target_braking_impulse_mps",
        "1.0.0",
        "m/s",
        1e-9,
        False,
    ),
    M6PrimaryMetricIdentity(
        M6_RESPONSE_TIMELINESS_METRIC,
        "1.0.0",
        "s",
        1e-9,
        False,
    ),
    M6PrimaryMetricIdentity(
        "minimum_longitudinal_bumper_gap_change_m",
        "1.0.0",
        "m",
        1e-6,
        True,
    ),
    M6PrimaryMetricIdentity(
        "target_progress_loss_m",
        "1.0.0",
        "m",
        1e-6,
        True,
    ),
)

_PRIMARY_METRICS_BY_NAME = {
    metric.metric_name: metric for metric in M6_PRIMARY_METRICS
}
_PRIMARY_ROLES_BY_NAME = {
    role.policy_name: role for role in M6_PRIMARY_POLICY_ROLES
}
_CANONICAL_CELL_IDENTITIES = tuple(
    (
        role.policy_name,
        role.access_role,
        metric.metric_name,
        metric.metric_version,
    )
    for role in M6_PRIMARY_POLICY_ROLES
    for metric in M6_PRIMARY_METRICS
)


@dataclass(frozen=True, slots=True)
class M6PrimaryCellSpec:
    """Identity of one registered primary metric × policy/access-role cell."""

    metric_name: str
    policy_name: str
    policy_access_role: str
    intervention_config_fingerprint: str
    metric_version: str = "1.0.0"

    def __post_init__(self) -> None:
        metric = _PRIMARY_METRICS_BY_NAME.get(self.metric_name)
        if metric is None:
            raise ValueError(f"unregistered M6 primary metric {self.metric_name!r}")
        if self.metric_version != metric.metric_version:
            raise ValueError(
                f"metric_version for {self.metric_name!r} must be exactly "
                f"{metric.metric_version!r}"
            )
        role = _PRIMARY_ROLES_BY_NAME.get(self.policy_name)
        if role is None:
            raise ValueError(f"unregistered M6 primary policy {self.policy_name!r}")
        if self.policy_access_role != role.access_role:
            raise ValueError(
                f"policy_access_role for {self.policy_name!r} must be exactly "
                f"{role.access_role!r}"
            )
        if (
            not isinstance(self.intervention_config_fingerprint, str)
            or _SHA256.fullmatch(self.intervention_config_fingerprint) is None
        ):
            raise ValueError(
                "intervention_config_fingerprint must be a lowercase SHA-256 hex "
                "digest"
            )

    @property
    def metric(self) -> M6PrimaryMetricIdentity:
        return _PRIMARY_METRICS_BY_NAME[self.metric_name]

    @property
    def value_unit(self) -> str:
        return self.metric.value_unit

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return (
            self.policy_name,
            self.policy_access_role,
            self.metric_name,
            self.metric_version,
        )


def m6_primary_cell_specs(
    intervention_config_fingerprint: str,
) -> tuple[M6PrimaryCellSpec, ...]:
    """Return the exact 12 primary cells in frozen policy-major order."""

    return tuple(
        M6PrimaryCellSpec(
            metric_name=metric.metric_name,
            metric_version=metric.metric_version,
            policy_name=role.policy_name,
            policy_access_role=role.access_role,
            intervention_config_fingerprint=intervention_config_fingerprint,
        )
        for role in M6_PRIMARY_POLICY_ROLES
        for metric in M6_PRIMARY_METRICS
    )


@dataclass(frozen=True, slots=True)
class M6SceneEffect:
    """One finite signed baseline/treatment effect for an opaque cohort member.

    ``responded`` and ``responder_latency_s`` are populated only for
    ``response_timeliness_s``.  A censored timeliness observation has exact scalar
    zero, ``responded=False``, and no responder-only latency.  An event at the window
    boundary may legitimately have exact scalar zero while ``responded=True``.
    """

    cohort_index: int
    value: float
    responded: bool | None = None
    responder_latency_s: float | None = None

    def __post_init__(self) -> None:
        cohort_index = _nonnegative_integer(self.cohort_index, "cohort_index")
        value = _finite_float(self.value, "value")
        if self.responded is not None and type(self.responded) is not bool:
            raise TypeError("responded must be a bool or None")
        latency = self.responder_latency_s
        if latency is not None:
            latency = _finite_float(latency, "responder_latency_s")
            if latency < 0.0:
                raise ValueError("responder_latency_s must be non-negative")
        object.__setattr__(self, "cohort_index", cohort_index)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "responder_latency_s", latency)


@dataclass(frozen=True, slots=True)
class M6PrimaryCellInput:
    """Complete ordered input for one registered M6 primary cell."""

    spec: M6PrimaryCellSpec
    scene_effects: tuple[M6SceneEffect, ...]
    source_pairing_complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.spec, M6PrimaryCellSpec):
            raise TypeError("spec must be an M6PrimaryCellSpec")
        if type(self.source_pairing_complete) is not bool:
            raise TypeError("source_pairing_complete must be a bool")
        rows = tuple(self.scene_effects)
        if any(not isinstance(row, M6SceneEffect) for row in rows):
            raise TypeError("scene_effects must contain M6SceneEffect values")
        by_index: dict[int, M6SceneEffect] = {}
        for row in rows:
            if row.cohort_index in by_index:
                raise ValueError(
                    "scene_effects contains duplicate cohort_index "
                    f"{row.cohort_index}"
                )
            by_index[row.cohort_index] = row
        if len(by_index) > M6_MAX_PRIMARY_PAIR_N:
            raise ValueError(
                f"M6 primary pair N cannot exceed {M6_MAX_PRIMARY_PAIR_N}"
            )
        canonical = tuple(by_index[index] for index in sorted(by_index))
        _validate_response_fields(self.spec, canonical)
        object.__setattr__(self, "scene_effects", canonical)

    @property
    def pair_n(self) -> int:
        return len(self.scene_effects)

    @property
    def cohort_indices(self) -> tuple[int, ...]:
        return tuple(row.cohort_index for row in self.scene_effects)


@dataclass(frozen=True, slots=True)
class M6ResamplingKey:
    """Canonical deterministic identity for one primary resampling stream."""

    canonical_json: str
    sha256: str
    digest_words: tuple[int, ...]
    pair_n: int
    resamples: int = M6_PRIMARY_RESAMPLES
    base_seed: int = M6_BASE_SEED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pair_n",
            _positive_integer(self.pair_n, "pair_n"),
        )
        object.__setattr__(
            self,
            "resamples",
            _positive_integer(self.resamples, "resamples"),
        )
        object.__setattr__(
            self,
            "base_seed",
            _positive_integer(self.base_seed, "base_seed"),
        )
        if self.pair_n > M6_MAX_PRIMARY_PAIR_N:
            raise ValueError(
                f"M6 primary pair N cannot exceed {M6_MAX_PRIMARY_PAIR_N}"
            )
        if self.resamples != M6_PRIMARY_RESAMPLES:
            raise ValueError(
                f"M6 primary resamples must be exactly {M6_PRIMARY_RESAMPLES}"
            )
        if self.base_seed != M6_BASE_SEED:
            raise ValueError(f"M6 base_seed must be exactly {M6_BASE_SEED}")
        self.revalidate()

    def revalidate(self) -> None:
        """Fail closed if any frozen key component no longer agrees."""

        if not isinstance(self.canonical_json, str):
            raise TypeError("canonical_json must be a string")
        try:
            payload = json.loads(self.canonical_json)
        except json.JSONDecodeError as exc:
            raise ValueError("canonical_json must contain valid JSON") from exc
        expected_fields = {
            "base_seed",
            "intervention_config_fingerprint",
            "metric_name",
            "metric_version",
            "paired_n",
            "policy_access_role",
            "policy_name",
            "resamples",
            "statistics_schema_version",
        }
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise ValueError("canonical_json fields do not match the M6 key schema")
        for field_name in ("base_seed", "paired_n", "resamples"):
            if type(payload[field_name]) is not int:
                raise ValueError(
                    f"canonical_json {field_name} must be an exact JSON integer"
                )
        if (
            payload["statistics_schema_version"]
            != M6_STATISTICS_SCHEMA_VERSION
            or payload["paired_n"] != self.pair_n
            or payload["resamples"] != self.resamples
            or payload["base_seed"] != self.base_seed
        ):
            raise ValueError("canonical_json does not match resampling key fields")
        # Reconstructing the typed spec validates every identity dimension rather
        # than trusting JSON strings that merely happen to hash.
        M6PrimaryCellSpec(
            metric_name=payload["metric_name"],
            metric_version=payload["metric_version"],
            policy_name=payload["policy_name"],
            policy_access_role=payload["policy_access_role"],
            intervention_config_fingerprint=payload[
                "intervention_config_fingerprint"
            ],
        )
        recoded = _canonical_json(payload)
        if recoded != self.canonical_json:
            raise ValueError("canonical_json is not in the frozen canonical form")
        digest = hashlib.sha256(self.canonical_json.encode("utf-8")).digest()
        expected_words = tuple(int(word) for word in struct.unpack(">8I", digest))
        if self.sha256 != digest.hex():
            raise ValueError("resampling key sha256 does not match canonical_json")
        if tuple(self.digest_words) != expected_words:
            raise ValueError(
                "resampling key digest_words do not match canonical_json"
            )


@dataclass(frozen=True, slots=True)
class M6ReweightingBand:
    """A deterministic fixed-cohort scenario-reweighting percentile band."""

    level: float
    lower: float
    upper: float
    interpretation: str = M6_REWEIGHTING_INTERPRETATION

    def __post_init__(self) -> None:
        level = _finite_float(self.level, "level")
        lower = _finite_float(self.lower, "lower")
        upper = _finite_float(self.upper, "upper")
        if not 0.0 < level < 1.0:
            raise ValueError("reweighting level must lie strictly inside (0, 1)")
        if lower > upper:
            raise ValueError("reweighting-band lower endpoint exceeds upper endpoint")
        if self.interpretation != M6_REWEIGHTING_INTERPRETATION:
            raise ValueError(
                "M6 bands must be labeled fixed-cohort reweighting sensitivity"
            )
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def to_dict(self) -> dict[str, float]:
        """Return the exact numeric band fields used by aggregate rows."""

        return {
            "level": self.level,
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass(frozen=True, slots=True)
class M6ConditionalLatencySummary:
    """Responder-only latency summary with the frozen ten-responder floor."""

    responder_n: int
    censor_n: int
    status: M6ResponderStatus
    arithmetic_mean_s: float | None
    median_s: float | None

    def __post_init__(self) -> None:
        responder_n = _nonnegative_integer(self.responder_n, "responder_n")
        censor_n = _nonnegative_integer(self.censor_n, "censor_n")
        if self.status not in {"responder_sparse", "descriptive"}:
            raise ValueError("unregistered conditional latency status")
        if responder_n < 10:
            if (
                self.status != "responder_sparse"
                or self.arithmetic_mean_s is not None
                or self.median_s is not None
            ):
                raise ValueError(
                    "conditional latency must be suppressed below 10 responders"
                )
        else:
            if (
                self.status != "descriptive"
                or self.arithmetic_mean_s is None
                or self.median_s is None
            ):
                raise ValueError(
                    "conditional latency must be reported from 10 responders"
                )
            arithmetic_mean_s = _finite_float(
                self.arithmetic_mean_s,
                "arithmetic_mean_s",
            )
            median_s = _finite_float(self.median_s, "median_s")
            if arithmetic_mean_s < 0.0 or median_s < 0.0:
                raise ValueError(
                    "conditional latency summaries must be non-negative"
                )
            object.__setattr__(
                self,
                "arithmetic_mean_s",
                arithmetic_mean_s,
            )
            object.__setattr__(self, "median_s", median_s)
        object.__setattr__(self, "responder_n", responder_n)
        object.__setattr__(self, "censor_n", censor_n)

    @property
    def suppression_reason(self) -> str | None:
        if self.status == "responder_sparse":
            return "responder_n_below_10"
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "arithmetic_mean_s": self.arithmetic_mean_s,
            "censor_n": self.censor_n,
            "median_s": self.median_s,
            "responder_n": self.responder_n,
            "status": self.status,
            "suppression_reason": self.suppression_reason,
        }


@dataclass(frozen=True, slots=True)
class M6PrimaryCellResult:
    """Complete aggregate for one frozen M6 primary paired-effect cell."""

    spec: M6PrimaryCellSpec
    pair_n: int
    thresholded_nonzero_n: int
    responder_n: int | None
    censor_n: int | None
    arithmetic_mean: float
    median: float
    pointwise_band: M6ReweightingBand
    adjusted_band: M6ReweightingBand
    status: M6PrimaryStatus
    suppression_reason: str | None
    source_pairing_complete: bool
    directional_language_allowed: bool
    directional_effect_sign: M6EffectSign | None
    conditional_responder_latency: M6ConditionalLatencySummary | None
    resampling_key: M6ResamplingKey

    def __post_init__(self) -> None:
        if not isinstance(self.spec, M6PrimaryCellSpec):
            raise TypeError("spec must be an M6PrimaryCellSpec")
        pair_n = _positive_integer(self.pair_n, "pair_n")
        nonzero_n = _nonnegative_integer(
            self.thresholded_nonzero_n,
            "thresholded_nonzero_n",
        )
        if pair_n < 10:
            raise ValueError("N < 10 blocks M6 primary outcome execution")
        if pair_n > M6_MAX_PRIMARY_PAIR_N or nonzero_n > pair_n:
            raise ValueError("M6 primary result counts are inconsistent")
        if type(self.source_pairing_complete) is not bool:
            raise TypeError("source_pairing_complete must be a bool")
        if not self.source_pairing_complete:
            raise ValueError(
                "a structural/pairing defect fails instead of producing a row"
            )
        if type(self.directional_language_allowed) is not bool:
            raise TypeError("directional_language_allowed must be a bool")
        _finite_float(self.arithmetic_mean, "arithmetic_mean")
        _finite_float(self.median, "median")
        if not isinstance(self.pointwise_band, M6ReweightingBand) or (
            self.pointwise_band.level != M6_POINTWISE_REWEIGHTING_LEVEL
        ):
            raise ValueError("pointwise_band must use the exact M6 pointwise level")
        if not isinstance(self.adjusted_band, M6ReweightingBand) or (
            self.adjusted_band.level != M6_ADJUSTED_REWEIGHTING_LEVEL
        ):
            raise ValueError("adjusted_band must use the exact M6 adjusted level")
        if not isinstance(self.resampling_key, M6ResamplingKey):
            raise TypeError("resampling_key must be an M6ResamplingKey")
        self.resampling_key.revalidate()
        expected_key = make_m6_resampling_key(self.spec, pair_n=pair_n)
        if self.resampling_key != expected_key:
            raise ValueError(
                "resampling key does not match the result cell identity and pair N"
            )

        expected_status, expected_reason = _primary_status(
            pair_n,
            nonzero_n,
            self.adjusted_band,
        )
        if self.status != expected_status or self.suppression_reason != expected_reason:
            raise ValueError("primary status/suppression reason violates priority")
        expected_allowed = expected_status == "direction_supported"
        if self.directional_language_allowed != expected_allowed:
            raise ValueError("directional language flag violates the adjusted-band gate")
        expected_sign: M6EffectSign | None = None
        if expected_allowed:
            expected_sign = (
                "positive" if self.adjusted_band.lower > 0.0 else "negative"
            )
        if self.directional_effect_sign != expected_sign:
            raise ValueError("directional effect sign violates the adjusted-band gate")

        timeliness = self.spec.metric_name == M6_RESPONSE_TIMELINESS_METRIC
        if timeliness:
            if (
                self.responder_n is None
                or self.censor_n is None
                or self.conditional_responder_latency is None
            ):
                raise ValueError(
                    "timeliness rows require responder/censor/latency accounting"
                )
            responder_n = _nonnegative_integer(self.responder_n, "responder_n")
            censor_n = _nonnegative_integer(self.censor_n, "censor_n")
            if responder_n + censor_n != pair_n:
                raise ValueError("responder and censor counts must sum to pair N")
            if (
                self.conditional_responder_latency.responder_n != responder_n
                or self.conditional_responder_latency.censor_n != censor_n
            ):
                raise ValueError("conditional latency counts drifted")
        elif (
            self.responder_n is not None
            or self.censor_n is not None
            or self.conditional_responder_latency is not None
        ):
            raise ValueError(
                "non-timeliness rows require null responder/censor/latency fields"
            )

        object.__setattr__(self, "pair_n", pair_n)
        object.__setattr__(self, "thresholded_nonzero_n", nonzero_n)

    @property
    def source_pairing_flag(self) -> bool:
        """Alias matching the promoted-schema wording."""

        return self.source_pairing_complete

    def to_promoted_dict(self) -> dict[str, object]:
        """Return the exact sanitized primary-row fields from plan §11.4.

        Configuration and result digests are deliberately absent.  They remain local
        mutation-detection provenance and are never public evidence.
        """

        return {
            "adjusted_band": self.adjusted_band.to_dict(),
            "arithmetic_mean": self.arithmetic_mean,
            "censor_n": self.censor_n,
            "directional_language_allowed": self.directional_language_allowed,
            "median": self.median,
            "metric_name": self.spec.metric_name,
            "metric_version": self.spec.metric_version,
            "pair_n": self.pair_n,
            "pointwise_band": self.pointwise_band.to_dict(),
            "policy_access_role": self.spec.policy_access_role,
            "policy_name": self.spec.policy_name,
            "responder_n": self.responder_n,
            "source_pairing_complete": self.source_pairing_complete,
            "status": self.status,
            "suppression_reason": self.suppression_reason,
            "thresholded_nonzero_n": self.thresholded_nonzero_n,
            "unit": self.spec.value_unit,
        }

    def to_local_dict(self) -> dict[str, object]:
        """Return aggregate-only local evidence including resampling provenance."""

        return {
            **self.to_promoted_dict(),
            "conditional_responder_latency": (
                None
                if self.conditional_responder_latency is None
                else self.conditional_responder_latency.to_dict()
            ),
            "directional_effect_sign": self.directional_effect_sign,
            "resampling": {
                "base_seed": self.resampling_key.base_seed,
                "canonical_cell_key": self.resampling_key.canonical_json,
                "digest_words": list(self.resampling_key.digest_words),
                "index_dtype": "int64",
                "interpretation": M6_REWEIGHTING_INTERPRETATION,
                "quantile_method": "linear",
                "resamples": self.resampling_key.resamples,
                "rng": "PCG64",
                "sha256": self.resampling_key.sha256,
                "statistics_schema_version": M6_STATISTICS_SCHEMA_VERSION,
            },
        }


@dataclass(frozen=True, slots=True)
class M6PrimaryMatrixResult:
    """The exact complete 12-cell M6 primary matrix in canonical order."""

    rows: tuple[M6PrimaryCellResult, ...]
    pair_n: int
    intervention_config_fingerprint: str
    evalsim_idm_response_event_expectation_met: bool

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if len(rows) != M6_PRIMARY_FAMILY_SIZE or any(
            not isinstance(row, M6PrimaryCellResult) for row in rows
        ):
            raise ValueError("M6 primary matrix must contain exactly 12 result rows")
        if tuple(row.spec.identity for row in rows) != _CANONICAL_CELL_IDENTITIES:
            raise ValueError("M6 primary matrix rows are not in canonical order")
        pair_n = _positive_integer(self.pair_n, "pair_n")
        if any(row.pair_n != pair_n for row in rows):
            raise ValueError("M6 primary matrix rows must share pair N")
        if (
            not isinstance(self.intervention_config_fingerprint, str)
            or _SHA256.fullmatch(self.intervention_config_fingerprint) is None
            or any(
                row.spec.intervention_config_fingerprint
                != self.intervention_config_fingerprint
                for row in rows
            )
        ):
            raise ValueError(
                "M6 primary matrix rows must share one intervention fingerprint"
            )
        if type(self.evalsim_idm_response_event_expectation_met) is not bool:
            raise TypeError(
                "evalsim_idm_response_event_expectation_met must be a bool"
            )
        expected = next(
            row
            for row in rows
            if row.spec.policy_name == "idm"
            and row.spec.metric_name == M6_RESPONSE_TIMELINESS_METRIC
        )
        if self.evalsim_idm_response_event_expectation_met != (
            expected.responder_n is not None and expected.responder_n >= 10
        ):
            raise ValueError("EvalSim IDM response-event expectation flag drifted")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "pair_n", pair_n)

    def to_promoted_rows(self) -> tuple[dict[str, object], ...]:
        """Return exactly 12 sanitized rows without local hashes or scene data."""

        return tuple(row.to_promoted_dict() for row in self.rows)

    def to_local_dict(self) -> dict[str, object]:
        return {
            "base_seed": M6_BASE_SEED,
            "evalsim_idm_response_event_expectation_met": (
                self.evalsim_idm_response_event_expectation_met
            ),
            "interpretation": M6_REWEIGHTING_INTERPRETATION,
            "pair_n": self.pair_n,
            "primary_family_size": M6_PRIMARY_FAMILY_SIZE,
            "rows": [row.to_local_dict() for row in self.rows],
            "statistics_schema_version": M6_STATISTICS_SCHEMA_VERSION,
        }


def make_m6_resampling_key(
    spec: M6PrimaryCellSpec,
    *,
    pair_n: int,
) -> M6ResamplingKey:
    """Build the exact canonical M6 cell key and SHA-256 entropy words."""

    if not isinstance(spec, M6PrimaryCellSpec):
        raise TypeError("spec must be an M6PrimaryCellSpec")
    pair_n = _positive_integer(pair_n, "pair_n")
    if pair_n > M6_MAX_PRIMARY_PAIR_N:
        raise ValueError(
            f"M6 primary pair N cannot exceed {M6_MAX_PRIMARY_PAIR_N}"
        )
    payload = {
        "base_seed": M6_BASE_SEED,
        "intervention_config_fingerprint": (
            spec.intervention_config_fingerprint
        ),
        "metric_name": spec.metric_name,
        "metric_version": spec.metric_version,
        "paired_n": pair_n,
        "policy_access_role": spec.policy_access_role,
        "policy_name": spec.policy_name,
        "resamples": M6_PRIMARY_RESAMPLES,
        "statistics_schema_version": M6_STATISTICS_SCHEMA_VERSION,
    }
    canonical_json = _canonical_json(payload)
    digest = hashlib.sha256(canonical_json.encode("utf-8")).digest()
    words = tuple(int(word) for word in struct.unpack(">8I", digest))
    return M6ResamplingKey(
        canonical_json=canonical_json,
        sha256=digest.hex(),
        digest_words=words,
        pair_n=pair_n,
    )


def draw_m6_resample_indices(key: M6ResamplingKey) -> np.ndarray:
    """Draw the one exact ``int64[100000, N]`` matrix for a primary cell."""

    if not isinstance(key, M6ResamplingKey):
        raise TypeError("key must be an M6ResamplingKey")
    key.revalidate()
    seed_sequence = np.random.SeedSequence(
        [M6_BASE_SEED, *key.digest_words]
    )
    rng = np.random.Generator(np.random.PCG64(seed_sequence))
    return rng.integers(
        0,
        key.pair_n,
        size=(M6_PRIMARY_RESAMPLES, key.pair_n),
        dtype=np.int64,
    )


def m6_resampled_means(
    effects: np.ndarray,
    draws: np.ndarray,
) -> np.ndarray:
    """Compute float64 sampled means with at most one million sampled scalars.

    The exact primary path supplies the full matrix from
    :func:`draw_m6_resample_indices`.  Accepting a generic row count here keeps the
    chunking primitive independently testable without weakening the exact draw API.
    """

    effects = np.asarray(effects)
    if effects.dtype != np.dtype(np.float64) or effects.ndim != 1:
        raise TypeError("effects must be a one-dimensional float64 array")
    if effects.size < 1 or not np.all(np.isfinite(effects)):
        raise ValueError("effects must be a non-empty finite array")
    if not isinstance(draws, np.ndarray) or draws.dtype != np.dtype(np.int64):
        raise TypeError("draws must be an int64 ndarray")
    if draws.ndim != 2 or draws.shape[1] != effects.size:
        raise ValueError("draw matrix has the wrong shape")
    if draws.size and (
        int(np.min(draws)) < 0 or int(np.max(draws)) >= effects.size
    ):
        raise ValueError("draw matrix contains an out-of-range scenario index")

    means = np.empty(draws.shape[0], dtype=np.float64)
    rows_per_chunk = max(
        1,
        min(
            draws.shape[0],
            M6_MAX_SAMPLED_SCALARS_PER_CHUNK // effects.size,
        ),
    )
    for start in range(0, draws.shape[0], rows_per_chunk):
        stop = min(draws.shape[0], start + rows_per_chunk)
        means[start:stop] = np.mean(
            effects[draws[start:stop]],
            axis=1,
            dtype=np.float64,
        )
    if not np.all(np.isfinite(means)):
        raise ValueError("scenario resampling produced a non-finite mean")
    return means


def analyze_m6_primary_cell(
    spec_or_input: M6PrimaryCellSpec | M6PrimaryCellInput,
    scene_effects: Iterable[M6SceneEffect] | None = None,
    *,
    source_pairing_complete: bool = True,
) -> M6PrimaryCellResult:
    """Analyze one complete registered M6 primary cell.

    Callers may pass an :class:`M6PrimaryCellInput`, or a spec plus scene effects.
    Caller order cannot affect the result: effects are canonicalized by opaque
    ``cohort_index`` before arithmetic or resampling.
    """

    cell = _coerce_cell_input(
        spec_or_input,
        scene_effects,
        source_pairing_complete=source_pairing_complete,
    )
    if not cell.source_pairing_complete:
        raise ValueError(
            "a structural/pairing defect fails instead of producing a row"
        )
    if cell.pair_n < 10:
        raise ValueError("N < 10 blocks M6 primary outcome execution")

    values = np.asarray(
        [row.value for row in cell.scene_effects],
        dtype=np.float64,
    )
    arithmetic_mean = _finite_mean(values)
    median = _finite_median(values)
    thresholded_nonzero_n = sum(
        cell.spec.metric.is_thresholded_nonzero(float(value))
        for value in values
    )

    resampling_key = make_m6_resampling_key(cell.spec, pair_n=cell.pair_n)
    draws = draw_m6_resample_indices(resampling_key)
    sampled_means = m6_resampled_means(values, draws)
    pointwise_band = _reweighting_band(
        sampled_means,
        M6_POINTWISE_REWEIGHTING_LEVEL,
    )
    adjusted_band = _reweighting_band(
        sampled_means,
        M6_ADJUSTED_REWEIGHTING_LEVEL,
    )
    status, suppression_reason = _primary_status(
        cell.pair_n,
        thresholded_nonzero_n,
        adjusted_band,
    )
    directional_allowed = status == "direction_supported"
    directional_sign: M6EffectSign | None = None
    if directional_allowed:
        directional_sign = (
            "positive" if adjusted_band.lower > 0.0 else "negative"
        )

    conditional_latency: M6ConditionalLatencySummary | None = None
    responder_n: int | None = None
    censor_n: int | None = None
    if cell.spec.metric_name == M6_RESPONSE_TIMELINESS_METRIC:
        responder_rows = tuple(
            row for row in cell.scene_effects if row.responded is True
        )
        responder_n = len(responder_rows)
        censor_n = cell.pair_n - responder_n
        if responder_n < 10:
            conditional_latency = M6ConditionalLatencySummary(
                responder_n=responder_n,
                censor_n=censor_n,
                status="responder_sparse",
                arithmetic_mean_s=None,
                median_s=None,
            )
        else:
            latencies = np.asarray(
                [
                    float(row.responder_latency_s)
                    for row in responder_rows
                    if row.responder_latency_s is not None
                ],
                dtype=np.float64,
            )
            # Input validation requires one latency for every responder.
            if latencies.size != responder_n:
                raise ValueError("responder-only latency accounting is incomplete")
            conditional_latency = M6ConditionalLatencySummary(
                responder_n=responder_n,
                censor_n=censor_n,
                status="descriptive",
                arithmetic_mean_s=_finite_mean(latencies),
                median_s=_finite_median(latencies),
            )

    return M6PrimaryCellResult(
        spec=cell.spec,
        pair_n=cell.pair_n,
        thresholded_nonzero_n=thresholded_nonzero_n,
        responder_n=responder_n,
        censor_n=censor_n,
        arithmetic_mean=arithmetic_mean,
        median=median,
        pointwise_band=pointwise_band,
        adjusted_band=adjusted_band,
        status=status,
        suppression_reason=suppression_reason,
        source_pairing_complete=True,
        directional_language_allowed=directional_allowed,
        directional_effect_sign=directional_sign,
        conditional_responder_latency=conditional_latency,
        resampling_key=resampling_key,
    )


def analyze_m6_primary_matrix(
    cells: Iterable[M6PrimaryCellInput],
) -> M6PrimaryMatrixResult:
    """Validate and analyze exactly three roles × four metrics.

    Validation of all 12 identities, one shared intervention fingerprint, and one
    exact cohort occurs before the first resampling draw.
    """

    cells_tuple = tuple(cells)
    if any(not isinstance(cell, M6PrimaryCellInput) for cell in cells_tuple):
        raise TypeError("cells must contain M6PrimaryCellInput values")
    if len(cells_tuple) != M6_PRIMARY_FAMILY_SIZE:
        raise ValueError("M6 primary matrix requires exactly 12 input cells")

    by_identity: dict[tuple[str, str, str, str], M6PrimaryCellInput] = {}
    for cell in cells_tuple:
        if cell.spec.identity in by_identity:
            raise ValueError(
                f"duplicate M6 primary cell identity {cell.spec.identity!r}"
            )
        by_identity[cell.spec.identity] = cell
    actual = set(by_identity)
    expected = set(_CANONICAL_CELL_IDENTITIES)
    if actual != expected:
        raise ValueError(
            "M6 primary matrix cell identities are incomplete or unexpected"
        )

    canonical_cells = tuple(
        by_identity[identity] for identity in _CANONICAL_CELL_IDENTITIES
    )
    fingerprints = {
        cell.spec.intervention_config_fingerprint for cell in canonical_cells
    }
    if len(fingerprints) != 1:
        raise ValueError(
            "M6 primary matrix cells must share one intervention fingerprint"
        )
    if any(not cell.source_pairing_complete for cell in canonical_cells):
        raise ValueError(
            "a structural/pairing defect fails instead of producing a matrix"
        )
    cohort_indices = canonical_cells[0].cohort_indices
    if any(cell.cohort_indices != cohort_indices for cell in canonical_cells[1:]):
        raise ValueError(
            "M6 primary matrix cells must contain the same complete cohort indices"
        )
    if len(cohort_indices) < 10:
        raise ValueError("N < 10 blocks M6 primary outcome execution")

    rows = tuple(analyze_m6_primary_cell(cell) for cell in canonical_cells)
    idm_timeliness = next(
        row
        for row in rows
        if row.spec.policy_name == "idm"
        and row.spec.metric_name == M6_RESPONSE_TIMELINESS_METRIC
    )
    expectation_met = (
        idm_timeliness.responder_n is not None
        and idm_timeliness.responder_n >= 10
    )
    return M6PrimaryMatrixResult(
        rows=rows,
        pair_n=len(cohort_indices),
        intervention_config_fingerprint=next(iter(fingerprints)),
        evalsim_idm_response_event_expectation_met=expectation_met,
    )


def _coerce_cell_input(
    spec_or_input: M6PrimaryCellSpec | M6PrimaryCellInput,
    scene_effects: Iterable[M6SceneEffect] | None,
    *,
    source_pairing_complete: bool,
) -> M6PrimaryCellInput:
    if isinstance(spec_or_input, M6PrimaryCellInput):
        if scene_effects is not None:
            raise TypeError(
                "scene_effects must be omitted with M6PrimaryCellInput"
            )
        if source_pairing_complete is not True:
            raise TypeError(
                "source_pairing_complete belongs to M6PrimaryCellInput"
            )
        return spec_or_input
    if not isinstance(spec_or_input, M6PrimaryCellSpec):
        raise TypeError(
            "first argument must be M6PrimaryCellSpec or M6PrimaryCellInput"
        )
    if scene_effects is None:
        raise TypeError("scene_effects are required with M6PrimaryCellSpec")
    return M6PrimaryCellInput(
        spec=spec_or_input,
        scene_effects=tuple(scene_effects),
        source_pairing_complete=source_pairing_complete,
    )


def _validate_response_fields(
    spec: M6PrimaryCellSpec,
    rows: tuple[M6SceneEffect, ...],
) -> None:
    timeliness = spec.metric_name == M6_RESPONSE_TIMELINESS_METRIC
    for row in rows:
        if timeliness:
            if row.responded is None:
                raise ValueError(
                    "timeliness effects require explicit responded accounting"
                )
            if row.responded:
                if row.responder_latency_s is None:
                    raise ValueError(
                        "responders require responder_latency_s"
                    )
            else:
                if row.responder_latency_s is not None:
                    raise ValueError(
                        "censored observations cannot have responder_latency_s"
                    )
                if row.value != 0.0:
                    raise ValueError(
                        "a censored timeliness effect must be exact zero"
                    )
        elif row.responded is not None or row.responder_latency_s is not None:
            raise ValueError(
                "response metadata is only valid for response_timeliness_s"
            )


def _primary_status(
    pair_n: int,
    thresholded_nonzero_n: int,
    adjusted_band: M6ReweightingBand,
) -> tuple[M6PrimaryStatus, str | None]:
    """Apply the exact registered priority after structural and N gates."""

    if thresholded_nonzero_n < 10:
        return "event_sparse", "thresholded_nonzero_n_below_10"
    if pair_n < 30:
        return "small_n", "pair_n_below_30"
    if adjusted_band.lower <= 0.0 <= adjusted_band.upper:
        return "descriptive", "adjusted_band_contains_zero"
    return "direction_supported", None


def _reweighting_band(
    sampled_means: np.ndarray,
    level: float,
) -> M6ReweightingBand:
    tail = (1.0 - level) / 2.0
    quantiles = np.quantile(
        sampled_means,
        [tail, 1.0 - tail],
        method="linear",
    )
    return M6ReweightingBand(
        level=float(level),
        lower=float(quantiles[0]),
        upper=float(quantiles[1]),
    )


def _finite_mean(values: np.ndarray) -> float:
    if values.size < 1:
        raise ValueError("cannot take the mean of an empty finite cohort")
    result = math.fsum(float(value) for value in values) / values.size
    if not math.isfinite(result):
        raise ValueError("finite-cohort mean is non-finite")
    return result


def _finite_median(values: np.ndarray) -> float:
    if values.size < 1:
        raise ValueError("cannot take the median of an empty finite cohort")
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    result = math.fsum((ordered[midpoint - 1], ordered[midpoint])) / 2.0
    if not math.isfinite(result):
        raise ValueError("finite-cohort median is non-finite")
    return result


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


__all__ = [
    "M6_ADJUSTED_REWEIGHTING_LEVEL",
    "M6_BASE_SEED",
    "M6_CONSTANT_VELOCITY_ROLE",
    "M6_IDM_ROLE",
    "M6_LOG_REPLAY_ROLE",
    "M6_MAX_PRIMARY_DRAW_MATRIX_BYTES",
    "M6_MAX_PRIMARY_PAIR_N",
    "M6_MAX_SAMPLED_SCALARS_PER_CHUNK",
    "M6_POINTWISE_REWEIGHTING_LEVEL",
    "M6_PRIMARY_FAMILY_SIZE",
    "M6_PRIMARY_METRICS",
    "M6_PRIMARY_POLICY_ROLES",
    "M6_PRIMARY_RESAMPLES",
    "M6_REWEIGHTING_INTERPRETATION",
    "M6_RESPONSE_TIMELINESS_METRIC",
    "M6_STATISTICS_SCHEMA_VERSION",
    "M6ConditionalLatencySummary",
    "M6PrimaryCellInput",
    "M6PrimaryCellResult",
    "M6PrimaryCellSpec",
    "M6PrimaryMatrixResult",
    "M6PrimaryMetricIdentity",
    "M6PrimaryPolicyRole",
    "M6ResamplingKey",
    "M6ReweightingBand",
    "M6SceneEffect",
    "analyze_m6_primary_cell",
    "analyze_m6_primary_matrix",
    "draw_m6_resample_indices",
    "m6_primary_cell_specs",
    "m6_resampled_means",
    "make_m6_resampling_key",
]
