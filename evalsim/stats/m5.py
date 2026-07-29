"""Deterministic paired finite-cohort statistics for M5.

The accepted M5 cohort is fixed and fully observed.  Consequently, the summaries in
this module are exact finite-cohort summaries.  The percentile bands deliberately have
the narrower interpretation frozen by the M5 pre-registration: they describe
sensitivity to empirical scenario reweighting and are *not* population confidence
intervals or hypothesis tests.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import re
import struct
from typing import Iterable, Literal

import numpy as np

from evalsim.metrics.m5 import M5_METRIC_SPECS
from evalsim.slices.m5 import (
    M5_SLICE_NAMES as _CANONICAL_SLICE_NAMES,
    M5_SLICE_VERSION,
)


M5_STATISTICS_SCHEMA_VERSION = "m5-paired-statistics-1.0.0"
M5_BASE_SEED = 20260728
M5_POINTWISE_STABILITY_LEVEL = 0.95
M5_PRIMARY_FAMILY_SIZE = 12
M5_PRIMARY_ADJUSTED_STABILITY_LEVEL = 1.0 - 0.05 / M5_PRIMARY_FAMILY_SIZE
M5_PRIMARY_RESAMPLES = 100_000
M5_OTHER_RESAMPLES = 10_000

MetricDirection = Literal["higher", "lower", "neutral"]
_REASON_CODE = re.compile(r"[a-z][a-z0-9_]*")

M5_METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    name: spec.direction  # type: ignore[dict-item]
    for name, spec in M5_METRIC_SPECS.items()
}

M5_PRIMARY_METRIC_NAMES = frozenset(
    {
        "position_error_m",
        "speed_error_mps",
        "oriented_box_overlap_rate",
        "waymax_kinematic_infeasibility_rate",
    }
)

M5_SLICE_NAMES = frozenset(_CANONICAL_SLICE_NAMES)


@dataclass(frozen=True, slots=True)
class PolicyContrast:
    """One frozen ordered M5 policy contrast, interpreted as ``A - B``."""

    policy_a: str
    policy_b: str

    def __post_init__(self) -> None:
        for field_name in ("policy_a", "policy_b"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.policy_a == self.policy_b:
            raise ValueError("an ordered policy contrast requires distinct policies")

    @property
    def label(self) -> str:
        return f"{self.policy_a} - {self.policy_b}"


CONSTANT_VELOCITY_MINUS_LOG_REPLAY = PolicyContrast(
    "constant_velocity",
    "log_replay",
)
IDM_MINUS_LOG_REPLAY = PolicyContrast("idm", "log_replay")
IDM_MINUS_CONSTANT_VELOCITY = PolicyContrast("idm", "constant_velocity")

M5_POLICY_CONTRASTS = (
    CONSTANT_VELOCITY_MINUS_LOG_REPLAY,
    IDM_MINUS_LOG_REPLAY,
    IDM_MINUS_CONSTANT_VELOCITY,
)


@dataclass(frozen=True, slots=True)
class PairedCellSpec:
    """Frozen identity and semantics for one metric × slice × policy-pair cell."""

    metric_name: str
    contrast: PolicyContrast
    slice_name: str = "all"
    metric_version: str = "1.0.0"
    slice_version: str = M5_SLICE_VERSION

    def __post_init__(self) -> None:
        metric_spec = M5_METRIC_SPECS.get(self.metric_name)
        if metric_spec is None:
            raise ValueError(f"unregistered M5 metric {self.metric_name!r}")
        if self.metric_version != metric_spec.version:
            raise ValueError(
                f"M5 metric_version for {self.metric_name!r} must be "
                f"exactly {metric_spec.version!r}"
            )
        if self.slice_name not in M5_SLICE_NAMES:
            raise ValueError(f"unregistered M5 slice {self.slice_name!r}")
        if self.slice_version != M5_SLICE_VERSION:
            raise ValueError(
                f"M5 slice_version must be exactly {M5_SLICE_VERSION!r}"
            )
        if self.contrast not in M5_POLICY_CONTRASTS:
            raise ValueError("contrast must be one of the three frozen M5 contrasts")

    @property
    def direction(self) -> MetricDirection:
        return M5_METRIC_DIRECTIONS[self.metric_name]

    @property
    def value_unit(self) -> str:
        return M5_METRIC_SPECS[self.metric_name].value_unit

    @property
    def invalid_reason_codes(self) -> tuple[str, ...]:
        return M5_METRIC_SPECS[self.metric_name].invalid_reason_codes

    @property
    def is_primary(self) -> bool:
        return (
            self.metric_name in M5_PRIMARY_METRIC_NAMES
            and self.slice_name == "all"
        )

    @property
    def resamples(self) -> int:
        return M5_PRIMARY_RESAMPLES if self.is_primary else M5_OTHER_RESAMPLES


@dataclass(frozen=True, slots=True)
class ScenarioScalar:
    """One source-accounted per-scenario scalar for one policy and metric.

    ``value=None`` represents only pre-registered source missingness and therefore
    requires an ``invalid_reason``.  Execution failures and non-finite values are
    rejected instead of being converted to missing observations.
    """

    cohort_index: int
    value: float | None
    invalid_reason: str | None = None
    eligible_components: int = 1
    total_components: int = 1

    def __post_init__(self) -> None:
        cohort_index = _nonnegative_integer(self.cohort_index, "cohort_index")
        eligible = _nonnegative_integer(
            self.eligible_components,
            "eligible_components",
        )
        total = _nonnegative_integer(self.total_components, "total_components")
        if eligible > total:
            raise ValueError("eligible_components cannot exceed total_components")
        object.__setattr__(self, "cohort_index", cohort_index)
        object.__setattr__(self, "eligible_components", eligible)
        object.__setattr__(self, "total_components", total)

        if self.value is None:
            if (
                not isinstance(self.invalid_reason, str)
                or _REASON_CODE.fullmatch(self.invalid_reason) is None
            ):
                raise ValueError(
                    "a missing scenario scalar requires a lower_snake_case "
                    "invalid_reason"
                )
            if eligible != 0:
                raise ValueError(
                    "a missing scenario scalar must have zero eligible_components"
                )
            return

        if isinstance(self.value, (bool, np.bool_)) or not isinstance(
            self.value,
            Real,
        ):
            raise TypeError("scenario scalar value must be a finite real or None")
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("scenario scalar value must be finite")
        if self.invalid_reason is not None:
            raise ValueError("a valid scenario scalar cannot have an invalid_reason")
        if eligible < 1:
            raise ValueError(
                "a valid scenario scalar requires at least one eligible component"
            )
        object.__setattr__(self, "value", value)

    @property
    def valid(self) -> bool:
        return self.value is not None

    @classmethod
    def missing(
        cls,
        cohort_index: int,
        reason: str,
        *,
        total_components: int = 0,
    ) -> "ScenarioScalar":
        return cls(
            cohort_index=cohort_index,
            value=None,
            invalid_reason=reason,
            eligible_components=0,
            total_components=total_components,
        )


@dataclass(frozen=True, slots=True)
class ResamplingKey:
    """Canonical deterministic identity for one cell's resampling stream."""

    canonical_json: str
    sha256: str
    digest_words: tuple[int, ...]
    paired_n: int
    resamples: int

    def __post_init__(self) -> None:
        paired_n = _nonnegative_integer(self.paired_n, "paired_n")
        resamples = _positive_integer(self.resamples, "resamples")
        if not isinstance(self.canonical_json, str):
            raise TypeError("canonical_json must be a string")
        try:
            payload = json.loads(self.canonical_json)
        except json.JSONDecodeError as exc:
            raise ValueError("canonical_json must contain valid JSON") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("paired_n") != paired_n
            or payload.get("resamples") != resamples
            or payload.get("statistics_schema_version")
            != M5_STATISTICS_SCHEMA_VERSION
        ):
            raise ValueError(
                "canonical_json does not match the resampling key fields"
            )
        recoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        if recoded != self.canonical_json:
            raise ValueError("canonical_json is not in the frozen canonical form")
        digest = hashlib.sha256(self.canonical_json.encode("utf-8")).digest()
        expected_words = tuple(
            int(value) for value in struct.unpack(">8I", digest)
        )
        if self.sha256 != digest.hex():
            raise ValueError("resampling key sha256 does not match canonical_json")
        if tuple(self.digest_words) != expected_words:
            raise ValueError(
                "resampling key digest_words do not match canonical_json"
            )
        object.__setattr__(self, "paired_n", paired_n)
        object.__setattr__(self, "resamples", resamples)
        object.__setattr__(self, "digest_words", expected_words)


@dataclass(frozen=True, slots=True)
class StabilityBand:
    """A descriptive empirical scenario-reweighting percentile band."""

    level: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not 0.0 < self.level < 1.0:
            raise ValueError("stability-band level must lie strictly inside (0, 1)")
        if not all(math.isfinite(value) for value in (self.lower, self.upper)):
            raise ValueError("stability-band endpoints must be finite")
        if self.lower > self.upper:
            raise ValueError("stability-band lower endpoint exceeds upper endpoint")


@dataclass(frozen=True, slots=True)
class PairedCellResult:
    """Complete aggregate result for one frozen paired cell."""

    spec: PairedCellSpec
    cohort_n: int
    valid_a_n: int
    valid_b_n: int
    paired_n: int
    excluded_n: int
    both_missing_n: int
    asymmetric_missing_n: int
    asymmetric_reason_n: int
    asymmetric_component_n: int
    missing_reasons_a: tuple[tuple[str, int], ...]
    missing_reasons_b: tuple[tuple[str, int], ...]
    eligible_components_a: int
    eligible_components_b: int
    total_components_a: int
    total_components_b: int
    source_pairing_complete: bool
    nonzero_effect_n: int | None
    raw_mean_difference: float | None
    raw_median_difference: float | None
    oriented_mean_advantage: float | None
    favorable_proportion: float | None
    standardized_signal_to_heterogeneity: float | None
    policy_a_mean: float | None
    policy_a_median: float | None
    policy_b_mean: float | None
    policy_b_median: float | None
    pointwise_stability_band: StabilityBand | None
    adjusted_primary_stability_band: StabilityBand | None
    status: str
    directional_language_allowed: bool
    resampling_key: ResamplingKey

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-native aggregate without per-scene values or identities."""

        def band_dict(band: StabilityBand | None) -> dict[str, float] | None:
            if band is None:
                return None
            return {
                "level": band.level,
                "lower": band.lower,
                "upper": band.upper,
            }

        return {
            "adjusted_primary_stability_band": band_dict(
                self.adjusted_primary_stability_band
            ),
            "asymmetric_component_n": self.asymmetric_component_n,
            "asymmetric_missing_n": self.asymmetric_missing_n,
            "asymmetric_reason_n": self.asymmetric_reason_n,
            "both_missing_n": self.both_missing_n,
            "cohort_n": self.cohort_n,
            "direction": self.spec.direction,
            "directional_language_allowed": self.directional_language_allowed,
            "eligible_components_a": self.eligible_components_a,
            "eligible_components_b": self.eligible_components_b,
            "excluded_n": self.excluded_n,
            "favorable_proportion": self.favorable_proportion,
            "metric_name": self.spec.metric_name,
            "metric_version": self.spec.metric_version,
            "missing_reasons_a": dict(self.missing_reasons_a),
            "missing_reasons_b": dict(self.missing_reasons_b),
            "nonzero_effect_n": self.nonzero_effect_n,
            "ordered_policy_pair": [
                self.spec.contrast.policy_a,
                self.spec.contrast.policy_b,
            ],
            "oriented_mean_advantage": self.oriented_mean_advantage,
            "paired_n": self.paired_n,
            "pointwise_stability_band": band_dict(
                self.pointwise_stability_band
            ),
            "policy_a_mean": self.policy_a_mean,
            "policy_a_median": self.policy_a_median,
            "policy_b_mean": self.policy_b_mean,
            "policy_b_median": self.policy_b_median,
            "raw_mean_difference": self.raw_mean_difference,
            "raw_median_difference": self.raw_median_difference,
            "resampling": {
                "base_seed": M5_BASE_SEED,
                "canonical_key": self.resampling_key.canonical_json,
                "digest_words": list(self.resampling_key.digest_words),
                "index_dtype": "int64",
                "quantile_method": "linear",
                "resamples": self.resampling_key.resamples,
                "rng": "PCG64",
                "sha256": self.resampling_key.sha256,
                "statistics_schema_version": M5_STATISTICS_SCHEMA_VERSION,
            },
            "slice_name": self.spec.slice_name,
            "slice_version": self.spec.slice_version,
            "source_pairing_complete": self.source_pairing_complete,
            "standardized_signal_to_heterogeneity": (
                self.standardized_signal_to_heterogeneity
            ),
            "status": self.status,
            "total_components_a": self.total_components_a,
            "total_components_b": self.total_components_b,
            "valid_a_n": self.valid_a_n,
            "valid_b_n": self.valid_b_n,
            "value_unit": self.spec.value_unit,
        }

    def to_json(self) -> str:
        """Return canonical byte-stable JSON for identical aggregate inputs."""

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )


def make_resampling_key(
    spec: PairedCellSpec,
    *,
    paired_n: int,
    resamples: int | None = None,
) -> ResamplingKey:
    """Construct the exact canonical key and SHA-256-derived entropy words."""

    paired_n = _nonnegative_integer(paired_n, "paired_n")
    if resamples is None:
        resamples = spec.resamples
    resamples = _positive_integer(resamples, "resamples")
    payload = {
        "metric_name": spec.metric_name,
        "metric_version": spec.metric_version,
        "ordered_policy_pair": [
            spec.contrast.policy_a,
            spec.contrast.policy_b,
        ],
        "paired_n": paired_n,
        "resamples": resamples,
        "slice_name": spec.slice_name,
        "slice_version": spec.slice_version,
        "statistics_schema_version": M5_STATISTICS_SCHEMA_VERSION,
    }
    canonical_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical_json.encode("utf-8")).digest()
    words = tuple(int(value) for value in struct.unpack(">8I", digest))
    return ResamplingKey(
        canonical_json=canonical_json,
        sha256=digest.hex(),
        digest_words=words,
        paired_n=paired_n,
        resamples=resamples,
    )


def draw_resample_indices(key: ResamplingKey) -> np.ndarray:
    """Generate the one exact int64 scenario-index matrix frozen for a cell."""

    if key.paired_n < 1:
        raise ValueError("cannot draw resample indices when paired_n is zero")
    seed_sequence = np.random.SeedSequence(
        [M5_BASE_SEED, *key.digest_words]
    )
    rng = np.random.Generator(np.random.PCG64(seed_sequence))
    return rng.integers(
        0,
        key.paired_n,
        size=(key.resamples, key.paired_n),
        dtype=np.int64,
    )


def analyze_paired_cell(
    spec: PairedCellSpec,
    values_a: Iterable[ScenarioScalar],
    values_b: Iterable[ScenarioScalar],
) -> PairedCellResult:
    """Compute one exact finite-cohort paired summary and stability bands.

    Inputs must enumerate the same unique cohort indices.  Rows are canonicalized by
    ``cohort_index`` before any arithmetic or resampling, so caller order cannot affect
    the result.  Source missingness is retained and accounted for; structural row loss,
    duplicate indices, and malformed/non-finite observations fail closed.
    """

    rows_a = _canonical_rows(values_a, "values_a")
    rows_b = _canonical_rows(values_b, "values_b")
    _validate_missing_reasons(spec, rows_a.values(), "values_a")
    _validate_missing_reasons(spec, rows_b.values(), "values_b")
    if tuple(rows_a) != tuple(rows_b):
        raise ValueError(
            "values_a and values_b must enumerate the same cohort_index set"
        )

    paired_a: list[float] = []
    paired_b: list[float] = []
    both_missing_n = 0
    asymmetric_missing_n = 0
    asymmetric_reason_n = 0
    asymmetric_component_n = 0

    for cohort_index in rows_a:
        row_a = rows_a[cohort_index]
        row_b = rows_b[cohort_index]
        if row_a.valid and row_b.valid:
            paired_a.append(float(row_a.value))
            paired_b.append(float(row_b.value))
        elif not row_a.valid and not row_b.valid:
            both_missing_n += 1
            asymmetric_reason_n += int(
                row_a.invalid_reason != row_b.invalid_reason
            )
        else:
            asymmetric_missing_n += 1
        asymmetric_component_n += int(
            row_a.eligible_components != row_b.eligible_components
            or row_a.total_components != row_b.total_components
        )

    cohort_n = len(rows_a)
    valid_a_n = sum(row.valid for row in rows_a.values())
    valid_b_n = sum(row.valid for row in rows_b.values())
    paired_n = len(paired_a)
    excluded_n = cohort_n - paired_n
    source_pairing_complete = (
        asymmetric_missing_n == 0
        and asymmetric_reason_n == 0
        and asymmetric_component_n == 0
    )
    reason_counts_a = _missing_reason_counts(rows_a.values())
    reason_counts_b = _missing_reason_counts(rows_b.values())
    resampling_key = make_resampling_key(spec, paired_n=paired_n)

    common = {
        "spec": spec,
        "cohort_n": cohort_n,
        "valid_a_n": valid_a_n,
        "valid_b_n": valid_b_n,
        "paired_n": paired_n,
        "excluded_n": excluded_n,
        "both_missing_n": both_missing_n,
        "asymmetric_missing_n": asymmetric_missing_n,
        "asymmetric_reason_n": asymmetric_reason_n,
        "asymmetric_component_n": asymmetric_component_n,
        "missing_reasons_a": reason_counts_a,
        "missing_reasons_b": reason_counts_b,
        "eligible_components_a": sum(
            row.eligible_components for row in rows_a.values()
        ),
        "eligible_components_b": sum(
            row.eligible_components for row in rows_b.values()
        ),
        "total_components_a": sum(
            row.total_components for row in rows_a.values()
        ),
        "total_components_b": sum(
            row.total_components for row in rows_b.values()
        ),
        "source_pairing_complete": source_pairing_complete,
        "resampling_key": resampling_key,
    }

    if paired_n < 10:
        return PairedCellResult(
            **common,
            nonzero_effect_n=None,
            raw_mean_difference=None,
            raw_median_difference=None,
            oriented_mean_advantage=None,
            favorable_proportion=None,
            standardized_signal_to_heterogeneity=None,
            policy_a_mean=None,
            policy_a_median=None,
            policy_b_mean=None,
            policy_b_median=None,
            pointwise_stability_band=None,
            adjusted_primary_stability_band=None,
            status=(
                "insufficient_n"
                if source_pairing_complete
                else "pairing_incomplete"
            ),
            directional_language_allowed=False,
        )

    array_a = np.asarray(paired_a, dtype=np.float64)
    array_b = np.asarray(paired_b, dtype=np.float64)
    raw_effects = array_a - array_b
    if not np.all(np.isfinite(raw_effects)):
        raise ValueError("paired subtraction produced a non-finite effect")
    nonzero_effect_n = int(np.count_nonzero(raw_effects != 0.0))
    raw_mean = _finite_mean(raw_effects)
    raw_median = _finite_median(raw_effects)
    policy_a_mean = _finite_mean(array_a)
    policy_b_mean = _finite_mean(array_b)
    policy_a_median = _finite_median(array_a)
    policy_b_median = _finite_median(array_b)

    oriented_effects: np.ndarray | None
    if spec.direction == "higher":
        oriented_effects = raw_effects
    elif spec.direction == "lower":
        oriented_effects = -raw_effects
    else:
        oriented_effects = None

    if oriented_effects is None:
        oriented_mean = None
        favorable_proportion = None
        standardized = None
    else:
        oriented_mean = _finite_mean(oriented_effects)
        wins = int(np.count_nonzero(oriented_effects > 0.0))
        ties = int(np.count_nonzero(raw_effects == 0.0))
        favorable_proportion = (wins + 0.5 * ties) / paired_n
        standardized = _standardized_signal(
            oriented_effects,
            nonzero_effect_n=nonzero_effect_n,
        )

    draws = draw_resample_indices(resampling_key)
    bootstrap_means = _bootstrap_means(raw_effects, draws)
    pointwise_band = _percentile_band(
        bootstrap_means,
        M5_POINTWISE_STABILITY_LEVEL,
    )
    adjusted_band = (
        _percentile_band(
            bootstrap_means,
            M5_PRIMARY_ADJUSTED_STABILITY_LEVEL,
        )
        if spec.is_primary
        else None
    )

    if not source_pairing_complete:
        status = "pairing_incomplete"
    elif spec.is_primary and nonzero_effect_n < 10:
        status = "event_sparse"
    elif paired_n < 30 or nonzero_effect_n < 10:
        status = "small_or_sparse"
    else:
        status = "descriptive"
    claim_band = adjusted_band if spec.is_primary else pointwise_band
    band_excludes_zero = (
        claim_band is not None
        and (claim_band.upper < 0.0 or claim_band.lower > 0.0)
    )
    directional_allowed = (
        spec.direction != "neutral"
        and paired_n >= 30
        and nonzero_effect_n >= 10
        and source_pairing_complete
        and band_excludes_zero
    )

    return PairedCellResult(
        **common,
        nonzero_effect_n=nonzero_effect_n,
        raw_mean_difference=raw_mean,
        raw_median_difference=raw_median,
        oriented_mean_advantage=oriented_mean,
        favorable_proportion=favorable_proportion,
        standardized_signal_to_heterogeneity=standardized,
        policy_a_mean=policy_a_mean,
        policy_a_median=policy_a_median,
        policy_b_mean=policy_b_mean,
        policy_b_median=policy_b_median,
        pointwise_stability_band=pointwise_band,
        adjusted_primary_stability_band=adjusted_band,
        status=status,
        directional_language_allowed=directional_allowed,
    )


def _canonical_rows(
    rows: Iterable[ScenarioScalar],
    name: str,
) -> dict[int, ScenarioScalar]:
    by_index: dict[int, ScenarioScalar] = {}
    for row in rows:
        if not isinstance(row, ScenarioScalar):
            raise TypeError(f"{name} must contain ScenarioScalar values")
        if row.cohort_index in by_index:
            raise ValueError(
                f"{name} contains duplicate cohort_index {row.cohort_index}"
            )
        by_index[row.cohort_index] = row
    return dict(sorted(by_index.items()))


def _missing_reason_counts(
    rows: Iterable[ScenarioScalar],
) -> tuple[tuple[str, int], ...]:
    counts = Counter(
        row.invalid_reason for row in rows if row.invalid_reason is not None
    )
    return tuple(sorted((str(reason), int(count)) for reason, count in counts.items()))


def _validate_missing_reasons(
    spec: PairedCellSpec,
    rows: Iterable[ScenarioScalar],
    name: str,
) -> None:
    allowed = frozenset(spec.invalid_reason_codes)
    for row in rows:
        if not row.valid and row.invalid_reason not in allowed:
            raise ValueError(
                f"{name} contains unregistered source-missing reason "
                f"{row.invalid_reason!r} for metric {spec.metric_name!r}; "
                "execution failures cannot be converted to missingness"
            )


def _bootstrap_means(
    effects: np.ndarray,
    draws: np.ndarray,
) -> np.ndarray:
    if draws.dtype != np.dtype(np.int64):
        raise TypeError("resampling draw matrix must have dtype int64")
    if draws.ndim != 2 or draws.shape[1] != effects.size:
        raise ValueError("resampling draw matrix has the wrong shape")
    means = np.empty(draws.shape[0], dtype=np.float64)
    # Retain the one frozen full draw matrix while bounding temporary sampled values.
    rows_per_chunk = max(1, min(draws.shape[0], 1_000_000 // effects.size))
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


def _percentile_band(
    bootstrap_values: np.ndarray,
    level: float,
) -> StabilityBand:
    tail = (1.0 - level) / 2.0
    quantiles = np.quantile(
        bootstrap_values,
        [tail, 1.0 - tail],
        method="linear",
    )
    return StabilityBand(
        level=float(level),
        lower=float(quantiles[0]),
        upper=float(quantiles[1]),
    )


def _standardized_signal(
    oriented_effects: np.ndarray,
    *,
    nonzero_effect_n: int,
) -> float | None:
    if oriented_effects.size < 30 or nonzero_effect_n < 10:
        return None
    mean = _finite_mean(oriented_effects)
    squared = math.fsum(
        (float(value) - mean) ** 2 for value in oriented_effects
    )
    variance = squared / (oriented_effects.size - 1)
    if not math.isfinite(variance) or variance <= 0.0:
        return None
    standard_deviation = math.sqrt(variance)
    result = mean / standard_deviation
    return result if math.isfinite(result) else None


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


__all__ = [
    "CONSTANT_VELOCITY_MINUS_LOG_REPLAY",
    "IDM_MINUS_CONSTANT_VELOCITY",
    "IDM_MINUS_LOG_REPLAY",
    "M5_BASE_SEED",
    "M5_METRIC_DIRECTIONS",
    "M5_OTHER_RESAMPLES",
    "M5_POINTWISE_STABILITY_LEVEL",
    "M5_POLICY_CONTRASTS",
    "M5_PRIMARY_ADJUSTED_STABILITY_LEVEL",
    "M5_PRIMARY_FAMILY_SIZE",
    "M5_PRIMARY_METRIC_NAMES",
    "M5_PRIMARY_RESAMPLES",
    "M5_SLICE_NAMES",
    "M5_SLICE_VERSION",
    "M5_STATISTICS_SCHEMA_VERSION",
    "PairedCellResult",
    "PairedCellSpec",
    "PolicyContrast",
    "ResamplingKey",
    "ScenarioScalar",
    "StabilityBand",
    "analyze_paired_cell",
    "draw_resample_indices",
    "make_resampling_key",
]
