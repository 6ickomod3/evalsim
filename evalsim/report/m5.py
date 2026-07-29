"""Deterministic, aggregate-only human-readable rendering for M5 scorecards.

The renderer deliberately accepts only the fixed scorecard schema.  It does not
accept a title, run path, provenance, scenario identifiers, or free-form notes, so
those values cannot accidentally cross from local result storage into the report.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import math
from numbers import Integral, Real
from typing import Any

from evalsim.metrics.m5 import M5_METRIC_SPECS
from evalsim.slices.m5 import M5_SLICE_SPECS, M5_SLICE_VERSION
from evalsim.stats.m5 import (
    M5_BASE_SEED,
    M5_POLICY_CONTRASTS,
    M5_POINTWISE_STABILITY_LEVEL,
    M5_PRIMARY_ADJUSTED_STABILITY_LEVEL,
    PairedCellSpec,
    PolicyContrast,
    make_resampling_key,
)


M5_SCORECARD_RENDERER_VERSION = "m5-scorecard-markdown-1.0.0"
M5_SCORECARD_SOURCE_SCHEMA_VERSION = "m5-result-store-1.0.0"
M5_SCORECARD_REPORT_PATH = "scorecard.md"
M5_SCORECARD_REPORT_MEDIA_TYPE = "text/markdown; charset=utf-8"

OFFICIAL_M5_PROFILE = "official_m5"
DATA_FREE_TEST_PROFILE = "data_free_test"

M5_SCORECARD_INPUT_FIELDS = frozenset(
    {
        "metric_name",
        "metric_version",
        "value_unit",
        "direction",
        "slice_name",
        "slice_version",
        "policy_a",
        "policy_b",
        "cohort_n",
        "valid_a_n",
        "valid_b_n",
        "paired_n",
        "excluded_n",
        "both_missing_n",
        "asymmetric_missing_n",
        "asymmetric_reason_n",
        "asymmetric_component_n",
        "missing_reasons_a_json",
        "missing_reasons_b_json",
        "eligible_components_a",
        "eligible_components_b",
        "total_components_a",
        "total_components_b",
        "source_pairing_complete",
        "nonzero_effect_n",
        "raw_mean_difference",
        "raw_median_difference",
        "oriented_mean_advantage",
        "favorable_proportion",
        "standardized_signal_to_heterogeneity",
        "policy_a_mean",
        "policy_a_median",
        "policy_b_mean",
        "policy_b_median",
        "pointwise_level",
        "pointwise_lower",
        "pointwise_upper",
        "adjusted_level",
        "adjusted_lower",
        "adjusted_upper",
        "status",
        "directional_language_allowed",
        "resampling_key_json",
        "resampling_sha256",
        "resampling_digest_words",
        "resamples",
        "base_seed",
        "rng",
        "index_dtype",
        "quantile_method",
    }
)

# These are the only per-row values permitted to enter the Markdown bytes.  The
# three resampling-substream identity fields are validated by the result store but
# intentionally excluded from this presentation boundary.
M5_SCORECARD_RENDERED_ROW_FIELDS = M5_SCORECARD_INPUT_FIELDS.difference(
    {
        "resampling_key_json",
        "resampling_sha256",
        "resampling_digest_words",
        "base_seed",
        "rng",
        "index_dtype",
        "quantile_method",
    }
)

_COUNT_FIELDS = (
    "cohort_n",
    "valid_a_n",
    "valid_b_n",
    "paired_n",
    "excluded_n",
    "both_missing_n",
    "asymmetric_missing_n",
    "asymmetric_reason_n",
    "asymmetric_component_n",
    "eligible_components_a",
    "eligible_components_b",
    "total_components_a",
    "total_components_b",
    "resamples",
    "base_seed",
)
_OPTIONAL_REAL_FIELDS = (
    "raw_mean_difference",
    "raw_median_difference",
    "oriented_mean_advantage",
    "favorable_proportion",
    "standardized_signal_to_heterogeneity",
    "policy_a_mean",
    "policy_a_median",
    "policy_b_mean",
    "policy_b_median",
    "pointwise_level",
    "pointwise_lower",
    "pointwise_upper",
    "adjusted_level",
    "adjusted_lower",
    "adjusted_upper",
)
_SLICE_NAMES = frozenset(spec.name for spec in M5_SLICE_SPECS)
_CONTRASTS = frozenset(
    (contrast.policy_a, contrast.policy_b)
    for contrast in M5_POLICY_CONTRASTS
)
_KEY_FIELDS = (
    "metric_name",
    "metric_version",
    "slice_name",
    "slice_version",
    "policy_a",
    "policy_b",
)


def render_m5_scorecard(
    rows: Iterable[Mapping[str, Any]],
    *,
    row_accounting_profile: str,
) -> bytes:
    """Render fixed M5 scorecard rows into deterministic UTF-8 Markdown bytes."""

    if row_accounting_profile not in {
        OFFICIAL_M5_PROFILE,
        DATA_FREE_TEST_PROFILE,
    }:
        raise ValueError("row_accounting_profile is not a recognized M5 profile")
    normalized = _normalize_rows(rows)
    keys = {
        tuple(row[field] for field in _KEY_FIELDS)
        for row in normalized
    }
    if len(keys) != len(normalized):
        raise ValueError("scorecard rows contain a duplicate canonical key")
    if row_accounting_profile == OFFICIAL_M5_PROFILE:
        expected = {
            (
                metric_name,
                metric_spec.version,
                slice_name,
                M5_SLICE_VERSION,
                contrast.policy_a,
                contrast.policy_b,
            )
            for metric_name, metric_spec in M5_METRIC_SPECS.items()
            for slice_name in _SLICE_NAMES
            for contrast in M5_POLICY_CONTRASTS
        }
        if keys != expected:
            raise ValueError(
                "official M5 rendering requires the exact 312-cell key domain"
            )

    lines = [
        "# EvalSim M5 scorecard",
        "",
        f"- Renderer: `{M5_SCORECARD_RENDERER_VERSION}`",
        f"- Source schema: `{M5_SCORECARD_SOURCE_SCHEMA_VERSION}`",
        f"- Row-accounting profile: `{row_accounting_profile}`",
        (
            f"- Frozen resampling: base seed `{M5_BASE_SEED}`, RNG `PCG64`, "
            "index dtype `int64`, quantile method `linear`."
        ),
        "",
        "## Interpretation boundary",
        "",
        (
            "- This is an aggregate local rendering of a manifest-bound scorecard "
            "table. An `official_m5` label is evidence only when the enclosing "
            "result store independently verifies."
        ),
        (
            "- Results are conditional on the accepted finite cohort and registered "
            "source-only slices; they do not estimate the WOMD population."
        ),
        (
            "- Policy comparisons use shared source decoding. `log_replay` is a "
            "privileged logged-future reference, not independent ground truth."
        ),
        (
            "- Stability bands describe empirical scenario reweighting. They are "
            "not population confidence intervals or hypothesis tests."
        ),
        (
            "- Rows use lexical metric/slice/policy order. This report makes no "
            "winner, ranking, or cross-metric aggregate claim."
        ),
        "",
    ]
    if not normalized:
        lines.extend(("No scorecard rows.", ""))
        return ("\n".join(lines)).encode("utf-8")

    grouped: dict[
        tuple[str, str, str, str],
        dict[tuple[str, str], list[dict[str, Any]]],
    ] = {}
    for row in normalized:
        metric_key = (
            row["metric_name"],
            row["metric_version"],
            row["value_unit"],
            row["direction"],
        )
        slice_key = (row["slice_name"], row["slice_version"])
        grouped.setdefault(metric_key, {}).setdefault(slice_key, []).append(row)

    for metric_key in sorted(grouped):
        metric_name, metric_version, value_unit, direction = metric_key
        lines.extend(
            (
                f"## Metric: {_markdown(metric_name)}",
                "",
                (
                    f"Version `{_markdown(metric_version)}`; unit "
                    f"`{_markdown(value_unit)}`; registered direction "
                    f"`{_markdown(direction)}`."
                ),
                "",
            )
        )
        for slice_key in sorted(grouped[metric_key]):
            slice_name, slice_version = slice_key
            slice_rows = sorted(
                grouped[metric_key][slice_key],
                key=lambda row: (row["policy_a"], row["policy_b"]),
            )
            lines.extend(
                (
                    f"### Slice: {_markdown(slice_name)}",
                    "",
                    f"Slice version `{_markdown(slice_version)}`.",
                    "",
                    "#### Effects and coverage",
                    "",
                    (
                        "| Policy A | Policy B | Cohort | Valid A | Valid B | "
                        "Paired | Excluded | Nonzero | Mean A | Median A | Mean B | "
                        "Median B | Mean(A-B) | Median(A-B) | Oriented advantage | "
                        "Favorable proportion | Standardized signal/heterogeneity | "
                        "Pointwise band | Adjusted band | Resamples | Status | "
                        "Directional language |"
                    ),
                    (
                        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
                        "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                        "---: | --- | --- | ---: | --- | --- |"
                    ),
                )
            )
            for row in slice_rows:
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            _markdown(row["policy_a"]),
                            _markdown(row["policy_b"]),
                            _scalar(row["cohort_n"]),
                            _scalar(row["valid_a_n"]),
                            _scalar(row["valid_b_n"]),
                            _scalar(row["paired_n"]),
                            _scalar(row["excluded_n"]),
                            _scalar(row["nonzero_effect_n"]),
                            _scalar(row["policy_a_mean"]),
                            _scalar(row["policy_a_median"]),
                            _scalar(row["policy_b_mean"]),
                            _scalar(row["policy_b_median"]),
                            _scalar(row["raw_mean_difference"]),
                            _scalar(row["raw_median_difference"]),
                            _scalar(row["oriented_mean_advantage"]),
                            _scalar(row["favorable_proportion"]),
                            _scalar(
                                row["standardized_signal_to_heterogeneity"]
                            ),
                            _band(row, "pointwise"),
                            _band(row, "adjusted"),
                            _scalar(row["resamples"]),
                            _markdown(row["status"]),
                            _scalar(row["directional_language_allowed"]),
                        )
                    )
                    + " |"
                )
            lines.extend(
                (
                    "",
                    "#### Accounting",
                    "",
                    (
                        "| Policy A | Policy B | Both missing | Asymmetric missing | "
                        "Asymmetric reason | Asymmetric components | Eligible A | "
                        "Eligible B | Total A | Total B | Missing reasons A | "
                        "Missing reasons B | Source pairing complete |"
                    ),
                    (
                        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
                        "---: | ---: | --- | --- | --- |"
                    ),
                )
            )
            for row in slice_rows:
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            _markdown(row["policy_a"]),
                            _markdown(row["policy_b"]),
                            _scalar(row["both_missing_n"]),
                            _scalar(row["asymmetric_missing_n"]),
                            _scalar(row["asymmetric_reason_n"]),
                            _scalar(row["asymmetric_component_n"]),
                            _scalar(row["eligible_components_a"]),
                            _scalar(row["eligible_components_b"]),
                            _scalar(row["total_components_a"]),
                            _scalar(row["total_components_b"]),
                            _markdown(row["missing_reasons_a_json"]),
                            _markdown(row["missing_reasons_b_json"]),
                            _scalar(row["source_pairing_complete"]),
                        )
                    )
                    + " |"
                )
            lines.append("")
    return ("\n".join(lines)).encode("utf-8")


def _normalize_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(rows, (str, bytes, Mapping)):
        raise TypeError("rows must be an iterable of row mappings")
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise TypeError("scorecard rows must be mappings")
        if set(raw) != M5_SCORECARD_INPUT_FIELDS:
            raise ValueError(
                "scorecard row fields must exactly match the fixed M5 schema"
            )
        row = dict(raw)
        try:
            cell = PairedCellSpec(
                metric_name=row["metric_name"],
                metric_version=row["metric_version"],
                slice_name=row["slice_name"],
                slice_version=row["slice_version"],
                contrast=PolicyContrast(row["policy_a"], row["policy_b"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "scorecard identity is not a canonical M5 cell"
            ) from exc
        if (
            row["value_unit"] != cell.value_unit
            or row["direction"] != cell.direction
            or row["slice_name"] not in _SLICE_NAMES
            or (row["policy_a"], row["policy_b"]) not in _CONTRASTS
        ):
            raise ValueError("scorecard semantics drifted from the M5 registry")
        for name in _COUNT_FIELDS:
            minimum = 1 if name == "resamples" else 0
            maximum = 2**32 - 1 if name == "base_seed" else None
            row[name] = _integer(
                row[name],
                name,
                minimum=minimum,
                maximum=maximum,
            )
        if row["nonzero_effect_n"] is not None:
            row["nonzero_effect_n"] = _integer(
                row["nonzero_effect_n"],
                "nonzero_effect_n",
                minimum=0,
            )
        for name in _OPTIONAL_REAL_FIELDS:
            if row[name] is not None:
                row[name] = _finite(row[name], name)
        for name in (
            "source_pairing_complete",
            "directional_language_allowed",
        ):
            if type(row[name]) is not bool:
                raise TypeError(f"{name} must be a boolean")
        for name in ("missing_reasons_a_json", "missing_reasons_b_json"):
            row[name] = _reason_counts(row[name], name, cell.invalid_reason_codes)
        words = row["resampling_digest_words"]
        if isinstance(words, (str, bytes)) or not isinstance(words, Sequence):
            raise TypeError("resampling_digest_words must be a sequence")
        normalized_words = tuple(
            _integer(
                word,
                "resampling digest word",
                minimum=0,
                maximum=2**32 - 1,
            )
            for word in words
        )
        expected_key = make_resampling_key(cell, paired_n=row["paired_n"])
        if (
            row["base_seed"] != M5_BASE_SEED
            or row["rng"] != "PCG64"
            or row["index_dtype"] != "int64"
            or row["quantile_method"] != "linear"
            or row["resamples"] != cell.resamples
            or row["resampling_key_json"] != expected_key.canonical_json
            or row["resampling_sha256"] != expected_key.sha256
            or normalized_words != expected_key.digest_words
        ):
            raise ValueError(
                "scorecard resampling metadata is not the frozen M5 substream"
            )
        row["resampling_digest_words"] = list(normalized_words)
        _validate_accounting(row)
        _validate_bands(row, cell)
        normalized.append(row)
    normalized.sort(key=lambda row: tuple(row[field] for field in _KEY_FIELDS))
    return normalized


def _validate_accounting(row: Mapping[str, Any]) -> None:
    if row["paired_n"] + row["excluded_n"] != row["cohort_n"]:
        raise ValueError("paired and excluded counts must equal cohort count")
    if (
        row["valid_a_n"] > row["cohort_n"]
        or row["valid_b_n"] > row["cohort_n"]
        or row["paired_n"] > min(row["valid_a_n"], row["valid_b_n"])
    ):
        raise ValueError("valid counts contradict cohort accounting")
    if (
        row["both_missing_n"] + row["asymmetric_missing_n"]
        != row["excluded_n"]
        or row["valid_a_n"] + row["valid_b_n"]
        != 2 * row["paired_n"] + row["asymmetric_missing_n"]
        or row["asymmetric_reason_n"] > row["both_missing_n"]
        or row["asymmetric_component_n"] > row["cohort_n"]
    ):
        raise ValueError("missingness counts contradict cohort accounting")
    if (
        row["eligible_components_a"] > row["total_components_a"]
        or row["eligible_components_b"] > row["total_components_b"]
        or row["eligible_components_a"] < row["valid_a_n"]
        or row["eligible_components_b"] < row["valid_b_n"]
    ):
        raise ValueError("component counts contradict valid counts")
    if (
        row["nonzero_effect_n"] is not None
        and row["nonzero_effect_n"] > row["paired_n"]
    ):
        raise ValueError("nonzero effect count exceeds paired count")
    reasons_a = json.loads(row["missing_reasons_a_json"])
    reasons_b = json.loads(row["missing_reasons_b_json"])
    if sum(reasons_a.values()) != row["cohort_n"] - row["valid_a_n"]:
        raise ValueError("policy A missing-reason counts are not exact")
    if sum(reasons_b.values()) != row["cohort_n"] - row["valid_b_n"]:
        raise ValueError("policy B missing-reason counts are not exact")
    expected_complete = (
        row["asymmetric_missing_n"] == 0
        and row["asymmetric_reason_n"] == 0
        and row["asymmetric_component_n"] == 0
    )
    if row["source_pairing_complete"] != expected_complete:
        raise ValueError("source pairing flag contradicts asymmetric accounting")


def _validate_bands(row: Mapping[str, Any], cell: PairedCellSpec) -> None:
    for prefix in ("pointwise", "adjusted"):
        values = tuple(
            row[f"{prefix}_{suffix}"]
            for suffix in ("level", "lower", "upper")
        )
        if any(value is None for value in values) and not all(
            value is None for value in values
        ):
            raise ValueError(f"{prefix} band must be wholly present or absent")
        if all(value is not None for value in values):
            level, lower, upper = values
            if not 0.0 < level < 1.0 or lower > upper:
                raise ValueError(f"{prefix} band is invalid")
    pointwise = tuple(
        row[f"pointwise_{suffix}"] for suffix in ("level", "lower", "upper")
    )
    adjusted = tuple(
        row[f"adjusted_{suffix}"] for suffix in ("level", "lower", "upper")
    )
    if row["paired_n"] < 10:
        suppressed = (
            "nonzero_effect_n",
            "raw_mean_difference",
            "raw_median_difference",
            "oriented_mean_advantage",
            "favorable_proportion",
            "standardized_signal_to_heterogeneity",
            "policy_a_mean",
            "policy_a_median",
            "policy_b_mean",
            "policy_b_median",
            "pointwise_level",
            "pointwise_lower",
            "pointwise_upper",
            "adjusted_level",
            "adjusted_lower",
            "adjusted_upper",
        )
        if any(row[name] is not None for name in suppressed):
            raise ValueError("paired count below ten requires effect suppression")
        expected_status = (
            "insufficient_n"
            if row["source_pairing_complete"]
            else "pairing_incomplete"
        )
        expected_directional = False
    else:
        required = (
            "nonzero_effect_n",
            "raw_mean_difference",
            "raw_median_difference",
            "policy_a_mean",
            "policy_a_median",
            "policy_b_mean",
            "policy_b_median",
        )
        if any(row[name] is None for name in required):
            raise ValueError("unsuppressed scorecards require exact effects")
        if pointwise[0] != M5_POINTWISE_STABILITY_LEVEL:
            raise ValueError("pointwise stability level is not frozen")
        if cell.is_primary:
            if (
                adjusted[0] != M5_PRIMARY_ADJUSTED_STABILITY_LEVEL
                or adjusted[1] > pointwise[1]
                or adjusted[2] < pointwise[2]
            ):
                raise ValueError("primary adjusted band is invalid")
            claim_band = adjusted
        else:
            if any(value is not None for value in adjusted):
                raise ValueError("exploratory cells cannot carry adjusted bands")
            claim_band = pointwise
        if cell.direction == "neutral":
            for name in (
                "oriented_mean_advantage",
                "favorable_proportion",
                "standardized_signal_to_heterogeneity",
            ):
                if row[name] is not None:
                    raise ValueError("neutral metrics cannot carry oriented effects")
        elif (
            row["oriented_mean_advantage"] is None
            or row["favorable_proportion"] is None
        ):
            raise ValueError("directional metrics require oriented effects")
        favorable = row["favorable_proportion"]
        if favorable is not None and not 0.0 <= favorable <= 1.0:
            raise ValueError("favorable proportion must lie in [0, 1]")
        standardized = row["standardized_signal_to_heterogeneity"]
        if standardized is not None:
            if row["paired_n"] < 30 or row["nonzero_effect_n"] < 10:
                raise ValueError(
                    "standardized signal requires the frozen sample thresholds"
                )
            oriented = row["oriented_mean_advantage"]
            if (
                (oriented > 0.0 and standardized <= 0.0)
                or (oriented < 0.0 and standardized >= 0.0)
                or (oriented == 0.0 and standardized != 0.0)
            ):
                raise ValueError(
                    "standardized signal contradicts the oriented effect"
                )
        if cell.direction != "neutral":
            ties = row["paired_n"] - row["nonzero_effect_n"]
            implied_wins = row["paired_n"] * favorable - 0.5 * ties
            nearest_wins = round(implied_wins)
            if (
                nearest_wins < 0
                or nearest_wins > row["nonzero_effect_n"]
                or not _finite_close(implied_wins, nearest_wins)
            ):
                raise ValueError(
                    "favorable proportion is outside the wins/ties lattice"
                )
        expected_raw_mean = row["policy_a_mean"] - row["policy_b_mean"]
        if not _finite_close(
            row["raw_mean_difference"],
            expected_raw_mean,
            scale_values=(row["policy_a_mean"], row["policy_b_mean"]),
        ):
            raise ValueError(
                "raw mean difference contradicts the policy means"
            )
        if cell.direction != "neutral":
            expected_oriented = (
                row["raw_mean_difference"]
                if cell.direction == "higher"
                else -row["raw_mean_difference"]
            )
            if row["oriented_mean_advantage"] != expected_oriented:
                raise ValueError(
                    "oriented effect contradicts metric direction"
                )
        if row["nonzero_effect_n"] == 0:
            required_zero = (
                "raw_mean_difference",
                "raw_median_difference",
                "pointwise_lower",
                "pointwise_upper",
            )
            if cell.direction != "neutral":
                required_zero += ("oriented_mean_advantage",)
                if row["favorable_proportion"] != 0.5:
                    raise ValueError(
                        "zero-effect directional cells require half-favorable ties"
                    )
            if cell.is_primary:
                required_zero += ("adjusted_lower", "adjusted_upper")
            if any(row[name] != 0.0 for name in required_zero):
                raise ValueError(
                    "zero nonzero-effect count requires zero effects and bands"
                )
            if row["standardized_signal_to_heterogeneity"] is not None:
                raise ValueError(
                    "zero-effect cells cannot carry standardized signal"
                )
        if not row["source_pairing_complete"]:
            expected_status = "pairing_incomplete"
        elif cell.is_primary and row["nonzero_effect_n"] < 10:
            expected_status = "event_sparse"
        elif row["paired_n"] < 30 or row["nonzero_effect_n"] < 10:
            expected_status = "small_or_sparse"
        else:
            expected_status = "descriptive"
        band_excludes_zero = claim_band[2] < 0.0 or claim_band[1] > 0.0
        expected_directional = (
            cell.direction != "neutral"
            and row["paired_n"] >= 30
            and row["nonzero_effect_n"] >= 10
            and row["source_pairing_complete"]
            and band_excludes_zero
        )
    if row["status"] != expected_status:
        raise ValueError("scorecard status contradicts frozen M5 rules")
    if row["directional_language_allowed"] != expected_directional:
        raise ValueError("directional claim flag contradicts frozen M5 rules")


def _reason_counts(value: Any, name: str, allowed: Sequence[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be canonical JSON text")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON") from exc
    if (
        not isinstance(payload, dict)
        or not set(payload).issubset(allowed)
        or any(
            not isinstance(key, str)
            or isinstance(count, bool)
            or not isinstance(count, Integral)
            or count < 1
            for key, count in payload.items()
        )
    ):
        raise ValueError(f"{name} must contain registered positive counts")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if value != canonical:
        raise ValueError(f"{name} must use canonical JSON")
    return value


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its accepted range")
    return result


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_close(
    actual: Any,
    expected: Any,
    *,
    scale_values: Sequence[Any] = (),
) -> bool:
    if not isinstance(actual, Real) or not isinstance(expected, Real):
        return False
    actual_float = float(actual)
    expected_float = float(expected)
    if not math.isfinite(actual_float) or not math.isfinite(expected_float):
        return False
    normalized_scales: list[float] = []
    for value in scale_values:
        if not isinstance(value, Real) or not math.isfinite(float(value)):
            return False
        normalized_scales.append(abs(float(value)))
    scale = max(
        abs(actual_float),
        abs(expected_float),
        *normalized_scales,
        1.0,
    )
    return abs(actual_float - expected_float) <= 32.0 * math.ulp(scale)


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if isinstance(value, Integral) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("rendered numeric values must be finite")
        return json.dumps(
            numeric,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
        )
    if isinstance(value, str):
        return _markdown(value)
    raise TypeError("unsupported rendered scorecard scalar")


def _band(row: Mapping[str, Any], prefix: str) -> str:
    level = row[f"{prefix}_level"]
    if level is None:
        return "null"
    return (
        f"[{_scalar(row[f'{prefix}_lower'])},"
        f"{_scalar(row[f'{prefix}_upper'])}]@{_scalar(level)}"
    )


def _markdown(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("Markdown text values must be strings")
    if any(
        (ord(character) < 0x20 and character not in {"\t", "\n", "\r"})
        or ord(character) == 0x7F
        for character in value
    ):
        raise ValueError("Markdown text cannot contain control characters")
    return (
        value.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r\n", "<br>")
        .replace("\r", "<br>")
        .replace("\n", "<br>")
        .replace("\t", " ")
    )


__all__ = [
    "DATA_FREE_TEST_PROFILE",
    "M5_SCORECARD_INPUT_FIELDS",
    "M5_SCORECARD_RENDERED_ROW_FIELDS",
    "M5_SCORECARD_RENDERER_VERSION",
    "M5_SCORECARD_REPORT_MEDIA_TYPE",
    "M5_SCORECARD_REPORT_PATH",
    "M5_SCORECARD_SOURCE_SCHEMA_VERSION",
    "OFFICIAL_M5_PROFILE",
    "render_m5_scorecard",
]
