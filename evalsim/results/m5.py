"""Immutable, project-contained local result storage for M5.

The store is deliberately small and append-never: every run name is created once,
every Parquet part is created exclusively, and a run becomes readable as accepted
evidence only after its final manifest and ``SUCCESS`` marker exist.  Failed or
interrupted runs remain in place and cannot be resumed into success.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
from types import MappingProxyType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from evalsim.metrics.m5 import (
    M5_KINEMATIC_METRIC_VERSION,
    M5_METRIC_SPECS,
    M5_METRIC_VERSION,
)
from evalsim.report.m5 import (
    M5_SCORECARD_RENDERER_VERSION,
    M5_SCORECARD_REPORT_MEDIA_TYPE,
    M5_SCORECARD_REPORT_PATH,
    render_m5_scorecard,
)
from evalsim.simulators.waymax_reference import WAYMAX_EXACT_LOG_NAME
from evalsim.simulators.waymax_reference import WAYMAX_REFERENCE_VERSION
from evalsim.slices.m5 import M5_SLICE_SPECS, M5_SLICE_VERSION
from evalsim.stats.m5 import (
    M5_BASE_SEED,
    M5_OTHER_RESAMPLES,
    M5_POINTWISE_STABILITY_LEVEL,
    M5_POLICY_CONTRASTS,
    M5_PRIMARY_ADJUSTED_STABILITY_LEVEL,
    M5_PRIMARY_METRIC_NAMES,
    M5_PRIMARY_RESAMPLES,
    M5_STATISTICS_SCHEMA_VERSION,
    PairedCellResult,
    PairedCellSpec,
    PolicyContrast,
    ScenarioScalar,
    analyze_paired_cell,
    make_resampling_key,
)


M5_RESULT_STORE_SCHEMA_VERSION = "m5-result-store-1.0.0"
OFFICIAL_M5_PROFILE = "official_m5"
DATA_FREE_TEST_PROFILE = "data_free_test"
M5_PARITY_ORDER_RECEIPT_SCHEMA_VERSION = (
    "m5-parity-order-receipt-1.0.0"
)
M5_DETERMINISM_RECEIPT_SCHEMA_VERSION = (
    "m5-determinism-receipt-1.0.0"
)
M5_PARITY_RANK_DOMAIN = "evalsim-m5-metric-parity-v1"
M5_PARITY_ORDER_VERSION = "m5-metric-parity-order-1"
M5_PARITY_SCENE_COUNT = 16
M5_PARITY_TRANSITION_COUNT = 20
M5_DETERMINISM_CANONICALIZATION_VERSION = (
    "m5-evaluation-canonical-json-digests-1"
)
M5_M4_REUSE_SCHEMA_VERSION = "m5-m4-reuse-1.0.0"
M5_M4_SELECTED_ORDER_VERSION = "m4-selected-order-1"
M5_M4_INTEGRITY_ASSUMPTION = (
    "explicit_ignored_m4_run_is_local_trust_root"
)

PARITY_ORDER_RECEIPT = "parity_order_receipt"
DETERMINISM_RECEIPT = "determinism_receipt"
_SUPPLEMENTAL_PATHS = MappingProxyType(
    {
        PARITY_ORDER_RECEIPT: "parity-order-receipt.json",
        DETERMINISM_RECEIPT: "determinism-receipt.json",
    }
)

OFFICIAL_WAYMAX_REFERENCE_PARAMETERS: Mapping[str, Any] = MappingProxyType(
    {
        "executed": True,
        "horizon_transitions": 80,
        "scenario_count": 128,
    }
)

METRIC_RESULTS = "metric-results"
SLICE_MEMBERSHIP = "slice-membership"
SCORECARDS = "scorecards"
WAYMAX_PARITY_SUMMARY = "waymax-parity-summary"

_DATASET_NAMES = (
    METRIC_RESULTS,
    SLICE_MEMBERSHIP,
    SCORECARDS,
    WAYMAX_PARITY_SUMMARY,
)
_RUN_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_LOWER_NAME = re.compile(r"[a-z][a-z0-9_]*")
_SEMANTIC_VERSION = re.compile(
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PART_NAME = re.compile(r"part-([0-9]{5})\.parquet")
_FLOAT32_MAX = 3.4028234663852886e38
_CONTROL_FILES = frozenset(
    {
        "COMMITTED",
        "FINALIZING",
        "FAILURE.json",
        "evaluation-manifest.json",
        "SUCCESS",
    }
)
_M5_SLICE_SPECS = MappingProxyType(
    {spec.name: spec for spec in M5_SLICE_SPECS}
)
_M5_POLICY_CONTRASTS = tuple(
    (contrast.policy_a, contrast.policy_b)
    for contrast in M5_POLICY_CONTRASTS
)
_M5_POLICY_NAMES = tuple(
    sorted(
        {
            name
            for contrast in M5_POLICY_CONTRASTS
            for name in (contrast.policy_a, contrast.policy_b)
        }
    )
)
_M5_EXECUTION_ROLES = MappingProxyType(
    {
        **{name: "policy" for name in _M5_POLICY_NAMES},
        WAYMAX_EXACT_LOG_NAME: "reference",
    }
)
_M5_PARITY_METRIC_NAMES = (
    "kinematic_infeasibility",
    "log_divergence",
    "overlap",
)
_M5_PARITY_METRIC_VERSIONS = MappingProxyType(
    {
        "kinematic_infeasibility": M5_KINEMATIC_METRIC_VERSION,
        "log_divergence": M5_METRIC_VERSION,
        "overlap": M5_METRIC_VERSION,
    }
)
_M5_ERROR_ORACLE_METRIC_NAMES = (
    "acceleration_error_mps2",
    "jerk_error_mps3",
    "position_error_m",
    "speed_error_mps",
    "yaw_rate_error_radps",
)
_OFFICIAL_EXECUTABLE_ROOTS = (
    "evalsim/cli",
    "evalsim/contracts",
    "evalsim/evaluation",
    "evalsim/metrics",
    "evalsim/report",
    "evalsim/results",
    "evalsim/rollout",
    "evalsim/simulators",
    "evalsim/slices",
    "evalsim/sources",
    "evalsim/stats",
    "tests",
)
_OFFICIAL_EXECUTABLE_FILES = (
    "AGENTS.md",
    "NOTICE.md",
    "docs/data/womd-waymax-m5-metric-crosswalk.md",
    "docs/plans/2026-07-28-m5-real-womd-metrics-scorecards.md",
    "pyproject.toml",
    "uv.lock",
)


def _schema(fields: Sequence[pa.Field], name: str) -> pa.Schema:
    return pa.schema(
        fields,
        metadata={
            b"evalsim.dataset": name.encode("ascii"),
            b"evalsim.schema_version": M5_RESULT_STORE_SCHEMA_VERSION.encode(
                "ascii"
            ),
        },
    )


METRIC_RESULTS_SCHEMA = _schema(
    (
        pa.field("cohort_index", pa.int32(), nullable=False),
        pa.field("execution_name", pa.string(), nullable=False),
        pa.field("execution_role", pa.string(), nullable=False),
        pa.field("seed", pa.uint32(), nullable=False),
        pa.field("metric_name", pa.string(), nullable=False),
        pa.field("metric_version", pa.string(), nullable=False),
        pa.field("value", pa.float64(), nullable=True),
        pa.field("valid", pa.bool_(), nullable=False),
        pa.field("invalid_reason", pa.string(), nullable=True),
        pa.field("eligible_components", pa.int64(), nullable=False),
        pa.field("total_components", pa.int64(), nullable=False),
        pa.field(
            "distribution",
            pa.list_(pa.field("element", pa.float64(), nullable=False)),
            nullable=False,
        ),
        pa.field("details_json", pa.string(), nullable=False),
    ),
    METRIC_RESULTS,
)

SLICE_MEMBERSHIP_SCHEMA = _schema(
    (
        pa.field("cohort_index", pa.int32(), nullable=False),
        pa.field("slice_name", pa.string(), nullable=False),
        pa.field("slice_version", pa.string(), nullable=False),
        pa.field("eligible", pa.bool_(), nullable=False),
        pa.field("member", pa.bool_(), nullable=False),
        pa.field("reason", pa.string(), nullable=True),
    ),
    SLICE_MEMBERSHIP,
)

SCORECARDS_SCHEMA = _schema(
    (
        pa.field("metric_name", pa.string(), nullable=False),
        pa.field("metric_version", pa.string(), nullable=False),
        pa.field("value_unit", pa.string(), nullable=False),
        pa.field("direction", pa.string(), nullable=False),
        pa.field("slice_name", pa.string(), nullable=False),
        pa.field("slice_version", pa.string(), nullable=False),
        pa.field("policy_a", pa.string(), nullable=False),
        pa.field("policy_b", pa.string(), nullable=False),
        pa.field("cohort_n", pa.int32(), nullable=False),
        pa.field("valid_a_n", pa.int32(), nullable=False),
        pa.field("valid_b_n", pa.int32(), nullable=False),
        pa.field("paired_n", pa.int32(), nullable=False),
        pa.field("excluded_n", pa.int32(), nullable=False),
        pa.field("both_missing_n", pa.int32(), nullable=False),
        pa.field("asymmetric_missing_n", pa.int32(), nullable=False),
        pa.field("asymmetric_reason_n", pa.int32(), nullable=False),
        pa.field("asymmetric_component_n", pa.int32(), nullable=False),
        pa.field("missing_reasons_a_json", pa.string(), nullable=False),
        pa.field("missing_reasons_b_json", pa.string(), nullable=False),
        pa.field("eligible_components_a", pa.int64(), nullable=False),
        pa.field("eligible_components_b", pa.int64(), nullable=False),
        pa.field("total_components_a", pa.int64(), nullable=False),
        pa.field("total_components_b", pa.int64(), nullable=False),
        pa.field("source_pairing_complete", pa.bool_(), nullable=False),
        pa.field("nonzero_effect_n", pa.int32(), nullable=True),
        pa.field("raw_mean_difference", pa.float64(), nullable=True),
        pa.field("raw_median_difference", pa.float64(), nullable=True),
        pa.field("oriented_mean_advantage", pa.float64(), nullable=True),
        pa.field("favorable_proportion", pa.float64(), nullable=True),
        pa.field(
            "standardized_signal_to_heterogeneity",
            pa.float64(),
            nullable=True,
        ),
        pa.field("policy_a_mean", pa.float64(), nullable=True),
        pa.field("policy_a_median", pa.float64(), nullable=True),
        pa.field("policy_b_mean", pa.float64(), nullable=True),
        pa.field("policy_b_median", pa.float64(), nullable=True),
        pa.field("pointwise_level", pa.float64(), nullable=True),
        pa.field("pointwise_lower", pa.float64(), nullable=True),
        pa.field("pointwise_upper", pa.float64(), nullable=True),
        pa.field("adjusted_level", pa.float64(), nullable=True),
        pa.field("adjusted_lower", pa.float64(), nullable=True),
        pa.field("adjusted_upper", pa.float64(), nullable=True),
        pa.field("status", pa.string(), nullable=False),
        pa.field("directional_language_allowed", pa.bool_(), nullable=False),
        pa.field("resampling_key_json", pa.string(), nullable=False),
        pa.field("resampling_sha256", pa.string(), nullable=False),
        pa.field(
            "resampling_digest_words",
            pa.list_(pa.field("element", pa.uint32(), nullable=False), 8),
            nullable=False,
        ),
        pa.field("resamples", pa.int32(), nullable=False),
        pa.field("base_seed", pa.uint32(), nullable=False),
        pa.field("rng", pa.string(), nullable=False),
        pa.field("index_dtype", pa.string(), nullable=False),
        pa.field("quantile_method", pa.string(), nullable=False),
    ),
    SCORECARDS,
)

WAYMAX_PARITY_SUMMARY_SCHEMA = _schema(
    (
        pa.field("parity_index", pa.int32(), nullable=False),
        pa.field("policy_name", pa.string(), nullable=False),
        pa.field("metric_name", pa.string(), nullable=False),
        pa.field("metric_version", pa.string(), nullable=False),
        pa.field("compared_components", pa.int64(), nullable=False),
        pa.field("mismatch_count", pa.int64(), nullable=False),
        pa.field("max_abs_error", pa.float64(), nullable=True),
        pa.field("max_tolerance_excess", pa.float64(), nullable=True),
        pa.field("exact_match", pa.bool_(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
    ),
    WAYMAX_PARITY_SUMMARY,
)

M5_RESULT_SCHEMAS: Mapping[str, pa.Schema] = MappingProxyType(
    {
        METRIC_RESULTS: METRIC_RESULTS_SCHEMA,
        SLICE_MEMBERSHIP: SLICE_MEMBERSHIP_SCHEMA,
        SCORECARDS: SCORECARDS_SCHEMA,
        WAYMAX_PARITY_SUMMARY: WAYMAX_PARITY_SUMMARY_SCHEMA,
    }
)

_KEY_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        METRIC_RESULTS: (
            "cohort_index",
            "execution_name",
            "seed",
            "metric_name",
            "metric_version",
        ),
        SLICE_MEMBERSHIP: (
            "cohort_index",
            "slice_name",
            "slice_version",
        ),
        SCORECARDS: (
            "metric_name",
            "metric_version",
            "slice_name",
            "slice_version",
            "policy_a",
            "policy_b",
        ),
        WAYMAX_PARITY_SUMMARY: (
            "parity_index",
            "policy_name",
            "metric_name",
            "metric_version",
        ),
    }
)


class M5ResultStoreError(RuntimeError):
    """Base error for M5 result-store contract violations."""


class M5ResultStoreStateError(M5ResultStoreError):
    """The requested mutation is incompatible with the run's terminal state."""


class M5ResultStoreIntegrityError(M5ResultStoreError):
    """Stored bytes, schemas, keys, or accounting failed verification."""


@dataclass(frozen=True, slots=True)
class ExpectedRowCounts:
    """Exact M5 row-accounting gate, overridable only for data-free tests."""

    metric_results: int = 6_656
    slice_membership: int = 1_024
    scorecards: int = 312
    waymax_parity_summary: int = 144

    def __post_init__(self) -> None:
        for name in (
            "metric_results",
            "slice_membership",
            "scorecards",
            "waymax_parity_summary",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name=name, minimum=0),
            )

    def to_dict(self) -> dict[str, int]:
        return {
            METRIC_RESULTS: self.metric_results,
            SLICE_MEMBERSHIP: self.slice_membership,
            SCORECARDS: self.scorecards,
            WAYMAX_PARITY_SUMMARY: self.waymax_parity_summary,
        }


@dataclass(frozen=True, slots=True)
class M5ParityOrderReceipt:
    """Immutable source-only selection receipt written before metric outcomes."""

    rank_domain: str
    order_version: str
    ordered_membership_sha256: str = field(repr=False)
    member_count: int = M5_PARITY_SCENE_COUNT
    transition_count: int = M5_PARITY_TRANSITION_COUNT

    def __post_init__(self) -> None:
        if self.rank_domain != M5_PARITY_RANK_DOMAIN:
            raise ValueError("rank_domain is not the frozen M5 parity domain")
        if self.order_version != M5_PARITY_ORDER_VERSION:
            raise ValueError("order_version is not the frozen M5 parity order")
        if (
            not isinstance(self.ordered_membership_sha256, str)
            or _SHA256.fullmatch(self.ordered_membership_sha256) is None
        ):
            raise ValueError(
                "ordered_membership_sha256 must be a SHA-256 digest"
            )
        if self.member_count != M5_PARITY_SCENE_COUNT:
            raise ValueError("member_count is not the frozen parity size")
        if self.transition_count != M5_PARITY_TRANSITION_COUNT:
            raise ValueError(
                "transition_count is not the frozen parity horizon"
            )
    def to_dict(self) -> dict[str, Any]:
        return {
            "member_count": self.member_count,
            "order_version": self.order_version,
            "ordered_membership_sha256": self.ordered_membership_sha256,
            "rank_domain": self.rank_domain,
            "transition_count": self.transition_count,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "M5ParityOrderReceipt":
        expected = {
            "member_count",
            "order_version",
            "ordered_membership_sha256",
            "rank_domain",
            "transition_count",
        }
        if set(payload) != expected:
            raise M5ResultStoreIntegrityError(
                "parity-order receipt has unexpected fields"
            )
        try:
            return cls(**dict(payload))
        except (TypeError, ValueError) as exc:
            raise M5ResultStoreIntegrityError(
                "parity-order receipt is invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class M5DeterminismReceipt:
    """Two-pass evaluator digests, later rebound to the stored official rows."""

    metric_pass_1_sha256: str = field(repr=False)
    metric_pass_2_sha256: str = field(repr=False)
    statistics_pass_1_sha256: str = field(repr=False)
    statistics_pass_2_sha256: str = field(repr=False)
    metric_row_count: int = 6_656
    statistics_row_count: int = 312
    metric_passes_equal: bool = True
    statistics_passes_equal: bool = True
    reference_matches_log_replay: bool = True
    zero_error_oracles_passed: bool = True
    canonicalization_version: str = (
        M5_DETERMINISM_CANONICALIZATION_VERSION
    )
    schema_version: str = M5_DETERMINISM_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "metric_pass_1_sha256",
            "metric_pass_2_sha256",
            "statistics_pass_1_sha256",
            "statistics_pass_2_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be a SHA-256 digest")
        object.__setattr__(
            self,
            "metric_row_count",
            _integer(
                self.metric_row_count,
                name="metric_row_count",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "statistics_row_count",
            _integer(
                self.statistics_row_count,
                name="statistics_row_count",
                minimum=0,
            ),
        )
        for name in (
            "metric_passes_equal",
            "statistics_passes_equal",
            "reference_matches_log_replay",
            "zero_error_oracles_passed",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"{name} must be exactly true")
        if (
            self.metric_pass_1_sha256 != self.metric_pass_2_sha256
            or not self.metric_passes_equal
        ):
            raise ValueError("metric pass digests must be identical")
        if (
            self.statistics_pass_1_sha256
            != self.statistics_pass_2_sha256
            or not self.statistics_passes_equal
        ):
            raise ValueError("statistics pass digests must be identical")
        if (
            self.metric_row_count
            != OFFICIAL_M5_ROW_COUNTS.metric_results
            or self.statistics_row_count
            != OFFICIAL_M5_ROW_COUNTS.scorecards
        ):
            raise ValueError(
                "determinism receipt row counts are not the official domains"
            )
        if (
            self.canonicalization_version
            != M5_DETERMINISM_CANONICALIZATION_VERSION
        ):
            raise ValueError(
                "determinism canonicalization version is unsupported"
            )
        if self.schema_version != M5_DETERMINISM_RECEIPT_SCHEMA_VERSION:
            raise ValueError("determinism receipt schema is unsupported")

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonicalization_version": self.canonicalization_version,
            "metric_passes_equal": self.metric_passes_equal,
            "metric_pass_1_sha256": self.metric_pass_1_sha256,
            "metric_pass_2_sha256": self.metric_pass_2_sha256,
            "metric_row_count": self.metric_row_count,
            "reference_matches_log_replay": (
                self.reference_matches_log_replay
            ),
            "schema_version": self.schema_version,
            "statistics_passes_equal": self.statistics_passes_equal,
            "statistics_pass_1_sha256": self.statistics_pass_1_sha256,
            "statistics_pass_2_sha256": self.statistics_pass_2_sha256,
            "statistics_row_count": self.statistics_row_count,
            "zero_error_oracles_passed": self.zero_error_oracles_passed,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "M5DeterminismReceipt":
        expected = {
            "canonicalization_version",
            "metric_passes_equal",
            "metric_pass_1_sha256",
            "metric_pass_2_sha256",
            "metric_row_count",
            "reference_matches_log_replay",
            "schema_version",
            "statistics_passes_equal",
            "statistics_pass_1_sha256",
            "statistics_pass_2_sha256",
            "statistics_row_count",
            "zero_error_oracles_passed",
        }
        if set(payload) != expected:
            raise M5ResultStoreIntegrityError(
                "determinism receipt has unexpected fields"
            )
        try:
            return cls(**dict(payload))
        except (TypeError, ValueError) as exc:
            raise M5ResultStoreIntegrityError(
                "determinism receipt is invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class M5RunProvenance:
    """Typed non-payload provenance required before a run can become success."""

    m4_manifest_sha256: str
    m4_execution_provenance_sha256: str
    selected_order_version: str
    selected_order_fingerprint_sha256: str
    executable_source_fingerprint_sha256: str
    executable_source_paths: tuple[str, ...]
    git_commit: str
    git_tree: str
    simulator_specs: Mapping[str, Any]
    runtime_versions: Mapping[str, str]
    m4_aggregate_summary_sha256: str | None = None
    m4_reuse_schema_version: str | None = None
    m4_integrity_assumption: str | None = None
    m4_receipt_verified: bool | None = None
    parity_order_version: str | None = None
    parity_order_fingerprint_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "m4_manifest_sha256",
            "m4_execution_provenance_sha256",
            "selected_order_fingerprint_sha256",
            "executable_source_fingerprint_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be a SHA-256 digest")
        for name in ("git_commit", "git_tree"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) not in {40, 64}
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    f"{name} must be a lowercase Git object identifier"
                )
        object.__setattr__(
            self,
            "selected_order_version",
            _text(self.selected_order_version, "selected_order_version"),
        )
        extension_names = (
            "m4_aggregate_summary_sha256",
            "m4_reuse_schema_version",
            "m4_integrity_assumption",
            "m4_receipt_verified",
            "parity_order_version",
            "parity_order_fingerprint_sha256",
        )
        extension_values = tuple(
            getattr(self, name) for name in extension_names
        )
        if any(value is None for value in extension_values) and not all(
            value is None for value in extension_values
        ):
            raise ValueError(
                "official M4/parity provenance fields must be all present or absent"
            )
        if all(value is not None for value in extension_values):
            for name in (
                "m4_aggregate_summary_sha256",
                "parity_order_fingerprint_sha256",
            ):
                value = getattr(self, name)
                if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                    raise ValueError(f"{name} must be a SHA-256 digest")
            if self.m4_reuse_schema_version != M5_M4_REUSE_SCHEMA_VERSION:
                raise ValueError("m4_reuse_schema_version is unsupported")
            if self.m4_integrity_assumption != M5_M4_INTEGRITY_ASSUMPTION:
                raise ValueError(
                    "m4_integrity_assumption is not the documented local rule"
                )
            if type(self.m4_receipt_verified) is not bool:
                raise TypeError("m4_receipt_verified must be a boolean")
            if self.parity_order_version != M5_PARITY_ORDER_VERSION:
                raise ValueError(
                    "parity_order_version is not the frozen M5 value"
                )
        if isinstance(self.executable_source_paths, (str, bytes)):
            raise ValueError(
                "executable_source_paths must be a sequence, not a string"
            )
        source_paths = tuple(
            _safe_project_relative_source_path(path).as_posix()
            for path in self.executable_source_paths
        )
        if (
            not source_paths
            or len(set(source_paths)) != len(source_paths)
            or source_paths != tuple(sorted(source_paths))
        ):
            raise ValueError(
                "executable_source_paths must be non-empty, unique, and sorted"
            )
        object.__setattr__(self, "executable_source_paths", source_paths)
        simulator_specs = _json_mapping(self.simulator_specs, "simulator_specs")
        if set(simulator_specs) != set(_M5_EXECUTION_ROLES):
            raise ValueError(
                "simulator_specs must bind the canonical four M5 executions"
            )
        for execution_name, raw_spec in simulator_specs.items():
            if not isinstance(raw_spec, Mapping):
                raise ValueError(
                    f"simulator_specs.{execution_name} must be an object"
                )
            if set(raw_spec) != {
                "deterministic",
                "execution_role",
                "parameters",
                "version",
            }:
                raise ValueError(
                    "each simulator spec must have the exact frozen field set"
                )
            if (
                not isinstance(raw_spec.get("version"), str)
                or not raw_spec["version"]
                or raw_spec.get("deterministic") is not True
                or raw_spec.get("execution_role")
                != _M5_EXECUTION_ROLES[execution_name]
                or not isinstance(raw_spec.get("parameters"), Mapping)
            ):
                raise ValueError(
                    "each simulator spec requires a version, deterministic=true, "
                    "canonical execution_role, and parameter object"
                )
        runtime_versions = _json_mapping(
            self.runtime_versions,
            "runtime_versions",
        )
        required_runtime_versions = {
            "flax",
            "jax",
            "jaxlib",
            "numpy",
            "pyarrow",
            "python",
            "tensorflow",
            "waymo_waymax",
        }
        if not required_runtime_versions.issubset(runtime_versions):
            raise ValueError(
                "runtime_versions omit a required M5/Waymax dependency version"
            )
        for key, value in runtime_versions.items():
            _text(key, "runtime version name")
            _text(value, f"runtime_versions.{key}")
        object.__setattr__(self, "simulator_specs", simulator_specs)
        object.__setattr__(self, "runtime_versions", runtime_versions)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "executable_source_fingerprint_sha256": (
                self.executable_source_fingerprint_sha256
            ),
            "executable_source_paths": list(self.executable_source_paths),
            "git_commit": self.git_commit,
            "git_tree": self.git_tree,
            "m4_execution_provenance_sha256": (
                self.m4_execution_provenance_sha256
            ),
            "m4_manifest_sha256": self.m4_manifest_sha256,
            "runtime_versions": _thaw_json(self.runtime_versions),
            "selected_order_fingerprint_sha256": (
                self.selected_order_fingerprint_sha256
            ),
            "selected_order_version": self.selected_order_version,
            "simulator_specs": _thaw_json(self.simulator_specs),
        }
        if self.has_official_extensions:
            payload.update(
                {
                    "m4_aggregate_summary_sha256": (
                        self.m4_aggregate_summary_sha256
                    ),
                    "m4_integrity_assumption": (
                        self.m4_integrity_assumption
                    ),
                    "m4_receipt_verified": self.m4_receipt_verified,
                    "m4_reuse_schema_version": (
                        self.m4_reuse_schema_version
                    ),
                    "parity_order_fingerprint_sha256": (
                        self.parity_order_fingerprint_sha256
                    ),
                    "parity_order_version": self.parity_order_version,
                }
            )
        return payload

    @property
    def has_official_extensions(self) -> bool:
        return self.m4_aggregate_summary_sha256 is not None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "M5RunProvenance":
        legacy = {
            "executable_source_fingerprint_sha256",
            "executable_source_paths",
            "git_commit",
            "git_tree",
            "m4_execution_provenance_sha256",
            "m4_manifest_sha256",
            "runtime_versions",
            "selected_order_fingerprint_sha256",
            "selected_order_version",
            "simulator_specs",
        }
        extended = legacy | {
            "m4_aggregate_summary_sha256",
            "m4_integrity_assumption",
            "m4_receipt_verified",
            "m4_reuse_schema_version",
            "parity_order_fingerprint_sha256",
            "parity_order_version",
        }
        fields = frozenset(payload)
        if fields not in {frozenset(legacy), frozenset(extended)}:
            raise M5ResultStoreIntegrityError(
                "manifest provenance has unexpected fields"
            )
        try:
            return cls(**dict(payload))
        except (TypeError, ValueError) as exc:
            raise M5ResultStoreIntegrityError(
                "manifest provenance is invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Manifest record for one immutable canonical Parquet part."""

    dataset: str
    path: str
    rows: int
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.dataset not in _DATASET_NAMES:
            raise ValueError(f"unknown result dataset {self.dataset!r}")
        relative = _safe_relative_artifact_path(self.path)
        if relative.as_posix() != self.path:
            raise ValueError("artifact path must use canonical POSIX separators")
        object.__setattr__(
            self,
            "rows",
            _integer(self.rows, name="rows", minimum=0),
        )
        object.__setattr__(
            self,
            "size_bytes",
            _integer(self.size_bytes, name="size_bytes", minimum=1),
        )
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must be a lowercase hexadecimal digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "path": self.path,
            "rows": self.rows,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactRecord":
        expected = {"dataset", "path", "rows", "sha256", "size_bytes"}
        if set(payload) != expected:
            raise M5ResultStoreIntegrityError(
                "manifest artifact record has unexpected fields"
            )
        try:
            return cls(**dict(payload))
        except (TypeError, ValueError) as exc:
            raise M5ResultStoreIntegrityError(
                "manifest artifact record is invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class SupplementalArtifactRecord:
    """Manifest record for one typed immutable official JSON artifact."""

    kind: str
    path: str
    schema_version: str
    size_bytes: int
    sha256: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.kind not in _SUPPLEMENTAL_PATHS:
            raise ValueError("supplemental artifact kind is unsupported")
        if self.path != _SUPPLEMENTAL_PATHS[self.kind]:
            raise ValueError("supplemental artifact path is not canonical")
        expected_schema = {
            PARITY_ORDER_RECEIPT: M5_PARITY_ORDER_RECEIPT_SCHEMA_VERSION,
            DETERMINISM_RECEIPT: M5_DETERMINISM_RECEIPT_SCHEMA_VERSION,
        }[self.kind]
        if self.schema_version != expected_schema:
            raise ValueError(
                "supplemental artifact schema version is unsupported"
            )
        object.__setattr__(
            self,
            "size_bytes",
            _integer(self.size_bytes, name="size_bytes", minimum=1),
        )
        if (
            not isinstance(self.sha256, str)
            or _SHA256.fullmatch(self.sha256) is None
        ):
            raise ValueError("sha256 must be a lowercase hexadecimal digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "SupplementalArtifactRecord":
        expected = {
            "kind",
            "path",
            "schema_version",
            "sha256",
            "size_bytes",
        }
        if set(payload) != expected:
            raise M5ResultStoreIntegrityError(
                "supplemental artifact record has unexpected fields"
            )
        try:
            return cls(**dict(payload))
        except (TypeError, ValueError) as exc:
            raise M5ResultStoreIntegrityError(
                "supplemental artifact record is invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class ScorecardReportRecord:
    """Manifest record for the deterministic human-readable scorecard."""

    path: str
    media_type: str
    renderer_version: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.path != M5_SCORECARD_REPORT_PATH:
            raise ValueError("scorecard report path is not canonical")
        if self.media_type != M5_SCORECARD_REPORT_MEDIA_TYPE:
            raise ValueError("scorecard report media type is not canonical")
        if self.renderer_version != M5_SCORECARD_RENDERER_VERSION:
            raise ValueError("scorecard renderer version is not supported")
        object.__setattr__(
            self,
            "size_bytes",
            _integer(self.size_bytes, name="size_bytes", minimum=1),
        )
        if (
            not isinstance(self.sha256, str)
            or _SHA256.fullmatch(self.sha256) is None
        ):
            raise ValueError("sha256 must be a lowercase hexadecimal digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "media_type": self.media_type,
            "path": self.path,
            "renderer_version": self.renderer_version,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "ScorecardReportRecord":
        expected = {
            "media_type",
            "path",
            "renderer_version",
            "sha256",
            "size_bytes",
        }
        if set(payload) != expected:
            raise M5ResultStoreIntegrityError(
                "manifest scorecard report has unexpected fields"
            )
        try:
            return cls(**dict(payload))
        except (TypeError, ValueError) as exc:
            raise M5ResultStoreIntegrityError(
                "manifest scorecard report is invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class VerifiedM5ResultStore:
    """Read-only verification result for one successful local run."""

    run_path: Path
    manifest: Mapping[str, Any]
    artifacts: tuple[ArtifactRecord, ...]
    scorecard_report: ScorecardReportRecord
    supplemental_artifacts: tuple[SupplementalArtifactRecord, ...] = ()
    parity_order_receipt: M5ParityOrderReceipt | None = None
    determinism_receipt: M5DeterminismReceipt | None = None


@dataclass(frozen=True, slots=True)
class _OfficialScanSummary:
    """Compact facts retained from one incremental official artifact scan."""

    metric_pass_sha256: str
    statistics_pass_sha256: str
    scorecard_rows: tuple[Mapping[str, Any], ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class PreparedM5Finalization:
    """Opaque capability for one verified manifest awaiting ``SUCCESS``."""

    run_path: Path
    _nonce: object = field(repr=False, compare=False)


class M5ResultStore:
    """Exclusive writer for one never-overwritten M5 result directory."""

    def __init__(
        self,
        *,
        project_root: Path,
        run_name: str,
        run_path: Path,
        expected_rows: ExpectedRowCounts,
        row_accounting_profile: str,
    ) -> None:
        self.project_root = project_root
        self.run_name = run_name
        self.run_path = run_path
        self.expected_rows = expected_rows
        self.row_accounting_profile = row_accounting_profile
        self._artifacts: dict[str, ArtifactRecord] = {}
        self._scorecard_report: ScorecardReportRecord | None = None
        self._supplemental_artifacts: dict[
            str,
            SupplementalArtifactRecord,
        ] = {}
        self._parity_order_receipt: M5ParityOrderReceipt | None = None
        self._determinism_receipt: M5DeterminismReceipt | None = None
        self._keys: dict[str, set[tuple[Any, ...]]] = {
            name: set() for name in _DATASET_NAMES
        }
        self._row_counts: dict[str, int] = {
            name: 0 for name in _DATASET_NAMES
        }
        self._failed = False
        self._finalizing = False
        self._committed_for_verification = False
        self._successful = False
        self._prepared_finalization: PreparedM5Finalization | None = None
        self._prepared_manifest_bytes: bytes | None = None

    @classmethod
    def create(
        cls,
        project_root: str | Path,
        run_name: str,
        *,
        expected_rows: ExpectedRowCounts | None = None,
        data_free: bool = False,
    ) -> "M5ResultStore":
        """Exclusively create ``outputs/m5/<run-name>`` under a real project root."""

        root = _validated_project_root(project_root)
        name = _validated_run_name(run_name)
        if type(data_free) is not bool:
            raise TypeError("data_free must be a boolean")
        counts = expected_rows or ExpectedRowCounts()
        if not isinstance(counts, ExpectedRowCounts):
            raise TypeError("expected_rows must be ExpectedRowCounts")
        if not data_free and counts != OFFICIAL_M5_ROW_COUNTS:
            raise ValueError(
                "official M5 runs require the frozen row-count matrix; "
                "custom counts are data-free-test-only"
            )
        profile = DATA_FREE_TEST_PROFILE if data_free else OFFICIAL_M5_PROFILE

        output_root = _ensure_directory(root, "outputs")
        m5_root = _ensure_directory(output_root, "m5")
        run_path = m5_root / name
        if os.path.lexists(run_path):
            raise FileExistsError(
                f"M5 run {name!r} already exists and cannot be overwritten"
            )
        created_run = False
        try:
            os.mkdir(run_path, mode=0o700)
            created_run = True
            _fsync_directory(m5_root)
            if (
                _path_kind(run_path) != "directory"
                or run_path.resolve(strict=True) != run_path
                or run_path.parent != m5_root
            ):
                raise M5ResultStoreError(
                    "created M5 run failed containment checks"
                )
            pending = _create_child_directory(run_path, "pending")
            _create_child_directory(pending, METRIC_RESULTS)
            _create_child_directory(run_path, METRIC_RESULTS)
        except FileExistsError as exc:
            if not created_run:
                raise
            _rollback_fresh_result_store(run_path)
            raise M5ResultStoreError(
                "could not create the M5 run directory"
            ) from exc
        except BaseException as exc:
            if created_run:
                _rollback_fresh_result_store(run_path)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, M5ResultStoreError):
                raise
            raise M5ResultStoreError(
                "could not create the M5 run directory"
            ) from exc
        return cls(
            project_root=root,
            run_name=name,
            run_path=run_path,
            expected_rows=counts,
            row_accounting_profile=profile,
        )

    @property
    def project_relative_path(self) -> Path:
        return Path("outputs") / "m5" / self.run_name

    @property
    def row_counts(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._row_counts))

    @property
    def artifacts(self) -> tuple[ArtifactRecord, ...]:
        return tuple(self._artifacts[path] for path in sorted(self._artifacts))

    @property
    def scorecard_report(self) -> ScorecardReportRecord | None:
        return self._scorecard_report

    @property
    def supplemental_artifacts(
        self,
    ) -> tuple[SupplementalArtifactRecord, ...]:
        return tuple(
            self._supplemental_artifacts[kind]
            for kind in sorted(self._supplemental_artifacts)
        )

    @property
    def parity_order_receipt(self) -> M5ParityOrderReceipt | None:
        return self._parity_order_receipt

    @property
    def determinism_receipt(self) -> M5DeterminismReceipt | None:
        return self._determinism_receipt

    def write_parity_order_receipt(
        self,
        receipt: M5ParityOrderReceipt,
    ) -> SupplementalArtifactRecord:
        """Write the official source-only parity order before any metric row."""

        self._ensure_active()
        if self.row_accounting_profile != OFFICIAL_M5_PROFILE:
            raise M5ResultStoreStateError(
                "parity-order receipts are official-run-only"
            )
        if not isinstance(receipt, M5ParityOrderReceipt):
            try:
                raw_receipt = receipt.to_dict()  # type: ignore[union-attr]
            except AttributeError as exc:
                raise TypeError(
                    "receipt must be a compatible M5ParityOrderReceipt"
                ) from exc
            if not isinstance(raw_receipt, Mapping):
                raise TypeError(
                    "receipt must be a compatible M5ParityOrderReceipt"
                )
            receipt = M5ParityOrderReceipt.from_dict(raw_receipt)
        if (
            self._row_counts[METRIC_RESULTS]
            or self._row_counts[SCORECARDS]
            or self._row_counts[WAYMAX_PARITY_SUMMARY]
            or any(
                record.dataset
                in {
                    METRIC_RESULTS,
                    SCORECARDS,
                    WAYMAX_PARITY_SUMMARY,
                }
                for record in self._artifacts.values()
            )
        ):
            self._poison("parity_order_receipt_late")
            raise M5ResultStoreStateError(
                "pre-metric parity-order receipt must precede metric-derived "
                "result rows"
            )
        try:
            payload = receipt.to_dict()
            payload["schema_version"] = (
                M5_PARITY_ORDER_RECEIPT_SCHEMA_VERSION
            )
            record = self._write_supplemental_artifact(
                PARITY_ORDER_RECEIPT,
                payload,
            )
            self._parity_order_receipt = receipt
            return record
        except BaseException:
            self._poison("parity_order_receipt_write_failed")
            raise

    def write_determinism_receipt(
        self,
        receipt: M5DeterminismReceipt,
    ) -> SupplementalArtifactRecord:
        """Write the two-pass receipt whose digests finalization will rederive."""

        self._ensure_active()
        if self.row_accounting_profile != OFFICIAL_M5_PROFILE:
            raise M5ResultStoreStateError(
                "determinism receipts are official-run-only"
            )
        if not isinstance(receipt, M5DeterminismReceipt):
            raise TypeError("receipt must be M5DeterminismReceipt")
        if self._parity_order_receipt is None:
            self._poison("determinism_receipt_before_parity_order")
            raise M5ResultStoreStateError(
                "determinism receipt requires the pre-metric parity receipt"
            )
        try:
            record = self._write_supplemental_artifact(
                DETERMINISM_RECEIPT,
                receipt.to_dict(),
            )
            self._determinism_receipt = receipt
            return record
        except BaseException:
            self._poison("determinism_receipt_write_failed")
            raise

    def write_metric_results_part(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        part_index: int = 0,
    ) -> ArtifactRecord:
        self._ensure_active()
        if (
            self.row_accounting_profile == OFFICIAL_M5_PROFILE
            and self._parity_order_receipt is None
        ):
            self._poison("parity_order_receipt_missing")
            raise M5ResultStoreStateError(
                "official metric rows require a pre-metric parity receipt"
            )
        index = _integer(part_index, name="part_index", minimum=0, maximum=99_999)
        relative = Path(METRIC_RESULTS) / f"part-{index:05d}.parquet"
        return self._write_dataset(METRIC_RESULTS, relative, rows)

    def write_slice_membership(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> ArtifactRecord:
        return self._write_dataset(
            SLICE_MEMBERSHIP,
            Path("slice-membership.parquet"),
            rows,
        )

    def write_scorecards(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> ArtifactRecord:
        """Write scorecards only after metric and slice accounting is exact."""

        self._ensure_active()
        expected = self.expected_rows.to_dict()
        if (
            self._row_counts[METRIC_RESULTS] != expected[METRIC_RESULTS]
            or self._row_counts[SLICE_MEMBERSHIP]
            != expected[SLICE_MEMBERSHIP]
        ):
            self._poison("scorecard_inputs_incomplete")
            raise M5ResultStoreIntegrityError(
                "scorecards require exact metric and slice row accounting"
            )
        return self._write_dataset(
            SCORECARDS,
            Path("scorecards.parquet"),
            rows,
        )

    def write_waymax_parity_summary(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> ArtifactRecord:
        return self._write_dataset(
            WAYMAX_PARITY_SUMMARY,
            Path("waymax-parity-summary.parquet"),
            rows,
        )

    def write_human_readable_scorecard(self) -> ScorecardReportRecord:
        """Render and exclusively publish the report from stored scorecard rows."""

        self._ensure_active()
        if (
            self._scorecard_report is not None
            or os.path.lexists(self.run_path / M5_SCORECARD_REPORT_PATH)
            or os.path.lexists(
                self.run_path / "pending" / M5_SCORECARD_REPORT_PATH
            )
        ):
            self._poison("scorecard_report_exists")
            raise FileExistsError("scorecard report already exists")
        expected = self.expected_rows.to_dict()[SCORECARDS]
        scorecard_records = tuple(
            record
            for record in self.artifacts
            if record.dataset == SCORECARDS
        )
        if (
            self._row_counts[SCORECARDS] != expected
            or len(scorecard_records) != 1
        ):
            self._poison("scorecard_report_inputs_incomplete")
            raise M5ResultStoreIntegrityError(
                "scorecard report requires the exact stored scorecard table"
            )
        try:
            payload = render_m5_scorecard(
                self.read_dataset(SCORECARDS).to_pylist(),
                row_accounting_profile=self.row_accounting_profile,
            )
            relative = Path(M5_SCORECARD_REPORT_PATH)
            pending_path = _pending_path(self.run_path, relative)
            canonical_path = _contained_artifact_path(
                self.run_path,
                M5_SCORECARD_REPORT_PATH,
            )
            _write_bytes_exclusive(pending_path, payload)
            os.chmod(pending_path, 0o600, follow_symlinks=False)
            try:
                os.link(
                    pending_path,
                    canonical_path,
                    follow_symlinks=False,
                )
                _fsync_directory(canonical_path.parent)
            except FileExistsError:
                raise
            except OSError as exc:
                raise M5ResultStoreError(
                    "could not exclusively publish the scorecard report"
                ) from exc
            if (
                pending_path.stat().st_dev,
                pending_path.stat().st_ino,
            ) != (
                canonical_path.stat().st_dev,
                canonical_path.stat().st_ino,
            ):
                raise M5ResultStoreIntegrityError(
                    "canonical report is not the immutable pending file"
                )
            digest, size = _file_sha256(canonical_path)
            record = ScorecardReportRecord(
                path=M5_SCORECARD_REPORT_PATH,
                media_type=M5_SCORECARD_REPORT_MEDIA_TYPE,
                renderer_version=M5_SCORECARD_RENDERER_VERSION,
                size_bytes=size,
                sha256=digest,
            )
            self._scorecard_report = record
            return record
        except BaseException:
            self._poison("scorecard_report_write_failed")
            raise

    def read_dataset(self, dataset: str) -> pa.Table:
        """Read currently written canonical parts after rechecking their hashes."""

        if dataset not in _DATASET_NAMES:
            raise KeyError(dataset)
        records = tuple(
            record
            for record in self.artifacts
            if record.dataset == dataset
        )
        if not records:
            raise M5ResultStoreStateError(f"{dataset} has not been written")
        tables: list[pa.Table] = []
        for record in records:
            path = _contained_artifact_path(self.run_path, record.path)
            digest, size = _file_sha256(path)
            if digest != record.sha256 or size != record.size_bytes:
                raise M5ResultStoreIntegrityError(
                    f"stored {record.path} no longer matches its part hash"
                )
            table = _read_and_validate_table(path, dataset)
            if table.num_rows != record.rows:
                raise M5ResultStoreIntegrityError(
                    f"stored {record.path} row count changed"
                )
            tables.append(table)
        if len(tables) == 1:
            return tables[0]
        return pa.concat_tables(tables)

    def fail(self, reason_code: str) -> Path:
        """Permanently mark an active run failed without deleting its artifacts."""

        self._ensure_active()
        reason = _lower_name(reason_code, name="reason_code")
        self._poison(reason)
        return self.run_path / "FAILURE.json"

    def finalize(
        self,
        *,
        provenance: M5RunProvenance,
    ) -> Path:
        """Prepare a verified manifest and immediately commit ``SUCCESS``."""

        prepared = self.prepare_finalization(provenance=provenance)
        if self.row_accounting_profile == OFFICIAL_M5_PROFILE:
            self.mark_committed_for_verification(prepared)
        return self.commit_finalization(prepared)

    def prepare_finalization(
        self,
        *,
        provenance: M5RunProvenance,
    ) -> PreparedM5Finalization:
        """Validate all bytes and publish the manifest without creating success.

        The split lifecycle lets a command restore and inspect an fd-level terminal
        capture after every fallible validation step.  Only
        :meth:`commit_finalization` may create ``SUCCESS``.
        """

        self._ensure_active()
        if not isinstance(provenance, M5RunProvenance):
            raise TypeError("provenance must be M5RunProvenance")
        try:
            preflight_summary = self._preflight_finalization(provenance)
            self._finalizing = True
            _write_bytes_exclusive(
                self.run_path / "FINALIZING",
                b"FINALIZING\n",
            )
            if self.row_accounting_profile == OFFICIAL_M5_PROFILE:
                if (
                    preflight_summary is None
                    or self._determinism_receipt is None
                ):
                    raise M5ResultStoreIntegrityError(
                        "official finalization lost its streamed preflight"
                    )
                records = self.artifacts
                _verify_artifact_bytes(self.run_path, records)
                supplemental = self._verified_supplemental_artifacts()
                scan_summary = preflight_summary
                report = self._verified_scorecard_report(
                    records,
                    scorecard_rows=scan_summary.scorecard_rows,
                )
            else:
                records = self._verify_artifacts()
                scan_summary = None
                supplemental = ()
                report = self._verified_scorecard_report(records)
            _validate_run_members(
                self.run_path,
                records,
                report,
                allowed_controls={"FINALIZING"},
                supplemental_records=supplemental,
            )
            payload = self._manifest_payload(records, provenance)
            manifest_path = self.run_path / "evaluation-manifest.json"
            manifest_bytes = _canonical_json_bytes(payload)
            _write_bytes_exclusive(manifest_path, manifest_bytes)
            # Guard the small manifest-to-success interval against external mutation.
            if self.row_accounting_profile == OFFICIAL_M5_PROFILE:
                if (
                    scan_summary is None
                    or self._determinism_receipt is None
                ):
                    raise M5ResultStoreIntegrityError(
                        "official scan summary disappeared"
                    )
                _verify_artifact_bytes(self.run_path, records)
                self._verified_supplemental_artifacts()
                self._verified_scorecard_report(
                    records,
                    scorecard_rows=scan_summary.scorecard_rows,
                )
            else:
                self._verify_artifacts()
                self._verified_scorecard_report(records)
            _validate_run_members(
                self.run_path,
                records,
                report,
                allowed_controls={"FINALIZING", "evaluation-manifest.json"},
                supplemental_records=supplemental,
            )
        except BaseException:
            self._poison("finalization_failed")
            raise
        prepared = PreparedM5Finalization(
            run_path=self.run_path,
            _nonce=object(),
        )
        self._prepared_finalization = prepared
        self._prepared_manifest_bytes = manifest_bytes
        return prepared

    def commit_finalization(
        self,
        prepared: PreparedM5Finalization,
    ) -> Path:
        """Exclusively create ``SUCCESS`` for this exact prepared capability.

        Creation of the exact marker is the irreversible point of no return.  A
        close or parent-directory fsync can raise after those bytes already exist;
        in that case the run must reconcile as success rather than creating a
        contradictory failure record.
        """

        if (
            not isinstance(prepared, PreparedM5Finalization)
            or prepared is not self._prepared_finalization
            or prepared.run_path != self.run_path
            or not self._finalizing
            or self._failed
            or self._successful
            or os.path.lexists(self.run_path / "FAILURE.json")
            or _path_kind(self.run_path / "FINALIZING") != "file"
            or _path_kind(self.run_path / "evaluation-manifest.json") != "file"
            or (
                self.row_accounting_profile == OFFICIAL_M5_PROFILE
                and (
                    not self._committed_for_verification
                    or not _regular_file_has_exact_bytes(
                        self.run_path / "COMMITTED",
                        b"COMMITTED\n",
                    )
                )
            )
            or os.path.lexists(self.run_path / "SUCCESS")
        ):
            raise M5ResultStoreStateError(
                "only the active prepared finalization can become success"
            )
        success_path = self.run_path / "SUCCESS"
        try:
            self._verify_terminal_byte_seal()
            _write_bytes_exclusive(success_path, b"SUCCESS\n")
            self._successful = True
            self._finalizing = False
            self._committed_for_verification = False
            self._prepared_finalization = None
            self._prepared_manifest_bytes = None
            return self.run_path
        except BaseException:
            if _regular_file_has_exact_bytes(
                success_path,
                b"SUCCESS\n",
            ):
                self._successful = True
                self._finalizing = False
                self._committed_for_verification = False
                self._prepared_finalization = None
                self._prepared_manifest_bytes = None
                return self.run_path
            self._poison("finalization_failed")
            raise

    def _verify_terminal_byte_seal(self) -> None:
        """Recheck every registered byte immediately before ``SUCCESS``."""

        expected_manifest = self._prepared_manifest_bytes
        manifest_path = self.run_path / "evaluation-manifest.json"
        if (
            type(expected_manifest) is not bytes
            or not _regular_file_has_exact_bytes(
                self.run_path / "FINALIZING",
                b"FINALIZING\n",
            )
            or _path_kind(manifest_path) != "file"
            or _read_regular_bytes(manifest_path) != expected_manifest
        ):
            raise M5ResultStoreIntegrityError(
                "prepared finalization bytes changed before success"
            )
        records = self.artifacts
        _verify_artifact_bytes(self.run_path, records)
        if self.row_accounting_profile == OFFICIAL_M5_PROFILE:
            if not _regular_file_has_exact_bytes(
                self.run_path / "COMMITTED",
                b"COMMITTED\n",
            ):
                raise M5ResultStoreIntegrityError(
                    "official commit checkpoint changed before success"
                )
            supplemental = self._verified_supplemental_artifacts()
            allowed_controls = {
                "COMMITTED",
                "FINALIZING",
                "evaluation-manifest.json",
            }
        else:
            supplemental = ()
            allowed_controls = {
                "FINALIZING",
                "evaluation-manifest.json",
            }
        report = self._verified_scorecard_report(records)
        _validate_run_members(
            self.run_path,
            records,
            report,
            allowed_controls=allowed_controls,
            supplemental_records=supplemental,
        )

    def mark_committed_for_verification(
        self,
        prepared: PreparedM5Finalization,
    ) -> Path:
        """Create an abortable in-process checkpoint before verification.

        ``COMMITTED`` is not success.  The prepared capability remains active so
        verification failure can still create ``FAILURE.json`` and can never be
        mistaken for an accepted run.
        """

        committed_path = self.run_path / "COMMITTED"
        if (
            self.row_accounting_profile != OFFICIAL_M5_PROFILE
            or not isinstance(prepared, PreparedM5Finalization)
            or prepared is not self._prepared_finalization
            or prepared.run_path != self.run_path
            or not self._finalizing
            or self._failed
            or self._successful
            or self._committed_for_verification
            or os.path.lexists(self.run_path / "FAILURE.json")
            or os.path.lexists(self.run_path / "SUCCESS")
            or os.path.lexists(committed_path)
            or _path_kind(self.run_path / "FINALIZING") != "file"
            or _path_kind(self.run_path / "evaluation-manifest.json") != "file"
        ):
            raise M5ResultStoreStateError(
                "only an active prepared official run can be committed "
                "for verification"
            )
        try:
            _write_bytes_exclusive(committed_path, b"COMMITTED\n")
            self._committed_for_verification = True
            return self.run_path
        except BaseException:
            if _regular_file_has_exact_bytes(
                committed_path,
                b"COMMITTED\n",
            ):
                self._committed_for_verification = True
                return self.run_path
            self._poison("finalization_failed")
            raise

    def abort_finalization(
        self,
        prepared: PreparedM5Finalization,
        reason_code: str,
    ) -> Path:
        """Permanently fail one prepared run without deleting its evidence."""

        if (
            not isinstance(prepared, PreparedM5Finalization)
            or prepared is not self._prepared_finalization
            or prepared.run_path != self.run_path
            or not self._finalizing
            or self._successful
            or os.path.lexists(self.run_path / "SUCCESS")
        ):
            raise M5ResultStoreStateError(
                "only the active prepared finalization can be aborted"
            )
        reason = _lower_name(reason_code, name="reason_code")
        self._poison(reason)
        self._prepared_finalization = None
        self._prepared_manifest_bytes = None
        return self.run_path / "FAILURE.json"

    def _write_supplemental_artifact(
        self,
        kind: str,
        payload: Mapping[str, Any],
    ) -> SupplementalArtifactRecord:
        if kind not in _SUPPLEMENTAL_PATHS:
            raise ValueError("supplemental artifact kind is unsupported")
        if kind in self._supplemental_artifacts:
            raise FileExistsError(
                f"supplemental artifact {kind!r} already exists"
            )
        path_text = _SUPPLEMENTAL_PATHS[kind]
        relative = Path(path_text)
        pending_path = _pending_path(self.run_path, relative)
        canonical_path = _contained_artifact_path(
            self.run_path,
            path_text,
        )
        encoded = _canonical_json_bytes(payload)
        _write_bytes_exclusive(pending_path, encoded)
        try:
            os.link(
                pending_path,
                canonical_path,
                follow_symlinks=False,
            )
            _fsync_directory(canonical_path.parent)
        except FileExistsError:
            raise
        except OSError as exc:
            raise M5ResultStoreError(
                "could not exclusively publish a supplemental artifact"
            ) from exc
        pending_stat = pending_path.stat()
        canonical_stat = canonical_path.stat()
        if (
            pending_stat.st_dev,
            pending_stat.st_ino,
        ) != (
            canonical_stat.st_dev,
            canonical_stat.st_ino,
        ):
            raise M5ResultStoreIntegrityError(
                "supplemental artifact is not its immutable pending file"
            )
        digest, size = _file_sha256(canonical_path)
        record = SupplementalArtifactRecord(
            kind=kind,
            path=path_text,
            schema_version=str(payload["schema_version"]),
            size_bytes=size,
            sha256=digest,
        )
        self._supplemental_artifacts[kind] = record
        return record

    def _write_dataset(
        self,
        dataset: str,
        relative: Path,
        rows: Iterable[Mapping[str, Any]],
    ) -> ArtifactRecord:
        self._ensure_active()
        canonical_text = relative.as_posix()
        if canonical_text in self._artifacts:
            self._poison("artifact_exists")
            raise FileExistsError(f"artifact {canonical_text} already exists")
        try:
            normalized, keys = _normalize_rows(dataset, rows)
            duplicate = self._keys[dataset].intersection(keys)
            if duplicate:
                raise M5ResultStoreIntegrityError(
                    f"{dataset} contains a duplicate canonical key"
                )
            table = pa.Table.from_pylist(
                normalized,
                schema=M5_RESULT_SCHEMAS[dataset],
            )
            table.validate(full=True)
            if not table.schema.equals(
                M5_RESULT_SCHEMAS[dataset],
                check_metadata=True,
            ):
                raise M5ResultStoreIntegrityError(
                    f"{dataset} did not materialize with its fixed schema"
                )
            pending_path = _pending_path(self.run_path, relative)
            canonical_path = _contained_artifact_path(
                self.run_path,
                canonical_text,
            )
            _write_parquet_pending(pending_path, table)
            digest, size = _file_sha256(pending_path)
            try:
                os.link(
                    pending_path,
                    canonical_path,
                    follow_symlinks=False,
                )
                _fsync_directory(canonical_path.parent)
            except FileExistsError:
                raise
            except OSError as exc:
                raise M5ResultStoreError(
                    "could not exclusively publish a Parquet part"
                ) from exc
            if (
                pending_path.stat().st_dev,
                pending_path.stat().st_ino,
            ) != (
                canonical_path.stat().st_dev,
                canonical_path.stat().st_ino,
            ):
                raise M5ResultStoreIntegrityError(
                    "canonical artifact is not the immutable pending part"
                )
            record = ArtifactRecord(
                dataset=dataset,
                path=canonical_text,
                rows=table.num_rows,
                size_bytes=size,
                sha256=digest,
            )
            self._artifacts[canonical_text] = record
            self._keys[dataset].update(keys)
            self._row_counts[dataset] += table.num_rows
            return record
        except BaseException:
            self._poison(f"{dataset.replace('-', '_')}_write_failed")
            raise

    def _ensure_active(self) -> None:
        if self._successful or os.path.lexists(self.run_path / "SUCCESS"):
            raise M5ResultStoreStateError("a successful run is immutable")
        if (
            self._failed
            or os.path.lexists(self.run_path / "FAILURE.json")
        ):
            raise M5ResultStoreStateError("a failed run cannot become success")
        if (
            self._finalizing
            or os.path.lexists(self.run_path / "FINALIZING")
        ):
            raise M5ResultStoreStateError(
                "an interrupted finalization cannot be resumed"
            )
        if os.path.lexists(self.run_path / "evaluation-manifest.json"):
            raise M5ResultStoreStateError(
                "a final manifest already exists and cannot be replaced"
            )

    def _poison(self, reason_code: str) -> None:
        self._failed = True
        reason = (
            reason_code
            if isinstance(reason_code, str)
            and _LOWER_NAME.fullmatch(reason_code) is not None
            else "result_store_failed"
        )
        path = self.run_path / "FAILURE.json"
        if os.path.lexists(path):
            return
        payload = {
            "complete": False,
            "reason_code": reason,
            "schema_version": M5_RESULT_STORE_SCHEMA_VERSION,
        }
        try:
            _write_bytes_exclusive(path, _canonical_json_bytes(payload))
        except (OSError, M5ResultStoreError):
            # FINALIZING and/or the in-memory terminal bit still prevent promotion.
            pass

    def _preflight_finalization(
        self,
        provenance: M5RunProvenance,
    ) -> _OfficialScanSummary | None:
        expected = self.expected_rows.to_dict()
        if self._row_counts != expected:
            raise M5ResultStoreIntegrityError(
                "result row accounting does not match the expected matrix"
        )
        _validate_artifact_layout(self.artifacts)
        self._verify_artifacts()
        if self.row_accounting_profile == OFFICIAL_M5_PROFILE:
            supplemental = self._verified_supplemental_artifacts()
            if (
                self._parity_order_receipt is None
                or self._determinism_receipt is None
                or provenance.parity_order_fingerprint_sha256
                != self._parity_order_receipt.ordered_membership_sha256
                or provenance.parity_order_version
                != self._parity_order_receipt.order_version
            ):
                raise M5ResultStoreIntegrityError(
                    "official parity-order receipt differs from provenance"
                )
            scan_summary = _scan_official_artifacts(
                self.run_path,
                self.artifacts,
                self._determinism_receipt,
            )
            report = self._verified_scorecard_report(
                self.artifacts,
                scorecard_rows=scan_summary.scorecard_rows,
            )
            _validate_official_provenance_contract(provenance)
            _validate_official_source_binding(
                self.project_root,
                provenance,
            )
        else:
            supplemental = ()
            report = self._verified_scorecard_report(self.artifacts)
            rows = _read_all_artifact_rows(
                self.run_path,
                self.artifacts,
            )
            _validate_scorecards_derived_from_rows(rows)
            scan_summary = None
        observed_source_fingerprint = executable_source_fingerprint(
            self.project_root,
            provenance.executable_source_paths,
        )
        if (
            observed_source_fingerprint
            != provenance.executable_source_fingerprint_sha256
        ):
            raise M5ResultStoreIntegrityError(
                "executable-source fingerprint does not match the bound paths"
            )
        _validate_run_members(
            self.run_path,
            self.artifacts,
            report,
            allowed_controls=set(),
            supplemental_records=supplemental,
        )
        return scan_summary

    def _verify_artifacts(self) -> tuple[ArtifactRecord, ...]:
        verified = self.artifacts
        actual = _verify_artifact_records(
            self.run_path,
            verified,
            streaming=(
                self.row_accounting_profile == OFFICIAL_M5_PROFILE
            ),
        )
        if actual != self._row_counts:
            raise M5ResultStoreIntegrityError(
                "artifact row counts differ from writer accounting"
            )
        return verified

    def _verified_scorecard_report(
        self,
        records: Sequence[ArtifactRecord],
        *,
        scorecard_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> ScorecardReportRecord:
        if self._scorecard_report is None:
            raise M5ResultStoreIntegrityError(
                "human-readable scorecard report is missing"
            )
        _verify_scorecard_report(
            self.run_path,
            self._scorecard_report,
            self.row_accounting_profile,
            records,
            scorecard_rows=scorecard_rows,
        )
        return self._scorecard_report

    def _verified_supplemental_artifacts(
        self,
    ) -> tuple[SupplementalArtifactRecord, ...]:
        records = self.supplemental_artifacts
        _validate_supplemental_layout(records)
        observed = {
            record.kind: _verify_supplemental_artifact(
                self.run_path,
                record,
            )
            for record in records
        }
        if (
            observed.get(PARITY_ORDER_RECEIPT)
            != self._parity_order_receipt
            or observed.get(DETERMINISM_RECEIPT)
            != self._determinism_receipt
        ):
            raise M5ResultStoreIntegrityError(
                "official receipt bytes differ from the writer state"
            )
        return records

    def _manifest_payload(
        self,
        records: tuple[ArtifactRecord, ...],
        provenance: M5RunProvenance,
    ) -> dict[str, Any]:
        payload = {
            "actual_rows": dict(self._row_counts),
            "artifacts": [record.to_dict() for record in records],
            "catalog_fingerprints": _catalog_fingerprints(),
            "complete": True,
            "expected_rows": self.expected_rows.to_dict(),
            "hash_policy": {
                "algorithm": "sha256",
                "manifest_self_hash": False,
            },
            "provenance": provenance.to_dict(),
            "result_path": self.project_relative_path.as_posix(),
            "row_accounting_profile": self.row_accounting_profile,
            "run_name": self.run_name,
            "scorecard_report": (
                self._scorecard_report.to_dict()
                if self._scorecard_report is not None
                else None
            ),
            "schema_fingerprints": {
                name: _schema_fingerprint(M5_RESULT_SCHEMAS[name])
                for name in _DATASET_NAMES
            },
            "schema_version": M5_RESULT_STORE_SCHEMA_VERSION,
            "write_mode": "exclusive_immutable_pending_hardlink",
        }
        if self.row_accounting_profile == OFFICIAL_M5_PROFILE:
            supplemental = self._verified_supplemental_artifacts()
            payload["official_artifacts"] = [
                record.to_dict() for record in supplemental
            ]
        return payload


def scorecard_row_from_result(
    result: PairedCellResult,
) -> dict[str, Any]:
    """Flatten one validated statistics result into the fixed scorecard schema."""

    if not isinstance(result, PairedCellResult):
        raise TypeError("result must be a PairedCellResult")

    def band_fields(prefix: str, band: Any) -> dict[str, float | None]:
        if band is None:
            return {
                f"{prefix}_level": None,
                f"{prefix}_lower": None,
                f"{prefix}_upper": None,
            }
        return {
            f"{prefix}_level": band.level,
            f"{prefix}_lower": band.lower,
            f"{prefix}_upper": band.upper,
        }

    row: dict[str, Any] = {
        "asymmetric_component_n": result.asymmetric_component_n,
        "asymmetric_missing_n": result.asymmetric_missing_n,
        "asymmetric_reason_n": result.asymmetric_reason_n,
        "base_seed": M5_BASE_SEED,
        "both_missing_n": result.both_missing_n,
        "cohort_n": result.cohort_n,
        "direction": result.spec.direction,
        "directional_language_allowed": result.directional_language_allowed,
        "eligible_components_a": result.eligible_components_a,
        "eligible_components_b": result.eligible_components_b,
        "excluded_n": result.excluded_n,
        "favorable_proportion": result.favorable_proportion,
        "index_dtype": "int64",
        "metric_name": result.spec.metric_name,
        "metric_version": result.spec.metric_version,
        "missing_reasons_a_json": json.dumps(
            dict(result.missing_reasons_a),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
        "missing_reasons_b_json": json.dumps(
            dict(result.missing_reasons_b),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
        "nonzero_effect_n": result.nonzero_effect_n,
        "oriented_mean_advantage": result.oriented_mean_advantage,
        "paired_n": result.paired_n,
        "policy_a": result.spec.contrast.policy_a,
        "policy_a_mean": result.policy_a_mean,
        "policy_a_median": result.policy_a_median,
        "policy_b": result.spec.contrast.policy_b,
        "policy_b_mean": result.policy_b_mean,
        "policy_b_median": result.policy_b_median,
        "quantile_method": "linear",
        "raw_mean_difference": result.raw_mean_difference,
        "raw_median_difference": result.raw_median_difference,
        "resamples": result.resampling_key.resamples,
        "resampling_digest_words": list(result.resampling_key.digest_words),
        "resampling_key_json": result.resampling_key.canonical_json,
        "resampling_sha256": result.resampling_key.sha256,
        "rng": "PCG64",
        "slice_name": result.spec.slice_name,
        "slice_version": result.spec.slice_version,
        "source_pairing_complete": result.source_pairing_complete,
        "standardized_signal_to_heterogeneity": (
            result.standardized_signal_to_heterogeneity
        ),
        "status": result.status,
        "total_components_a": result.total_components_a,
        "total_components_b": result.total_components_b,
        "valid_a_n": result.valid_a_n,
        "valid_b_n": result.valid_b_n,
        "value_unit": result.spec.value_unit,
    }
    row.update(
        band_fields("pointwise", result.pointwise_stability_band)
    )
    row.update(
        band_fields(
            "adjusted",
            result.adjusted_primary_stability_band,
        )
    )
    if set(row) != set(SCORECARDS_SCHEMA.names):
        raise RuntimeError("scorecard adapter drifted from the fixed schema")
    return _scorecard_row(row)


def official_executable_source_paths(
    project_root: str | Path,
) -> tuple[str, ...]:
    """Return the exact clean tracked source set frozen for official M5 runs."""

    root = _validated_project_root(project_root)
    top_level = _git_text(root, "rev-parse", "--show-toplevel")
    try:
        git_root = Path(top_level).resolve(strict=True)
    except OSError as exc:
        raise M5ResultStoreIntegrityError(
            "official M5 requires a valid Git worktree"
        ) from exc
    if git_root != root:
        raise M5ResultStoreIntegrityError(
            "official M5 project_root must be the Git worktree root"
        )

    status = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "-z",
    )
    if status:
        raise M5ResultStoreIntegrityError(
            "official M5 requires a clean Git worktree"
        )

    pathspecs = (*_OFFICIAL_EXECUTABLE_ROOTS, *_OFFICIAL_EXECUTABLE_FILES)
    tracked_bytes = _git_bytes(root, "ls-files", "-z", "--", *pathspecs)
    try:
        tracked = tuple(
            item.decode("utf-8", errors="strict")
            for item in tracked_bytes.split(b"\0")
            if item
        )
    except UnicodeDecodeError as exc:
        raise M5ResultStoreIntegrityError(
            "official executable-source paths must be UTF-8"
        ) from exc
    canonical = tuple(sorted(tracked))
    if (
        not canonical
        or len(set(canonical)) != len(canonical)
        or tracked != canonical
    ):
        raise M5ResultStoreIntegrityError(
            "Git did not return a unique canonical executable-source set"
        )
    tracked_set = set(canonical)
    if not set(_OFFICIAL_EXECUTABLE_FILES).issubset(tracked_set):
        raise M5ResultStoreIntegrityError(
            "an official named executable-source file is not tracked"
        )
    for root_name in _OFFICIAL_EXECUTABLE_ROOTS:
        prefix = f"{root_name}/"
        if not any(path.startswith(prefix) for path in canonical):
            raise M5ResultStoreIntegrityError(
                "an official executable-source root has no tracked files"
            )
    # The fingerprint itself checks containment, regular-file type, and symlinks.
    executable_source_fingerprint(root, canonical)
    _reject_untracked_source_candidates(root, tracked_set)
    return canonical


def executable_source_fingerprint(
    project_root: str | Path,
    paths: Sequence[str],
) -> str:
    """Hash a canonical, length-prefixed list of contained regular source files."""

    root = _validated_project_root(project_root)
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise TypeError("paths must be a sequence of project-relative strings")
    normalized = tuple(
        _safe_project_relative_source_path(path).as_posix()
        for path in paths
    )
    if (
        not normalized
        or normalized != tuple(sorted(normalized))
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError("source paths must be non-empty, unique, and sorted")
    digest = hashlib.sha256(b"evalsim-m5-executable-source-v1\0")
    for relative_text in normalized:
        relative = Path(relative_text)
        path = root.joinpath(*relative.parts)
        if (
            root not in path.parents
            or _path_kind(path) != "file"
            or path.resolve(strict=True) != path
        ):
            raise M5ResultStoreIntegrityError(
                "an executable-source path is missing, linked, or outside the project"
            )
        file_digest, _ = _file_sha256(path)
        path_bytes = relative_text.encode("utf-8", errors="strict")
        digest.update(struct.pack(">Q", len(path_bytes)))
        digest.update(path_bytes)
        digest.update(bytes.fromhex(file_digest))
    return digest.hexdigest()


def _validate_official_source_binding(
    project_root: Path,
    provenance: M5RunProvenance,
) -> None:
    expected_paths = official_executable_source_paths(project_root)
    if provenance.executable_source_paths != expected_paths:
        raise M5ResultStoreIntegrityError(
            "official M5 provenance omits or adds executable-source paths"
        )
    head = _git_text(project_root, "rev-parse", "--verify", "HEAD")
    tree = _git_text(project_root, "rev-parse", "--verify", "HEAD^{tree}")
    if provenance.git_commit != head or provenance.git_tree != tree:
        raise M5ResultStoreIntegrityError(
            "official M5 provenance is not bound to the checked-out Git tree"
        )
    branch = _git_text(project_root, "branch", "--show-current")
    upstream = _git_text(
        project_root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
    )
    upstream_head = _git_text(
        project_root,
        "rev-parse",
        "--verify",
        "@{upstream}",
    )
    if (
        branch != "main"
        or not upstream.endswith("/main")
        or upstream_head != head
    ):
        raise M5ResultStoreIntegrityError(
            "official M5 requires checked-out main at its pushed upstream commit"
        )


def _validate_official_provenance_contract(
    provenance: M5RunProvenance,
) -> None:
    if not provenance.has_official_extensions:
        raise M5ResultStoreIntegrityError(
            "official M5 provenance omits M4 reuse or parity-order evidence"
        )
    if provenance.selected_order_version != M5_M4_SELECTED_ORDER_VERSION:
        raise M5ResultStoreIntegrityError(
            "official M5 provenance uses the wrong accepted-M4 order"
        )
    reference_spec = provenance.simulator_specs[WAYMAX_EXACT_LOG_NAME]
    if (
        reference_spec["version"] != WAYMAX_REFERENCE_VERSION
        or reference_spec["deterministic"] is not True
        or reference_spec["execution_role"] != "reference"
        or dict(reference_spec["parameters"])
        != dict(OFFICIAL_WAYMAX_REFERENCE_PARAMETERS)
    ):
        raise M5ResultStoreIntegrityError(
            "official Waymax exact-log execution spec is not frozen or executed"
        )


def _validate_recorded_official_source_binding(
    project_root: Path,
    provenance: M5RunProvenance,
) -> None:
    commit = provenance.git_commit
    tree = _git_text(
        project_root,
        "rev-parse",
        "--verify",
        f"{commit}^{{tree}}",
    )
    if tree != provenance.git_tree:
        raise M5ResultStoreIntegrityError(
            "recorded official Git commit does not match its bound tree"
        )
    entries = _official_git_source_entries(project_root, commit)
    paths = tuple(path for path, _ in entries)
    if paths != provenance.executable_source_paths:
        raise M5ResultStoreIntegrityError(
            "recorded official commit has a different executable-source set"
        )

    digest = hashlib.sha256(b"evalsim-m5-executable-source-v1\0")
    for relative_text, object_id in entries:
        blob = _git_bytes(project_root, "cat-file", "blob", object_id)
        path_bytes = relative_text.encode("utf-8", errors="strict")
        digest.update(struct.pack(">Q", len(path_bytes)))
        digest.update(path_bytes)
        digest.update(hashlib.sha256(blob).digest())
    if digest.hexdigest() != provenance.executable_source_fingerprint_sha256:
        raise M5ResultStoreIntegrityError(
            "recorded official executable-source fingerprint is invalid"
        )


def _official_git_source_entries(
    project_root: Path,
    revision: str,
) -> tuple[tuple[str, str], ...]:
    pathspecs = (*_OFFICIAL_EXECUTABLE_ROOTS, *_OFFICIAL_EXECUTABLE_FILES)
    payload = _git_bytes(
        project_root,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        revision,
        "--",
        *pathspecs,
    )
    entries: list[tuple[str, str]] = []
    for raw_record in payload.split(b"\0"):
        if not raw_record:
            continue
        try:
            metadata, raw_path = raw_record.split(b"\t", 1)
            mode, object_type, raw_object_id = metadata.split(b" ", 2)
            relative = raw_path.decode("utf-8", errors="strict")
            object_id = raw_object_id.decode("ascii", errors="strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise M5ResultStoreIntegrityError(
                "recorded Git source tree is malformed"
            ) from exc
        if (
            mode not in {b"100644", b"100755"}
            or object_type != b"blob"
            or len(object_id) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in object_id)
        ):
            raise M5ResultStoreIntegrityError(
                "official executable-source entries must be regular Git blobs"
            )
        try:
            canonical_path = _safe_project_relative_source_path(
                relative
            ).as_posix()
        except (TypeError, ValueError) as exc:
            raise M5ResultStoreIntegrityError(
                "recorded Git source path is not canonical"
            ) from exc
        entries.append((canonical_path, object_id))
    canonical = tuple(sorted(entries, key=lambda item: item[0]))
    if (
        not canonical
        or tuple(entries) != canonical
        or len({path for path, _ in canonical}) != len(canonical)
    ):
        raise M5ResultStoreIntegrityError(
            "recorded Git source tree is not a unique canonical path set"
        )
    paths = {path for path, _ in canonical}
    if not set(_OFFICIAL_EXECUTABLE_FILES).issubset(paths):
        raise M5ResultStoreIntegrityError(
            "recorded Git tree omits a named executable-source file"
        )
    for root_name in _OFFICIAL_EXECUTABLE_ROOTS:
        prefix = f"{root_name}/"
        if not any(path.startswith(prefix) for path in paths):
            raise M5ResultStoreIntegrityError(
                "recorded Git tree omits an executable-source root"
            )
    return canonical


def _git_bytes(project_root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ("git", "-C", os.fspath(project_root), *arguments),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise M5ResultStoreIntegrityError(
            "Git is required for official executable-source binding"
        ) from exc
    if completed.returncode != 0:
        raise M5ResultStoreIntegrityError(
            "Git could not validate official executable-source binding"
        )
    return completed.stdout


def _git_text(project_root: Path, *arguments: str) -> str:
    payload = _git_bytes(project_root, *arguments)
    try:
        value = payload.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise M5ResultStoreIntegrityError(
            "Git returned non-UTF-8 provenance"
        ) from exc
    if not value or "\n" in value or "\r" in value:
        raise M5ResultStoreIntegrityError(
            "Git returned invalid provenance"
        )
    return value


def _reject_untracked_source_candidates(
    project_root: Path,
    tracked_paths: set[str],
) -> None:
    cache_directories = {
        ".hypothesis",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
    }
    executable_suffixes = {
        ".bash",
        ".c",
        ".cc",
        ".cpp",
        ".fish",
        ".h",
        ".hpp",
        ".json",
        ".lock",
        ".py",
        ".pyi",
        ".pyx",
        ".sh",
        ".toml",
        ".yaml",
        ".yml",
        ".zsh",
    }

    for root_name in _OFFICIAL_EXECUTABLE_ROOTS:
        source_root = project_root.joinpath(*Path(root_name).parts)
        if (
            _path_kind(source_root) != "directory"
            or source_root.resolve(strict=True) != source_root
        ):
            raise M5ResultStoreIntegrityError(
                "an official executable-source root is missing or linked"
            )
        for directory, directory_names, file_names in os.walk(
            source_root,
            topdown=True,
            followlinks=False,
        ):
            directory_path = Path(directory)
            retained_directories: list[str] = []
            for name in directory_names:
                candidate = directory_path / name
                if name in cache_directories:
                    continue
                if candidate.is_symlink():
                    raise M5ResultStoreIntegrityError(
                        "official executable-source roots cannot contain symlinks"
                    )
                retained_directories.append(name)
            directory_names[:] = retained_directories
            for name in file_names:
                candidate = directory_path / name
                relative = candidate.relative_to(project_root).as_posix()
                if relative in tracked_paths:
                    continue
                try:
                    mode = candidate.lstat().st_mode
                except OSError as exc:
                    raise M5ResultStoreIntegrityError(
                        "could not inspect an executable-source candidate"
                    ) from exc
                if stat.S_ISLNK(mode):
                    raise M5ResultStoreIntegrityError(
                        "official executable-source roots cannot contain symlinks"
                    )
                if (
                    stat.S_ISREG(mode)
                    and (
                        candidate.suffix.lower() in executable_suffixes
                        or mode & 0o111
                    )
                ):
                    raise M5ResultStoreIntegrityError(
                        "an untracked executable file exists in a frozen source root"
                    )


def verify_m5_result_store(
    project_root: str | Path,
    run_name: str,
    *,
    allow_data_free: bool = False,
) -> VerifiedM5ResultStore:
    """Fail closed unless the named local run is a complete immutable result set."""

    return _verify_m5_result_store(
        project_root,
        run_name,
        allow_data_free=allow_data_free,
        state="success",
    )


def verify_prepared_m5_result_store(
    project_root: str | Path,
    run_name: str,
    *,
    allow_data_free: bool = False,
) -> VerifiedM5ResultStore:
    """Independently verify a prepared run before irreversible ``SUCCESS``.

    A prepared run must contain the exact ``FINALIZING`` marker and final
    manifest, must not yet contain ``SUCCESS`` or ``FAILURE.json``, and must pass
    every artifact, provenance, receipt, source, statistics, and report check
    applied to a successful run.
    """

    return _verify_m5_result_store(
        project_root,
        run_name,
        allow_data_free=allow_data_free,
        state="prepared",
    )


def verify_committed_m5_result_store(
    project_root: str | Path,
    run_name: str,
    *,
    allow_data_free: bool = False,
) -> VerifiedM5ResultStore:
    """Verify the abortable in-process checkpoint before ``SUCCESS``."""

    return _verify_m5_result_store(
        project_root,
        run_name,
        allow_data_free=allow_data_free,
        state="committed",
    )


def _verify_m5_result_store(
    project_root: str | Path,
    run_name: str,
    *,
    allow_data_free: bool,
    state: str,
) -> VerifiedM5ResultStore:
    """Shared complete-store verifier for prepared and committed states."""

    if type(allow_data_free) is not bool:
        raise TypeError("allow_data_free must be a boolean")
    if state not in {"prepared", "committed", "success"}:
        raise ValueError("state must be prepared, committed, or success")
    root = _validated_project_root(project_root)
    name = _validated_run_name(run_name)
    run_path = root / "outputs" / "m5" / name
    if (
        _path_kind(run_path) != "directory"
        or run_path.resolve(strict=True) != run_path
    ):
        raise M5ResultStoreIntegrityError("M5 result path is missing or unsafe")
    if os.path.lexists(run_path / "FAILURE.json"):
        raise M5ResultStoreIntegrityError("failed runs cannot verify as success")
    controls = [("FINALIZING", b"FINALIZING\n")]
    if state == "prepared":
        if os.path.lexists(run_path / "COMMITTED") or os.path.lexists(
            run_path / "SUCCESS"
        ):
            raise M5ResultStoreIntegrityError(
                "a prepared M5 result store cannot already be committed"
            )
    elif state == "committed":
        controls.append(("COMMITTED", b"COMMITTED\n"))
        if os.path.lexists(run_path / "SUCCESS"):
            raise M5ResultStoreIntegrityError(
                "a committed verification checkpoint cannot contain SUCCESS"
            )
    else:
        controls.append(("SUCCESS", b"SUCCESS\n"))
    for control, expected in controls:
        path = run_path / control
        if _path_kind(path) != "file" or path.read_bytes() != expected:
            raise M5ResultStoreIntegrityError(
                f"M5 result store has an invalid {control} marker"
            )

    manifest_path = run_path / "evaluation-manifest.json"
    if _path_kind(manifest_path) != "file":
        raise M5ResultStoreIntegrityError("final evaluation manifest is missing")
    encoded = manifest_path.read_bytes()
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M5ResultStoreIntegrityError(
            "final evaluation manifest is not canonical JSON"
        ) from exc
    if not isinstance(payload, dict) or _canonical_json_bytes(payload) != encoded:
        raise M5ResultStoreIntegrityError(
            "final evaluation manifest is not canonical JSON"
        )
    _validate_manifest_identity(payload, name)
    profile = payload["row_accounting_profile"]
    if (
        state == "committed"
        and profile != OFFICIAL_M5_PROFILE
    ):
        raise M5ResultStoreIntegrityError(
            "only official M5 runs use a committed verification checkpoint"
        )
    if (
        state == "success"
        and profile == OFFICIAL_M5_PROFILE
        and not _regular_file_has_exact_bytes(
            run_path / "COMMITTED",
            b"COMMITTED\n",
        )
    ):
        raise M5ResultStoreIntegrityError(
            "official M5 success omits its committed verification checkpoint"
        )
    if profile == DATA_FREE_TEST_PROFILE and not allow_data_free:
        raise M5ResultStoreIntegrityError(
            "production verification rejects data-free row-count overrides"
        )
    raw_scorecard_report = payload.get("scorecard_report")
    if not isinstance(raw_scorecard_report, Mapping):
        raise M5ResultStoreIntegrityError(
            "manifest scorecard report must be an object"
        )
    scorecard_report = ScorecardReportRecord.from_dict(raw_scorecard_report)

    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise M5ResultStoreIntegrityError("manifest artifacts must be an array")
    records = tuple(
        ArtifactRecord.from_dict(item)
        if isinstance(item, Mapping)
        else _raise_invalid_artifact()
        for item in raw_artifacts
    )
    if tuple(record.path for record in records) != tuple(
        sorted(record.path for record in records)
    ):
        raise M5ResultStoreIntegrityError(
            "manifest artifacts are not canonically ordered"
        )
    if len({record.path for record in records}) != len(records):
        raise M5ResultStoreIntegrityError("manifest artifact paths are not unique")
    if any(
        record.path in {"evaluation-manifest.json", "SUCCESS", "FINALIZING"}
        for record in records
    ):
        raise M5ResultStoreIntegrityError("manifest cannot hash itself or markers")
    _validate_artifact_layout(records)

    expected_raw = payload.get("expected_rows")
    actual_raw = payload.get("actual_rows")
    if (
        not isinstance(expected_raw, dict)
        or set(expected_raw) != set(_DATASET_NAMES)
        or not isinstance(actual_raw, dict)
        or set(actual_raw) != set(_DATASET_NAMES)
    ):
        raise M5ResultStoreIntegrityError("manifest row accounting is incomplete")
    expected_counts = {
        name_: _integer(
            expected_raw[name_],
            name=f"expected_rows.{name_}",
            minimum=0,
        )
        for name_ in _DATASET_NAMES
    }
    actual_counts = {
        name_: _integer(
            actual_raw[name_],
            name=f"actual_rows.{name_}",
            minimum=0,
        )
        for name_ in _DATASET_NAMES
    }
    observed_counts = {
        dataset: sum(
            record.rows for record in records if record.dataset == dataset
        )
        for dataset in _DATASET_NAMES
    }
    if expected_counts != actual_counts or actual_counts != observed_counts:
        raise M5ResultStoreIntegrityError(
            "manifest row accounting is not exact"
        )
    if (
        profile == OFFICIAL_M5_PROFILE
        and expected_counts != OFFICIAL_M5_ROW_COUNTS.to_dict()
    ):
        raise M5ResultStoreIntegrityError(
            "official M5 row accounting differs from the frozen matrix"
        )

    schema_fingerprints = payload.get("schema_fingerprints")
    expected_fingerprints = {
        dataset: _schema_fingerprint(M5_RESULT_SCHEMAS[dataset])
        for dataset in _DATASET_NAMES
    }
    if schema_fingerprints != expected_fingerprints:
        raise M5ResultStoreIntegrityError(
            "manifest schema fingerprints do not match the fixed schemas"
        )
    if payload.get("catalog_fingerprints") != _catalog_fingerprints():
        raise M5ResultStoreIntegrityError(
            "manifest catalog fingerprints do not match canonical M5 semantics"
        )
    provenance_payload = payload.get("provenance")
    if not isinstance(provenance_payload, Mapping):
        raise M5ResultStoreIntegrityError("manifest provenance must be an object")
    bound_provenance = M5RunProvenance.from_dict(provenance_payload)
    if profile == OFFICIAL_M5_PROFILE:
        _validate_official_provenance_contract(bound_provenance)
        _validate_recorded_official_source_binding(root, bound_provenance)
    elif executable_source_fingerprint(
        root,
        bound_provenance.executable_source_paths,
    ) != bound_provenance.executable_source_fingerprint_sha256:
        raise M5ResultStoreIntegrityError(
            "bound executable-source files changed after finalization"
        )

    verified_counts = _verify_artifact_records(
        run_path,
        records,
        streaming=(profile == OFFICIAL_M5_PROFILE),
    )
    if verified_counts != actual_counts:
        raise M5ResultStoreIntegrityError(
            "verified artifact rows differ from the manifest"
        )
    if profile == OFFICIAL_M5_PROFILE:
        raw_supplemental = payload.get("official_artifacts")
        if not isinstance(raw_supplemental, list):
            raise M5ResultStoreIntegrityError(
                "official supplemental artifacts must be an array"
            )
        supplemental = tuple(
            SupplementalArtifactRecord.from_dict(item)
            if isinstance(item, Mapping)
            else _raise_invalid_supplemental_artifact()
            for item in raw_supplemental
        )
        _validate_supplemental_layout(supplemental)
        supplemental_payloads = {
            record.kind: _verify_supplemental_artifact(
                run_path,
                record,
            )
            for record in supplemental
        }
        parity_receipt = supplemental_payloads[PARITY_ORDER_RECEIPT]
        determinism_receipt = supplemental_payloads[DETERMINISM_RECEIPT]
        if (
            not isinstance(parity_receipt, M5ParityOrderReceipt)
            or not isinstance(determinism_receipt, M5DeterminismReceipt)
            or bound_provenance.parity_order_version
            != parity_receipt.order_version
            or bound_provenance.parity_order_fingerprint_sha256
            != parity_receipt.ordered_membership_sha256
        ):
            raise M5ResultStoreIntegrityError(
                "official supplemental evidence differs from provenance"
            )
        scan_summary = _scan_official_artifacts(
            run_path,
            records,
            determinism_receipt,
        )
        scorecard_rows = scan_summary.scorecard_rows
    else:
        supplemental = ()
        parity_receipt = None
        determinism_receipt = None
        result_rows = _read_all_artifact_rows(run_path, records)
        _validate_scorecards_derived_from_rows(result_rows)
        scorecard_rows = None
    _verify_scorecard_report(
        run_path,
        scorecard_report,
        profile,
        records,
        scorecard_rows=scorecard_rows,
    )
    _validate_run_members(
        run_path,
        records,
        scorecard_report,
        allowed_controls=(
            {"FINALIZING", "evaluation-manifest.json"}
            | ({"COMMITTED"} if state == "committed" else set())
            | ({"SUCCESS"} if state == "success" else set())
            | (
                {"COMMITTED"}
                if state == "success" and profile == OFFICIAL_M5_PROFILE
                else set()
            )
        ),
        supplemental_records=supplemental,
    )
    return VerifiedM5ResultStore(
        run_path=run_path,
        manifest=MappingProxyType(payload),
        artifacts=records,
        scorecard_report=scorecard_report,
        supplemental_artifacts=supplemental,
        parity_order_receipt=parity_receipt,
        determinism_receipt=determinism_receipt,
    )


def _normalize_rows(
    dataset: str,
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], set[tuple[Any, ...]]]:
    if dataset not in _DATASET_NAMES:
        raise KeyError(dataset)
    if isinstance(rows, (str, bytes, Mapping)):
        raise TypeError("rows must be an iterable of row mappings")
    normalized: list[dict[str, Any]] = []
    keys: set[tuple[Any, ...]] = set()
    for raw in rows:
        row = _normalize_row(dataset, raw)
        key = tuple(row[field] for field in _KEY_FIELDS[dataset])
        if key in keys:
            raise M5ResultStoreIntegrityError(
                f"{dataset} contains a duplicate canonical key"
            )
        keys.add(key)
        normalized.append(row)
    normalized.sort(
        key=lambda row: tuple(
            row[field] for field in _KEY_FIELDS[dataset]
        )
    )
    return normalized, keys


def _normalize_row(
    dataset: str,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    if dataset not in _DATASET_NAMES:
        raise KeyError(dataset)
    if not isinstance(raw, Mapping):
        raise TypeError(f"{dataset} rows must be mappings")
    if set(raw) != set(M5_RESULT_SCHEMAS[dataset].names):
        raise ValueError(
            f"{dataset} row fields must exactly match its fixed schema"
        )
    validator = {
        METRIC_RESULTS: _metric_row,
        SLICE_MEMBERSHIP: _slice_row,
        SCORECARDS: _scorecard_row,
        WAYMAX_PARITY_SUMMARY: _parity_row,
    }[dataset]
    return validator(raw)


def _metric_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    row["cohort_index"] = _integer(
        row["cohort_index"],
        name="cohort_index",
        minimum=0,
        maximum=2**31 - 1,
    )
    row["execution_name"] = _lower_name(
        row["execution_name"],
        name="execution_name",
    )
    expected_role = _M5_EXECUTION_ROLES.get(row["execution_name"])
    if expected_role is None or row["execution_role"] != expected_role:
        raise ValueError(
            "execution name/role must match the canonical four M5 executions"
        )
    row["seed"] = _integer(
        row["seed"],
        name="seed",
        minimum=0,
        maximum=2**32 - 1,
    )
    row["metric_name"] = _lower_name(row["metric_name"], name="metric_name")
    row["metric_version"] = _version(row["metric_version"], "metric_version")
    metric_spec = M5_METRIC_SPECS.get(row["metric_name"])
    if (
        metric_spec is None
        or row["metric_version"] != metric_spec.version
    ):
        raise ValueError(
            "metric name/version must match the canonical M5 catalog"
        )
    row["valid"] = _boolean(row["valid"], "valid")
    eligible = _integer(
        row["eligible_components"],
        name="eligible_components",
        minimum=0,
    )
    total = _integer(
        row["total_components"],
        name="total_components",
        minimum=0,
    )
    if eligible > total:
        raise ValueError("eligible_components cannot exceed total_components")
    distribution = _finite_sequence(row["distribution"], "distribution")
    if len(distribution) != eligible:
        raise ValueError("distribution length must equal eligible_components")
    row["eligible_components"] = eligible
    row["total_components"] = total
    row["distribution"] = distribution
    row["details_json"] = _canonical_json_text(
        row["details_json"],
        "details_json",
        require_object=True,
    )
    if row["valid"]:
        row["value"] = _finite(row["value"], "value")
        if row["invalid_reason"] is not None or eligible < 1:
            raise ValueError(
                "valid metric rows require components and no invalid_reason"
            )
        if metric_spec.aggregation in {"mean", "rate"}:
            expected_value = math.fsum(distribution) / len(distribution)
        elif metric_spec.aggregation == "minimum":
            expected_value = min(distribution)
        else:  # pragma: no cover - protected by the frozen M5 registry
            raise RuntimeError(
                "M5 result store does not recognize the registered reducer"
            )
        if row["value"] != expected_value:
            raise ValueError(
                "metric value contradicts its registered distribution reducer"
            )
    else:
        if row["value"] is not None or eligible != 0 or distribution:
            raise ValueError(
                "invalid metric rows require null value and zero components"
            )
        row["invalid_reason"] = _lower_name(
            row["invalid_reason"],
            name="invalid_reason",
        )
        if row["invalid_reason"] not in metric_spec.invalid_reason_codes:
            raise ValueError(
                "invalid_reason is not registered for this M5 metric"
            )
    return row


def _slice_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    row["cohort_index"] = _integer(
        row["cohort_index"],
        name="cohort_index",
        minimum=0,
        maximum=2**31 - 1,
    )
    row["slice_name"] = _lower_name(row["slice_name"], name="slice_name")
    row["slice_version"] = _slice_version(row["slice_version"])
    slice_spec = _M5_SLICE_SPECS.get(row["slice_name"])
    if slice_spec is None:
        raise ValueError("slice_name is not in the canonical M5 catalog")
    row["eligible"] = _boolean(row["eligible"], "eligible")
    row["member"] = _boolean(row["member"], "member")
    if row["eligible"]:
        if row["reason"] is not None:
            raise ValueError("eligible slice rows cannot have a reason")
    else:
        if row["member"]:
            raise ValueError("ineligible slice rows cannot be members")
        row["reason"] = _lower_name(row["reason"], name="reason")
        if row["reason"] not in slice_spec.ineligible_reasons:
            raise ValueError(
                "slice reason is not registered for this M5 slice"
            )
    return row


def _scorecard_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    row["metric_name"] = _lower_name(row["metric_name"], name="metric_name")
    row["metric_version"] = _version(row["metric_version"], "metric_version")
    row["value_unit"] = _text(row["value_unit"], "value_unit")
    row["slice_name"] = _lower_name(row["slice_name"], name="slice_name")
    row["slice_version"] = _slice_version(row["slice_version"])
    row["policy_a"] = _lower_name(row["policy_a"], name="policy_a")
    row["policy_b"] = _lower_name(row["policy_b"], name="policy_b")
    try:
        cell_spec = PairedCellSpec(
            metric_name=row["metric_name"],
            metric_version=row["metric_version"],
            slice_name=row["slice_name"],
            slice_version=row["slice_version"],
            contrast=PolicyContrast(row["policy_a"], row["policy_b"]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "scorecard identity is not a canonical M5 metric/slice/contrast"
        ) from exc
    if (
        row["value_unit"] != cell_spec.value_unit
        or row["direction"] != cell_spec.direction
    ):
        raise ValueError(
            "scorecard unit/direction drifted from the canonical metric spec"
        )

    count_fields = (
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
    for name in count_fields:
        maximum = 2**32 - 1 if name == "base_seed" else None
        minimum = 1 if name == "resamples" else 0
        row[name] = _integer(
            row[name],
            name=name,
            minimum=minimum,
            maximum=maximum,
        )
    if row["paired_n"] + row["excluded_n"] != row["cohort_n"]:
        raise ValueError("paired_n plus excluded_n must equal cohort_n")
    if (
        row["valid_a_n"] > row["cohort_n"]
        or row["valid_b_n"] > row["cohort_n"]
        or row["paired_n"] > min(row["valid_a_n"], row["valid_b_n"])
    ):
        raise ValueError("valid policy counts cannot exceed cohort_n")
    if (
        row["both_missing_n"] + row["asymmetric_missing_n"]
        != row["excluded_n"]
    ):
        raise ValueError(
            "excluded_n must equal both-missing plus asymmetric-missing rows"
        )
    if (
        row["valid_a_n"] + row["valid_b_n"]
        != 2 * row["paired_n"] + row["asymmetric_missing_n"]
    ):
        raise ValueError(
            "valid counts do not match paired and asymmetric missingness"
        )
    if row["asymmetric_reason_n"] > row["both_missing_n"]:
        raise ValueError("asymmetric_reason_n cannot exceed both_missing_n")
    if row["asymmetric_component_n"] > row["cohort_n"]:
        raise ValueError("asymmetric_component_n cannot exceed cohort_n")
    if (
        row["eligible_components_a"] > row["total_components_a"]
        or row["eligible_components_b"] > row["total_components_b"]
    ):
        raise ValueError("eligible component totals cannot exceed component totals")
    if (
        row["eligible_components_a"] < row["valid_a_n"]
        or row["eligible_components_b"] < row["valid_b_n"]
    ):
        raise ValueError(
            "each valid scenario scalar requires an eligible component"
        )

    if row["nonzero_effect_n"] is not None:
        row["nonzero_effect_n"] = _integer(
            row["nonzero_effect_n"],
            name="nonzero_effect_n",
            minimum=0,
        )
        if row["nonzero_effect_n"] > row["paired_n"]:
            raise ValueError("nonzero_effect_n cannot exceed paired_n")
    for name in (
        "raw_mean_difference",
        "raw_median_difference",
        "oriented_mean_advantage",
        "favorable_proportion",
        "standardized_signal_to_heterogeneity",
        "policy_a_mean",
        "policy_a_median",
        "policy_b_mean",
        "policy_b_median",
    ):
        if row[name] is not None:
            row[name] = _finite(row[name], name)
    if (
        row["favorable_proportion"] is not None
        and not 0.0 <= row["favorable_proportion"] <= 1.0
    ):
        raise ValueError("favorable_proportion must lie in [0, 1]")
    _normalize_band(row, "pointwise")
    _normalize_band(row, "adjusted")
    row["source_pairing_complete"] = _boolean(
        row["source_pairing_complete"],
        "source_pairing_complete",
    )
    row["directional_language_allowed"] = _boolean(
        row["directional_language_allowed"],
        "directional_language_allowed",
    )
    row["status"] = _lower_name(row["status"], name="status")
    row["missing_reasons_a_json"] = _reason_counts_json(
        row["missing_reasons_a_json"],
        "missing_reasons_a_json",
    )
    row["missing_reasons_b_json"] = _reason_counts_json(
        row["missing_reasons_b_json"],
        "missing_reasons_b_json",
    )
    reasons_a = json.loads(row["missing_reasons_a_json"])
    reasons_b = json.loads(row["missing_reasons_b_json"])
    allowed_reasons = set(cell_spec.invalid_reason_codes)
    if not set(reasons_a).issubset(allowed_reasons) or not set(
        reasons_b
    ).issubset(allowed_reasons):
        raise ValueError(
            "scorecard missing reasons drifted from the metric registry"
        )
    if sum(reasons_a.values()) != row["cohort_n"] - row["valid_a_n"]:
        raise ValueError("policy A missing-reason counts are not exact")
    if sum(reasons_b.values()) != row["cohort_n"] - row["valid_b_n"]:
        raise ValueError("policy B missing-reason counts are not exact")
    expected_source_complete = (
        row["asymmetric_missing_n"] == 0
        and row["asymmetric_reason_n"] == 0
        and row["asymmetric_component_n"] == 0
    )
    if row["source_pairing_complete"] != expected_source_complete:
        raise ValueError(
            "source_pairing_complete contradicts asymmetric accounting"
        )

    row["resampling_key_json"] = _canonical_json_text(
        row["resampling_key_json"],
        "resampling_key_json",
        require_object=True,
    )
    if (
        not isinstance(row["resampling_sha256"], str)
        or _SHA256.fullmatch(row["resampling_sha256"]) is None
    ):
        raise ValueError("resampling_sha256 must be a SHA-256 digest")
    expected_digest = hashlib.sha256(
        row["resampling_key_json"].encode("utf-8")
    ).hexdigest()
    if row["resampling_sha256"] != expected_digest:
        raise ValueError("resampling_sha256 does not match resampling_key_json")
    words = row["resampling_digest_words"]
    if isinstance(words, (str, bytes)) or not isinstance(words, Sequence):
        raise TypeError("resampling_digest_words must be a sequence")
    normalized_words = [
        _integer(word, name="digest word", minimum=0, maximum=2**32 - 1)
        for word in words
    ]
    if len(normalized_words) != 8:
        raise ValueError("resampling_digest_words must contain exactly eight words")
    expected_words = [
        int.from_bytes(bytes.fromhex(expected_digest)[offset : offset + 4], "big")
        for offset in range(0, 32, 4)
    ]
    if normalized_words != expected_words:
        raise ValueError("resampling digest words do not match the key")
    row["resampling_digest_words"] = normalized_words
    row["rng"] = _text(row["rng"], "rng")
    row["index_dtype"] = _text(row["index_dtype"], "index_dtype")
    row["quantile_method"] = _text(row["quantile_method"], "quantile_method")

    expected_key = make_resampling_key(
        cell_spec,
        paired_n=row["paired_n"],
    )
    if (
        row["resampling_key_json"] != expected_key.canonical_json
        or row["resampling_sha256"] != expected_key.sha256
        or tuple(row["resampling_digest_words"]) != expected_key.digest_words
        or row["resamples"] != expected_key.resamples
        or row["base_seed"] != M5_BASE_SEED
        or row["rng"] != "PCG64"
        or row["index_dtype"] != "int64"
        or row["quantile_method"] != "linear"
    ):
        raise ValueError(
            "scorecard resampling metadata is not the frozen M5 substream"
        )

    summary_fields = (
        "raw_mean_difference",
        "raw_median_difference",
        "oriented_mean_advantage",
        "favorable_proportion",
        "standardized_signal_to_heterogeneity",
        "policy_a_mean",
        "policy_a_median",
        "policy_b_mean",
        "policy_b_median",
    )
    pointwise = (
        row["pointwise_level"],
        row["pointwise_lower"],
        row["pointwise_upper"],
    )
    adjusted = (
        row["adjusted_level"],
        row["adjusted_lower"],
        row["adjusted_upper"],
    )
    if row["paired_n"] < 10:
        if (
            row["nonzero_effect_n"] is not None
            or any(row[name] is not None for name in summary_fields)
            or any(value is not None for value in pointwise)
            or any(value is not None for value in adjusted)
        ):
            raise ValueError(
                "paired_n below ten requires complete effect suppression"
            )
        expected_status = (
            "insufficient_n"
            if row["source_pairing_complete"]
            else "pairing_incomplete"
        )
        expected_directional = False
    else:
        required_finite = (
            "raw_mean_difference",
            "raw_median_difference",
            "policy_a_mean",
            "policy_a_median",
            "policy_b_mean",
            "policy_b_median",
        )
        if (
            row["nonzero_effect_n"] is None
            or any(row[name] is None for name in required_finite)
            or pointwise[0] != M5_POINTWISE_STABILITY_LEVEL
        ):
            raise ValueError(
                "unsuppressed scorecards require exact effects and pointwise band"
            )
        if cell_spec.is_primary:
            if adjusted[0] != M5_PRIMARY_ADJUSTED_STABILITY_LEVEL:
                raise ValueError(
                    "primary scorecards require the adjusted stability level"
                )
            if (
                adjusted[1] > pointwise[1]
                or adjusted[2] < pointwise[2]
            ):
                raise ValueError(
                    "primary adjusted band must contain its pointwise band"
                )
            claim_band = adjusted
        else:
            if any(value is not None for value in adjusted):
                raise ValueError(
                    "exploratory scorecards cannot carry a primary adjusted band"
                )
            claim_band = pointwise
        if cell_spec.direction == "neutral":
            if any(
                row[name] is not None
                for name in (
                    "oriented_mean_advantage",
                    "favorable_proportion",
                    "standardized_signal_to_heterogeneity",
                )
            ):
                raise ValueError(
                    "neutral metrics cannot carry oriented claim fields"
                )
        elif (
            row["oriented_mean_advantage"] is None
            or row["favorable_proportion"] is None
        ):
            raise ValueError(
                "directional metrics require oriented effect fields"
            )
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
                    "standardized signal contradicts the oriented mean sign"
                )
        if cell_spec.direction != "neutral":
            ties = row["paired_n"] - row["nonzero_effect_n"]
            implied_wins = (
                row["paired_n"] * row["favorable_proportion"]
                - 0.5 * ties
            )
            nearest_wins = round(implied_wins)
            if (
                nearest_wins < 0
                or nearest_wins > row["nonzero_effect_n"]
                or not _finite_close(implied_wins, nearest_wins)
            ):
                raise ValueError(
                    "favorable proportion is outside the exact wins/ties lattice"
                )
        expected_raw_mean = (
            row["policy_a_mean"] - row["policy_b_mean"]
        )
        if not _finite_close(
            row["raw_mean_difference"],
            expected_raw_mean,
            scale_values=(
                row["policy_a_mean"],
                row["policy_b_mean"],
            ),
        ):
            raise ValueError(
                "raw mean difference contradicts the two policy means"
            )
        if cell_spec.direction != "neutral":
            expected_oriented = (
                row["raw_mean_difference"]
                if cell_spec.direction == "higher"
                else -row["raw_mean_difference"]
            )
            if row["oriented_mean_advantage"] != expected_oriented:
                raise ValueError(
                    "oriented mean contradicts metric direction and raw mean"
                )
        if row["nonzero_effect_n"] == 0:
            required_zero = (
                "raw_mean_difference",
                "raw_median_difference",
                "pointwise_lower",
                "pointwise_upper",
            )
            if cell_spec.direction != "neutral":
                required_zero += ("oriented_mean_advantage",)
                if row["favorable_proportion"] != 0.5:
                    raise ValueError(
                        "zero-effect directional cells require half-favorable ties"
                    )
            if cell_spec.is_primary:
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
        elif cell_spec.is_primary and row["nonzero_effect_n"] < 10:
            expected_status = "event_sparse"
        elif row["paired_n"] < 30 or row["nonzero_effect_n"] < 10:
            expected_status = "small_or_sparse"
        else:
            expected_status = "descriptive"
        band_excludes_zero = (
            claim_band[1] is not None
            and claim_band[2] is not None
            and (claim_band[2] < 0.0 or claim_band[1] > 0.0)
        )
        expected_directional = (
            cell_spec.direction != "neutral"
            and row["paired_n"] >= 30
            and row["nonzero_effect_n"] >= 10
            and row["source_pairing_complete"]
            and band_excludes_zero
        )
    if row["status"] != expected_status:
        raise ValueError("scorecard status contradicts frozen M5 rules")
    if row["directional_language_allowed"] != expected_directional:
        raise ValueError(
            "directional_language_allowed contradicts the claim gate"
        )
    return row


def _parity_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    row["parity_index"] = _integer(
        row["parity_index"],
        name="parity_index",
        minimum=0,
        maximum=2**31 - 1,
    )
    row["policy_name"] = _lower_name(row["policy_name"], name="policy_name")
    if row["policy_name"] not in _M5_POLICY_NAMES:
        raise ValueError("parity policy is not a canonical M5 policy")
    row["metric_name"] = _lower_name(row["metric_name"], name="metric_name")
    if row["metric_name"] not in _M5_PARITY_METRIC_NAMES:
        raise ValueError("parity metric is not a canonical M5 anchor")
    row["metric_version"] = _version(row["metric_version"], "metric_version")
    expected_version = _M5_PARITY_METRIC_VERSIONS[row["metric_name"]]
    if row["metric_version"] != expected_version:
        raise ValueError(
            "parity metric_version differs from the frozen anchor version"
        )
    row["compared_components"] = _integer(
        row["compared_components"],
        name="compared_components",
        minimum=0,
    )
    row["mismatch_count"] = _integer(
        row["mismatch_count"],
        name="mismatch_count",
        minimum=0,
    )
    if row["mismatch_count"] > row["compared_components"]:
        raise ValueError("mismatch_count cannot exceed compared_components")
    row["exact_match"] = _boolean(row["exact_match"], "exact_match")
    if row["exact_match"] != (row["mismatch_count"] == 0):
        raise ValueError("exact_match must agree with mismatch_count")
    if row["max_abs_error"] is not None:
        row["max_abs_error"] = _finite(row["max_abs_error"], "max_abs_error")
        if row["max_abs_error"] < 0.0:
            raise ValueError("max_abs_error must be non-negative")
    if row["max_tolerance_excess"] is not None:
        row["max_tolerance_excess"] = _finite(
            row["max_tolerance_excess"],
            "max_tolerance_excess",
        )
    if row["compared_components"] > 0 and (
        row["max_abs_error"] is None
        or row["max_tolerance_excess"] is None
    ):
        raise ValueError(
            "compared parity components require error and tolerance evidence"
        )
    if row["compared_components"] == 0 and (
        row["max_abs_error"] is not None
        or row["max_tolerance_excess"] is not None
    ):
        raise ValueError(
            "empty parity comparisons cannot carry error evidence"
        )
    row["status"] = _lower_name(row["status"], name="status")
    if row["status"] == "accepted" and (
        row["mismatch_count"] != 0 or not row["exact_match"]
    ):
        raise ValueError(
            "accepted parity rows cannot contain mismatches"
        )
    if (
        row["metric_name"] in {"kinematic_infeasibility", "overlap"}
        and row["compared_components"] > 0
        and (
            row["max_tolerance_excess"] != row["max_abs_error"]
            or row["max_abs_error"] not in {0.0, 1.0}
            or (row["mismatch_count"] > 0)
            != (row["max_abs_error"] == 1.0)
        )
    ):
        raise ValueError(
            "discrete parity tolerance evidence must equal exact absolute error"
        )
    if row["metric_name"] == "log_divergence":
        if (
            row["max_abs_error"] is not None
            and row["max_abs_error"] > _FLOAT32_MAX
        ):
            raise ValueError(
                "log-divergence parity error exceeds the finite float32 range"
            )
        if (
            row["max_tolerance_excess"] is not None
            and abs(row["max_tolerance_excess"]) > _FLOAT32_MAX
        ):
            raise ValueError(
                "log-divergence tolerance excess exceeds the float32 range"
            )
        if (
            row["max_abs_error"] is not None
            and row["max_tolerance_excess"] is not None
        ):
            maximum_possible_excess = row["max_abs_error"] - 1e-6
            if (
                row["max_tolerance_excess"] > maximum_possible_excess
                and not _finite_close(
                    row["max_tolerance_excess"],
                    maximum_possible_excess,
                )
            ):
                raise ValueError(
                    "log-divergence tolerance excess violates the absolute floor"
                )
        has_tolerance_mismatch = (
            row["max_tolerance_excess"] is not None
            and row["max_tolerance_excess"] > 0.0
        )
        if has_tolerance_mismatch != (row["mismatch_count"] > 0):
            raise ValueError(
                "log-divergence mismatch count contradicts tolerance excess"
            )
    return row


def _normalize_band(row: dict[str, Any], prefix: str) -> None:
    names = (f"{prefix}_level", f"{prefix}_lower", f"{prefix}_upper")
    values = tuple(row[name] for name in names)
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError(f"{prefix} stability band must be all-null or complete")
    level = _finite(row[names[0]], names[0])
    lower = _finite(row[names[1]], names[1])
    upper = _finite(row[names[2]], names[2])
    if not 0.0 < level < 1.0 or lower > upper:
        raise ValueError(f"{prefix} stability band is invalid")
    row[names[0]], row[names[1]], row[names[2]] = level, lower, upper


def _reason_counts_json(value: Any, name: str) -> str:
    canonical = _canonical_json_text(value, name, require_object=True)
    payload = json.loads(canonical)
    for reason, count in payload.items():
        _lower_name(reason, name=f"{name} reason")
        _integer(count, name=f"{name} count", minimum=1)
    return canonical


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


def _read_and_validate_table(path: Path, dataset: str) -> pa.Table:
    try:
        table = pq.read_table(path)
        table.validate(full=True)
    except (OSError, pa.ArrowException) as exc:
        raise M5ResultStoreIntegrityError(
            f"{dataset} Parquet part could not be read"
        ) from exc
    if not table.schema.equals(M5_RESULT_SCHEMAS[dataset], check_metadata=True):
        raise M5ResultStoreIntegrityError(
            f"{dataset} Parquet part does not use the fixed schema"
        )
    return table


def _iter_normalized_artifact_rows(
    path: Path,
    dataset: str,
    *,
    batch_size: int = 64,
) -> Iterable[dict[str, Any]]:
    """Incrementally validate and yield one canonical Parquet artifact."""

    try:
        parquet_file = pq.ParquetFile(path)
        if not parquet_file.schema_arrow.equals(
            M5_RESULT_SCHEMAS[dataset],
            check_metadata=True,
        ):
            raise M5ResultStoreIntegrityError(
                f"{dataset} Parquet part does not use the fixed schema"
            )
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batch.validate(full=True)
            for raw in batch.to_pylist():
                yield _normalize_row(dataset, raw)
    except M5ResultStoreIntegrityError:
        raise
    except (OSError, pa.ArrowException, TypeError, ValueError) as exc:
        raise M5ResultStoreIntegrityError(
            f"{dataset} Parquet part could not be incrementally validated"
        ) from exc


def _write_parquet_pending(path: Path, table: pa.Table) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"pending part {path.name!r} already exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            pq.write_table(
                table,
                handle,
                compression="NONE",
                use_dictionary=False,
                write_statistics=True,
                version="2.6",
                data_page_version="2.0",
            )
            handle.flush()
            _fsync_descriptor(handle.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    if type(payload) is not bytes:
        raise TypeError("exclusive payload must be exact bytes")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            _fsync_descriptor(handle.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _regular_file_has_exact_bytes(
    path: Path,
    expected: bytes,
) -> bool:
    """Read one control file without following links or trusting path metadata."""

    if type(expected) is not bytes:
        return False
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != len(expected)
        ):
            return False
        payload = bytearray()
        while len(payload) <= len(expected):
            chunk = os.read(
                descriptor,
                len(expected) + 1 - len(payload),
            )
            if not chunk:
                break
            payload.extend(chunk)
        return bytes(payload) == expected
    except OSError:
        return False
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _fsync_descriptor(descriptor: int) -> None:
    os.fsync(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        _fsync_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> tuple[str, int]:
    if _path_kind(path) != "file":
        raise M5ResultStoreIntegrityError("hash target is not a regular file")
    before = path.stat()
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise M5ResultStoreIntegrityError("artifact hashing failed") from exc
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise M5ResultStoreIntegrityError(
            "artifact changed while its SHA-256 was computed"
        )
    return digest.hexdigest(), after.st_size


def _read_regular_bytes(path: Path) -> bytes:
    if _path_kind(path) != "file":
        raise M5ResultStoreIntegrityError("byte target is not a regular file")
    before = path.stat()
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise M5ResultStoreIntegrityError("artifact byte read failed") from exc
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or len(payload) != after.st_size:
        raise M5ResultStoreIntegrityError(
            "artifact changed while its bytes were read"
        )
    return payload


def _verify_supplemental_artifact(
    run_path: Path,
    record: SupplementalArtifactRecord,
) -> M5ParityOrderReceipt | M5DeterminismReceipt:
    if not isinstance(record, SupplementalArtifactRecord):
        raise M5ResultStoreIntegrityError(
            "supplemental artifact record has the wrong type"
        )
    canonical = _contained_artifact_path(run_path, record.path)
    pending = _pending_path(run_path, Path(record.path))
    if _path_kind(canonical) != "file" or _path_kind(pending) != "file":
        raise M5ResultStoreIntegrityError(
            "supplemental artifact is missing or unsafe"
        )
    canonical_stat = canonical.stat()
    pending_stat = pending.stat()
    if (
        canonical_stat.st_dev,
        canonical_stat.st_ino,
    ) != (
        pending_stat.st_dev,
        pending_stat.st_ino,
    ):
        raise M5ResultStoreIntegrityError(
            "supplemental artifact is detached from its pending hard link"
        )
    if (
        stat.S_IMODE(canonical_stat.st_mode) != 0o600
        or stat.S_IMODE(pending_stat.st_mode) != 0o600
    ):
        raise M5ResultStoreIntegrityError(
            "supplemental artifact permissions are not owner-only"
        )
    encoded = _read_regular_bytes(canonical)
    if (
        len(encoded) != record.size_bytes
        or hashlib.sha256(encoded).hexdigest() != record.sha256
    ):
        raise M5ResultStoreIntegrityError(
            "supplemental artifact failed SHA-256 verification"
        )
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M5ResultStoreIntegrityError(
            "supplemental artifact is not canonical JSON"
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or _canonical_json_bytes(payload) != encoded
    ):
        raise M5ResultStoreIntegrityError(
            "supplemental artifact is not canonical JSON"
        )
    if payload.get("schema_version") != record.schema_version:
        raise M5ResultStoreIntegrityError(
            "supplemental payload schema differs from its manifest record"
        )
    if record.kind == PARITY_ORDER_RECEIPT:
        receipt_payload = dict(payload)
        del receipt_payload["schema_version"]
        return M5ParityOrderReceipt.from_dict(receipt_payload)
    if record.kind != DETERMINISM_RECEIPT:
        raise M5ResultStoreIntegrityError(
            "supplemental artifact kind is unsupported"
        )
    return M5DeterminismReceipt.from_dict(payload)


def _verify_scorecard_report(
    run_path: Path,
    record: ScorecardReportRecord,
    row_accounting_profile: str,
    records: Sequence[ArtifactRecord],
    *,
    scorecard_rows: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    if not isinstance(record, ScorecardReportRecord):
        raise M5ResultStoreIntegrityError(
            "scorecard report record has the wrong type"
        )
    canonical = _contained_artifact_path(run_path, record.path)
    pending = _pending_path(run_path, Path(record.path))
    if _path_kind(canonical) != "file" or _path_kind(pending) != "file":
        raise M5ResultStoreIntegrityError(
            "scorecard report is missing or unsafe"
        )
    canonical_stat = canonical.stat()
    pending_stat = pending.stat()
    if (
        canonical_stat.st_dev,
        canonical_stat.st_ino,
    ) != (
        pending_stat.st_dev,
        pending_stat.st_ino,
    ):
        raise M5ResultStoreIntegrityError(
            "scorecard report is detached from its pending hard link"
        )
    if (
        stat.S_IMODE(canonical_stat.st_mode) != 0o600
        or stat.S_IMODE(pending_stat.st_mode) != 0o600
    ):
        raise M5ResultStoreIntegrityError(
            "scorecard report permissions are not owner-only"
        )
    actual = _read_regular_bytes(canonical)
    if (
        len(actual) != record.size_bytes
        or hashlib.sha256(actual).hexdigest() != record.sha256
    ):
        raise M5ResultStoreIntegrityError(
            "scorecard report failed SHA-256 verification"
        )

    scorecard_records = tuple(
        artifact for artifact in records if artifact.dataset == SCORECARDS
    )
    if (
        len(scorecard_records) != 1
        or scorecard_records[0].path != "scorecards.parquet"
    ):
        raise M5ResultStoreIntegrityError(
            "scorecard report has no canonical source table"
        )
    source_record = scorecard_records[0]
    source_path = _contained_artifact_path(run_path, source_record.path)
    source_digest, source_size = _file_sha256(source_path)
    if (
        source_digest != source_record.sha256
        or source_size != source_record.size_bytes
    ):
        raise M5ResultStoreIntegrityError(
            "scorecard report source table failed SHA-256 verification"
        )
    try:
        if scorecard_rows is None:
            table = _read_and_validate_table(source_path, SCORECARDS)
            if table.num_rows != source_record.rows:
                raise M5ResultStoreIntegrityError(
                    "scorecard report source row count changed"
                )
            normalized, _ = _normalize_rows(
                SCORECARDS,
                table.to_pylist(),
            )
        else:
            normalized = [
                _normalize_row(SCORECARDS, row)
                for row in scorecard_rows
            ]
            if len(normalized) != source_record.rows:
                raise M5ResultStoreIntegrityError(
                    "scorecard report source row count changed"
                )
        expected = render_m5_scorecard(
            normalized,
            row_accounting_profile=row_accounting_profile,
        )
    except (TypeError, ValueError) as exc:
        raise M5ResultStoreIntegrityError(
            "scorecard report could not be deterministically re-rendered"
        ) from exc
    if actual != expected:
        raise M5ResultStoreIntegrityError(
            "scorecard report differs from its deterministic rendering"
        )


def _schema_fingerprint(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _validated_project_root(value: str | Path) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError("project_root must be a path")
    absolute = Path(os.path.abspath(os.fspath(value)))
    if (
        _path_kind(absolute) != "directory"
        or absolute.resolve(strict=True) != absolute
    ):
        raise M5ResultStoreError(
            "project_root must be an existing real directory without symlinks"
        )
    return absolute


def _validated_run_name(value: Any) -> str:
    if not isinstance(value, str) or _RUN_NAME.fullmatch(value) is None:
        raise ValueError(
            "run_name must be a simple lowercase filename without traversal"
        )
    if value in {".", ".."}:
        raise ValueError("run_name cannot be a traversal component")
    return value


def _ensure_directory(parent: Path, name: str) -> Path:
    path = parent / name
    try:
        kind = _path_kind(path)
        if kind == "missing":
            os.mkdir(path, mode=0o700)
            _fsync_directory(parent)
            kind = _path_kind(path)
    except OSError as exc:
        raise M5ResultStoreError(
            f"could not create required output directory {name!r}"
        ) from exc
    if (
        kind != "directory"
        or path.resolve(strict=True) != path
        or path.parent != parent
    ):
        raise M5ResultStoreError(
            f"required output directory {name!r} is unsafe"
        )
    return path


def _create_child_directory(parent: Path, name: str) -> Path:
    path = parent / name
    try:
        os.mkdir(path, mode=0o700)
        _fsync_directory(parent)
    except OSError as exc:
        raise M5ResultStoreError(
            f"could not create internal result directory {name!r}"
        ) from exc
    if _path_kind(path) != "directory" or path.resolve(strict=True) != path:
        raise M5ResultStoreError(
            f"internal result directory {name!r} is unsafe"
        )
    return path


def _rollback_fresh_result_store(run_path: Path) -> None:
    """Best-effort rollback of only the known, newly-created empty layout.

    If concurrent or unexpected members make exact ``rmdir`` rollback
    impossible, retain the directory and publish a canonical failure marker so
    it cannot be mistaken for a resumable or successful run.
    """

    candidates = (
        run_path / "pending" / METRIC_RESULTS,
        run_path / "pending",
        run_path / METRIC_RESULTS,
        run_path,
    )
    for candidate in candidates:
        try:
            if _path_kind(candidate) == "directory":
                os.rmdir(candidate)
        except OSError:
            pass
    if _path_kind(run_path) != "directory":
        return
    failure_path = run_path / "FAILURE.json"
    if os.path.lexists(failure_path):
        return
    try:
        _write_bytes_exclusive(
            failure_path,
            _canonical_json_bytes(
                {
                    "complete": False,
                    "reason_code": "result_store_failed",
                    "schema_version": M5_RESULT_STORE_SCHEMA_VERSION,
                }
            ),
        )
    except (OSError, M5ResultStoreError):
        # Exclusive creation can report a post-create fsync error.  Preserve an
        # exact marker if it nevertheless reached disk.
        if not _regular_file_has_exact_bytes(
            failure_path,
            _canonical_json_bytes(
                {
                    "complete": False,
                    "reason_code": "result_store_failed",
                    "schema_version": M5_RESULT_STORE_SCHEMA_VERSION,
                }
            ),
        ):
            return


def _path_kind(path: Path) -> str:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return "missing"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    return "other"


def _safe_relative_artifact_path(value: str | Path) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError("artifact path must be path-like")
    raw = os.fspath(value)
    path = Path(raw)
    if (
        path.is_absolute()
        or not path.parts
        or "\\" in raw
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("artifact path must be a contained relative path")
    return path


def _safe_project_relative_source_path(value: Any) -> Path:
    if not isinstance(value, str):
        raise TypeError("executable source paths must be strings")
    path = Path(value)
    if (
        path.is_absolute()
        or not path.parts
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(
            "executable source paths must be canonical project-relative paths"
        )
    return path


def _contained_artifact_path(run_path: Path, value: str | Path) -> Path:
    relative = _safe_relative_artifact_path(value)
    path = run_path.joinpath(*relative.parts)
    if path == run_path or run_path not in path.parents:
        raise M5ResultStoreIntegrityError("artifact path escapes the run")
    if path.parent.resolve(strict=True) not in {
        run_path,
        run_path / METRIC_RESULTS,
    }:
        raise M5ResultStoreIntegrityError("artifact parent is not canonical")
    return path


def _pending_path(run_path: Path, relative: Path) -> Path:
    relative = _safe_relative_artifact_path(relative)
    path = run_path / "pending" / relative
    expected_parents = {
        run_path / "pending",
        run_path / "pending" / METRIC_RESULTS,
    }
    if path.parent not in expected_parents:
        raise M5ResultStoreIntegrityError("pending artifact parent is not canonical")
    return path


def _validate_run_members(
    run_path: Path,
    records: Sequence[ArtifactRecord],
    scorecard_report: ScorecardReportRecord,
    *,
    allowed_controls: set[str],
    supplemental_records: Sequence[SupplementalArtifactRecord] = (),
) -> None:
    canonical = {record.path for record in records}
    pending = {f"pending/{record.path}" for record in records}
    canonical.add(scorecard_report.path)
    pending.add(f"pending/{scorecard_report.path}")
    canonical.update(record.path for record in supplemental_records)
    pending.update(
        f"pending/{record.path}" for record in supplemental_records
    )
    allowed_files = canonical | pending | allowed_controls
    allowed_directories = {
        ".",
        "pending",
        METRIC_RESULTS,
        f"pending/{METRIC_RESULTS}",
    }
    observed_files: set[str] = set()
    observed_directories: set[str] = {"."}
    for current_text, directory_names, file_names in os.walk(
        run_path,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_text)
        relative_current = current.relative_to(run_path).as_posix()
        if relative_current == ".":
            relative_current = "."
        if _path_kind(current) != "directory":
            raise M5ResultStoreIntegrityError("run contains an unsafe directory")
        for directory_name in directory_names:
            child = current / directory_name
            relative = child.relative_to(run_path).as_posix()
            if _path_kind(child) != "directory":
                raise M5ResultStoreIntegrityError(
                    "run contains a symlink or special directory entry"
                )
            observed_directories.add(relative)
        for file_name in file_names:
            child = current / file_name
            relative = child.relative_to(run_path).as_posix()
            if _path_kind(child) != "file":
                raise M5ResultStoreIntegrityError(
                    "run contains a symlink or special file"
                )
            observed_files.add(relative)
    if observed_directories != allowed_directories:
        raise M5ResultStoreIntegrityError(
            "run contains an unregistered output directory"
        )
    if observed_files != allowed_files:
        raise M5ResultStoreIntegrityError(
            "run contains an unregistered or missing output artifact"
        )


def _validate_artifact_layout(
    records: Sequence[ArtifactRecord],
) -> None:
    by_dataset = {
        name: tuple(record for record in records if record.dataset == name)
        for name in _DATASET_NAMES
    }
    if any(not dataset_records for dataset_records in by_dataset.values()):
        raise M5ResultStoreIntegrityError(
            "every fixed result dataset requires at least one Parquet part"
        )
    singleton_paths = {
        SLICE_MEMBERSHIP: "slice-membership.parquet",
        SCORECARDS: "scorecards.parquet",
        WAYMAX_PARITY_SUMMARY: "waymax-parity-summary.parquet",
    }
    for dataset, path in singleton_paths.items():
        dataset_records = by_dataset[dataset]
        if len(dataset_records) != 1 or dataset_records[0].path != path:
            raise M5ResultStoreIntegrityError(
                f"{dataset} must have exactly its canonical single part"
            )
    indices: list[int] = []
    for record in by_dataset[METRIC_RESULTS]:
        path = Path(record.path)
        match = _PART_NAME.fullmatch(path.name)
        if path.parent.as_posix() != METRIC_RESULTS or match is None:
            raise M5ResultStoreIntegrityError(
                "metric result parts have noncanonical paths"
            )
        indices.append(int(match.group(1)))
    if sorted(indices) != list(range(len(indices))):
        raise M5ResultStoreIntegrityError(
            "metric result part indices must be contiguous from zero"
        )


def _validate_supplemental_layout(
    records: Sequence[SupplementalArtifactRecord],
) -> None:
    expected_kinds = tuple(sorted(_SUPPLEMENTAL_PATHS))
    observed_kinds = tuple(record.kind for record in records)
    if observed_kinds != expected_kinds:
        raise M5ResultStoreIntegrityError(
            "official supplemental artifacts are incomplete or unordered"
        )
    if len({record.path for record in records}) != len(records):
        raise M5ResultStoreIntegrityError(
            "official supplemental artifact paths are not unique"
        )


def _verify_artifact_records(
    run_path: Path,
    records: Sequence[ArtifactRecord],
    *,
    streaming: bool,
) -> dict[str, int]:
    """Verify immutable links, hashes, schemas, keys, order, and row counts."""

    _verify_artifact_bytes(run_path, records)
    seen: dict[str, set[tuple[Any, ...]]] = {
        name: set() for name in _DATASET_NAMES
    }
    counts = {name: 0 for name in _DATASET_NAMES}
    for record in records:
        canonical = _contained_artifact_path(run_path, record.path)
        if streaming:
            row_count = 0
            previous_key: tuple[Any, ...] | None = None
            for row in _iter_normalized_artifact_rows(
                canonical,
                record.dataset,
            ):
                key = tuple(
                    row[field] for field in _KEY_FIELDS[record.dataset]
                )
                if previous_key is not None and key <= previous_key:
                    raise M5ResultStoreIntegrityError(
                        f"artifact {record.path} is not in canonical key order"
                    )
                if key in seen[record.dataset]:
                    raise M5ResultStoreIntegrityError(
                        f"{record.dataset} has duplicate keys across parts"
                    )
                seen[record.dataset].add(key)
                previous_key = key
                row_count += 1
        else:
            table = _read_and_validate_table(canonical, record.dataset)
            normalized, keys = _normalize_rows(
                record.dataset,
                table.to_pylist(),
            )
            ordered = [
                tuple(row[field] for field in _KEY_FIELDS[record.dataset])
                for row in normalized
            ]
            if ordered != sorted(ordered):
                raise M5ResultStoreIntegrityError(
                    f"artifact {record.path} is not in canonical key order"
                )
            if seen[record.dataset].intersection(keys):
                raise M5ResultStoreIntegrityError(
                    f"{record.dataset} has duplicate keys across parts"
                )
            seen[record.dataset].update(keys)
            row_count = table.num_rows
        if row_count != record.rows:
            raise M5ResultStoreIntegrityError(
                f"artifact {record.path} failed row accounting"
            )
        counts[record.dataset] += row_count
    return counts


def _verify_artifact_bytes(
    run_path: Path,
    records: Sequence[ArtifactRecord],
) -> None:
    """Recheck immutable identities and hashes without materializing row data."""

    for record in records:
        canonical = _contained_artifact_path(run_path, record.path)
        pending = _pending_path(
            run_path,
            _safe_relative_artifact_path(record.path),
        )
        if _path_kind(canonical) != "file" or _path_kind(pending) != "file":
            raise M5ResultStoreIntegrityError(
                f"artifact {record.path} is not a regular file"
            )
        canonical_stat = canonical.stat()
        pending_stat = pending.stat()
        if (
            canonical_stat.st_dev,
            canonical_stat.st_ino,
        ) != (
            pending_stat.st_dev,
            pending_stat.st_ino,
        ):
            raise M5ResultStoreIntegrityError(
                f"artifact {record.path} is no longer its pending hard link"
            )
        digest, size = _file_sha256(canonical)
        if digest != record.sha256 or size != record.size_bytes:
            raise M5ResultStoreIntegrityError(
                f"artifact {record.path} failed SHA-256 verification"
            )


def _read_all_artifact_rows(
    run_path: Path,
    records: Sequence[ArtifactRecord],
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {
        dataset: [] for dataset in _DATASET_NAMES
    }
    for record in records:
        table = _read_and_validate_table(
            _contained_artifact_path(run_path, record.path),
            record.dataset,
        )
        normalized, _ = _normalize_rows(record.dataset, table.to_pylist())
        rows[record.dataset].extend(normalized)
    return rows


def _canonical_evaluation_digest(domain: str, payload: Any) -> str:
    encoded = json.dumps(
        _thaw_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(domain.encode("ascii") + b"\0" + encoded).hexdigest()


def _iter_dataset_rows(
    run_path: Path,
    records: Sequence[ArtifactRecord],
    dataset: str,
) -> Iterable[dict[str, Any]]:
    dataset_records = tuple(
        record for record in records if record.dataset == dataset
    )
    for record in dataset_records:
        yield from _iter_normalized_artifact_rows(
            _contained_artifact_path(run_path, record.path),
            dataset,
        )


def _reference_equivalence_digest(row: Mapping[str, Any]) -> str:
    payload = {
        name: row[name]
        for name in M5_RESULT_SCHEMAS[METRIC_RESULTS].names
        if name not in {"execution_name", "execution_role"}
    }
    return _canonical_evaluation_digest(
        "evalsim-m5-reference-log-replay-equivalence-v1",
        payload,
    )


def _validate_exact_log_zero_oracle(
    row: Mapping[str, Any],
) -> int:
    if row["metric_name"] not in _M5_ERROR_ORACLE_METRIC_NAMES:
        raise ValueError("metric row is not an exact-log zero oracle")
    distribution = row["distribution"]
    if (
        row["valid"] is not True
        or row["value"] != 0.0
        or not distribution
        or any(component != 0.0 for component in distribution)
    ):
        raise M5ResultStoreIntegrityError(
            "Waymax exact-log reference failed a zero-error oracle"
        )
    return len(distribution)


def _scenario_scalar_from_metric_row(
    row: Mapping[str, Any],
) -> ScenarioScalar:
    cohort_index = int(row["cohort_index"])
    if row["valid"]:
        return ScenarioScalar(
            cohort_index=cohort_index,
            value=float(row["value"]),
            eligible_components=int(row["eligible_components"]),
            total_components=int(row["total_components"]),
        )
    return ScenarioScalar.missing(
        cohort_index,
        str(row["invalid_reason"]),
        total_components=int(row["total_components"]),
    )


def _scan_official_artifacts(
    run_path: Path,
    records: Sequence[ArtifactRecord],
    determinism_receipt: M5DeterminismReceipt,
) -> _OfficialScanSummary:
    """Stream the official matrix while retaining only compact review facts."""

    if not isinstance(determinism_receipt, M5DeterminismReceipt):
        raise M5ResultStoreIntegrityError(
            "official M5 determinism receipt is missing"
        )
    cohort_indices = range(128)
    metric_names = tuple(M5_METRIC_SPECS)
    execution_names = tuple(sorted(_M5_EXECUTION_ROLES))
    expected_metric_domain = {
        (cohort_index, execution_name, metric_name)
        for cohort_index in cohort_indices
        for execution_name in execution_names
        for metric_name in metric_names
    }
    observed_metric_domain: set[tuple[int, str, str]] = set()
    policy_scalars: dict[
        tuple[int, str, str],
        ScenarioScalar,
    ] = {}
    reference_digests: dict[tuple[int, str], str] = {}
    log_replay_digests: dict[tuple[int, str], str] = {}
    case_digests: dict[int, str] = {}
    case_rows: list[Mapping[str, Any]] = []
    active_case: int | None = None
    previous_metric_key: tuple[Any, ...] | None = None

    def finish_case() -> None:
        nonlocal case_rows
        if active_case is None:
            return
        if len(case_rows) != len(_M5_EXECUTION_ROLES) * len(
            M5_METRIC_SPECS
        ):
            raise M5ResultStoreIntegrityError(
                "official case metric matrix is incomplete"
            )
        case_digests[active_case] = _canonical_evaluation_digest(
            "evalsim-m5-case-metric-pass-v1",
            tuple(case_rows),
        )
        case_rows = []

    for row in _iter_dataset_rows(run_path, records, METRIC_RESULTS):
        key = tuple(row[field] for field in _KEY_FIELDS[METRIC_RESULTS])
        if previous_metric_key is not None and key <= previous_metric_key:
            raise M5ResultStoreIntegrityError(
                "official metric rows are not globally canonical"
            )
        previous_metric_key = key
        cohort_index = int(row["cohort_index"])
        execution_name = str(row["execution_name"])
        metric_name = str(row["metric_name"])
        domain_key = (cohort_index, execution_name, metric_name)
        observed_metric_domain.add(domain_key)
        if row["seed"] != 0:
            raise M5ResultStoreIntegrityError(
                "each official execution must use the frozen seed zero"
            )
        if (
            metric_name in M5_PRIMARY_METRIC_NAMES
            and row["valid"] is not True
        ):
            raise M5ResultStoreIntegrityError(
                "every official primary metric must be valid for all four executions"
            )
        if execution_name in _M5_POLICY_NAMES:
            policy_scalars[domain_key] = _scenario_scalar_from_metric_row(row)
        equivalence_key = (cohort_index, metric_name)
        if execution_name == "log_replay":
            log_replay_digests[equivalence_key] = (
                _reference_equivalence_digest(row)
            )
        elif execution_name == WAYMAX_EXACT_LOG_NAME:
            reference_digests[equivalence_key] = (
                _reference_equivalence_digest(row)
            )
            if metric_name in _M5_ERROR_ORACLE_METRIC_NAMES:
                _validate_exact_log_zero_oracle(row)

        if active_case is None:
            active_case = cohort_index
        elif cohort_index != active_case:
            finish_case()
            if cohort_index != active_case + 1:
                raise M5ResultStoreIntegrityError(
                    "official metric cohort order is not contiguous"
                )
            active_case = cohort_index
        case_rows.append(row)
    finish_case()

    if (
        observed_metric_domain != expected_metric_domain
        or len(observed_metric_domain)
        != OFFICIAL_M5_ROW_COUNTS.metric_results
        or tuple(sorted(case_digests)) != tuple(cohort_indices)
    ):
        raise M5ResultStoreIntegrityError(
            "metric results do not cover the frozen 128×4×13 domain"
        )
    if set(policy_scalars) != {
        (cohort_index, policy_name, metric_name)
        for cohort_index in cohort_indices
        for policy_name in _M5_POLICY_NAMES
        for metric_name in metric_names
    }:
        raise M5ResultStoreIntegrityError(
            "official policy scalar matrix is incomplete"
        )
    if set(reference_digests) != set(log_replay_digests):
        raise M5ResultStoreIntegrityError(
            "Waymax reference/log-replay metric identity differs"
        )
    reference_mismatch_rows = sum(
        reference_digests[key] != log_replay_digests[key]
        for key in reference_digests
    )
    if reference_mismatch_rows:
        raise M5ResultStoreIntegrityError(
            "Waymax exact-log results differ from EvalSim log replay"
        )
    metric_pass_sha256 = _canonical_evaluation_digest(
        "evalsim-m5-cohort-metric-pass-v1",
        tuple(
            {
                "cohort_index": cohort_index,
                "sha256": case_digests[cohort_index],
            }
            for cohort_index in cohort_indices
        ),
    )
    if (
        determinism_receipt.metric_pass_1_sha256
        != metric_pass_sha256
        or determinism_receipt.metric_pass_2_sha256
        != metric_pass_sha256
    ):
        raise M5ResultStoreIntegrityError(
            "stored metric rows differ from the determinism receipt"
        )

    slice_names = tuple(_M5_SLICE_SPECS)
    expected_slice_domain = {
        (cohort_index, slice_name)
        for cohort_index in cohort_indices
        for slice_name in slice_names
    }
    observed_slice_domain: set[tuple[int, str]] = set()
    slice_members: dict[str, list[int]] = {
        name: [] for name in slice_names
    }
    for row in _iter_dataset_rows(run_path, records, SLICE_MEMBERSHIP):
        key = (int(row["cohort_index"]), str(row["slice_name"]))
        observed_slice_domain.add(key)
        if row["eligible"] and row["member"]:
            slice_members[key[1]].append(key[0])
        if (
            key[1] == "all"
            and (not row["eligible"] or not row["member"])
        ):
            raise M5ResultStoreIntegrityError(
                "the official all slice must contain every accepted scenario"
            )
    if observed_slice_domain != expected_slice_domain:
        raise M5ResultStoreIntegrityError(
            "slice membership does not cover the frozen 128×8 domain"
        )
    for members in slice_members.values():
        members.sort()

    expected_scorecard_domain = {
        (metric_name, slice_name, policy_a, policy_b)
        for metric_name in metric_names
        for slice_name in slice_names
        for policy_a, policy_b in _M5_POLICY_CONTRASTS
    }
    observed_scorecard_domain: set[tuple[str, str, str, str]] = set()
    scorecard_rows: list[Mapping[str, Any]] = []
    derived_results: dict[
        tuple[str, str, str, str],
        PairedCellResult,
    ] = {}
    for row in _iter_dataset_rows(run_path, records, SCORECARDS):
        domain_key = (
            str(row["metric_name"]),
            str(row["slice_name"]),
            str(row["policy_a"]),
            str(row["policy_b"]),
        )
        observed_scorecard_domain.add(domain_key)
        members = slice_members[domain_key[1]]
        values_a = tuple(
            policy_scalars[
                (cohort_index, domain_key[2], domain_key[0])
            ]
            for cohort_index in members
        )
        values_b = tuple(
            policy_scalars[
                (cohort_index, domain_key[3], domain_key[0])
            ]
            for cohort_index in members
        )
        try:
            result = analyze_paired_cell(
                PairedCellSpec(
                    metric_name=domain_key[0],
                    metric_version=str(row["metric_version"]),
                    slice_name=domain_key[1],
                    slice_version=str(row["slice_version"]),
                    contrast=PolicyContrast(
                        domain_key[2],
                        domain_key[3],
                    ),
                ),
                values_a,
                values_b,
            )
            expected = scorecard_row_from_result(result)
        except (KeyError, TypeError, ValueError) as exc:
            raise M5ResultStoreIntegrityError(
                "scorecard could not be derived from streamed source rows"
            ) from exc
        if dict(row) != expected:
            raise M5ResultStoreIntegrityError(
                "scorecard differs from its exact stored metric/slice derivation"
            )
        if (
            row["cohort_n"] != len(members)
            or row["source_pairing_complete"] is not True
        ):
            raise M5ResultStoreIntegrityError(
                "official scorecard pairing contradicts source-only membership"
            )
        if (
            domain_key[0] in M5_PRIMARY_METRIC_NAMES
            and domain_key[1] == "all"
            and not (
                row["cohort_n"]
                == row["valid_a_n"]
                == row["valid_b_n"]
                == row["paired_n"]
                == 128
            )
        ):
            raise M5ResultStoreIntegrityError(
                "official primary all-slice cells require 128 valid paired scenarios"
            )
        derived_results[domain_key] = result
        scorecard_rows.append(row)
    if observed_scorecard_domain != expected_scorecard_domain:
        raise M5ResultStoreIntegrityError(
            "scorecards do not cover the frozen 13×8×3 domain"
        )
    registry_results = tuple(
        derived_results[
            (
                metric_spec.name,
                slice_spec.name,
                policy_a,
                policy_b,
            )
        ]
        for metric_spec in M5_METRIC_SPECS.values()
        for slice_spec in M5_SLICE_SPECS
        for policy_a, policy_b in _M5_POLICY_CONTRASTS
    )
    statistics_pass_sha256 = _canonical_evaluation_digest(
        "evalsim-m5-statistics-pass-v1",
        tuple(result.to_dict() for result in registry_results),
    )
    if (
        determinism_receipt.statistics_pass_1_sha256
        != statistics_pass_sha256
        or determinism_receipt.statistics_pass_2_sha256
        != statistics_pass_sha256
    ):
        raise M5ResultStoreIntegrityError(
            "stored scorecards differ from the determinism receipt"
        )

    expected_parity_domain = {
        (parity_index, policy_name, metric_name)
        for parity_index in range(M5_PARITY_SCENE_COUNT)
        for policy_name in _M5_POLICY_NAMES
        for metric_name in _M5_PARITY_METRIC_NAMES
    }
    observed_parity_domain: set[tuple[int, str, str]] = set()
    for row in _iter_dataset_rows(
        run_path,
        records,
        WAYMAX_PARITY_SUMMARY,
    ):
        observed_parity_domain.add(
            (
                int(row["parity_index"]),
                str(row["policy_name"]),
                str(row["metric_name"]),
            )
        )
        if (
            row["compared_components"] < 1
            or row["status"] != "accepted"
            or row["mismatch_count"] != 0
            or row["exact_match"] is not True
        ):
            raise M5ResultStoreIntegrityError(
                "Waymax parity contains a rejected or mismatched row"
            )
    if observed_parity_domain != expected_parity_domain:
        raise M5ResultStoreIntegrityError(
            "Waymax parity does not cover the frozen 16×3×3 domain"
        )

    return _OfficialScanSummary(
        metric_pass_sha256=metric_pass_sha256,
        statistics_pass_sha256=statistics_pass_sha256,
        scorecard_rows=tuple(scorecard_rows),
    )


def _validate_scorecards_derived_from_rows(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    try:
        metric_rows = rows[METRIC_RESULTS]
        slice_rows = rows[SLICE_MEMBERSHIP]
        scorecard_rows = rows[SCORECARDS]
    except KeyError as exc:
        raise M5ResultStoreIntegrityError(
            "scorecard derivation inputs are incomplete"
        ) from exc

    metric_lookup: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in metric_rows:
        key = (
            row["cohort_index"],
            row["execution_name"],
            row["metric_name"],
        )
        if key in metric_lookup:
            raise M5ResultStoreIntegrityError(
                "scorecard derivation found duplicate metric inputs"
            )
        metric_lookup[key] = row
    slice_members: dict[str, list[int]] = {
        name: [] for name in _M5_SLICE_SPECS
    }
    for row in slice_rows:
        if row["eligible"] and row["member"]:
            slice_members[row["slice_name"]].append(row["cohort_index"])
    for members in slice_members.values():
        members.sort()

    def scalar(
        cohort_index: int,
        execution_name: str,
        metric_name: str,
    ) -> ScenarioScalar:
        try:
            metric_row = metric_lookup[
                (cohort_index, execution_name, metric_name)
            ]
        except KeyError as exc:
            raise M5ResultStoreIntegrityError(
                "scorecard derivation is missing a policy metric row"
            ) from exc
        if metric_row["valid"]:
            return ScenarioScalar(
                cohort_index=cohort_index,
                value=metric_row["value"],
                eligible_components=metric_row["eligible_components"],
                total_components=metric_row["total_components"],
            )
        return ScenarioScalar.missing(
            cohort_index,
            metric_row["invalid_reason"],
            total_components=metric_row["total_components"],
        )

    for observed in scorecard_rows:
        try:
            spec = PairedCellSpec(
                metric_name=observed["metric_name"],
                metric_version=observed["metric_version"],
                slice_name=observed["slice_name"],
                slice_version=observed["slice_version"],
                contrast=PolicyContrast(
                    observed["policy_a"],
                    observed["policy_b"],
                ),
            )
            members = slice_members[spec.slice_name]
            result = analyze_paired_cell(
                spec,
                (
                    scalar(
                        cohort_index,
                        spec.contrast.policy_a,
                        spec.metric_name,
                    )
                    for cohort_index in members
                ),
                (
                    scalar(
                        cohort_index,
                        spec.contrast.policy_b,
                        spec.metric_name,
                    )
                    for cohort_index in members
                ),
            )
            expected = scorecard_row_from_result(result)
        except (KeyError, TypeError, ValueError) as exc:
            raise M5ResultStoreIntegrityError(
                "scorecard could not be derived from stored source rows"
            ) from exc
        if dict(observed) != expected:
            raise M5ResultStoreIntegrityError(
                "scorecard differs from its exact stored metric/slice derivation"
            )


def _validate_official_key_domains(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    if set(rows) != set(_DATASET_NAMES):
        raise M5ResultStoreIntegrityError(
            "official M5 domain accounting is incomplete"
        )
    cohort_indices = range(128)
    metric_names = tuple(M5_METRIC_SPECS)
    execution_names = tuple(sorted(_M5_EXECUTION_ROLES))
    expected_metric_domain = {
        (cohort_index, execution_name, metric_name)
        for cohort_index in cohort_indices
        for execution_name in execution_names
        for metric_name in metric_names
    }
    metric_rows = rows[METRIC_RESULTS]
    observed_metric_domain = {
        (
            row["cohort_index"],
            row["execution_name"],
            row["metric_name"],
        )
        for row in metric_rows
    }
    if (
        len(metric_rows) != OFFICIAL_M5_ROW_COUNTS.metric_results
        or observed_metric_domain != expected_metric_domain
    ):
        raise M5ResultStoreIntegrityError(
            "metric results do not cover the frozen 128×4×13 domain"
        )
    for execution_name in execution_names:
        seeds = {
            row["seed"]
            for row in metric_rows
            if row["execution_name"] == execution_name
        }
        if seeds != {0}:
            raise M5ResultStoreIntegrityError(
                "each official execution must use the frozen seed zero"
            )
    if any(
        row["valid"] is not True
        for row in metric_rows
        if row["metric_name"] in M5_PRIMARY_METRIC_NAMES
    ):
        raise M5ResultStoreIntegrityError(
            "every official primary metric must be valid for all four executions"
        )

    slice_names = tuple(_M5_SLICE_SPECS)
    expected_slice_domain = {
        (cohort_index, slice_name)
        for cohort_index in cohort_indices
        for slice_name in slice_names
    }
    slice_rows = rows[SLICE_MEMBERSHIP]
    observed_slice_domain = {
        (row["cohort_index"], row["slice_name"])
        for row in slice_rows
    }
    if (
        len(slice_rows) != OFFICIAL_M5_ROW_COUNTS.slice_membership
        or observed_slice_domain != expected_slice_domain
    ):
        raise M5ResultStoreIntegrityError(
            "slice membership does not cover the frozen 128×8 domain"
        )
    if any(
        not row["eligible"] or not row["member"]
        for row in slice_rows
        if row["slice_name"] == "all"
    ):
        raise M5ResultStoreIntegrityError(
            "the official all slice must contain every accepted scenario"
        )
    slice_cohort_sizes = {
        slice_name: sum(
            bool(row["eligible"] and row["member"])
            for row in slice_rows
            if row["slice_name"] == slice_name
        )
        for slice_name in slice_names
    }

    expected_scorecard_domain = {
        (metric_name, slice_name, policy_a, policy_b)
        for metric_name in metric_names
        for slice_name in slice_names
        for policy_a, policy_b in _M5_POLICY_CONTRASTS
    }
    scorecard_rows = rows[SCORECARDS]
    observed_scorecard_domain = {
        (
            row["metric_name"],
            row["slice_name"],
            row["policy_a"],
            row["policy_b"],
        )
        for row in scorecard_rows
    }
    if (
        len(scorecard_rows) != OFFICIAL_M5_ROW_COUNTS.scorecards
        or observed_scorecard_domain != expected_scorecard_domain
    ):
        raise M5ResultStoreIntegrityError(
            "scorecards do not cover the frozen 13×8×3 domain"
        )
    if any(
        row["cohort_n"] != slice_cohort_sizes[row["slice_name"]]
        for row in scorecard_rows
    ):
        raise M5ResultStoreIntegrityError(
            "scorecard cohort_n contradicts source-only slice membership"
        )
    if any(
        row["source_pairing_complete"] is not True
        for row in scorecard_rows
    ):
        raise M5ResultStoreIntegrityError(
            "official M5 scorecards require policy-independent source pairing"
        )
    if any(
        not (
            row["cohort_n"]
            == row["valid_a_n"]
            == row["valid_b_n"]
            == row["paired_n"]
            == 128
        )
        for row in scorecard_rows
        if (
            row["metric_name"] in M5_PRIMARY_METRIC_NAMES
            and row["slice_name"] == "all"
        )
    ):
        raise M5ResultStoreIntegrityError(
            "official primary all-slice cells require 128 valid paired scenarios"
        )

    expected_parity_domain = {
        (parity_index, policy_name, metric_name)
        for parity_index in range(16)
        for policy_name in _M5_POLICY_NAMES
        for metric_name in _M5_PARITY_METRIC_NAMES
    }
    parity_rows = rows[WAYMAX_PARITY_SUMMARY]
    observed_parity_domain = {
        (
            row["parity_index"],
            row["policy_name"],
            row["metric_name"],
        )
        for row in parity_rows
    }
    if (
        len(parity_rows) != OFFICIAL_M5_ROW_COUNTS.waymax_parity_summary
        or observed_parity_domain != expected_parity_domain
        or any(row["compared_components"] < 1 for row in parity_rows)
        or any(
            row["status"] != "accepted"
            or row["mismatch_count"] != 0
            or row["exact_match"] is not True
            for row in parity_rows
        )
    ):
        raise M5ResultStoreIntegrityError(
            "Waymax parity does not cover the frozen 16×3×3 domain"
        )


def _catalog_fingerprints() -> dict[str, str]:
    metric_payload = [
        {
            "aggregation": spec.aggregation,
            "agent_scope": spec.agent_scope,
            "depends_on": list(spec.depends_on),
            "deterministic": spec.deterministic,
            "direction": spec.direction,
            "eligibility": spec.eligibility,
            "evaluation_window": spec.evaluation_window,
            "invalid_reason_codes": list(spec.invalid_reason_codes),
            "known_failure_modes": list(spec.known_failure_modes),
            "name": spec.name,
            "required_fields": list(spec.required_fields),
            "unit_of_analysis": spec.unit_of_analysis,
            "value_unit": spec.value_unit,
            "version": spec.version,
        }
        for spec in M5_METRIC_SPECS.values()
    ]
    slice_payload = [
        {
            "description": spec.description,
            "ineligible_reasons": list(spec.ineligible_reasons),
            "name": spec.name,
            "version": spec.version,
        }
        for spec in M5_SLICE_SPECS
    ]
    statistics_payload = {
        "base_seed": M5_BASE_SEED,
        "contrasts": [list(contrast) for contrast in _M5_POLICY_CONTRASTS],
        "other_resamples": M5_OTHER_RESAMPLES,
        "parity": {
            "discrete_metrics": [
                "kinematic_infeasibility",
                "overlap",
            ],
            "log_divergence_absolute_floor": 1e-6,
            "log_divergence_float32_ulps": 8,
            "summary_excess": "max(abs_error-allowed_tolerance)",
        },
        "pointwise_stability_level": M5_POINTWISE_STABILITY_LEVEL,
        "primary_adjusted_stability_level": (
            M5_PRIMARY_ADJUSTED_STABILITY_LEVEL
        ),
        "primary_metrics": sorted(M5_PRIMARY_METRIC_NAMES),
        "primary_resamples": M5_PRIMARY_RESAMPLES,
        "schema_version": M5_STATISTICS_SCHEMA_VERSION,
    }

    def fingerprint(payload: Any) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    return {
        "metrics": fingerprint(metric_payload),
        "slices": fingerprint(slice_payload),
        "statistics": fingerprint(statistics_payload),
    }


def _validate_manifest_identity(payload: Mapping[str, Any], run_name: str) -> None:
    required = {
        "actual_rows",
        "artifacts",
        "catalog_fingerprints",
        "complete",
        "expected_rows",
        "hash_policy",
        "provenance",
        "result_path",
        "row_accounting_profile",
        "run_name",
        "scorecard_report",
        "schema_fingerprints",
        "schema_version",
        "write_mode",
    }
    if payload.get("row_accounting_profile") == OFFICIAL_M5_PROFILE:
        required.add("official_artifacts")
    if set(payload) != required:
        raise M5ResultStoreIntegrityError(
            "final evaluation manifest has unexpected fields"
        )
    if payload.get("schema_version") != M5_RESULT_STORE_SCHEMA_VERSION:
        raise M5ResultStoreIntegrityError("unsupported result-store schema version")
    if payload.get("complete") is not True:
        raise M5ResultStoreIntegrityError("final evaluation manifest is incomplete")
    if payload.get("run_name") != run_name:
        raise M5ResultStoreIntegrityError("manifest run name does not match its path")
    if payload.get("result_path") != f"outputs/m5/{run_name}":
        raise M5ResultStoreIntegrityError(
            "manifest result path is not project-relative and canonical"
        )
    if payload.get("write_mode") != "exclusive_immutable_pending_hardlink":
        raise M5ResultStoreIntegrityError("manifest write mode is not frozen")
    if payload.get("hash_policy") != {
        "algorithm": "sha256",
        "manifest_self_hash": False,
    }:
        raise M5ResultStoreIntegrityError("manifest hash policy is not frozen")
    if payload.get("row_accounting_profile") not in {
        OFFICIAL_M5_PROFILE,
        DATA_FREE_TEST_PROFILE,
    }:
        raise M5ResultStoreIntegrityError(
            "manifest row-accounting profile is invalid"
        )


def _raise_invalid_artifact() -> ArtifactRecord:
    raise M5ResultStoreIntegrityError("manifest artifact must be an object")


def _raise_invalid_supplemental_artifact() -> SupplementalArtifactRecord:
    raise M5ResultStoreIntegrityError(
        "manifest supplemental artifact must be an object"
    )


def _integer(
    value: Any,
    *,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        if maximum is None:
            raise ValueError(f"{name} must be at least {minimum}")
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return result


OFFICIAL_M5_ROW_COUNTS = ExpectedRowCounts()


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _lower_name(value: Any, *, name: str) -> str:
    value = _text(value, name)
    if _LOWER_NAME.fullmatch(value) is None:
        raise ValueError(f"{name} must be lower_snake_case")
    return value


def _version(value: Any, name: str) -> str:
    value = _text(value, name)
    if _SEMANTIC_VERSION.fullmatch(value) is None:
        raise ValueError(f"{name} must be a semantic version")
    return value


def _slice_version(value: Any) -> str:
    value = _text(value, "slice_version")
    if value != M5_SLICE_VERSION:
        raise ValueError(
            f"slice_version must be exactly {M5_SLICE_VERSION!r}"
        )
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_sequence(value: Any, name: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return [_finite(item, f"{name} item") for item in value]


def _canonical_json_text(
    value: Any,
    name: str,
    *,
    require_object: bool,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be canonical JSON text")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON") from exc
    if require_object and not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if value != canonical:
        raise ValueError(f"{name} must use canonical JSON encoding")
    return value


def _json_mapping(value: Any, path: str) -> Mapping[str, Any]:
    normalized = _json_value(value, path=path, active=set())
    if not isinstance(normalized, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return normalized


def _json_value(value: Any, *, path: str, active: set[int]) -> Any:
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} contains a reference cycle")
        active.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(f"{path} keys must be non-empty strings")
                result[key] = _json_value(
                    item,
                    path=f"{path}.{key}",
                    active=active,
                )
            return MappingProxyType(result)
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} contains a reference cycle")
        active.add(identity)
        try:
            return tuple(
                _json_value(item, path=f"{path}[{index}]", active=active)
                for index, item in enumerate(value)
            )
        finally:
            active.remove(identity)
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError(f"{path} must contain exact finite JSON values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise M5ResultStoreError("payload is not canonical finite JSON") from exc


__all__ = [
    "ArtifactRecord",
    "DETERMINISM_RECEIPT",
    "ExpectedRowCounts",
    "M5_DETERMINISM_CANONICALIZATION_VERSION",
    "M5_DETERMINISM_RECEIPT_SCHEMA_VERSION",
    "M5_M4_INTEGRITY_ASSUMPTION",
    "M5_M4_REUSE_SCHEMA_VERSION",
    "M5_M4_SELECTED_ORDER_VERSION",
    "M5_PARITY_ORDER_RECEIPT_SCHEMA_VERSION",
    "M5_PARITY_ORDER_VERSION",
    "M5_PARITY_RANK_DOMAIN",
    "M5_PARITY_SCENE_COUNT",
    "M5_PARITY_TRANSITION_COUNT",
    "M5_RESULT_SCHEMAS",
    "M5_RESULT_STORE_SCHEMA_VERSION",
    "M5_SLICE_VERSION",
    "M5DeterminismReceipt",
    "M5ParityOrderReceipt",
    "M5RunProvenance",
    "M5ResultStore",
    "M5ResultStoreError",
    "M5ResultStoreIntegrityError",
    "M5ResultStoreStateError",
    "METRIC_RESULTS",
    "METRIC_RESULTS_SCHEMA",
    "OFFICIAL_M5_ROW_COUNTS",
    "OFFICIAL_WAYMAX_REFERENCE_PARAMETERS",
    "PARITY_ORDER_RECEIPT",
    "PreparedM5Finalization",
    "SCORECARDS",
    "SCORECARDS_SCHEMA",
    "ScorecardReportRecord",
    "SLICE_MEMBERSHIP",
    "SLICE_MEMBERSHIP_SCHEMA",
    "SupplementalArtifactRecord",
    "VerifiedM5ResultStore",
    "WAYMAX_PARITY_SUMMARY",
    "WAYMAX_PARITY_SUMMARY_SCHEMA",
    "executable_source_fingerprint",
    "official_executable_source_paths",
    "scorecard_row_from_result",
    "verify_m5_result_store",
    "verify_committed_m5_result_store",
    "verify_prepared_m5_result_store",
]
