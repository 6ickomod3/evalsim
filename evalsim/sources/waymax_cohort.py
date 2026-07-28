"""Pure M4 cohort selection and local-manifest contracts.

This module deliberately imports no Waymax, JAX, or TensorFlow runtime.  It owns the
result-independent source predicate, domain-separated ranking, deterministic cohort
selection, and the immutable local-only manifest schema.  Filesystem reading and WOMD
decoding remain in :mod:`evalsim.sources.waymax_loader`.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, NoReturn

import numpy as np

from .waymax import (
    MAX_MAP_DIRECTION_ERROR_DEGREES,
    MAX_MAP_SEGMENT_METERS,
    MIN_MAP_SEGMENT_METERS,
    WAYMAX_ADAPTER_SCHEMA_VERSION,
    WAYMAX_ADAPTER_VERSION,
    WAYMAX_COMMIT,
    WOMD_DATASET_VERSION,
)

M4_SELECTOR_VERSION = "3"
M4_MANIFEST_SCHEMA_VERSION = "1"

M4_SHARD_SUFFIXES = tuple(f"{index:05d}" for index in range(10))
M4_TFRECORD_SUFFIXES = tuple(
    f"tfrecord-{suffix}-of-00150" for suffix in M4_SHARD_SUFFIXES
)
M4_COHORT_TARGET = 128
M4_FALLBACK_FLOOR = 32
M4_IDM_SUBSET_TARGET = 16
M4_IDM_SUBSET_FLOOR = 8
M4_VMAP_PAIR_SIZE = 2

M4_COHORT_DOMAIN = "evalsim-m4-cohort-v1"
M4_REDISTRIBUTION_DOMAIN = "evalsim-m4-redistribution-v1"
M4_IDM_SUBSET_DOMAIN = "evalsim-m4-idm-subset-v1"
M4_VMAP_DOMAIN = "evalsim-m4-vmap-v1"

SOURCE_REJECTION_CODES = (
    "source_sdc_count_not_one",
    "source_sdc_future_incomplete",
    "source_no_world_vehicle_transition",
    "source_no_supported_map",
)

M4_INITIAL_QUOTAS = MappingProxyType(
    {
        suffix: 13 if index < 8 else 12
        for index, suffix in enumerate(M4_SHARD_SUFFIXES)
    }
)

_PRE_JAX_AUDIT_SCHEMA = (
    ("state/id", (128,), np.dtype(np.float32)),
    ("state/type", (128,), np.dtype(np.float32)),
    ("state/is_sdc", (128,), np.dtype(np.int64)),
    ("state/which_time", (91,), np.dtype(np.float32)),
    ("state/all/valid", (128, 91), np.dtype(np.int64)),
    ("state/all/x", (128, 91), np.dtype(np.float32)),
    ("state/all/y", (128, 91), np.dtype(np.float32)),
    ("state/all/velocity_x", (128, 91), np.dtype(np.float32)),
    ("state/all/velocity_y", (128, 91), np.dtype(np.float32)),
    ("state/all/bbox_yaw", (128, 91), np.dtype(np.float32)),
    ("state/all/timestamp_micros", (128, 91), np.dtype(np.int64)),
    ("state/all/length", (128, 91), np.dtype(np.float32)),
    ("state/all/width", (128, 91), np.dtype(np.float32)),
    ("roadgraph_samples/xyz", (30000, 3), np.dtype(np.float32)),
    ("roadgraph_samples/dir", (30000, 3), np.dtype(np.float32)),
    ("roadgraph_samples/type", (30000, 1), np.dtype(np.int64)),
    ("roadgraph_samples/id", (30000, 1), np.dtype(np.int64)),
    ("roadgraph_samples/valid", (30000, 1), np.dtype(np.int64)),
)
_AUDIT_FIELDS = tuple(name for name, _, _ in _PRE_JAX_AUDIT_SCHEMA)
_PRE_JAX_AUDIT_SCHEMA_BY_FIELD = MappingProxyType(
    {
        name: (shape, dtype)
        for name, shape, dtype in _PRE_JAX_AUDIT_SCHEMA
    }
)
_VALID_MOTION_FIELDS = (
    "state/all/x",
    "state/all/y",
    "state/all/velocity_x",
    "state/all/velocity_y",
    "state/all/bbox_yaw",
)
_DIMENSION_FIELDS = ("state/all/length", "state/all/width")
_LANE_TYPE_IDS = frozenset({0, 1, 2, 3})
_ROAD_EDGE_TYPE_IDS = frozenset({14, 15, 16})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_EXPECTED_WHICH_TIME = np.concatenate(
    (
        -np.ones(10, dtype=np.int8),
        np.zeros(1, dtype=np.int8),
        np.ones(80, dtype=np.int8),
    )
)
_SHARD_ORDER = {suffix: index for index, suffix in enumerate(M4_SHARD_SUFFIXES)}


class CohortInvariantError(ValueError):
    """A fatal M4 population, selector, or manifest invariant failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class CohortSelectionError(CohortInvariantError):
    """A pre-registered cohort or nested-subset evidence floor was not met."""


def selector_config_payload() -> dict[str, Any]:
    """Return a fresh JSON-native payload freezing every M4 selector rule."""

    return {
        "selector_version": M4_SELECTOR_VERSION,
        "population": {
            "ordered_shard_suffixes": list(M4_SHARD_SUFFIXES),
            "exact_tfrecord_suffixes": list(M4_TFRECORD_SUFFIXES),
            "womd_split": "validation",
            "womd_version": WOMD_DATASET_VERSION,
            "dataset_config": {
                "repeat": 1,
                "shuffle_seed": None,
                "deterministic": True,
                "num_shards": 1,
                "batch_dims": [],
                "batch_by_scenario": True,
                "max_num_objects": 128,
                "max_num_rg_points": 30000,
                "include_sdc_paths": True,
                "aggregate_timesteps": True,
                "past_steps": 10,
                "current_steps": 1,
                "future_steps": 80,
            },
        },
        "source_predicate": {
            "audit_boundary": (
                "after_pinned_waymax_tensorflow_preprocess_"
                "before_jax_waymax_factory"
            ),
            "audit_fields": list(_AUDIT_FIELDS),
            "pre_jax_audit_schema": {
                name: {
                    "shape": list(shape),
                    "dtype": dtype.name,
                }
                for name, shape, dtype in _PRE_JAX_AUDIT_SCHEMA
            },
            "pre_jax_to_semantic_normalization": {
                "state/id": {
                    "semantic_dtype": "int32",
                    "gates": [
                        "finite",
                        "signed_int32_range",
                        "float32_to_int32_to_float32_exact_roundtrip",
                    ],
                },
                "state/type": {
                    "semantic_dtype": "int32",
                    "gates": [
                        "finite",
                        "signed_int32_range",
                        "float32_to_int32_to_float32_exact_roundtrip",
                    ],
                },
                "state/is_sdc": {
                    "semantic_dtype": "bool",
                    "gates": [
                        "int64_exactly_minus_one_zero_or_one",
                        "minus_one_only_on_never_valid_object_slots",
                        "one_only_on_retained_object_slots",
                        "semantic_true_exactly_equals_one",
                    ],
                    "retained_object_source": (
                        "strict_binary_state/all/valid_any_frames_0_through_90"
                    ),
                },
                "state/all/valid": {
                    "semantic_dtype": "bool",
                    "gates": ["binary_int64_exactly_zero_or_one"],
                },
                "state/all/timestamp_micros": {
                    "semantic_dtype": "int64",
                    "gates": [
                        "signed_int32_range",
                        "int64_to_int32_to_int64_exact_roundtrip",
                    ],
                    "consensus_dtype": "int64",
                },
                "roadgraph_samples/type": {
                    "semantic_dtype": "int32",
                    "gates": [
                        "signed_int32_range",
                        "int64_to_int32_to_int64_exact_roundtrip",
                    ],
                },
                "roadgraph_samples/id": {
                    "semantic_dtype": "int32",
                    "gates": [
                        "signed_int32_range",
                        "int64_to_int32_to_int64_exact_roundtrip",
                    ],
                },
                "roadgraph_samples/valid": {
                    "semantic_dtype": "bool",
                    "gates": ["binary_int64_exactly_zero_or_one"],
                },
            },
            "native_scenario_id": {
                "tf_source": {"dtype": "tf.string", "shape": [1]},
                "decoded": {"dtype": "uint8", "shape": [1, "N"], "N_min": 1},
                "gates": [
                    "decoded_bytes_exactly_equal_tf_string_payload",
                    "strict_utf8",
                    "nonempty_hex",
                    "preserve_original_case_and_length",
                ],
            },
            "rejection_priority": list(SOURCE_REJECTION_CODES),
            "retained_object_rule": "any_valid_frames_0_through_90",
            "sdc_count_scope": "retained_object_slots",
            "sdc_complete_frames_inclusive": [10, 90],
            "world_vehicle_type_id": 1,
            "world_vehicle_transition_frames": [10, 11],
            "valid_motion_fields": list(_VALID_MOTION_FIELDS),
            "dimensions": {
                "fields": list(_DIMENSION_FIELDS),
                "rule": (
                    "finite_positive_exactly_constant_valid_frames_only_"
                    "ignore_invalid_payload"
                ),
            },
            "timestamps": {
                "field": "state/all/timestamp_micros",
                "rule": "valid_object_exact_consensus_each_frame_strictly_increasing",
            },
            "which_time": [-1] * 10 + [0] + [1] * 80,
            "map": {
                "grouping": "valid_points_by_id_in_first_source_order",
                "one_source_type_per_group": True,
                "lane_type_ids": sorted(_LANE_TYPE_IDS),
                "road_edge_type_ids": sorted(_ROAD_EDGE_TYPE_IDS),
                "minimum_unique_xy_points": 2,
                "finite_xy_and_direction": True,
                "min_segment_meters_exclusive": MIN_MAP_SEGMENT_METERS,
                "max_segment_meters_inclusive": MAX_MAP_SEGMENT_METERS,
                "max_nonterminal_direction_error_degrees_inclusive": (
                    MAX_MAP_DIRECTION_ERROR_DEGREES
                ),
                "nonterminal_direction_must_be_nonzero": True,
                "terminal_zero_direction_allowed": True,
            },
        },
        "ranking": {
            "encoding": [
                "strict_ascii_domain",
                "zero_byte",
                "five_ascii_shard_suffix_bytes",
                "zero_byte",
                "unsigned_uint64_big_endian_ordinal",
                "zero_byte",
                "strict_utf8_native_scenario_id",
            ],
            "digest": "sha256",
            "domains": {
                "cohort": M4_COHORT_DOMAIN,
                "redistribution": M4_REDISTRIBUTION_DOMAIN,
                "idm_subset": M4_IDM_SUBSET_DOMAIN,
                "vmap_pair": M4_VMAP_DOMAIN,
            },
            "within_shard_tie_break": ["rank", "record_ordinal"],
            "redistribution_tie_break": [
                "rank",
                "shard_suffix",
                "record_ordinal",
            ],
        },
        "cohort": {
            "target": M4_COHORT_TARGET,
            "initial_quotas": dict(M4_INITIAL_QUOTAS),
            "canonical_selected_order": [
                "shard_suffix",
                "cohort_rank",
                "record_ordinal",
            ],
            "redistribute_deficits_globally_without_replacement": True,
            "fallback": {
                "condition": "total_eligible_below_target",
                "minimum_total": M4_FALLBACK_FLOOR,
                "require_every_shard_represented": True,
                "selection": "complete_eligible_population",
            },
        },
        "idm_subset": {
            "target": M4_IDM_SUBSET_TARGET,
            "fallback_floor": M4_IDM_SUBSET_FLOOR,
            "continuously_valid_non_sdc_vehicle_frames_inclusive": [10, 30],
            "initialized_overlap_frame": 0,
            "initialized_overlap_uses_all_128_boxes": True,
            "initialized_overlap_uses_validity_mask": False,
            "initialized_overlap_exclusion": (
                "exclude_any_actor_overlapping_any_other_box_after_diagonal_removal"
            ),
            "tie_break": ["rank", "shard_suffix", "record_ordinal"],
            "future_transitions": 20,
            "minimum_effective_controlled_transitions_per_scene": 20,
            "nonfallback_difference_threshold_exclusive": 1e-6,
            "actor_control_mask": {
                "not_sdc": True,
                "object_type_id": 1,
                "logged_valid_frames": ["current", "next"],
                "initialized_overlap_excluded": True,
            },
            "lifecycle_fallback": (
                "log_for_birth_disappearance_or_invalid_transition"
            ),
            "upstream_idm_defaults": {
                "desired_velocity_mps": 30.0,
                "minimum_spacing_m": 2.0,
                "safe_headway_s": 2.0,
                "maximum_acceleration_mps2": 2.0,
                "maximum_deceleration_mps2": 4.0,
                "exponent": 4,
                "maximum_lookahead": 10,
                "lookahead_from_current_position": True,
                "additional_lookahead_points": 10,
                "additional_lookahead_distance_m": 10.0,
                "invalidate_on_end": False,
            },
        },
        "vmap": {
            "pair_size": M4_VMAP_PAIR_SIZE,
            "tie_break": ["rank", "shard_suffix", "record_ordinal"],
            "static_shapes": {
                "object_slots": 128,
                "trajectory_frames": 91,
                "roadgraph_points": 30000,
                "sdc_paths": [45, 800],
            },
            "requires_sequential_jit_vmap_and_permutation_parity": True,
        },
        "parity": {
            "float_rtol": 0.0,
            "float_atol": 1e-6,
            "exact_fields": [
                "native_identity",
                "agent_order",
                "sdc",
                "valid",
                "timestamp_micros",
                "selection",
            ],
            "compact_float_fields": [
                "x",
                "y",
                "yaw",
                "vel_x",
                "vel_y",
            ],
        },
        "manifest": {
            "schema_version": M4_MANIFEST_SCHEMA_VERSION,
            "event_outcomes": ["eligible", "rejected"],
            "counter_invariant": (
                "raw_seen==decode_attempted==event_emitted==eligible+rejected"
            ),
            "clean_eof_required": True,
            "duplicate_native_identity": "fatal",
            "write_mode": "exclusive_create",
            "canonical_json": "utf8_sort_keys_compact_newline",
        },
        "versions": {
            "waymax_commit": WAYMAX_COMMIT,
            "adapter_version": WAYMAX_ADAPTER_VERSION,
            "adapter_schema_version": WAYMAX_ADAPTER_SCHEMA_VERSION,
        },
    }


def selector_config_fingerprint() -> str:
    """Return the SHA-256 of the complete canonical selector payload."""

    encoded = json.dumps(
        selector_config_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8", errors="strict")
    return hashlib.sha256(encoded).hexdigest()


M4_SELECTOR_CONFIG_FINGERPRINT = selector_config_fingerprint()


def source_rejection_code(audit: Mapping[str, Any]) -> str | None:
    """Apply the four source-only rejection predicates in their frozen priority.

    Malformed inputs and source/contract drift raise :class:`CohortInvariantError`.
    The function does not call the M3 adapter and does not inspect policy output.
    """

    arrays = _validated_audit_arrays(audit)
    valid = arrays["state/all/valid"]
    retained = np.any(valid, axis=1)
    retained_sdc = arrays["state/is_sdc"][retained]

    if int(np.count_nonzero(retained_sdc)) != 1:
        return SOURCE_REJECTION_CODES[0]

    sdc_source_index = int(
        np.flatnonzero(retained)[np.flatnonzero(retained_sdc)[0]]
    )
    if not np.all(valid[sdc_source_index, 10:91]):
        return SOURCE_REJECTION_CODES[1]

    is_world_vehicle = (
        retained
        & ~arrays["state/is_sdc"]
        & (arrays["state/type"] == 1)
    )
    if not np.any(is_world_vehicle & valid[:, 10] & valid[:, 11]):
        return SOURCE_REJECTION_CODES[2]

    if not _has_supported_map(arrays):
        return SOURCE_REJECTION_CODES[3]
    return None


def ranking_message(
    domain: str,
    shard_suffix: str,
    record_ordinal: int,
    native_scenario_id: str,
) -> bytes:
    """Encode one unambiguous M4 ranking message."""

    if not isinstance(domain, str) or not domain:
        raise CohortInvariantError(
            "rank_domain_invalid",
            "ranking domain must be a non-empty string",
        )
    try:
        domain_bytes = domain.encode("ascii", errors="strict")
    except UnicodeEncodeError as exc:
        raise CohortInvariantError(
            "rank_domain_invalid",
            "ranking domain must contain strict ASCII only",
        ) from exc
    if b"\x00" in domain_bytes:
        raise CohortInvariantError(
            "rank_domain_invalid",
            "ranking domain may not contain a zero byte",
        )
    _validate_shard_suffix(shard_suffix)
    ordinal = _validate_uint64(record_ordinal, "record_ordinal")
    scenario_bytes = _strict_identity_bytes(native_scenario_id)
    return (
        domain_bytes
        + b"\x00"
        + shard_suffix.encode("ascii")
        + b"\x00"
        + ordinal.to_bytes(8, byteorder="big", signed=False)
        + b"\x00"
        + scenario_bytes
    )


def rank_record(
    domain: str,
    shard_suffix: str,
    record_ordinal: int,
    native_scenario_id: str,
) -> str:
    """Return the lowercase SHA-256 rank for one locator and identity."""

    return hashlib.sha256(
        ranking_message(
            domain,
            shard_suffix,
            record_ordinal,
            native_scenario_id,
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ScanEvent:
    """One local-only M4 scan result after all required checks for the record."""

    shard_suffix: str
    record_ordinal: int
    native_scenario_id: str
    shard_sha256: str
    dataset_config_fingerprint: str
    outcome: str
    rejection_code: str | None
    selection_rank: str | None
    selected: bool = False
    womd_dataset_version: str = WOMD_DATASET_VERSION
    waymax_commit: str = WAYMAX_COMMIT
    adapter_version: str = WAYMAX_ADAPTER_VERSION
    adapter_schema_version: str = WAYMAX_ADAPTER_SCHEMA_VERSION
    selector_version: str = M4_SELECTOR_VERSION
    manifest_schema_version: str = M4_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_shard_suffix(self.shard_suffix)
        object.__setattr__(
            self,
            "record_ordinal",
            _validate_uint64(self.record_ordinal, "record_ordinal"),
        )
        _strict_identity_bytes(self.native_scenario_id)
        _validate_sha256(self.shard_sha256, "shard_sha256")
        _validate_sha256(
            self.dataset_config_fingerprint,
            "dataset_config_fingerprint",
        )
        if type(self.selected) is not bool:
            _fail("scan_event_invalid", "selected must be a boolean")
        if self.outcome not in {"eligible", "rejected"}:
            _fail(
                "scan_event_invalid",
                "outcome must be exactly 'eligible' or 'rejected'",
            )
        expected_rank = rank_record(
            M4_COHORT_DOMAIN,
            self.shard_suffix,
            self.record_ordinal,
            self.native_scenario_id,
        )
        if self.outcome == "eligible":
            if self.rejection_code is not None:
                _fail(
                    "scan_event_invalid",
                    "an eligible event cannot contain a rejection code",
                )
            if self.selection_rank != expected_rank:
                _fail(
                    "scan_event_invalid",
                    "eligible selection_rank does not match the frozen encoding",
                )
        else:
            if self.rejection_code not in SOURCE_REJECTION_CODES:
                _fail(
                    "scan_event_invalid",
                    "a rejected event must use one registered source reason",
                )
            if self.selection_rank is not None or self.selected:
                _fail(
                    "scan_event_invalid",
                    "a rejected event cannot have a rank or cohort membership",
                )
        expected_versions = {
            "womd_dataset_version": WOMD_DATASET_VERSION,
            "waymax_commit": WAYMAX_COMMIT,
            "adapter_version": WAYMAX_ADAPTER_VERSION,
            "adapter_schema_version": WAYMAX_ADAPTER_SCHEMA_VERSION,
            "selector_version": M4_SELECTOR_VERSION,
            "manifest_schema_version": M4_MANIFEST_SCHEMA_VERSION,
        }
        for name, expected in expected_versions.items():
            if getattr(self, name) != expected:
                _fail(
                    "scan_event_version_drift",
                    f"{name} differs from the locked M4 value",
                )

    @classmethod
    def eligible_event(
        cls,
        *,
        shard_suffix: str,
        record_ordinal: int,
        native_scenario_id: str,
        shard_sha256: str,
        dataset_config_fingerprint: str,
    ) -> "ScanEvent":
        return cls(
            shard_suffix=shard_suffix,
            record_ordinal=record_ordinal,
            native_scenario_id=native_scenario_id,
            shard_sha256=shard_sha256,
            dataset_config_fingerprint=dataset_config_fingerprint,
            outcome="eligible",
            rejection_code=None,
            selection_rank=rank_record(
                M4_COHORT_DOMAIN,
                shard_suffix,
                record_ordinal,
                native_scenario_id,
            ),
        )

    @classmethod
    def rejected_event(
        cls,
        *,
        shard_suffix: str,
        record_ordinal: int,
        native_scenario_id: str,
        shard_sha256: str,
        dataset_config_fingerprint: str,
        rejection_code: str,
    ) -> "ScanEvent":
        return cls(
            shard_suffix=shard_suffix,
            record_ordinal=record_ordinal,
            native_scenario_id=native_scenario_id,
            shard_sha256=shard_sha256,
            dataset_config_fingerprint=dataset_config_fingerprint,
            outcome="rejected",
            rejection_code=rejection_code,
            selection_rank=None,
        )

    @property
    def locator(self) -> tuple[str, int]:
        return (self.shard_suffix, self.record_ordinal)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_schema_version": self.adapter_schema_version,
            "adapter_version": self.adapter_version,
            "dataset_config_fingerprint": self.dataset_config_fingerprint,
            "manifest_schema_version": self.manifest_schema_version,
            "native_scenario_id": self.native_scenario_id,
            "outcome": self.outcome,
            "record_ordinal": self.record_ordinal,
            "rejection_code": self.rejection_code,
            "selected": self.selected,
            "selection_rank": self.selection_rank,
            "selector_version": self.selector_version,
            "shard_sha256": self.shard_sha256,
            "shard_suffix": self.shard_suffix,
            "waymax_commit": self.waymax_commit,
            "womd_dataset_version": self.womd_dataset_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScanEvent":
        expected = frozenset(cls.__dataclass_fields__)
        _require_exact_keys(payload, expected, "scan event")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class ShardScanCounts:
    """Independent per-shard raw/decode/event accounting."""

    shard_suffix: str
    raw_seen: int
    decode_attempted: int
    event_emitted: int
    eligible: int
    rejected: int
    clean_eof: bool

    def __post_init__(self) -> None:
        _validate_shard_suffix(self.shard_suffix)
        for name in (
            "raw_seen",
            "decode_attempted",
            "event_emitted",
            "eligible",
            "rejected",
        ):
            object.__setattr__(
                self,
                name,
                _validate_nonnegative_int(getattr(self, name), name),
            )
        if self.clean_eof is not True:
            _fail(
                "scan_not_clean_eof",
                "per-shard manifest accounting requires a clean raw-stream EOF",
            )
        if not (
            self.raw_seen
            == self.decode_attempted
            == self.event_emitted
            == self.eligible + self.rejected
        ):
            _fail(
                "scan_counter_mismatch",
                "raw_seen, decode_attempted, event_emitted, and "
                "eligible + rejected must be equal",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean_eof": self.clean_eof,
            "decode_attempted": self.decode_attempted,
            "eligible": self.eligible,
            "event_emitted": self.event_emitted,
            "raw_seen": self.raw_seen,
            "rejected": self.rejected,
            "shard_suffix": self.shard_suffix,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShardScanCounts":
        expected = frozenset(cls.__dataclass_fields__)
        _require_exact_keys(payload, expected, "shard scan counts")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class _SelectionParts:
    events: tuple[ScanEvent, ...]
    quota_deficits: tuple[tuple[str, int], ...]
    redistributed_count: int
    fallback_used: bool


@dataclass(frozen=True, slots=True)
class CohortSelection:
    """Frozen result of the M4 quota, redistribution, and fallback rules."""

    events: tuple[ScanEvent, ...]
    quota_deficits: tuple[tuple[str, int], ...]
    redistributed_count: int
    fallback_used: bool

    def __post_init__(self) -> None:
        events = tuple(self.events)
        try:
            deficits = tuple(
                (
                    item[0],
                    _validate_nonnegative_int(item[1], "quota_deficit"),
                )
                for item in self.quota_deficits
            )
        except (IndexError, TypeError) as exc:
            raise CohortInvariantError(
                "selection_invalid",
                "quota_deficits must contain (shard_suffix, count) pairs",
            ) from exc
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "quota_deficits", deficits)
        if type(self.fallback_used) is not bool:
            _fail("selection_invalid", "fallback_used must be a boolean")
        object.__setattr__(
            self,
            "redistributed_count",
            _validate_nonnegative_int(
                self.redistributed_count,
                "redistributed_count",
            ),
        )
        reset_events = tuple(replace(event, selected=False) for event in events)
        expected = _compute_selection(reset_events)
        if (
            events != expected.events
            or deficits != expected.quota_deficits
            or self.redistributed_count != expected.redistributed_count
            or self.fallback_used != expected.fallback_used
        ):
            _fail(
                "selection_tampered",
                "cohort membership or accounting differs from the frozen selector",
            )

    @property
    def selected_events(self) -> tuple[ScanEvent, ...]:
        return tuple(
            sorted(
                (event for event in self.events if event.selected),
                key=lambda event: (
                    _SHARD_ORDER[event.shard_suffix],
                    event.selection_rank,
                    event.record_ordinal,
                ),
            )
        )

    @property
    def actual_count(self) -> int:
        return len(self.selected_events)


def select_cohort(events: Sequence[ScanEvent]) -> CohortSelection:
    """Select the fixed M4 cohort from complete eligible/rejected scan events."""

    source_events = tuple(events)
    if any(event.selected for event in source_events):
        _fail(
            "selection_already_applied",
            "select_cohort requires unselected scan events",
        )
    parts = _compute_selection(source_events)
    return CohortSelection(
        events=parts.events,
        quota_deficits=parts.quota_deficits,
        redistributed_count=parts.redistributed_count,
        fallback_used=parts.fallback_used,
    )


def select_idm_subset(
    selected_events: Sequence[ScanEvent],
    qualification: Mapping[tuple[str, int], bool],
) -> tuple[ScanEvent, ...]:
    """Choose the result-independent nested IDM subset from the frozen cohort.

    ``qualification`` must classify every selected locator exactly once.  It is
    computed by the runtime seam from continuous validity and the independent
    initialized-overlap oracle before any IDM output is inspected.
    """

    population = _validate_selected_population(selected_events)
    expected_keys = {event.locator for event in population}
    if not isinstance(qualification, Mapping):
        _fail("idm_qualification_invalid", "qualification must be a mapping")
    if set(qualification) != expected_keys:
        _fail(
            "idm_qualification_invalid",
            "qualification must classify every selected locator exactly once",
        )
    for value in qualification.values():
        if type(value) is not bool:
            _fail(
                "idm_qualification_invalid",
                "qualification values must be booleans",
            )
    qualifying = [
        event for event in population if qualification[event.locator]
    ]
    qualifying.sort(
        key=lambda event: (
            rank_record(
                M4_IDM_SUBSET_DOMAIN,
                event.shard_suffix,
                event.record_ordinal,
                event.native_scenario_id,
            ),
            event.shard_suffix,
            event.record_ordinal,
        )
    )
    if len(qualifying) < M4_IDM_SUBSET_FLOOR:
        raise CohortSelectionError(
            "idm_subset_floor_not_met",
            f"the nested IDM subset requires at least {M4_IDM_SUBSET_FLOOR} "
            "qualifying scenarios",
        )
    return tuple(qualifying[:M4_IDM_SUBSET_TARGET])


def select_vmap_pair(
    selected_events: Sequence[ScanEvent],
) -> tuple[ScanEvent, ...]:
    """Choose the fixed two-scene explicit-vmap pair before execution."""

    population = list(_validate_selected_population(selected_events))
    if len(population) < M4_VMAP_PAIR_SIZE:
        raise CohortSelectionError(
            "vmap_pair_floor_not_met",
            f"vmap selection requires {M4_VMAP_PAIR_SIZE} selected scenarios",
        )
    population.sort(
        key=lambda event: (
            rank_record(
                M4_VMAP_DOMAIN,
                event.shard_suffix,
                event.record_ordinal,
                event.native_scenario_id,
            ),
            event.shard_suffix,
            event.record_ordinal,
        )
    )
    return tuple(population[:M4_VMAP_PAIR_SIZE])


@dataclass(frozen=True, slots=True)
class WaymaxCohortManifest:
    """Canonical, immutable, local-only M4 manifest."""

    selection: CohortSelection
    shard_counts: tuple[ShardScanCounts, ...]
    selector_config_fingerprint: str = M4_SELECTOR_CONFIG_FINGERPRINT
    selector_version: str = M4_SELECTOR_VERSION
    schema_version: str = M4_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        counts = tuple(self.shard_counts)
        object.__setattr__(self, "shard_counts", counts)
        if self.selector_config_fingerprint != M4_SELECTOR_CONFIG_FINGERPRINT:
            _fail(
                "manifest_selector_drift",
                "selector_config_fingerprint differs from this implementation",
            )
        if self.selector_version != M4_SELECTOR_VERSION:
            _fail(
                "manifest_selector_drift",
                "selector_version differs from this implementation",
            )
        if self.schema_version != M4_MANIFEST_SCHEMA_VERSION:
            _fail(
                "manifest_schema_unsupported",
                "manifest schema version is unsupported",
            )
        if not isinstance(self.selection, CohortSelection):
            _fail(
                "manifest_invalid",
                "selection must be a CohortSelection",
            )
        if tuple(item.shard_suffix for item in counts) != M4_SHARD_SUFFIXES:
            _fail(
                "manifest_shard_order",
                "shard_counts must contain the exact ten suffixes in order",
            )
        events = self.selection.events
        expected_event_order = tuple(
            sorted(
                events,
                key=lambda event: (
                    _SHARD_ORDER[event.shard_suffix],
                    event.record_ordinal,
                ),
            )
        )
        if events != expected_event_order:
            _fail(
                "manifest_event_order",
                "events must be in canonical shard/ordinal scan order",
            )
        for count in counts:
            shard_events = tuple(
                event
                for event in events
                if event.shard_suffix == count.shard_suffix
            )
            if tuple(event.record_ordinal for event in shard_events) != tuple(
                range(count.raw_seen)
            ):
                _fail(
                    "manifest_ordinal_sequence",
                    "each shard must contain every ordinal from zero to raw_seen - 1",
                )
            eligible = sum(event.outcome == "eligible" for event in shard_events)
            rejected = sum(event.outcome == "rejected" for event in shard_events)
            if (
                len(shard_events) != count.event_emitted
                or eligible != count.eligible
                or rejected != count.rejected
            ):
                _fail(
                    "manifest_counter_mismatch",
                    "manifest events disagree with per-shard scan counters",
                )
            shard_digests = {event.shard_sha256 for event in shard_events}
            if len(shard_digests) > 1:
                _fail(
                    "manifest_shard_digest_drift",
                    "one shard has more than one source digest",
                )
        config_fingerprints = {
            event.dataset_config_fingerprint for event in events
        }
        if len(config_fingerprints) != 1:
            _fail(
                "manifest_dataset_config_drift",
                "all events must share one path-independent dataset config",
            )

    @property
    def events(self) -> tuple[ScanEvent, ...]:
        return self.selection.events

    @property
    def selected_events(self) -> tuple[ScanEvent, ...]:
        return self.selection.selected_events

    @classmethod
    def build(
        cls,
        *,
        events: Sequence[ScanEvent],
        shard_counts: Sequence[ShardScanCounts],
    ) -> "WaymaxCohortManifest":
        return cls(
            selection=select_cohort(events),
            shard_counts=tuple(shard_counts),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [event.to_dict() for event in self.events],
            "schema_version": self.schema_version,
            "selection": _selection_payload(self.selection),
            "selector_config_fingerprint": self.selector_config_fingerprint,
            "selector_version": self.selector_version,
            "shard_counts": [count.to_dict() for count in self.shard_counts],
        }

    def to_json(self) -> str:
        return _canonical_json_bytes(self.to_dict()).decode("utf-8")

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.to_dict()) + b"\n"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def to_file(self, path: str | Path) -> None:
        """Write canonical bytes once; never replace an existing manifest."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(self.canonical_bytes())

    @classmethod
    def from_json(cls, text: str) -> "WaymaxCohortManifest":
        payload = _strict_json_loads(text)
        if not isinstance(payload, Mapping):
            _fail("manifest_json_invalid", "manifest JSON must contain an object")
        if _canonical_json_bytes(payload).decode("utf-8") != text:
            _fail(
                "manifest_json_noncanonical",
                "manifest JSON must use the exact canonical byte encoding",
            )
        _require_exact_keys(
            payload,
            {
                "events",
                "schema_version",
                "selection",
                "selector_config_fingerprint",
                "selector_version",
                "shard_counts",
            },
            "manifest",
        )
        raw_events = payload["events"]
        raw_counts = payload["shard_counts"]
        raw_selection = payload["selection"]
        if not isinstance(raw_events, list) or not isinstance(raw_counts, list):
            _fail(
                "manifest_json_invalid",
                "events and shard_counts must be JSON arrays",
            )
        if not isinstance(raw_selection, Mapping):
            _fail(
                "manifest_json_invalid",
                "selection must be a JSON object",
            )
        events = tuple(
            ScanEvent.from_dict(item)
            if isinstance(item, Mapping)
            else _fail("manifest_json_invalid", "every event must be an object")
            for item in raw_events
        )
        counts = tuple(
            ShardScanCounts.from_dict(item)
            if isinstance(item, Mapping)
            else _fail(
                "manifest_json_invalid",
                "every shard count must be an object",
            )
            for item in raw_counts
        )
        selection = _selection_from_payload(events, raw_selection)
        return cls(
            selection=selection,
            shard_counts=counts,
            selector_config_fingerprint=payload[
                "selector_config_fingerprint"
            ],
            selector_version=payload["selector_version"],
            schema_version=payload["schema_version"],
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "WaymaxCohortManifest":
        try:
            raw = Path(path).read_bytes()
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise CohortInvariantError(
                "manifest_json_invalid",
                "manifest is not strict UTF-8",
            ) from exc
        if not text.endswith("\n"):
            _fail(
                "manifest_json_noncanonical",
                "manifest file must end in one canonical newline",
            )
        manifest = cls.from_json(text[:-1])
        if raw != manifest.canonical_bytes():
            _fail(
                "manifest_json_noncanonical",
                "manifest file bytes differ from the canonical encoding",
            )
        return manifest


def _validated_audit_arrays(
    audit: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if not isinstance(audit, Mapping):
        _fail("audit_invalid", "audit must be a mapping")
    arrays: dict[str, np.ndarray] = {}
    for name in _AUDIT_FIELDS:
        if name not in audit:
            _fail("audit_field_missing", f"audit omitted required field {name}")
        try:
            array = np.asarray(audit[name])
        except (TypeError, ValueError) as exc:
            raise CohortInvariantError(
                "audit_array_invalid",
                f"{name} is not a rectangular eager array",
            ) from exc
        expected_shape, expected_dtype = _PRE_JAX_AUDIT_SCHEMA_BY_FIELD[name]
        if array.shape != expected_shape or array.dtype != expected_dtype:
            _fail(
                "audit_shape_or_dtype_drift",
                f"{name} must have exact pre-JAX shape {expected_shape} "
                f"and dtype {expected_dtype.name}",
            )
        arrays[name] = array

    arrays["state/id"] = _lossless_float32_to_int32(
        arrays["state/id"],
        "state/id",
    )
    arrays["state/type"] = _lossless_float32_to_int32(
        arrays["state/type"],
        "state/type",
    )
    for name in ("state/all/valid", "roadgraph_samples/valid"):
        arrays[name] = _binary_int64_to_bool(arrays[name], name)
    retained = np.any(arrays["state/all/valid"], axis=1)
    arrays["state/is_sdc"] = _sdc_int64_to_bool(
        arrays["state/is_sdc"],
        retained,
    )
    arrays["roadgraph_samples/type"] = _lossless_int64_to_int32(
        arrays["roadgraph_samples/type"],
        "roadgraph_samples/type",
    )
    arrays["roadgraph_samples/id"] = _lossless_int64_to_int32(
        arrays["roadgraph_samples/id"],
        "roadgraph_samples/id",
    )

    valid = arrays["state/all/valid"]
    which_time = arrays["state/which_time"]
    if (
        not np.all(np.isfinite(which_time))
        or not np.array_equal(which_time, _EXPECTED_WHICH_TIME)
    ):
        _fail(
            "which_time_drift",
            "state/which_time must preserve the exact 10/1/80 boundary",
        )

    timestamps = arrays["state/all/timestamp_micros"]
    _lossless_int64_to_int32(
        timestamps,
        "state/all/timestamp_micros",
    )

    retained_ids = arrays["state/id"][retained]
    if np.unique(retained_ids).size != retained_ids.size:
        _fail(
            "duplicate_native_object_id",
            "retained source object IDs must be unique",
        )
    for name in _VALID_MOTION_FIELDS:
        if not np.all(np.isfinite(arrays[name][valid])):
            _fail(
                "nonfinite_valid_motion",
                f"{name} contains a non-finite valid value",
            )
    for name in _DIMENSION_FIELDS:
        for source_index in np.flatnonzero(retained):
            values = arrays[name][source_index, valid[source_index]]
            if (
                not np.all(np.isfinite(values))
                or np.any(values <= 0.0)
                or not np.all(values == values[0])
            ):
                _fail(
                    "dimension_not_constant",
                    f"{name} must be finite, positive, and exactly constant "
                    "over valid frames",
                )

    canonical_timestamps = np.empty(91, dtype=np.int64)
    for step in range(91):
        contributors = timestamps[:, step][valid[:, step]]
        if contributors.size == 0:
            _fail(
                "timestamp_absent",
                "at least one valid object must provide each source timestamp",
            )
        if not np.all(contributors == contributors[0]):
            _fail(
                "timestamp_disagreement",
                "valid objects disagree on a source timestamp",
            )
        canonical_timestamps[step] = int(contributors[0])
    if np.any(np.diff(canonical_timestamps) <= 0):
        _fail(
            "timestamp_not_increasing",
            "source timestamps must be strictly increasing",
        )

    return arrays


def _has_supported_map(arrays: Mapping[str, np.ndarray]) -> bool:
    xyz = arrays["roadgraph_samples/xyz"]
    directions = arrays["roadgraph_samples/dir"]
    types = arrays["roadgraph_samples/type"][:, 0]
    ids = arrays["roadgraph_samples/id"][:, 0]
    valid = arrays["roadgraph_samples/valid"][:, 0]
    groups: OrderedDict[int, list[int]] = OrderedDict()
    for source_index in np.flatnonzero(valid):
        groups.setdefault(int(ids[source_index]), []).append(int(source_index))

    threshold = math.cos(math.radians(MAX_MAP_DIRECTION_ERROR_DEGREES))
    for source_indices in groups.values():
        indices = np.asarray(source_indices, dtype=np.int64)
        group_types = np.unique(types[indices])
        if group_types.size != 1:
            continue
        source_type = int(group_types[0])
        if (
            source_type not in _LANE_TYPE_IDS
            and source_type not in _ROAD_EDGE_TYPE_IDS
        ):
            continue
        xy = np.asarray(xyz[indices, :2], dtype=np.float64)
        direction_xy = np.asarray(directions[indices, :2], dtype=np.float64)
        if (
            not np.all(np.isfinite(xy))
            or not np.all(np.isfinite(direction_xy))
            or xy.shape[0] < 2
            or np.unique(xy, axis=0).shape[0] < 2
        ):
            continue
        segments = np.diff(xy, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        if (
            np.any(lengths <= MIN_MAP_SEGMENT_METERS)
            or np.any(lengths > MAX_MAP_SEGMENT_METERS)
        ):
            continue
        source_directions = direction_xy[:-1]
        norms = np.linalg.norm(source_directions, axis=1)
        if np.any(norms <= 0.0):
            continue
        cosines = np.sum(
            (segments / lengths[:, np.newaxis])
            * (source_directions / norms[:, np.newaxis]),
            axis=1,
        )
        if np.any(cosines < threshold):
            continue
        return True
    return False


def _compute_selection(events: Sequence[ScanEvent]) -> _SelectionParts:
    population = tuple(events)
    if any(not isinstance(event, ScanEvent) for event in population):
        _fail("selection_invalid", "events must contain only ScanEvent values")
    canonical = tuple(
        sorted(
            population,
            key=lambda event: (
                _SHARD_ORDER[event.shard_suffix],
                event.record_ordinal,
            ),
        )
    )
    locators = [event.locator for event in canonical]
    if len(set(locators)) != len(locators):
        _fail("duplicate_locator", "scan event locators must be unique")
    identities = [event.native_scenario_id for event in canonical]
    if len(set(identities)) != len(identities):
        _fail(
            "duplicate_native_scenario_id",
            "native scenario identity must be unique across all ten shards",
        )

    eligible_by_shard: dict[str, list[ScanEvent]] = {
        suffix: [] for suffix in M4_SHARD_SUFFIXES
    }
    for event in canonical:
        if event.outcome == "eligible":
            eligible_by_shard[event.shard_suffix].append(event)
    for shard_events in eligible_by_shard.values():
        shard_events.sort(
            key=lambda event: (event.selection_rank, event.record_ordinal)
        )
    eligible_total = sum(len(items) for items in eligible_by_shard.values())
    deficits = tuple(
        (
            suffix,
            max(0, M4_INITIAL_QUOTAS[suffix] - len(eligible_by_shard[suffix])),
        )
        for suffix in M4_SHARD_SUFFIXES
    )

    selected_locators: set[tuple[str, int]]
    fallback_used = eligible_total < M4_COHORT_TARGET
    redistributed_count = 0
    if fallback_used:
        represented = all(eligible_by_shard[suffix] for suffix in M4_SHARD_SUFFIXES)
        if eligible_total < M4_FALLBACK_FLOOR or not represented:
            raise CohortSelectionError(
                "cohort_fallback_floor_not_met",
                "an under-128 cohort requires at least 32 eligible records and "
                "representation from all ten shards",
            )
        selected_locators = {
            event.locator
            for shard_events in eligible_by_shard.values()
            for event in shard_events
        }
    else:
        initially_selected = [
            event
            for suffix in M4_SHARD_SUFFIXES
            for event in eligible_by_shard[suffix][
                : M4_INITIAL_QUOTAS[suffix]
            ]
        ]
        selected_locators = {event.locator for event in initially_selected}
        needed = M4_COHORT_TARGET - len(selected_locators)
        remaining = [
            event
            for shard_events in eligible_by_shard.values()
            for event in shard_events
            if event.locator not in selected_locators
        ]
        remaining.sort(
            key=lambda event: (
                rank_record(
                    M4_REDISTRIBUTION_DOMAIN,
                    event.shard_suffix,
                    event.record_ordinal,
                    event.native_scenario_id,
                ),
                event.shard_suffix,
                event.record_ordinal,
            )
        )
        for event in remaining[:needed]:
            selected_locators.add(event.locator)
        redistributed_count = needed
        if len(selected_locators) != M4_COHORT_TARGET:
            _fail(
                "selection_internal_error",
                "eligible total met the target but redistribution did not",
            )

    selected_events = tuple(
        replace(event, selected=event.locator in selected_locators)
        for event in canonical
    )
    return _SelectionParts(
        events=selected_events,
        quota_deficits=deficits,
        redistributed_count=redistributed_count,
        fallback_used=fallback_used,
    )


def _validate_selected_population(
    selected_events: Sequence[ScanEvent],
) -> tuple[ScanEvent, ...]:
    population = tuple(selected_events)
    if any(
        not isinstance(event, ScanEvent)
        or event.outcome != "eligible"
        or not event.selected
        for event in population
    ):
        _fail(
            "selected_population_invalid",
            "nested selection requires eligible selected ScanEvent values",
        )
    locators = [event.locator for event in population]
    identities = [event.native_scenario_id for event in population]
    if len(set(locators)) != len(locators) or len(set(identities)) != len(
        identities
    ):
        _fail(
            "selected_population_invalid",
            "selected locators and native identities must be unique",
        )
    return population


def _selection_payload(selection: CohortSelection) -> dict[str, Any]:
    return {
        "actual_count": selection.actual_count,
        "fallback_used": selection.fallback_used,
        "quota_deficits": dict(selection.quota_deficits),
        "redistributed_count": selection.redistributed_count,
        "selected": [
            {
                "native_scenario_id": event.native_scenario_id,
                "record_ordinal": event.record_ordinal,
                "selection_rank": event.selection_rank,
                "shard_suffix": event.shard_suffix,
            }
            for event in selection.selected_events
        ],
        "target_count": M4_COHORT_TARGET,
    }


def _selection_from_payload(
    events: tuple[ScanEvent, ...],
    payload: Mapping[str, Any],
) -> CohortSelection:
    _require_exact_keys(
        payload,
        {
            "actual_count",
            "fallback_used",
            "quota_deficits",
            "redistributed_count",
            "selected",
            "target_count",
        },
        "manifest selection",
    )
    raw_deficits = payload["quota_deficits"]
    if not isinstance(raw_deficits, Mapping):
        _fail(
            "manifest_json_invalid",
            "selection quota_deficits must be an object",
        )
    if tuple(raw_deficits) != tuple(sorted(M4_SHARD_SUFFIXES)):
        # Canonical JSON key order is lexical, which equals suffix order here.
        _fail(
            "manifest_json_invalid",
            "selection quota_deficits must contain the exact ten suffixes",
        )
    deficits = tuple(
        (suffix, raw_deficits[suffix]) for suffix in M4_SHARD_SUFFIXES
    )
    selection = CohortSelection(
        events=events,
        quota_deficits=deficits,
        redistributed_count=payload["redistributed_count"],
        fallback_used=payload["fallback_used"],
    )
    if payload != _selection_payload(selection):
        _fail(
            "selection_tampered",
            "serialized selection summary differs from cohort membership",
        )
    return selection


def _strict_json_loads(text: str) -> Any:
    if not isinstance(text, str):
        _fail("manifest_json_invalid", "manifest JSON must be text")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CohortInvariantError(
                    "manifest_json_duplicate_key",
                    f"duplicate JSON object key {key!r}",
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        raise CohortInvariantError(
            "manifest_json_invalid",
            f"non-finite JSON constant {value!r} is forbidden",
        )

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise CohortInvariantError(
            "manifest_json_invalid",
            "manifest is not valid JSON",
        ) from exc


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CohortInvariantError(
            "canonical_json_invalid",
            "value is not canonical JSON data",
        ) from exc


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    context: str,
) -> None:
    if not isinstance(payload, Mapping):
        _fail("manifest_json_invalid", f"{context} must be an object")
    actual = set(payload)
    if actual != set(expected):
        _fail(
            "manifest_json_invalid",
            f"{context} field set differs from schema",
        )


def _lossless_float32_to_int32(
    array: np.ndarray,
    name: str,
) -> np.ndarray:
    values = array.astype(np.float64)
    if (
        not np.all(np.isfinite(values))
        or np.any(values != np.trunc(values))
        or np.any(values < -(2**31))
        or np.any(values >= 2**31)
    ):
        _fail(
            "audit_nonintegral_encoding",
            f"{name} must contain finite signed-int32 encodings",
        )
    normalized = array.astype(np.int32)
    if not np.array_equal(normalized.astype(np.float32), array):
        _fail(
            "audit_nonintegral_encoding",
            f"{name} failed the exact float32/int32 roundtrip",
        )
    return normalized


def _lossless_int64_to_int32(
    array: np.ndarray,
    name: str,
) -> np.ndarray:
    if np.any(array < -(2**31)) or np.any(array >= 2**31):
        _fail(
            "audit_int32_range",
            f"{name} must fit signed int32 losslessly",
        )
    normalized = array.astype(np.int32)
    if not np.array_equal(normalized.astype(np.int64), array):
        _fail(
            "audit_int32_range",
            f"{name} failed the exact int64/int32 roundtrip",
        )
    return normalized


def _binary_int64_to_bool(
    array: np.ndarray,
    name: str,
) -> np.ndarray:
    if np.any((array != 0) & (array != 1)):
        _fail(
            "audit_nonbinary_encoding",
            f"{name} must contain only binary int64 0/1 encodings",
        )
    return array.astype(np.bool_)


def _sdc_int64_to_bool(
    array: np.ndarray,
    retained: np.ndarray,
) -> np.ndarray:
    if np.any((array != -1) & (array != 0) & (array != 1)):
        _fail(
            "audit_is_sdc_encoding",
            "state/is_sdc must contain only int64 -1/0/1 encodings",
        )
    if np.any((array == -1) & retained):
        _fail(
            "active_sdc_padding",
            "a retained object slot may not use the -1 SDC padding sentinel",
        )
    semantic = array == 1
    if np.any(semantic & ~retained):
        _fail(
            "inactive_sdc_marker",
            "an SDC marker may not refer to a never-valid object slot",
        )
    return semantic


def _validate_shard_suffix(shard_suffix: str) -> None:
    if shard_suffix not in _SHARD_ORDER:
        _fail(
            "shard_suffix_invalid",
            "shard suffix must be one of exact M4 suffixes 00000 through 00009",
        )


def _strict_identity_bytes(native_scenario_id: str) -> bytes:
    if not isinstance(native_scenario_id, str) or not native_scenario_id:
        _fail(
            "native_scenario_id_invalid",
            "native scenario identity must be a non-empty string",
        )
    try:
        return native_scenario_id.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise CohortInvariantError(
            "native_scenario_id_invalid",
            "native scenario identity must be strict UTF-8",
        ) from exc


def _validate_uint64(value: Any, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 0 <= int(value) <= 2**64 - 1
    ):
        _fail(f"{name}_invalid", f"{name} must be an unsigned 64-bit integer")
    return int(value)


def _validate_nonnegative_int(value: Any, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        _fail(f"{name}_invalid", f"{name} must be a non-negative integer")
    return int(value)


def _validate_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        _fail(f"{name}_invalid", f"{name} must be lowercase SHA-256 hex")


def _fail(code: str, message: str) -> NoReturn:
    raise CohortInvariantError(code, message)


__all__ = [
    "CohortInvariantError",
    "CohortSelection",
    "CohortSelectionError",
    "M4_COHORT_DOMAIN",
    "M4_COHORT_TARGET",
    "M4_FALLBACK_FLOOR",
    "M4_IDM_SUBSET_DOMAIN",
    "M4_IDM_SUBSET_FLOOR",
    "M4_IDM_SUBSET_TARGET",
    "M4_INITIAL_QUOTAS",
    "M4_MANIFEST_SCHEMA_VERSION",
    "M4_REDISTRIBUTION_DOMAIN",
    "M4_SELECTOR_CONFIG_FINGERPRINT",
    "M4_SELECTOR_VERSION",
    "M4_SHARD_SUFFIXES",
    "M4_TFRECORD_SUFFIXES",
    "M4_VMAP_DOMAIN",
    "M4_VMAP_PAIR_SIZE",
    "SOURCE_REJECTION_CODES",
    "ScanEvent",
    "ShardScanCounts",
    "WaymaxCohortManifest",
    "rank_record",
    "ranking_message",
    "select_cohort",
    "select_idm_subset",
    "select_vmap_pair",
    "selector_config_fingerprint",
    "selector_config_payload",
    "source_rejection_code",
]
