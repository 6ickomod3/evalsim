"""Immutable, mode-bound local evidence stores for EvalSim M6.

M6 does not reuse or reinterpret the accepted M5 store.  Every run is created once
under ``outputs/m6/<safe-run-name>`` and is bound to one explicit command mode:

``eligibility_only``
    The complete 128-member source-only ledger and receipt.  No outcomes.
``compute_pilot``
    The source-only ledger plus one bounded, outcome-suppressed timing/RSS row.
``official``
    The complete real-WOMD local evidence catalog with at least ten primary members.
``data_free``
    The complete synthetic command profile with exactly ten primary members.

The writer is an in-process capability returned only by :meth:`M6ResultStore.create`.
It cannot be reconstructed from disk.  Its one-way lifecycle is:

``ABSENT -> PENDING -> COMMITTED -> TERMINAL_SUCCESS``

or ``ABSENT/PENDING/COMMITTED -> TERMINAL_FAILURE``.  Every mutation uses exclusive,
no-follow creation; every read rechecks containment, ownership, mode, node type, link
count, and file identity.  Terminal success is the final reconciled filesystem write:
all independent verification happens before that marker is created.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
from types import MappingProxyType
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from evalsim.sources.waymax import WAYMAX_COMMIT
from evalsim.perturb.m6 import (
    M6_INTERVENTION_VERSION,
    PRIMARY_BRAKE_MAGNITUDE_MPS2,
    SECONDARY_BRAKE_MAGNITUDE_MPS2,
    longitudinal_brake_pulse_spec,
)
from evalsim.evaluation.m6_waymax_metrics import (
    M6WaymaxIssuedScalarTable,
    M6WaymaxMatrixResult,
    M6WaymaxSceneScalar,
    analyze_m6_waymax_cells,
    build_m6_waymax_data_free_determinism_table,
    m6_waymax_measure_contract,
    parse_m6_waymax_scene_scalar_table,
    reconstruct_m6_waymax_stored_cells,
    verify_m6_waymax_stored_selection,
)
from evalsim.simulators.waymax_m6 import (
    M6_WAYMAX_BUNDLES,
    M6_WAYMAX_FLOAT_ATOL,
    M6_WAYMAX_FLOAT_RTOL,
    M6_WAYMAX_LOGGED_WORLD,
    M6_WAYMAX_MAX_SCENES,
    M6_WAYMAX_PRIVILEGED_IDM,
    M6_WAYMAX_TRANSITIONS,
    M6_WAYMAX_YAW_ATOL,
    M6WaymaxSelection,
    m6_waymax_rank_sha256,
)
from evalsim.stats.m6 import (
    M6_ADJUSTED_REWEIGHTING_LEVEL,
    M6_BASE_SEED,
    M6_POINTWISE_REWEIGHTING_LEVEL,
    M6_PRIMARY_RESAMPLES,
    M6_REWEIGHTING_INTERPRETATION,
    M6_STATISTICS_SCHEMA_VERSION,
    M6PrimaryCellInput,
    M6PrimaryCellResult,
    M6PrimaryCellSpec,
    M6SceneEffect,
    analyze_m6_primary_matrix,
)


M6_RESULT_STORE_SCHEMA_VERSION = "m6-result-store-3.0.0"
M6_ELIGIBILITY_RECEIPT_SCHEMA_VERSION = "m6-eligibility-receipt-3.0.0"
M6_WAYMAX_SELECTION_RECEIPT_SCHEMA_VERSION = (
    "m6-waymax-selection-receipt-1.0.0"
)
M6_DETERMINISM_RECEIPT_SCHEMA_VERSION = "m6-determinism-receipt-3.0.0"
M6_PROMOTED_AGGREGATE_SCHEMA_VERSION = "m6-promoted-aggregate-2.0.0"
M6_TYPED_PROVENANCE_SCHEMA_VERSION = "m6-typed-provenance-1.0.0"
M6_EXECUTION_SCHEMA_VERSION = "m6-execution-evidence-1.0.0"
M6_CLAIM_LIMITATIONS_SCHEMA_VERSION = "m6-claim-limitations-1.0.0"
M6_CONFIG_VERSION = "m6-counterfactual-config-1.0.0"
M6_PLAN_VERSION = "m6-counterfactual-reactivity-1.0.0"

ELIGIBILITY_ONLY_MODE = "eligibility_only"
COMPUTE_PILOT_MODE = "compute_pilot"
OFFICIAL_MODE = "official"
DATA_FREE_MODE = "data_free"
M6_RUN_MODES = (
    ELIGIBILITY_ONLY_MODE,
    COMPUTE_PILOT_MODE,
    OFFICIAL_MODE,
    DATA_FREE_MODE,
)

ELIGIBILITY_LEDGER = "eligibility-ledger"
COMPUTE_PILOT_SUMMARY = "compute-pilot-summary"
PRIMARY_SCENE_SCALARS = "primary-paired-scene-scalars"
PRIMARY_MATRIX = "primary-matrix"
PRIMARY_REPEAT_SCENE_SCALARS = "primary-repeat-paired-scene-scalars"
PRIMARY_REPEAT_MATRIX = "primary-repeat-matrix"
SECONDARY_SCENE_SCALARS = "secondary-b4-paired-scene-scalars"
SECONDARY_MATRIX = "secondary-b4-matrix"
NEGATIVE_TIMING_OBSERVATIONS = "negative-timing-observations"
NEGATIVE_TIMING_GATES = "negative-timing-gates"
WAYMAX_ACCOUNTING = "waymax-accounting"
WAYMAX_QUALIFICATION = "waymax-qualification"
WAYMAX_SCENE_SCALARS = "waymax-scene-scalars"
WAYMAX_FIELD_COMPARISONS = "waymax-field-comparisons"
WAYMAX_DETERMINISM = "waymax-determinism"
TYPED_PROVENANCE = "typed-provenance"
EXECUTION_SUMMARY = "execution-summary"
STAGE_TIMINGS = "stage-timings"
REVIEW_DECISIONS = "review-decisions"

ELIGIBILITY_RECEIPT_PATH = "eligibility-receipt.json"
WAYMAX_SELECTION_RECEIPT_PATH = "waymax-selection-receipt.json"
DETERMINISM_RECEIPT_PATH = "determinism-receipt.json"
CLAIM_LIMITATIONS_PATH = "claim-limitations.json"
MANIFEST_PATH = "result-manifest.json"
PENDING_MARKER = "PENDING"
COMMITTED_MARKER = "COMMITTED"
TERMINAL_SUCCESS_MARKER = "TERMINAL_SUCCESS"
TERMINAL_FAILURE_MARKER = "TERMINAL_FAILURE"

_DATASET_PATHS: Mapping[str, str] = MappingProxyType(
    {
        ELIGIBILITY_LEDGER: "eligibility-ledger.parquet",
        COMPUTE_PILOT_SUMMARY: "compute-pilot-summary.parquet",
        PRIMARY_SCENE_SCALARS: "primary-paired-scene-scalars.parquet",
        PRIMARY_MATRIX: "primary-matrix.parquet",
        PRIMARY_REPEAT_SCENE_SCALARS: (
            "primary-repeat-paired-scene-scalars.parquet"
        ),
        PRIMARY_REPEAT_MATRIX: "primary-repeat-matrix.parquet",
        SECONDARY_SCENE_SCALARS: "secondary-b4-paired-scene-scalars.parquet",
        SECONDARY_MATRIX: "secondary-b4-matrix.parquet",
        NEGATIVE_TIMING_OBSERVATIONS: "negative-timing-observations.parquet",
        NEGATIVE_TIMING_GATES: "negative-timing-gates.parquet",
        WAYMAX_ACCOUNTING: "waymax-accounting.parquet",
        WAYMAX_QUALIFICATION: "waymax-qualification.parquet",
        WAYMAX_SCENE_SCALARS: "waymax-scene-scalars.parquet",
        WAYMAX_FIELD_COMPARISONS: "waymax-field-comparisons.parquet",
        WAYMAX_DETERMINISM: "waymax-determinism.parquet",
        TYPED_PROVENANCE: "typed-provenance.parquet",
        EXECUTION_SUMMARY: "execution-summary.parquet",
        STAGE_TIMINGS: "stage-timings.parquet",
        REVIEW_DECISIONS: "review-decisions.parquet",
    }
)

M6_PRIMARY_REJECTION_REASONS = (
    "insufficient_future_horizon",
    "ego_invalid_in_window",
    "ego_speed_below_5_mps",
    "source_ego_path_degenerate",
    "zero_dose_reconstruction_mismatch",
    "primary_ego_plan_infeasible",
    "no_stable_aligned_follower",
    "current_ego_follower_overlap",
)
M6_PRIMARY_POLICY_ROLES = (
    ("log_replay", "privileged"),
    ("constant_velocity", "history_only"),
    ("idm", "history_only"),
)
M6_PRIMARY_METRICS = (
    ("additional_target_braking_impulse_mps", "1.0.0", "m/s"),
    ("response_timeliness_s", "1.0.0", "s"),
    ("minimum_longitudinal_bumper_gap_change_m", "1.0.0", "m"),
    ("target_progress_loss_m", "1.0.0", "m"),
)
M6_PRIMARY_CELL_DOMAIN = tuple(
    (policy, access_role, metric, version)
    for policy, access_role in M6_PRIMARY_POLICY_ROLES
    for metric, version, _unit in M6_PRIMARY_METRICS
)
M6_NEGATIVE_TIMING_GATE_DOMAIN = (
    "log_replay_world_tensor_equality",
    "constant_velocity_world_tensor_equality",
    "sham_legacy_equality",
    "synchronous_response_floor",
    "primary_plan_feasibility",
    "nested_dose_monotonicity",
)
M6_STAGE_DOMAIN = (
    "eligibility",
    "numpy_rollouts",
    "paired_metrics",
    "statistics",
    "waymax",
    "verification",
)
M6_REVIEW_ROLE_DOMAIN = (
    "architecture",
    "methods_statistics",
    "privacy_claim",
)

M6_WAYMAX_REJECTION_REASONS = (
    "waymax_cadence_mismatch",
    "waymax_target_control_incomplete",
    "waymax_target_overlap_excluded",
)
M6_WAYMAX_QUALIFICATION_REJECTION_REASONS = (
    "source_cadence_not_100ms",
    "target_not_requested_all_transitions",
    "target_initialized_overlap_excluded",
)
M6_WAYMAX_QUALIFICATION_TO_ACCOUNTING_REASON: Mapping[str, str] = (
    MappingProxyType(
        {
            "source_cadence_not_100ms": "waymax_cadence_mismatch",
            "target_not_requested_all_transitions": (
                "waymax_target_control_incomplete"
            ),
            "target_initialized_overlap_excluded": (
                "waymax_target_overlap_excluded"
            ),
        }
    )
)
_M6_WAYMAX_MEASURE_CONTRACT = m6_waymax_measure_contract()
M6_WAYMAX_IDENTITY_CONFIGURATION_FINGERPRINT = str(
    _M6_WAYMAX_MEASURE_CONTRACT["identity_configuration_fingerprint"]
)
M6_WAYMAX_PRIMARY_B2_CONFIGURATION_FINGERPRINT = str(
    _M6_WAYMAX_MEASURE_CONTRACT["primary_b2_configuration_fingerprint"]
)
M6_DATA_FREE_WAYMAX_PRIMARY_DOMAIN_SHA256 = hashlib.sha256(
    b"evalsim-m6-data-free-waymax-primary-domain-not-applicable-v1"
).hexdigest()
M6_DATA_FREE_WAYMAX_SELECTION_BINDING_SHA256 = hashlib.sha256(
    b"evalsim-m6-data-free-waymax-selection-not-applicable-v1"
).hexdigest()
M6_WAYMAX_CONDITIONS = ("identity", "primary_brake")
M6_WAYMAX_MAX_SELECTED = M6_WAYMAX_MAX_SCENES
M6_WAYMAX_DETERMINISM_ROW_COUNT = (
    M6_WAYMAX_MAX_SELECTED
    * len(M6_WAYMAX_BUNDLES)
    * len(M6_WAYMAX_CONDITIONS)
)
M6_WAYMAX_RANK_DOMAIN = "evalsim-m6-waymax-reactivity-v1"
M6_WAYMAX_COMPARISON_FIELDS = (
    "agent_identity",
    "timestamps",
    "validity",
    "actor_mask",
    "lifecycle_category",
    "x",
    "y",
    "vx",
    "vy",
    "heading",
)
M6_WAYMAX_EXACT_FIELDS = frozenset(
    {
        "agent_identity",
        "timestamps",
        "validity",
        "actor_mask",
        "lifecycle_category",
    }
)
M6_WAYMAX_CONTROL_COUNTS = (
    "target_requested_control",
    "target_effective_control",
    "target_logged_lifecycle_fallback",
    "target_initialized_overlap_exclusion",
)
M6_PRIMARY_INTERVENTION = longitudinal_brake_pulse_spec(
    PRIMARY_BRAKE_MAGNITUDE_MPS2
)
M6_SECONDARY_INTERVENTION = longitudinal_brake_pulse_spec(
    SECONDARY_BRAKE_MAGNITUDE_MPS2
)
M6_PRIMARY_INTERVENTION_FINGERPRINT = (
    M6_PRIMARY_INTERVENTION.configuration_fingerprint
)
M6_SECONDARY_INTERVENTION_FINGERPRINT = (
    M6_SECONDARY_INTERVENTION.configuration_fingerprint
)
assert M6_PRIMARY_INTERVENTION_FINGERPRINT is not None
assert M6_SECONDARY_INTERVENTION_FINGERPRINT is not None

M6_NEGATIVE_TIMING_OBSERVATION_POLICIES: Mapping[
    str, tuple[str | None, ...]
] = MappingProxyType(
    {
        "log_replay_world_tensor_equality": ("log_replay",),
        "constant_velocity_world_tensor_equality": ("constant_velocity",),
        "sham_legacy_equality": tuple(
            policy for policy, _role in M6_PRIMARY_POLICY_ROLES
        ),
        "synchronous_response_floor": tuple(
            policy for policy, _role in M6_PRIMARY_POLICY_ROLES
        ),
        "primary_plan_feasibility": (None,),
        "nested_dose_monotonicity": (None,),
    }
)
M6_WAYMAX_ROW_DOMAIN = (
    (
        ("scope", "qualified_count", None, None, None),
        ("scope", "selected_count", None, None, None),
        ("scope", "transition_count", None, None, None),
    )
    + tuple(
        ("selection_rejection", reason, None, None, None)
        for reason in M6_WAYMAX_REJECTION_REASONS
    )
    + tuple(
        ("field_comparison", field_name, bundle, condition, None)
        for bundle in M6_WAYMAX_BUNDLES
        for condition in M6_WAYMAX_CONDITIONS
        for field_name in M6_WAYMAX_COMPARISON_FIELDS
    )
    + tuple(
        (
            "control_partition",
            name,
            M6_WAYMAX_PRIVILEGED_IDM,
            None,
            None,
        )
        for name in M6_WAYMAX_CONTROL_COUNTS
    )
    + tuple(
        ("secondary_cell", "paired_effect", bundle, None, metric)
        for bundle in M6_WAYMAX_BUNDLES
        for metric, _version, _unit in M6_PRIMARY_METRICS
    )
)

M6_PROMOTED_PRIMARY_FIELDS = (
    "metric_name",
    "metric_version",
    "unit",
    "policy_name",
    "policy_access_role",
    "pair_n",
    "thresholded_nonzero_n",
    "responder_n",
    "censor_n",
    "arithmetic_mean",
    "median",
    "pointwise_band",
    "adjusted_band",
    "status",
    "suppression_reason",
    "source_pairing_complete",
    "directional_language_allowed",
)
M6_PROMOTED_TOP_LEVEL_DOMAINS = (
    "provenance_labels",
    "eligibility",
    "primary_matrix",
    "negative_control_and_timing_gates",
    "waymax_scope",
    "execution",
    "claim_and_limitations",
)

M6_ACCEPTED_BOUNDED_CLAIM = (
    "Implemented and evaluated a typed paired ego-braking intervention on a fixed, "
    "source-eligible subset of the accepted local WOMD cohort. Audited history-only "
    "world policies received fixed observed history/static context at initialization "
    "and only realized current state thereafter; privileged log-replay, ego-plan, "
    "and Waymax references remained explicitly separated. Independent reactivity "
    "and response-cost measures detected nonresponse and controlled synthetic "
    "overreaction without a simulator winner or real-world causal/safety claim."
)
M6_FIXED_LIMITATIONS = (
    "No real-world, human-driver, fleet, or WOMD-population causal effect is "
    "estimated.",
    "No collision-prevention, safety, route-compliance, lane-following, "
    "offroad, traffic-rule, or driving-quality claim is made.",
    "Waymax IDM is a privileged logged-trajectory waypoint-following "
    "reference, not causal policy or independent ground truth.",
    "EvalSim IDM and Waymax IDM are not claimed to be numerical twins.",
    "No simulator winner, replacement decision, composite reactivity score, "
    "or production-readiness claim is made.",
    "Results are conditional on the accepted ten-shard cohort and are not "
    "representative beyond it.",
    "M6 response measures remain subject to M7 construct-validity evaluation.",
    "World-agent lifecycle births, deaths, and re-entries remain source scheduled.",
    "The aligned geometric follower is not a lane follower.",
    "Relational gap changes combine ego and world motion and alone do not "
    "establish world reactivity.",
    "A simulated braking response is not evidence of better driving quality.",
    "Pinned Waymax actor logic assumes fixed 0.1 second cadence on its bounded subset.",
)
_M6_DATA_FREE_BOUNDED_CLAIM = (
    "Executed the fixed M6 synthetic acceptance fixtures for the typed paired "
    "ego-braking implementation. This data-free evidence contains no WOMD or "
    "real-scene Waymax outcome and supports no real-data reactivity claim."
)
_M6_DATA_FREE_LIMITATIONS = (
    "This is synthetic data-free software evidence only.",
    "No WOMD record, accepted-M4 cohort member, or real-scene Waymax outcome was used.",
    "No real-world causal, safety, simulator-ranking, or production-readiness claim "
    "is made.",
)

_RUN_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_REASON_CODE = re.compile(r"[a-z][a-z0-9_]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}")
_SAFE_VERSION = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}")
_ALLOWED_PRIMARY_STATUSES = frozenset(
    {"event_sparse", "small_n", "descriptive", "direction_supported"}
)
_CREATE_SENTINEL = object()
_AGGREGATE_SENTINEL = object()
_OBSERVED_PREFLIGHT_SENTINEL = object()
_TERMINAL_CAPABILITY_SENTINEL = object()
M6_PREFLIGHT_CHECK_DOMAIN = (
    "accepted_m4_verified",
    "git_live_main_verified",
    "output_layout_verified",
    "result_store_verified",
    "shards_verified",
    "source_runtime_verified",
    "terminal_capture_verified",
)


@dataclass(frozen=True, slots=True)
class M6ResultProfile:
    """Exact mode, population, and publication boundary."""

    mode: str
    population_size: int
    data_free: bool

    def __post_init__(self) -> None:
        if self.mode not in M6_RUN_MODES:
            raise ValueError("unregistered M6 mode")
        if type(self.population_size) is not int or self.population_size < 1:
            raise ValueError("population_size must be an exact positive int")
        if type(self.data_free) is not bool:
            raise TypeError("data_free must be an exact bool")
        expected_population = 10 if self.mode == DATA_FREE_MODE else 128
        if self.population_size != expected_population:
            raise ValueError("M6 mode has the wrong fixed population")
        if self.data_free != (self.mode == DATA_FREE_MODE):
            raise ValueError("only data_free mode may use synthetic evidence")

    @property
    def complete_results(self) -> bool:
        return self.mode in {OFFICIAL_MODE, DATA_FREE_MODE}


ELIGIBILITY_ONLY_M6_PROFILE = M6ResultProfile(
    ELIGIBILITY_ONLY_MODE, 128, False
)
COMPUTE_PILOT_M6_PROFILE = M6ResultProfile(COMPUTE_PILOT_MODE, 128, False)
OFFICIAL_M6_PROFILE = M6ResultProfile(OFFICIAL_MODE, 128, False)
DATA_FREE_M6_TEST_PROFILE = M6ResultProfile(DATA_FREE_MODE, 10, True)
_PROFILES: Mapping[str, M6ResultProfile] = MappingProxyType(
    {
        profile.mode: profile
        for profile in (
            ELIGIBILITY_ONLY_M6_PROFILE,
            COMPUTE_PILOT_M6_PROFILE,
            OFFICIAL_M6_PROFILE,
            DATA_FREE_M6_TEST_PROFILE,
        )
    }
)


def _schema(fields: Sequence[pa.Field], name: str) -> pa.Schema:
    return pa.schema(
        fields,
        metadata={
            b"evalsim.dataset": name.encode("ascii"),
            b"evalsim.schema_version": M6_RESULT_STORE_SCHEMA_VERSION.encode(
                "ascii"
            ),
        },
    )


ELIGIBILITY_LEDGER_SCHEMA = _schema(
    (
        pa.field("cohort_index", pa.int32(), nullable=False),
        pa.field("primary_eligible", pa.bool_(), nullable=False),
        pa.field("rejection_reason", pa.string(), nullable=True),
        pa.field("secondary_b4_feasible", pa.bool_(), nullable=True),
    ),
    ELIGIBILITY_LEDGER,
)
COMPUTE_PILOT_SUMMARY_SCHEMA = _schema(
    (
        pa.field("pilot_scene_n", pa.int32(), nullable=False),
        pa.field("total_wall_ms", pa.int64(), nullable=False),
        pa.field("max_scene_ms", pa.int64(), nullable=False),
        pa.field("decode_ms", pa.int64(), nullable=False),
        pa.field("numpy_ms", pa.int64(), nullable=False),
        pa.field("waymax_ms", pa.int64(), nullable=False),
        pa.field("verification_ms", pa.int64(), nullable=False),
        pa.field("fresh_worker_peak_rss_bytes", pa.int64(), nullable=False),
        pa.field("passed", pa.bool_(), nullable=False),
    ),
    COMPUTE_PILOT_SUMMARY,
)
_SCENE_FIELDS = (
    pa.field("cohort_index", pa.int32(), nullable=False),
    pa.field("policy_name", pa.string(), nullable=False),
    pa.field("policy_access_role", pa.string(), nullable=False),
    pa.field("metric_name", pa.string(), nullable=False),
    pa.field("metric_version", pa.string(), nullable=False),
    pa.field("unit", pa.string(), nullable=False),
    pa.field("value", pa.float64(), nullable=False),
    pa.field("responded", pa.bool_(), nullable=True),
    pa.field("responder_latency_s", pa.float64(), nullable=True),
    pa.field("source_pairing_complete", pa.bool_(), nullable=False),
    pa.field(
        "intervention_config_fingerprint",
        pa.string(),
        nullable=False,
    ),
)
PRIMARY_SCENE_SCALARS_SCHEMA = _schema(_SCENE_FIELDS, PRIMARY_SCENE_SCALARS)
PRIMARY_REPEAT_SCENE_SCALARS_SCHEMA = _schema(
    _SCENE_FIELDS,
    PRIMARY_REPEAT_SCENE_SCALARS,
)
SECONDARY_SCENE_SCALARS_SCHEMA = _schema(
    _SCENE_FIELDS,
    SECONDARY_SCENE_SCALARS,
)
PRIMARY_MATRIX_SCHEMA = _schema(
    (
        pa.field("metric_name", pa.string(), nullable=False),
        pa.field("metric_version", pa.string(), nullable=False),
        pa.field("unit", pa.string(), nullable=False),
        pa.field("policy_name", pa.string(), nullable=False),
        pa.field("policy_access_role", pa.string(), nullable=False),
        pa.field(
            "intervention_config_fingerprint",
            pa.string(),
            nullable=False,
        ),
        pa.field("pair_n", pa.int32(), nullable=False),
        pa.field("thresholded_nonzero_n", pa.int32(), nullable=False),
        pa.field("responder_n", pa.int32(), nullable=True),
        pa.field("censor_n", pa.int32(), nullable=True),
        pa.field("arithmetic_mean", pa.float64(), nullable=False),
        pa.field("median", pa.float64(), nullable=False),
        pa.field("pointwise_level", pa.float64(), nullable=False),
        pa.field("pointwise_lower", pa.float64(), nullable=False),
        pa.field("pointwise_upper", pa.float64(), nullable=False),
        pa.field("adjusted_level", pa.float64(), nullable=False),
        pa.field("adjusted_lower", pa.float64(), nullable=False),
        pa.field("adjusted_upper", pa.float64(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("suppression_reason", pa.string(), nullable=True),
        pa.field("source_pairing_complete", pa.bool_(), nullable=False),
        pa.field(
            "directional_language_allowed",
            pa.bool_(),
            nullable=False,
        ),
        pa.field("directional_effect_sign", pa.string(), nullable=True),
        pa.field(
            "conditional_latency_status",
            pa.string(),
            nullable=True,
        ),
        pa.field(
            "conditional_latency_suppression_reason",
            pa.string(),
            nullable=True,
        ),
        pa.field(
            "conditional_latency_mean_s",
            pa.float64(),
            nullable=True,
        ),
        pa.field(
            "conditional_latency_median_s",
            pa.float64(),
            nullable=True,
        ),
        pa.field("resampling_key_json", pa.string(), nullable=False),
        pa.field("resampling_sha256", pa.string(), nullable=False),
        pa.field(
            "resampling_digest_words",
            pa.list_(
                pa.field("element", pa.uint32(), nullable=False),
                8,
            ),
            nullable=False,
        ),
        pa.field("resamples", pa.int32(), nullable=False),
        pa.field("base_seed", pa.uint32(), nullable=False),
        pa.field("rng", pa.string(), nullable=False),
        pa.field("index_dtype", pa.string(), nullable=False),
        pa.field("quantile_method", pa.string(), nullable=False),
        pa.field("interpretation", pa.string(), nullable=False),
        pa.field(
            "statistics_schema_version",
            pa.string(),
            nullable=False,
        ),
    ),
    PRIMARY_MATRIX,
)
PRIMARY_REPEAT_MATRIX_SCHEMA = _schema(
    tuple(PRIMARY_MATRIX_SCHEMA),
    PRIMARY_REPEAT_MATRIX,
)
SECONDARY_MATRIX_SCHEMA = _schema(
    (
        pa.field("metric_name", pa.string(), nullable=False),
        pa.field("metric_version", pa.string(), nullable=False),
        pa.field("unit", pa.string(), nullable=False),
        pa.field("policy_name", pa.string(), nullable=False),
        pa.field("policy_access_role", pa.string(), nullable=False),
        pa.field(
            "intervention_config_fingerprint",
            pa.string(),
            nullable=False,
        ),
        pa.field("pair_n", pa.int32(), nullable=False),
        pa.field("thresholded_nonzero_n", pa.int32(), nullable=False),
        pa.field("responder_n", pa.int32(), nullable=True),
        pa.field("censor_n", pa.int32(), nullable=True),
        pa.field("arithmetic_mean", pa.float64(), nullable=True),
        pa.field("median", pa.float64(), nullable=True),
        pa.field("status", pa.string(), nullable=False),
        pa.field("suppression_reason", pa.string(), nullable=True),
        pa.field("source_pairing_complete", pa.bool_(), nullable=False),
    ),
    SECONDARY_MATRIX,
)
NEGATIVE_TIMING_OBSERVATIONS_SCHEMA = _schema(
    (
        pa.field("gate_name", pa.string(), nullable=False),
        pa.field("cohort_index", pa.int32(), nullable=False),
        pa.field("policy_name", pa.string(), nullable=True),
        pa.field("assessed_n", pa.int32(), nullable=False),
        pa.field("violation_n", pa.int32(), nullable=False),
        pa.field("observation_sha256", pa.string(), nullable=False),
    ),
    NEGATIVE_TIMING_OBSERVATIONS,
)
NEGATIVE_TIMING_GATES_SCHEMA = _schema(
    (
        pa.field("gate_name", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("assessed_n", pa.int32(), nullable=False),
        pa.field("passed_n", pa.int32(), nullable=False),
        pa.field("violation_n", pa.int32(), nullable=False),
        pa.field("local_evidence_sha256", pa.string(), nullable=False),
    ),
    NEGATIVE_TIMING_GATES,
)
WAYMAX_ACCOUNTING_SCHEMA = _schema(
    (
        pa.field("record_type", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("bundle", pa.string(), nullable=True),
        pa.field("condition", pa.string(), nullable=True),
        pa.field("metric_name", pa.string(), nullable=True),
        pa.field("metric_version", pa.string(), nullable=True),
        pa.field("unit", pa.string(), nullable=True),
        pa.field("comparison_kind", pa.string(), nullable=True),
        pa.field("count", pa.int64(), nullable=True),
        pa.field("opportunity_n", pa.int64(), nullable=True),
        pa.field("denominator", pa.int64(), nullable=True),
        pa.field("max_abs_error", pa.float64(), nullable=True),
        pa.field("tolerance_failures", pa.int64(), nullable=True),
        pa.field("binary_mismatches", pa.int64(), nullable=True),
        pa.field("pair_n", pa.int32(), nullable=True),
        pa.field("thresholded_nonzero_n", pa.int32(), nullable=True),
        pa.field("responder_n", pa.int32(), nullable=True),
        pa.field("censor_n", pa.int32(), nullable=True),
        pa.field("arithmetic_mean", pa.float64(), nullable=True),
        pa.field("median", pa.float64(), nullable=True),
        pa.field("pointwise_level", pa.float64(), nullable=True),
        pa.field("pointwise_lower", pa.float64(), nullable=True),
        pa.field("pointwise_upper", pa.float64(), nullable=True),
        pa.field("suppression_reason", pa.string(), nullable=True),
        pa.field("source_pairing_complete", pa.bool_(), nullable=True),
        pa.field(
            "directional_language_allowed",
            pa.bool_(),
            nullable=True,
        ),
        pa.field("status", pa.string(), nullable=False),
    ),
    WAYMAX_ACCOUNTING,
)
WAYMAX_QUALIFICATION_SCHEMA = _schema(
    (
        pa.field("cohort_index", pa.int32(), nullable=False),
        pa.field("assessment_status", pa.string(), nullable=False),
        pa.field("rejection_reason", pa.string(), nullable=True),
        pa.field("rank_sha256", pa.string(), nullable=True),
        pa.field("source_binding_sha256", pa.string(), nullable=True),
        pa.field("primary_entry_sha256", pa.string(), nullable=True),
        pa.field("qualification_binding_sha256", pa.string(), nullable=True),
        pa.field("selected", pa.bool_(), nullable=False),
        pa.field("selection_position", pa.int32(), nullable=True),
    ),
    WAYMAX_QUALIFICATION,
)
WAYMAX_SCENE_SCALARS_SCHEMA = _schema(
    (
        pa.field("selection_position", pa.int32(), nullable=False),
        pa.field("cohort_index", pa.int32(), nullable=True),
        pa.field("qualification_binding_sha256", pa.string(), nullable=True),
        pa.field("primary_domain_sha256", pa.string(), nullable=False),
        pa.field("selection_binding_sha256", pa.string(), nullable=False),
        pa.field("selection_supported", pa.bool_(), nullable=False),
        pa.field("selection_member_count", pa.int32(), nullable=False),
        pa.field(
            "identity_configuration_fingerprint",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "primary_b2_configuration_fingerprint",
            pa.string(),
            nullable=False,
        ),
        pa.field("bundle", pa.string(), nullable=False),
        pa.field("metric_name", pa.string(), nullable=False),
        pa.field("metric_version", pa.string(), nullable=False),
        pa.field("value_unit", pa.string(), nullable=False),
        pa.field("value", pa.float64(), nullable=True),
        pa.field("responded", pa.bool_(), nullable=True),
        pa.field("responder_latency_s", pa.float64(), nullable=True),
        pa.field("source_pairing_complete", pa.bool_(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
    ),
    WAYMAX_SCENE_SCALARS,
)
WAYMAX_FIELD_COMPARISONS_SCHEMA = _schema(
    (
        pa.field("selection_position", pa.int32(), nullable=False),
        pa.field("bundle", pa.string(), nullable=False),
        pa.field("condition", pa.string(), nullable=False),
        pa.field("field_name", pa.string(), nullable=False),
        pa.field("cohort_index", pa.int32(), nullable=True),
        pa.field("qualification_binding_sha256", pa.string(), nullable=True),
        pa.field("comparison_kind", pa.string(), nullable=False),
        pa.field("denominator", pa.int64(), nullable=True),
        pa.field("max_abs_error", pa.float64(), nullable=True),
        pa.field("max_normalized_error", pa.float64(), nullable=True),
        pa.field("tolerance_failures", pa.int64(), nullable=True),
        pa.field("binary_mismatches", pa.int64(), nullable=True),
        pa.field("status", pa.string(), nullable=False),
    ),
    WAYMAX_FIELD_COMPARISONS,
)
WAYMAX_DETERMINISM_SCHEMA = _schema(
    (
        pa.field("selection_position", pa.int32(), nullable=False),
        pa.field("bundle", pa.string(), nullable=False),
        pa.field("condition", pa.string(), nullable=False),
        pa.field("cohort_index", pa.int32(), nullable=True),
        pa.field(
            "qualification_binding_sha256",
            pa.string(),
            nullable=True,
        ),
        pa.field("status", pa.string(), nullable=False),
        pa.field("eager_pass_1_sha256", pa.string(), nullable=True),
        pa.field("eager_pass_2_sha256", pa.string(), nullable=True),
        pa.field("jit_eager_sha256", pa.string(), nullable=True),
        pa.field("jit_compiled_sha256", pa.string(), nullable=True),
    ),
    WAYMAX_DETERMINISM,
)
TYPED_PROVENANCE_SCHEMA = _schema(
    (
        pa.field("plan_version", pa.string(), nullable=False),
        pa.field("config_version", pa.string(), nullable=False),
        pa.field("statistics_schema_version", pa.string(), nullable=False),
        pa.field("population_label", pa.string(), nullable=False),
        pa.field("source_shard_start", pa.string(), nullable=True),
        pa.field("source_shard_end", pa.string(), nullable=True),
        pa.field("approved_git_commit", pa.string(), nullable=False),
        pa.field("git_tree", pa.string(), nullable=False),
        pa.field("executable_source_sha256", pa.string(), nullable=False),
        pa.field("uv_lock_sha256", pa.string(), nullable=False),
        pa.field("runtime_config_sha256", pa.string(), nullable=False),
        pa.field("accepted_m4_manifest_sha256", pa.string(), nullable=True),
        pa.field(
            "accepted_m4_provenance_sha256",
            pa.string(),
            nullable=True,
        ),
        pa.field("python_version", pa.string(), nullable=False),
        pa.field("numpy_version", pa.string(), nullable=False),
        pa.field("pyarrow_version", pa.string(), nullable=False),
        pa.field("jax_version", pa.string(), nullable=True),
        pa.field("jaxlib_version", pa.string(), nullable=True),
        pa.field("tensorflow_version", pa.string(), nullable=True),
        pa.field("waymax_commit", pa.string(), nullable=True),
        pa.field("jax_backend", pa.string(), nullable=True),
        pa.field("jax_device_class", pa.string(), nullable=True),
        pa.field(
            "primary_intervention_fingerprint",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "secondary_intervention_fingerprint",
            pa.string(),
            nullable=False,
        ),
    ),
    TYPED_PROVENANCE,
)
EXECUTION_SUMMARY_SCHEMA = _schema(
    (
        pa.field("deterministic_repeat_status", pa.string(), nullable=False),
        pa.field("waymax_gate_status", pa.string(), nullable=False),
        pa.field(
            "real_reactivity_claim_status",
            pa.string(),
            nullable=False,
        ),
        pa.field("release_gate_status", pa.string(), nullable=False),
        pa.field("fresh_worker_peak_rss_bytes", pa.int64(), nullable=False),
        pa.field("eligibility_rows", pa.int32(), nullable=False),
        pa.field("primary_scene_rows", pa.int32(), nullable=False),
        pa.field("primary_matrix_rows", pa.int32(), nullable=False),
        pa.field("secondary_scene_rows", pa.int32(), nullable=False),
        pa.field("secondary_matrix_rows", pa.int32(), nullable=False),
        pa.field(
            "primary_repeat_scene_rows",
            pa.int32(),
            nullable=False,
        ),
        pa.field(
            "primary_repeat_matrix_rows",
            pa.int32(),
            nullable=False,
        ),
        pa.field(
            "negative_timing_observation_rows",
            pa.int32(),
            nullable=False,
        ),
        pa.field("negative_timing_gate_rows", pa.int32(), nullable=False),
        pa.field("waymax_accounting_rows", pa.int32(), nullable=False),
        pa.field("waymax_qualification_rows", pa.int32(), nullable=False),
        pa.field("waymax_scene_scalar_rows", pa.int32(), nullable=False),
        pa.field(
            "waymax_field_comparison_rows",
            pa.int32(),
            nullable=False,
        ),
        pa.field("waymax_determinism_rows", pa.int32(), nullable=False),
        pa.field("stage_timing_rows", pa.int32(), nullable=False),
        pa.field("review_decision_rows", pa.int32(), nullable=False),
    ),
    EXECUTION_SUMMARY,
)
STAGE_TIMINGS_SCHEMA = _schema(
    (
        pa.field("stage_name", pa.string(), nullable=False),
        pa.field("duration_ms", pa.int64(), nullable=False),
    ),
    STAGE_TIMINGS,
)
REVIEW_DECISIONS_SCHEMA = _schema(
    (
        pa.field("role", pa.string(), nullable=False),
        pa.field("decision", pa.string(), nullable=False),
        pa.field("p1_count", pa.int32(), nullable=False),
        pa.field("p2_count", pa.int32(), nullable=False),
        pa.field("p3_count", pa.int32(), nullable=False),
        pa.field(
            "evidence_catalog_sha256",
            pa.string(),
            nullable=False,
        ),
    ),
    REVIEW_DECISIONS,
)

M6_RESULT_SCHEMAS: Mapping[str, pa.Schema] = MappingProxyType(
    {
        ELIGIBILITY_LEDGER: ELIGIBILITY_LEDGER_SCHEMA,
        COMPUTE_PILOT_SUMMARY: COMPUTE_PILOT_SUMMARY_SCHEMA,
        PRIMARY_SCENE_SCALARS: PRIMARY_SCENE_SCALARS_SCHEMA,
        PRIMARY_MATRIX: PRIMARY_MATRIX_SCHEMA,
        PRIMARY_REPEAT_SCENE_SCALARS: PRIMARY_REPEAT_SCENE_SCALARS_SCHEMA,
        PRIMARY_REPEAT_MATRIX: PRIMARY_REPEAT_MATRIX_SCHEMA,
        SECONDARY_SCENE_SCALARS: SECONDARY_SCENE_SCALARS_SCHEMA,
        SECONDARY_MATRIX: SECONDARY_MATRIX_SCHEMA,
        NEGATIVE_TIMING_OBSERVATIONS: NEGATIVE_TIMING_OBSERVATIONS_SCHEMA,
        NEGATIVE_TIMING_GATES: NEGATIVE_TIMING_GATES_SCHEMA,
        WAYMAX_ACCOUNTING: WAYMAX_ACCOUNTING_SCHEMA,
        WAYMAX_QUALIFICATION: WAYMAX_QUALIFICATION_SCHEMA,
        WAYMAX_SCENE_SCALARS: WAYMAX_SCENE_SCALARS_SCHEMA,
        WAYMAX_FIELD_COMPARISONS: WAYMAX_FIELD_COMPARISONS_SCHEMA,
        WAYMAX_DETERMINISM: WAYMAX_DETERMINISM_SCHEMA,
        TYPED_PROVENANCE: TYPED_PROVENANCE_SCHEMA,
        EXECUTION_SUMMARY: EXECUTION_SUMMARY_SCHEMA,
        STAGE_TIMINGS: STAGE_TIMINGS_SCHEMA,
        REVIEW_DECISIONS: REVIEW_DECISIONS_SCHEMA,
    }
)


class M6ResultStoreError(RuntimeError):
    """Base M6 result-store contract error."""


class M6ResultStoreStateError(M6ResultStoreError):
    """Requested operation contradicts the one-way lifecycle."""


class M6ResultStoreIntegrityError(M6ResultStoreError):
    """Stored bytes, permissions, schemas, or domains are invalid."""


@dataclass(frozen=True, slots=True)
class M6ArtifactRecord:
    """One immutable file listed by the manifest."""

    path: str
    schema_identity: str
    rows: int | None
    size_bytes: int
    sha256: str = field(repr=False)

    def __post_init__(self) -> None:
        if (
            type(self.path) is not str
            or Path(self.path).name != self.path
            or self.path in {"", ".", ".."}
        ):
            raise ValueError("artifact path must be one canonical filename")
        if type(self.schema_identity) is not str or not self.schema_identity:
            raise ValueError("artifact schema identity must be non-empty")
        if self.rows is not None:
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
        if type(self.sha256) is not str or _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("artifact sha256 must be lowercase hexadecimal")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "rows": self.rows,
            "schema_identity": self.schema_identity,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "M6ArtifactRecord":
        if set(value) != {
            "path",
            "rows",
            "schema_identity",
            "sha256",
            "size_bytes",
        }:
            raise M6ResultStoreIntegrityError(
                "manifest artifact fields are not exact"
            )
        try:
            return cls(**dict(value))
        except (TypeError, ValueError, M6ResultStoreError) as exc:
            raise M6ResultStoreIntegrityError(
                "manifest artifact is invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class M6EligibilityReceipt:
    """Source-only membership and exact mode-dependent row accounting."""

    mode: str
    population_size: int
    eligible_cohort_indices: tuple[int, ...]
    secondary_b4_cohort_indices: tuple[int, ...]
    rejection_reason_counts: Mapping[str, int]
    primary_intervention_fingerprint: str = field(repr=False)
    secondary_intervention_fingerprint: str = field(repr=False)

    def __post_init__(self) -> None:
        profile = _profile(self.mode)
        if type(self.population_size) is not int or (
            self.population_size != profile.population_size
        ):
            raise ValueError("eligibility population differs from mode")
        eligible = _ordered_indices(
            self.eligible_cohort_indices,
            population_size=profile.population_size,
            name="eligible_cohort_indices",
        )
        secondary = _ordered_indices(
            self.secondary_b4_cohort_indices,
            population_size=profile.population_size,
            name="secondary_b4_cohort_indices",
        )
        if not set(secondary).issubset(eligible):
            raise ValueError("secondary b4 subset must be nested in primary")
        if profile.mode in {OFFICIAL_MODE, COMPUTE_PILOT_MODE} and len(
            eligible
        ) < 10:
            raise ValueError(f"{profile.mode} requires primary N >= 10")
        if profile.mode == DATA_FREE_MODE and (
            eligible != tuple(range(10)) or secondary != tuple(range(10))
        ):
            raise ValueError("data_free mode requires exact primary/secondary N=10")
        if profile.mode == ELIGIBILITY_ONLY_MODE and secondary:
            raise ValueError(
                "eligibility_only cannot retain the later b4 severity subset"
            )
        counts = dict(self.rejection_reason_counts)
        if set(counts) != set(M6_PRIMARY_REJECTION_REASONS):
            raise ValueError("eligibility rejection domain is not exact")
        normalized_counts = {
            reason: _integer(counts[reason], name=reason, minimum=0)
            for reason in M6_PRIMARY_REJECTION_REASONS
        }
        if len(eligible) + sum(normalized_counts.values()) != profile.population_size:
            raise ValueError("eligible plus rejected must equal population")
        for name in (
            "primary_intervention_fingerprint",
            "secondary_intervention_fingerprint",
        ):
            value = getattr(self, name)
            if type(value) is not str or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be SHA-256")
        if (
            self.primary_intervention_fingerprint
            != M6_PRIMARY_INTERVENTION_FINGERPRINT
            or self.secondary_intervention_fingerprint
            != M6_SECONDARY_INTERVENTION_FINGERPRINT
        ):
            raise ValueError(
                "eligibility receipt must use the exact registered b=2/b=4 "
                "longitudinal_brake_pulse/v1 fingerprints"
            )
        object.__setattr__(self, "eligible_cohort_indices", eligible)
        object.__setattr__(self, "secondary_b4_cohort_indices", secondary)
        object.__setattr__(
            self,
            "rejection_reason_counts",
            MappingProxyType(normalized_counts),
        )

    @property
    def eligible_count(self) -> int:
        return len(self.eligible_cohort_indices)

    @property
    def secondary_b4_count(self) -> int:
        return len(self.secondary_b4_cohort_indices)

    @property
    def expected_rows(self) -> Mapping[str, int]:
        rows: dict[str, int] = {
            ELIGIBILITY_LEDGER: self.population_size,
            WAYMAX_QUALIFICATION: self.eligible_count,
        }
        if self.mode == COMPUTE_PILOT_MODE:
            rows[COMPUTE_PILOT_SUMMARY] = 1
        elif self.mode in {OFFICIAL_MODE, DATA_FREE_MODE}:
            rows.update(
                {
                    PRIMARY_SCENE_SCALARS: self.eligible_count * 12,
                    PRIMARY_MATRIX: 12,
                    PRIMARY_REPEAT_SCENE_SCALARS: self.eligible_count * 12,
                    PRIMARY_REPEAT_MATRIX: 12,
                    SECONDARY_SCENE_SCALARS: self.secondary_b4_count * 12,
                    SECONDARY_MATRIX: 12,
                    NEGATIVE_TIMING_OBSERVATIONS: (
                        self.eligible_count * 9 + self.secondary_b4_count
                    ),
                    NEGATIVE_TIMING_GATES: len(
                        M6_NEGATIVE_TIMING_GATE_DOMAIN
                    ),
                    WAYMAX_ACCOUNTING: len(M6_WAYMAX_ROW_DOMAIN),
                    WAYMAX_SCENE_SCALARS: (
                        M6_WAYMAX_MAX_SELECTED
                        * len(M6_WAYMAX_BUNDLES)
                        * len(M6_PRIMARY_METRICS)
                    ),
                    WAYMAX_FIELD_COMPARISONS: (
                        M6_WAYMAX_MAX_SELECTED
                        * len(M6_WAYMAX_BUNDLES)
                        * len(M6_WAYMAX_CONDITIONS)
                        * len(M6_WAYMAX_COMPARISON_FIELDS)
                    ),
                    WAYMAX_DETERMINISM: M6_WAYMAX_DETERMINISM_ROW_COUNT,
                    TYPED_PROVENANCE: 1,
                    EXECUTION_SUMMARY: 1,
                    STAGE_TIMINGS: len(M6_STAGE_DOMAIN),
                    REVIEW_DECISIONS: len(M6_REVIEW_ROLE_DOMAIN),
                }
            )
        return MappingProxyType(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible_cohort_indices": list(self.eligible_cohort_indices),
            "eligible_count": self.eligible_count,
            "expected_rows": dict(self.expected_rows),
            "mode": self.mode,
            "population_size": self.population_size,
            "primary_intervention_fingerprint": (
                self.primary_intervention_fingerprint
            ),
            "rejection_reason_counts": dict(self.rejection_reason_counts),
            "schema_version": M6_ELIGIBILITY_RECEIPT_SCHEMA_VERSION,
            "secondary_b4_cohort_indices": list(
                self.secondary_b4_cohort_indices
            ),
            "secondary_b4_count": self.secondary_b4_count,
            "secondary_intervention_fingerprint": (
                self.secondary_intervention_fingerprint
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "M6EligibilityReceipt":
        expected = {
            "eligible_cohort_indices",
            "eligible_count",
            "expected_rows",
            "mode",
            "population_size",
            "primary_intervention_fingerprint",
            "rejection_reason_counts",
            "schema_version",
            "secondary_b4_cohort_indices",
            "secondary_b4_count",
            "secondary_intervention_fingerprint",
        }
        if set(value) != expected or (
            value.get("schema_version")
            != M6_ELIGIBILITY_RECEIPT_SCHEMA_VERSION
        ):
            raise M6ResultStoreIntegrityError(
                "eligibility receipt fields/schema are not exact"
            )
        try:
            receipt = cls(
                mode=_json_text(value["mode"], "mode"),
                population_size=_json_integer(
                    value["population_size"], "population_size", minimum=1
                ),
                eligible_cohort_indices=tuple(
                    _json_integer(item, "eligible cohort index", minimum=0)
                    for item in _json_array(
                        value["eligible_cohort_indices"],
                        "eligible_cohort_indices",
                    )
                ),
                secondary_b4_cohort_indices=tuple(
                    _json_integer(item, "secondary cohort index", minimum=0)
                    for item in _json_array(
                        value["secondary_b4_cohort_indices"],
                        "secondary_b4_cohort_indices",
                    )
                ),
                rejection_reason_counts=_json_mapping(
                    value["rejection_reason_counts"],
                    "rejection_reason_counts",
                ),
                primary_intervention_fingerprint=_json_text(
                    value["primary_intervention_fingerprint"],
                    "primary_intervention_fingerprint",
                ),
                secondary_intervention_fingerprint=_json_text(
                    value["secondary_intervention_fingerprint"],
                    "secondary_intervention_fingerprint",
                ),
            )
        except M6ResultStoreIntegrityError:
            raise
        except (TypeError, ValueError, M6ResultStoreError) as exc:
            raise M6ResultStoreIntegrityError(
                "eligibility receipt is invalid"
            ) from exc
        expected_rows = _json_mapping(
            value["expected_rows"],
            "expected_rows",
        )
        normalized_expected_rows = {
            _json_text(name, "expected row dataset"): _json_integer(
                count,
                f"expected row count for {name}",
                minimum=0,
            )
            for name, count in expected_rows.items()
        }
        if (
            _json_integer(value["eligible_count"], "eligible_count", minimum=0)
            != receipt.eligible_count
            or _json_integer(
                value["secondary_b4_count"],
                "secondary_b4_count",
                minimum=0,
            )
            != receipt.secondary_b4_count
            or normalized_expected_rows != dict(receipt.expected_rows)
        ):
            raise M6ResultStoreIntegrityError(
                "eligibility receipt accounting drifted"
            )
        return receipt


def _m6_waymax_selection_binding_sha256(
    selection: M6WaymaxSelection,
) -> str:
    """Mirror the frozen metric bridge's public-safe selection commitment."""

    if not isinstance(selection, M6WaymaxSelection):
        raise TypeError("selection must be an M6WaymaxSelection")
    selection.revalidate()
    payload = {
        "eligible_count": selection.eligible_count,
        "qualification_ledger_sha256": (
            selection.qualification_ledger_sha256
        ),
        "members": [
            {
                "cohort_index": member.cohort_index,
                "qualification_binding_sha256": (
                    member.qualification_binding_sha256
                ),
                "rank_sha256": member.rank_sha256,
            }
            for member in selection.members
        ],
        "primary_domain_member_count": (
            selection.primary_domain_member_count
        ),
        "primary_domain_sha256": selection.primary_domain_sha256,
        "supported": selection.supported,
        "selector_selection_sha256": selection.selection_sha256,
    }
    return hashlib.sha256(
        b"evalsim-m6-waymax-selection-binding-v1"
        + b"\x00"
        + _canonical_json_text(payload).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class M6WaymaxSelectionReceipt:
    """Safe commitment to the typed qualification ledger and canonical selection."""

    mode: str
    status: str
    primary_domain_sha256: str
    primary_domain_member_count: int
    qualification_ledger_sha256: str | None
    selector_selection_sha256: str | None
    selection_binding_sha256: str
    selection_supported: bool
    eligible_count: int
    selection_member_count: int
    identity_configuration_fingerprint: str
    primary_b2_configuration_fingerprint: str

    def __post_init__(self) -> None:
        profile = _profile(self.mode)
        if self.status not in {"sealed", "not_applicable"}:
            raise ValueError("Waymax selection receipt status is invalid")
        for name in (
            "primary_domain_sha256",
            "selection_binding_sha256",
            "identity_configuration_fingerprint",
            "primary_b2_configuration_fingerprint",
        ):
            value = getattr(self, name)
            if type(value) is not str or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be SHA-256")
        for name in (
            "qualification_ledger_sha256",
            "selector_selection_sha256",
        ):
            value = getattr(self, name)
            if value is not None and (
                type(value) is not str or _SHA256.fullmatch(value) is None
            ):
                raise ValueError(f"{name} must be null or SHA-256")
        for name in (
            "primary_domain_member_count",
            "eligible_count",
            "selection_member_count",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name=name, minimum=0),
            )
        if type(self.selection_supported) is not bool:
            raise TypeError("selection_supported must be an exact bool")
        if (
            self.identity_configuration_fingerprint
            != M6_WAYMAX_IDENTITY_CONFIGURATION_FINGERPRINT
            or self.primary_b2_configuration_fingerprint
            != M6_WAYMAX_PRIMARY_B2_CONFIGURATION_FINGERPRINT
        ):
            raise ValueError(
                "Waymax receipt requires exact identity and primary b=2 "
                "fingerprints"
            )
        if profile.data_free:
            if (
                self.status != "not_applicable"
                or self.primary_domain_sha256
                != M6_DATA_FREE_WAYMAX_PRIMARY_DOMAIN_SHA256
                or self.selection_binding_sha256
                != M6_DATA_FREE_WAYMAX_SELECTION_BINDING_SHA256
                or self.primary_domain_member_count != 0
                or self.qualification_ledger_sha256 is not None
                or self.selector_selection_sha256 is not None
                or self.selection_supported
                or self.eligible_count != 0
                or self.selection_member_count != 0
            ):
                raise ValueError(
                    "data_free Waymax selection receipt is not the exact "
                    "non-applicable placeholder"
                )
            return
        expected_supported = self.eligible_count >= 8
        expected_members = (
            min(self.eligible_count, M6_WAYMAX_MAX_SELECTED)
            if expected_supported
            else 0
        )
        if (
            self.status != "sealed"
            or self.qualification_ledger_sha256 is None
            or self.selector_selection_sha256 is None
            or self.selection_supported != expected_supported
            or self.selection_member_count != expected_members
            or self.eligible_count > self.primary_domain_member_count
        ):
            raise ValueError(
                "Waymax selection receipt does not describe the canonical "
                "complete-ledger 16-or-floor selection"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible_count": self.eligible_count,
            "identity_configuration_fingerprint": (
                self.identity_configuration_fingerprint
            ),
            "mode": self.mode,
            "primary_b2_configuration_fingerprint": (
                self.primary_b2_configuration_fingerprint
            ),
            "primary_domain_member_count": self.primary_domain_member_count,
            "primary_domain_sha256": self.primary_domain_sha256,
            "qualification_ledger_sha256": self.qualification_ledger_sha256,
            "schema_version": M6_WAYMAX_SELECTION_RECEIPT_SCHEMA_VERSION,
            "selection_binding_sha256": self.selection_binding_sha256,
            "selection_member_count": self.selection_member_count,
            "selection_supported": self.selection_supported,
            "selector_selection_sha256": self.selector_selection_sha256,
            "status": self.status,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "M6WaymaxSelectionReceipt":
        expected = {
            "eligible_count",
            "identity_configuration_fingerprint",
            "mode",
            "primary_b2_configuration_fingerprint",
            "primary_domain_member_count",
            "primary_domain_sha256",
            "qualification_ledger_sha256",
            "schema_version",
            "selection_binding_sha256",
            "selection_member_count",
            "selection_supported",
            "selector_selection_sha256",
            "status",
        }
        if set(value) != expected or value.get("schema_version") != (
            M6_WAYMAX_SELECTION_RECEIPT_SCHEMA_VERSION
        ):
            raise M6ResultStoreIntegrityError(
                "Waymax selection receipt fields/schema are not exact"
            )
        try:
            return cls(
                mode=_json_text(value["mode"], "mode"),
                status=_json_text(value["status"], "status"),
                primary_domain_sha256=_json_text(
                    value["primary_domain_sha256"],
                    "primary_domain_sha256",
                ),
                primary_domain_member_count=_json_integer(
                    value["primary_domain_member_count"],
                    "primary_domain_member_count",
                    minimum=0,
                ),
                qualification_ledger_sha256=_json_optional_text(
                    value["qualification_ledger_sha256"],
                    "qualification_ledger_sha256",
                ),
                selector_selection_sha256=_json_optional_text(
                    value["selector_selection_sha256"],
                    "selector_selection_sha256",
                ),
                selection_binding_sha256=_json_text(
                    value["selection_binding_sha256"],
                    "selection_binding_sha256",
                ),
                selection_supported=_json_boolean(
                    value["selection_supported"],
                    "selection_supported",
                ),
                eligible_count=_json_integer(
                    value["eligible_count"],
                    "eligible_count",
                    minimum=0,
                ),
                selection_member_count=_json_integer(
                    value["selection_member_count"],
                    "selection_member_count",
                    minimum=0,
                ),
                identity_configuration_fingerprint=_json_text(
                    value["identity_configuration_fingerprint"],
                    "identity_configuration_fingerprint",
                ),
                primary_b2_configuration_fingerprint=_json_text(
                    value["primary_b2_configuration_fingerprint"],
                    "primary_b2_configuration_fingerprint",
                ),
            )
        except M6ResultStoreIntegrityError:
            raise
        except (TypeError, ValueError, M6ResultStoreError) as exc:
            raise M6ResultStoreIntegrityError(
                "Waymax selection receipt is invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class M6DeterminismReceipt:
    """Two independently produced canonical logical-content passes."""

    mode: str
    primary_scene_pass_1_sha256: str = field(repr=False)
    primary_scene_pass_2_sha256: str = field(repr=False)
    primary_matrix_pass_1_sha256: str = field(repr=False)
    primary_matrix_pass_2_sha256: str = field(repr=False)
    waymax_repeat_status: str
    waymax_repeat_rows: int

    def __post_init__(self) -> None:
        profile = _profile(self.mode)
        if not profile.complete_results:
            raise ValueError("determinism receipt is complete-result-only")
        for name in (
            "primary_scene_pass_1_sha256",
            "primary_scene_pass_2_sha256",
            "primary_matrix_pass_1_sha256",
            "primary_matrix_pass_2_sha256",
        ):
            value = getattr(self, name)
            if type(value) is not str or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be SHA-256")
        if (
            self.primary_scene_pass_1_sha256
            != self.primary_scene_pass_2_sha256
            or self.primary_matrix_pass_1_sha256
            != self.primary_matrix_pass_2_sha256
        ):
            raise ValueError("deterministic pass-1/pass-2 content disagrees")
        if self.waymax_repeat_status not in {
            "passed",
            "failed",
            "not_applicable",
        }:
            raise ValueError("waymax repeat status is not registered")
        object.__setattr__(
            self,
            "waymax_repeat_rows",
            _integer(
                self.waymax_repeat_rows,
                name="waymax_repeat_rows",
                minimum=0,
                maximum=M6_WAYMAX_DETERMINISM_ROW_COUNT,
            ),
        )
        if (
            self.waymax_repeat_status == "not_applicable"
            and self.waymax_repeat_rows != 0
        ) or (
            self.waymax_repeat_status != "not_applicable"
            and self.waymax_repeat_rows == 0
        ):
            raise ValueError("Waymax repeat status/row accounting disagrees")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "primary_matrix_pass_1_sha256": (
                self.primary_matrix_pass_1_sha256
            ),
            "primary_matrix_pass_2_sha256": (
                self.primary_matrix_pass_2_sha256
            ),
            "primary_scene_pass_1_sha256": (
                self.primary_scene_pass_1_sha256
            ),
            "primary_scene_pass_2_sha256": (
                self.primary_scene_pass_2_sha256
            ),
            "schema_version": M6_DETERMINISM_RECEIPT_SCHEMA_VERSION,
            "waymax_repeat_rows": self.waymax_repeat_rows,
            "waymax_repeat_status": self.waymax_repeat_status,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "M6DeterminismReceipt":
        expected = {
            "mode",
            "primary_matrix_pass_1_sha256",
            "primary_matrix_pass_2_sha256",
            "primary_scene_pass_1_sha256",
            "primary_scene_pass_2_sha256",
            "schema_version",
            "waymax_repeat_rows",
            "waymax_repeat_status",
        }
        if set(value) != expected or (
            value.get("schema_version")
            != M6_DETERMINISM_RECEIPT_SCHEMA_VERSION
        ):
            raise M6ResultStoreIntegrityError(
                "determinism receipt fields/schema are not exact"
            )
        try:
            return cls(
                mode=_json_text(value["mode"], "mode"),
                primary_scene_pass_1_sha256=_json_text(
                    value["primary_scene_pass_1_sha256"], "scene pass 1"
                ),
                primary_scene_pass_2_sha256=_json_text(
                    value["primary_scene_pass_2_sha256"], "scene pass 2"
                ),
                primary_matrix_pass_1_sha256=_json_text(
                    value["primary_matrix_pass_1_sha256"], "matrix pass 1"
                ),
                primary_matrix_pass_2_sha256=_json_text(
                    value["primary_matrix_pass_2_sha256"], "matrix pass 2"
                ),
                waymax_repeat_status=_json_text(
                    value["waymax_repeat_status"],
                    "waymax_repeat_status",
                ),
                waymax_repeat_rows=_json_integer(
                    value["waymax_repeat_rows"],
                    "waymax_repeat_rows",
                    minimum=0,
                ),
            )
        except M6ResultStoreIntegrityError:
            raise
        except (TypeError, ValueError, M6ResultStoreError) as exc:
            raise M6ResultStoreIntegrityError(
                "determinism receipt is invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class VerifiedM6ResultStore:
    """Read-only result of independent on-disk verification."""

    run_path: Path
    profile: M6ResultProfile
    receipt: M6EligibilityReceipt
    waymax_selection_receipt: M6WaymaxSelectionReceipt
    manifest: Mapping[str, Any]
    artifacts: tuple[M6ArtifactRecord, ...]
    tables: Mapping[str, pa.Table] = field(repr=False)

    def read_dataset(self, name: str) -> pa.Table:
        try:
            return self.tables[name]
        except KeyError as exc:
            raise KeyError(f"dataset {name!r} is unavailable in this mode") from exc


@dataclass(frozen=True, slots=True)
class M6SanitizedAggregate:
    """Immutable canonical ASCII JSON representation of the seven public domains."""

    canonical_json: str
    _factory_sentinel: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_sentinel is not _AGGREGATE_SENTINEL:
            raise M6ResultStoreStateError(
                "sanitized aggregates can only be constructed from a verified "
                "terminal official store"
            )
        if type(self.canonical_json) is not str:
            raise TypeError("canonical_json must be exact str")
        try:
            payload = json.loads(self.canonical_json)
        except json.JSONDecodeError as exc:
            raise ValueError("canonical_json is invalid") from exc
        if (
            type(payload) is not dict
            or tuple(payload) != tuple(sorted(M6_PROMOTED_TOP_LEVEL_DOMAINS))
            or _canonical_json_text(payload) != self.canonical_json
        ):
            raise ValueError("aggregate is not the canonical seven-domain object")
        _validate_sanitized_aggregate_payload(payload)
        _assert_promoted_privacy(payload)

    @property
    def canonical_bytes(self) -> bytes:
        return self.canonical_json.encode("ascii")

    def to_dict(self) -> dict[str, Any]:
        value = json.loads(self.canonical_json)
        assert type(value) is dict
        return value


@dataclass(frozen=True, slots=True)
class M6ObservedPreflightResult:
    """Exact observation minted only by the trusted official verifier boundary.

    This type deliberately does not perform Git, shard, runtime, or terminal-capture
    checks itself.  It is the narrow hand-off from the future official verifier,
    whose trust is limited to those seven observed booleans and one local evidence
    catalog precursor digest.  The store independently binds that observation to the
    already-created COMMITTED and manifest bytes.
    """

    mode: str
    result_path: str
    manifest_sha256: str
    committed_sha256: str
    evidence_catalog_sha256: str
    checks: Mapping[str, bool]
    _factory_sentinel: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_sentinel is not _OBSERVED_PREFLIGHT_SENTINEL:
            raise M6ResultStoreStateError(
                "observed preflight results are verifier-minted only"
            )
        profile = _profile(self.mode)
        if profile.data_free:
            raise ValueError("data_free terminalization self-verifies")
        if (
            type(self.result_path) is not str
            or not self.result_path.startswith("outputs/m6/")
            or Path(self.result_path).as_posix() != self.result_path
        ):
            raise ValueError("observed preflight result path is invalid")
        for name in (
            "manifest_sha256",
            "committed_sha256",
            "evidence_catalog_sha256",
        ):
            if (
                type(getattr(self, name)) is not str
                or _SHA256.fullmatch(getattr(self, name)) is None
            ):
                raise ValueError(f"{name} must be SHA-256")
        checks = dict(self.checks)
        if set(checks) != set(M6_PREFLIGHT_CHECK_DOMAIN) or any(
            type(value) is not bool or value is not True
            for value in checks.values()
        ):
            raise ValueError(
                "observed preflight must contain the exact all-passed check domain"
            )
        object.__setattr__(
            self,
            "checks",
            MappingProxyType(
                {name: checks[name] for name in M6_PREFLIGHT_CHECK_DOMAIN}
            ),
        )

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json_bytes(
                {
                    "checks": dict(self.checks),
                    "committed_sha256": self.committed_sha256,
                    "evidence_catalog_sha256": (
                        self.evidence_catalog_sha256
                    ),
                    "manifest_sha256": self.manifest_sha256,
                    "mode": self.mode,
                    "result_path": self.result_path,
                }
            )
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class M6TerminalCapability:
    """One-use success authority bound to one exact committed store."""

    mode: str
    result_path: str
    manifest_sha256: str
    committed_sha256: str
    evidence_catalog_sha256: str
    observed_preflight_sha256: str
    nonce: bytes = field(repr=False)
    _factory_sentinel: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_sentinel is not _TERMINAL_CAPABILITY_SENTINEL:
            raise M6ResultStoreStateError(
                "terminal capabilities can only be minted by the M6 verifier hook"
            )
        if type(self.nonce) is not bytes or len(self.nonce) != 32:
            raise ValueError("terminal capability nonce is invalid")
        for name in (
            "manifest_sha256",
            "committed_sha256",
            "evidence_catalog_sha256",
            "observed_preflight_sha256",
        ):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be SHA-256")


class M6ResultStore:
    """Capability-bound exclusive writer for one never-resumed M6 run."""

    def __init__(
        self,
        *,
        project_root: Path,
        run_name: str,
        run_path: Path,
        profile: M6ResultProfile,
        capability_nonce: bytes,
        _create_sentinel: object,
    ) -> None:
        if _create_sentinel is not _CREATE_SENTINEL:
            raise M6ResultStoreStateError(
                "M6 writers can only be created by M6ResultStore.create"
            )
        if type(capability_nonce) is not bytes or len(capability_nonce) != 32:
            raise M6ResultStoreStateError("writer capability is invalid")
        self.project_root = project_root
        self.run_name = run_name
        self.run_path = run_path
        self.profile = profile
        self._capability_nonce: bytes | None = capability_nonce
        self._capability_sha256 = hashlib.sha256(capability_nonce).hexdigest()
        self._receipt: M6EligibilityReceipt | None = None
        self._waymax_selection: M6WaymaxSelection | None = None
        self._waymax_selection_receipt: M6WaymaxSelectionReceipt | None = None
        self._waymax_live_matrix: M6WaymaxMatrixResult | None = None
        self._artifacts: dict[str, M6ArtifactRecord] = {}
        self._phase = "pending"

    @classmethod
    def create(
        cls,
        project_root: str | Path,
        run_name: str,
        *,
        mode: str = OFFICIAL_MODE,
    ) -> "M6ResultStore":
        root = _validated_project_root(project_root)
        name = _validated_run_name(run_name)
        profile = _profile(mode)
        relative = Path("outputs") / "m6" / name
        _require_git_invisible(root, relative)
        outputs = _ensure_directory(root, "outputs", require_owner_mode=False)
        m6_root = _ensure_directory(outputs, "m6", require_owner_mode=True)
        run_path = m6_root / name
        if os.path.lexists(run_path):
            raise FileExistsError(
                f"M6 run {name!r} already exists and cannot be resumed"
            )
        nonce = secrets.token_bytes(32)
        capability_sha256 = hashlib.sha256(nonce).hexdigest()
        created = False
        try:
            os.mkdir(run_path, 0o700)
            created = True
            _fsync_directory(m6_root)
            _guard_run_directory(run_path)
            pending = _pending_payload(
                name,
                profile,
                capability_sha256,
            )
            _write_bytes_exclusive(
                run_path / PENDING_MARKER,
                _canonical_json_bytes(pending),
                run_path,
            )
        except BaseException:
            if created:
                _best_effort_failure(
                    run_path,
                    profile.mode,
                    "creation_failed",
                )
            raise
        return cls(
            project_root=root,
            run_name=name,
            run_path=run_path,
            profile=profile,
            capability_nonce=nonce,
            _create_sentinel=_CREATE_SENTINEL,
        )

    @property
    def project_relative_path(self) -> Path:
        return Path("outputs") / "m6" / self.run_name

    @property
    def eligibility_receipt(self) -> M6EligibilityReceipt | None:
        return self._receipt

    @property
    def artifacts(self) -> tuple[M6ArtifactRecord, ...]:
        return tuple(self._artifacts[path] for path in sorted(self._artifacts))

    def write_eligibility_ledger(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        primary_intervention_fingerprint: str = (
            M6_PRIMARY_INTERVENTION_FINGERPRINT
        ),
        secondary_intervention_fingerprint: str = (
            M6_SECONDARY_INTERVENTION_FINGERPRINT
        ),
    ) -> M6EligibilityReceipt:
        try:
            self._assert_pending_capability()
            if self._receipt is not None:
                raise M6ResultStoreStateError(
                    "eligibility receipt already exists"
                )
            normalized = _normalize_eligibility(rows, self.profile)
            self._write_dataset(ELIGIBILITY_LEDGER, normalized)
            eligible = tuple(
                row["cohort_index"]
                for row in normalized
                if row["primary_eligible"]
            )
            secondary = tuple(
                row["cohort_index"]
                for row in normalized
                if row["secondary_b4_feasible"] is True
            )
            reasons = {
                reason: sum(
                    row["rejection_reason"] == reason for row in normalized
                )
                for reason in M6_PRIMARY_REJECTION_REASONS
            }
            receipt = M6EligibilityReceipt(
                mode=self.profile.mode,
                population_size=self.profile.population_size,
                eligible_cohort_indices=eligible,
                secondary_b4_cohort_indices=secondary,
                rejection_reason_counts=reasons,
                primary_intervention_fingerprint=(
                    primary_intervention_fingerprint
                ),
                secondary_intervention_fingerprint=(
                    secondary_intervention_fingerprint
                ),
            )
            record = _write_json_artifact(
                self.run_path,
                ELIGIBILITY_RECEIPT_PATH,
                M6_ELIGIBILITY_RECEIPT_SCHEMA_VERSION,
                receipt.to_dict(),
            )
            self._artifacts[record.path] = record
            self._receipt = receipt
            return receipt
        except BaseException:
            self._poison("eligibility_write_failed")
            raise

    def write_waymax_qualification(
        self,
        selection: M6WaymaxSelection | None = None,
    ) -> tuple[M6ArtifactRecord, M6ArtifactRecord]:
        """Seal one canonical typed selection and its safe complete ledger.

        Data-free mode has no live Waymax selection and therefore accepts no
        caller evidence; it writes the one exact non-applicable placeholder.
        """

        try:
            self._assert_pending_capability()
            receipt = self._require_receipt()
            if self._waymax_selection_receipt is not None:
                raise M6ResultStoreStateError(
                    "Waymax qualification/selection receipt already exists"
                )
            if self.profile.data_free:
                if selection is not None:
                    raise TypeError(
                        "data_free qualification accepts no caller selection"
                    )
                selection_receipt = _data_free_waymax_selection_receipt(
                    receipt
                )
                rows = m6_data_free_waymax_qualification_rows(
                    receipt.eligible_cohort_indices
                )
            else:
                if not isinstance(selection, M6WaymaxSelection):
                    raise TypeError(
                        "non-data-free qualification requires one canonical "
                        "M6WaymaxSelection"
                    )
                selection_receipt = (
                    _waymax_selection_receipt_from_selection(
                        selection,
                        receipt,
                    )
                )
                rows = _waymax_qualification_rows_from_selection(
                    selection,
                    receipt,
                )
            normalized = _normalize_waymax_qualification(rows, receipt)
            qualification_record = self._write_dataset(
                WAYMAX_QUALIFICATION,
                normalized,
            )
            receipt_record = _write_json_artifact(
                self.run_path,
                WAYMAX_SELECTION_RECEIPT_PATH,
                M6_WAYMAX_SELECTION_RECEIPT_SCHEMA_VERSION,
                selection_receipt.to_dict(),
            )
            if receipt_record.path in self._artifacts:
                raise FileExistsError(
                    "Waymax selection receipt already exists"
                )
            self._artifacts[receipt_record.path] = receipt_record
            self._waymax_selection = selection
            self._waymax_selection_receipt = selection_receipt
            return qualification_record, receipt_record
        except BaseException:
            self._poison("waymax_qualification_write_failed")
            raise

    def write_compute_pilot_summary(
        self,
        row: Mapping[str, Any],
    ) -> M6ArtifactRecord:
        try:
            self._assert_pending_capability()
            receipt = self._require_receipt()
            if self.profile.mode != COMPUTE_PILOT_MODE:
                raise M6ResultStoreStateError(
                    "compute pilot evidence is compute_pilot-mode-only"
                )
            normalized = _normalize_compute_pilot((row,), receipt)
            return self._write_dataset(COMPUTE_PILOT_SUMMARY, normalized)
        except BaseException:
            self._poison("compute_pilot_write_failed")
            raise

    def write_primary_scene_scalars(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> M6ArtifactRecord:
        return self._write_complete_dataset(
            PRIMARY_SCENE_SCALARS,
            rows,
            _normalize_primary_scene_scalars,
        )

    def write_primary_matrix(
        self,
        rows: Iterable[Mapping[str, Any]] | None = None,
    ) -> M6ArtifactRecord:
        try:
            self._assert_complete_pending()
            receipt = self._require_receipt()
            scene_rows = self._read_dataset_rows(PRIMARY_SCENE_SCALARS)
            expected = _derive_primary_matrix_rows(scene_rows, receipt)
            if rows is not None:
                supplied = _normalize_primary_matrix(rows, receipt)
                if supplied != expected:
                    raise M6ResultStoreIntegrityError(
                        "primary matrix differs from independent stats.m6 "
                        "reconstruction"
                    )
            return self._write_dataset(PRIMARY_MATRIX, expected)
        except BaseException:
            self._poison("primary_matrix_write_failed")
            raise

    def write_primary_repeat_scene_scalars(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> M6ArtifactRecord:
        """Seal an independently produced normalized primary pass-2 table."""

        return self._write_complete_dataset(
            PRIMARY_REPEAT_SCENE_SCALARS,
            rows,
            _normalize_primary_scene_scalars,
        )

    def write_primary_repeat_matrix(self) -> M6ArtifactRecord:
        """Derive the pass-2 matrix from the separately sealed pass-2 scenes."""

        try:
            self._assert_complete_pending()
            receipt = self._require_receipt()
            scene_rows = self._read_dataset_rows(
                PRIMARY_REPEAT_SCENE_SCALARS
            )
            expected = _derive_primary_matrix_rows(scene_rows, receipt)
            return self._write_dataset(PRIMARY_REPEAT_MATRIX, expected)
        except BaseException:
            self._poison("primary_repeat_matrix_write_failed")
            raise

    def write_secondary_scene_scalars(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> M6ArtifactRecord:
        return self._write_complete_dataset(
            SECONDARY_SCENE_SCALARS,
            rows,
            _normalize_secondary_scene_scalars,
        )

    def write_secondary_matrix(
        self,
        rows: Iterable[Mapping[str, Any]] | None = None,
    ) -> M6ArtifactRecord:
        try:
            self._assert_complete_pending()
            receipt = self._require_receipt()
            scene_rows = self._read_dataset_rows(SECONDARY_SCENE_SCALARS)
            expected = _derive_secondary_matrix_rows(scene_rows, receipt)
            if rows is not None:
                supplied = _normalize_secondary_matrix(rows, receipt)
                if supplied != expected:
                    raise M6ResultStoreIntegrityError(
                        "secondary b4 matrix differs from sealed scene rows"
                    )
            return self._write_dataset(SECONDARY_MATRIX, expected)
        except BaseException:
            self._poison("secondary_matrix_write_failed")
            raise

    def write_negative_timing_observations(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> tuple[M6ArtifactRecord, M6ArtifactRecord]:
        """Seal exact observations and mechanically derive the six gates."""

        try:
            self._assert_complete_pending()
            receipt = self._require_receipt()
            normalized = _normalize_negative_timing_observations(
                rows,
                receipt,
            )
            observations = self._write_dataset(
                NEGATIVE_TIMING_OBSERVATIONS,
                normalized,
            )
            gates = self._write_dataset(
                NEGATIVE_TIMING_GATES,
                _derive_negative_timing_gates(normalized, receipt),
            )
            return observations, gates
        except BaseException:
            self._poison("negative_timing_observation_write_failed")
            raise

    def write_waymax_scene_scalars(
        self,
        issued_table: M6WaymaxIssuedScalarTable | None = None,
        *,
        selection: M6WaymaxSelection | None = None,
    ) -> M6ArtifactRecord:
        try:
            self._assert_complete_pending()
            receipt = self._require_receipt()
            selection_receipt = self._require_waymax_selection_receipt()
            qualification = self._read_dataset_rows(WAYMAX_QUALIFICATION)
            if self.profile.data_free:
                if issued_table is not None or selection is not None:
                    raise TypeError(
                        "data_free scalar placeholders accept no caller outcomes"
                    )
                parsed = parse_m6_waymax_scene_scalar_table(
                    m6_data_free_waymax_scene_scalar_rows()
                )
                parsed.revalidate()
                scalar_rows = tuple(
                    row.to_store_dict() for row in parsed.rows
                )
            else:
                if not isinstance(issued_table, M6WaymaxIssuedScalarTable):
                    raise TypeError(
                        "live scalar writing requires "
                        "M6WaymaxIssuedScalarTable"
                    )
                if not isinstance(selection, M6WaymaxSelection):
                    raise TypeError(
                        "live scalar writing requires the canonical "
                        "M6WaymaxSelection"
                    )
                _verify_waymax_selection_receipt_against_selection(
                    selection_receipt,
                    selection,
                    receipt,
                )
                issued_table.revalidate(selection=selection)
                matrix = analyze_m6_waymax_cells(
                    issued_table,
                    selection=selection,
                    intervention_configuration_fingerprint=(
                        M6_WAYMAX_PRIMARY_B2_CONFIGURATION_FINGERPRINT
                    ),
                )
                matrix.revalidate()
                self._waymax_live_matrix = matrix
                scalar_rows = tuple(
                    row.to_store_dict() for row in issued_table.rows
                )
            normalized = _normalize_waymax_scene_scalars_from_qualification(
                scalar_rows,
                receipt,
                qualification,
                selection_receipt,
            )
            return self._write_dataset(WAYMAX_SCENE_SCALARS, normalized)
        except BaseException:
            self._poison("waymax_scene_scalars_write_failed")
            raise

    def write_waymax_field_comparisons(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> M6ArtifactRecord:
        try:
            self._assert_complete_pending()
            receipt = self._require_receipt()
            self._require_waymax_selection_receipt()
            qualification = self._read_dataset_rows(WAYMAX_QUALIFICATION)
            normalized = (
                _normalize_waymax_field_comparisons_from_qualification(
                    rows,
                    receipt,
                    qualification,
                )
            )
            return self._write_dataset(
                WAYMAX_FIELD_COMPARISONS,
                normalized,
            )
        except BaseException:
            self._poison("waymax_field_comparisons_write_failed")
            raise

    def write_waymax_determinism(
        self,
        rows: object | None = None,
    ) -> M6ArtifactRecord:
        try:
            self._assert_pending_capability()
            if not self.profile.data_free:
                raise M6ResultStoreStateError(
                    "non-data-free Waymax determinism writing is disabled "
                    "until runner-issued evidence is implemented and reviewed"
                )
            receipt = self._require_receipt()
            self._assert_complete_pending()
            if rows is not None:
                raise TypeError(
                    "data_free Waymax determinism accepts no caller rows"
                )
            self._require_waymax_selection_receipt()
            qualification = self._read_dataset_rows(WAYMAX_QUALIFICATION)
            issued = build_m6_waymax_data_free_determinism_table()
            issued.revalidate()
            normalized = _normalize_waymax_determinism_from_qualification(
                issued.to_store_rows(),
                receipt,
                qualification,
            )
            return self._write_dataset(WAYMAX_DETERMINISM, normalized)
        except BaseException:
            self._poison("waymax_determinism_write_failed")
            raise

    def write_waymax_accounting(self) -> M6ArtifactRecord:
        """Derive all 58 Waymax scope/comparison/control/cell rows."""

        try:
            self._assert_complete_pending()
            receipt = self._require_receipt()
            qualification = self._read_dataset_rows(WAYMAX_QUALIFICATION)
            scalars = self._read_dataset_rows(WAYMAX_SCENE_SCALARS)
            comparisons = self._read_dataset_rows(
                WAYMAX_FIELD_COMPARISONS
            )
            rows = _derive_waymax_accounting(
                qualification,
                scalars,
                comparisons,
                receipt,
                self._require_waymax_selection_receipt(),
                matrix=self._waymax_live_matrix,
            )
            return self._write_dataset(WAYMAX_ACCOUNTING, rows)
        except BaseException:
            self._poison("waymax_accounting_write_failed")
            raise

    def write_typed_provenance(
        self,
        row: Mapping[str, Any],
    ) -> M6ArtifactRecord:
        return self._write_complete_dataset(
            TYPED_PROVENANCE,
            (row,),
            _normalize_typed_provenance,
        )

    def write_execution_summary(
        self,
        *,
        fresh_worker_peak_rss_bytes: int,
    ) -> M6ArtifactRecord:
        """Derive execution statuses/counts from already sealed evidence."""

        try:
            self._assert_complete_pending()
            receipt = self._require_receipt()
            row = _derive_execution_summary(
                receipt=receipt,
                tables={
                    name: self._read_dataset_rows(name)
                    for name in receipt.expected_rows
                    if name != EXECUTION_SUMMARY
                },
                determinism=M6DeterminismReceipt.from_dict(
                    _decode_canonical_mapping(
                        _read_guarded_bytes(
                            self.run_path / DETERMINISM_RECEIPT_PATH,
                            self.run_path,
                        ),
                        "determinism receipt",
                    )
                ),
                fresh_worker_peak_rss_bytes=fresh_worker_peak_rss_bytes,
            )
            return self._write_dataset(EXECUTION_SUMMARY, (row,))
        except BaseException:
            self._poison("execution_summary_write_failed")
            raise

    def write_stage_timings(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> M6ArtifactRecord:
        return self._write_complete_dataset(
            STAGE_TIMINGS,
            rows,
            _normalize_stage_timings,
        )

    def write_review_decisions(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> M6ArtifactRecord:
        try:
            self._assert_complete_pending()
            receipt = self._require_receipt()
            evidence_catalog_sha256 = _review_precursor_sha256(
                receipt,
                self.artifacts,
            )
            caller_fields = tuple(
                field.name
                for field in REVIEW_DECISIONS_SCHEMA
                if field.name != "evidence_catalog_sha256"
            )
            injected = tuple(
                {
                    **_exact_row(raw, caller_fields, REVIEW_DECISIONS),
                    "evidence_catalog_sha256": evidence_catalog_sha256,
                }
                for raw in rows
            )
            normalized = _normalize_review_decisions(
                injected,
                receipt,
                expected_evidence_catalog_sha256=(
                    evidence_catalog_sha256
                ),
            )
            return self._write_dataset(REVIEW_DECISIONS, normalized)
        except BaseException:
            self._poison("review_decisions_write_failed")
            raise

    def write_claim_limitations(self) -> M6ArtifactRecord:
        try:
            self._assert_complete_pending()
            payload = _claim_limitations_payload(self.profile.mode)
            record = _write_json_artifact(
                self.run_path,
                CLAIM_LIMITATIONS_PATH,
                M6_CLAIM_LIMITATIONS_SCHEMA_VERSION,
                payload,
            )
            if record.path in self._artifacts:
                raise FileExistsError("claim/limitations already exists")
            self._artifacts[record.path] = record
            return record
        except BaseException:
            self._poison("claim_limitations_write_failed")
            raise

    def write_determinism_receipt(self) -> M6ArtifactRecord:
        """Derive the receipt from two sealed primary passes and 64 Waymax rows."""

        try:
            self._assert_complete_pending()
            receipt = self._derive_determinism_receipt()
            record = _write_json_artifact(
                self.run_path,
                DETERMINISM_RECEIPT_PATH,
                M6_DETERMINISM_RECEIPT_SCHEMA_VERSION,
                receipt.to_dict(),
            )
            if record.path in self._artifacts:
                raise FileExistsError("determinism receipt already exists")
            self._artifacts[record.path] = record
            return record
        except BaseException:
            self._poison("determinism_receipt_write_failed")
            raise

    def commit(self) -> Path:
        try:
            self._assert_pending_capability()
            receipt = self._require_receipt()
            self._require_complete_artifacts(receipt)
            _verify_uncommitted_artifacts(
                self.run_path,
                self.profile,
                receipt,
                self.artifacts,
                waymax_selection=self._waymax_selection,
            )
            row_domain_sha256 = _row_domain_sha256(receipt)
            schema_fingerprints = _schema_fingerprints_for_receipt(receipt)
            manifest = {
                "artifacts": [
                    record.to_dict() for record in self.artifacts
                ],
                "capability_sha256": self._capability_sha256,
                "complete": True,
                "expected_rows": dict(receipt.expected_rows),
                "hash_policy": {
                    "algorithm": "sha256",
                    "manifest_self_hash": False,
                },
                "mode": self.profile.mode,
                "population_size": self.profile.population_size,
                "result_path": self.project_relative_path.as_posix(),
                "row_domain_sha256": row_domain_sha256,
                "schema_fingerprints": schema_fingerprints,
                "schema_version": M6_RESULT_STORE_SCHEMA_VERSION,
            }
            manifest_bytes = _canonical_json_bytes(manifest)
            _write_bytes_exclusive(
                self.run_path / MANIFEST_PATH,
                manifest_bytes,
                self.run_path,
            )
            committed = {
                "expected_rows": dict(receipt.expected_rows),
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "manifest_size_bytes": len(manifest_bytes),
                "mode": self.profile.mode,
                "row_domain_sha256": row_domain_sha256,
                "schema_fingerprints": schema_fingerprints,
                "schema_version": M6_RESULT_STORE_SCHEMA_VERSION,
                "state": "COMMITTED",
            }
            _write_bytes_exclusive(
                self.run_path / COMMITTED_MARKER,
                _canonical_json_bytes(committed),
                self.run_path,
            )
            self._phase = "committed"
            verify_committed_m6_result_store(
                self.project_root,
                self.run_name,
                allow_data_free=self.profile.data_free,
                expected_mode=self.profile.mode,
            )
            return self.run_path
        except BaseException:
            self._poison("commit_failed")
            raise

    def mark_terminal_success(
        self,
        *,
        capability: M6TerminalCapability | None = None,
    ) -> Path:
        try:
            self._assert_committed_capability()
            if not self.profile.data_free:
                raise M6ResultStoreStateError(
                    "non-data-free terminal success is disabled until the "
                    "official M6 verifier/CLI is implemented and reviewed"
                )
            verify_committed_m6_result_store(
                self.project_root,
                self.run_name,
                allow_data_free=self.profile.data_free,
                expected_mode=self.profile.mode,
            )
            manifest_bytes = _read_guarded_bytes(
                self.run_path / MANIFEST_PATH,
                self.run_path,
            )
            committed_bytes = _read_guarded_bytes(
                self.run_path / COMMITTED_MARKER,
                self.run_path,
            )
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            committed_sha256 = hashlib.sha256(committed_bytes).hexdigest()
            if capability is not None:
                raise M6ResultStoreStateError(
                    "data_free success self-verifies and accepts no external "
                    "terminal capability"
                )
            evidence_catalog_sha256 = self._review_evidence_catalog_sha256()
            observed_preflight_sha256 = hashlib.sha256(
                b"evalsim-m6-data-free-self-verification-v1\x00"
                + bytes.fromhex(manifest_sha256)
                + bytes.fromhex(committed_sha256)
                + bytes.fromhex(evidence_catalog_sha256)
            ).hexdigest()
            success_bytes = _canonical_json_bytes(
                {
                    "committed_sha256": committed_sha256,
                    "evidence_catalog_sha256": evidence_catalog_sha256,
                    "manifest_sha256": manifest_sha256,
                    "mode": self.profile.mode,
                    "observed_preflight_sha256": observed_preflight_sha256,
                    "schema_version": M6_RESULT_STORE_SCHEMA_VERSION,
                    "state": "TERMINAL_SUCCESS",
                }
            )
        except BaseException:
            if self._phase != "failure":
                self._poison("pre_terminal_verification_failed")
            raise

        # This is deliberately the last fallible operation.  A post-create fsync
        # error is reconciled as success only if the exact guarded marker exists.
        try:
            _write_terminal_success_final(
                self.run_path / TERMINAL_SUCCESS_MARKER,
                success_bytes,
                self.run_path,
            )
        except BaseException:
            if _path_kind(
                self.run_path / TERMINAL_SUCCESS_MARKER
            ) == "missing":
                self._poison("terminal_success_write_failed")
            else:
                # The marker exists but directory durability could not be
                # established.  This writer is permanently ambiguous and must not
                # report success or create a contradictory failure marker.
                self._phase = "ambiguous"
                self._capability_nonce = None
            raise
        self._phase = "success"
        self._capability_nonce = None
        return self.run_path

    def finalize(self) -> Path:
        if not self.profile.data_free:
            raise M6ResultStoreStateError(
                "non-data-free runs must commit, pass the official verifier hook, "
                "and present its terminal capability"
            )
        self.commit()
        return self.mark_terminal_success()

    def fail(self, reason_code: str) -> Path:
        self._assert_not_successful()
        return self._poison(reason_code)

    def _write_complete_dataset(
        self,
        name: str,
        rows: Iterable[Mapping[str, Any]],
        normalizer: Callable[
            [Iterable[Mapping[str, Any]], M6EligibilityReceipt],
            tuple[dict[str, Any], ...],
        ],
    ) -> M6ArtifactRecord:
        try:
            self._assert_complete_pending()
            receipt = self._require_receipt()
            normalized = normalizer(rows, receipt)
            return self._write_dataset(name, normalized)
        except BaseException:
            self._poison(f"{name.replace('-', '_')}_write_failed")
            raise

    def _write_dataset(
        self,
        name: str,
        rows: Sequence[Mapping[str, Any]],
    ) -> M6ArtifactRecord:
        self._assert_pending_capability()
        path_name = _DATASET_PATHS[name]
        if path_name in self._artifacts or os.path.lexists(
            self.run_path / path_name
        ):
            raise FileExistsError(f"{name} already exists")
        table = pa.Table.from_pylist(list(rows), schema=M6_RESULT_SCHEMAS[name])
        table.validate(full=True)
        path = self.run_path / path_name
        _write_parquet_exclusive(path, table, self.run_path)
        digest, size = _guarded_sha256(path, self.run_path)
        record = M6ArtifactRecord(
            path=path_name,
            schema_identity=name,
            rows=table.num_rows,
            size_bytes=size,
            sha256=digest,
        )
        self._artifacts[path_name] = record
        return record

    def _read_dataset_rows(self, name: str) -> tuple[dict[str, Any], ...]:
        path_name = _DATASET_PATHS[name]
        if path_name not in self._artifacts:
            raise M6ResultStoreStateError(f"{name} must be sealed first")
        return tuple(
            _read_guarded_parquet(
                self.run_path / path_name,
                self.run_path,
                name,
            ).to_pylist()
        )

    def _validate_determinism_receipt(
        self,
        receipt: M6DeterminismReceipt,
    ) -> None:
        if receipt.mode != self.profile.mode:
            raise M6ResultStoreIntegrityError(
                "determinism receipt mode drifted"
            )
        expected_scene = _canonical_rows_sha256(
            PRIMARY_SCENE_SCALARS,
            self._read_dataset_rows(PRIMARY_SCENE_SCALARS),
        )
        expected_repeat_scene = _canonical_rows_sha256(
            PRIMARY_SCENE_SCALARS,
            self._read_dataset_rows(PRIMARY_REPEAT_SCENE_SCALARS),
        )
        expected_matrix = _canonical_rows_sha256(
            PRIMARY_MATRIX,
            self._read_dataset_rows(PRIMARY_MATRIX),
        )
        expected_repeat_matrix = _canonical_rows_sha256(
            PRIMARY_MATRIX,
            self._read_dataset_rows(PRIMARY_REPEAT_MATRIX),
        )
        if (
            receipt.primary_scene_pass_1_sha256 != expected_scene
            or receipt.primary_scene_pass_2_sha256 != expected_repeat_scene
            or receipt.primary_matrix_pass_1_sha256 != expected_matrix
            or receipt.primary_matrix_pass_2_sha256 != expected_repeat_matrix
            or expected_scene != expected_repeat_scene
            or expected_matrix != expected_repeat_matrix
        ):
            raise M6ResultStoreIntegrityError(
                "determinism receipt does not bind canonical sealed content"
            )

    def _derive_determinism_receipt(self) -> M6DeterminismReceipt:
        scene_pass_1 = _canonical_rows_sha256(
            PRIMARY_SCENE_SCALARS,
            self._read_dataset_rows(PRIMARY_SCENE_SCALARS),
        )
        scene_pass_2 = _canonical_rows_sha256(
            PRIMARY_SCENE_SCALARS,
            self._read_dataset_rows(PRIMARY_REPEAT_SCENE_SCALARS),
        )
        matrix_pass_1 = _canonical_rows_sha256(
            PRIMARY_MATRIX,
            self._read_dataset_rows(PRIMARY_MATRIX),
        )
        matrix_pass_2 = _canonical_rows_sha256(
            PRIMARY_MATRIX,
            self._read_dataset_rows(PRIMARY_REPEAT_MATRIX),
        )
        waymax = self._read_dataset_rows(WAYMAX_DETERMINISM)
        selected = [row for row in waymax if row["status"] != "not_applicable"]
        waymax_status = (
            "not_applicable"
            if not selected
            else (
                "passed"
                if all(row["status"] == "passed" for row in selected)
                else "failed"
            )
        )
        return M6DeterminismReceipt(
            mode=self.profile.mode,
            primary_scene_pass_1_sha256=scene_pass_1,
            primary_scene_pass_2_sha256=scene_pass_2,
            primary_matrix_pass_1_sha256=matrix_pass_1,
            primary_matrix_pass_2_sha256=matrix_pass_2,
            waymax_repeat_status=waymax_status,
            waymax_repeat_rows=len(selected),
        )

    def _review_evidence_catalog_sha256(self) -> str:
        rows = self._read_dataset_rows(REVIEW_DECISIONS)
        values = {row["evidence_catalog_sha256"] for row in rows}
        if len(values) != 1:
            raise M6ResultStoreIntegrityError(
                "review decisions do not bind one evidence catalog precursor"
            )
        return next(iter(values))

    def _require_complete_artifacts(
        self,
        receipt: M6EligibilityReceipt,
    ) -> None:
        expected = _expected_artifact_paths(receipt)
        if set(self._artifacts) != expected:
            raise M6ResultStoreIntegrityError(
                "mode is missing required artifacts or has unexpected artifacts"
            )
        by_path = {record.path: record for record in self.artifacts}
        for dataset, expected_rows in receipt.expected_rows.items():
            record = by_path[_DATASET_PATHS[dataset]]
            if record.rows != expected_rows:
                raise M6ResultStoreIntegrityError(
                    f"{dataset} rows differ from eligibility receipt"
                )

    def _require_receipt(self) -> M6EligibilityReceipt:
        if self._receipt is None:
            raise M6ResultStoreStateError(
                "eligibility receipt must precede this artifact"
            )
        return self._receipt

    def _require_waymax_selection_receipt(
        self,
    ) -> M6WaymaxSelectionReceipt:
        if self._waymax_selection_receipt is None:
            raise M6ResultStoreStateError(
                "Waymax qualification/selection receipt must precede outcomes"
            )
        return self._waymax_selection_receipt

    def _assert_complete_pending(self) -> None:
        self._assert_pending_capability()
        if not self.profile.complete_results:
            raise M6ResultStoreStateError(
                "complete-result artifact is unavailable in this mode"
            )
        self._require_receipt()

    def _assert_pending_capability(self) -> None:
        if self._phase != "pending" or self._capability_nonce is None:
            raise M6ResultStoreStateError("writer is not pending")
        if hashlib.sha256(self._capability_nonce).hexdigest() != (
            self._capability_sha256
        ):
            raise M6ResultStoreStateError("writer capability no longer matches")
        _guard_run_directory(self.run_path)
        _validate_marker_exclusivity(self.run_path)
        if any(
            _path_kind(self.run_path / name) != "missing"
            for name in (
                COMMITTED_MARKER,
                TERMINAL_SUCCESS_MARKER,
                TERMINAL_FAILURE_MARKER,
            )
        ):
            raise M6ResultStoreStateError(
                "pending writer has an unexpected later-state marker"
            )
        expected = _canonical_json_bytes(
            _pending_payload(
                self.run_name,
                self.profile,
                self._capability_sha256,
            )
        )
        if not _guarded_exact_bytes(
            self.run_path / PENDING_MARKER,
            expected,
            self.run_path,
        ):
            raise M6ResultStoreStateError("PENDING marker/capability is not exact")

    def _assert_committed_capability(self) -> None:
        if self._phase != "committed" or self._capability_nonce is None:
            raise M6ResultStoreStateError(
                "terminal success requires this writer's committed capability"
            )
        if hashlib.sha256(self._capability_nonce).hexdigest() != (
            self._capability_sha256
        ):
            raise M6ResultStoreStateError("committed capability no longer matches")
        _validate_marker_exclusivity(self.run_path)
        if (
            _path_kind(self.run_path / COMMITTED_MARKER) != "file"
            or _path_kind(self.run_path / TERMINAL_FAILURE_MARKER) != "missing"
            or _path_kind(self.run_path / TERMINAL_SUCCESS_MARKER) != "missing"
        ):
            raise M6ResultStoreStateError(
                "committed marker state is not exact"
            )
        expected_pending = _canonical_json_bytes(
            _pending_payload(
                self.run_name,
                self.profile,
                self._capability_sha256,
            )
        )
        if not _guarded_exact_bytes(
            self.run_path / PENDING_MARKER,
            expected_pending,
            self.run_path,
        ):
            raise M6ResultStoreStateError("PENDING capability binding drifted")

    def _assert_not_successful(self) -> None:
        if self._phase == "success" or _path_kind(
            self.run_path / TERMINAL_SUCCESS_MARKER
        ) != "missing":
            raise M6ResultStoreStateError("successful run is immutable")

    def _poison(self, reason_code: str) -> Path:
        if type(reason_code) is not str or _REASON_CODE.fullmatch(reason_code) is None:
            reason_code = "result_store_failure"
        failure_path = self.run_path / TERMINAL_FAILURE_MARKER
        if (
            self._phase == "failure"
            and _path_kind(failure_path) == "file"
        ):
            # Preserve the first exact failure reason.  Follow-up API calls on a
            # poisoned capability remain errors but must not mask those original
            # errors by attempting a contradictory second terminal transition.
            return failure_path
        if _path_kind(self.run_path) != "directory":
            self._phase = "failure"
            self._capability_nonce = None
            return failure_path
        if _path_kind(self.run_path / TERMINAL_SUCCESS_MARKER) != "missing":
            raise M6ResultStoreStateError(
                "TERMINAL_FAILURE cannot follow TERMINAL_SUCCESS"
            )
        payload = _canonical_json_bytes(
            {
                "mode": self.profile.mode,
                "reason_code": reason_code,
                "schema_version": M6_RESULT_STORE_SCHEMA_VERSION,
                "state": "TERMINAL_FAILURE",
            }
        )
        if _path_kind(failure_path) == "missing":
            try:
                _write_bytes_exclusive(failure_path, payload, self.run_path)
            except OSError:
                if not _guarded_exact_bytes(
                    failure_path,
                    payload,
                    self.run_path,
                ):
                    raise
        elif not _guarded_exact_bytes(failure_path, payload, self.run_path):
            raise M6ResultStoreIntegrityError(
                "existing TERMINAL_FAILURE is contradictory"
            )
        self._phase = "failure"
        self._capability_nonce = None
        return failure_path


def _make_m6_observed_preflight_result(
    store: M6ResultStore,
    *,
    checks: Mapping[str, bool],
    evidence_catalog_sha256: str,
) -> M6ObservedPreflightResult:
    """Disabled: asserted booleans cannot substitute for an official verifier."""

    del checks, evidence_catalog_sha256
    if not isinstance(store, M6ResultStore):
        raise TypeError("store must be an M6ResultStore")
    raise M6ResultStoreStateError(
        "asserted preflight booleans are disabled until the official M6 "
        "verifier/CLI exists"
    )


def _mint_m6_terminal_capability(
    store: M6ResultStore,
    observed: M6ObservedPreflightResult,
) -> M6TerminalCapability:
    """Disabled until the official verifier can issue observed capabilities."""

    del observed
    if not isinstance(store, M6ResultStore):
        raise TypeError("store must be an M6ResultStore")
    raise M6ResultStoreStateError(
        "terminal capability minting is disabled until the official M6 "
        "verifier/CLI exists"
    )


def _normalize_eligibility(
    rows: Iterable[Mapping[str, Any]],
    profile: M6ResultProfile,
    *,
    expected_receipt: M6EligibilityReceipt | None = None,
) -> tuple[dict[str, Any], ...]:
    fields = tuple(field.name for field in ELIGIBILITY_LEDGER_SCHEMA)
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in rows:
        row = _exact_row(raw, fields, ELIGIBILITY_LEDGER)
        index = _integer(
            row["cohort_index"],
            name="cohort_index",
            minimum=0,
            maximum=profile.population_size - 1,
        )
        if index in seen:
            raise M6ResultStoreIntegrityError(
                "eligibility ledger has duplicate cohort_index"
            )
        seen.add(index)
        eligible = _boolean(row["primary_eligible"], "primary_eligible")
        reason = row["rejection_reason"]
        secondary = row["secondary_b4_feasible"]
        if eligible:
            if reason is not None:
                raise M6ResultStoreIntegrityError(
                    "eligible row cannot have a rejection reason"
                )
            if profile.mode == ELIGIBILITY_ONLY_MODE:
                if secondary is not None:
                    raise M6ResultStoreIntegrityError(
                        "eligibility_only requires null b4 status"
                    )
            else:
                secondary = _boolean(
                    secondary,
                    "secondary_b4_feasible",
                )
        else:
            if reason not in M6_PRIMARY_REJECTION_REASONS:
                raise M6ResultStoreIntegrityError(
                    "rejected row has an unregistered reason"
                )
            if secondary is not None:
                raise M6ResultStoreIntegrityError(
                    "rejected row cannot retain b4 status"
                )
        normalized.append(
            {
                "cohort_index": index,
                "primary_eligible": eligible,
                "rejection_reason": reason,
                "secondary_b4_feasible": secondary,
            }
        )
    if seen != set(range(profile.population_size)):
        raise M6ResultStoreIntegrityError(
            "eligibility ledger must contain every cohort index exactly once"
        )
    normalized.sort(key=lambda row: row["cohort_index"])
    if profile.mode == DATA_FREE_MODE and (
        any(not row["primary_eligible"] for row in normalized)
        or any(row["secondary_b4_feasible"] is not True for row in normalized)
    ):
        raise M6ResultStoreIntegrityError(
            "data_free ledger must have exact primary/secondary N=10"
        )
    if expected_receipt is not None:
        eligible = tuple(
            row["cohort_index"]
            for row in normalized
            if row["primary_eligible"]
        )
        secondary = tuple(
            row["cohort_index"]
            for row in normalized
            if row["secondary_b4_feasible"] is True
        )
        reasons = {
            reason: sum(
                row["rejection_reason"] == reason for row in normalized
            )
            for reason in M6_PRIMARY_REJECTION_REASONS
        }
        if (
            eligible != expected_receipt.eligible_cohort_indices
            or secondary != expected_receipt.secondary_b4_cohort_indices
            or reasons != dict(expected_receipt.rejection_reason_counts)
        ):
            raise M6ResultStoreIntegrityError(
                "eligibility ledger differs from its frozen receipt"
            )
    return tuple(normalized)


def _normalize_compute_pilot(
    rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
) -> tuple[dict[str, Any], ...]:
    fields = tuple(field.name for field in COMPUTE_PILOT_SUMMARY_SCHEMA)
    raw_rows = tuple(rows)
    if len(raw_rows) != 1:
        raise M6ResultStoreIntegrityError(
            "compute pilot requires exactly one aggregate row"
        )
    row = _exact_row(raw_rows[0], fields, COMPUTE_PILOT_SUMMARY)
    pilot_n = _integer(
        row["pilot_scene_n"],
        name="pilot_scene_n",
        minimum=1,
        maximum=min(8, receipt.eligible_count),
    )
    durations = {
        name: _integer(row[name], name=name, minimum=1)
        for name in (
            "total_wall_ms",
            "max_scene_ms",
            "decode_ms",
            "numpy_ms",
            "waymax_ms",
            "verification_ms",
        )
    }
    if durations["total_wall_ms"] < sum(
        durations[name]
        for name in ("decode_ms", "numpy_ms", "waymax_ms", "verification_ms")
    ):
        raise M6ResultStoreIntegrityError(
            "pilot stage durations exceed total wall time"
        )
    if durations["max_scene_ms"] > durations["total_wall_ms"]:
        raise M6ResultStoreIntegrityError(
            "pilot max scene time exceeds total wall time"
        )
    rss = _integer(
        row["fresh_worker_peak_rss_bytes"],
        name="fresh_worker_peak_rss_bytes",
        minimum=1,
    )
    expected_passed = (
        durations["total_wall_ms"] <= 30 * 60 * 1000
        and durations["max_scene_ms"] <= 10 * 60 * 1000
        and rss <= 16 * 1024**3
    )
    passed = _boolean(row["passed"], "passed")
    if passed != expected_passed:
        raise M6ResultStoreIntegrityError(
            "pilot passed flag disagrees with frozen thresholds"
        )
    return (
        {
            "pilot_scene_n": pilot_n,
            **durations,
            "fresh_worker_peak_rss_bytes": rss,
            "passed": passed,
        },
    )


def _normalize_primary_scene_scalars(
    rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
) -> tuple[dict[str, Any], ...]:
    return _normalize_scene_scalars(
        rows,
        receipt,
        dataset=PRIMARY_SCENE_SCALARS,
        cohort_indices=receipt.eligible_cohort_indices,
        fingerprint=receipt.primary_intervention_fingerprint,
    )


def _normalize_secondary_scene_scalars(
    rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
) -> tuple[dict[str, Any], ...]:
    return _normalize_scene_scalars(
        rows,
        receipt,
        dataset=SECONDARY_SCENE_SCALARS,
        cohort_indices=receipt.secondary_b4_cohort_indices,
        fingerprint=receipt.secondary_intervention_fingerprint,
    )


def _normalize_scene_scalars(
    rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
    *,
    dataset: str,
    cohort_indices: tuple[int, ...],
    fingerprint: str,
) -> tuple[dict[str, Any], ...]:
    schema = M6_RESULT_SCHEMAS[dataset]
    fields = tuple(field.name for field in schema)
    expected = tuple(
        (index, *cell)
        for index in cohort_indices
        for cell in M6_PRIMARY_CELL_DOMAIN
    )
    order = {key: position for position, key in enumerate(expected)}
    units = {metric: unit for metric, _version, unit in M6_PRIMARY_METRICS}
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for raw in rows:
        row = _exact_row(raw, fields, dataset)
        key = (
            _integer(row["cohort_index"], name="cohort_index", minimum=0),
            _text(row["policy_name"], "policy_name"),
            _text(row["policy_access_role"], "policy_access_role"),
            _text(row["metric_name"], "metric_name"),
            _text(row["metric_version"], "metric_version"),
        )
        if key not in order or key in seen:
            raise M6ResultStoreIntegrityError(
                f"{dataset} has a duplicate or unexpected key"
            )
        seen.add(key)
        metric = key[3]
        unit = _text(row["unit"], "unit")
        if unit != units[metric]:
            raise M6ResultStoreIntegrityError(f"{dataset} unit drifted")
        row_fingerprint = _text(
            row["intervention_config_fingerprint"],
            "intervention_config_fingerprint",
        )
        if row_fingerprint != fingerprint:
            raise M6ResultStoreIntegrityError(
                f"{dataset} intervention fingerprint drifted"
            )
        value = _finite(row["value"], "value")
        responded = row["responded"]
        latency = row["responder_latency_s"]
        if metric == "response_timeliness_s":
            responded = _boolean(responded, "responded")
            if responded:
                latency = _finite(
                    latency,
                    "responder_latency_s",
                    minimum=0.0,
                )
            else:
                if latency is not None or value != 0.0:
                    raise M6ResultStoreIntegrityError(
                        "censored timeliness must have null latency and exact zero"
                    )
        elif responded is not None or latency is not None:
            raise M6ResultStoreIntegrityError(
                "response metadata is timeliness-only"
            )
        pairing = _boolean(
            row["source_pairing_complete"],
            "source_pairing_complete",
        )
        if not pairing:
            raise M6ResultStoreIntegrityError(
                "incomplete pair must fail instead of producing a scalar"
            )
        normalized.append(
            {
                "cohort_index": key[0],
                "policy_name": key[1],
                "policy_access_role": key[2],
                "metric_name": metric,
                "metric_version": key[4],
                "unit": unit,
                "value": value,
                "responded": responded,
                "responder_latency_s": latency,
                "source_pairing_complete": pairing,
                "intervention_config_fingerprint": row_fingerprint,
            }
        )
    if seen != set(expected):
        raise M6ResultStoreIntegrityError(
            f"{dataset} is missing registered scene rows"
        )
    normalized.sort(
        key=lambda row: order[
            (
                row["cohort_index"],
                row["policy_name"],
                row["policy_access_role"],
                row["metric_name"],
                row["metric_version"],
            )
        ]
    )
    return tuple(normalized)


def _derive_primary_matrix_rows(
    scene_rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
) -> tuple[dict[str, Any], ...]:
    normalized = _normalize_primary_scene_scalars(scene_rows, receipt)
    by_cell: dict[tuple[str, str, str, str], list[M6SceneEffect]] = {
        cell: [] for cell in M6_PRIMARY_CELL_DOMAIN
    }
    for row in normalized:
        cell = (
            row["policy_name"],
            row["policy_access_role"],
            row["metric_name"],
            row["metric_version"],
        )
        by_cell[cell].append(
            M6SceneEffect(
                cohort_index=row["cohort_index"],
                value=row["value"],
                responded=row["responded"],
                responder_latency_s=row["responder_latency_s"],
            )
        )
    inputs = tuple(
        M6PrimaryCellInput(
            spec=M6PrimaryCellSpec(
                policy_name=policy,
                policy_access_role=access,
                metric_name=metric,
                metric_version=version,
                intervention_config_fingerprint=(
                    receipt.primary_intervention_fingerprint
                ),
            ),
            scene_effects=tuple(by_cell[(policy, access, metric, version)]),
            source_pairing_complete=True,
        )
        for policy, access, metric, version in M6_PRIMARY_CELL_DOMAIN
    )
    result = analyze_m6_primary_matrix(inputs)
    return tuple(_primary_matrix_row_from_result(row) for row in result.rows)


def _primary_matrix_row_from_result(
    result: M6PrimaryCellResult,
) -> dict[str, Any]:
    local = result.to_local_dict()
    conditional = local["conditional_responder_latency"]
    resampling = local["resampling"]
    if not isinstance(resampling, Mapping):
        raise M6ResultStoreIntegrityError("stats.m6 resampling payload drifted")
    return {
        "metric_name": result.spec.metric_name,
        "metric_version": result.spec.metric_version,
        "unit": result.spec.value_unit,
        "policy_name": result.spec.policy_name,
        "policy_access_role": result.spec.policy_access_role,
        "intervention_config_fingerprint": (
            result.spec.intervention_config_fingerprint
        ),
        "pair_n": result.pair_n,
        "thresholded_nonzero_n": result.thresholded_nonzero_n,
        "responder_n": result.responder_n,
        "censor_n": result.censor_n,
        "arithmetic_mean": result.arithmetic_mean,
        "median": result.median,
        "pointwise_level": result.pointwise_band.level,
        "pointwise_lower": result.pointwise_band.lower,
        "pointwise_upper": result.pointwise_band.upper,
        "adjusted_level": result.adjusted_band.level,
        "adjusted_lower": result.adjusted_band.lower,
        "adjusted_upper": result.adjusted_band.upper,
        "status": result.status,
        "suppression_reason": result.suppression_reason,
        "source_pairing_complete": result.source_pairing_complete,
        "directional_language_allowed": (
            result.directional_language_allowed
        ),
        "directional_effect_sign": result.directional_effect_sign,
        "conditional_latency_status": (
            None if conditional is None else conditional["status"]
        ),
        "conditional_latency_suppression_reason": (
            None if conditional is None else conditional["suppression_reason"]
        ),
        "conditional_latency_mean_s": (
            None if conditional is None else conditional["arithmetic_mean_s"]
        ),
        "conditional_latency_median_s": (
            None if conditional is None else conditional["median_s"]
        ),
        "resampling_key_json": resampling["canonical_cell_key"],
        "resampling_sha256": resampling["sha256"],
        "resampling_digest_words": list(resampling["digest_words"]),
        "resamples": resampling["resamples"],
        "base_seed": resampling["base_seed"],
        "rng": resampling["rng"],
        "index_dtype": resampling["index_dtype"],
        "quantile_method": resampling["quantile_method"],
        "interpretation": resampling["interpretation"],
        "statistics_schema_version": resampling[
            "statistics_schema_version"
        ],
    }


def _normalize_primary_matrix(
    rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
) -> tuple[dict[str, Any], ...]:
    fields = tuple(field.name for field in PRIMARY_MATRIX_SCHEMA)
    order = {key: index for index, key in enumerate(M6_PRIMARY_CELL_DOMAIN)}
    units = {metric: unit for metric, _version, unit in M6_PRIMARY_METRICS}
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for raw in rows:
        row = _exact_row(raw, fields, PRIMARY_MATRIX)
        key = (
            _text(row["policy_name"], "policy_name"),
            _text(row["policy_access_role"], "policy_access_role"),
            _text(row["metric_name"], "metric_name"),
            _text(row["metric_version"], "metric_version"),
        )
        if key not in order or key in seen:
            raise M6ResultStoreIntegrityError(
                "primary matrix cell is duplicate or unexpected"
            )
        seen.add(key)
        pair_n = _integer(row["pair_n"], name="pair_n", minimum=10)
        if pair_n != receipt.eligible_count:
            raise M6ResultStoreIntegrityError(
                "primary matrix pair N differs from eligibility"
            )
        nonzero = _integer(
            row["thresholded_nonzero_n"],
            name="thresholded_nonzero_n",
            minimum=0,
            maximum=pair_n,
        )
        responder = row["responder_n"]
        censor = row["censor_n"]
        timeliness = key[2] == "response_timeliness_s"
        if timeliness:
            responder = _integer(
                responder,
                name="responder_n",
                minimum=0,
                maximum=pair_n,
            )
            censor = _integer(
                censor,
                name="censor_n",
                minimum=0,
                maximum=pair_n,
            )
            if responder + censor != pair_n:
                raise M6ResultStoreIntegrityError(
                    "primary responder/censor counts do not sum to N"
                )
        elif responder is not None or censor is not None:
            raise M6ResultStoreIntegrityError(
                "non-timeliness primary cell has response counts"
            )
        status = _text(row["status"], "status")
        if status not in _ALLOWED_PRIMARY_STATUSES:
            raise M6ResultStoreIntegrityError(
                "primary matrix status is unregistered"
            )
        suppression = _optional_text(
            row["suppression_reason"],
            "suppression_reason",
        )
        directional = _boolean(
            row["directional_language_allowed"],
            "directional_language_allowed",
        )
        sign = _optional_text(
            row["directional_effect_sign"],
            "directional_effect_sign",
        )
        if sign not in {None, "positive", "negative"}:
            raise M6ResultStoreIntegrityError(
                "primary directional sign is unregistered"
            )
        pairing = _boolean(
            row["source_pairing_complete"],
            "source_pairing_complete",
        )
        if not pairing:
            raise M6ResultStoreIntegrityError(
                "incomplete source pairing cannot produce a primary cell"
            )
        point_level = _finite(row["pointwise_level"], "pointwise_level")
        adjusted_level = _finite(row["adjusted_level"], "adjusted_level")
        if (
            point_level != M6_POINTWISE_REWEIGHTING_LEVEL
            or adjusted_level != M6_ADJUSTED_REWEIGHTING_LEVEL
        ):
            raise M6ResultStoreIntegrityError(
                "primary matrix band levels drifted"
            )
        point_lower = _finite(row["pointwise_lower"], "pointwise_lower")
        point_upper = _finite(row["pointwise_upper"], "pointwise_upper")
        adjusted_lower = _finite(row["adjusted_lower"], "adjusted_lower")
        adjusted_upper = _finite(row["adjusted_upper"], "adjusted_upper")
        if point_lower > point_upper or adjusted_lower > adjusted_upper:
            raise M6ResultStoreIntegrityError("primary band endpoints reversed")
        expected_status, expected_suppression = _primary_status(
            pair_n,
            nonzero,
            adjusted_lower,
            adjusted_upper,
        )
        if status != expected_status or suppression != expected_suppression:
            raise M6ResultStoreIntegrityError(
                "primary status/suppression priority drifted"
            )
        if directional != (status == "direction_supported"):
            raise M6ResultStoreIntegrityError(
                "primary directional flag drifted"
            )
        expected_sign = None
        if directional:
            expected_sign = "positive" if adjusted_lower > 0.0 else "negative"
        if sign != expected_sign:
            raise M6ResultStoreIntegrityError("primary direction sign drifted")
        conditional_status = _optional_text(
            row["conditional_latency_status"],
            "conditional_latency_status",
        )
        conditional_suppression = _optional_text(
            row["conditional_latency_suppression_reason"],
            "conditional_latency_suppression_reason",
        )
        conditional_mean = _optional_finite(
            row["conditional_latency_mean_s"],
            "conditional_latency_mean_s",
            minimum=0.0,
        )
        conditional_median = _optional_finite(
            row["conditional_latency_median_s"],
            "conditional_latency_median_s",
            minimum=0.0,
        )
        if timeliness:
            if responder < 10:
                expected_conditional = (
                    "responder_sparse",
                    "responder_n_below_10",
                    None,
                    None,
                )
            else:
                expected_conditional = (
                    "descriptive",
                    None,
                    conditional_mean,
                    conditional_median,
                )
                if conditional_mean is None or conditional_median is None:
                    raise M6ResultStoreIntegrityError(
                        "ten responders require conditional latency values"
                    )
            if (
                conditional_status,
                conditional_suppression,
                conditional_mean,
                conditional_median,
            ) != expected_conditional:
                raise M6ResultStoreIntegrityError(
                    "conditional responder-latency fields drifted"
                )
        elif any(
            value is not None
            for value in (
                conditional_status,
                conditional_suppression,
                conditional_mean,
                conditional_median,
            )
        ):
            raise M6ResultStoreIntegrityError(
                "conditional latency is timeliness-only"
            )
        fingerprint = _text(
            row["intervention_config_fingerprint"],
            "intervention_config_fingerprint",
        )
        if fingerprint != receipt.primary_intervention_fingerprint:
            raise M6ResultStoreIntegrityError(
                "primary matrix intervention fingerprint drifted"
            )
        resampling_json = _text(
            row["resampling_key_json"],
            "resampling_key_json",
        )
        resampling_sha = _text(
            row["resampling_sha256"],
            "resampling_sha256",
        )
        if _SHA256.fullmatch(resampling_sha) is None or (
            hashlib.sha256(resampling_json.encode("utf-8")).hexdigest()
            != resampling_sha
        ):
            raise M6ResultStoreIntegrityError(
                "primary resampling key digest drifted"
            )
        words_raw = row["resampling_digest_words"]
        if type(words_raw) is not list or len(words_raw) != 8:
            raise M6ResultStoreIntegrityError(
                "primary resampling words must have exact length eight"
            )
        words = [
            _integer(word, name="resampling word", minimum=0, maximum=2**32 - 1)
            for word in words_raw
        ]
        expected_words = list(
            np.frombuffer(
                bytes.fromhex(resampling_sha),
                dtype=">u4",
            ).astype(np.uint32)
        )
        if [int(word) for word in expected_words] != words:
            raise M6ResultStoreIntegrityError(
                "primary resampling digest words drifted"
            )
        if (
            _integer(row["resamples"], name="resamples", minimum=1)
            != M6_PRIMARY_RESAMPLES
            or _integer(row["base_seed"], name="base_seed", minimum=1)
            != M6_BASE_SEED
            or row["rng"] != "PCG64"
            or row["index_dtype"] != "int64"
            or row["quantile_method"] != "linear"
            or row["interpretation"] != M6_REWEIGHTING_INTERPRETATION
            or row["statistics_schema_version"]
            != M6_STATISTICS_SCHEMA_VERSION
        ):
            raise M6ResultStoreIntegrityError(
                "primary resampling metadata drifted"
            )
        if row["unit"] != units[key[2]]:
            raise M6ResultStoreIntegrityError("primary matrix unit drifted")
        normalized.append(
            {
                "metric_name": key[2],
                "metric_version": key[3],
                "unit": row["unit"],
                "policy_name": key[0],
                "policy_access_role": key[1],
                "intervention_config_fingerprint": fingerprint,
                "pair_n": pair_n,
                "thresholded_nonzero_n": nonzero,
                "responder_n": responder,
                "censor_n": censor,
                "arithmetic_mean": _finite(
                    row["arithmetic_mean"],
                    "arithmetic_mean",
                ),
                "median": _finite(row["median"], "median"),
                "pointwise_level": point_level,
                "pointwise_lower": point_lower,
                "pointwise_upper": point_upper,
                "adjusted_level": adjusted_level,
                "adjusted_lower": adjusted_lower,
                "adjusted_upper": adjusted_upper,
                "status": status,
                "suppression_reason": suppression,
                "source_pairing_complete": pairing,
                "directional_language_allowed": directional,
                "directional_effect_sign": sign,
                "conditional_latency_status": conditional_status,
                "conditional_latency_suppression_reason": (
                    conditional_suppression
                ),
                "conditional_latency_mean_s": conditional_mean,
                "conditional_latency_median_s": conditional_median,
                "resampling_key_json": resampling_json,
                "resampling_sha256": resampling_sha,
                "resampling_digest_words": words,
                "resamples": M6_PRIMARY_RESAMPLES,
                "base_seed": M6_BASE_SEED,
                "rng": "PCG64",
                "index_dtype": "int64",
                "quantile_method": "linear",
                "interpretation": M6_REWEIGHTING_INTERPRETATION,
                "statistics_schema_version": M6_STATISTICS_SCHEMA_VERSION,
            }
        )
    if seen != set(M6_PRIMARY_CELL_DOMAIN):
        raise M6ResultStoreIntegrityError(
            "primary matrix must contain exact 12-cell domain"
        )
    normalized.sort(
        key=lambda row: order[
            (
                row["policy_name"],
                row["policy_access_role"],
                row["metric_name"],
                row["metric_version"],
            )
        ]
    )
    return tuple(normalized)


def _derive_secondary_matrix_rows(
    scene_rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
) -> tuple[dict[str, Any], ...]:
    rows = _normalize_secondary_scene_scalars(scene_rows, receipt)
    by_cell: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {
        cell: [] for cell in M6_PRIMARY_CELL_DOMAIN
    }
    metric_specs = {
        metric.metric_name: metric
        for metric in (
            M6PrimaryCellSpec(
                policy_name="log_replay",
                policy_access_role="privileged",
                metric_name=name,
                metric_version=version,
                intervention_config_fingerprint=(
                    receipt.secondary_intervention_fingerprint
                ),
            ).metric
            for name, version, _unit in M6_PRIMARY_METRICS
        )
    }
    for row in rows:
        key = (
            row["policy_name"],
            row["policy_access_role"],
            row["metric_name"],
            row["metric_version"],
        )
        by_cell[key].append(row)
    units = {metric: unit for metric, _version, unit in M6_PRIMARY_METRICS}
    results: list[dict[str, Any]] = []
    for key in M6_PRIMARY_CELL_DOMAIN:
        cell_rows = by_cell[key]
        n = len(cell_rows)
        values = [float(row["value"]) for row in cell_rows]
        nonzero = sum(
            metric_specs[key[2]].is_thresholded_nonzero(value)
            for value in values
        )
        responder: int | None = None
        censor: int | None = None
        if key[2] == "response_timeliness_s":
            responder = sum(row["responded"] is True for row in cell_rows)
            censor = n - responder
        if n:
            ordered = sorted(values)
            mean: float | None = math.fsum(values) / n
            middle = n // 2
            median: float | None = (
                ordered[middle]
                if n % 2
                else math.fsum(
                    (ordered[middle - 1], ordered[middle])
                )
                / 2.0
            )
            status = "descriptive"
            suppression = None
        else:
            mean = None
            median = None
            status = "empty"
            suppression = "secondary_b4_n_zero"
        results.append(
            {
                "metric_name": key[2],
                "metric_version": key[3],
                "unit": units[key[2]],
                "policy_name": key[0],
                "policy_access_role": key[1],
                "intervention_config_fingerprint": (
                    receipt.secondary_intervention_fingerprint
                ),
                "pair_n": n,
                "thresholded_nonzero_n": nonzero,
                "responder_n": responder,
                "censor_n": censor,
                "arithmetic_mean": mean,
                "median": median,
                "status": status,
                "suppression_reason": suppression,
                "source_pairing_complete": True,
            }
        )
    return tuple(results)


def _normalize_secondary_matrix(
    rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
) -> tuple[dict[str, Any], ...]:
    fields = tuple(field.name for field in SECONDARY_MATRIX_SCHEMA)
    order = {key: index for index, key in enumerate(M6_PRIMARY_CELL_DOMAIN)}
    units = {metric: unit for metric, _version, unit in M6_PRIMARY_METRICS}
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for raw in rows:
        row = _exact_row(raw, fields, SECONDARY_MATRIX)
        key = (
            _text(row["policy_name"], "policy_name"),
            _text(row["policy_access_role"], "policy_access_role"),
            _text(row["metric_name"], "metric_name"),
            _text(row["metric_version"], "metric_version"),
        )
        if key not in order or key in seen:
            raise M6ResultStoreIntegrityError(
                "secondary matrix cell is duplicate or unexpected"
            )
        seen.add(key)
        n = _integer(row["pair_n"], name="pair_n", minimum=0)
        if n != receipt.secondary_b4_count:
            raise M6ResultStoreIntegrityError(
                "secondary matrix N differs from frozen b4 subset"
            )
        nonzero = _integer(
            row["thresholded_nonzero_n"],
            name="thresholded_nonzero_n",
            minimum=0,
            maximum=n,
        )
        responder = row["responder_n"]
        censor = row["censor_n"]
        if key[2] == "response_timeliness_s":
            responder = _integer(
                responder,
                name="responder_n",
                minimum=0,
                maximum=n,
            )
            censor = _integer(
                censor,
                name="censor_n",
                minimum=0,
                maximum=n,
            )
            if responder + censor != n:
                raise M6ResultStoreIntegrityError(
                    "secondary responder/censor counts drifted"
                )
        elif responder is not None or censor is not None:
            raise M6ResultStoreIntegrityError(
                "secondary response counts are timeliness-only"
            )
        mean = _optional_finite(row["arithmetic_mean"], "arithmetic_mean")
        median = _optional_finite(row["median"], "median")
        status = _text(row["status"], "status")
        suppression = _optional_text(
            row["suppression_reason"],
            "suppression_reason",
        )
        expected = (
            ("empty", "secondary_b4_n_zero", None, None)
            if n == 0
            else ("descriptive", None, mean, median)
        )
        if n > 0 and (mean is None or median is None):
            raise M6ResultStoreIntegrityError(
                "nonempty secondary cell requires mean and median"
            )
        if (status, suppression, mean, median) != expected:
            raise M6ResultStoreIntegrityError(
                "secondary matrix status/value suppression drifted"
            )
        fingerprint = _text(
            row["intervention_config_fingerprint"],
            "intervention_config_fingerprint",
        )
        if fingerprint != receipt.secondary_intervention_fingerprint:
            raise M6ResultStoreIntegrityError(
                "secondary intervention fingerprint drifted"
            )
        pairing = _boolean(
            row["source_pairing_complete"],
            "source_pairing_complete",
        )
        if not pairing or row["unit"] != units[key[2]]:
            raise M6ResultStoreIntegrityError(
                "secondary pairing/unit drifted"
            )
        normalized.append(
            {
                "metric_name": key[2],
                "metric_version": key[3],
                "unit": row["unit"],
                "policy_name": key[0],
                "policy_access_role": key[1],
                "intervention_config_fingerprint": fingerprint,
                "pair_n": n,
                "thresholded_nonzero_n": nonzero,
                "responder_n": responder,
                "censor_n": censor,
                "arithmetic_mean": mean,
                "median": median,
                "status": status,
                "suppression_reason": suppression,
                "source_pairing_complete": pairing,
            }
        )
    if seen != set(M6_PRIMARY_CELL_DOMAIN):
        raise M6ResultStoreIntegrityError(
            "secondary matrix must contain exact 12-cell domain"
        )
    normalized.sort(
        key=lambda row: order[
            (
                row["policy_name"],
                row["policy_access_role"],
                row["metric_name"],
                row["metric_version"],
            )
        ]
    )
    return tuple(normalized)


def _negative_timing_observation_domain(
    receipt: M6EligibilityReceipt,
) -> tuple[tuple[str, int, str | None], ...]:
    rows: list[tuple[str, int, str | None]] = []
    for gate_name in M6_NEGATIVE_TIMING_GATE_DOMAIN:
        indices = (
            receipt.secondary_b4_cohort_indices
            if gate_name == "nested_dose_monotonicity"
            else receipt.eligible_cohort_indices
        )
        for cohort_index in indices:
            for policy_name in M6_NEGATIVE_TIMING_OBSERVATION_POLICIES[
                gate_name
            ]:
                rows.append((gate_name, cohort_index, policy_name))
    return tuple(rows)


def _normalize_negative_timing_observations(
    rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
) -> tuple[dict[str, Any], ...]:
    fields = tuple(field.name for field in NEGATIVE_TIMING_OBSERVATIONS_SCHEMA)
    domain = _negative_timing_observation_domain(receipt)
    order = {key: index for index, key in enumerate(domain)}
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str | None]] = set()
    for raw in rows:
        row = _exact_row(raw, fields, NEGATIVE_TIMING_OBSERVATIONS)
        gate_name = _text(row["gate_name"], "gate_name")
        cohort_index = _integer(
            row["cohort_index"],
            name="cohort_index",
            minimum=0,
            maximum=receipt.population_size - 1,
        )
        policy_name = _optional_text(row["policy_name"], "policy_name")
        key = (gate_name, cohort_index, policy_name)
        if key not in order or key in seen:
            raise M6ResultStoreIntegrityError(
                "negative/timing observation is duplicate or outside the exact "
                "per-case/per-policy domain"
            )
        seen.add(key)
        assessed = _integer(row["assessed_n"], name="assessed_n", minimum=0)
        violation = _integer(
            row["violation_n"],
            name="violation_n",
            minimum=0,
            maximum=1,
        )
        if assessed != 1 or violation > assessed:
            raise M6ResultStoreIntegrityError(
                "each negative/timing observation must represent exactly one "
                "assessed case"
            )
        digest = _text(row["observation_sha256"], "observation_sha256")
        if _SHA256.fullmatch(digest) is None:
            raise M6ResultStoreIntegrityError(
                "negative/timing observation digest must be SHA-256"
            )
        normalized.append(
            {
                "gate_name": gate_name,
                "cohort_index": cohort_index,
                "policy_name": policy_name,
                "assessed_n": 1,
                "violation_n": violation,
                "observation_sha256": digest,
            }
        )
    if seen != set(domain):
        raise M6ResultStoreIntegrityError(
            "negative/timing observation domain is incomplete"
        )
    normalized.sort(
        key=lambda row: order[
            (
                row["gate_name"],
                row["cohort_index"],
                row["policy_name"],
            )
        ]
    )
    return tuple(normalized)


def _derive_negative_timing_gates(
    observations: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
) -> tuple[dict[str, Any], ...]:
    rows = _normalize_negative_timing_observations(observations, receipt)
    result: list[dict[str, Any]] = []
    for gate_name in M6_NEGATIVE_TIMING_GATE_DOMAIN:
        members = [row for row in rows if row["gate_name"] == gate_name]
        assessed = len(members)
        violations = sum(int(row["violation_n"]) for row in members)
        status = (
            "unsupported"
            if assessed == 0
            else ("passed" if violations == 0 else "failed")
        )
        evidence = hashlib.sha256(
            b"evalsim-m6-negative-timing-gate-v1\x00"
            + gate_name.encode("ascii")
            + b"\x00"
            + _canonical_json_bytes(
                {
                    "observations": [
                        {
                            "cohort_index": row["cohort_index"],
                            "observation_sha256": row[
                                "observation_sha256"
                            ],
                            "policy_name": row["policy_name"],
                            "violation_n": row["violation_n"],
                        }
                        for row in members
                    ]
                }
            )
        ).hexdigest()
        result.append(
            {
                "gate_name": gate_name,
                "status": status,
                "assessed_n": assessed,
                "passed_n": assessed - violations,
                "violation_n": violations,
                "local_evidence_sha256": evidence,
            }
        )
    return tuple(result)


def _normalize_negative_timing_gates(
    rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
) -> tuple[dict[str, Any], ...]:
    fields = tuple(field.name for field in NEGATIVE_TIMING_GATES_SCHEMA)
    order = {
        name: index
        for index, name in enumerate(M6_NEGATIVE_TIMING_GATE_DOMAIN)
    }
    expected_assessed = {
        "log_replay_world_tensor_equality": receipt.eligible_count,
        "constant_velocity_world_tensor_equality": receipt.eligible_count,
        "sham_legacy_equality": receipt.eligible_count * 3,
        "synchronous_response_floor": receipt.eligible_count * 3,
        "primary_plan_feasibility": receipt.eligible_count,
        "nested_dose_monotonicity": receipt.secondary_b4_count,
    }
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        row = _exact_row(raw, fields, NEGATIVE_TIMING_GATES)
        name = _text(row["gate_name"], "gate_name")
        if name not in order or name in seen:
            raise M6ResultStoreIntegrityError(
                "negative/timing gate is duplicate or unexpected"
            )
        seen.add(name)
        assessed = _integer(row["assessed_n"], name="assessed_n", minimum=0)
        if assessed != expected_assessed[name]:
            raise M6ResultStoreIntegrityError(
                f"{name} assessed domain drifted"
            )
        passed_n = _integer(
            row["passed_n"],
            name="passed_n",
            minimum=0,
            maximum=assessed,
        )
        violations = _integer(
            row["violation_n"],
            name="violation_n",
            minimum=0,
            maximum=assessed,
        )
        if passed_n + violations != assessed:
            raise M6ResultStoreIntegrityError(
                "gate passed/violation counts must partition assessed N"
            )
        status = _text(row["status"], "status")
        if assessed == 0:
            if (
                name != "nested_dose_monotonicity"
                or status != "unsupported"
                or passed_n != 0
                or violations != 0
            ):
                raise M6ResultStoreIntegrityError(
                    "only an empty nested-dose gate may be unsupported"
                )
        elif status not in {"passed", "failed"}:
            raise M6ResultStoreIntegrityError(
                "assessed gate status must be passed or failed"
            )
        elif (status == "passed") != (violations == 0):
            raise M6ResultStoreIntegrityError(
                "gate status disagrees with violations"
            )
        evidence = _text(
            row["local_evidence_sha256"],
            "local_evidence_sha256",
        )
        if _SHA256.fullmatch(evidence) is None:
            raise M6ResultStoreIntegrityError(
                "gate local evidence must be SHA-256"
            )
        normalized.append(
            {
                "gate_name": name,
                "status": status,
                "assessed_n": assessed,
                "passed_n": passed_n,
                "violation_n": violations,
                "local_evidence_sha256": evidence,
            }
        )
    if seen != set(M6_NEGATIVE_TIMING_GATE_DOMAIN):
        raise M6ResultStoreIntegrityError(
            "negative/timing gate domain is incomplete"
        )
    normalized.sort(key=lambda row: order[row["gate_name"]])
    return tuple(normalized)


def _waymax_selection_receipt_from_selection(
    selection: M6WaymaxSelection,
    receipt: M6EligibilityReceipt,
) -> M6WaymaxSelectionReceipt:
    if receipt.mode == DATA_FREE_MODE:
        raise M6ResultStoreIntegrityError(
            "data_free mode cannot accept a live Waymax selection"
        )
    if not isinstance(selection, M6WaymaxSelection):
        raise TypeError(
            "non-data-free qualification requires M6WaymaxSelection"
        )
    selection.revalidate()
    ledger_indices = tuple(
        row.cohort_index for row in selection.qualification_ledger.rows
    )
    if (
        selection.primary_domain_member_count != receipt.eligible_count
        or ledger_indices != receipt.eligible_cohort_indices
    ):
        raise M6ResultStoreIntegrityError(
            "canonical Waymax selection does not cover the exact primary-eligible "
            "cohort domain"
        )
    return M6WaymaxSelectionReceipt(
        mode=receipt.mode,
        status="sealed",
        primary_domain_sha256=selection.primary_domain_sha256,
        primary_domain_member_count=selection.primary_domain_member_count,
        qualification_ledger_sha256=(
            selection.qualification_ledger_sha256
        ),
        selector_selection_sha256=selection.selection_sha256,
        selection_binding_sha256=(
            _m6_waymax_selection_binding_sha256(selection)
        ),
        selection_supported=selection.supported,
        eligible_count=selection.eligible_count,
        selection_member_count=len(selection.members),
        identity_configuration_fingerprint=(
            M6_WAYMAX_IDENTITY_CONFIGURATION_FINGERPRINT
        ),
        primary_b2_configuration_fingerprint=(
            M6_WAYMAX_PRIMARY_B2_CONFIGURATION_FINGERPRINT
        ),
    )


def _data_free_waymax_selection_receipt(
    receipt: M6EligibilityReceipt,
) -> M6WaymaxSelectionReceipt:
    if receipt.mode != DATA_FREE_MODE:
        raise M6ResultStoreIntegrityError(
            "non-applicable selection receipt is data_free-only"
        )
    return M6WaymaxSelectionReceipt(
        mode=receipt.mode,
        status="not_applicable",
        primary_domain_sha256=M6_DATA_FREE_WAYMAX_PRIMARY_DOMAIN_SHA256,
        primary_domain_member_count=0,
        qualification_ledger_sha256=None,
        selector_selection_sha256=None,
        selection_binding_sha256=(
            M6_DATA_FREE_WAYMAX_SELECTION_BINDING_SHA256
        ),
        selection_supported=False,
        eligible_count=0,
        selection_member_count=0,
        identity_configuration_fingerprint=(
            M6_WAYMAX_IDENTITY_CONFIGURATION_FINGERPRINT
        ),
        primary_b2_configuration_fingerprint=(
            M6_WAYMAX_PRIMARY_B2_CONFIGURATION_FINGERPRINT
        ),
    )


def _waymax_qualification_rows_from_selection(
    selection: M6WaymaxSelection,
    receipt: M6EligibilityReceipt,
) -> tuple[dict[str, Any], ...]:
    _waymax_selection_receipt_from_selection(selection, receipt)
    selected_positions = {
        member.cohort_index: position
        for position, member in enumerate(selection.members)
    }
    return tuple(
        {
            "cohort_index": row.cohort_index,
            "assessment_status": (
                "qualified" if row.eligible else "rejected"
            ),
            "rejection_reason": row.reason,
            "rank_sha256": row.rank_sha256,
            "source_binding_sha256": row.source_binding_sha256,
            "primary_entry_sha256": row.primary_entry_sha256,
            "qualification_binding_sha256": (
                row.qualification_binding_sha256
            ),
            "selected": row.cohort_index in selected_positions,
            "selection_position": selected_positions.get(row.cohort_index),
        }
        for row in selection.qualification_ledger.rows
    )


def _verify_waymax_selection_receipt_against_selection(
    sealed: M6WaymaxSelectionReceipt,
    selection: M6WaymaxSelection,
    receipt: M6EligibilityReceipt,
) -> None:
    expected = _waymax_selection_receipt_from_selection(
        selection,
        receipt,
    )
    if sealed.to_dict() != expected.to_dict():
        raise M6ResultStoreIntegrityError(
            "canonical Waymax selection differs from the sealed "
            "qualification/selection receipt"
        )


def _verify_waymax_selection_receipt_against_stored_qualification(
    sealed: M6WaymaxSelectionReceipt,
    qualification: Sequence[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
) -> tuple[tuple[int, int, str], ...]:
    """Verify only the manifest-bound safe projection; never mint live authority."""

    if receipt.mode == DATA_FREE_MODE:
        raise M6ResultStoreIntegrityError(
            "data_free mode has no stored Waymax selection"
        )
    selected = tuple(
        sorted(
            (row for row in qualification if row["selected"] is True),
            key=lambda row: int(row["selection_position"]),
        )
    )
    qualified_count = sum(
        row["assessment_status"] == "qualified" for row in qualification
    )
    if (
        sealed.mode != receipt.mode
        or sealed.status != "sealed"
        or sealed.primary_domain_member_count != receipt.eligible_count
        or sealed.eligible_count != qualified_count
        or sealed.selection_member_count != len(selected)
        or sealed.selection_supported is not (qualified_count >= 8)
        or tuple(int(row["selection_position"]) for row in selected)
        != tuple(range(len(selected)))
    ):
        raise M6ResultStoreIntegrityError(
            "stored qualification ledger differs from its sealed selection "
            "receipt"
        )
    payload = {
        "eligible_count": sealed.eligible_count,
        "qualification_ledger_sha256": (
            sealed.qualification_ledger_sha256
        ),
        "members": [
            {
                "cohort_index": int(row["cohort_index"]),
                "qualification_binding_sha256": (
                    row["qualification_binding_sha256"]
                ),
                "rank_sha256": row["rank_sha256"],
            }
            for row in selected
        ],
        "primary_domain_member_count": (
            sealed.primary_domain_member_count
        ),
        "primary_domain_sha256": sealed.primary_domain_sha256,
        "supported": sealed.selection_supported,
        "selector_selection_sha256": sealed.selector_selection_sha256,
    }
    expected_binding = hashlib.sha256(
        b"evalsim-m6-waymax-selection-binding-v1"
        + b"\x00"
        + _canonical_json_text(payload).encode("ascii")
    ).hexdigest()
    if expected_binding != sealed.selection_binding_sha256:
        raise M6ResultStoreIntegrityError(
            "stored selection binding does not bind its manifest-verified "
            "qualification projection"
        )
    return tuple(
        (
            int(row["selection_position"]),
            int(row["cohort_index"]),
            str(row["qualification_binding_sha256"]),
        )
        for row in selected
    )


def m6_data_free_waymax_qualification_rows(
    eligible_cohort_indices: Sequence[int],
) -> tuple[dict[str, Any], ...]:
    """Build the exact data-free source-qualification placeholder."""

    indices = _ordered_indices(
        eligible_cohort_indices,
        population_size=10,
        name="eligible_cohort_indices",
    )
    return tuple(
        {
            "cohort_index": cohort_index,
            "assessment_status": "not_applicable",
            "rejection_reason": None,
            "rank_sha256": None,
            "source_binding_sha256": None,
            "primary_entry_sha256": None,
            "qualification_binding_sha256": None,
            "selected": False,
            "selection_position": None,
        }
        for cohort_index in indices
    )


def m6_data_free_waymax_scene_scalar_rows() -> tuple[dict[str, Any], ...]:
    """Build the exact fixed 128-row no-Waymax scalar grid."""

    units = {name: unit for name, _version, unit in M6_PRIMARY_METRICS}
    versions = {name: version for name, version, _unit in M6_PRIMARY_METRICS}
    return tuple(
        {
            "selection_position": position,
            "cohort_index": None,
            "qualification_binding_sha256": None,
            "primary_domain_sha256": (
                M6_DATA_FREE_WAYMAX_PRIMARY_DOMAIN_SHA256
            ),
            "selection_binding_sha256": (
                M6_DATA_FREE_WAYMAX_SELECTION_BINDING_SHA256
            ),
            "selection_supported": False,
            "selection_member_count": 0,
            "identity_configuration_fingerprint": (
                M6_WAYMAX_IDENTITY_CONFIGURATION_FINGERPRINT
            ),
            "primary_b2_configuration_fingerprint": (
                M6_WAYMAX_PRIMARY_B2_CONFIGURATION_FINGERPRINT
            ),
            "bundle": bundle,
            "metric_name": metric,
            "metric_version": versions[metric],
            "value_unit": units[metric],
            "value": None,
            "responded": None,
            "responder_latency_s": None,
            "source_pairing_complete": False,
            "status": "not_selected",
        }
        for position in range(M6_WAYMAX_MAX_SELECTED)
        for bundle in M6_WAYMAX_BUNDLES
        for metric, _version, _unit in M6_PRIMARY_METRICS
    )


def m6_data_free_waymax_field_comparison_rows(
) -> tuple[dict[str, Any], ...]:
    """Build the exact fixed 640-row no-Waymax field grid."""

    return tuple(
        {
            "selection_position": position,
            "bundle": bundle,
            "condition": condition,
            "field_name": field_name,
            "cohort_index": None,
            "qualification_binding_sha256": None,
            "comparison_kind": (
                "exact"
                if field_name in M6_WAYMAX_EXACT_FIELDS
                else "tolerance"
            ),
            "denominator": None,
            "max_abs_error": None,
            "max_normalized_error": None,
            "tolerance_failures": None,
            "binary_mismatches": None,
            "status": "not_applicable",
        }
        for position in range(M6_WAYMAX_MAX_SELECTED)
        for bundle in M6_WAYMAX_BUNDLES
        for condition in M6_WAYMAX_CONDITIONS
        for field_name in M6_WAYMAX_COMPARISON_FIELDS
    )


def m6_data_free_waymax_determinism_rows(
) -> tuple[dict[str, Any], ...]:
    """Build the exact fixed 64-row no-Waymax repeat/JIT grid."""

    return tuple(
        {
            "selection_position": position,
            "bundle": bundle,
            "condition": condition,
            "cohort_index": None,
            "qualification_binding_sha256": None,
            "status": "not_applicable",
            "eager_pass_1_sha256": None,
            "eager_pass_2_sha256": None,
            "jit_eager_sha256": None,
            "jit_compiled_sha256": None,
        }
        for position in range(M6_WAYMAX_MAX_SELECTED)
        for bundle in M6_WAYMAX_BUNDLES
        for condition in M6_WAYMAX_CONDITIONS
    )


def _normalize_waymax_qualification(
    rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
) -> tuple[dict[str, Any], ...]:
    fields = tuple(field.name for field in WAYMAX_QUALIFICATION_SCHEMA)
    expected_indices = receipt.eligible_cohort_indices
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in rows:
        row = _exact_row(raw, fields, WAYMAX_QUALIFICATION)
        cohort_index = _integer(
            row["cohort_index"],
            name="cohort_index",
            minimum=0,
            maximum=receipt.population_size - 1,
        )
        if cohort_index not in expected_indices or cohort_index in seen:
            raise M6ResultStoreIntegrityError(
                "Waymax qualification must cover the exact primary-eligible "
                "cohort domain once"
            )
        seen.add(cohort_index)
        assessment = _text(row["assessment_status"], "assessment_status")
        reason = _optional_text(row["rejection_reason"], "rejection_reason")
        rank = _optional_text(row["rank_sha256"], "rank_sha256")
        source = _optional_text(
            row["source_binding_sha256"],
            "source_binding_sha256",
        )
        primary_entry = _optional_text(
            row["primary_entry_sha256"],
            "primary_entry_sha256",
        )
        qualification = _optional_text(
            row["qualification_binding_sha256"],
            "qualification_binding_sha256",
        )
        selected = _boolean(row["selected"], "selected")
        position = (
            None
            if row["selection_position"] is None
            else _integer(
                row["selection_position"],
                name="selection_position",
                minimum=0,
                maximum=M6_WAYMAX_MAX_SELECTED - 1,
            )
        )
        if receipt.mode == DATA_FREE_MODE:
            if (
                assessment != "not_applicable"
                or reason is not None
                or any(
                    value is not None
                    for value in (
                        rank,
                        source,
                        primary_entry,
                        qualification,
                        position,
                    )
                )
                or selected
            ):
                raise M6ResultStoreIntegrityError(
                    "data_free Waymax qualification must be exact "
                    "not_applicable/null/zero evidence"
                )
        else:
            if assessment not in {"qualified", "rejected"}:
                raise M6ResultStoreIntegrityError(
                    "real-data Waymax assessment must be qualified or rejected"
                )
            expected_rank = m6_waymax_rank_sha256(cohort_index)
            if rank != expected_rank:
                raise M6ResultStoreIntegrityError(
                    "Waymax rank differs from the frozen source-only rule"
                )
            for name, value in (
                ("source_binding_sha256", source),
                ("primary_entry_sha256", primary_entry),
                ("qualification_binding_sha256", qualification),
            ):
                if value is None or _SHA256.fullmatch(value) is None:
                    raise M6ResultStoreIntegrityError(
                        f"real-data {name} must be a complete SHA-256 binding"
                    )
            if assessment == "qualified":
                if reason is not None:
                    raise M6ResultStoreIntegrityError(
                        "qualified Waymax rows cannot retain rejection reasons"
                    )
            elif reason not in M6_WAYMAX_QUALIFICATION_REJECTION_REASONS:
                raise M6ResultStoreIntegrityError(
                    "rejected Waymax rows require a registered source-only reason"
                )
        normalized.append(
            {
                "cohort_index": cohort_index,
                "assessment_status": assessment,
                "rejection_reason": reason,
                "rank_sha256": rank,
                "source_binding_sha256": source,
                "primary_entry_sha256": primary_entry,
                "qualification_binding_sha256": qualification,
                "selected": selected,
                "selection_position": position,
            }
        )
    if seen != set(expected_indices):
        raise M6ResultStoreIntegrityError(
            "Waymax qualification ledger is incomplete"
        )
    if receipt.mode != DATA_FREE_MODE:
        for field_name in (
            "source_binding_sha256",
            "primary_entry_sha256",
            "qualification_binding_sha256",
        ):
            values = [str(row[field_name]) for row in normalized]
            if len(values) != len(set(values)):
                raise M6ResultStoreIntegrityError(
                    f"Waymax {field_name} is reused across eligible scenes"
                )
        qualified = sorted(
            (
                row
                for row in normalized
                if row["assessment_status"] == "qualified"
            ),
            key=lambda row: (
                bytes.fromhex(str(row["rank_sha256"])),
                int(row["cohort_index"]),
            ),
        )
        selected_rows = (
            qualified[:M6_WAYMAX_MAX_SELECTED]
            if len(qualified) >= 8
            else []
        )
        selected_positions = {
            int(row["cohort_index"]): position
            for position, row in enumerate(selected_rows)
        }
        for row in normalized:
            expected_position = selected_positions.get(
                int(row["cohort_index"])
            )
            if (
                row["selected"] is not (expected_position is not None)
                or row["selection_position"] != expected_position
            ):
                raise M6ResultStoreIntegrityError(
                    "Waymax selected flag/position differs from the exact "
                    "ranked 16-or-floor rule"
                )
    normalized.sort(key=lambda row: int(row["cohort_index"]))
    return tuple(normalized)


def m6_waymax_unsupported_rows(
    primary_eligible_n: int,
    *,
    rejection_reason: str = "waymax_cadence_mismatch",
) -> tuple[dict[str, Any], ...]:
    """Build a complete bounded unsupported Waymax domain without Waymax import."""

    n = _integer(
        primary_eligible_n,
        name="primary_eligible_n",
        minimum=0,
        maximum=128,
    )
    if rejection_reason not in M6_WAYMAX_REJECTION_REASONS:
        raise ValueError("rejection_reason is not registered")
    units = {metric: unit for metric, _version, unit in M6_PRIMARY_METRICS}
    rows: list[dict[str, Any]] = []
    for record_type, name, bundle, condition, metric in M6_WAYMAX_ROW_DOMAIN:
        row = {field.name: None for field in WAYMAX_ACCOUNTING_SCHEMA}
        row.update(
            {
                "record_type": record_type,
                "name": name,
                "bundle": bundle,
                "condition": condition,
                "metric_name": metric,
                "status": "unsupported",
            }
        )
        if record_type == "scope":
            row["count"] = {
                "qualified_count": 0,
                "selected_count": 0,
                "transition_count": 20,
            }[name]
        elif record_type == "selection_rejection":
            row["count"] = n if name == rejection_reason else 0
            row["opportunity_n"] = n
        elif record_type == "field_comparison":
            row["comparison_kind"] = (
                "exact" if name in M6_WAYMAX_EXACT_FIELDS else "tolerance"
            )
            row["denominator"] = 0
            row["binary_mismatches"] = 0
            if name not in M6_WAYMAX_EXACT_FIELDS:
                row["tolerance_failures"] = 0
        elif record_type == "control_partition":
            row["opportunity_n"] = 0
            row["count"] = 0
        else:
            row.update(
                {
                    "metric_version": "1.0.0",
                    "unit": units[metric],
                    "pair_n": 0,
                    "thresholded_nonzero_n": 0,
                    "responder_n": (
                        0 if metric == "response_timeliness_s" else None
                    ),
                    "censor_n": (
                        0 if metric == "response_timeliness_s" else None
                    ),
                    "suppression_reason": "waymax_selected_n_below_8",
                    "source_pairing_complete": True,
                    "directional_language_allowed": False,
                }
            )
        rows.append(row)
    return tuple(rows)


def m6_data_free_waymax_placeholder_rows(
    primary_eligible_n: int,
) -> tuple[dict[str, Any], ...]:
    """Return a factual no-Waymax placeholder for synthetic data-free evidence."""

    rows = [
        dict(row)
        for row in m6_waymax_unsupported_rows(primary_eligible_n)
    ]
    for row in rows:
        if row["record_type"] == "selection_rejection":
            row["count"] = 0
            row["opportunity_n"] = 0
    return tuple(rows)


def _selected_waymax_by_position(
    qualification_rows: Sequence[Mapping[str, Any]],
) -> Mapping[int, Mapping[str, Any]]:
    selected = {
        int(row["selection_position"]): row
        for row in qualification_rows
        if row["selected"] is True
    }
    if set(selected) != set(range(len(selected))):
        raise M6ResultStoreIntegrityError(
            "Waymax selected positions are not one exact prefix"
        )
    return MappingProxyType(selected)


def _normalize_waymax_scene_scalars_from_qualification(
    rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
    qualification_rows: Iterable[Mapping[str, Any]],
    selection_receipt: M6WaymaxSelectionReceipt,
) -> tuple[dict[str, Any], ...]:
    fields = tuple(field.name for field in WAYMAX_SCENE_SCALARS_SCHEMA)
    qualification = _normalize_waymax_qualification(
        qualification_rows,
        receipt,
    )
    selected = _selected_waymax_by_position(qualification)
    if (
        selection_receipt.mode != receipt.mode
        or len(selected) != selection_receipt.selection_member_count
    ):
        raise M6ResultStoreIntegrityError(
            "Waymax qualification differs from its sealed selection receipt"
        )
    expected_keys = tuple(
        (position, bundle, metric_name)
        for position in range(M6_WAYMAX_MAX_SELECTED)
        for bundle in M6_WAYMAX_BUNDLES
        for metric_name, _version, _unit in M6_PRIMARY_METRICS
    )
    order = {key: index for index, key in enumerate(expected_keys)}
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for raw in rows:
        row = _exact_row(raw, fields, WAYMAX_SCENE_SCALARS)
        try:
            scalar = M6WaymaxSceneScalar.from_store_dict(row)
            scalar.revalidate()
        except (TypeError, ValueError) as exc:
            raise M6ResultStoreIntegrityError(
                "Waymax safe scalar row failed its contract"
            ) from exc
        key = (
            int(scalar.selection_position),
            scalar.bundle,
            scalar.metric_name,
        )
        if key not in order or key in seen:
            raise M6ResultStoreIntegrityError(
                "Waymax scalar is duplicate or outside the fixed 16x2x4 grid"
            )
        seen.add(key)
        selected_row = selected.get(int(scalar.selection_position))
        if (
            scalar.primary_domain_sha256
            != selection_receipt.primary_domain_sha256
            or scalar.selection_binding_sha256
            != selection_receipt.selection_binding_sha256
            or scalar.selection_supported
            is not selection_receipt.selection_supported
            or scalar.selection_member_count
            != selection_receipt.selection_member_count
            or scalar.identity_configuration_fingerprint
            != selection_receipt.identity_configuration_fingerprint
            or scalar.primary_b2_configuration_fingerprint
            != selection_receipt.primary_b2_configuration_fingerprint
        ):
            raise M6ResultStoreIntegrityError(
                "Waymax scalar provenance differs from the sealed "
                "selection/domain/intervention receipt"
            )
        if selected_row is None:
            if scalar.status != "not_selected":
                raise M6ResultStoreIntegrityError(
                    "unused Waymax scalar positions must be exact not_selected NA"
                )
        elif (
            scalar.status != "selected"
            or scalar.cohort_index != selected_row["cohort_index"]
            or scalar.qualification_binding_sha256
            != selected_row["qualification_binding_sha256"]
            or scalar.source_pairing_complete is not True
        ):
            raise M6ResultStoreIntegrityError(
                "selected Waymax scalar does not cross-bind its frozen "
                "qualification position"
            )
        normalized.append(scalar.to_store_dict())
    if seen != set(expected_keys):
        raise M6ResultStoreIntegrityError(
            "Waymax scalar table is not the fixed 128-row grid"
        )
    normalized.sort(
        key=lambda row: order[
            (
                row["selection_position"],
                row["bundle"],
                row["metric_name"],
            )
        ]
    )
    return tuple(normalized)


def _normalize_waymax_field_comparisons_from_qualification(
    rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
    qualification_rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    fields = tuple(field.name for field in WAYMAX_FIELD_COMPARISONS_SCHEMA)
    qualification = _normalize_waymax_qualification(
        qualification_rows,
        receipt,
    )
    selected = _selected_waymax_by_position(qualification)
    expected_keys = tuple(
        (position, bundle, condition, field_name)
        for position in range(M6_WAYMAX_MAX_SELECTED)
        for bundle in M6_WAYMAX_BUNDLES
        for condition in M6_WAYMAX_CONDITIONS
        for field_name in M6_WAYMAX_COMPARISON_FIELDS
    )
    order = {key: index for index, key in enumerate(expected_keys)}
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str, str]] = set()
    for raw in rows:
        row = _exact_row(raw, fields, WAYMAX_FIELD_COMPARISONS)
        position = _integer(
            row["selection_position"],
            name="selection_position",
            minimum=0,
            maximum=M6_WAYMAX_MAX_SELECTED - 1,
        )
        bundle = _text(row["bundle"], "bundle")
        condition = _text(row["condition"], "condition")
        field_name = _text(row["field_name"], "field_name")
        key = (position, bundle, condition, field_name)
        if key not in order or key in seen:
            raise M6ResultStoreIntegrityError(
                "Waymax field comparison is duplicate or outside the fixed "
                "16x2x2x10 grid"
            )
        seen.add(key)
        expected_kind = (
            "exact"
            if field_name in M6_WAYMAX_EXACT_FIELDS
            else "tolerance"
        )
        kind = _text(row["comparison_kind"], "comparison_kind")
        if kind != expected_kind:
            raise M6ResultStoreIntegrityError(
                "Waymax field comparison kind drifted"
            )
        status = _text(row["status"], "status")
        selected_row = selected.get(position)
        cohort_index = (
            None
            if row["cohort_index"] is None
            else _integer(
                row["cohort_index"],
                name="cohort_index",
                minimum=0,
                maximum=receipt.population_size - 1,
            )
        )
        binding = _optional_text(
            row["qualification_binding_sha256"],
            "qualification_binding_sha256",
        )
        numeric_names = (
            "denominator",
            "tolerance_failures",
            "binary_mismatches",
        )
        numeric = {
            name: (
                None
                if row[name] is None
                else _integer(
                    row[name],
                    name=name,
                    minimum=0,
                    maximum=2**63 - 1,
                )
            )
            for name in numeric_names
        }
        max_error = (
            None
            if row["max_abs_error"] is None
            else _finite(
                row["max_abs_error"],
                "max_abs_error",
                minimum=0.0,
            )
        )
        max_normalized = (
            None
            if row["max_normalized_error"] is None
            else _finite(
                row["max_normalized_error"],
                "max_normalized_error",
                minimum=0.0,
            )
        )
        if selected_row is None:
            if (
                status != "not_applicable"
                or cohort_index is not None
                or binding is not None
                or max_error is not None
                or max_normalized is not None
                or any(value is not None for value in numeric.values())
            ):
                raise M6ResultStoreIntegrityError(
                    "unused Waymax comparison positions must be exact NA"
                )
        else:
            if (
                cohort_index != selected_row["cohort_index"]
                or binding
                != selected_row["qualification_binding_sha256"]
            ):
                raise M6ResultStoreIntegrityError(
                    "Waymax comparison does not cross-bind selected "
                    "qualification"
                )
            denominator = numeric["denominator"]
            tolerance_failures = numeric["tolerance_failures"]
            binary_mismatches = numeric["binary_mismatches"]
            if (
                denominator is None
                or denominator < M6_WAYMAX_TRANSITIONS
                or tolerance_failures is None
                or binary_mismatches is None
                or tolerance_failures > denominator
                or binary_mismatches > denominator
                or max_error is None
            ):
                raise M6ResultStoreIntegrityError(
                    "selected Waymax comparison lacks complete denominator/error "
                    "evidence"
                )
            if kind == "exact":
                if (
                    tolerance_failures != 0
                    or max_normalized is not None
                    or (binary_mismatches == 0) != (max_error == 0.0)
                ):
                    raise M6ResultStoreIntegrityError(
                        "exact Waymax comparison mismatch/error evidence "
                        "contradicts itself"
                    )
            else:
                if binary_mismatches != 0 or max_normalized is None:
                    raise M6ResultStoreIntegrityError(
                        "tolerance Waymax comparisons require normalized error "
                        "and no binary mismatch count"
                    )
                if (max_error == 0.0) != (max_normalized == 0.0):
                    raise M6ResultStoreIntegrityError(
                        "Waymax absolute and normalized maxima contradict"
                    )
            expected_status = (
                "passed"
                if tolerance_failures == 0 and binary_mismatches == 0
                else "failed"
            )
            if status != expected_status:
                raise M6ResultStoreIntegrityError(
                    "Waymax comparison status differs from its failure counts"
                )
            if kind == "tolerance" and (
                (status == "passed" and max_normalized > 1.0)
                or (
                    status == "failed"
                    and tolerance_failures > 0
                    and max_normalized <= 1.0
                )
            ):
                raise M6ResultStoreIntegrityError(
                    "Waymax normalized maximum contradicts tolerance failures"
                )
        normalized.append(
            {
                "selection_position": position,
                "bundle": bundle,
                "condition": condition,
                "field_name": field_name,
                "cohort_index": cohort_index,
                "qualification_binding_sha256": binding,
                "comparison_kind": kind,
                "denominator": numeric["denominator"],
                "max_abs_error": max_error,
                "max_normalized_error": max_normalized,
                "tolerance_failures": numeric["tolerance_failures"],
                "binary_mismatches": numeric["binary_mismatches"],
                "status": status,
            }
        )
    if seen != set(expected_keys):
        raise M6ResultStoreIntegrityError(
            "Waymax field comparison table is not exactly 640 rows"
        )
    normalized.sort(
        key=lambda row: order[
            (
                row["selection_position"],
                row["bundle"],
                row["condition"],
                row["field_name"],
            )
        ]
    )
    return tuple(normalized)


def _normalize_waymax_determinism_from_qualification(
    rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
    qualification_rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    fields = tuple(field.name for field in WAYMAX_DETERMINISM_SCHEMA)
    qualification = _normalize_waymax_qualification(
        qualification_rows,
        receipt,
    )
    selected = _selected_waymax_by_position(qualification)
    expected_keys = tuple(
        (position, bundle, condition)
        for position in range(M6_WAYMAX_MAX_SELECTED)
        for bundle in M6_WAYMAX_BUNDLES
        for condition in M6_WAYMAX_CONDITIONS
    )
    order = {key: index for index, key in enumerate(expected_keys)}
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for raw in rows:
        row = _exact_row(raw, fields, WAYMAX_DETERMINISM)
        position = _integer(
            row["selection_position"],
            name="selection_position",
            minimum=0,
            maximum=M6_WAYMAX_MAX_SELECTED - 1,
        )
        bundle = _text(row["bundle"], "bundle")
        condition = _text(row["condition"], "condition")
        key = (position, bundle, condition)
        if key not in order or key in seen:
            raise M6ResultStoreIntegrityError(
                "Waymax determinism row is duplicate or outside fixed 64 rows"
            )
        seen.add(key)
        selected_row = selected.get(position)
        cohort_index = (
            None
            if row["cohort_index"] is None
            else _integer(
                row["cohort_index"],
                name="cohort_index",
                minimum=0,
                maximum=receipt.population_size - 1,
            )
        )
        binding = _optional_text(
            row["qualification_binding_sha256"],
            "qualification_binding_sha256",
        )
        hashes = {
            name: _optional_text(row[name], name)
            for name in (
                "eager_pass_1_sha256",
                "eager_pass_2_sha256",
                "jit_eager_sha256",
                "jit_compiled_sha256",
            )
        }
        for name, value in hashes.items():
            if value is not None and _SHA256.fullmatch(value) is None:
                raise M6ResultStoreIntegrityError(
                    f"{name} must be SHA-256 when present"
                )
        supplied_status = _text(row["status"], "status")
        if selected_row is None:
            expected_status = "not_applicable"
            if (
                cohort_index is not None
                or binding is not None
                or any(value is not None for value in hashes.values())
            ):
                raise M6ResultStoreIntegrityError(
                    "unused Waymax determinism rows must be exact NA"
                )
        else:
            if (
                cohort_index != selected_row["cohort_index"]
                or binding
                != selected_row["qualification_binding_sha256"]
                or hashes["eager_pass_1_sha256"] is None
                or hashes["eager_pass_2_sha256"] is None
            ):
                raise M6ResultStoreIntegrityError(
                    "selected Waymax determinism row lacks qualification-bound "
                    "independent eager passes"
                )
            passed = (
                hashes["eager_pass_1_sha256"]
                == hashes["eager_pass_2_sha256"]
            )
            if position == 0:
                if (
                    hashes["jit_eager_sha256"] is None
                    or hashes["jit_compiled_sha256"] is None
                ):
                    raise M6ResultStoreIntegrityError(
                        "selection position zero requires eager/JIT evidence"
                    )
                passed = passed and len(
                    {
                        hashes["eager_pass_1_sha256"],
                        hashes["jit_eager_sha256"],
                        hashes["jit_compiled_sha256"],
                    }
                ) == 1
            elif (
                hashes["jit_eager_sha256"] is not None
                or hashes["jit_compiled_sha256"] is not None
            ):
                raise M6ResultStoreIntegrityError(
                    "only selection position zero may carry JIT evidence"
                )
            expected_status = "passed" if passed else "failed"
        if supplied_status != expected_status:
            raise M6ResultStoreIntegrityError(
                "Waymax determinism status differs from sealed hashes"
            )
        normalized.append(
            {
                "selection_position": position,
                "bundle": bundle,
                "condition": condition,
                "cohort_index": cohort_index,
                "qualification_binding_sha256": binding,
                "status": expected_status,
                **hashes,
            }
        )
    if seen != set(expected_keys):
        raise M6ResultStoreIntegrityError(
            "Waymax determinism table is not the exact 64-row grid"
        )
    normalized.sort(
        key=lambda row: order[
            (
                row["selection_position"],
                row["bundle"],
                row["condition"],
            )
        ]
    )
    return tuple(normalized)


def _waymax_thresholded_nonzero(metric_name: str, value: float) -> bool:
    threshold = {
        "additional_target_braking_impulse_mps": 1e-9,
        "response_timeliness_s": 1e-9,
        "minimum_longitudinal_bumper_gap_change_m": 1e-6,
        "target_progress_loss_m": 1e-6,
    }[metric_name]
    return abs(float(value)) > threshold


def _derive_waymax_accounting(
    qualification_rows: Iterable[Mapping[str, Any]],
    scalar_rows: Iterable[Mapping[str, Any]],
    comparison_rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
    selection_receipt: M6WaymaxSelectionReceipt,
    *,
    matrix: M6WaymaxMatrixResult | None,
    stored_cell_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    qualification = _normalize_waymax_qualification(
        qualification_rows,
        receipt,
    )
    # These calls independently repeat the fixed-grid and cross-binding checks.
    scalars = _normalize_waymax_scene_scalars_from_qualification(
        scalar_rows,
        receipt,
        qualification,
        selection_receipt,
    )
    comparisons = _normalize_waymax_field_comparisons_from_qualification(
        comparison_rows,
        receipt,
        qualification,
    )
    if receipt.mode == DATA_FREE_MODE:
        if matrix is not None:
            raise M6ResultStoreIntegrityError(
                "data_free Waymax accounting cannot accept live matrix evidence"
            )
        cell_evidence = tuple(
            {
                "bundle": bundle,
                "metric_name": metric,
                "metric_version": version,
                "value_unit": unit,
                "pair_n": 0,
                "status": "unsupported",
                "responder_n": 0 if metric == "response_timeliness_s" else None,
                "censor_n": 0 if metric == "response_timeliness_s" else None,
                "arithmetic_mean": None,
                "median": None,
                "pointwise_level": None,
                "pointwise_lower": None,
                "pointwise_upper": None,
            }
            for bundle in M6_WAYMAX_BUNDLES
            for metric, version, unit in M6_PRIMARY_METRICS
        )
    elif isinstance(matrix, M6WaymaxMatrixResult):
        matrix.revalidate()
        if (
            matrix.intervention_configuration_fingerprint
            != selection_receipt.primary_b2_configuration_fingerprint
            or matrix.pair_n != selection_receipt.selection_member_count
        ):
            raise M6ResultStoreIntegrityError(
                "live Waymax matrix differs from the sealed selection receipt"
            )
        cell_evidence = tuple(
            {
                "bundle": cell.bundle,
                "metric_name": cell.metric_name,
                "metric_version": cell.metric_version,
                "value_unit": cell.value_unit,
                "pair_n": cell.pair_n,
                "status": cell.status,
                "responder_n": cell.responder_n,
                "censor_n": cell.censor_n,
                "arithmetic_mean": cell.arithmetic_mean,
                "median": cell.median,
                "pointwise_level": (
                    None
                    if cell.pointwise_band is None
                    else cell.pointwise_band.level
                ),
                "pointwise_lower": (
                    None
                    if cell.pointwise_band is None
                    else cell.pointwise_band.lower
                ),
                "pointwise_upper": (
                    None
                    if cell.pointwise_band is None
                    else cell.pointwise_band.upper
                ),
            }
            for cell in matrix.cells
        )
    elif stored_cell_rows is not None:
        normalized_stored = tuple(stored_cell_rows)
        if len(normalized_stored) != len(M6_WAYMAX_BUNDLES) * len(
            M6_PRIMARY_METRICS
        ):
            raise M6ResultStoreIntegrityError(
                "stored Waymax cell accounting domain is not exactly eight rows"
            )
        cell_evidence = tuple(
            {
                "bundle": row["bundle"],
                "metric_name": row["metric_name"],
                "metric_version": row["metric_version"],
                "value_unit": row["unit"],
                "pair_n": row["pair_n"],
                "status": row["status"],
                "responder_n": row["responder_n"],
                "censor_n": row["censor_n"],
                "arithmetic_mean": row["arithmetic_mean"],
                "median": row["median"],
                "pointwise_level": row["pointwise_level"],
                "pointwise_lower": row["pointwise_lower"],
                "pointwise_upper": row["pointwise_upper"],
            }
            for row in normalized_stored
        )
    else:
        raise M6ResultStoreIntegrityError(
            "Waymax accounting requires issued or verified stored cell evidence"
        )
    selected = sum(row["selected"] is True for row in qualification)
    qualified = sum(
        row["assessment_status"] == "qualified" for row in qualification
    )
    status = "accepted" if selected >= 8 else "unsupported"
    rows: list[dict[str, Any]] = []
    for name, count in (
        ("qualified_count", qualified),
        ("selected_count", selected),
        ("transition_count", M6_WAYMAX_TRANSITIONS),
    ):
        row = _empty_waymax_accounting_row(
            "scope",
            name,
            status=status,
        )
        row["count"] = count
        rows.append(row)
    for reason in M6_WAYMAX_REJECTION_REASONS:
        row = _empty_waymax_accounting_row(
            "selection_rejection",
            reason,
            status=status,
        )
        qualification_reason = next(
            source_reason
            for source_reason, accounting_reason in (
                M6_WAYMAX_QUALIFICATION_TO_ACCOUNTING_REASON.items()
            )
            if accounting_reason == reason
        )
        row["count"] = sum(
            item["rejection_reason"] == qualification_reason
            for item in qualification
        )
        row["opportunity_n"] = (
            0 if receipt.mode == DATA_FREE_MODE else receipt.eligible_count
        )
        rows.append(row)
    for bundle in M6_WAYMAX_BUNDLES:
        for condition in M6_WAYMAX_CONDITIONS:
            for field_name in M6_WAYMAX_COMPARISON_FIELDS:
                members = [
                    row
                    for row in comparisons
                    if row["bundle"] == bundle
                    and row["condition"] == condition
                    and row["field_name"] == field_name
                    and row["status"] != "not_applicable"
                ]
                row = _empty_waymax_accounting_row(
                    "field_comparison",
                    field_name,
                    bundle=bundle,
                    condition=condition,
                    status=(
                        "unsupported"
                        if not members
                        else (
                            "accepted"
                            if all(item["status"] == "passed" for item in members)
                            else "failed"
                        )
                    ),
                )
                row["comparison_kind"] = (
                    "exact"
                    if field_name in M6_WAYMAX_EXACT_FIELDS
                    else "tolerance"
                )
                row["denominator"] = sum(
                    int(item["denominator"]) for item in members
                )
                row["binary_mismatches"] = sum(
                    int(item["binary_mismatches"]) for item in members
                )
                row["max_abs_error"] = (
                    max(float(item["max_abs_error"]) for item in members)
                    if members
                    else None
                )
                if field_name not in M6_WAYMAX_EXACT_FIELDS:
                    row["tolerance_failures"] = sum(
                        int(item["tolerance_failures"]) for item in members
                    )
                rows.append(row)
    opportunity = selected * M6_WAYMAX_TRANSITIONS * len(
        M6_WAYMAX_CONDITIONS
    )
    control_counts = {
        "target_requested_control": opportunity,
        "target_effective_control": opportunity,
        "target_logged_lifecycle_fallback": 0,
        "target_initialized_overlap_exclusion": 0,
    }
    for name in M6_WAYMAX_CONTROL_COUNTS:
        row = _empty_waymax_accounting_row(
            "control_partition",
            name,
            bundle=M6_WAYMAX_PRIVILEGED_IDM,
            status=status,
        )
        row["opportunity_n"] = opportunity
        row["count"] = control_counts[name]
        rows.append(row)
    scalar_by_cell = {
        (bundle, metric): [
            row
            for row in scalars
            if row["bundle"] == bundle
            and row["metric_name"] == metric
            and row["status"] == "selected"
        ]
        for bundle in M6_WAYMAX_BUNDLES
        for metric, _version, _unit in M6_PRIMARY_METRICS
    }
    for cell in cell_evidence:
        row = _empty_waymax_accounting_row(
            "secondary_cell",
            "paired_effect",
            bundle=str(cell["bundle"]),
            metric_name=str(cell["metric_name"]),
            status=str(cell["status"]),
        )
        row.update(
            {
                "metric_version": cell["metric_version"],
                "unit": cell["value_unit"],
                "pair_n": cell["pair_n"],
                "thresholded_nonzero_n": sum(
                    _waymax_thresholded_nonzero(
                        str(cell["metric_name"]),
                        float(item["value"]),
                    )
                    for item in scalar_by_cell[
                        (str(cell["bundle"]), str(cell["metric_name"]))
                    ]
                ),
                "responder_n": cell["responder_n"],
                "censor_n": cell["censor_n"],
                "arithmetic_mean": cell["arithmetic_mean"],
                "median": cell["median"],
                "pointwise_level": cell["pointwise_level"],
                "pointwise_lower": cell["pointwise_lower"],
                "pointwise_upper": cell["pointwise_upper"],
                "suppression_reason": (
                    "waymax_selected_n_below_8"
                    if cell["status"] == "unsupported"
                    else (
                        "waymax_pair_n_below_10"
                        if cell["status"] == "insufficient_n"
                        else None
                    )
                ),
                "source_pairing_complete": True,
                "directional_language_allowed": False,
            }
        )
        rows.append(row)
    return _normalize_waymax_accounting(rows, receipt)


def _empty_waymax_accounting_row(
    record_type: str,
    name: str,
    *,
    bundle: str | None = None,
    condition: str | None = None,
    metric_name: str | None = None,
    status: str,
) -> dict[str, Any]:
    row = {field.name: None for field in WAYMAX_ACCOUNTING_SCHEMA}
    row.update(
        {
            "record_type": record_type,
            "name": name,
            "bundle": bundle,
            "condition": condition,
            "metric_name": metric_name,
            "status": status,
        }
    )
    return row


def _normalize_waymax_accounting(
    rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
) -> tuple[dict[str, Any], ...]:
    fields = tuple(field.name for field in WAYMAX_ACCOUNTING_SCHEMA)
    order = {key: index for index, key in enumerate(M6_WAYMAX_ROW_DOMAIN)}
    units = {metric: unit for metric, _version, unit in M6_PRIMARY_METRICS}
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for raw in rows:
        row = _exact_row(raw, fields, WAYMAX_ACCOUNTING)
        key = (
            _text(row["record_type"], "record_type"),
            _text(row["name"], "name"),
            _optional_text(row["bundle"], "bundle"),
            _optional_text(row["condition"], "condition"),
            _optional_text(row["metric_name"], "metric_name"),
        )
        if key not in order or key in seen:
            raise M6ResultStoreIntegrityError(
                "Waymax row is duplicate or outside fixed domain"
            )
        seen.add(key)
        normalized_row = {
            name: row[name] for name in fields
        }
        normalized_row.update(
            {
                "record_type": key[0],
                "name": key[1],
                "bundle": key[2],
                "condition": key[3],
                "metric_name": key[4],
                "status": _text(row["status"], "status"),
                "metric_version": _optional_text(
                    row["metric_version"],
                    "metric_version",
                ),
                "unit": _optional_text(row["unit"], "unit"),
                "comparison_kind": _optional_text(
                    row["comparison_kind"],
                    "comparison_kind",
                ),
                "suppression_reason": _optional_text(
                    row["suppression_reason"],
                    "suppression_reason",
                ),
            }
        )
        for name in (
            "count",
            "opportunity_n",
            "denominator",
            "tolerance_failures",
            "binary_mismatches",
            "pair_n",
            "thresholded_nonzero_n",
            "responder_n",
            "censor_n",
        ):
            value = row[name]
            normalized_row[name] = (
                None
                if value is None
                else _integer(
                    value,
                    name=name,
                    minimum=0,
                    maximum=2**63 - 1,
                )
            )
        for name in (
            "max_abs_error",
            "arithmetic_mean",
            "median",
            "pointwise_level",
            "pointwise_lower",
            "pointwise_upper",
        ):
            value = row[name]
            normalized_row[name] = (
                None
                if value is None
                else _finite(
                    value,
                    name,
                    minimum=0.0 if name == "max_abs_error" else None,
                )
            )
        source_pairing = row["source_pairing_complete"]
        normalized_row["source_pairing_complete"] = (
            None
            if source_pairing is None
            else _boolean(source_pairing, "source_pairing_complete")
        )
        directional = row["directional_language_allowed"]
        normalized_row["directional_language_allowed"] = (
            None
            if directional is None
            else _boolean(
                directional,
                "directional_language_allowed",
            )
        )
        _validate_waymax_subtype(normalized_row)
        normalized.append(normalized_row)
    if seen != set(M6_WAYMAX_ROW_DOMAIN):
        raise M6ResultStoreIntegrityError(
            "Waymax accounting domain is incomplete"
        )
    normalized.sort(
        key=lambda row: order[
            (
                row["record_type"],
                row["name"],
                row["bundle"],
                row["condition"],
                row["metric_name"],
            )
        ]
    )
    _validate_waymax_cross_rows(normalized, receipt)
    return tuple(normalized)


def _validate_waymax_subtype(row: Mapping[str, Any]) -> None:
    record_type = row["record_type"]
    status = row["status"]
    common_identity = {
        "record_type",
        "name",
        "bundle",
        "condition",
        "metric_name",
        "status",
    }
    allowed_nonnull: set[str]
    if record_type == "scope":
        allowed_nonnull = common_identity | {"count"}
        if status not in {"accepted", "unsupported"} or row["count"] is None:
            raise M6ResultStoreIntegrityError("Waymax scope subtype is invalid")
    elif record_type == "selection_rejection":
        allowed_nonnull = common_identity | {"count", "opportunity_n"}
        if (
            status not in {"accepted", "unsupported"}
            or row["count"] is None
            or row["opportunity_n"] is None
            or row["count"] > row["opportunity_n"]
        ):
            raise M6ResultStoreIntegrityError(
                "Waymax selection-rejection subtype is invalid"
            )
    elif record_type == "field_comparison":
        allowed_nonnull = common_identity | {
            "comparison_kind",
            "denominator",
            "binary_mismatches",
        }
        expected_kind = (
            "exact"
            if row["name"] in M6_WAYMAX_EXACT_FIELDS
            else "tolerance"
        )
        if row["comparison_kind"] != expected_kind:
            raise M6ResultStoreIntegrityError(
                "Waymax comparison kind drifted"
            )
        denominator = row["denominator"]
        binary = row["binary_mismatches"]
        if denominator is None or binary is None or binary > denominator:
            raise M6ResultStoreIntegrityError(
                "Waymax comparison denominator/mismatch is invalid"
            )
        if expected_kind == "exact":
            if row["tolerance_failures"] is not None:
                raise M6ResultStoreIntegrityError(
                    "exact Waymax field cannot carry tolerance-failure values"
                )
        else:
            allowed_nonnull |= {"tolerance_failures"}
            tolerance = row["tolerance_failures"]
            if tolerance is None or tolerance > denominator:
                raise M6ResultStoreIntegrityError(
                    "Waymax tolerance failure count is invalid"
                )
        if denominator > 0:
            allowed_nonnull.add("max_abs_error")
            if row["max_abs_error"] is None:
                raise M6ResultStoreIntegrityError(
                    "executed field comparison requires maximum absolute error"
                )
        if status not in {"accepted", "failed", "unsupported"}:
            raise M6ResultStoreIntegrityError(
                "Waymax field status is invalid"
            )
        if status == "accepted" and (
            binary != 0
            or (
                expected_kind == "tolerance"
                and row["tolerance_failures"] != 0
            )
        ):
            raise M6ResultStoreIntegrityError(
                "accepted Waymax field cannot retain failures"
            )
        if status == "failed" and (
            binary == 0
            and (
                expected_kind == "exact"
                or row["tolerance_failures"] == 0
            )
        ):
            raise M6ResultStoreIntegrityError(
                "failed Waymax field requires a registered failure count"
            )
    elif record_type == "control_partition":
        allowed_nonnull = common_identity | {"count", "opportunity_n"}
        if (
            status not in {"accepted", "failed", "unsupported"}
            or row["count"] is None
            or row["opportunity_n"] is None
            or row["count"] > row["opportunity_n"]
        ):
            raise M6ResultStoreIntegrityError(
                "Waymax control partition subtype is invalid"
            )
    elif record_type == "secondary_cell":
        allowed_nonnull = common_identity | {
            "metric_version",
            "unit",
            "pair_n",
            "thresholded_nonzero_n",
            "source_pairing_complete",
            "directional_language_allowed",
        }
        metric = row["metric_name"]
        if (
            row["metric_version"] != "1.0.0"
            or row["unit"]
            != {
                name: unit for name, _version, unit in M6_PRIMARY_METRICS
            }[metric]
            or row["source_pairing_complete"] is not True
            or row["directional_language_allowed"] is not False
        ):
            raise M6ResultStoreIntegrityError(
                "Waymax cell identity/pairing/directionality drifted"
            )
        pair_n = row["pair_n"]
        nonzero = row["thresholded_nonzero_n"]
        if pair_n is None or nonzero is None or nonzero > pair_n:
            raise M6ResultStoreIntegrityError(
                "Waymax cell counts are invalid"
            )
        timeliness = metric == "response_timeliness_s"
        if timeliness:
            allowed_nonnull |= {"responder_n", "censor_n"}
            if (
                row["responder_n"] is None
                or row["censor_n"] is None
                or row["responder_n"] + row["censor_n"] != pair_n
            ):
                raise M6ResultStoreIntegrityError(
                    "Waymax timeliness response counts drifted"
                )
        elif row["responder_n"] is not None or row["censor_n"] is not None:
            raise M6ResultStoreIntegrityError(
                "Waymax non-timeliness cell has response counts"
            )
        if pair_n < 8:
            allowed_nonnull.add("suppression_reason")
            if (
                status != "unsupported"
                or row["suppression_reason"]
                != "waymax_selected_n_below_8"
            ):
                raise M6ResultStoreIntegrityError(
                    "Waymax N<8 cell suppression drifted"
                )
        elif pair_n < 10:
            allowed_nonnull.add("suppression_reason")
            if (
                status != "insufficient_n"
                or row["suppression_reason"] != "waymax_pair_n_below_10"
            ):
                raise M6ResultStoreIntegrityError(
                    "Waymax N 8-9 cell suppression drifted"
                )
        else:
            allowed_nonnull |= {
                "arithmetic_mean",
                "median",
                "pointwise_level",
                "pointwise_lower",
                "pointwise_upper",
            }
            if (
                pair_n > 16
                or status != "descriptive"
                or row["suppression_reason"] is not None
                or row["arithmetic_mean"] is None
                or row["median"] is None
                or row["pointwise_level"] != 0.95
                or row["pointwise_lower"] is None
                or row["pointwise_upper"] is None
                or row["pointwise_lower"] > row["pointwise_upper"]
            ):
                raise M6ResultStoreIntegrityError(
                    "executed Waymax cell statistics drifted"
                )
    else:
        raise M6ResultStoreIntegrityError("unknown Waymax record subtype")
    for key, value in row.items():
        if key not in allowed_nonnull and value is not None:
            raise M6ResultStoreIntegrityError(
                f"Waymax {record_type} row has forbidden non-null field {key}"
            )


def _validate_waymax_cross_rows(
    rows: Sequence[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
) -> None:
    by_key = {
        (
            row["record_type"],
            row["name"],
            row["bundle"],
            row["condition"],
            row["metric_name"],
        ): row
        for row in rows
    }
    qualified = by_key[("scope", "qualified_count", None, None, None)][
        "count"
    ]
    selected = by_key[("scope", "selected_count", None, None, None)]["count"]
    transitions = by_key[("scope", "transition_count", None, None, None)][
        "count"
    ]
    rejections = [
        by_key[("selection_rejection", reason, None, None, None)]
        for reason in M6_WAYMAX_REJECTION_REASONS
    ]
    if receipt.mode == DATA_FREE_MODE:
        if (
            qualified != 0
            or selected != 0
            or any(
                row["opportunity_n"] != 0 or row["count"] != 0
                for row in rejections
            )
        ):
            raise M6ResultStoreIntegrityError(
                "data-free Waymax scope must remain unassessed and unsupported"
            )
    elif (
        any(row["opportunity_n"] != receipt.eligible_count for row in rejections)
        or qualified + sum(row["count"] for row in rejections)
        != receipt.eligible_count
    ):
        raise M6ResultStoreIntegrityError(
            "Waymax qualified/rejections do not partition primary N"
        )
    expected_selected = min(16, qualified) if qualified >= 8 else 0
    if selected != expected_selected or transitions != 20:
        raise M6ResultStoreIntegrityError(
            "Waymax selection is not exact 16-or-floor / 20 transitions"
        )
    executed = selected >= 8
    expected_scope_status = "accepted" if executed else "unsupported"
    if any(
        row["status"] != expected_scope_status
        for row in rows
        if row["record_type"] in {"scope", "selection_rejection"}
    ):
        raise M6ResultStoreIntegrityError(
            "Waymax scope/rejection status disagrees with selected scope"
        )
    for row in rows:
        if row["record_type"] == "field_comparison":
            if executed:
                if row["denominator"] < 1 or row["status"] == "unsupported":
                    raise M6ResultStoreIntegrityError(
                        "selected Waymax field comparison is unsupported/empty"
                    )
            elif (
                row["denominator"] != 0
                or row["status"] != "unsupported"
                or row["binary_mismatches"] != 0
                or (
                    row["comparison_kind"] == "tolerance"
                    and row["tolerance_failures"] != 0
                )
            ):
                raise M6ResultStoreIntegrityError(
                    "unselected Waymax comparison must be exact unsupported zero"
                )
        elif row["record_type"] == "secondary_cell":
            if row["pair_n"] != selected:
                raise M6ResultStoreIntegrityError(
                    "Waymax cell N differs from selected count"
                )
    control = {
        row["name"]: row
        for row in rows
        if row["record_type"] == "control_partition"
    }
    opportunity = selected * 20 * len(M6_WAYMAX_CONDITIONS)
    if any(row["opportunity_n"] != opportunity for row in control.values()):
        raise M6ResultStoreIntegrityError(
            "Waymax control opportunity denominator drifted"
        )
    if control["target_requested_control"]["count"] != opportunity:
        raise M6ResultStoreIntegrityError(
            "Waymax target requested control must cover every opportunity"
        )
    if (
        control["target_effective_control"]["count"] != opportunity
        or control["target_logged_lifecycle_fallback"]["count"] != 0
        or control["target_initialized_overlap_exclusion"]["count"] != 0
    ):
        raise M6ResultStoreIntegrityError(
            "Waymax frozen target must be effectively controlled at every "
            "opportunity without lifecycle fallback or overlap exclusion"
        )
    expected_control_status = "accepted" if executed else "unsupported"
    if any(
        row["status"] != expected_control_status for row in control.values()
    ):
        raise M6ResultStoreIntegrityError(
            "Waymax control status disagrees with selected scope"
        )


def _normalize_typed_provenance(
    rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
) -> tuple[dict[str, Any], ...]:
    fields = tuple(field.name for field in TYPED_PROVENANCE_SCHEMA)
    raw_rows = tuple(rows)
    if len(raw_rows) != 1:
        raise M6ResultStoreIntegrityError(
            "typed provenance requires exactly one row"
        )
    row = _exact_row(raw_rows[0], fields, TYPED_PROVENANCE)
    fixed = {
        "plan_version": M6_PLAN_VERSION,
        "config_version": M6_CONFIG_VERSION,
        "statistics_schema_version": M6_STATISTICS_SCHEMA_VERSION,
    }
    if receipt.mode == DATA_FREE_MODE:
        fixed.update(
            {
                "population_label": "synthetic_data_free_n10",
                "source_shard_start": None,
                "source_shard_end": None,
            }
        )
    else:
        fixed.update(
            {
                "population_label": (
                    "accepted_m4_complete_case_ten_shard_cohort"
                ),
                "source_shard_start": "00000",
                "source_shard_end": "00009",
            }
        )
    if any(row[name] != value for name, value in fixed.items()):
        raise M6ResultStoreIntegrityError(
            "typed provenance fixed labels drifted"
        )
    hashes = {}
    for name in (
        "executable_source_sha256",
        "uv_lock_sha256",
        "runtime_config_sha256",
    ):
        value = _text(row[name], name)
        if _SHA256.fullmatch(value) is None:
            raise M6ResultStoreIntegrityError(
                f"typed provenance {name} must be SHA-256"
            )
        hashes[name] = value
    for name in (
        "accepted_m4_manifest_sha256",
        "accepted_m4_provenance_sha256",
    ):
        value = _optional_text(row[name], name)
        if receipt.mode == DATA_FREE_MODE:
            if value is not None:
                raise M6ResultStoreIntegrityError(
                    f"data-free typed provenance must omit {name}"
                )
        elif value is None or _SHA256.fullmatch(value) is None:
            raise M6ResultStoreIntegrityError(
                f"official typed provenance {name} must be SHA-256"
            )
        hashes[name] = value
    git_objects = {}
    for name in ("approved_git_commit", "git_tree"):
        value = _text(row[name], name)
        if _GIT_OBJECT.fullmatch(value) is None:
            raise M6ResultStoreIntegrityError(
                f"typed provenance {name} must be a 40-hex Git object"
            )
        git_objects[name] = value
    versions = {}
    for name in ("python_version", "numpy_version", "pyarrow_version"):
        versions[name] = _safe_version(row[name], name, nullable=False)
    optional_versions = {}
    for name in (
        "jax_version",
        "jaxlib_version",
        "tensorflow_version",
    ):
        optional_versions[name] = _safe_version(
            row[name],
            name,
            nullable=True,
        )
    waymax_commit = _optional_text(row["waymax_commit"], "waymax_commit")
    if waymax_commit is not None and re.fullmatch(
        r"[0-9a-f]{7,40}",
        waymax_commit,
    ) is None:
        raise M6ResultStoreIntegrityError(
            "waymax_commit must be 7-40 lowercase hex"
        )
    jax_backend = _optional_text(row["jax_backend"], "jax_backend")
    jax_device = _optional_text(
        row["jax_device_class"],
        "jax_device_class",
    )
    if (jax_backend is None) != (jax_device is None) or jax_backend not in {
        None,
        "cpu",
        "gpu",
        "tpu",
    } or jax_device not in {None, "cpu", "gpu", "tpu"}:
        raise M6ResultStoreIntegrityError(
            "JAX backend/device class domain is invalid"
        )
    optional_runtime = (
        *optional_versions.values(),
        waymax_commit,
        jax_backend,
        jax_device,
    )
    if receipt.mode == OFFICIAL_MODE:
        if any(value is None for value in optional_runtime):
            raise M6ResultStoreIntegrityError(
                "official typed provenance requires the complete "
                "JAX/TensorFlow/Waymax runtime"
            )
        if (
            waymax_commit != WAYMAX_COMMIT
            or jax_backend != "cpu"
            or jax_device != "cpu"
        ):
            raise M6ResultStoreIntegrityError(
                "official typed provenance drifted from the pinned Waymax/local "
                "CPU runtime"
            )
    elif any(value is not None for value in optional_runtime):
        raise M6ResultStoreIntegrityError(
            "data-free typed provenance cannot claim JAX/TensorFlow/Waymax execution"
        )
    primary_fingerprint = _text(
        row["primary_intervention_fingerprint"],
        "primary_intervention_fingerprint",
    )
    secondary_fingerprint = _text(
        row["secondary_intervention_fingerprint"],
        "secondary_intervention_fingerprint",
    )
    if (
        primary_fingerprint != receipt.primary_intervention_fingerprint
        or secondary_fingerprint
        != receipt.secondary_intervention_fingerprint
    ):
        raise M6ResultStoreIntegrityError(
            "typed provenance intervention fingerprints drifted"
        )
    return (
        {
            **fixed,
            **git_objects,
            **hashes,
            **versions,
            **optional_versions,
            "waymax_commit": waymax_commit,
            "jax_backend": jax_backend,
            "jax_device_class": jax_device,
            "primary_intervention_fingerprint": primary_fingerprint,
            "secondary_intervention_fingerprint": secondary_fingerprint,
        },
    )


def _normalize_execution_summary(
    rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
) -> tuple[dict[str, Any], ...]:
    fields = tuple(field.name for field in EXECUTION_SUMMARY_SCHEMA)
    raw_rows = tuple(rows)
    if len(raw_rows) != 1:
        raise M6ResultStoreIntegrityError(
            "execution summary requires exactly one row"
        )
    row = _exact_row(raw_rows[0], fields, EXECUTION_SUMMARY)
    deterministic = _text(
        row["deterministic_repeat_status"],
        "deterministic_repeat_status",
    )
    waymax_status = _text(row["waymax_gate_status"], "waymax_gate_status")
    claim_status = _text(
        row["real_reactivity_claim_status"],
        "real_reactivity_claim_status",
    )
    release_status = _text(
        row["release_gate_status"],
        "release_gate_status",
    )
    if deterministic != "passed":
        raise M6ResultStoreIntegrityError(
            "terminal evidence requires deterministic repeat passed"
        )
    if waymax_status not in {"accepted", "failed", "unsupported"}:
        raise M6ResultStoreIntegrityError(
            "Waymax gate status must be accepted, failed, or unsupported"
        )
    if claim_status not in {"supported", "blocked"}:
        raise M6ResultStoreIntegrityError(
            "real reactivity claim status is invalid"
        )
    if release_status != "accepted":
        raise M6ResultStoreIntegrityError(
            "terminal evidence requires accepted release gate"
        )
    rss = _integer(
        row["fresh_worker_peak_rss_bytes"],
        name="fresh_worker_peak_rss_bytes",
        minimum=0,
    )
    expected_counts = {
        "eligibility_rows": receipt.expected_rows[ELIGIBILITY_LEDGER],
        "primary_scene_rows": receipt.expected_rows[PRIMARY_SCENE_SCALARS],
        "primary_matrix_rows": receipt.expected_rows[PRIMARY_MATRIX],
        "primary_repeat_scene_rows": receipt.expected_rows[
            PRIMARY_REPEAT_SCENE_SCALARS
        ],
        "primary_repeat_matrix_rows": receipt.expected_rows[
            PRIMARY_REPEAT_MATRIX
        ],
        "secondary_scene_rows": receipt.expected_rows[
            SECONDARY_SCENE_SCALARS
        ],
        "secondary_matrix_rows": receipt.expected_rows[SECONDARY_MATRIX],
        "negative_timing_observation_rows": receipt.expected_rows[
            NEGATIVE_TIMING_OBSERVATIONS
        ],
        "negative_timing_gate_rows": receipt.expected_rows[
            NEGATIVE_TIMING_GATES
        ],
        "waymax_accounting_rows": receipt.expected_rows[WAYMAX_ACCOUNTING],
        "waymax_qualification_rows": receipt.expected_rows[
            WAYMAX_QUALIFICATION
        ],
        "waymax_scene_scalar_rows": receipt.expected_rows[
            WAYMAX_SCENE_SCALARS
        ],
        "waymax_field_comparison_rows": receipt.expected_rows[
            WAYMAX_FIELD_COMPARISONS
        ],
        "waymax_determinism_rows": receipt.expected_rows[
            WAYMAX_DETERMINISM
        ],
        "stage_timing_rows": receipt.expected_rows[STAGE_TIMINGS],
        "review_decision_rows": receipt.expected_rows[REVIEW_DECISIONS],
    }
    normalized_counts = {
        name: _integer(row[name], name=name, minimum=0)
        for name in expected_counts
    }
    if normalized_counts != expected_counts:
        raise M6ResultStoreIntegrityError(
            "execution row-domain accounting drifted"
        )
    return (
        {
            "deterministic_repeat_status": deterministic,
            "waymax_gate_status": waymax_status,
            "real_reactivity_claim_status": claim_status,
            "release_gate_status": release_status,
            "fresh_worker_peak_rss_bytes": rss,
            **normalized_counts,
        },
    )


def _derive_execution_summary(
    *,
    receipt: M6EligibilityReceipt,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    determinism: M6DeterminismReceipt,
    fresh_worker_peak_rss_bytes: int,
) -> dict[str, Any]:
    rss = _integer(
        fresh_worker_peak_rss_bytes,
        name="fresh_worker_peak_rss_bytes",
        minimum=1 if receipt.mode == OFFICIAL_MODE else 0,
    )
    primary = _normalize_primary_matrix(tables[PRIMARY_MATRIX], receipt)
    repeat_scene = _normalize_primary_scene_scalars(
        tables[PRIMARY_REPEAT_SCENE_SCALARS],
        receipt,
    )
    repeat_matrix = _normalize_primary_matrix(
        tables[PRIMARY_REPEAT_MATRIX],
        receipt,
    )
    primary_scene = _normalize_primary_scene_scalars(
        tables[PRIMARY_SCENE_SCALARS],
        receipt,
    )
    deterministic = (
        _canonical_rows_sha256(PRIMARY_SCENE_SCALARS, primary_scene)
        == _canonical_rows_sha256(PRIMARY_SCENE_SCALARS, repeat_scene)
        and _canonical_rows_sha256(PRIMARY_MATRIX, primary)
        == _canonical_rows_sha256(PRIMARY_MATRIX, repeat_matrix)
        and determinism.waymax_repeat_status != "failed"
    )
    qualification = _normalize_waymax_qualification(
        tables[WAYMAX_QUALIFICATION],
        receipt,
    )
    selected = sum(row["selected"] is True for row in qualification)
    accounting = _normalize_waymax_accounting(
        tables[WAYMAX_ACCOUNTING],
        receipt,
    )
    failed_comparison = any(
        row["record_type"] == "field_comparison"
        and row["status"] == "failed"
        for row in accounting
    )
    waymax_status = (
        "unsupported"
        if selected < 8
        else (
            "accepted"
            if (
                not failed_comparison
                and determinism.waymax_repeat_status == "passed"
            )
            else "failed"
        )
    )
    gates = _normalize_negative_timing_gates(
        tables[NEGATIVE_TIMING_GATES],
        receipt,
    )
    reviews = _normalize_review_decisions(
        tables[REVIEW_DECISIONS],
        receipt,
    )
    idm_timing = next(
        row
        for row in primary
        if row["policy_name"] == "idm"
        and row["metric_name"] == "response_timeliness_s"
    )
    claim_status = (
        "blocked"
        if receipt.mode == DATA_FREE_MODE
        else (
            "supported"
            if int(idm_timing["responder_n"]) >= 10
            else "blocked"
        )
    )
    release = (
        deterministic
        and waymax_status != "failed"
        and all(row["status"] != "failed" for row in gates)
        and all(row["decision"] == "accept" for row in reviews)
    )
    counts = {
        "eligibility_rows": receipt.expected_rows[ELIGIBILITY_LEDGER],
        "primary_scene_rows": receipt.expected_rows[PRIMARY_SCENE_SCALARS],
        "primary_matrix_rows": receipt.expected_rows[PRIMARY_MATRIX],
        "primary_repeat_scene_rows": receipt.expected_rows[
            PRIMARY_REPEAT_SCENE_SCALARS
        ],
        "primary_repeat_matrix_rows": receipt.expected_rows[
            PRIMARY_REPEAT_MATRIX
        ],
        "secondary_scene_rows": receipt.expected_rows[
            SECONDARY_SCENE_SCALARS
        ],
        "secondary_matrix_rows": receipt.expected_rows[SECONDARY_MATRIX],
        "negative_timing_observation_rows": receipt.expected_rows[
            NEGATIVE_TIMING_OBSERVATIONS
        ],
        "negative_timing_gate_rows": receipt.expected_rows[
            NEGATIVE_TIMING_GATES
        ],
        "waymax_accounting_rows": receipt.expected_rows[WAYMAX_ACCOUNTING],
        "waymax_qualification_rows": receipt.expected_rows[
            WAYMAX_QUALIFICATION
        ],
        "waymax_scene_scalar_rows": receipt.expected_rows[
            WAYMAX_SCENE_SCALARS
        ],
        "waymax_field_comparison_rows": receipt.expected_rows[
            WAYMAX_FIELD_COMPARISONS
        ],
        "waymax_determinism_rows": receipt.expected_rows[
            WAYMAX_DETERMINISM
        ],
        "stage_timing_rows": receipt.expected_rows[STAGE_TIMINGS],
        "review_decision_rows": receipt.expected_rows[REVIEW_DECISIONS],
    }
    for dataset_name, expected in receipt.expected_rows.items():
        if dataset_name == EXECUTION_SUMMARY:
            continue
        if len(tables[dataset_name]) != expected:
            raise M6ResultStoreIntegrityError(
                "execution derivation observed a row-domain mismatch"
            )
    row = {
        "deterministic_repeat_status": (
            "passed" if deterministic else "failed"
        ),
        "waymax_gate_status": waymax_status,
        "real_reactivity_claim_status": claim_status,
        "release_gate_status": "accepted" if release else "rejected",
        "fresh_worker_peak_rss_bytes": rss,
        **counts,
    }
    return _normalize_execution_summary((row,), receipt)[0]


def _normalize_stage_timings(
    rows: Iterable[Mapping[str, Any]],
    receipt: M6EligibilityReceipt,
) -> tuple[dict[str, Any], ...]:
    fields = tuple(field.name for field in STAGE_TIMINGS_SCHEMA)
    order = {name: index for index, name in enumerate(M6_STAGE_DOMAIN)}
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        row = _exact_row(raw, fields, STAGE_TIMINGS)
        name = _text(row["stage_name"], "stage_name")
        if name not in order or name in seen:
            raise M6ResultStoreIntegrityError(
                "stage timing is duplicate or unexpected"
            )
        seen.add(name)
        duration = _integer(
            row["duration_ms"],
            name="duration_ms",
            minimum=0,
        )
        if receipt.mode == OFFICIAL_MODE and duration <= 0:
            raise M6ResultStoreIntegrityError(
                "official stage timings must be positive observations"
            )
        normalized.append(
            {
                "stage_name": name,
                "duration_ms": duration,
            }
        )
    if seen != set(M6_STAGE_DOMAIN):
        raise M6ResultStoreIntegrityError(
            "stage timing domain is incomplete"
        )
    normalized.sort(key=lambda row: order[row["stage_name"]])
    return tuple(normalized)


def _normalize_review_decisions(
    rows: Iterable[Mapping[str, Any]],
    _receipt: M6EligibilityReceipt,
    *,
    expected_evidence_catalog_sha256: str | None = None,
) -> tuple[dict[str, Any], ...]:
    if (
        expected_evidence_catalog_sha256 is not None
        and (
            type(expected_evidence_catalog_sha256) is not str
            or _SHA256.fullmatch(
                expected_evidence_catalog_sha256
            )
            is None
        )
    ):
        raise M6ResultStoreIntegrityError(
            "expected review precursor digest must be SHA-256"
        )
    fields = tuple(field.name for field in REVIEW_DECISIONS_SCHEMA)
    order = {
        role: index for index, role in enumerate(M6_REVIEW_ROLE_DOMAIN)
    }
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        row = _exact_row(raw, fields, REVIEW_DECISIONS)
        role = _text(row["role"], "role")
        if role not in order or role in seen:
            raise M6ResultStoreIntegrityError(
                "review decision is duplicate or unexpected"
            )
        seen.add(role)
        decision = _text(row["decision"], "decision")
        if decision not in {"accept", "reject"}:
            raise M6ResultStoreIntegrityError(
                "review decision enum is invalid"
            )
        counts = {
            name: _integer(row[name], name=name, minimum=0)
            for name in ("p1_count", "p2_count", "p3_count")
        }
        expected_decision = (
            "accept"
            if counts["p1_count"] == 0 and counts["p2_count"] == 0
            else "reject"
        )
        if decision != expected_decision:
            raise M6ResultStoreIntegrityError(
                "review decision differs from P1/P2 counts"
            )
        evidence_catalog_sha256 = _text(
            row["evidence_catalog_sha256"],
            "evidence_catalog_sha256",
        )
        if _SHA256.fullmatch(evidence_catalog_sha256) is None:
            raise M6ResultStoreIntegrityError(
                "review evidence catalog binding must be SHA-256"
            )
        if (
            expected_evidence_catalog_sha256 is not None
            and evidence_catalog_sha256
            != expected_evidence_catalog_sha256
        ):
            raise M6ResultStoreIntegrityError(
                "review evidence catalog differs from sealed precursor "
                "artifacts"
            )
        normalized.append(
            {
                "role": role,
                "decision": decision,
                **counts,
                "evidence_catalog_sha256": evidence_catalog_sha256,
            }
        )
    if seen != set(M6_REVIEW_ROLE_DOMAIN):
        raise M6ResultStoreIntegrityError(
            "review decision domain is incomplete"
        )
    normalized.sort(key=lambda row: order[row["role"]])
    if (
        len(
            {
                row["evidence_catalog_sha256"]
                for row in normalized
            }
        )
        != 1
    ):
        raise M6ResultStoreIntegrityError(
            "all reviews must bind one evidence-catalog precursor digest"
        )
    return tuple(normalized)


def verify_committed_m6_result_store(
    project_root: str | Path,
    run_name: str,
    *,
    allow_data_free: bool = False,
    expected_mode: str | None = None,
) -> VerifiedM6ResultStore:
    """Independently reopen a COMMITTED pre-terminal checkpoint."""

    return _verify_m6_result_store(
        project_root,
        run_name,
        require_success=False,
        allow_data_free=allow_data_free,
        expected_mode=expected_mode,
    )


def verify_m6_result_store(
    project_root: str | Path,
    run_name: str,
    *,
    allow_data_free: bool = False,
    expected_mode: str | None = None,
) -> VerifiedM6ResultStore:
    """Independently reopen a terminal-success M6 store."""

    return _verify_m6_result_store(
        project_root,
        run_name,
        require_success=True,
        allow_data_free=allow_data_free,
        expected_mode=expected_mode,
    )


def _verify_m6_result_store(
    project_root: str | Path,
    run_name: str,
    *,
    require_success: bool,
    allow_data_free: bool,
    expected_mode: str | None,
) -> VerifiedM6ResultStore:
    root = _validated_project_root(project_root)
    name = _validated_run_name(run_name)
    if type(allow_data_free) is not bool:
        raise TypeError("allow_data_free must be an exact bool")
    if expected_mode is not None:
        _profile(expected_mode)
    relative = Path("outputs") / "m6" / name
    _require_git_invisible(root, relative)
    run_path = root / relative
    _guard_run_directory(run_path)
    _validate_marker_exclusivity(run_path)
    if _path_kind(run_path / TERMINAL_FAILURE_MARKER) != "missing":
        raise M6ResultStoreIntegrityError("run is terminally failed")
    if not require_success and _path_kind(
        run_path / TERMINAL_SUCCESS_MARKER
    ) != "missing":
        raise M6ResultStoreIntegrityError(
            "committed verifier requires a pre-terminal checkpoint"
        )

    pending_bytes = _read_guarded_bytes(
        run_path / PENDING_MARKER,
        run_path,
    )
    manifest_bytes = _read_guarded_bytes(
        run_path / MANIFEST_PATH,
        run_path,
    )
    committed_bytes = _read_guarded_bytes(
        run_path / COMMITTED_MARKER,
        run_path,
    )
    success_bytes = (
        _read_guarded_bytes(
            run_path / TERMINAL_SUCCESS_MARKER,
            run_path,
        )
        if require_success
        else None
    )
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    committed_sha256 = hashlib.sha256(committed_bytes).hexdigest()
    if _raw_canonical_sha256_field(
        committed_bytes,
        "manifest_sha256",
        "COMMITTED marker",
    ) != manifest_sha256:
        raise M6ResultStoreIntegrityError(
            "raw COMMITTED marker does not authenticate manifest bytes"
        )
    if success_bytes is not None and (
        _raw_canonical_sha256_field(
            success_bytes,
            "manifest_sha256",
            "TERMINAL_SUCCESS marker",
        )
        != manifest_sha256
        or _raw_canonical_sha256_field(
            success_bytes,
            "committed_sha256",
            "TERMINAL_SUCCESS marker",
        )
        != committed_sha256
    ):
        raise M6ResultStoreIntegrityError(
            "raw TERMINAL_SUCCESS marker does not authenticate manifest/"
            "COMMITTED bytes"
        )

    pending = _decode_canonical_mapping(pending_bytes, "PENDING marker")
    if set(pending) != {
        "capability_sha256",
        "mode",
        "result_path",
        "schema_version",
        "state",
    }:
        raise M6ResultStoreIntegrityError("PENDING marker fields are not exact")
    mode = _json_text(pending["mode"], "mode")
    profile = _profile(mode)
    if expected_mode is not None and mode != expected_mode:
        raise M6ResultStoreIntegrityError("run mode differs from expected mode")
    if profile.data_free and allow_data_free is not True:
        raise M6ResultStoreIntegrityError(
            "data_free stores require explicit verification opt-in"
        )
    capability_sha = _json_text(
        pending["capability_sha256"],
        "capability_sha256",
    )
    if (
        _SHA256.fullmatch(capability_sha) is None
        or pending
        != _pending_payload(name, profile, capability_sha)
    ):
        raise M6ResultStoreIntegrityError(
            "PENDING marker identity/mode binding is invalid"
        )

    manifest = _decode_canonical_mapping(manifest_bytes, "result manifest")
    expected_manifest_fields = {
        "artifacts",
        "capability_sha256",
        "complete",
        "expected_rows",
        "hash_policy",
        "mode",
        "population_size",
        "result_path",
        "row_domain_sha256",
        "schema_fingerprints",
        "schema_version",
    }
    if set(manifest) != expected_manifest_fields:
        raise M6ResultStoreIntegrityError("manifest fields are not exact")
    manifest_schema_version = _json_text(
        manifest["schema_version"],
        "manifest schema_version",
    )
    manifest_mode = _json_text(manifest["mode"], "manifest mode")
    manifest_capability = _json_text(
        manifest["capability_sha256"],
        "manifest capability_sha256",
    )
    manifest_population = _json_integer(
        manifest["population_size"],
        "manifest population_size",
        minimum=1,
    )
    manifest_result_path = _json_text(
        manifest["result_path"],
        "manifest result_path",
    )
    manifest_complete = _json_boolean(
        manifest["complete"],
        "manifest complete",
    )
    manifest_hash_policy = _json_mapping(
        manifest["hash_policy"],
        "manifest hash_policy",
    )
    if set(manifest_hash_policy) != {
        "algorithm",
        "manifest_self_hash",
    }:
        raise M6ResultStoreIntegrityError(
            "manifest hash policy fields are not exact"
        )
    hash_algorithm = _json_text(
        manifest_hash_policy["algorithm"],
        "manifest hash algorithm",
    )
    manifest_self_hash = _json_boolean(
        manifest_hash_policy["manifest_self_hash"],
        "manifest self-hash flag",
    )
    manifest_expected_rows = _json_integer_mapping(
        manifest["expected_rows"],
        "manifest expected_rows",
    )
    manifest_schema_fingerprints = _json_sha256_mapping(
        manifest["schema_fingerprints"],
        "manifest schema_fingerprints",
    )
    manifest_row_domain = _json_text(
        manifest["row_domain_sha256"],
        "manifest row_domain_sha256",
    )
    if (
        _SHA256.fullmatch(manifest_capability) is None
        or _SHA256.fullmatch(manifest_row_domain) is None
        or manifest_schema_version != M6_RESULT_STORE_SCHEMA_VERSION
        or manifest_mode != mode
        or manifest_capability != capability_sha
        or manifest_population != profile.population_size
        or manifest_result_path != relative.as_posix()
        or manifest_complete is not True
        or hash_algorithm != "sha256"
        or manifest_self_hash is not False
    ):
        raise M6ResultStoreIntegrityError(
            "manifest mode/identity boundary is invalid"
        )
    raw_records = manifest["artifacts"]
    if type(raw_records) is not list:
        raise M6ResultStoreIntegrityError("manifest artifacts must be an array")
    records = tuple(
        M6ArtifactRecord.from_dict(
            _json_mapping(item, "manifest artifact")
        )
        for item in raw_records
    )
    paths = tuple(record.path for record in records)
    if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
        raise M6ResultStoreIntegrityError(
            "manifest artifact ordering/domain is noncanonical"
        )
    # Authenticate every listed artifact byte-for-byte before invoking any
    # artifact JSON or Parquet parser.  This prevents a rewritten artifact and
    # rewritten manifest from reaching parser code unless the already-sealed
    # COMMITTED marker authenticates that new manifest.
    for record in records:
        digest, size = _guarded_sha256(
            run_path / record.path,
            run_path,
        )
        if digest != record.sha256 or size != record.size_bytes:
            raise M6ResultStoreIntegrityError(
                f"{record.path} failed size/SHA-256 verification"
            )
    receipt_record = next(
        (
            record
            for record in records
            if record.path == ELIGIBILITY_RECEIPT_PATH
        ),
        None,
    )
    if receipt_record is None:
        raise M6ResultStoreIntegrityError(
            "manifest is missing eligibility receipt"
        )
    receipt_bytes = _read_guarded_bytes(
        run_path / ELIGIBILITY_RECEIPT_PATH,
        run_path,
    )
    receipt = M6EligibilityReceipt.from_dict(
        _decode_canonical_mapping(receipt_bytes, "eligibility receipt")
    )
    if receipt.mode != mode:
        raise M6ResultStoreIntegrityError(
            "eligibility receipt mode differs from PENDING"
        )
    expected_artifact_paths = _expected_artifact_paths(receipt)
    if set(paths) != expected_artifact_paths:
        raise M6ResultStoreIntegrityError(
            "manifest artifact domain differs from mode"
        )
    if MANIFEST_PATH in paths:
        raise M6ResultStoreIntegrityError("manifest cannot list/hash itself")
    if manifest_expected_rows != dict(receipt.expected_rows):
        raise M6ResultStoreIntegrityError(
            "manifest rows differ from eligibility receipt"
        )
    expected_schema_fingerprints = _schema_fingerprints_for_receipt(receipt)
    if manifest_schema_fingerprints != expected_schema_fingerprints:
        raise M6ResultStoreIntegrityError(
            "manifest schema fingerprint catalog drifted"
        )
    if manifest_row_domain != _row_domain_sha256(receipt):
        raise M6ResultStoreIntegrityError("manifest row-domain hash drifted")

    allowed_files = {
        PENDING_MARKER,
        MANIFEST_PATH,
        COMMITTED_MARKER,
        *expected_artifact_paths,
    }
    if require_success:
        allowed_files.add(TERMINAL_SUCCESS_MARKER)
    _validate_run_tree(run_path, allowed_files=allowed_files)
    record_by_path = {record.path: record for record in records}
    tables: dict[str, pa.Table] = {}
    waymax_selection_receipt: M6WaymaxSelectionReceipt | None = None
    for record in records:
        path = run_path / record.path
        dataset = _dataset_for_path(record.path)
        if dataset is not None:
            if record.schema_identity != dataset or record.rows is None:
                raise M6ResultStoreIntegrityError(
                    "manifest dataset identity/rows drifted"
                )
            table = _read_guarded_parquet(path, run_path, dataset)
            if table.num_rows != record.rows:
                raise M6ResultStoreIntegrityError(
                    f"{dataset} row count drifted"
                )
            tables[dataset] = table
        else:
            expected_identity = {
                ELIGIBILITY_RECEIPT_PATH: (
                    M6_ELIGIBILITY_RECEIPT_SCHEMA_VERSION
                ),
                WAYMAX_SELECTION_RECEIPT_PATH: (
                    M6_WAYMAX_SELECTION_RECEIPT_SCHEMA_VERSION
                ),
                DETERMINISM_RECEIPT_PATH: (
                    M6_DETERMINISM_RECEIPT_SCHEMA_VERSION
                ),
                CLAIM_LIMITATIONS_PATH: (
                    M6_CLAIM_LIMITATIONS_SCHEMA_VERSION
                ),
            }.get(record.path)
            if (
                expected_identity is None
                or record.schema_identity != expected_identity
                or record.rows is not None
            ):
                raise M6ResultStoreIntegrityError(
                    "manifest JSON artifact identity drifted"
                )
            if record.path == WAYMAX_SELECTION_RECEIPT_PATH:
                waymax_selection_receipt = (
                    M6WaymaxSelectionReceipt.from_dict(
                        _decode_canonical_mapping(
                            _read_guarded_bytes(path, run_path),
                            "Waymax selection receipt",
                        )
                    )
                )

    if (
        waymax_selection_receipt is None
        or waymax_selection_receipt.mode != mode
    ):
        raise M6ResultStoreIntegrityError(
            "Waymax selection receipt is missing or mode-mismatched"
        )

    _verify_semantic_artifacts(
        run_path,
        profile,
        receipt,
        tables,
        waymax_selection_receipt,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        live_waymax_selection=None,
        review_precursor_sha256=(
            _review_precursor_sha256(receipt, records)
            if profile.complete_results
            else None
        ),
    )
    for dataset, expected_rows in receipt.expected_rows.items():
        if (
            tables[dataset].num_rows != expected_rows
            or record_by_path[_DATASET_PATHS[dataset]].rows != expected_rows
        ):
            raise M6ResultStoreIntegrityError(
                f"{dataset} differs from exact receipt rows"
            )

    committed = _decode_canonical_mapping(committed_bytes, "COMMITTED marker")
    expected_committed = {
        "expected_rows": dict(receipt.expected_rows),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_size_bytes": len(manifest_bytes),
        "mode": mode,
        "row_domain_sha256": _row_domain_sha256(receipt),
        "schema_fingerprints": expected_schema_fingerprints,
        "schema_version": M6_RESULT_STORE_SCHEMA_VERSION,
        "state": "COMMITTED",
    }
    if set(committed) != set(expected_committed):
        raise M6ResultStoreIntegrityError(
            "COMMITTED marker fields are not exact"
        )
    committed_manifest_sha = _json_text(
        committed["manifest_sha256"],
        "COMMITTED manifest_sha256",
    )
    committed_row_domain = _json_text(
        committed["row_domain_sha256"],
        "COMMITTED row_domain_sha256",
    )
    committed_schema_fingerprints = _json_sha256_mapping(
        committed["schema_fingerprints"],
        "COMMITTED schema_fingerprints",
    )
    normalized_committed = {
        "expected_rows": _json_integer_mapping(
            committed["expected_rows"],
            "COMMITTED expected_rows",
        ),
        "manifest_sha256": committed_manifest_sha,
        "manifest_size_bytes": _json_integer(
            committed["manifest_size_bytes"],
            "COMMITTED manifest_size_bytes",
            minimum=1,
        ),
        "mode": _json_text(committed["mode"], "COMMITTED mode"),
        "row_domain_sha256": committed_row_domain,
        "schema_fingerprints": committed_schema_fingerprints,
        "schema_version": _json_text(
            committed["schema_version"],
            "COMMITTED schema_version",
        ),
        "state": _json_text(committed["state"], "COMMITTED state"),
    }
    if (
        _SHA256.fullmatch(committed_manifest_sha) is None
        or _SHA256.fullmatch(committed_row_domain) is None
        or normalized_committed != expected_committed
    ):
        raise M6ResultStoreIntegrityError(
            "COMMITTED does not bind manifest/mode/schema/rows"
        )
    if require_success:
        assert success_bytes is not None
        success = _decode_canonical_mapping(
            success_bytes,
            "TERMINAL_SUCCESS marker",
        )
        expected_success_fields = {
            "committed_sha256": hashlib.sha256(committed_bytes).hexdigest(),
            "evidence_catalog_sha256": success.get(
                "evidence_catalog_sha256"
            ),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "mode": mode,
            "observed_preflight_sha256": success.get(
                "observed_preflight_sha256"
            ),
            "schema_version": M6_RESULT_STORE_SCHEMA_VERSION,
            "state": "TERMINAL_SUCCESS",
        }
        evidence_catalog_sha256 = _json_text(
            success.get("evidence_catalog_sha256"),
            "TERMINAL_SUCCESS evidence_catalog_sha256",
        )
        observed_preflight_sha256 = _json_text(
            success.get("observed_preflight_sha256"),
            "TERMINAL_SUCCESS observed_preflight_sha256",
        )
        if (
            set(success) != set(expected_success_fields)
            or _SHA256.fullmatch(evidence_catalog_sha256) is None
            or _SHA256.fullmatch(observed_preflight_sha256) is None
            or success != expected_success_fields
        ):
            raise M6ResultStoreIntegrityError(
                "TERMINAL_SUCCESS binding is invalid"
            )
        if profile.complete_results:
            reviews = tables[REVIEW_DECISIONS].to_pylist()
            review_digests = {
                row["evidence_catalog_sha256"] for row in reviews
            }
            if review_digests != {evidence_catalog_sha256}:
                raise M6ResultStoreIntegrityError(
                    "TERMINAL_SUCCESS evidence catalog differs from reviews"
                )
        if profile.data_free:
            expected_self = hashlib.sha256(
                b"evalsim-m6-data-free-self-verification-v1\x00"
                + bytes.fromhex(
                    hashlib.sha256(manifest_bytes).hexdigest()
                )
                + bytes.fromhex(
                    hashlib.sha256(committed_bytes).hexdigest()
                )
                + bytes.fromhex(evidence_catalog_sha256)
            ).hexdigest()
            if observed_preflight_sha256 != expected_self:
                raise M6ResultStoreIntegrityError(
                    "data_free terminal self-verification binding drifted"
                )
    return VerifiedM6ResultStore(
        run_path=run_path,
        profile=profile,
        receipt=receipt,
        waymax_selection_receipt=waymax_selection_receipt,
        manifest=MappingProxyType(dict(manifest)),
        artifacts=records,
        tables=MappingProxyType(tables),
    )


def _verify_uncommitted_artifacts(
    run_path: Path,
    profile: M6ResultProfile,
    receipt: M6EligibilityReceipt,
    records: Sequence[M6ArtifactRecord],
    *,
    waymax_selection: M6WaymaxSelection | None,
) -> None:
    expected_paths = _expected_artifact_paths(receipt)
    if {record.path for record in records} != expected_paths:
        raise M6ResultStoreIntegrityError(
            "uncommitted mode artifact domain is incomplete"
        )
    allowed = {PENDING_MARKER, *expected_paths}
    _validate_run_tree(run_path, allowed_files=allowed)
    tables: dict[str, pa.Table] = {}
    disk_receipt: M6EligibilityReceipt | None = None
    waymax_selection_receipt: M6WaymaxSelectionReceipt | None = None
    for record in records:
        path = run_path / record.path
        digest, size = _guarded_sha256(path, run_path)
        if digest != record.sha256 or size != record.size_bytes:
            raise M6ResultStoreIntegrityError(
                "uncommitted artifact digest/size drifted"
            )
        dataset = _dataset_for_path(record.path)
        if dataset is not None:
            table = _read_guarded_parquet(path, run_path, dataset)
            if record.rows != table.num_rows:
                raise M6ResultStoreIntegrityError(
                    "uncommitted artifact rows drifted"
                )
            tables[dataset] = table
        elif record.path == ELIGIBILITY_RECEIPT_PATH:
            if (
                record.schema_identity
                != M6_ELIGIBILITY_RECEIPT_SCHEMA_VERSION
                or record.rows is not None
            ):
                raise M6ResultStoreIntegrityError(
                    "uncommitted eligibility receipt identity drifted"
                )
            disk_receipt = M6EligibilityReceipt.from_dict(
                _decode_canonical_mapping(
                    _read_guarded_bytes(path, run_path),
                    "eligibility receipt",
                )
            )
        elif record.path == WAYMAX_SELECTION_RECEIPT_PATH:
            if (
                record.schema_identity
                != M6_WAYMAX_SELECTION_RECEIPT_SCHEMA_VERSION
                or record.rows is not None
            ):
                raise M6ResultStoreIntegrityError(
                    "uncommitted Waymax selection receipt identity drifted"
                )
            waymax_selection_receipt = M6WaymaxSelectionReceipt.from_dict(
                _decode_canonical_mapping(
                    _read_guarded_bytes(path, run_path),
                    "Waymax selection receipt",
                )
            )
        elif record.path == DETERMINISM_RECEIPT_PATH:
            if (
                record.schema_identity
                != M6_DETERMINISM_RECEIPT_SCHEMA_VERSION
                or record.rows is not None
            ):
                raise M6ResultStoreIntegrityError(
                    "uncommitted determinism receipt identity drifted"
                )
        elif record.path == CLAIM_LIMITATIONS_PATH:
            if (
                record.schema_identity
                != M6_CLAIM_LIMITATIONS_SCHEMA_VERSION
                or record.rows is not None
            ):
                raise M6ResultStoreIntegrityError(
                    "uncommitted claim/limitations identity drifted"
                )
        else:
            raise M6ResultStoreIntegrityError(
                "uncommitted JSON artifact domain drifted"
            )
    if (
        disk_receipt is None
        or disk_receipt.to_dict() != receipt.to_dict()
    ):
        raise M6ResultStoreIntegrityError(
            "uncommitted eligibility receipt bytes drifted"
        )
    if (
        waymax_selection_receipt is None
        or waymax_selection_receipt.mode != receipt.mode
    ):
        raise M6ResultStoreIntegrityError(
            "uncommitted Waymax selection receipt is missing/mismatched"
        )
    _verify_semantic_artifacts(
        run_path,
        profile,
        receipt,
        tables,
        waymax_selection_receipt,
        manifest_sha256=None,
        live_waymax_selection=waymax_selection,
        review_precursor_sha256=(
            _review_precursor_sha256(receipt, records)
            if profile.complete_results
            else None
        ),
    )


def _verify_semantic_artifacts(
    run_path: Path,
    profile: M6ResultProfile,
    receipt: M6EligibilityReceipt,
    tables: Mapping[str, pa.Table],
    waymax_selection_receipt: M6WaymaxSelectionReceipt,
    *,
    manifest_sha256: str | None,
    live_waymax_selection: M6WaymaxSelection | None,
    review_precursor_sha256: str | None,
) -> None:
    eligibility = _normalize_eligibility(
        tables[ELIGIBILITY_LEDGER].to_pylist(),
        profile,
        expected_receipt=receipt,
    )
    del eligibility
    qualification = _normalize_waymax_qualification(
        tables[WAYMAX_QUALIFICATION].to_pylist(),
        receipt,
    )
    stored_members: tuple[tuple[int, int, str], ...] | None = None
    if profile.data_free:
        if live_waymax_selection is not None:
            raise M6ResultStoreIntegrityError(
                "data_free verification cannot accept a live Waymax selection"
            )
        expected_selection_receipt = (
            _data_free_waymax_selection_receipt(receipt)
        )
        if (
            waymax_selection_receipt.to_dict()
            != expected_selection_receipt.to_dict()
            or qualification
            != m6_data_free_waymax_qualification_rows(
                receipt.eligible_cohort_indices
            )
        ):
            raise M6ResultStoreIntegrityError(
                "data_free Waymax selection boundary is not exact"
            )
    elif live_waymax_selection is not None:
        if (
            manifest_sha256 is not None
            or not isinstance(live_waymax_selection, M6WaymaxSelection)
        ):
            raise M6ResultStoreIntegrityError(
                "pre-commit Waymax verification boundary is contradictory"
            )
        _verify_waymax_selection_receipt_against_selection(
            waymax_selection_receipt,
            live_waymax_selection,
            receipt,
        )
        if qualification != _waymax_qualification_rows_from_selection(
            live_waymax_selection,
            receipt,
        ):
            raise M6ResultStoreIntegrityError(
                "stored Waymax qualification differs from the canonical "
                "typed selection ledger"
            )
    else:
        if (
            manifest_sha256 is None
            or _SHA256.fullmatch(manifest_sha256) is None
        ):
            raise M6ResultStoreIntegrityError(
                "reopened Waymax verification requires the verified manifest "
                "binding"
            )
        stored_members = (
            _verify_waymax_selection_receipt_against_stored_qualification(
                waymax_selection_receipt,
                qualification,
                receipt,
            )
        )
    if profile.mode == ELIGIBILITY_ONLY_MODE:
        if set(tables) != {
            ELIGIBILITY_LEDGER,
            WAYMAX_QUALIFICATION,
        }:
            raise M6ResultStoreIntegrityError(
                "eligibility_only contains outcome artifacts"
            )
        return
    if profile.mode == COMPUTE_PILOT_MODE:
        if set(tables) != {
            ELIGIBILITY_LEDGER,
            WAYMAX_QUALIFICATION,
            COMPUTE_PILOT_SUMMARY,
        }:
            raise M6ResultStoreIntegrityError(
                "compute_pilot table domain drifted"
            )
        _normalize_compute_pilot(
            tables[COMPUTE_PILOT_SUMMARY].to_pylist(),
            receipt,
        )
        return

    expected_tables = set(receipt.expected_rows)
    if set(tables) != expected_tables:
        raise M6ResultStoreIntegrityError(
            "complete-result table domain drifted"
        )
    primary_scene = _normalize_primary_scene_scalars(
        tables[PRIMARY_SCENE_SCALARS].to_pylist(),
        receipt,
    )
    primary_matrix = _normalize_primary_matrix(
        tables[PRIMARY_MATRIX].to_pylist(),
        receipt,
    )
    independently_reconstructed = _derive_primary_matrix_rows(
        primary_scene,
        receipt,
    )
    if primary_matrix != independently_reconstructed:
        raise M6ResultStoreIntegrityError(
            "sealed primary matrix differs from stats.m6 reconstruction"
        )
    repeat_scene = _normalize_primary_scene_scalars(
        tables[PRIMARY_REPEAT_SCENE_SCALARS].to_pylist(),
        receipt,
    )
    repeat_matrix = _normalize_primary_matrix(
        tables[PRIMARY_REPEAT_MATRIX].to_pylist(),
        receipt,
    )
    if (
        repeat_matrix != _derive_primary_matrix_rows(repeat_scene, receipt)
        or _canonical_rows_sha256(
            PRIMARY_SCENE_SCALARS,
            repeat_scene,
        )
        != _canonical_rows_sha256(
            PRIMARY_SCENE_SCALARS,
            primary_scene,
        )
        or _canonical_rows_sha256(PRIMARY_MATRIX, repeat_matrix)
        != _canonical_rows_sha256(PRIMARY_MATRIX, primary_matrix)
    ):
        raise M6ResultStoreIntegrityError(
            "sealed primary pass-2 logical rows differ from pass 1"
        )
    secondary_scene = _normalize_secondary_scene_scalars(
        tables[SECONDARY_SCENE_SCALARS].to_pylist(),
        receipt,
    )
    secondary_matrix = _normalize_secondary_matrix(
        tables[SECONDARY_MATRIX].to_pylist(),
        receipt,
    )
    if secondary_matrix != _derive_secondary_matrix_rows(
        secondary_scene,
        receipt,
    ):
        raise M6ResultStoreIntegrityError(
            "sealed b4 matrix differs from its complete scene rows"
        )
    observations = _normalize_negative_timing_observations(
        tables[NEGATIVE_TIMING_OBSERVATIONS].to_pylist(),
        receipt,
    )
    gates = _normalize_negative_timing_gates(
        tables[NEGATIVE_TIMING_GATES].to_pylist(),
        receipt,
    )
    if gates != _derive_negative_timing_gates(observations, receipt):
        raise M6ResultStoreIntegrityError(
            "negative/timing gates differ from sealed per-case observations"
        )
    qualification = _normalize_waymax_qualification(
        tables[WAYMAX_QUALIFICATION].to_pylist(),
        receipt,
    )
    scalar_rows = _normalize_waymax_scene_scalars_from_qualification(
        tables[WAYMAX_SCENE_SCALARS].to_pylist(),
        receipt,
        qualification,
        waymax_selection_receipt,
    )
    parsed_scalar_table = parse_m6_waymax_scene_scalar_table(scalar_rows)
    parsed_scalar_table.revalidate()
    stored_reconstruction = None
    if profile.data_free:
        if scalar_rows != m6_data_free_waymax_scene_scalar_rows():
            raise M6ResultStoreIntegrityError(
                "data_free Waymax scalar placeholders are not exact"
            )
    elif live_waymax_selection is not None:
        stored_reconstruction = reconstruct_m6_waymax_stored_cells(
            parsed_scalar_table,
            selection=live_waymax_selection,
            verified_selection_binding_sha256=(
                waymax_selection_receipt.selection_binding_sha256
            ),
            intervention_configuration_fingerprint=(
                waymax_selection_receipt.primary_b2_configuration_fingerprint
            ),
        )
    else:
        assert manifest_sha256 is not None
        assert stored_members is not None
        stored_selection = verify_m6_waymax_stored_selection(
            parsed_scalar_table,
            manifest_sha256=manifest_sha256,
            selection_binding_sha256=(
                waymax_selection_receipt.selection_binding_sha256
            ),
            primary_domain_sha256=(
                waymax_selection_receipt.primary_domain_sha256
            ),
            supported=waymax_selection_receipt.selection_supported,
            members=stored_members,
        )
        if stored_selection.promotable is not False:
            raise M6ResultStoreIntegrityError(
                "verified stored Waymax selection became promotable"
            )
        stored_reconstruction = reconstruct_m6_waymax_stored_cells(
            parsed_scalar_table,
            stored_selection=stored_selection,
            intervention_configuration_fingerprint=(
                waymax_selection_receipt.primary_b2_configuration_fingerprint
            ),
        )
    if stored_reconstruction is not None:
        if stored_reconstruction.promotable is not False:
            raise M6ResultStoreIntegrityError(
                "reopened Waymax scalar reconstruction became promotable"
            )
    comparison_rows = _normalize_waymax_field_comparisons_from_qualification(
        tables[WAYMAX_FIELD_COMPARISONS].to_pylist(),
        receipt,
        qualification,
    )
    determinism_rows = _normalize_waymax_determinism_from_qualification(
        tables[WAYMAX_DETERMINISM].to_pylist(),
        receipt,
        qualification,
    )
    waymax = _normalize_waymax_accounting(
        tables[WAYMAX_ACCOUNTING].to_pylist(),
        receipt,
    )
    stored_cells = tuple(
        row for row in waymax if row["record_type"] == "secondary_cell"
    )
    if stored_reconstruction is not None and any(
        row["pair_n"] != stored_reconstruction.pair_n
        or row["status"] != stored_reconstruction.status
        for row in stored_cells
    ):
        raise M6ResultStoreIntegrityError(
            "Waymax stored cell accounting differs from non-promotable "
            "reconstruction"
        )
    if waymax != _derive_waymax_accounting(
        qualification,
        scalar_rows,
        comparison_rows,
        receipt,
        waymax_selection_receipt,
        matrix=None,
        stored_cell_rows=(
            None if profile.data_free else stored_cells
        ),
    ):
        raise M6ResultStoreIntegrityError(
            "Waymax accounting differs from sealed qualification/scalar/"
            "comparison rows"
        )
    provenance = _normalize_typed_provenance(
        tables[TYPED_PROVENANCE].to_pylist(),
        receipt,
    )
    del provenance
    execution = _normalize_execution_summary(
        tables[EXECUTION_SUMMARY].to_pylist(),
        receipt,
    )[0]
    stage_rows = _normalize_stage_timings(
        tables[STAGE_TIMINGS].to_pylist(),
        receipt,
    )
    review_rows = _normalize_review_decisions(
        tables[REVIEW_DECISIONS].to_pylist(),
        receipt,
        expected_evidence_catalog_sha256=review_precursor_sha256,
    )
    if any(row["status"] == "failed" for row in gates):
        raise M6ResultStoreIntegrityError(
            "failed negative/timing gate cannot reach COMMITTED"
        )
    determinism = M6DeterminismReceipt.from_dict(
        _decode_canonical_mapping(
            _read_guarded_bytes(
                run_path / DETERMINISM_RECEIPT_PATH,
                run_path,
            ),
            "determinism receipt",
        )
    )
    scene_digest = _canonical_rows_sha256(
        PRIMARY_SCENE_SCALARS,
        primary_scene,
    )
    matrix_digest = _canonical_rows_sha256(
        PRIMARY_MATRIX,
        primary_matrix,
    )
    repeat_scene_digest = _canonical_rows_sha256(
        PRIMARY_SCENE_SCALARS,
        repeat_scene,
    )
    repeat_matrix_digest = _canonical_rows_sha256(
        PRIMARY_MATRIX,
        repeat_matrix,
    )
    selected_determinism = [
        row
        for row in determinism_rows
        if row["status"] != "not_applicable"
    ]
    expected_repeat_status = (
        "not_applicable"
        if not selected_determinism
        else (
            "passed"
            if all(row["status"] == "passed" for row in selected_determinism)
            else "failed"
        )
    )
    if (
        determinism.mode != profile.mode
        or determinism.primary_scene_pass_1_sha256 != scene_digest
        or determinism.primary_scene_pass_2_sha256 != repeat_scene_digest
        or determinism.primary_matrix_pass_1_sha256 != matrix_digest
        or determinism.primary_matrix_pass_2_sha256 != repeat_matrix_digest
        or determinism.waymax_repeat_status != expected_repeat_status
        or determinism.waymax_repeat_rows != len(selected_determinism)
    ):
        raise M6ResultStoreIntegrityError(
            "determinism receipt does not bind both passes / Waymax scope"
        )
    expected_execution = _derive_execution_summary(
        receipt=receipt,
        tables={
            name: tuple(table.to_pylist())
            for name, table in tables.items()
            if name != EXECUTION_SUMMARY
        },
        determinism=determinism,
        fresh_worker_peak_rss_bytes=execution[
            "fresh_worker_peak_rss_bytes"
        ],
    )
    if execution != expected_execution:
        raise M6ResultStoreIntegrityError(
            "execution summary differs from mechanically derived evidence"
        )
    del stage_rows, review_rows
    claim = _decode_canonical_mapping(
        _read_guarded_bytes(
            run_path / CLAIM_LIMITATIONS_PATH,
            run_path,
        ),
        "claim/limitations",
    )
    if claim != _claim_limitations_payload(profile.mode):
        raise M6ResultStoreIntegrityError(
            "claim/limitations artifact is not exact"
        )


def reconstruct_sanitized_m6_aggregate(
    project_root: str | Path,
    run_name: str,
    *,
    allow_data_free: bool = False,
) -> M6SanitizedAggregate:
    """Reopen a terminal official store and emit the exact seven public domains.

    Data-free stores are deliberately non-promotable: their synthetic N=10 evidence
    cannot truthfully carry the official WOMD population and shard claims.
    """

    verified = verify_m6_result_store(
        project_root,
        run_name,
        allow_data_free=allow_data_free,
    )
    if verified.profile.mode != OFFICIAL_MODE:
        raise M6ResultStoreIntegrityError(
            "only a terminal official store can be promoted; data-free evidence is "
            "synthetic and non-promotable"
        )
    receipt = verified.receipt
    matrix = _normalize_primary_matrix(
        verified.read_dataset(PRIMARY_MATRIX).to_pylist(),
        receipt,
    )
    promoted_matrix = [_promoted_primary_row(row) for row in matrix]
    if len(promoted_matrix) != 12 or any(
        tuple(row) != M6_PROMOTED_PRIMARY_FIELDS
        for row in promoted_matrix
    ):
        raise M6ResultStoreIntegrityError(
            "promoted primary matrix is not exact 12-row schema"
        )
    gates = _normalize_negative_timing_gates(
        verified.read_dataset(NEGATIVE_TIMING_GATES).to_pylist(),
        receipt,
    )
    waymax_rows = _normalize_waymax_accounting(
        verified.read_dataset(WAYMAX_ACCOUNTING).to_pylist(),
        receipt,
    )
    execution = _normalize_execution_summary(
        verified.read_dataset(EXECUTION_SUMMARY).to_pylist(),
        receipt,
    )[0]
    stages = _normalize_stage_timings(
        verified.read_dataset(STAGE_TIMINGS).to_pylist(),
        receipt,
    )
    reviews = _normalize_review_decisions(
        verified.read_dataset(REVIEW_DECISIONS).to_pylist(),
        receipt,
    )
    aggregate = {
        "claim_and_limitations": {
            "bounded_claim": M6_ACCEPTED_BOUNDED_CLAIM,
            "limitations": list(M6_FIXED_LIMITATIONS),
        },
        "eligibility": {
            "primary_eligible_count": receipt.eligible_count,
            "rejection_reason_counts": dict(
                receipt.rejection_reason_counts
            ),
            "total": receipt.population_size,
        },
        "execution": {
            "aggregate_stage_durations_ms": {
                row["stage_name"]: row["duration_ms"] for row in stages
            },
            "deterministic_repeat_status": execution[
                "deterministic_repeat_status"
            ],
            "fresh_worker_peak_rss_bytes": execution[
                "fresh_worker_peak_rss_bytes"
            ],
            "gate_status": {
                "release": execution["release_gate_status"],
                "real_reactivity_claim": execution[
                    "real_reactivity_claim_status"
                ],
                "waymax": execution["waymax_gate_status"],
            },
            "required_row_domain_counts": {
                key: value
                for key, value in execution.items()
                if key.endswith("_rows")
            },
            "review_decisions": [
                {
                    "decision": row["decision"],
                    "p1_count": row["p1_count"],
                    "p2_count": row["p2_count"],
                    "p3_count": row["p3_count"],
                    "role": row["role"],
                }
                for row in reviews
            ],
        },
        "negative_control_and_timing_gates": [
            {
                "assessed_n": row["assessed_n"],
                "gate_name": row["gate_name"],
                "passed_n": row["passed_n"],
                "status": row["status"],
                "violation_n": row["violation_n"],
            }
            for row in gates
        ],
        "primary_matrix": promoted_matrix,
        "provenance_labels": {
            "aggregate_schema_version": (
                M6_PROMOTED_AGGREGATE_SCHEMA_VERSION
            ),
            "config_version": M6_CONFIG_VERSION,
            "fixed_limitations": list(M6_FIXED_LIMITATIONS),
            "horizons": {
                "numpy_transitions": 40,
                "waymax_transitions": 20,
            },
            "interventions": [
                {
                    "deceleration_mps2": 0.0,
                    "name": "identity",
                    "version": M6_INTERVENTION_VERSION,
                },
                {
                    "deceleration_mps2": 2.0,
                    "duration_s": 1.0,
                    "name": "longitudinal_brake_pulse",
                    "role": "primary",
                    "version": M6_INTERVENTION_VERSION,
                },
                {
                    "deceleration_mps2": 4.0,
                    "duration_s": 1.0,
                    "name": "longitudinal_brake_pulse",
                    "role": "local_secondary_not_numeric_public",
                    "version": M6_INTERVENTION_VERSION,
                },
            ],
            "plan_version": M6_PLAN_VERSION,
            "policies": [
                {"access_role": access, "name": policy}
                for policy, access in M6_PRIMARY_POLICY_ROLES
            ],
            "population_label": (
                "accepted_m4_complete_case_ten_shard_cohort"
            ),
            "result_schema_version": M6_RESULT_STORE_SCHEMA_VERSION,
            "source_shard_suffix_range": ["00000", "00009"],
            "statistics_schema_version": M6_STATISTICS_SCHEMA_VERSION,
        },
        "waymax_scope": _promoted_waymax_scope(waymax_rows),
    }
    if tuple(aggregate) != tuple(sorted(M6_PROMOTED_TOP_LEVEL_DOMAINS)):
        raise M6ResultStoreIntegrityError(
            "promoted aggregate top-level domains drifted"
        )
    _assert_promoted_privacy(aggregate)
    canonical = _canonical_json_text(aggregate)
    return M6SanitizedAggregate(
        canonical,
        _factory_sentinel=_AGGREGATE_SENTINEL,
    )


def _promoted_primary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metric_name": row["metric_name"],
        "metric_version": row["metric_version"],
        "unit": row["unit"],
        "policy_name": row["policy_name"],
        "policy_access_role": row["policy_access_role"],
        "pair_n": row["pair_n"],
        "thresholded_nonzero_n": row["thresholded_nonzero_n"],
        "responder_n": row["responder_n"],
        "censor_n": row["censor_n"],
        "arithmetic_mean": row["arithmetic_mean"],
        "median": row["median"],
        "pointwise_band": {
            "level": row["pointwise_level"],
            "lower": row["pointwise_lower"],
            "upper": row["pointwise_upper"],
        },
        "adjusted_band": {
            "level": row["adjusted_level"],
            "lower": row["adjusted_lower"],
            "upper": row["adjusted_upper"],
        },
        "status": row["status"],
        "suppression_reason": row["suppression_reason"],
        "source_pairing_complete": row["source_pairing_complete"],
        "directional_language_allowed": row[
            "directional_language_allowed"
        ],
    }


def _promoted_waymax_scope(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scope = {
        row["name"]: row["count"]
        for row in rows
        if row["record_type"] == "scope"
    }
    rejections = {
        row["name"]: row["count"]
        for row in rows
        if row["record_type"] == "selection_rejection"
    }
    comparisons = [
        {
            "binary_mismatches": row["binary_mismatches"],
            "bundle": row["bundle"],
            "comparison_kind": row["comparison_kind"],
            "condition": row["condition"],
            "denominator": row["denominator"],
            "field": row["name"],
            "max_abs_error": row["max_abs_error"],
            "status": row["status"],
            "tolerance_failures": row["tolerance_failures"],
        }
        for row in rows
        if row["record_type"] == "field_comparison"
    ]
    controls = {
        row["name"]: {
            "count": row["count"],
            "opportunity_n": row["opportunity_n"],
            "status": row["status"],
        }
        for row in rows
        if row["record_type"] == "control_partition"
    }
    cells = []
    for row in rows:
        if row["record_type"] != "secondary_cell":
            continue
        cells.append(
            {
                "arithmetic_mean": row["arithmetic_mean"],
                "bundle": row["bundle"],
                "censor_n": row["censor_n"],
                "median": row["median"],
                "metric_name": row["metric_name"],
                "metric_version": row["metric_version"],
                "pair_n": row["pair_n"],
                "pointwise_band": (
                    None
                    if row["pointwise_level"] is None
                    else {
                        "level": row["pointwise_level"],
                        "lower": row["pointwise_lower"],
                        "upper": row["pointwise_upper"],
                    }
                ),
                "responder_n": row["responder_n"],
                "source_pairing_complete": row[
                    "source_pairing_complete"
                ],
                "directional_language_allowed": row[
                    "directional_language_allowed"
                ],
                "status": row["status"],
                "suppression_reason": row["suppression_reason"],
                "thresholded_nonzero_n": row[
                    "thresholded_nonzero_n"
                ],
                "unit": row["unit"],
            }
        )
    if len(cells) != 8:
        raise M6ResultStoreIntegrityError(
            "promoted Waymax cell domain must have exactly eight rows"
        )
    return {
        "cell_rows": cells,
        "control_accounting": controls,
        "field_comparisons": comparisons,
        "qualified_count": scope["qualified_count"],
        "rejection_reason_counts": rejections,
        "selected_count": scope["selected_count"],
        "selection_rule": "16_or_floor",
        "transition_count": scope["transition_count"],
    }


def _expected_artifact_paths(
    receipt: M6EligibilityReceipt,
) -> set[str]:
    paths = {
        _DATASET_PATHS[ELIGIBILITY_LEDGER],
        _DATASET_PATHS[WAYMAX_QUALIFICATION],
        ELIGIBILITY_RECEIPT_PATH,
        WAYMAX_SELECTION_RECEIPT_PATH,
    }
    if receipt.mode == COMPUTE_PILOT_MODE:
        paths.add(_DATASET_PATHS[COMPUTE_PILOT_SUMMARY])
    elif receipt.mode in {OFFICIAL_MODE, DATA_FREE_MODE}:
        paths.update(
            {
                _DATASET_PATHS[name]
                for name in (
                    PRIMARY_SCENE_SCALARS,
                    PRIMARY_MATRIX,
                    PRIMARY_REPEAT_SCENE_SCALARS,
                    PRIMARY_REPEAT_MATRIX,
                    SECONDARY_SCENE_SCALARS,
                    SECONDARY_MATRIX,
                    NEGATIVE_TIMING_OBSERVATIONS,
                    NEGATIVE_TIMING_GATES,
                    WAYMAX_ACCOUNTING,
                    WAYMAX_SCENE_SCALARS,
                    WAYMAX_FIELD_COMPARISONS,
                    WAYMAX_DETERMINISM,
                    TYPED_PROVENANCE,
                    EXECUTION_SUMMARY,
                    STAGE_TIMINGS,
                    REVIEW_DECISIONS,
                )
            }
        )
        paths.update(
            {
                DETERMINISM_RECEIPT_PATH,
                CLAIM_LIMITATIONS_PATH,
            }
        )
    return paths


def _review_precursor_sha256(
    receipt: M6EligibilityReceipt,
    records: Sequence[M6ArtifactRecord],
) -> str:
    """Bind reviews to every sealed prerequisite without a circular self-hash."""

    excluded = {
        _DATASET_PATHS[REVIEW_DECISIONS],
        _DATASET_PATHS[EXECUTION_SUMMARY],
    }
    expected_paths = _expected_artifact_paths(receipt) - excluded
    all_expected_paths = _expected_artifact_paths(receipt)
    by_path: dict[str, M6ArtifactRecord] = {}
    input_paths: set[str] = set()
    for record in records:
        if not isinstance(record, M6ArtifactRecord):
            raise M6ResultStoreIntegrityError(
                "review precursor catalog contains a non-artifact record"
            )
        if record.path in input_paths:
            raise M6ResultStoreIntegrityError(
                "review precursor catalog contains duplicate paths"
            )
        input_paths.add(record.path)
        if record.path not in excluded:
            by_path[record.path] = record
    if (
        input_paths != expected_paths
        and input_paths != all_expected_paths
    ):
        raise M6ResultStoreStateError(
            "all noncircular review prerequisite artifacts must be sealed first"
        )
    catalog = {
        "artifacts": [
            by_path[path].to_dict() for path in sorted(by_path)
        ],
        "mode": receipt.mode,
        "schema_version": M6_RESULT_STORE_SCHEMA_VERSION,
    }
    return hashlib.sha256(
        b"evalsim-m6-review-precursor-catalog-v1\x00"
        + _canonical_json_bytes(catalog)
    ).hexdigest()


def _claim_limitations_payload(mode: str) -> dict[str, Any]:
    if mode not in {OFFICIAL_MODE, DATA_FREE_MODE}:
        raise M6ResultStoreIntegrityError(
            "claim/limitations are complete-result-only"
        )
    data_free = mode == DATA_FREE_MODE
    return {
        "bounded_claim": (
            _M6_DATA_FREE_BOUNDED_CLAIM
            if data_free
            else M6_ACCEPTED_BOUNDED_CLAIM
        ),
        "limitations": list(
            _M6_DATA_FREE_LIMITATIONS if data_free else M6_FIXED_LIMITATIONS
        ),
        "mode": mode,
        "schema_version": M6_CLAIM_LIMITATIONS_SCHEMA_VERSION,
    }


def _pending_payload(
    run_name: str,
    profile: M6ResultProfile,
    capability_sha256: str,
) -> dict[str, Any]:
    return {
        "capability_sha256": capability_sha256,
        "mode": profile.mode,
        "result_path": f"outputs/m6/{run_name}",
        "schema_version": M6_RESULT_STORE_SCHEMA_VERSION,
        "state": "PENDING",
    }


def _profile(value: Any) -> M6ResultProfile:
    if type(value) is not str or value not in _PROFILES:
        raise M6ResultStoreIntegrityError("unregistered M6 mode")
    return _PROFILES[value]


def _ordered_indices(
    values: Iterable[Any],
    *,
    population_size: int,
    name: str,
) -> tuple[int, ...]:
    indices = tuple(
        _integer(value, name=name, minimum=0, maximum=population_size - 1)
        for value in values
    )
    if indices != tuple(sorted(set(indices))):
        raise ValueError(f"{name} must be unique and sorted")
    return indices


def _primary_status(
    pair_n: int,
    nonzero_n: int,
    adjusted_lower: float,
    adjusted_upper: float,
) -> tuple[str, str | None]:
    if nonzero_n < 10:
        return "event_sparse", "thresholded_nonzero_n_below_10"
    if pair_n < 30:
        return "small_n", "pair_n_below_30"
    if adjusted_lower <= 0.0 <= adjusted_upper:
        return "descriptive", "adjusted_band_contains_zero"
    return "direction_supported", None


def _schema_fingerprints_for_receipt(
    receipt: M6EligibilityReceipt,
) -> dict[str, str]:
    return {
        name: hashlib.sha256(
            M6_RESULT_SCHEMAS[name].serialize().to_pybytes()
        ).hexdigest()
        for name in sorted(receipt.expected_rows)
    }


def _row_domain_sha256(receipt: M6EligibilityReceipt) -> str:
    payload: dict[str, Any] = {
        "eligible_cohort_indices": list(receipt.eligible_cohort_indices),
        "expected_rows": dict(receipt.expected_rows),
        "mode": receipt.mode,
        "population_size": receipt.population_size,
        "primary_cell_domain": [
            list(value) for value in M6_PRIMARY_CELL_DOMAIN
        ],
        "secondary_b4_cohort_indices": list(
            receipt.secondary_b4_cohort_indices
        ),
    }
    if receipt.mode in {OFFICIAL_MODE, DATA_FREE_MODE}:
        payload.update(
            {
                "negative_timing_gate_domain": list(
                    M6_NEGATIVE_TIMING_GATE_DOMAIN
                ),
                "review_role_domain": list(M6_REVIEW_ROLE_DOMAIN),
                "stage_domain": list(M6_STAGE_DOMAIN),
                "waymax_row_domain": [
                    list(value) for value in M6_WAYMAX_ROW_DOMAIN
                ],
            }
        )
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_rows_sha256(
    dataset: str,
    rows: Iterable[Mapping[str, Any]],
) -> str:
    if dataset not in M6_RESULT_SCHEMAS:
        raise ValueError("unknown M6 dataset")
    payload = {
        "dataset": dataset,
        "rows": [_json_safe_row(row) for row in rows],
        "schema_version": M6_RESULT_STORE_SCHEMA_VERSION,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def m6_canonical_dataset_sha256(
    dataset: str,
    rows: Iterable[Mapping[str, Any]],
) -> str:
    """Public deterministic logical-row digest helper for pass-2 workers."""

    return _canonical_rows_sha256(dataset, rows)


def _json_safe_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            value = [
                item.item() if isinstance(item, np.generic) else item
                for item in value
            ]
        result[str(key)] = value
    return result


def _dataset_for_path(path: str) -> str | None:
    for dataset, candidate in _DATASET_PATHS.items():
        if candidate == path:
            return dataset
    return None


def _write_json_artifact(
    run_path: Path,
    path_name: str,
    schema_identity: str,
    payload: Mapping[str, Any],
) -> M6ArtifactRecord:
    encoded = _canonical_json_bytes(payload)
    path = run_path / path_name
    _write_bytes_exclusive(path, encoded, run_path)
    digest, size = _guarded_sha256(path, run_path)
    return M6ArtifactRecord(
        path=path_name,
        schema_identity=schema_identity,
        rows=None,
        size_bytes=size,
        sha256=digest,
    )


def _exact_row(
    value: Mapping[str, Any],
    fields: Sequence[str],
    dataset: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise M6ResultStoreIntegrityError(
            f"{dataset} row fields do not match fixed schema"
        )
    return {field: value[field] for field in fields}


def _integer(
    value: Any,
    *,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise M6ResultStoreIntegrityError(f"{name} must be an exact integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise M6ResultStoreIntegrityError(f"{name} is outside its domain")
    return result


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise M6ResultStoreIntegrityError(f"{name} must be exact bool")
    return value


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise M6ResultStoreIntegrityError(f"{name} must be non-empty text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _finite(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise M6ResultStoreIntegrityError(f"{name} must be a finite real")
    result = float(value)
    if not math.isfinite(result) or (
        minimum is not None and result < minimum
    ):
        raise M6ResultStoreIntegrityError(f"{name} is outside its domain")
    return result


def _optional_finite(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _finite(value, name, minimum=minimum)


def _safe_version(
    value: Any,
    name: str,
    *,
    nullable: bool,
) -> str | None:
    if value is None and nullable:
        return None
    text = _text(value, name)
    if _SAFE_VERSION.fullmatch(text) is None:
        raise M6ResultStoreIntegrityError(
            f"{name} is outside safe version domain"
        )
    return text


def _json_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise M6ResultStoreIntegrityError(f"{name} must be exact JSON object")
    return value


def _json_array(value: Any, name: str) -> list[Any]:
    if type(value) is not list:
        raise M6ResultStoreIntegrityError(f"{name} must be exact JSON array")
    return value


def _json_integer(
    value: Any,
    name: str,
    *,
    minimum: int,
) -> int:
    if type(value) is not int or value < minimum:
        raise M6ResultStoreIntegrityError(
            f"{name} must be exact JSON integer"
        )
    return value


def _json_boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise M6ResultStoreIntegrityError(f"{name} must be exact JSON bool")
    return value


def _json_text(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise M6ResultStoreIntegrityError(f"{name} must be exact JSON text")
    return value


def _json_optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _json_text(value, name)


def _json_integer_mapping(
    value: Any,
    name: str,
) -> dict[str, int]:
    mapping = _json_mapping(value, name)
    return {
        _json_text(key, f"{name} key"): _json_integer(
            count,
            f"{name}[{key!r}]",
            minimum=0,
        )
        for key, count in mapping.items()
    }


def _json_sha256_mapping(
    value: Any,
    name: str,
) -> dict[str, str]:
    mapping = _json_mapping(value, name)
    normalized: dict[str, str] = {}
    for key, digest in mapping.items():
        normalized_key = _json_text(key, f"{name} key")
        normalized_digest = _json_text(
            digest,
            f"{name}[{normalized_key!r}]",
        )
        if _SHA256.fullmatch(normalized_digest) is None:
            raise M6ResultStoreIntegrityError(
                f"{name}[{normalized_key!r}] must be SHA-256"
            )
        normalized[normalized_key] = normalized_digest
    return normalized


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (_canonical_json_text(value) + "\n").encode("ascii")


def _canonical_json_text(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise M6ResultStoreIntegrityError(
            "payload is not canonical JSON-compatible"
        ) from exc


def _decode_canonical_mapping(
    payload: bytes,
    name: str,
) -> Mapping[str, Any]:
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise M6ResultStoreIntegrityError(
            f"{name} is not canonical ASCII JSON"
        ) from exc
    if (
        type(value) is not dict
        or _canonical_json_bytes(value) != payload
    ):
        raise M6ResultStoreIntegrityError(
            f"{name} is not canonical ASCII JSON"
        )
    return value


def _raw_canonical_sha256_field(
    payload: bytes,
    field_name: str,
    artifact_name: str,
) -> str:
    """Extract one canonical SHA-256 token without invoking a data parser."""

    if type(payload) is not bytes or type(field_name) is not str:
        raise TypeError("raw SHA-256 extraction requires exact bytes/str")
    token = re.compile(
        rb'"'
        + re.escape(field_name.encode("ascii"))
        + rb'":"([0-9a-f]{64})"'
    )
    matches = token.findall(payload)
    if len(matches) != 1:
        raise M6ResultStoreIntegrityError(
            f"{artifact_name} lacks one exact raw {field_name} binding"
        )
    return matches[0].decode("ascii")


def _validate_sanitized_aggregate_payload(
    payload: Mapping[str, Any],
) -> None:
    """Validate every public leaf, nested field, domain, and accounting rule."""

    def exact_mapping(
        value: Any,
        fields: set[str],
        name: str,
    ) -> Mapping[str, Any]:
        mapping = _json_mapping(value, name)
        if set(mapping) != fields:
            raise M6ResultStoreIntegrityError(
                f"{name} fields are not exact"
            )
        return mapping

    def optional_integer(
        value: Any,
        name: str,
        *,
        minimum: int = 0,
    ) -> int | None:
        if value is None:
            return None
        return _json_integer(value, name, minimum=minimum)

    def optional_number(
        value: Any,
        name: str,
        *,
        minimum: float | None = None,
    ) -> float | None:
        if value is None:
            return None
        return _finite(value, name, minimum=minimum)

    def band(
        value: Any,
        name: str,
        *,
        expected_level: float,
    ) -> tuple[float, float]:
        item = exact_mapping(
            value,
            {"level", "lower", "upper"},
            name,
        )
        level = _finite(item["level"], f"{name}.level")
        lower = _finite(item["lower"], f"{name}.lower")
        upper = _finite(item["upper"], f"{name}.upper")
        if level != expected_level or lower > upper:
            raise M6ResultStoreIntegrityError(
                f"{name} level/endpoints drifted"
            )
        return lower, upper

    exact_mapping(
        payload,
        set(M6_PROMOTED_TOP_LEVEL_DOMAINS),
        "promoted aggregate",
    )

    claim = exact_mapping(
        payload["claim_and_limitations"],
        {"bounded_claim", "limitations"},
        "claim_and_limitations",
    )
    if (
        _json_text(
            claim["bounded_claim"],
            "claim_and_limitations.bounded_claim",
        )
        != M6_ACCEPTED_BOUNDED_CLAIM
        or _json_array(
            claim["limitations"],
            "claim_and_limitations.limitations",
        )
        != list(M6_FIXED_LIMITATIONS)
    ):
        raise M6ResultStoreIntegrityError(
            "promoted claim/limitations domain drifted"
        )

    eligibility = exact_mapping(
        payload["eligibility"],
        {
            "primary_eligible_count",
            "rejection_reason_counts",
            "total",
        },
        "eligibility",
    )
    total = _json_integer(
        eligibility["total"],
        "eligibility.total",
        minimum=1,
    )
    eligible = _json_integer(
        eligibility["primary_eligible_count"],
        "eligibility.primary_eligible_count",
        minimum=10,
    )
    reasons = _json_integer_mapping(
        eligibility["rejection_reason_counts"],
        "eligibility.rejection_reason_counts",
    )
    if (
        total != OFFICIAL_M6_PROFILE.population_size
        or eligible > total
        or set(reasons) != set(M6_PRIMARY_REJECTION_REASONS)
        or eligible + sum(reasons.values()) != total
    ):
        raise M6ResultStoreIntegrityError(
            "promoted eligibility accounting drifted"
        )

    units = {
        metric: unit
        for metric, _version, unit in M6_PRIMARY_METRICS
    }
    matrix = _json_array(payload["primary_matrix"], "primary_matrix")
    if len(matrix) != len(M6_PRIMARY_CELL_DOMAIN):
        raise M6ResultStoreIntegrityError(
            "promoted primary matrix must contain exactly 12 rows"
        )
    primary_identities: list[tuple[str, str, str, str]] = []
    idm_timing_responder_n: int | None = None
    for index, raw in enumerate(matrix):
        row = exact_mapping(
            raw,
            set(M6_PROMOTED_PRIMARY_FIELDS),
            f"primary_matrix[{index}]",
        )
        policy = _json_text(
            row["policy_name"],
            f"primary_matrix[{index}].policy_name",
        )
        access = _json_text(
            row["policy_access_role"],
            f"primary_matrix[{index}].policy_access_role",
        )
        metric = _json_text(
            row["metric_name"],
            f"primary_matrix[{index}].metric_name",
        )
        version = _json_text(
            row["metric_version"],
            f"primary_matrix[{index}].metric_version",
        )
        identity = (policy, access, metric, version)
        primary_identities.append(identity)
        if (
            identity not in M6_PRIMARY_CELL_DOMAIN
            or _json_text(
                row["unit"],
                f"primary_matrix[{index}].unit",
            )
            != units[metric]
        ):
            raise M6ResultStoreIntegrityError(
                "promoted primary identity/unit drifted"
            )
        pair_n = _json_integer(
            row["pair_n"],
            f"primary_matrix[{index}].pair_n",
            minimum=10,
        )
        nonzero_n = _json_integer(
            row["thresholded_nonzero_n"],
            f"primary_matrix[{index}].thresholded_nonzero_n",
            minimum=0,
        )
        if pair_n != eligible or nonzero_n > pair_n:
            raise M6ResultStoreIntegrityError(
                "promoted primary counts drifted"
            )
        responder_n = optional_integer(
            row["responder_n"],
            f"primary_matrix[{index}].responder_n",
        )
        censor_n = optional_integer(
            row["censor_n"],
            f"primary_matrix[{index}].censor_n",
        )
        if metric == "response_timeliness_s":
            if (
                responder_n is None
                or censor_n is None
                or responder_n + censor_n != pair_n
            ):
                raise M6ResultStoreIntegrityError(
                    "promoted primary responder/censor accounting drifted"
                )
            if policy == "idm":
                idm_timing_responder_n = responder_n
        elif responder_n is not None or censor_n is not None:
            raise M6ResultStoreIntegrityError(
                "non-timeliness primary row has response counts"
            )
        _finite(
            row["arithmetic_mean"],
            f"primary_matrix[{index}].arithmetic_mean",
        )
        _finite(
            row["median"],
            f"primary_matrix[{index}].median",
        )
        band(
            row["pointwise_band"],
            f"primary_matrix[{index}].pointwise_band",
            expected_level=M6_POINTWISE_REWEIGHTING_LEVEL,
        )
        adjusted_lower, adjusted_upper = band(
            row["adjusted_band"],
            f"primary_matrix[{index}].adjusted_band",
            expected_level=M6_ADJUSTED_REWEIGHTING_LEVEL,
        )
        status = _json_text(
            row["status"],
            f"primary_matrix[{index}].status",
        )
        suppression = _json_optional_text(
            row["suppression_reason"],
            f"primary_matrix[{index}].suppression_reason",
        )
        expected_status, expected_suppression = _primary_status(
            pair_n,
            nonzero_n,
            adjusted_lower,
            adjusted_upper,
        )
        pairing = _json_boolean(
            row["source_pairing_complete"],
            f"primary_matrix[{index}].source_pairing_complete",
        )
        directional = _json_boolean(
            row["directional_language_allowed"],
            f"primary_matrix[{index}].directional_language_allowed",
        )
        if (
            status != expected_status
            or suppression != expected_suppression
            or pairing is not True
            or directional != (status == "direction_supported")
        ):
            raise M6ResultStoreIntegrityError(
                "promoted primary status/pairing/directionality drifted"
            )
    if tuple(primary_identities) != M6_PRIMARY_CELL_DOMAIN:
        raise M6ResultStoreIntegrityError(
            "promoted primary matrix identity/order drifted"
        )
    assert idm_timing_responder_n is not None

    gates = _json_array(
        payload["negative_control_and_timing_gates"],
        "negative_control_and_timing_gates",
    )
    if len(gates) != len(M6_NEGATIVE_TIMING_GATE_DOMAIN):
        raise M6ResultStoreIntegrityError(
            "promoted negative/timing gate count drifted"
        )
    normalized_gates: dict[str, tuple[int, int, int, str]] = {}
    gate_order: list[str] = []
    for index, raw in enumerate(gates):
        row = exact_mapping(
            raw,
            {
                "assessed_n",
                "gate_name",
                "passed_n",
                "status",
                "violation_n",
            },
            f"negative_control_and_timing_gates[{index}]",
        )
        name = _json_text(
            row["gate_name"],
            f"negative_control_and_timing_gates[{index}].gate_name",
        )
        assessed = _json_integer(
            row["assessed_n"],
            f"negative_control_and_timing_gates[{index}].assessed_n",
            minimum=0,
        )
        passed_n = _json_integer(
            row["passed_n"],
            f"negative_control_and_timing_gates[{index}].passed_n",
            minimum=0,
        )
        violation_n = _json_integer(
            row["violation_n"],
            f"negative_control_and_timing_gates[{index}].violation_n",
            minimum=0,
        )
        status = _json_text(
            row["status"],
            f"negative_control_and_timing_gates[{index}].status",
        )
        if (
            name in normalized_gates
            or name not in M6_NEGATIVE_TIMING_GATE_DOMAIN
            or passed_n + violation_n != assessed
            or violation_n != 0
            or (
                assessed == 0
                and (
                    name != "nested_dose_monotonicity"
                    or status != "unsupported"
                )
            )
            or (assessed > 0 and status != "passed")
        ):
            raise M6ResultStoreIntegrityError(
                "promoted negative/timing gate accounting/status drifted"
            )
        gate_order.append(name)
        normalized_gates[name] = (
            assessed,
            passed_n,
            violation_n,
            status,
        )
    if tuple(gate_order) != M6_NEGATIVE_TIMING_GATE_DOMAIN:
        raise M6ResultStoreIntegrityError(
            "promoted negative/timing gate identity/order drifted"
        )
    expected_gate_n = {
        "log_replay_world_tensor_equality": eligible,
        "constant_velocity_world_tensor_equality": eligible,
        "sham_legacy_equality": eligible * 3,
        "synchronous_response_floor": eligible * 3,
        "primary_plan_feasibility": eligible,
    }
    if any(
        normalized_gates[name][0] != assessed
        for name, assessed in expected_gate_n.items()
    ):
        raise M6ResultStoreIntegrityError(
            "promoted negative/timing gate assessed domains drifted"
        )
    secondary_n = normalized_gates["nested_dose_monotonicity"][0]
    if secondary_n > eligible:
        raise M6ResultStoreIntegrityError(
            "promoted nested-dose domain exceeds primary eligibility"
        )

    waymax = exact_mapping(
        payload["waymax_scope"],
        {
            "cell_rows",
            "control_accounting",
            "field_comparisons",
            "qualified_count",
            "rejection_reason_counts",
            "selected_count",
            "selection_rule",
            "transition_count",
        },
        "waymax_scope",
    )
    qualified = _json_integer(
        waymax["qualified_count"],
        "waymax_scope.qualified_count",
        minimum=0,
    )
    selected = _json_integer(
        waymax["selected_count"],
        "waymax_scope.selected_count",
        minimum=0,
    )
    transitions = _json_integer(
        waymax["transition_count"],
        "waymax_scope.transition_count",
        minimum=1,
    )
    selection_rule = _json_text(
        waymax["selection_rule"],
        "waymax_scope.selection_rule",
    )
    rejection_counts = _json_integer_mapping(
        waymax["rejection_reason_counts"],
        "waymax_scope.rejection_reason_counts",
    )
    expected_selected = min(M6_WAYMAX_MAX_SELECTED, qualified) if (
        qualified >= 8
    ) else 0
    if (
        qualified > eligible
        or selected != expected_selected
        or transitions != M6_WAYMAX_TRANSITIONS
        or selection_rule != "16_or_floor"
        or set(rejection_counts) != set(M6_WAYMAX_REJECTION_REASONS)
        or qualified + sum(rejection_counts.values()) != eligible
    ):
        raise M6ResultStoreIntegrityError(
            "promoted Waymax scope/selection accounting drifted"
        )
    executed_waymax = selected >= 8

    cells = _json_array(
        waymax["cell_rows"],
        "waymax_scope.cell_rows",
    )
    expected_cell_domain = tuple(
        (bundle_name, metric_name)
        for bundle_name in M6_WAYMAX_BUNDLES
        for metric_name, _version, _unit in M6_PRIMARY_METRICS
    )
    if len(cells) != len(expected_cell_domain):
        raise M6ResultStoreIntegrityError(
            "promoted Waymax cell count drifted"
        )
    cell_domain: list[tuple[str, str]] = []
    cell_fields = {
        "arithmetic_mean",
        "bundle",
        "censor_n",
        "median",
        "metric_name",
        "metric_version",
        "pair_n",
        "pointwise_band",
        "responder_n",
        "source_pairing_complete",
        "directional_language_allowed",
        "status",
        "suppression_reason",
        "thresholded_nonzero_n",
        "unit",
    }
    for index, raw in enumerate(cells):
        row = exact_mapping(
            raw,
            cell_fields,
            f"waymax_scope.cell_rows[{index}]",
        )
        bundle_name = _json_text(
            row["bundle"],
            f"waymax_scope.cell_rows[{index}].bundle",
        )
        metric_name = _json_text(
            row["metric_name"],
            f"waymax_scope.cell_rows[{index}].metric_name",
        )
        cell_domain.append((bundle_name, metric_name))
        if (
            (bundle_name, metric_name) not in expected_cell_domain
            or _json_text(
                row["metric_version"],
                f"waymax_scope.cell_rows[{index}].metric_version",
            )
            != "1.0.0"
            or _json_text(
                row["unit"],
                f"waymax_scope.cell_rows[{index}].unit",
            )
            != units[metric_name]
        ):
            raise M6ResultStoreIntegrityError(
                "promoted Waymax cell identity/unit drifted"
            )
        pair_n = _json_integer(
            row["pair_n"],
            f"waymax_scope.cell_rows[{index}].pair_n",
            minimum=0,
        )
        nonzero_n = _json_integer(
            row["thresholded_nonzero_n"],
            f"waymax_scope.cell_rows[{index}].thresholded_nonzero_n",
            minimum=0,
        )
        responder_n = optional_integer(
            row["responder_n"],
            f"waymax_scope.cell_rows[{index}].responder_n",
        )
        censor_n = optional_integer(
            row["censor_n"],
            f"waymax_scope.cell_rows[{index}].censor_n",
        )
        if (
            pair_n != selected
            or pair_n > M6_WAYMAX_MAX_SELECTED
            or nonzero_n > pair_n
            or (
                metric_name == "response_timeliness_s"
                and (
                    responder_n is None
                    or censor_n is None
                    or responder_n + censor_n != pair_n
                )
            )
            or (
                metric_name != "response_timeliness_s"
                and (responder_n is not None or censor_n is not None)
            )
            or _json_boolean(
                row["source_pairing_complete"],
                f"waymax_scope.cell_rows[{index}].source_pairing_complete",
            )
            is not True
            or _json_boolean(
                row["directional_language_allowed"],
                f"waymax_scope.cell_rows[{index}]."
                "directional_language_allowed",
            )
            is not False
        ):
            raise M6ResultStoreIntegrityError(
                "promoted Waymax cell counts/pairing drifted"
            )
        status = _json_text(
            row["status"],
            f"waymax_scope.cell_rows[{index}].status",
        )
        suppression = _json_optional_text(
            row["suppression_reason"],
            f"waymax_scope.cell_rows[{index}].suppression_reason",
        )
        arithmetic_mean = optional_number(
            row["arithmetic_mean"],
            f"waymax_scope.cell_rows[{index}].arithmetic_mean",
        )
        median = optional_number(
            row["median"],
            f"waymax_scope.cell_rows[{index}].median",
        )
        public_band = row["pointwise_band"]
        if pair_n < 8:
            expected_status = "unsupported"
            expected_suppression = "waymax_selected_n_below_8"
        elif pair_n < 10:
            expected_status = "insufficient_n"
            expected_suppression = "waymax_pair_n_below_10"
        else:
            expected_status = "descriptive"
            expected_suppression = None
        if (
            status != expected_status
            or suppression != expected_suppression
            or (
                pair_n < 10
                and (
                    arithmetic_mean is not None
                    or median is not None
                    or public_band is not None
                )
            )
            or (
                pair_n >= 10
                and (
                    arithmetic_mean is None
                    or median is None
                    or public_band is None
                )
            )
        ):
            raise M6ResultStoreIntegrityError(
                "promoted Waymax cell statistic/suppression drifted"
            )
        if public_band is not None:
            band(
                public_band,
                f"waymax_scope.cell_rows[{index}].pointwise_band",
                expected_level=M6_POINTWISE_REWEIGHTING_LEVEL,
            )
    if tuple(cell_domain) != expected_cell_domain:
        raise M6ResultStoreIntegrityError(
            "promoted Waymax cell identity/order drifted"
        )

    comparisons = _json_array(
        waymax["field_comparisons"],
        "waymax_scope.field_comparisons",
    )
    expected_comparison_domain = tuple(
        (bundle_name, condition, field_name)
        for bundle_name in M6_WAYMAX_BUNDLES
        for condition in M6_WAYMAX_CONDITIONS
        for field_name in M6_WAYMAX_COMPARISON_FIELDS
    )
    comparison_fields = {
        "binary_mismatches",
        "bundle",
        "comparison_kind",
        "condition",
        "denominator",
        "field",
        "max_abs_error",
        "status",
        "tolerance_failures",
    }
    if len(comparisons) != len(expected_comparison_domain):
        raise M6ResultStoreIntegrityError(
            "promoted Waymax field-comparison count drifted"
        )
    comparison_domain: list[tuple[str, str, str]] = []
    for index, raw in enumerate(comparisons):
        row = exact_mapping(
            raw,
            comparison_fields,
            f"waymax_scope.field_comparisons[{index}]",
        )
        identity = (
            _json_text(
                row["bundle"],
                f"waymax_scope.field_comparisons[{index}].bundle",
            ),
            _json_text(
                row["condition"],
                f"waymax_scope.field_comparisons[{index}].condition",
            ),
            _json_text(
                row["field"],
                f"waymax_scope.field_comparisons[{index}].field",
            ),
        )
        comparison_domain.append(identity)
        if identity not in expected_comparison_domain:
            raise M6ResultStoreIntegrityError(
                "promoted Waymax comparison identity drifted"
            )
        expected_kind = (
            "exact"
            if identity[2] in M6_WAYMAX_EXACT_FIELDS
            else "tolerance"
        )
        kind = _json_text(
            row["comparison_kind"],
            f"waymax_scope.field_comparisons[{index}].comparison_kind",
        )
        denominator = _json_integer(
            row["denominator"],
            f"waymax_scope.field_comparisons[{index}].denominator",
            minimum=0,
        )
        binary_mismatches = _json_integer(
            row["binary_mismatches"],
            f"waymax_scope.field_comparisons[{index}].binary_mismatches",
            minimum=0,
        )
        tolerance_failures = optional_integer(
            row["tolerance_failures"],
            f"waymax_scope.field_comparisons[{index}].tolerance_failures",
        )
        max_abs_error = optional_number(
            row["max_abs_error"],
            f"waymax_scope.field_comparisons[{index}].max_abs_error",
            minimum=0.0,
        )
        status = _json_text(
            row["status"],
            f"waymax_scope.field_comparisons[{index}].status",
        )
        if (
            kind != expected_kind
            or binary_mismatches > denominator
            or (
                expected_kind == "exact"
                and tolerance_failures is not None
            )
            or (
                expected_kind == "tolerance"
                and (
                    tolerance_failures is None
                    or tolerance_failures > denominator
                )
            )
            or (
                executed_waymax
                and (
                    denominator < 1
                    or max_abs_error is None
                    or binary_mismatches != 0
                    or (
                        tolerance_failures is not None
                        and tolerance_failures != 0
                    )
                    or status != "accepted"
                )
            )
            or (
                not executed_waymax
                and (
                    denominator != 0
                    or binary_mismatches != 0
                    or max_abs_error is not None
                    or (
                        tolerance_failures is not None
                        and tolerance_failures != 0
                    )
                    or status != "unsupported"
                )
            )
        ):
            raise M6ResultStoreIntegrityError(
                "promoted Waymax comparison accounting/status drifted"
            )
    if tuple(comparison_domain) != expected_comparison_domain:
        raise M6ResultStoreIntegrityError(
            "promoted Waymax comparison identity/order drifted"
        )

    controls = exact_mapping(
        waymax["control_accounting"],
        set(M6_WAYMAX_CONTROL_COUNTS),
        "waymax_scope.control_accounting",
    )
    opportunity = (
        selected
        * M6_WAYMAX_TRANSITIONS
        * len(M6_WAYMAX_CONDITIONS)
    )
    expected_control_counts = {
        "target_requested_control": opportunity,
        "target_effective_control": opportunity,
        "target_logged_lifecycle_fallback": 0,
        "target_initialized_overlap_exclusion": 0,
    }
    expected_control_status = (
        "accepted" if executed_waymax else "unsupported"
    )
    for name in M6_WAYMAX_CONTROL_COUNTS:
        row = exact_mapping(
            controls[name],
            {"count", "opportunity_n", "status"},
            f"waymax_scope.control_accounting.{name}",
        )
        if (
            _json_integer(
                row["count"],
                f"waymax_scope.control_accounting.{name}.count",
                minimum=0,
            )
            != expected_control_counts[name]
            or _json_integer(
                row["opportunity_n"],
                f"waymax_scope.control_accounting.{name}.opportunity_n",
                minimum=0,
            )
            != opportunity
            or _json_text(
                row["status"],
                f"waymax_scope.control_accounting.{name}.status",
            )
            != expected_control_status
        ):
            raise M6ResultStoreIntegrityError(
                "promoted Waymax control accounting/status drifted"
            )

    execution = exact_mapping(
        payload["execution"],
        {
            "aggregate_stage_durations_ms",
            "deterministic_repeat_status",
            "fresh_worker_peak_rss_bytes",
            "gate_status",
            "required_row_domain_counts",
            "review_decisions",
        },
        "execution",
    )
    stages = _json_integer_mapping(
        execution["aggregate_stage_durations_ms"],
        "execution.aggregate_stage_durations_ms",
    )
    if set(stages) != set(M6_STAGE_DOMAIN) or any(
        duration <= 0 for duration in stages.values()
    ):
        raise M6ResultStoreIntegrityError(
            "promoted stage timing domain/positivity drifted"
        )
    if _json_text(
        execution["deterministic_repeat_status"],
        "execution.deterministic_repeat_status",
    ) != "passed":
        raise M6ResultStoreIntegrityError(
            "promoted deterministic repeat status drifted"
        )
    _json_integer(
        execution["fresh_worker_peak_rss_bytes"],
        "execution.fresh_worker_peak_rss_bytes",
        minimum=1,
    )
    gate_status = exact_mapping(
        execution["gate_status"],
        {"release", "real_reactivity_claim", "waymax"},
        "execution.gate_status",
    )
    expected_claim_status = (
        "supported" if idm_timing_responder_n >= 10 else "blocked"
    )
    if (
        _json_text(
            gate_status["release"],
            "execution.gate_status.release",
        )
        != "accepted"
        or _json_text(
            gate_status["real_reactivity_claim"],
            "execution.gate_status.real_reactivity_claim",
        )
        != expected_claim_status
        or _json_text(
            gate_status["waymax"],
            "execution.gate_status.waymax",
        )
        != ("accepted" if executed_waymax else "unsupported")
    ):
        raise M6ResultStoreIntegrityError(
            "promoted execution gate statuses drifted"
        )
    required_counts = _json_integer_mapping(
        execution["required_row_domain_counts"],
        "execution.required_row_domain_counts",
    )
    expected_counts = {
        "eligibility_rows": total,
        "primary_scene_rows": eligible * len(M6_PRIMARY_CELL_DOMAIN),
        "primary_matrix_rows": len(M6_PRIMARY_CELL_DOMAIN),
        "primary_repeat_scene_rows": (
            eligible * len(M6_PRIMARY_CELL_DOMAIN)
        ),
        "primary_repeat_matrix_rows": len(M6_PRIMARY_CELL_DOMAIN),
        "secondary_scene_rows": (
            secondary_n * len(M6_PRIMARY_CELL_DOMAIN)
        ),
        "secondary_matrix_rows": len(M6_PRIMARY_CELL_DOMAIN),
        "negative_timing_observation_rows": eligible * 9 + secondary_n,
        "negative_timing_gate_rows": len(M6_NEGATIVE_TIMING_GATE_DOMAIN),
        "waymax_accounting_rows": len(M6_WAYMAX_ROW_DOMAIN),
        "waymax_qualification_rows": eligible,
        "waymax_scene_scalar_rows": (
            M6_WAYMAX_MAX_SELECTED
            * len(M6_WAYMAX_BUNDLES)
            * len(M6_PRIMARY_METRICS)
        ),
        "waymax_field_comparison_rows": (
            M6_WAYMAX_MAX_SELECTED
            * len(M6_WAYMAX_BUNDLES)
            * len(M6_WAYMAX_CONDITIONS)
            * len(M6_WAYMAX_COMPARISON_FIELDS)
        ),
        "waymax_determinism_rows": M6_WAYMAX_DETERMINISM_ROW_COUNT,
        "stage_timing_rows": len(M6_STAGE_DOMAIN),
        "review_decision_rows": len(M6_REVIEW_ROLE_DOMAIN),
    }
    if required_counts != expected_counts:
        raise M6ResultStoreIntegrityError(
            "promoted execution row-domain accounting drifted"
        )
    reviews = _json_array(
        execution["review_decisions"],
        "execution.review_decisions",
    )
    if len(reviews) != len(M6_REVIEW_ROLE_DOMAIN):
        raise M6ResultStoreIntegrityError(
            "promoted review decision count drifted"
        )
    review_roles: list[str] = []
    for index, raw in enumerate(reviews):
        row = exact_mapping(
            raw,
            {"decision", "p1_count", "p2_count", "p3_count", "role"},
            f"execution.review_decisions[{index}]",
        )
        role = _json_text(
            row["role"],
            f"execution.review_decisions[{index}].role",
        )
        decision = _json_text(
            row["decision"],
            f"execution.review_decisions[{index}].decision",
        )
        p1_count = _json_integer(
            row["p1_count"],
            f"execution.review_decisions[{index}].p1_count",
            minimum=0,
        )
        p2_count = _json_integer(
            row["p2_count"],
            f"execution.review_decisions[{index}].p2_count",
            minimum=0,
        )
        _json_integer(
            row["p3_count"],
            f"execution.review_decisions[{index}].p3_count",
            minimum=0,
        )
        if decision != (
            "accept" if p1_count == 0 and p2_count == 0 else "reject"
        ) or decision != "accept":
            raise M6ResultStoreIntegrityError(
                "promoted review decision/count drifted"
            )
        review_roles.append(role)
    if tuple(review_roles) != M6_REVIEW_ROLE_DOMAIN:
        raise M6ResultStoreIntegrityError(
            "promoted review role identity/order drifted"
        )

    provenance = exact_mapping(
        payload["provenance_labels"],
        {
            "aggregate_schema_version",
            "config_version",
            "fixed_limitations",
            "horizons",
            "interventions",
            "plan_version",
            "policies",
            "population_label",
            "result_schema_version",
            "source_shard_suffix_range",
            "statistics_schema_version",
        },
        "provenance_labels",
    )
    fixed_text = {
        "aggregate_schema_version": M6_PROMOTED_AGGREGATE_SCHEMA_VERSION,
        "config_version": M6_CONFIG_VERSION,
        "plan_version": M6_PLAN_VERSION,
        "population_label": "accepted_m4_complete_case_ten_shard_cohort",
        "result_schema_version": M6_RESULT_STORE_SCHEMA_VERSION,
        "statistics_schema_version": M6_STATISTICS_SCHEMA_VERSION,
    }
    if any(
        _json_text(
            provenance[name],
            f"provenance_labels.{name}",
        )
        != expected
        for name, expected in fixed_text.items()
    ):
        raise M6ResultStoreIntegrityError(
            "promoted provenance fixed labels drifted"
        )
    if _json_array(
        provenance["fixed_limitations"],
        "provenance_labels.fixed_limitations",
    ) != list(M6_FIXED_LIMITATIONS):
        raise M6ResultStoreIntegrityError(
            "promoted provenance limitations drifted"
        )
    horizons = exact_mapping(
        provenance["horizons"],
        {"numpy_transitions", "waymax_transitions"},
        "provenance_labels.horizons",
    )
    if (
        _json_integer(
            horizons["numpy_transitions"],
            "provenance_labels.horizons.numpy_transitions",
            minimum=1,
        )
        != 40
        or _json_integer(
            horizons["waymax_transitions"],
            "provenance_labels.horizons.waymax_transitions",
            minimum=1,
        )
        != M6_WAYMAX_TRANSITIONS
    ):
        raise M6ResultStoreIntegrityError(
            "promoted provenance horizons drifted"
        )
    shard_range = _json_array(
        provenance["source_shard_suffix_range"],
        "provenance_labels.source_shard_suffix_range",
    )
    if (
        len(shard_range) != 2
        or _json_text(
            shard_range[0],
            "provenance_labels.source_shard_suffix_range[0]",
        )
        != "00000"
        or _json_text(
            shard_range[1],
            "provenance_labels.source_shard_suffix_range[1]",
        )
        != "00009"
    ):
        raise M6ResultStoreIntegrityError(
            "promoted provenance shard range drifted"
        )
    policies = _json_array(
        provenance["policies"],
        "provenance_labels.policies",
    )
    if len(policies) != len(M6_PRIMARY_POLICY_ROLES):
        raise M6ResultStoreIntegrityError(
            "promoted provenance policy count drifted"
        )
    normalized_policies = []
    for index, raw in enumerate(policies):
        row = exact_mapping(
            raw,
            {"access_role", "name"},
            f"provenance_labels.policies[{index}]",
        )
        normalized_policies.append(
            (
                _json_text(
                    row["name"],
                    f"provenance_labels.policies[{index}].name",
                ),
                _json_text(
                    row["access_role"],
                    f"provenance_labels.policies[{index}].access_role",
                ),
            )
        )
    if tuple(normalized_policies) != M6_PRIMARY_POLICY_ROLES:
        raise M6ResultStoreIntegrityError(
            "promoted provenance policy identity/order drifted"
        )
    interventions = _json_array(
        provenance["interventions"],
        "provenance_labels.interventions",
    )
    if len(interventions) != 3:
        raise M6ResultStoreIntegrityError(
            "promoted intervention count drifted"
        )
    identity_intervention = exact_mapping(
        interventions[0],
        {"deceleration_mps2", "name", "version"},
        "provenance_labels.interventions[0]",
    )
    primary_intervention = exact_mapping(
        interventions[1],
        {"deceleration_mps2", "duration_s", "name", "role", "version"},
        "provenance_labels.interventions[1]",
    )
    secondary_intervention = exact_mapping(
        interventions[2],
        {"deceleration_mps2", "duration_s", "name", "role", "version"},
        "provenance_labels.interventions[2]",
    )
    if (
        _finite(
            identity_intervention["deceleration_mps2"],
            "provenance_labels.interventions[0].deceleration_mps2",
        )
        != 0.0
        or _json_text(
            identity_intervention["name"],
            "provenance_labels.interventions[0].name",
        )
        != "identity"
        or _finite(
            primary_intervention["deceleration_mps2"],
            "provenance_labels.interventions[1].deceleration_mps2",
        )
        != PRIMARY_BRAKE_MAGNITUDE_MPS2
        or _finite(
            primary_intervention["duration_s"],
            "provenance_labels.interventions[1].duration_s",
        )
        != 1.0
        or _json_text(
            primary_intervention["name"],
            "provenance_labels.interventions[1].name",
        )
        != "longitudinal_brake_pulse"
        or _json_text(
            primary_intervention["role"],
            "provenance_labels.interventions[1].role",
        )
        != "primary"
        or _finite(
            secondary_intervention["deceleration_mps2"],
            "provenance_labels.interventions[2].deceleration_mps2",
        )
        != SECONDARY_BRAKE_MAGNITUDE_MPS2
        or _finite(
            secondary_intervention["duration_s"],
            "provenance_labels.interventions[2].duration_s",
        )
        != 1.0
        or _json_text(
            secondary_intervention["name"],
            "provenance_labels.interventions[2].name",
        )
        != "longitudinal_brake_pulse"
        or _json_text(
            secondary_intervention["role"],
            "provenance_labels.interventions[2].role",
        )
        != "local_secondary_not_numeric_public"
        or any(
            _json_text(
                row["version"],
                f"provenance_labels.interventions[{index}].version",
            )
            != M6_INTERVENTION_VERSION
            for index, row in enumerate(
                (
                    identity_intervention,
                    primary_intervention,
                    secondary_intervention,
                )
            )
        )
    ):
        raise M6ResultStoreIntegrityError(
            "promoted intervention identities/configuration drifted"
        )


def _assert_promoted_privacy(value: Any, *, path: str = "$") -> None:
    forbidden_keys = {
        "cohort_index",
        "cohort_indices",
        "scenario_id",
        "native_id",
        "target_index",
        "locator",
        "coordinates",
        "local_path",
        "sha256",
        "hash",
        "digest",
        "fingerprint",
        "intervention_config_fingerprint",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is not str or key.lower() in forbidden_keys:
                raise M6ResultStoreIntegrityError(
                    f"promoted aggregate contains private key at {path}"
                )
            _assert_promoted_privacy(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_promoted_privacy(child, path=f"{path}[{index}]")
    elif isinstance(value, str):
        if (
            value.startswith("/")
            or _SHA256.fullmatch(value) is not None
            or _GIT_OBJECT.fullmatch(value) is not None
        ):
            raise M6ResultStoreIntegrityError(
                f"promoted aggregate contains local path/hash at {path}"
            )


def _write_parquet_exclusive(
    path: Path,
    table: pa.Table,
    run_path: Path,
) -> None:
    _guard_artifact_parent(path, run_path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        _validate_file_metadata(os.fstat(descriptor))
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
            os.fsync(handle.fileno())
        _fsync_directory(run_path)
        descriptor, _metadata = _guard_file(path, run_path)
        os.close(descriptor)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _write_bytes_exclusive(
    path: Path,
    payload: bytes,
    run_path: Path,
) -> None:
    if type(payload) is not bytes:
        raise TypeError("exclusive payload must be exact bytes")
    _guard_artifact_parent(path, run_path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        _validate_file_metadata(os.fstat(descriptor))
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(run_path)
        descriptor, _metadata = _guard_file(path, run_path)
        os.close(descriptor)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _write_terminal_success_final(
    path: Path,
    payload: bytes,
    run_path: Path,
) -> None:
    """Create the final marker and establish directory durability.

    A transient directory-fsync failure is retried.  If durability still cannot be
    established, the function raises even when the exact marker bytes exist; an
    in-process writer must never report success for that ambiguous transition.
    """

    if type(payload) is not bytes:
        raise TypeError("terminal payload must be exact bytes")
    _guard_artifact_parent(path, run_path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        _validate_file_metadata(os.fstat(descriptor))
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        last_error: BaseException | None = None
        for _attempt in range(3):
            try:
                _fsync_directory(run_path)
                last_error = None
                break
            except BaseException as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _read_guarded_bytes(path: Path, run_path: Path) -> bytes:
    descriptor, before = _guard_file(path, run_path)
    try:
        payload = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise M6ResultStoreIntegrityError(
                "file changed during guarded read"
            )
        return bytes(payload)
    finally:
        os.close(descriptor)


def _read_guarded_parquet(
    path: Path,
    run_path: Path,
    dataset: str,
) -> pa.Table:
    descriptor, before = _guard_file(path, run_path)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            table = pq.read_table(handle)
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise M6ResultStoreIntegrityError(
                "Parquet artifact changed during guarded read"
            )
    except M6ResultStoreIntegrityError:
        raise
    except (OSError, pa.ArrowException) as exc:
        raise M6ResultStoreIntegrityError(
            f"{dataset} could not be read"
        ) from exc
    finally:
        os.close(descriptor)
    if not table.schema.equals(
        M6_RESULT_SCHEMAS[dataset],
        check_metadata=True,
    ):
        raise M6ResultStoreIntegrityError(
            f"{dataset} does not use its fixed schema"
        )
    table.validate(full=True)
    return table


def _guarded_sha256(path: Path, run_path: Path) -> tuple[str, int]:
    descriptor, before = _guard_file(path, run_path)
    digest = hashlib.sha256()
    try:
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            raise M6ResultStoreIntegrityError(
                "artifact changed while hashing"
            )
        return digest.hexdigest(), after.st_size
    finally:
        os.close(descriptor)


def _guard_file(path: Path, run_path: Path) -> tuple[int, os.stat_result]:
    _guard_artifact_parent(path, run_path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise M6ResultStoreIntegrityError(
            f"unsafe or missing result file {path.name!r}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        _validate_file_metadata(metadata)
        lexical = path.lstat()
        if _file_identity(metadata) != _file_identity(lexical):
            raise M6ResultStoreIntegrityError(
                "opened file differs from directory entry"
            )
        return descriptor, metadata
    except BaseException:
        os.close(descriptor)
        raise


def _validate_file_metadata(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
    ):
        raise M6ResultStoreIntegrityError(
            "result file must be regular, owner-only, owned, and singly linked"
        )


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _guard_artifact_parent(path: Path, run_path: Path) -> None:
    _guard_run_directory(run_path)
    if path.parent != run_path or path.name in {"", ".", ".."}:
        raise M6ResultStoreIntegrityError(
            "artifact must be a direct contained child"
        )


def _guard_directory(
    path: Path,
    *,
    parent: Path,
    require_mode: bool,
) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise M6ResultStoreIntegrityError(
            f"required directory {path.name!r} is missing"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or (require_mode and stat.S_IMODE(metadata.st_mode) != 0o700)
        or path.parent != parent
        or path.resolve(strict=True) != path
    ):
        raise M6ResultStoreIntegrityError(
            f"required directory {path.name!r} is unsafe"
        )


def _guard_run_directory(run_path: Path) -> None:
    m6_root = run_path.parent
    outputs_root = m6_root.parent
    if m6_root.name != "m6" or outputs_root.name != "outputs":
        raise M6ResultStoreIntegrityError("run containment is not canonical")
    _guard_directory(
        m6_root,
        parent=outputs_root,
        require_mode=True,
    )
    _guard_directory(
        run_path,
        parent=m6_root,
        require_mode=True,
    )


def _validate_run_tree(run_path: Path, *, allowed_files: set[str]) -> None:
    _guard_run_directory(run_path)
    observed: set[str] = set()
    with os.scandir(run_path) as entries:
        for entry in entries:
            if entry.name not in allowed_files:
                raise M6ResultStoreIntegrityError(
                    "run contains an unexpected member"
                )
            descriptor, _metadata = _guard_file(
                run_path / entry.name,
                run_path,
            )
            os.close(descriptor)
            observed.add(entry.name)
    if observed != allowed_files:
        raise M6ResultStoreIntegrityError(
            "run is missing a required artifact or marker"
        )


def _validate_marker_exclusivity(run_path: Path) -> None:
    success = _path_kind(run_path / TERMINAL_SUCCESS_MARKER) != "missing"
    failure = _path_kind(run_path / TERMINAL_FAILURE_MARKER) != "missing"
    if success and failure:
        raise M6ResultStoreIntegrityError(
            "success and failure markers are mutually exclusive"
        )
    if success and _path_kind(run_path / COMMITTED_MARKER) == "missing":
        raise M6ResultStoreIntegrityError(
            "TERMINAL_SUCCESS cannot precede COMMITTED"
        )


def _guarded_exact_bytes(
    path: Path,
    expected: bytes,
    run_path: Path,
) -> bool:
    try:
        return _read_guarded_bytes(path, run_path) == expected
    except M6ResultStoreError:
        return False


def _best_effort_failure(
    run_path: Path,
    mode: str,
    reason_code: str,
) -> None:
    if _path_kind(run_path) != "directory":
        return
    failure = run_path / TERMINAL_FAILURE_MARKER
    if os.path.lexists(failure):
        return
    try:
        _write_bytes_exclusive(
            failure,
            _canonical_json_bytes(
                {
                    "mode": mode,
                    "reason_code": reason_code,
                    "schema_version": M6_RESULT_STORE_SCHEMA_VERSION,
                    "state": "TERMINAL_FAILURE",
                }
            ),
            run_path,
        )
    except BaseException:
        return


def _validated_project_root(value: str | Path) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError("project_root must be path-like")
    absolute = Path(os.path.abspath(os.fspath(value)))
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise M6ResultStoreError("project_root must exist") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or absolute.resolve(strict=True) != absolute
    ):
        raise M6ResultStoreError(
            "project_root must be one canonical real directory"
        )
    return absolute


def _validated_run_name(value: Any) -> str:
    if (
        type(value) is not str
        or _RUN_NAME.fullmatch(value) is None
        or value in {".", ".."}
    ):
        raise ValueError("run_name must be one safe lowercase path component")
    return value


def _ensure_directory(
    parent: Path,
    name: str,
    *,
    require_owner_mode: bool,
) -> Path:
    path = parent / name
    try:
        if _path_kind(path) == "missing":
            os.mkdir(path, 0o700)
            _fsync_directory(parent)
    except OSError as exc:
        raise M6ResultStoreError(
            f"could not create required directory {name!r}"
        ) from exc
    _guard_directory(
        path,
        parent=parent,
        require_mode=require_owner_mode,
    )
    return path


def _require_git_invisible(project_root: Path, relative: Path) -> None:
    if (
        relative.is_absolute()
        or relative.parts[:2] != ("outputs", "m6")
        or len(relative.parts) != 3
    ):
        raise M6ResultStoreIntegrityError("M6 result path is not canonical")
    try:
        ignored = subprocess.run(
            (
                "git",
                "check-ignore",
                "--quiet",
                "--no-index",
                "--",
                relative.as_posix(),
            ),
            cwd=project_root,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tracked = subprocess.run(
            (
                "git",
                "ls-files",
                "--error-unmatch",
                "--",
                relative.as_posix(),
            ),
            cwd=project_root,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise M6ResultStoreError("Git result-path checks could not run") from exc
    if ignored.returncode != 0 or tracked.returncode == 0:
        raise M6ResultStoreIntegrityError(
            "M6 result path is visible to Git"
        )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


__all__ = [
    "CLAIM_LIMITATIONS_PATH",
    "COMMITTED_MARKER",
    "COMPUTE_PILOT_M6_PROFILE",
    "COMPUTE_PILOT_MODE",
    "COMPUTE_PILOT_SUMMARY",
    "COMPUTE_PILOT_SUMMARY_SCHEMA",
    "DATA_FREE_M6_TEST_PROFILE",
    "DATA_FREE_MODE",
    "DETERMINISM_RECEIPT_PATH",
    "ELIGIBILITY_LEDGER",
    "ELIGIBILITY_LEDGER_SCHEMA",
    "ELIGIBILITY_ONLY_M6_PROFILE",
    "ELIGIBILITY_ONLY_MODE",
    "ELIGIBILITY_RECEIPT_PATH",
    "EXECUTION_SUMMARY",
    "EXECUTION_SUMMARY_SCHEMA",
    "M6ArtifactRecord",
    "M6DeterminismReceipt",
    "M6EligibilityReceipt",
    "M6WaymaxSelectionReceipt",
    "M6ResultProfile",
    "M6ResultStore",
    "M6ResultStoreError",
    "M6ResultStoreIntegrityError",
    "M6ResultStoreStateError",
    "M6SanitizedAggregate",
    "M6ObservedPreflightResult",
    "M6TerminalCapability",
    "M6_ACCEPTED_BOUNDED_CLAIM",
    "M6_CLAIM_LIMITATIONS_SCHEMA_VERSION",
    "M6_CONFIG_VERSION",
    "M6_DATA_FREE_WAYMAX_PRIMARY_DOMAIN_SHA256",
    "M6_DATA_FREE_WAYMAX_SELECTION_BINDING_SHA256",
    "M6_DETERMINISM_RECEIPT_SCHEMA_VERSION",
    "M6_ELIGIBILITY_RECEIPT_SCHEMA_VERSION",
    "M6_WAYMAX_SELECTION_RECEIPT_SCHEMA_VERSION",
    "M6_EXECUTION_SCHEMA_VERSION",
    "M6_FIXED_LIMITATIONS",
    "M6_NEGATIVE_TIMING_GATE_DOMAIN",
    "M6_NEGATIVE_TIMING_OBSERVATION_POLICIES",
    "M6_PLAN_VERSION",
    "M6_PRIMARY_CELL_DOMAIN",
    "M6_PRIMARY_METRICS",
    "M6_PRIMARY_INTERVENTION_FINGERPRINT",
    "M6_PRIMARY_POLICY_ROLES",
    "M6_PRIMARY_REJECTION_REASONS",
    "M6_SECONDARY_INTERVENTION_FINGERPRINT",
    "M6_PROMOTED_AGGREGATE_SCHEMA_VERSION",
    "M6_PROMOTED_PRIMARY_FIELDS",
    "M6_PROMOTED_TOP_LEVEL_DOMAINS",
    "M6_RESULT_SCHEMAS",
    "M6_RESULT_STORE_SCHEMA_VERSION",
    "M6_REVIEW_ROLE_DOMAIN",
    "M6_RUN_MODES",
    "M6_STAGE_DOMAIN",
    "M6_TYPED_PROVENANCE_SCHEMA_VERSION",
    "M6_WAYMAX_BUNDLES",
    "M6_WAYMAX_IDENTITY_CONFIGURATION_FINGERPRINT",
    "M6_WAYMAX_PRIMARY_B2_CONFIGURATION_FINGERPRINT",
    "M6_WAYMAX_COMPARISON_FIELDS",
    "M6_WAYMAX_CONDITIONS",
    "M6_WAYMAX_CONTROL_COUNTS",
    "M6_WAYMAX_EXACT_FIELDS",
    "M6_WAYMAX_REJECTION_REASONS",
    "M6_WAYMAX_QUALIFICATION_REJECTION_REASONS",
    "M6_WAYMAX_ROW_DOMAIN",
    "MANIFEST_PATH",
    "NEGATIVE_TIMING_GATES",
    "NEGATIVE_TIMING_GATES_SCHEMA",
    "NEGATIVE_TIMING_OBSERVATIONS",
    "NEGATIVE_TIMING_OBSERVATIONS_SCHEMA",
    "OFFICIAL_M6_PROFILE",
    "OFFICIAL_MODE",
    "PENDING_MARKER",
    "PRIMARY_MATRIX",
    "PRIMARY_MATRIX_SCHEMA",
    "PRIMARY_REPEAT_MATRIX",
    "PRIMARY_REPEAT_MATRIX_SCHEMA",
    "PRIMARY_REPEAT_SCENE_SCALARS",
    "PRIMARY_REPEAT_SCENE_SCALARS_SCHEMA",
    "PRIMARY_SCENE_SCALARS",
    "PRIMARY_SCENE_SCALARS_SCHEMA",
    "REVIEW_DECISIONS",
    "REVIEW_DECISIONS_SCHEMA",
    "SECONDARY_MATRIX",
    "SECONDARY_MATRIX_SCHEMA",
    "SECONDARY_SCENE_SCALARS",
    "SECONDARY_SCENE_SCALARS_SCHEMA",
    "STAGE_TIMINGS",
    "STAGE_TIMINGS_SCHEMA",
    "TERMINAL_FAILURE_MARKER",
    "TERMINAL_SUCCESS_MARKER",
    "TYPED_PROVENANCE",
    "TYPED_PROVENANCE_SCHEMA",
    "VerifiedM6ResultStore",
    "WAYMAX_ACCOUNTING",
    "WAYMAX_ACCOUNTING_SCHEMA",
    "WAYMAX_DETERMINISM",
    "WAYMAX_DETERMINISM_SCHEMA",
    "WAYMAX_FIELD_COMPARISONS",
    "WAYMAX_FIELD_COMPARISONS_SCHEMA",
    "WAYMAX_QUALIFICATION",
    "WAYMAX_QUALIFICATION_SCHEMA",
    "WAYMAX_SCENE_SCALARS",
    "WAYMAX_SCENE_SCALARS_SCHEMA",
    "WAYMAX_SELECTION_RECEIPT_PATH",
    "m6_canonical_dataset_sha256",
    "m6_data_free_waymax_placeholder_rows",
    "m6_data_free_waymax_determinism_rows",
    "m6_data_free_waymax_field_comparison_rows",
    "m6_data_free_waymax_qualification_rows",
    "m6_data_free_waymax_scene_scalar_rows",
    "m6_waymax_unsupported_rows",
    "reconstruct_sanitized_m6_aggregate",
    "verify_committed_m6_result_store",
    "verify_m6_result_store",
]
