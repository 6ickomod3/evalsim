"""Pinned, source-specific Waymax reference and M5 metric-parity adapters.

This module is deliberately import-safe without JAX, TensorFlow, or Waymax.  The
optional native runtime is imported only while evaluating native metric components.
Private source identities and locators are used to derive an opaque deterministic
selection receipt, but never appear in public representations or error messages.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, NoReturn

import numpy as np

from evalsim.contracts import AgentType, Rollout, Scenario
from evalsim.metrics.m5 import (
    M5_METRIC_VERSION,
    canonical_float32_view,
    kinematic_infeasibility_components,
    oriented_box_overlap_components,
    position_divergence_components,
)
from evalsim.simulators.waymax_reference import (
    M4_EXACT_LOG_TRANSITIONS,
    M4_INIT_STEPS,
    M4_MAX_OBJECTS,
    WAYMAX_EXACT_LOG_NAME,
    WAYMAX_REFERENCE_VERSION,
    compact_exact_log_rollout,
    compact_waymax_to_rollout,
    validate_exact_log_compact,
)
from evalsim.sources.m5_m4_reuse import AcceptedM4MemberRef
from evalsim.sources.waymax_cohort import M4_COHORT_TARGET, rank_record
from evalsim.sources.waymax_loader import WaymaxRecord

from .m5 import EvaluationCase, ExecutionRollout, ExecutionSpec


M5_PARITY_RANK_DOMAIN = "evalsim-m5-metric-parity-v1"
M5_PARITY_ORDER_VERSION = "m5-metric-parity-order-1"
M5_PARITY_SCENE_COUNT = 16
M5_PARITY_TRANSITION_COUNT = 20
M5_PARITY_ROW_COUNT = 144

M5_PARITY_POLICY_NAMES = (
    "constant_velocity",
    "idm",
    "log_replay",
)
M5_PARITY_METRIC_NAMES = (
    "log_divergence",
    "overlap",
    "kinematic_infeasibility",
)

_PARITY_ORDER_HASH_DOMAIN = b"evalsim-m5-metric-parity-order-v1\0"
_STATE_HASH_DOMAIN = b"evalsim-m5-waymax-source-state-v1\0"
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_TRAJECTORY_FIELDS = (
    "x",
    "y",
    "z",
    "vel_x",
    "vel_y",
    "yaw",
    "valid",
    "timestamp_micros",
    "length",
    "width",
    "height",
)
_INJECTED_TRAJECTORY_FIELDS = (
    "x",
    "y",
    "vel_x",
    "vel_y",
    "yaw",
    "valid",
    "length",
    "width",
)
_SERIES_PAIRS = (
    ("x", "x"),
    ("y", "y"),
    ("yaw", "heading"),
    ("vel_x", "vx"),
    ("vel_y", "vy"),
)
_CURRENT_INDEX = M4_INIT_STEPS - 1
_FIRST_PARITY_FRAME = M4_INIT_STEPS
_PARITY_FRAME_STOP = M4_INIT_STEPS + M5_PARITY_TRANSITION_COUNT

_ERROR_CODES = frozenset(
    {
        "candidate_injection_invalid",
        "case_invalid",
        "compact_invalid",
        "component_empty",
        "component_mask_mismatch",
        "component_nonfinite",
        "component_shape_mismatch",
        "execution_invalid",
        "exact_log_mapping_mismatch",
        "native_dependency_missing",
        "native_metric_invalid",
        "parity_matrix_invalid",
        "parity_mismatch",
        "parity_selection_invalid",
        "receipt_invalid",
        "source_cadence_drift",
        "source_identity_mismatch",
        "source_mutated",
        "source_state_invalid",
    }
)


class M5WaymaxEvaluationError(RuntimeError):
    """Privacy-safe failure at the M5 Waymax evaluation boundary."""

    def __init__(self, code: str, message: str) -> None:
        if code not in _ERROR_CODES:
            raise ValueError("M5 Waymax error code is not registered")
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> NoReturn:
    raise M5WaymaxEvaluationError(code, message)


def _plain_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < minimum
    ):
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return int(value)


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _LOWER_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


@dataclass(frozen=True, slots=True)
class M5ParityOrderReceipt:
    """Aggregate-safe commitment to the frozen ordered 16-member subset."""

    rank_domain: str
    order_version: str
    ordered_membership_sha256: str = field(repr=False)
    member_count: int = M5_PARITY_SCENE_COUNT
    transition_count: int = M5_PARITY_TRANSITION_COUNT

    def __post_init__(self) -> None:
        if self.rank_domain != M5_PARITY_RANK_DOMAIN:
            raise ValueError("rank_domain differs from the frozen M5 value")
        if self.order_version != M5_PARITY_ORDER_VERSION:
            raise ValueError("order_version differs from the frozen M5 value")
        object.__setattr__(
            self,
            "ordered_membership_sha256",
            _sha256(
                self.ordered_membership_sha256,
                "ordered_membership_sha256",
            ),
        )
        member_count = _plain_int(
            self.member_count,
            "member_count",
            minimum=1,
        )
        transition_count = _plain_int(
            self.transition_count,
            "transition_count",
            minimum=1,
        )
        if (
            member_count != M5_PARITY_SCENE_COUNT
            or transition_count != M5_PARITY_TRANSITION_COUNT
        ):
            raise ValueError("parity receipt dimensions differ from frozen M5 values")
        object.__setattr__(self, "member_count", member_count)
        object.__setattr__(self, "transition_count", transition_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank_domain": self.rank_domain,
            "order_version": self.order_version,
            "ordered_membership_sha256": self.ordered_membership_sha256,
            "member_count": self.member_count,
            "transition_count": self.transition_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "M5ParityOrderReceipt":
        if not isinstance(payload, Mapping):
            raise TypeError("parity receipt payload must be a mapping")
        expected = {
            "rank_domain",
            "order_version",
            "ordered_membership_sha256",
            "member_count",
            "transition_count",
        }
        if set(payload) != expected:
            _fail("receipt_invalid", "parity receipt fields differ from the schema")
        try:
            return cls(**dict(payload))
        except (TypeError, ValueError) as exc:
            raise M5WaymaxEvaluationError(
                "receipt_invalid",
                "parity receipt values failed the frozen schema",
            ) from exc


@dataclass(frozen=True, slots=True)
class M5ParityMember:
    """One privately bound member in deterministic parity order."""

    parity_index: int
    cohort_index: int
    rank_sha256: str = field(repr=False)
    source_ref: AcceptedM4MemberRef = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        parity_index = _plain_int(self.parity_index, "parity_index")
        cohort_index = _plain_int(self.cohort_index, "cohort_index")
        if parity_index >= M5_PARITY_SCENE_COUNT:
            raise ValueError("parity_index lies outside the frozen subset")
        if cohort_index >= M4_COHORT_TARGET:
            raise ValueError("cohort_index lies outside the accepted M4 cohort")
        if not isinstance(self.source_ref, AcceptedM4MemberRef):
            raise TypeError("source_ref must be an AcceptedM4MemberRef")
        if self.source_ref.cohort_index != cohort_index:
            raise ValueError("source_ref and cohort_index differ")
        object.__setattr__(self, "parity_index", parity_index)
        object.__setattr__(self, "cohort_index", cohort_index)
        object.__setattr__(
            self,
            "rank_sha256",
            _sha256(self.rank_sha256, "rank_sha256"),
        )


@dataclass(frozen=True, slots=True)
class M5ParitySelection:
    """Frozen private membership plus its aggregate-safe public receipt."""

    members: tuple[M5ParityMember, ...] = field(repr=False)
    receipt: M5ParityOrderReceipt

    def __post_init__(self) -> None:
        members = tuple(self.members)
        if (
            len(members) != M5_PARITY_SCENE_COUNT
            or any(not isinstance(member, M5ParityMember) for member in members)
            or tuple(member.parity_index for member in members)
            != tuple(range(M5_PARITY_SCENE_COUNT))
            or len({member.cohort_index for member in members})
            != M5_PARITY_SCENE_COUNT
        ):
            raise ValueError("members do not form the frozen ordered parity subset")
        if not isinstance(self.receipt, M5ParityOrderReceipt):
            raise TypeError("receipt must be an M5ParityOrderReceipt")
        if (
            self.receipt.ordered_membership_sha256
            != _ordered_membership_digest(members)
        ):
            raise ValueError("receipt does not commit to the supplied parity members")
        object.__setattr__(self, "members", members)

    @property
    def parity_index_by_cohort_index(self) -> Mapping[int, int]:
        return {
            member.cohort_index: member.parity_index
            for member in self.members
        }


def _validate_member_source_binding(member: AcceptedM4MemberRef) -> None:
    event = member.event
    expectation = member.expectation
    if (
        expectation.locator.shard_suffix != event.shard_suffix
        or expectation.locator.record_ordinal != event.record_ordinal
        or expectation.expected_scenario_id != event.native_scenario_id
        or expectation.expected_shard_sha256 != event.shard_sha256
        or expectation.expected_dataset_config_fingerprint
        != event.dataset_config_fingerprint
    ):
        _fail(
            "parity_selection_invalid",
            "an accepted member's private source binding is inconsistent",
        )


def _ordered_membership_digest(members: Sequence[M5ParityMember]) -> str:
    payload = {
        "members": [
            {
                "cohort_index": member.cohort_index,
                "parity_index": member.parity_index,
                "rank_sha256": member.rank_sha256,
                "source_event": member.source_ref.event.to_dict(),
            }
            for member in members
        ],
        "rank_domain": M5_PARITY_RANK_DOMAIN,
        "version": M5_PARITY_ORDER_VERSION,
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise M5WaymaxEvaluationError(
            "parity_selection_invalid",
            "ordered parity membership cannot be canonically encoded",
        ) from exc
    digest = hashlib.sha256()
    digest.update(_PARITY_ORDER_HASH_DOMAIN)
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)
    return digest.hexdigest()


def select_m5_parity_members(
    accepted_members: Sequence[AcceptedM4MemberRef],
) -> M5ParitySelection:
    """Select and commit to the first 16 source-ranked accepted M4 members."""

    members = tuple(accepted_members)
    if (
        len(members) != M4_COHORT_TARGET
        or any(not isinstance(member, AcceptedM4MemberRef) for member in members)
    ):
        _fail(
            "parity_selection_invalid",
            "selection requires exactly 128 accepted M4 member references",
        )
    by_cohort = sorted(members, key=lambda member: member.cohort_index)
    if tuple(member.cohort_index for member in by_cohort) != tuple(
        range(M4_COHORT_TARGET)
    ):
        _fail(
            "parity_selection_invalid",
            "accepted members must cover canonical cohort indices 0 through 127",
        )
    locators = tuple(
        (member.event.shard_suffix, member.event.record_ordinal)
        for member in by_cohort
    )
    native_ids = tuple(member.event.native_scenario_id for member in by_cohort)
    if (
        len(set(locators)) != M4_COHORT_TARGET
        or len(set(native_ids)) != M4_COHORT_TARGET
    ):
        _fail(
            "parity_selection_invalid",
            "accepted members must retain unique source locators and identities",
        )
    ranked: list[tuple[str, str, int, AcceptedM4MemberRef]] = []
    for member in by_cohort:
        _validate_member_source_binding(member)
        event = member.event
        rank = rank_record(
            M5_PARITY_RANK_DOMAIN,
            event.shard_suffix,
            event.record_ordinal,
            event.native_scenario_id,
        )
        ranked.append((rank, event.shard_suffix, event.record_ordinal, member))
    ranked.sort(key=lambda item: item[:3])
    selected = tuple(
        M5ParityMember(
            parity_index=parity_index,
            cohort_index=item[3].cohort_index,
            rank_sha256=item[0],
            source_ref=item[3],
        )
        for parity_index, item in enumerate(ranked[:M5_PARITY_SCENE_COUNT])
    )
    receipt = M5ParityOrderReceipt(
        rank_domain=M5_PARITY_RANK_DOMAIN,
        order_version=M5_PARITY_ORDER_VERSION,
        ordered_membership_sha256=_ordered_membership_digest(selected),
    )
    return M5ParitySelection(members=selected, receipt=receipt)


def _hash_piece(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _hash_state_value(digest: Any, value: Any) -> None:
    if value is None:
        _hash_piece(digest, b"none")
        return
    if isinstance(value, Enum):
        _hash_piece(digest, b"enum")
        _hash_piece(digest, type(value).__qualname__.encode("utf-8"))
        _hash_piece(digest, str(value.value).encode("utf-8"))
        return
    if isinstance(value, (bool, np.bool_)):
        _hash_piece(digest, b"bool")
        _hash_piece(digest, b"1" if bool(value) else b"0")
        return
    if isinstance(value, (int, np.integer)):
        _hash_piece(digest, b"int")
        _hash_piece(digest, str(int(value)).encode("ascii"))
        return
    if isinstance(value, (float, np.floating)):
        _hash_piece(digest, b"float")
        _hash_piece(digest, np.float64(value).tobytes())
        return
    if isinstance(value, str):
        _hash_piece(digest, b"str")
        _hash_piece(digest, value.encode("utf-8", errors="strict"))
        return
    if isinstance(value, bytes):
        _hash_piece(digest, b"bytes")
        _hash_piece(digest, value)
        return
    if is_dataclass(value) and not isinstance(value, type):
        _hash_piece(digest, b"dataclass")
        _hash_piece(digest, type(value).__qualname__.encode("utf-8"))
        for item in fields(value):
            _hash_piece(digest, item.name.encode("utf-8"))
            _hash_state_value(digest, getattr(value, item.name))
        return
    if isinstance(value, Mapping):
        _hash_piece(digest, b"mapping")
        if any(not isinstance(key, str) for key in value):
            _fail(
                "source_state_invalid",
                "source-state mappings must use string keys",
            )
        for key in sorted(value):
            _hash_piece(digest, key.encode("utf-8", errors="strict"))
            _hash_state_value(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        _hash_piece(digest, b"tuple" if isinstance(value, tuple) else b"list")
        for item in value:
            _hash_state_value(digest, item)
        return
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise M5WaymaxEvaluationError(
            "source_state_invalid",
            "source state contains an unsupported value",
        ) from exc
    if array.dtype.hasobject:
        _fail(
            "source_state_invalid",
            "source state contains an unsupported object array",
        )
    contiguous = np.ascontiguousarray(array)
    _hash_piece(digest, b"array")
    _hash_piece(digest, contiguous.dtype.str.encode("ascii"))
    _hash_piece(
        digest,
        json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"),
    )
    _hash_piece(digest, contiguous.tobytes(order="C"))


def _state_fingerprint(state: Any) -> str:
    digest = hashlib.sha256()
    digest.update(_STATE_HASH_DOMAIN)
    _hash_state_value(digest, state)
    return digest.hexdigest()


def _require_case_record(case: EvaluationCase) -> WaymaxRecord:
    if not isinstance(case, EvaluationCase):
        raise TypeError("case must be an EvaluationCase")
    if not isinstance(case.reference_payload, WaymaxRecord):
        _fail(
            "case_invalid",
            "a Waymax evaluation case requires one typed source record",
        )
    if case.reference_payload.scenario_id != case.scenario.scenario_id:
        _fail(
            "source_identity_mismatch",
            "source record and contract scenario identities differ",
        )
    return case.reference_payload


def _validate_selected_case_binding(
    case: EvaluationCase,
    member: M5ParityMember,
) -> None:
    record = _require_case_record(case)
    event = member.source_ref.event
    if (
        case.cohort_index != member.cohort_index
        or record.scenario_id != event.native_scenario_id
        or record.shard_suffix != event.shard_suffix
        or record.record_ordinal != event.record_ordinal
        or record.shard_sha256 != event.shard_sha256
        or record.dataset_config_fingerprint
        != event.dataset_config_fingerprint
    ):
        _fail(
            "source_identity_mismatch",
            "a parity case does not match its frozen private source membership",
        )


def _seed_zero(seed: Any) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer")
    if int(seed) != 0:
        _fail("execution_invalid", "the frozen Waymax reference seed must be zero")
    return 0


def _block_compact(compact: Any) -> Any:
    try:
        leaves = []
        for value in compact:
            block = getattr(value, "block_until_ready", None)
            leaves.append(block() if callable(block) else value)
        return type(compact)(*leaves)
    except (TypeError, ValueError, AttributeError) as exc:
        raise M5WaymaxEvaluationError(
            "compact_invalid",
            "the exact-log compact result failed synchronization",
        ) from exc


def _validate_exact_mapping(
    scenario: Scenario,
    rollout: Rollout,
    *,
    seed: int,
) -> None:
    if (
        scenario.metadata.get("current_index") != _CURRENT_INDEX
        or scenario.num_steps != M4_INIT_STEPS + M4_EXACT_LOG_TRANSITIONS
    ):
        _fail(
            "case_invalid",
            "the exact-log case must have current index 10 and 91 frames",
        )
    if (
        rollout.scenario_id != scenario.scenario_id
        or rollout.sim_name != WAYMAX_EXACT_LOG_NAME
        or rollout.sim_version != WAYMAX_REFERENCE_VERSION
        or rollout.seed != seed
        or rollout.num_steps != scenario.num_steps
        or rollout.num_agents != scenario.num_agents
        or not np.array_equal(rollout.timestamps, scenario.timestamps)
    ):
        _fail(
            "exact_log_mapping_mismatch",
            "exact-log rollout identity or execution provenance differs",
        )
    for logged, candidate in zip(
        scenario.agents,
        rollout.agents,
        strict=True,
    ):
        if (
            logged.id != candidate.id
            or logged.type != candidate.type
            or logged.length != candidate.length
            or logged.width != candidate.width
            or not np.array_equal(logged.valid, candidate.valid)
        ):
            _fail(
                "exact_log_mapping_mismatch",
                "exact-log agent identity, dimensions, or validity differs",
            )
    view = canonical_float32_view(scenario, rollout)
    if not np.array_equal(view["sim_valid"], view["log_valid"]):
        _fail(
            "exact_log_mapping_mismatch",
            "exact-log canonical validity differs",
        )
    for name in ("x", "y", "heading", "vx", "vy"):
        if not np.array_equal(view[f"sim_{name}"], view[f"log_{name}"]):
            _fail(
                "exact_log_mapping_mismatch",
                "exact-log canonical motion differs",
            )


@dataclass(frozen=True, slots=True)
class WaymaxExactLogReferenceExecutor:
    """Execute the pinned full-horizon exact-log mapping oracle."""

    def execution_spec(self, seed: int = 0) -> ExecutionSpec:
        clean_seed = _seed_zero(seed)
        return ExecutionSpec(
            name=WAYMAX_EXACT_LOG_NAME,
            version=WAYMAX_REFERENCE_VERSION,
            role="reference",
            seed=clean_seed,
        )

    def execute(
        self,
        case: EvaluationCase,
        *,
        seed: int = 0,
    ) -> ExecutionRollout:
        clean_seed = _seed_zero(seed)
        record = _require_case_record(case)
        source_before = _state_fingerprint(record.state)
        try:
            compact = compact_exact_log_rollout(
                record.state,
                num_steps=M4_EXACT_LOG_TRANSITIONS,
            )
            compact = _block_compact(compact)
            checks = validate_exact_log_compact(record.state, compact)
            if (
                not isinstance(checks, Mapping)
                or set(checks) != {"fields", "timestamps", "timesteps", "validity"}
                or any(value is not True for value in checks.values())
            ):
                _fail(
                    "compact_invalid",
                    "the exact-log compact oracle did not pass every check",
                )
            rollout = compact_waymax_to_rollout(
                compact,
                state=record.state,
                scenario=case.scenario,
                sim_name=WAYMAX_EXACT_LOG_NAME,
                seed=clean_seed,
            )
            _validate_exact_mapping(case.scenario, rollout, seed=clean_seed)
            return ExecutionRollout(
                spec=self.execution_spec(seed=clean_seed),
                rollout=rollout,
            )
        finally:
            source_after = _state_fingerprint(record.state)
            if source_after != source_before:
                _fail(
                    "source_mutated",
                    "exact-log execution changed the reloaded source state",
                )


@dataclass(frozen=True, slots=True)
class M5WaymaxParityRow:
    """One aggregate-safe policy × metric parity result."""

    parity_index: int
    policy_name: str
    metric_name: str
    metric_version: str
    compared_components: int
    mismatch_count: int
    max_abs_error: float
    max_tolerance_excess: float
    exact_match: bool
    status: str

    def __post_init__(self) -> None:
        index = _plain_int(self.parity_index, "parity_index")
        if index >= M5_PARITY_SCENE_COUNT:
            raise ValueError("parity_index lies outside the frozen subset")
        if self.policy_name not in M5_PARITY_POLICY_NAMES:
            raise ValueError("policy_name is not registered for M5 parity")
        if self.metric_name not in M5_PARITY_METRIC_NAMES:
            raise ValueError("metric_name is not an M5 parity anchor")
        if self.metric_version != M5_METRIC_VERSION:
            raise ValueError("metric_version differs from the frozen M5 version")
        components = _plain_int(
            self.compared_components,
            "compared_components",
            minimum=1,
        )
        mismatches = _plain_int(self.mismatch_count, "mismatch_count")
        if mismatches > components:
            raise ValueError("mismatch_count exceeds compared_components")
        for name in ("max_abs_error", "max_tolerance_excess"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(f"{name} must be one finite Python float")
        if self.max_abs_error < 0.0:
            raise ValueError("max_abs_error must be non-negative")
        expected_match = mismatches == 0
        if type(self.exact_match) is not bool or self.exact_match != expected_match:
            raise ValueError("exact_match differs from mismatch_count")
        expected_status = "accepted" if expected_match else "rejected"
        if self.status != expected_status:
            raise ValueError("status differs from mismatch_count")
        object.__setattr__(self, "parity_index", index)
        object.__setattr__(self, "compared_components", components)
        object.__setattr__(self, "mismatch_count", mismatches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parity_index": self.parity_index,
            "policy_name": self.policy_name,
            "metric_name": self.metric_name,
            "metric_version": self.metric_version,
            "compared_components": self.compared_components,
            "mismatch_count": self.mismatch_count,
            "max_abs_error": self.max_abs_error,
            "max_tolerance_excess": self.max_tolerance_excess,
            "exact_match": self.exact_match,
            "status": self.status,
        }


def _component_arrays(
    custom: Any,
    reference: Any,
    custom_mask: Any,
    reference_mask: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    custom_values = np.asarray(custom)
    reference_values = np.asarray(reference)
    mask = np.asarray(custom_mask)
    other_mask = np.asarray(reference_mask)
    if custom_values.shape != reference_values.shape:
        _fail("component_shape_mismatch", "parity component value shapes differ")
    if (
        mask.dtype != np.bool_
        or other_mask.dtype != np.bool_
        or mask.shape != custom_values.shape
        or other_mask.shape != custom_values.shape
    ):
        _fail("component_shape_mismatch", "parity component masks have invalid shapes")
    if not np.array_equal(mask, other_mask):
        _fail("component_mask_mismatch", "parity component masks differ")
    if not np.any(mask):
        _fail("component_empty", "parity comparison has no eligible component")
    return custom_values, reference_values, mask


def build_continuous_parity_row(
    *,
    parity_index: int,
    policy_name: str,
    metric_name: str,
    custom: Any,
    reference: Any,
    custom_mask: Any,
    reference_mask: Any,
) -> M5WaymaxParityRow:
    """Compare float32 components under the frozen ULP-aware absolute bound."""

    if metric_name != "log_divergence":
        raise ValueError("continuous parity is registered only for log_divergence")
    custom_values, reference_values, mask = _component_arrays(
        custom,
        reference,
        custom_mask,
        reference_mask,
    )
    if custom_values.dtype != np.float32 or reference_values.dtype != np.float32:
        _fail(
            "native_metric_invalid",
            "continuous parity values must retain canonical float32 dtype",
        )
    selected_custom = custom_values[mask]
    selected_reference = reference_values[mask]
    if (
        not np.all(np.isfinite(selected_custom))
        or not np.all(np.isfinite(selected_reference))
    ):
        _fail(
            "component_nonfinite",
            "an eligible continuous parity component is non-finite",
        )
    custom64 = selected_custom.astype(np.float64, copy=False)
    reference64 = selected_reference.astype(np.float64, copy=False)
    errors = np.abs(custom64 - reference64)
    absolute_reference = np.abs(selected_reference).astype(
        np.float32,
        copy=False,
    )
    ulps = (
        np.nextafter(
            absolute_reference,
            np.float32(np.inf),
            dtype=np.float32,
        )
        - absolute_reference
    ).astype(np.float64)
    tolerances = np.maximum(np.float64(1e-6), np.float64(8.0) * ulps)
    if not np.all(np.isfinite(tolerances)):
        _fail(
            "component_nonfinite",
            "continuous parity tolerance is non-finite",
        )
    excess = errors - tolerances
    mismatch_count = int(np.count_nonzero(errors > tolerances))
    return M5WaymaxParityRow(
        parity_index=parity_index,
        policy_name=policy_name,
        metric_name=metric_name,
        metric_version=M5_METRIC_VERSION,
        compared_components=int(np.count_nonzero(mask)),
        mismatch_count=mismatch_count,
        max_abs_error=float(np.max(errors)),
        max_tolerance_excess=float(np.max(excess)),
        exact_match=mismatch_count == 0,
        status="accepted" if mismatch_count == 0 else "rejected",
    )


def build_discrete_parity_row(
    *,
    parity_index: int,
    policy_name: str,
    metric_name: str,
    custom: Any,
    reference: Any,
    custom_mask: Any,
    reference_mask: Any,
    additional_mismatches: Sequence[Any] = (),
) -> M5WaymaxParityRow:
    """Compare observed boolean flags exactly on one identical semantic mask."""

    if metric_name not in {"overlap", "kinematic_infeasibility"}:
        raise ValueError("metric_name is not a discrete M5 parity anchor")
    custom_values, reference_values, mask = _component_arrays(
        custom,
        reference,
        custom_mask,
        reference_mask,
    )
    custom_bool = np.asarray(custom_values, dtype=bool)
    reference_bool = np.asarray(reference_values, dtype=bool)
    mismatch = (custom_bool != reference_bool) & mask
    for additional in additional_mismatches:
        values = np.asarray(additional)
        if values.dtype != np.bool_ or values.shape != mask.shape:
            _fail(
                "component_shape_mismatch",
                "an additional discrete parity branch mask is invalid",
            )
        mismatch |= values & mask
    mismatch_count = int(np.count_nonzero(mismatch))
    binary_error = 0.0 if mismatch_count == 0 else 1.0
    return M5WaymaxParityRow(
        parity_index=parity_index,
        policy_name=policy_name,
        metric_name=metric_name,
        metric_version=M5_METRIC_VERSION,
        compared_components=int(np.count_nonzero(mask)),
        mismatch_count=mismatch_count,
        max_abs_error=binary_error,
        max_tolerance_excess=binary_error,
        exact_match=mismatch_count == 0,
        status="accepted" if mismatch_count == 0 else "rejected",
    )


def _require_native_runtime() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import jax.numpy as jnp
        from waymax.dynamics import bicycle_model
        from waymax.metrics import comfort, imitation, overlap
    except ImportError as exc:
        raise M5WaymaxEvaluationError(
            "native_dependency_missing",
            "the optional pinned Waymax runtime is unavailable",
        ) from exc
    return jnp, bicycle_model, comfort, imitation, overlap


def _trajectory_arrays(trajectory: Any) -> dict[str, np.ndarray]:
    missing = [name for name in _TRAJECTORY_FIELDS if not hasattr(trajectory, name)]
    if missing:
        _fail("source_state_invalid", "source trajectory fields are incomplete")
    arrays = {
        name: np.asarray(getattr(trajectory, name))
        for name in _TRAJECTORY_FIELDS
    }
    if any(
        array.shape != (M4_MAX_OBJECTS, M4_INIT_STEPS + M4_EXACT_LOG_TRANSITIONS)
        for array in arrays.values()
    ):
        _fail(
            "source_state_invalid",
            "source trajectories must have shape [128, 91]",
        )
    return arrays


def _normalize_yaw(values: np.ndarray) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    return (
        (values64 + np.pi) % (2.0 * np.pi) - np.pi
    ).astype(np.float32)


def _source_retained_slots(
    state: Any,
    scenario: Scenario,
    view: Mapping[str, np.ndarray],
) -> np.ndarray:
    if tuple(getattr(state, "shape", ())) != ():
        _fail("source_state_invalid", "native metric parity requires one unbatched state")
    if (
        not hasattr(state, "log_trajectory")
        or not hasattr(state, "sim_trajectory")
        or not hasattr(state, "object_metadata")
    ):
        _fail("source_state_invalid", "source state fields are incomplete")
    log_arrays = _trajectory_arrays(state.log_trajectory)
    _trajectory_arrays(state.sim_trajectory)
    retained = np.flatnonzero(np.any(log_arrays["valid"], axis=1))
    if retained.size != scenario.num_agents:
        _fail(
            "source_identity_mismatch",
            "retained source and contract agent counts differ",
        )
    metadata = state.object_metadata
    if (
        not hasattr(metadata, "ids")
        or not hasattr(metadata, "is_valid")
        or not hasattr(metadata, "is_sdc")
        or not hasattr(metadata, "object_types")
    ):
        _fail("source_state_invalid", "source object metadata is incomplete")
    metadata_valid = np.asarray(metadata.is_valid, dtype=bool)
    if (
        metadata_valid.shape != (M4_MAX_OBJECTS,)
        or not np.array_equal(metadata_valid, np.any(log_arrays["valid"], axis=1))
    ):
        _fail(
            "source_identity_mismatch",
            "source metadata and trajectory retention masks differ",
        )
    source_ids = np.asarray(metadata.ids)
    if source_ids.shape != (M4_MAX_OBJECTS,):
        _fail("source_state_invalid", "source object IDs have an invalid shape")
    try:
        contract_ids = np.asarray(
            [int(agent.id) for agent in scenario.agents],
            dtype=source_ids.dtype,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise M5WaymaxEvaluationError(
            "source_identity_mismatch",
            "contract object identities cannot map to native slots",
        ) from exc
    if not np.array_equal(source_ids[retained], contract_ids):
        _fail(
            "source_identity_mismatch",
            "contract object identity or order differs from native slots",
        )
    source_types = np.asarray(metadata.object_types)
    source_sdc = np.asarray(metadata.is_sdc, dtype=bool)
    if (
        source_types.shape != (M4_MAX_OBJECTS,)
        or source_sdc.shape != (M4_MAX_OBJECTS,)
    ):
        _fail("source_state_invalid", "source object roles have invalid shapes")
    type_map = {
        1: AgentType.VEHICLE,
        2: AgentType.PEDESTRIAN,
        3: AgentType.CYCLIST,
    }
    expected_types = tuple(
        type_map.get(int(source_types[source_index]), AgentType.UNKNOWN)
        for source_index in retained
    )
    if expected_types != tuple(agent.type for agent in scenario.agents):
        _fail(
            "source_identity_mismatch",
            "contract object types differ from native slots",
        )
    retained_sdc = np.flatnonzero(source_sdc[retained])
    if not np.array_equal(
        retained_sdc,
        np.asarray([scenario.ego_index], dtype=retained_sdc.dtype),
    ):
        _fail(
            "source_identity_mismatch",
            "contract ego identity differs from the native source",
        )
    if not np.array_equal(
        log_arrays["valid"][retained].T,
        view["log_valid"],
    ):
        _fail(
            "source_identity_mismatch",
            "contract logged validity differs from the native source",
        )
    for native_name, contract_name in _SERIES_PAIRS:
        source = np.asarray(
            log_arrays[native_name][retained].T,
            dtype=np.float32,
        )
        expected = view[f"log_{contract_name}"]
        if native_name == "yaw":
            source = _normalize_yaw(source)
        valid = view["log_valid"]
        if not np.array_equal(source[valid], expected[valid]):
            _fail(
                "source_identity_mismatch",
                "contract logged motion differs from the native source",
            )
    timestamps = log_arrays["timestamp_micros"][retained]
    valid_window = log_arrays["valid"][
        retained,
        _CURRENT_INDEX:_PARITY_FRAME_STOP,
    ]
    deltas = np.diff(
        timestamps[:, _CURRENT_INDEX:_PARITY_FRAME_STOP],
        axis=1,
    )
    contiguous = valid_window[:, :-1] & valid_window[:, 1:]
    if not np.all(deltas[contiguous] == 100_000):
        _fail(
            "source_cadence_drift",
            "scored source transitions do not have exact 100000 us cadence",
        )
    canonical = np.empty(scenario.num_steps, dtype=np.int64)
    source_valid = log_arrays["valid"][retained]
    for frame in range(scenario.num_steps):
        contributors = timestamps[:, frame][source_valid[:, frame]].astype(
            np.int64,
            copy=False,
        )
        if contributors.size == 0 or not np.all(contributors == contributors[0]):
            _fail(
                "source_identity_mismatch",
                "valid source objects do not define one canonical timestamp",
            )
        canonical[frame] = contributors[0]
    expected_timestamps = (canonical - canonical[0]).astype(np.float64) * 1e-6
    if not np.array_equal(
        np.asarray(scenario.timestamps, dtype=np.float64),
        expected_timestamps,
    ):
        _fail(
            "source_identity_mismatch",
            "contract timestamps differ from the native source",
        )
    return retained


def _replace_trajectory(
    trajectory: Any,
    arrays: Mapping[str, np.ndarray],
    *,
    jnp: Any,
) -> Any:
    try:
        return replace(
            trajectory,
            **{
                name: jnp.asarray(arrays[name])
                for name in _INJECTED_TRAJECTORY_FIELDS
            },
        )
    except (TypeError, ValueError) as exc:
        raise M5WaymaxEvaluationError(
            "candidate_injection_invalid",
            "native trajectory replacement failed",
        ) from exc


def _assert_candidate_injection(
    source_state: Any,
    candidate_state: Any,
    *,
    retained: np.ndarray,
    expected: Mapping[str, np.ndarray],
) -> None:
    source_sim = _trajectory_arrays(source_state.sim_trajectory)
    candidate_sim = _trajectory_arrays(candidate_state.sim_trajectory)
    for name in ("z", "height", "timestamp_micros"):
        if not np.array_equal(candidate_sim[name], source_sim[name]):
            _fail(
                "candidate_injection_invalid",
                "a preserved native trajectory field changed",
            )
    padding = np.ones(M4_MAX_OBJECTS, dtype=bool)
    padding[retained] = False
    for name in _INJECTED_TRAJECTORY_FIELDS:
        if not np.array_equal(candidate_sim[name][padding], source_sim[name][padding]):
            _fail(
                "candidate_injection_invalid",
                "a native padding slot changed during candidate injection",
            )
        if not np.array_equal(candidate_sim[name][retained], expected[name]):
            _fail(
                "candidate_injection_invalid",
                "a retained candidate field differs after native injection",
            )
    for item in fields(source_state):
        if item.name in {"sim_trajectory", "timestep"}:
            continue
        if getattr(candidate_state, item.name) is not getattr(
            source_state,
            item.name,
        ):
            _fail(
                "candidate_injection_invalid",
                "candidate injection replaced a source-state field",
            )
    if int(np.asarray(candidate_state.timestep)) != _CURRENT_INDEX:
        _fail(
            "candidate_injection_invalid",
            "candidate state does not use the frozen current timestep",
        )


def _inject_candidate_state(
    state: Any,
    scenario: Scenario,
    rollout: Rollout,
    *,
    jnp: Any,
) -> tuple[Any, np.ndarray, dict[str, np.ndarray]]:
    if (
        scenario.metadata.get("current_index") != _CURRENT_INDEX
        or scenario.num_steps != M4_INIT_STEPS + M4_EXACT_LOG_TRANSITIONS
    ):
        _fail(
            "case_invalid",
            "native parity requires current index 10 and 91 frames",
        )
    view = canonical_float32_view(scenario, rollout)
    retained = _source_retained_slots(state, scenario, view)
    source_arrays = _trajectory_arrays(state.sim_trajectory)
    injected = {
        name: np.array(source_arrays[name], copy=True)
        for name in _INJECTED_TRAJECTORY_FIELDS
    }
    mapping = {
        "x": "sim_x",
        "y": "sim_y",
        "vel_x": "sim_vx",
        "vel_y": "sim_vy",
        "yaw": "sim_heading",
        "valid": "sim_valid",
    }
    for native_name, view_name in mapping.items():
        injected[native_name][retained] = view[view_name].T
    injected["length"][retained] = np.broadcast_to(
        view["length"][:, None],
        (scenario.num_agents, scenario.num_steps),
    )
    injected["width"][retained] = np.broadcast_to(
        view["width"][:, None],
        (scenario.num_agents, scenario.num_steps),
    )
    candidate_trajectory = _replace_trajectory(
        state.sim_trajectory,
        injected,
        jnp=jnp,
    )
    try:
        timestep_dtype = np.asarray(state.timestep).dtype
        candidate_state = replace(
            state,
            sim_trajectory=candidate_trajectory,
            timestep=jnp.asarray(_CURRENT_INDEX, dtype=timestep_dtype),
        )
    except (TypeError, ValueError) as exc:
        raise M5WaymaxEvaluationError(
            "candidate_injection_invalid",
            "native candidate-state replacement failed",
        ) from exc
    expected = {
        name: np.asarray(injected[name][retained])
        for name in _INJECTED_TRAJECTORY_FIELDS
    }
    _assert_candidate_injection(
        state,
        candidate_state,
        retained=retained,
        expected=expected,
    )
    return candidate_state, retained, view


def _native_metric_rows(
    *,
    state: Any,
    scenario: Scenario,
    rollout: Rollout,
    parity_index: int,
    policy_name: str,
) -> tuple[M5WaymaxParityRow, ...]:
    jnp, bicycle_model, comfort, imitation, overlap = _require_native_runtime()
    candidate_state, retained, _ = _inject_candidate_state(
        state,
        scenario,
        rollout,
        jnp=jnp,
    )
    candidate = candidate_state.sim_trajectory
    logged = candidate_state.log_trajectory

    custom_position, custom_position_mask = position_divergence_components(
        scenario,
        rollout,
    )
    try:
        native_position = np.asarray(
            imitation.LogDivergenceMetric.compute_log_divergence(
                candidate.xy,
                logged.xy,
            )
        ).T[:, retained]
    except (TypeError, ValueError, RuntimeError) as exc:
        raise M5WaymaxEvaluationError(
            "native_metric_invalid",
            "native log-divergence execution failed",
        ) from exc
    native_position = np.asarray(native_position, dtype=np.float32)
    native_position_mask = (
        np.asarray(candidate.valid, dtype=bool)
        & np.asarray(logged.valid, dtype=bool)
    ).T[:, retained]
    frame_slice = slice(_FIRST_PARITY_FRAME, _PARITY_FRAME_STOP)
    position_row = build_continuous_parity_row(
        parity_index=parity_index,
        policy_name=policy_name,
        metric_name="log_divergence",
        custom=custom_position[frame_slice],
        reference=native_position[frame_slice],
        custom_mask=custom_position_mask[frame_slice],
        reference_mask=native_position_mask[frame_slice],
    )

    custom_overlap, custom_overlap_mask = oriented_box_overlap_components(
        scenario,
        rollout,
    )
    native_overlap = np.zeros(
        (M5_PARITY_TRANSITION_COUNT, scenario.num_agents),
        dtype=bool,
    )
    native_overlap_mask = np.zeros_like(native_overlap)
    overlap_metric = overlap.OverlapMetric()
    for offset, frame in enumerate(
        range(_FIRST_PARITY_FRAME, _PARITY_FRAME_STOP)
    ):
        try:
            frame_state = replace(
                candidate_state,
                timestep=jnp.asarray(
                    frame,
                    dtype=np.asarray(candidate_state.timestep).dtype,
                ),
            )
            result = overlap_metric.compute_overlap(
                frame_state.current_sim_trajectory
            )
            raw_value = np.asarray(result.value)[retained]
            raw_valid = np.asarray(result.valid, dtype=bool)[retained]
        except (TypeError, ValueError, RuntimeError) as exc:
            raise M5WaymaxEvaluationError(
                "native_metric_invalid",
                "native overlap execution failed",
            ) from exc
        if raw_value.shape != (scenario.num_agents,) or raw_valid.shape != (
            scenario.num_agents,
        ):
            _fail("native_metric_invalid", "native overlap result shape is invalid")
        if np.any(raw_valid & ~np.isfinite(raw_value)):
            _fail(
                "component_nonfinite",
                "an eligible native overlap component is non-finite",
            )
        if np.any(raw_valid & ~np.isin(raw_value, (0, 1))):
            _fail(
                "native_metric_invalid",
                "an eligible native overlap component is not binary",
            )
        native_overlap[offset] = np.asarray(raw_value, dtype=bool)
        native_overlap_mask[offset] = raw_valid
    overlap_row = build_discrete_parity_row(
        parity_index=parity_index,
        policy_name=policy_name,
        metric_name="overlap",
        custom=custom_overlap[frame_slice],
        reference=native_overlap,
        custom_mask=custom_overlap_mask[frame_slice],
        reference_mask=native_overlap_mask,
    )

    (
        custom_kinematic,
        custom_kinematic_mask,
        custom_acceleration,
        custom_curvature,
    ) = kinematic_infeasibility_components(scenario, rollout)
    transition_slice = slice(_CURRENT_INDEX, _CURRENT_INDEX + M5_PARITY_TRANSITION_COUNT)
    native_kinematic = np.zeros(
        (M5_PARITY_TRANSITION_COUNT, scenario.num_agents),
        dtype=bool,
    )
    native_kinematic_mask = np.zeros_like(native_kinematic)
    native_acceleration = np.zeros(
        native_kinematic.shape,
        dtype=np.float32,
    )
    native_curvature = np.zeros_like(native_acceleration)
    native_action_mask = np.zeros_like(native_kinematic)
    kinematic_metric = comfort.KinematicsInfeasibilityMetric()
    for offset, frame in enumerate(
        range(_FIRST_PARITY_FRAME, _PARITY_FRAME_STOP)
    ):
        try:
            old_timestep = jnp.asarray(
                frame - 1,
                dtype=np.asarray(candidate_state.timestep).dtype,
            )
            timestep = jnp.asarray(
                frame,
                dtype=np.asarray(candidate_state.timestep).dtype,
            )
            action = bicycle_model.compute_inverse(
                candidate,
                old_timestep,
                dt=0.1,
            )
            result = kinematic_metric.compute_kinematics_infeasibility(
                candidate,
                timestep,
            )
            action_data = np.asarray(action.data)[retained]
            action_valid = np.asarray(action.valid, dtype=bool)[retained, 0]
            result_value = np.asarray(result.value)[retained]
            result_valid = np.asarray(result.valid, dtype=bool)[retained]
        except (TypeError, ValueError, RuntimeError) as exc:
            raise M5WaymaxEvaluationError(
                "native_metric_invalid",
                "native kinematic execution failed",
            ) from exc
        if (
            action_data.shape != (scenario.num_agents, 2)
            or action_valid.shape != (scenario.num_agents,)
            or result_value.shape != (scenario.num_agents,)
            or result_valid.shape != (scenario.num_agents,)
        ):
            _fail("native_metric_invalid", "native kinematic result shape is invalid")
        if (
            np.any(action_valid & ~np.isfinite(action_data[:, 0]))
            or np.any(action_valid & ~np.isfinite(action_data[:, 1]))
            or np.any(result_valid & ~np.isfinite(result_value))
        ):
            _fail(
                "component_nonfinite",
                "an eligible native kinematic component is non-finite",
            )
        if np.any(result_valid & ~np.isin(result_value, (0, 1))):
            _fail(
                "native_metric_invalid",
                "an eligible native kinematic flag is not binary",
            )
        native_acceleration[offset] = np.asarray(
            action_data[:, 0],
            dtype=np.float32,
        )
        native_curvature[offset] = np.asarray(
            action_data[:, 1],
            dtype=np.float32,
        )
        native_action_mask[offset] = action_valid
        native_kinematic[offset] = np.asarray(result_value, dtype=bool)
        native_kinematic_mask[offset] = result_valid
    custom_mask_window = custom_kinematic_mask[transition_slice]
    if not np.array_equal(native_action_mask, native_kinematic_mask):
        _fail(
            "component_mask_mismatch",
            "native action and kinematic result masks differ",
        )
    if not np.array_equal(custom_mask_window, native_action_mask):
        _fail(
            "component_mask_mismatch",
            "custom and native kinematic action masks differ",
        )
    custom_acc_window = custom_acceleration[transition_slice]
    custom_curvature_window = custom_curvature[transition_slice]
    custom_acc_branch = np.abs(custom_acc_window) > np.float32(10.401)
    native_acc_branch = np.abs(native_acceleration) > np.float32(10.401)
    custom_curvature_branch = np.abs(custom_curvature_window) > np.float32(0.301)
    native_curvature_branch = np.abs(native_curvature) > np.float32(0.301)
    kinematic_row = build_discrete_parity_row(
        parity_index=parity_index,
        policy_name=policy_name,
        metric_name="kinematic_infeasibility",
        custom=custom_kinematic[transition_slice],
        reference=native_kinematic,
        custom_mask=custom_mask_window,
        reference_mask=native_kinematic_mask,
        additional_mismatches=(
            custom_acc_branch != native_acc_branch,
            custom_curvature_branch != native_curvature_branch,
        ),
    )
    return (position_row, overlap_row, kinematic_row)


def _policy_execution_map(
    executions: Sequence[ExecutionRollout],
) -> Mapping[str, ExecutionRollout]:
    values = tuple(executions)
    if any(not isinstance(value, ExecutionRollout) for value in values):
        raise TypeError("executions must contain ExecutionRollout values")
    policies = [value for value in values if value.spec.role == "policy"]
    if (
        len(values) != len(M5_PARITY_POLICY_NAMES)
        or len(policies) != len(M5_PARITY_POLICY_NAMES)
        or {value.spec.name for value in policies} != set(M5_PARITY_POLICY_NAMES)
        or any(value.spec.seed != 0 for value in policies)
    ):
        _fail(
            "execution_invalid",
            "parity requires exactly the three frozen seed-zero policy executions",
        )
    return {value.spec.name: value for value in policies}


@dataclass(frozen=True, slots=True)
class WaymaxM5MetricParityAdapter:
    """Inject contract rollouts and compare native component definitions."""

    def evaluate_case(
        self,
        case: EvaluationCase,
        *,
        parity_index: int,
        executions: Sequence[ExecutionRollout],
    ) -> tuple[M5WaymaxParityRow, ...]:
        index = _plain_int(parity_index, "parity_index")
        if index >= M5_PARITY_SCENE_COUNT:
            raise ValueError("parity_index lies outside the frozen subset")
        record = _require_case_record(case)
        policy_map = _policy_execution_map(executions)
        source_before = _state_fingerprint(record.state)
        try:
            rows: list[M5WaymaxParityRow] = []
            for policy_name in M5_PARITY_POLICY_NAMES:
                execution = policy_map[policy_name]
                if (
                    execution.rollout.sim_name != execution.spec.name
                    or execution.rollout.sim_version != execution.spec.version
                    or execution.rollout.seed != execution.spec.seed
                ):
                    _fail(
                        "execution_invalid",
                        "policy rollout provenance differs from its execution spec",
                    )
                rows.extend(
                    _native_metric_rows(
                        state=record.state,
                        scenario=case.scenario,
                        rollout=execution.rollout,
                        parity_index=index,
                        policy_name=policy_name,
                    )
                )
            return tuple(rows)
        finally:
            source_after = _state_fingerprint(record.state)
            if source_after != source_before:
                _fail(
                    "source_mutated",
                    "native metric parity changed the reloaded source state",
                )


@dataclass(frozen=True, slots=True)
class M5ParityCaseInput:
    """One selected parity case plus its three policy executions."""

    parity_index: int
    case: EvaluationCase = field(repr=False)
    executions: tuple[ExecutionRollout, ...] = field(repr=False)

    def __post_init__(self) -> None:
        index = _plain_int(self.parity_index, "parity_index")
        if index >= M5_PARITY_SCENE_COUNT:
            raise ValueError("parity_index lies outside the frozen subset")
        if not isinstance(self.case, EvaluationCase):
            raise TypeError("case must be an EvaluationCase")
        executions = tuple(self.executions)
        _policy_execution_map(executions)
        object.__setattr__(self, "parity_index", index)
        object.__setattr__(self, "executions", executions)


class M5StreamingParityAccumulator:
    """Evaluate selected parity cases without retaining source-backed cases.

    The accumulator keeps only the frozen source-only selection and aggregate-safe
    parity rows.  Each :meth:`add_case` call validates the private source binding,
    executes the native comparison immediately, and then releases the caller-owned
    :class:`EvaluationCase` and rollouts.
    """

    __slots__ = ("_adapter", "_finalized", "_members", "_rows")

    def __init__(
        self,
        selection: M5ParitySelection,
        *,
        adapter: WaymaxM5MetricParityAdapter | None = None,
    ) -> None:
        if not isinstance(selection, M5ParitySelection):
            raise TypeError("selection must be an M5ParitySelection")
        if adapter is not None and not callable(
            getattr(adapter, "evaluate_case", None)
        ):
            raise TypeError("adapter must provide evaluate_case or be None")
        self._adapter = adapter or WaymaxM5MetricParityAdapter()
        self._finalized = False
        self._members = {
            member.cohort_index: member for member in selection.members
        }
        self._rows: dict[int, tuple[M5WaymaxParityRow, ...]] = {}

    @property
    def case_count(self) -> int:
        """Return the number of selected cases already consumed."""

        return len(self._rows)

    def add_case(
        self,
        case: EvaluationCase,
        executions: Sequence[ExecutionRollout],
    ) -> tuple[M5WaymaxParityRow, ...]:
        """Consume one selected case and return its aggregate-safe nine rows."""

        if self._finalized:
            _fail(
                "parity_matrix_invalid",
                "a finalized parity accumulator is immutable",
            )
        if not isinstance(case, EvaluationCase):
            raise TypeError("case must be an EvaluationCase")
        member = self._members.get(case.cohort_index)
        if member is None:
            _fail(
                "parity_matrix_invalid",
                "a parity case is not in the frozen selected membership",
            )
        if case.cohort_index in self._rows:
            _fail(
                "parity_matrix_invalid",
                "a selected parity case was evaluated more than once",
            )
        _validate_selected_case_binding(case, member)
        rows = tuple(
            self._adapter.evaluate_case(
                case,
                parity_index=member.parity_index,
                executions=executions,
            )
        )
        expected_keys = {
            (member.parity_index, policy_name, metric_name)
            for policy_name in M5_PARITY_POLICY_NAMES
            for metric_name in M5_PARITY_METRIC_NAMES
        }
        actual_keys = {
            (row.parity_index, row.policy_name, row.metric_name)
            for row in rows
            if isinstance(row, M5WaymaxParityRow)
        }
        if (
            len(rows)
            != len(M5_PARITY_POLICY_NAMES) * len(M5_PARITY_METRIC_NAMES)
            or len(actual_keys) != len(rows)
            or actual_keys != expected_keys
        ):
            _fail(
                "parity_matrix_invalid",
                "a parity case did not produce the exact nine-row matrix",
            )
        if any(row.status != "accepted" for row in rows):
            _fail(
                "parity_mismatch",
                "at least one observed native metric component failed parity",
            )
        policy_order = {
            name: index for index, name in enumerate(M5_PARITY_POLICY_NAMES)
        }
        metric_order = {
            name: index for index, name in enumerate(M5_PARITY_METRIC_NAMES)
        }
        ordered = tuple(
            sorted(
                rows,
                key=lambda row: (
                    policy_order[row.policy_name],
                    metric_order[row.metric_name],
                ),
            )
        )
        self._rows[case.cohort_index] = ordered
        return ordered

    def finalize(self) -> tuple[M5WaymaxParityRow, ...]:
        """Return the exact canonical 144-row matrix and become immutable."""

        if self._finalized:
            _fail(
                "parity_matrix_invalid",
                "a finalized parity accumulator is immutable",
            )
        missing = set(self._members).difference(self._rows)
        extra = set(self._rows).difference(self._members)
        if missing or extra or len(self._rows) != M5_PARITY_SCENE_COUNT:
            _fail(
                "parity_matrix_invalid",
                "parity rows do not cover the frozen selected membership",
            )
        ordered_members = sorted(
            self._members.values(),
            key=lambda member: member.parity_index,
        )
        rows = tuple(
            row
            for member in ordered_members
            for row in self._rows[member.cohort_index]
        )
        expected_keys = {
            (parity_index, policy_name, metric_name)
            for parity_index in range(M5_PARITY_SCENE_COUNT)
            for policy_name in M5_PARITY_POLICY_NAMES
            for metric_name in M5_PARITY_METRIC_NAMES
        }
        actual_keys = {
            (row.parity_index, row.policy_name, row.metric_name)
            for row in rows
        }
        if (
            len(rows) != M5_PARITY_ROW_COUNT
            or len(actual_keys) != M5_PARITY_ROW_COUNT
            or actual_keys != expected_keys
        ):
            _fail(
                "parity_matrix_invalid",
                "parity rows do not form the exact frozen 144-row matrix",
            )
        self._finalized = True
        return rows


def build_waymax_parity_rows(
    case_inputs: Sequence[M5ParityCaseInput],
    *,
    selection: M5ParitySelection,
    adapter: WaymaxM5MetricParityAdapter | None = None,
) -> tuple[M5WaymaxParityRow, ...]:
    """Build the exact 144-row matrix and fail closed on any rejected row."""

    if not isinstance(selection, M5ParitySelection):
        raise TypeError("selection must be an M5ParitySelection")
    inputs = tuple(case_inputs)
    if (
        len(inputs) != M5_PARITY_SCENE_COUNT
        or any(not isinstance(item, M5ParityCaseInput) for item in inputs)
    ):
        _fail(
            "parity_matrix_invalid",
            "the official parity matrix requires exactly 16 case inputs",
        )
    ordered = sorted(inputs, key=lambda item: item.parity_index)
    if (
        tuple(item.parity_index for item in ordered)
        != tuple(range(M5_PARITY_SCENE_COUNT))
        or len({item.case.cohort_index for item in ordered})
        != M5_PARITY_SCENE_COUNT
        or tuple(item.case.cohort_index for item in ordered)
        != tuple(member.cohort_index for member in selection.members)
    ):
        _fail(
            "parity_matrix_invalid",
            "parity inputs do not match the frozen ordered membership",
        )
    accumulator = M5StreamingParityAccumulator(
        selection,
        adapter=adapter,
    )
    for item in ordered:
        accumulator.add_case(
            item.case,
            item.executions,
        )
    return accumulator.finalize()


__all__ = [
    "M5_PARITY_METRIC_NAMES",
    "M5_PARITY_ORDER_VERSION",
    "M5_PARITY_POLICY_NAMES",
    "M5_PARITY_RANK_DOMAIN",
    "M5_PARITY_ROW_COUNT",
    "M5_PARITY_SCENE_COUNT",
    "M5_PARITY_TRANSITION_COUNT",
    "M5ParityCaseInput",
    "M5ParityMember",
    "M5ParityOrderReceipt",
    "M5ParitySelection",
    "M5StreamingParityAccumulator",
    "M5WaymaxEvaluationError",
    "M5WaymaxParityRow",
    "WaymaxExactLogReferenceExecutor",
    "WaymaxM5MetricParityAdapter",
    "build_continuous_parity_row",
    "build_discrete_parity_row",
    "build_waymax_parity_rows",
    "select_m5_parity_members",
]
