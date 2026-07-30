"""Bounded M6 counterfactual execution through the pinned Waymax runtime.

This module is deliberately import-safe without JAX, TensorFlow, or Waymax.  The
optional runtime is imported only by execution functions.  Source-only eligibility,
ranking, plan-view construction, conversion, and validation use NumPy and the EvalSim
contracts.

The Waymax IDM bundle is a privileged logged-trajectory waypoint-following reference.
It is not a causal policy, independent ground truth, or a numerical twin of EvalSim
IDM.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from types import MappingProxyType
from typing import Any, NamedTuple, NoReturn

import numpy as np

from evalsim.contracts import (
    Agent,
    AgentType,
    EgoTrajectoryPlan,
    InterventionEligibility,
    Rollout,
    Scenario,
)
from evalsim.contracts.counterfactual import canonical_configuration_json
from evalsim.perturb.m6 import (
    IDENTITY_FAMILY,
    LONGITUDINAL_BRAKE_PULSE_FAMILY,
    M6_INTERVENTION_VERSION,
    PRIMARY_BRAKE_MAGNITUDE_MPS2,
    evaluate_primary_brake_eligibility,
    validate_registered_ego_plan,
)
from evalsim.simulators.waymax_reference import (
    M4_INIT_STEPS,
    M4_MAX_OBJECTS,
    WAYMAX_IDM_DEFAULTS,
    assert_waymax_idm_defaults,
    initialized_overlap_mask_numpy,
)
from evalsim.sources.waymax import WAYMAX_COMMIT

M6_WAYMAX_VERSION = "0.1.0"
M6_WAYMAX_TRANSITIONS = 20
M6_WAYMAX_FRAME_COUNT = M6_WAYMAX_TRANSITIONS + 1
M6_WAYMAX_MIN_SCENES = 8
M6_WAYMAX_MAX_SCENES = 16
M6_WAYMAX_SOURCE_COHORT_SIZE = 128
M6_WAYMAX_CADENCE_MICROS = 100_000
M6_WAYMAX_RANK_DOMAIN = "evalsim-m6-waymax-reactivity-v1"
M6_WAYMAX_PLAN_VIEW_DOMAIN = b"evalsim-m6-waymax-ego-plan-view-v1"
M6_WAYMAX_SOURCE_HASH_DOMAIN = b"evalsim-m6-waymax-source-state-v1"
M6_WAYMAX_PRIMARY_DOMAIN = b"evalsim-m6-waymax-primary-domain-v1"
M6_WAYMAX_PRIMARY_ENTRY_DOMAIN = b"evalsim-m6-waymax-primary-entry-v1"
M6_WAYMAX_QUALIFICATION_DOMAIN = b"evalsim-m6-waymax-qualification-v1"
M6_WAYMAX_QUALIFICATION_LEDGER_DOMAIN = (
    b"evalsim-m6-waymax-qualification-ledger-v1"
)
M6_WAYMAX_SELECTION_DOMAIN = b"evalsim-m6-waymax-selection-v1"
M6_WAYMAX_LOGGED_WORLD = "waymax_logged_world_fallback"
M6_WAYMAX_PRIVILEGED_IDM = (
    "waymax_privileged_logged_trajectory_waypoint_following_idm"
)
M6_WAYMAX_BUNDLES = (
    M6_WAYMAX_LOGGED_WORLD,
    M6_WAYMAX_PRIVILEGED_IDM,
)
M6_WAYMAX_FLOAT_ATOL = 1e-5
M6_WAYMAX_FLOAT_RTOL = 1e-6
M6_WAYMAX_YAW_ATOL = 1e-5

_FLOAT_FIELDS = ("x", "y", "yaw", "vx", "vy")
_PLAN_FIELDS = ("x", "y", "heading", "vx", "vy")
_SOURCE_TRAJECTORY_FIELDS = (
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
_MASK_FIELDS = (
    "requested_control",
    "effective_control",
    "lifecycle_fallback",
    "initialized_overlap_excluded",
)
_ELIGIBILITY_REASONS = frozenset(
    {
        "primary_ineligible",
        "source_cadence_not_100ms",
        "target_not_requested_all_transitions",
        "target_initialized_overlap_excluded",
    }
)
_ELIGIBILITY_ISSUER = object()
_QUALIFICATION_LEDGER_ISSUER = object()
_SELECTION_ISSUER = object()


class M6WaymaxDependencyError(ImportError):
    """The optional pinned Waymax execution stack is unavailable."""


class M6WaymaxError(ValueError):
    """A fail-closed M6 Waymax adapter error with a stable reason code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> NoReturn:
    raise M6WaymaxError(code, message)


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _readonly_vector(
    value: Any,
    *,
    dtype: np.dtype[Any] | type[Any],
    name: str,
    size: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape [{size}]")
    immutable = np.frombuffer(
        np.ascontiguousarray(array).tobytes(order="C"),
        dtype=np.dtype(dtype),
    )
    immutable.setflags(write=False)
    return immutable


def _require_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
        from waymax import agents, config, datatypes, dynamics, env
    except ImportError as exc:
        raise M6WaymaxDependencyError(
            "M6 Waymax execution is optional; install it with "
            "`uv sync --extra dev --extra waymo`."
        ) from exc
    return jax, jnp, agents, config, datatypes, (dynamics, env)


def m6_waymax_rank_sha256(cohort_index: int) -> str:
    """Return the exact frozen source-only rank digest for one cohort index."""

    index = _strict_int(cohort_index, "cohort_index")
    if index > np.iinfo(np.uint32).max:
        raise ValueError("cohort_index must fit uint32")
    payload = (
        M6_WAYMAX_RANK_DOMAIN.encode("ascii")
        + b"\x00"
        + struct.pack(">I", index)
    )
    return hashlib.sha256(payload).hexdigest()


def _sha256_hex(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def _eligibility_canonical_bytes(value: InterventionEligibility) -> bytes:
    if not isinstance(value, InterventionEligibility):
        raise TypeError("upstream_eligibility must be an InterventionEligibility")
    return canonical_configuration_json(value.to_dict()).encode("utf-8")


def _primary_entry_sha256(
    *,
    cohort_index: int,
    scenario_id: str,
    source_state_sha256: str,
    upstream_eligibility: InterventionEligibility,
    target_contract_id: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(M6_WAYMAX_PRIMARY_ENTRY_DOMAIN)
    digest.update(b"\x00")
    digest.update(struct.pack(">I", cohort_index))
    scenario_bytes = scenario_id.encode("utf-8")
    digest.update(len(scenario_bytes).to_bytes(8, "big"))
    digest.update(scenario_bytes)
    digest.update(bytes.fromhex(source_state_sha256))
    eligibility_bytes = _eligibility_canonical_bytes(upstream_eligibility)
    digest.update(len(eligibility_bytes).to_bytes(8, "big"))
    digest.update(eligibility_bytes)
    digest.update(struct.pack(">q", target_contract_id))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M6WaymaxPrimaryDomainEntry:
    """One complete upstream primary-eligibility source commitment."""

    cohort_index: int
    scenario_id: str = field(repr=False)
    source_state_sha256: str = field(repr=False)
    upstream_eligibility: InterventionEligibility = field(repr=False)
    target_contract_id: int = field(repr=False)
    entry_sha256: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        index = _strict_int(self.cohort_index, "cohort_index")
        if index >= M6_WAYMAX_SOURCE_COHORT_SIZE:
            raise ValueError("primary entry lies outside the accepted cohort")
        if not isinstance(self.scenario_id, str) or not self.scenario_id:
            raise ValueError("scenario_id must be a non-empty string")
        source_sha256 = _sha256_hex(
            self.source_state_sha256,
            "source_state_sha256",
        )
        if not isinstance(self.upstream_eligibility, InterventionEligibility):
            raise TypeError("upstream_eligibility must be an InterventionEligibility")
        # Canonical roundtrip makes the entry independent of caller object identity.
        eligibility = InterventionEligibility.from_dict(
            self.upstream_eligibility.to_dict()
        )
        if not eligibility.eligible or eligibility.target_index is None:
            raise ValueError("primary-domain entries must be primary eligible")
        target_contract_id = _strict_int(
            self.target_contract_id,
            "target_contract_id",
        )
        object.__setattr__(self, "cohort_index", index)
        object.__setattr__(self, "source_state_sha256", source_sha256)
        object.__setattr__(self, "upstream_eligibility", eligibility)
        object.__setattr__(self, "target_contract_id", target_contract_id)
        expected = _primary_entry_sha256(
            cohort_index=index,
            scenario_id=self.scenario_id,
            source_state_sha256=source_sha256,
            upstream_eligibility=eligibility,
            target_contract_id=target_contract_id,
        )
        if self.entry_sha256 is not None and self.entry_sha256 != expected:
            raise ValueError("entry_sha256 does not bind the complete primary entry")
        object.__setattr__(self, "entry_sha256", expected)

    @property
    def eligibility_canonical_bytes(self) -> bytes:
        return _eligibility_canonical_bytes(self.upstream_eligibility)

    def revalidate(self) -> None:
        expected = _primary_entry_sha256(
            cohort_index=self.cohort_index,
            scenario_id=self.scenario_id,
            source_state_sha256=self.source_state_sha256,
            upstream_eligibility=self.upstream_eligibility,
            target_contract_id=self.target_contract_id,
        )
        if expected != self.entry_sha256:
            raise M6WaymaxError(
                "primary_entry_mutated",
                "the primary-domain entry failed its complete local binding",
            )


def _primary_domain_sha256(
    entries_by_index: Mapping[int, M6WaymaxPrimaryDomainEntry],
) -> str:
    digest = hashlib.sha256()
    digest.update(M6_WAYMAX_PRIMARY_DOMAIN)
    digest.update(b"\x00")
    digest.update(len(entries_by_index).to_bytes(4, "big"))
    for index, entry in sorted(entries_by_index.items()):
        digest.update(struct.pack(">I", index))
        digest.update(bytes.fromhex(entry.entry_sha256))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M6WaymaxPrimaryDomain:
    """Complete upstream primary-eligible entry ledger."""

    entries: Sequence[M6WaymaxPrimaryDomainEntry] = field(
        repr=False,
        compare=False,
    )
    domain_sha256: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.entries, (str, bytes)) or not isinstance(
            self.entries,
            Sequence,
        ):
            raise TypeError("entries must be a sequence of primary-domain entries")
        normalized: dict[int, M6WaymaxPrimaryDomainEntry] = {}
        for entry in self.entries:
            if not isinstance(entry, M6WaymaxPrimaryDomainEntry):
                raise TypeError("entries must contain M6WaymaxPrimaryDomainEntry")
            entry.revalidate()
            if entry.cohort_index in normalized:
                raise ValueError("primary-domain cohort indices must be unique")
            normalized[entry.cohort_index] = entry
        normalized = dict(sorted(normalized.items()))
        object.__setattr__(
            self,
            "entries",
            tuple(normalized.values()),
        )
        expected = _primary_domain_sha256(normalized)
        if self.domain_sha256 is not None and self.domain_sha256 != expected:
            raise ValueError("domain_sha256 does not bind the complete primary domain")
        object.__setattr__(self, "domain_sha256", expected)

    @property
    def member_count(self) -> int:
        return len(self.entries)

    @property
    def entry_by_cohort_index(
        self,
    ) -> Mapping[int, M6WaymaxPrimaryDomainEntry]:
        return MappingProxyType(
            {entry.cohort_index: entry for entry in self.entries}
        )

    @property
    def source_binding_by_cohort_index(self) -> Mapping[int, str]:
        return MappingProxyType(
            {
                entry.cohort_index: entry.source_state_sha256
                for entry in self.entries
            }
        )

    def revalidate(self) -> None:
        entries_by_index = self.entry_by_cohort_index
        for entry in entries_by_index.values():
            entry.revalidate()
        if _primary_domain_sha256(entries_by_index) != self.domain_sha256:
            raise M6WaymaxError(
                "primary_domain_mutated",
                "the primary eligibility domain failed its local binding",
            )


def _qualification_binding_sha256(
    *,
    scenario_id: str,
    source_binding_sha256: str,
    primary_entry_sha256: str,
    cohort_index: int,
    eligible: bool,
    reason: str | None,
    target_index: int | None,
    target_agent_id: int | None,
    target_slot: int | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(M6_WAYMAX_QUALIFICATION_DOMAIN)
    digest.update(b"\x00")
    scenario_bytes = scenario_id.encode("utf-8")
    digest.update(len(scenario_bytes).to_bytes(8, "big"))
    digest.update(scenario_bytes)
    digest.update(bytes.fromhex(source_binding_sha256))
    digest.update(bytes.fromhex(primary_entry_sha256))
    digest.update(struct.pack(">I", cohort_index))
    digest.update(b"\x01" if eligible else b"\x00")
    reason_bytes = b"" if reason is None else reason.encode("ascii")
    digest.update(len(reason_bytes).to_bytes(4, "big"))
    digest.update(reason_bytes)
    for value in (target_index, target_slot):
        digest.update(struct.pack(">I", 0xFFFFFFFF if value is None else value))
    digest.update(
        struct.pack(
            ">q",
            -1 if target_agent_id is None else target_agent_id,
        )
    )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M6WaymaxEligibility:
    """Source-only Waymax qualification for one primary-eligible scene."""

    cohort_index: int
    eligible: bool
    reason: str | None
    scenario_id: str = field(repr=False)
    target_index: int | None
    target_agent_id: int | None = field(repr=False)
    target_slot: int | None
    rank_sha256: str
    source_binding_sha256: str = field(repr=False, compare=False)
    primary_entry_sha256: str = field(repr=False, compare=False)
    qualification_binding_sha256: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _issued_original_binding_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _ELIGIBILITY_ISSUER:
            raise TypeError("M6WaymaxEligibility is evaluator-issued only")
        index = _strict_int(self.cohort_index, "cohort_index")
        if index >= M6_WAYMAX_SOURCE_COHORT_SIZE:
            raise ValueError("cohort_index lies outside the accepted 128-scene cohort")
        expected_rank = m6_waymax_rank_sha256(index)
        if self.rank_sha256 != expected_rank:
            raise ValueError("rank_sha256 does not match the frozen ranking rule")
        source_sha256 = _sha256_hex(
            self.source_binding_sha256,
            "source_binding_sha256",
        )
        primary_entry_sha256 = _sha256_hex(
            self.primary_entry_sha256,
            "primary_entry_sha256",
        )
        if not isinstance(self.scenario_id, str) or not self.scenario_id:
            raise ValueError("scenario_id must be a non-empty string")
        if type(self.eligible) is not bool:
            raise ValueError("eligible must be a boolean")
        if self.eligible:
            if self.reason is not None:
                raise ValueError("eligible input cannot have a reason")
            for name in ("target_index", "target_agent_id", "target_slot"):
                if getattr(self, name) is None:
                    raise ValueError(f"eligible input requires {name}")
        else:
            if self.reason not in _ELIGIBILITY_REASONS:
                raise ValueError("ineligible input requires a registered reason")
        for name in ("target_index", "target_agent_id", "target_slot"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _strict_int(value, name),
                )
        object.__setattr__(self, "cohort_index", index)
        expected_binding = _qualification_binding_sha256(
            scenario_id=self.scenario_id,
            source_binding_sha256=source_sha256,
            primary_entry_sha256=primary_entry_sha256,
            cohort_index=index,
            eligible=self.eligible,
            reason=self.reason,
            target_index=self.target_index,
            target_agent_id=self.target_agent_id,
            target_slot=self.target_slot,
        )
        if (
            self.qualification_binding_sha256 is not None
            and self.qualification_binding_sha256 != expected_binding
        ):
            raise ValueError(
                "qualification_binding_sha256 does not bind scenario/source/target"
            )
        object.__setattr__(
            self,
            "qualification_binding_sha256",
            expected_binding,
        )
        object.__setattr__(
            self,
            "_issued_original_binding_sha256",
            expected_binding,
        )

    def revalidate(self) -> None:
        expected = _qualification_binding_sha256(
            scenario_id=self.scenario_id,
            source_binding_sha256=self.source_binding_sha256,
            primary_entry_sha256=self.primary_entry_sha256,
            cohort_index=self.cohort_index,
            eligible=self.eligible,
            reason=self.reason,
            target_index=self.target_index,
            target_agent_id=self.target_agent_id,
            target_slot=self.target_slot,
        )
        if (
            expected != self.qualification_binding_sha256
            or expected != self._issued_original_binding_sha256
        ):
            raise M6WaymaxError(
                "qualification_mutated",
                "Waymax qualification failed its local scenario/source/target binding",
            )


def _qualification_ledger_sha256(
    *,
    primary_domain_sha256: str,
    primary_domain_member_count: int,
    rows: Sequence[M6WaymaxEligibility],
) -> str:
    digest = hashlib.sha256()
    digest.update(M6_WAYMAX_QUALIFICATION_LEDGER_DOMAIN)
    digest.update(b"\x00")
    digest.update(bytes.fromhex(primary_domain_sha256))
    digest.update(struct.pack(">I", primary_domain_member_count))
    for row in rows:
        digest.update(struct.pack(">I", row.cohort_index))
        digest.update(bytes.fromhex(row.qualification_binding_sha256))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M6WaymaxQualificationLedger:
    """Builder-issued complete source-only qualification ledger."""

    primary_domain_sha256: str = field(repr=False)
    primary_domain_member_count: int
    rows: tuple[M6WaymaxEligibility, ...]
    ledger_sha256: str | None = field(default=None, repr=False)
    _issued_original_ledger_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _QUALIFICATION_LEDGER_ISSUER:
            raise TypeError(
                "M6WaymaxQualificationLedger is builder-issued only"
            )
        domain_sha256 = _sha256_hex(
            self.primary_domain_sha256,
            "primary_domain_sha256",
        )
        count = _strict_int(
            self.primary_domain_member_count,
            "primary_domain_member_count",
        )
        rows = tuple(sorted(self.rows, key=lambda row: row.cohort_index))
        if len(rows) != count:
            raise ValueError(
                "qualification ledger must cover every primary-domain member"
            )
        if any(not isinstance(row, M6WaymaxEligibility) for row in rows):
            raise TypeError(
                "qualification ledger rows must be M6WaymaxEligibility"
            )
        for row in rows:
            row.revalidate()
        indices = tuple(row.cohort_index for row in rows)
        if indices != tuple(sorted(set(indices))):
            raise ValueError("qualification ledger cohort indices must be unique")
        object.__setattr__(self, "primary_domain_sha256", domain_sha256)
        object.__setattr__(self, "primary_domain_member_count", count)
        object.__setattr__(self, "rows", rows)
        expected = _qualification_ledger_sha256(
            primary_domain_sha256=domain_sha256,
            primary_domain_member_count=count,
            rows=rows,
        )
        if self.ledger_sha256 is not None and self.ledger_sha256 != expected:
            raise ValueError("ledger_sha256 does not bind the complete ledger")
        object.__setattr__(self, "ledger_sha256", expected)
        object.__setattr__(
            self,
            "_issued_original_ledger_sha256",
            expected,
        )

    def revalidate(self) -> None:
        for row in self.rows:
            row.revalidate()
        if (
            len(self.rows) != self.primary_domain_member_count
            or tuple(row.cohort_index for row in self.rows)
            != tuple(sorted({row.cohort_index for row in self.rows}))
            or _qualification_ledger_sha256(
                primary_domain_sha256=self.primary_domain_sha256,
                primary_domain_member_count=self.primary_domain_member_count,
                rows=self.rows,
            )
            != self.ledger_sha256
            or self.ledger_sha256 != self._issued_original_ledger_sha256
        ):
            _fail(
                "qualification_ledger_mutated",
                "complete Waymax qualification ledger failed revalidation",
            )


def _canonical_selection_members(
    ledger: M6WaymaxQualificationLedger,
) -> tuple[M6WaymaxEligibility, ...]:
    eligible = tuple(
        sorted(
            (row for row in ledger.rows if row.eligible),
            key=lambda item: (
                bytes.fromhex(item.rank_sha256),
                item.cohort_index,
            ),
        )
    )
    if len(eligible) < M6_WAYMAX_MIN_SCENES:
        return ()
    return eligible[:M6_WAYMAX_MAX_SCENES]


def _selection_sha256(
    *,
    ledger_sha256: str,
    supported: bool,
    eligible_count: int,
    members: Sequence[M6WaymaxEligibility],
) -> str:
    digest = hashlib.sha256()
    digest.update(M6_WAYMAX_SELECTION_DOMAIN)
    digest.update(b"\x00")
    digest.update(bytes.fromhex(ledger_sha256))
    digest.update(b"\x01" if supported else b"\x00")
    digest.update(struct.pack(">I", eligible_count))
    digest.update(struct.pack(">I", len(members)))
    for position, member in enumerate(members):
        digest.update(struct.pack(">I", position))
        digest.update(struct.pack(">I", member.cohort_index))
        digest.update(bytes.fromhex(member.qualification_binding_sha256))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class M6WaymaxSelection:
    """Selector-issued canonical 16-or-floor result over a complete ledger."""

    supported: bool
    primary_domain_sha256: str = field(repr=False)
    primary_domain_member_count: int
    eligible_count: int
    members: tuple[M6WaymaxEligibility, ...]
    qualification_ledger: M6WaymaxQualificationLedger = field(
        repr=False,
        compare=False,
    )
    qualification_ledger_sha256: str = field(repr=False)
    selection_sha256: str | None = field(default=None, repr=False)
    _issued_original_selection_sha256: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _SELECTION_ISSUER:
            raise TypeError("M6WaymaxSelection is selector-issued only")
        self._validate_and_normalize()
        object.__setattr__(
            self,
            "_issued_original_selection_sha256",
            self.selection_sha256,
        )

    def _validate_and_normalize(self) -> None:
        if not isinstance(
            self.qualification_ledger,
            M6WaymaxQualificationLedger,
        ):
            raise TypeError(
                "selection requires a complete M6WaymaxQualificationLedger"
            )
        self.qualification_ledger.revalidate()
        if (
            self.qualification_ledger_sha256
            != self.qualification_ledger.ledger_sha256
        ):
            raise ValueError(
                "selection qualification_ledger_sha256 does not match its ledger"
            )
        if (
            self.primary_domain_sha256
            != self.qualification_ledger.primary_domain_sha256
            or self.primary_domain_member_count
            != self.qualification_ledger.primary_domain_member_count
        ):
            raise ValueError("selection primary domain differs from its ledger")
        eligible_count = sum(
            row.eligible for row in self.qualification_ledger.rows
        )
        canonical_members = _canonical_selection_members(
            self.qualification_ledger
        )
        supported = eligible_count >= M6_WAYMAX_MIN_SCENES
        if (
            self.supported is not supported
            or self.eligible_count != eligible_count
            or tuple(self.members) != canonical_members
        ):
            raise ValueError(
                "selection is not the canonical complete-ledger 16-or-floor result"
            )
        object.__setattr__(self, "eligible_count", eligible_count)
        object.__setattr__(self, "members", canonical_members)
        expected = _selection_sha256(
            ledger_sha256=self.qualification_ledger.ledger_sha256,
            supported=supported,
            eligible_count=eligible_count,
            members=canonical_members,
        )
        if self.selection_sha256 is not None and self.selection_sha256 != expected:
            raise ValueError("selection_sha256 does not bind the canonical selection")
        object.__setattr__(self, "selection_sha256", expected)

    def revalidate(
        self,
        *,
        primary_domain: M6WaymaxPrimaryDomain | None = None,
    ) -> None:
        try:
            self._validate_and_normalize()
            if (
                self.selection_sha256
                != self._issued_original_selection_sha256
            ):
                raise ValueError(
                    "selection differs from its selector-issued binding"
                )
            if primary_domain is not None:
                primary_domain.revalidate()
                if (
                    primary_domain.domain_sha256
                    != self.primary_domain_sha256
                    or primary_domain.member_count
                    != self.primary_domain_member_count
                ):
                    raise ValueError(
                        "selection differs from supplied primary domain"
                    )
        except (TypeError, ValueError) as exc:
            _fail(
                "selection_mutated",
                f"canonical Waymax selection failed revalidation: {exc}",
            )


def build_m6_waymax_qualification_ledger(
    eligibility: Sequence[M6WaymaxEligibility],
    *,
    primary_domain: M6WaymaxPrimaryDomain,
    primary_scenarios: Mapping[int, Scenario],
) -> M6WaymaxQualificationLedger:
    """Validate and issue the complete source-only qualification ledger."""

    if not isinstance(primary_domain, M6WaymaxPrimaryDomain):
        raise TypeError("primary_domain must be an M6WaymaxPrimaryDomain")
    primary_domain.revalidate()
    if not isinstance(primary_scenarios, Mapping):
        raise TypeError("primary_scenarios must be a cohort-index mapping")
    for scenario_index, scenario in primary_scenarios.items():
        _strict_int(scenario_index, "primary_scenarios cohort index")
        if not isinstance(scenario, Scenario):
            raise TypeError("primary_scenarios values must be Scenario instances")
    rows = tuple(eligibility)
    if any(not isinstance(row, M6WaymaxEligibility) for row in rows):
        raise TypeError("eligibility rows must be M6WaymaxEligibility values")
    if len({row.cohort_index for row in rows}) != len(rows):
        raise ValueError("eligibility rows contain duplicate cohort indices")
    entry_by_index = primary_domain.entry_by_cohort_index
    expected_indices = set(entry_by_index)
    actual_indices = {row.cohort_index for row in rows}
    scenario_indices = set(primary_scenarios)
    if (
        actual_indices != expected_indices
        or scenario_indices != expected_indices
    ):
        raise M6WaymaxError(
            "primary_domain_incomplete",
            "Waymax rows do not exactly cover the upstream primary domain",
        )
    for row in rows:
        row.revalidate()
        entry = entry_by_index[row.cohort_index]
        scenario = primary_scenarios[row.cohort_index]
        recomputed = _validate_primary_entry_against_scenario(
            entry,
            scenario,
            source_state_sha256=None,
        )
        if (
            row.source_binding_sha256 != entry.source_state_sha256
            or row.primary_entry_sha256 != entry.entry_sha256
            or row.scenario_id != entry.scenario_id
            or row.target_index != recomputed.target_index
            or row.target_agent_id != entry.target_contract_id
        ):
            raise M6WaymaxError(
                "primary_domain_source_mismatch",
                "Waymax row differs from the complete primary-domain entry",
            )
    return M6WaymaxQualificationLedger(
        primary_domain_sha256=primary_domain.domain_sha256,
        primary_domain_member_count=primary_domain.member_count,
        rows=rows,
        _issuance_capability=_QUALIFICATION_LEDGER_ISSUER,
    )


def select_m6_waymax_subset(
    eligibility: Sequence[M6WaymaxEligibility]
    | M6WaymaxQualificationLedger,
    *,
    primary_domain: M6WaymaxPrimaryDomain,
    primary_scenarios: Mapping[int, Scenario] | None = None,
) -> M6WaymaxSelection:
    """Issue the exact first-16/all-8--15 selection from a complete ledger."""

    if isinstance(eligibility, M6WaymaxQualificationLedger):
        ledger = eligibility
        ledger.revalidate()
        primary_domain.revalidate()
        if (
            ledger.primary_domain_sha256 != primary_domain.domain_sha256
            or ledger.primary_domain_member_count != primary_domain.member_count
        ):
            _fail(
                "selection_domain",
                "qualification ledger differs from the supplied primary domain",
            )
    else:
        if primary_scenarios is None:
            raise TypeError(
                "primary_scenarios is required when building a ledger from rows"
            )
        ledger = build_m6_waymax_qualification_ledger(
            eligibility,
            primary_domain=primary_domain,
            primary_scenarios=primary_scenarios,
        )
    eligible_count = sum(row.eligible for row in ledger.rows)
    members = _canonical_selection_members(ledger)
    return M6WaymaxSelection(
        supported=eligible_count >= M6_WAYMAX_MIN_SCENES,
        primary_domain_sha256=ledger.primary_domain_sha256,
        primary_domain_member_count=ledger.primary_domain_member_count,
        eligible_count=eligible_count,
        members=members,
        qualification_ledger=ledger,
        qualification_ledger_sha256=ledger.ledger_sha256,
        _issuance_capability=_SELECTION_ISSUER,
    )


def _hash_tree(hasher: Any, value: Any) -> None:
    """Deterministically hash an in-memory state without importing JAX."""

    if dataclasses.is_dataclass(value):
        hasher.update(b"D")
        for item in dataclasses.fields(value):
            encoded = item.name.encode("utf-8")
            hasher.update(len(encoded).to_bytes(4, "big"))
            hasher.update(encoded)
            _hash_tree(hasher, getattr(value, item.name))
        return
    if isinstance(value, Mapping):
        hasher.update(b"M")
        for key in sorted(value, key=lambda item: str(item)):
            _hash_tree(hasher, key)
            _hash_tree(hasher, value[key])
        return
    if isinstance(value, (tuple, list)):
        hasher.update(b"S")
        hasher.update(len(value).to_bytes(8, "big"))
        for item in value:
            _hash_tree(hasher, item)
        return
    if value is None:
        hasher.update(b"N")
        return
    if isinstance(value, (str, bytes)):
        raw = value.encode("utf-8") if isinstance(value, str) else value
        hasher.update(b"T")
        hasher.update(len(raw).to_bytes(8, "big"))
        hasher.update(raw)
        return
    if isinstance(value, (bool, int, float, np.generic)):
        raw = repr(np.asarray(value).item()).encode("ascii")
        hasher.update(b"V")
        hasher.update(len(raw).to_bytes(8, "big"))
        hasher.update(raw)
        return
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"unsupported source-state leaf {type(value).__name__}"
        ) from exc
    if array.dtype == object:
        raise TypeError("source-state object arrays cannot be mutation-hashed")
    contiguous = np.ascontiguousarray(array)
    header = (
        contiguous.dtype.str.encode("ascii")
        + b"\x00"
        + repr(contiguous.shape).encode("ascii")
    )
    hasher.update(b"A")
    hasher.update(len(header).to_bytes(8, "big"))
    hasher.update(header)
    hasher.update(contiguous.tobytes(order="C"))


def source_state_mutation_sha256(state: Any) -> str:
    """Return a local-only mutation detector over the complete state pytree."""

    digest = hashlib.sha256()
    digest.update(M6_WAYMAX_SOURCE_HASH_DOMAIN)
    digest.update(b"\x00")
    _hash_tree(digest, state)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _SourceBinding:
    scenario_id: str = field(repr=False)
    source_state_sha256: str = field(repr=False)
    retained_slots: np.ndarray = field(repr=False, compare=False)
    ego_slot: int
    current_index: int


def _validate_source_binding(state: Any, scenario: Scenario) -> _SourceBinding:
    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    if tuple(getattr(state, "shape", ())) != ():
        _fail("state_batched", "M6 requires sequential unbatched scene execution")
    trajectory = getattr(state, "log_trajectory", None)
    metadata = getattr(state, "object_metadata", None)
    if trajectory is None or metadata is None:
        _fail("state_schema", "log_trajectory and object_metadata are required")
    if (
        int(getattr(trajectory, "num_objects", -1)) != M4_MAX_OBJECTS
        or int(getattr(trajectory, "num_timesteps", -1)) != 91
    ):
        _fail("state_shape", "M6 requires one 128-slot, 91-frame Waymax state")
    current = scenario.metadata.get("current_index")
    if (
        isinstance(current, (bool, np.bool_))
        or not isinstance(current, (int, np.integer))
        or int(current) != M4_INIT_STEPS - 1
    ):
        _fail("current_boundary", "the source current_index must be exactly 10")
    if scenario.num_steps != 91:
        _fail("scenario_horizon", "the source-neutral scenario must have 91 frames")

    valid = np.asarray(trajectory.valid, dtype=bool)
    retained = np.flatnonzero(np.any(valid, axis=1))
    declared = np.asarray(metadata.is_valid, dtype=bool)
    if not np.array_equal(np.any(valid, axis=1), declared):
        _fail("source_validity", "metadata validity differs from trajectory validity")
    ids = np.asarray(metadata.ids)[retained]
    scenario_ids = np.asarray(
        [int(agent.id) for agent in scenario.agents],
        dtype=ids.dtype,
    )
    if not np.array_equal(ids, scenario_ids):
        _fail("agent_binding", "scenario agent order/IDs differ from source slots")
    is_sdc = np.asarray(metadata.is_sdc, dtype=bool)
    if np.count_nonzero(is_sdc & declared) != 1:
        _fail("sdc_binding", "source state must contain exactly one retained SDC")
    ego_slot = int(np.flatnonzero(is_sdc & declared)[0])
    if (
        scenario.ego_index >= retained.size
        or int(retained[scenario.ego_index]) != ego_slot
        or int(scenario.ego.id) != int(np.asarray(metadata.ids)[ego_slot])
    ):
        _fail("sdc_binding", "source SDC differs from the scenario ego")

    # Re-establish the complete pure-adapter seam.  This prevents a caller from
    # pairing a canonical plan compiled from one Scenario with a native state that
    # happens to reuse its IDs.
    source_timestamps = np.asarray(trajectory.timestamp_micros)
    canonical_micros = np.empty(91, dtype=np.int64)
    for frame in range(91):
        contributors = source_timestamps[:, frame][valid[:, frame]].astype(
            np.int64,
            copy=False,
        )
        if contributors.size == 0 or not np.all(contributors == contributors[0]):
            _fail(
                "source_timeline",
                "valid source objects do not provide one timestamp consensus",
            )
        canonical_micros[frame] = contributors[0]
    expected_timestamps = (
        canonical_micros - canonical_micros[0]
    ).astype(np.float64) * 1e-6
    if not np.array_equal(
        np.asarray(scenario.timestamps, dtype=np.float64),
        expected_timestamps,
    ):
        _fail("source_timeline", "Scenario timestamps differ from source microseconds")
    source_fields = {
        "x": "x",
        "y": "y",
        "heading": "yaw",
        "vx": "vel_x",
        "vy": "vel_y",
    }
    type_map = {
        1: AgentType.VEHICLE,
        2: AgentType.PEDESTRIAN,
        3: AgentType.CYCLIST,
    }
    source_types = np.asarray(metadata.object_types)
    for scenario_index, slot in enumerate(retained):
        agent = scenario.agents[scenario_index]
        slot_valid = valid[slot]
        if agent.type != type_map.get(int(source_types[slot]), AgentType.UNKNOWN):
            _fail("agent_binding", "Scenario type differs from its source slot")
        if not np.array_equal(np.asarray(agent.valid), slot_valid):
            _fail("agent_binding", "Scenario validity differs from its source slot")
        valid_frames = np.flatnonzero(slot_valid)
        if not valid_frames.size:
            _fail("agent_binding", "retained source slot has no valid frame")
        first_valid = int(valid_frames[0])
        for name in ("length", "width"):
            dimensions = np.asarray(getattr(trajectory, name)[slot], dtype=np.float64)
            if (
                not np.all(dimensions[slot_valid] == dimensions[first_valid])
                or float(getattr(agent, name)) != float(dimensions[first_valid])
            ):
                _fail(
                    "agent_binding",
                    f"Scenario {name} differs from its source slot",
                )
        for target_name, source_name in source_fields.items():
            expected = np.asarray(
                getattr(trajectory, source_name)[slot],
                dtype=np.float64,
            )
            if target_name == "heading":
                expected = (expected + np.pi) % (2.0 * np.pi) - np.pi
            expected = np.where(slot_valid, expected, 0.0)
            if not np.array_equal(
                np.asarray(getattr(agent, target_name), dtype=np.float64),
                expected,
            ):
                _fail(
                    "agent_binding",
                    f"Scenario {target_name} differs from its source slot",
                )
    return _SourceBinding(
        scenario_id=scenario.scenario_id,
        source_state_sha256=source_state_mutation_sha256(state),
        retained_slots=retained,
        ego_slot=ego_slot,
        current_index=int(current),
    )


def _validate_primary_entry_against_scenario(
    entry: M6WaymaxPrimaryDomainEntry,
    scenario: Scenario,
    *,
    source_state_sha256: str | None,
) -> InterventionEligibility:
    """Independently recompute and bind the upstream primary decision."""

    if not isinstance(entry, M6WaymaxPrimaryDomainEntry):
        raise TypeError("entry must be an M6WaymaxPrimaryDomainEntry")
    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    entry.revalidate()
    recomputed = evaluate_primary_brake_eligibility(scenario)
    if (
        entry.scenario_id != scenario.scenario_id
        or entry.eligibility_canonical_bytes
        != _eligibility_canonical_bytes(recomputed)
    ):
        _fail(
            "primary_entry_mismatch",
            "scenario identity or recomputed primary eligibility differs from entry",
        )
    if source_state_sha256 is not None and (
        entry.source_state_sha256 != source_state_sha256
    ):
        _fail(
            "primary_entry_mismatch",
            "source-state identity differs from the primary entry",
        )
    if not recomputed.eligible or recomputed.target_index is None:
        _fail(
            "primary_entry_mismatch",
            "the independently recomputed primary decision is not eligible",
        )
    if (
        recomputed.target_index >= scenario.num_agents
        or int(scenario.agents[recomputed.target_index].id)
        != entry.target_contract_id
    ):
        _fail(
            "primary_entry_mismatch",
            "scenario-derived target contract ID differs from the primary entry",
        )
    return recomputed


def build_m6_waymax_primary_domain_entry(
    state: Any,
    scenario: Scenario,
    upstream_eligibility: InterventionEligibility,
    *,
    cohort_index: int,
) -> M6WaymaxPrimaryDomainEntry:
    """Bind one upstream primary-eligible member before Waymax qualification."""

    if not isinstance(upstream_eligibility, InterventionEligibility):
        raise TypeError("upstream_eligibility must be an InterventionEligibility")
    binding = _validate_source_binding(state, scenario)
    recomputed = evaluate_primary_brake_eligibility(scenario)
    if _eligibility_canonical_bytes(upstream_eligibility) != (
        _eligibility_canonical_bytes(recomputed)
    ):
        _fail(
            "primary_entry_mismatch",
            "supplied upstream eligibility differs from independent recomputation",
        )
    if not recomputed.eligible or recomputed.target_index is None:
        _fail(
            "primary_entry_mismatch",
            "only primary-eligible scenes belong to the Waymax primary domain",
        )
    entry = M6WaymaxPrimaryDomainEntry(
        cohort_index=cohort_index,
        scenario_id=scenario.scenario_id,
        source_state_sha256=binding.source_state_sha256,
        upstream_eligibility=recomputed,
        target_contract_id=int(
            scenario.agents[recomputed.target_index].id
        ),
    )
    _validate_primary_entry_against_scenario(
        entry,
        scenario,
        source_state_sha256=binding.source_state_sha256,
    )
    return entry


def _source_control_masks(
    state: Any,
    *,
    current_index: int,
    bundle: str,
) -> dict[str, np.ndarray]:
    if bundle not in M6_WAYMAX_BUNDLES:
        raise ValueError(f"bundle must be one of {M6_WAYMAX_BUNDLES}")
    metadata = state.object_metadata
    non_sdc_vehicle = (
        ~np.asarray(metadata.is_sdc, dtype=bool)
        & (np.asarray(metadata.object_types) == 1)
    )
    if bundle == M6_WAYMAX_PRIVILEGED_IDM:
        valid = np.asarray(state.log_trajectory.valid, dtype=bool)
        requested = np.stack(
            [
                non_sdc_vehicle & valid[:, frame] & valid[:, frame + 1]
                for frame in range(
                    current_index,
                    current_index + M6_WAYMAX_TRANSITIONS,
                )
            ],
            axis=0,
        )
        overlap = initialized_overlap_mask_numpy(state)
        effective = requested & ~overlap[np.newaxis, :]
        overlap_excluded = requested & overlap[np.newaxis, :]
    else:
        requested = np.zeros(
            (M6_WAYMAX_TRANSITIONS, M4_MAX_OBJECTS),
            dtype=bool,
        )
        effective = np.zeros_like(requested)
        overlap_excluded = np.zeros_like(requested)
    lifecycle = np.broadcast_to(
        non_sdc_vehicle,
        requested.shape,
    ).copy() & ~requested
    return {
        "requested_control": requested,
        "effective_control": effective,
        "lifecycle_fallback": lifecycle,
        "initialized_overlap_excluded": overlap_excluded,
    }


def evaluate_m6_waymax_eligibility(
    state: Any,
    scenario: Scenario,
    primary_eligibility: InterventionEligibility,
    *,
    cohort_index: int,
    primary_entry: M6WaymaxPrimaryDomainEntry,
) -> M6WaymaxEligibility:
    """Apply the frozen source-only cadence, target-control, and overlap gates."""

    if not isinstance(primary_eligibility, InterventionEligibility):
        raise TypeError("primary_eligibility must be an InterventionEligibility")
    if not isinstance(primary_entry, M6WaymaxPrimaryDomainEntry):
        raise TypeError("primary_entry must be an M6WaymaxPrimaryDomainEntry")
    index = _strict_int(cohort_index, "cohort_index")
    binding = _validate_source_binding(state, scenario)
    if primary_entry.cohort_index != index:
        _fail(
            "primary_entry_mismatch",
            "the primary entry belongs to a different cohort index",
        )
    recomputed = _validate_primary_entry_against_scenario(
        primary_entry,
        scenario,
        source_state_sha256=binding.source_state_sha256,
    )
    if (
        _eligibility_canonical_bytes(primary_eligibility)
        != primary_entry.eligibility_canonical_bytes
        or _eligibility_canonical_bytes(primary_eligibility)
        != _eligibility_canonical_bytes(recomputed)
    ):
        _fail(
            "primary_entry_mismatch",
            "supplied primary eligibility differs from the bound upstream entry",
        )
    rank = m6_waymax_rank_sha256(index)
    target_index = recomputed.target_index

    def rejected(
        reason: str,
        *,
        target_slot: int | None = None,
        target_agent_id: int | None = None,
    ) -> M6WaymaxEligibility:
        return M6WaymaxEligibility(
            cohort_index=index,
            eligible=False,
            reason=reason,
            scenario_id=scenario.scenario_id,
            target_index=target_index,
            target_agent_id=target_agent_id,
            target_slot=target_slot,
            rank_sha256=rank,
            source_binding_sha256=binding.source_state_sha256,
            primary_entry_sha256=primary_entry.entry_sha256,
            _issuance_capability=_ELIGIBILITY_ISSUER,
        )

    if target_index is None:
        _fail(
            "primary_entry_mismatch",
            "the independently recomputed primary target is absent",
        )
    if recomputed.analysis_window != (
        binding.current_index,
        binding.current_index + 40,
    ):
        _fail("primary_binding", "primary eligibility uses a different source window")
    if not 0 <= target_index < scenario.num_agents:
        _fail("target_binding", "primary target_index is outside the scenario")
    target_slot = int(binding.retained_slots[target_index])
    target_agent_id = int(scenario.agents[target_index].id)
    if (
        target_agent_id != primary_entry.target_contract_id
        or int(np.asarray(state.object_metadata.ids)[target_slot]) != target_agent_id
    ):
        _fail("target_binding", "primary target identity differs from source")

    source_micros = np.asarray(state.log_trajectory.timestamp_micros)
    ego_micros = source_micros[
        binding.ego_slot,
        binding.current_index : binding.current_index + M6_WAYMAX_FRAME_COUNT,
    ]
    if (
        ego_micros.shape != (M6_WAYMAX_FRAME_COUNT,)
        or not np.all(np.diff(ego_micros.astype(np.int64)) == M6_WAYMAX_CADENCE_MICROS)
    ):
        return rejected(
            "source_cadence_not_100ms",
            target_slot=target_slot,
            target_agent_id=target_agent_id,
        )
    masks = _source_control_masks(
        state,
        current_index=binding.current_index,
        bundle=M6_WAYMAX_PRIVILEGED_IDM,
    )
    if not bool(np.all(masks["requested_control"][:, target_slot])):
        return rejected(
            "target_not_requested_all_transitions",
            target_slot=target_slot,
            target_agent_id=target_agent_id,
        )
    if bool(np.any(masks["initialized_overlap_excluded"][:, target_slot])):
        return rejected(
            "target_initialized_overlap_excluded",
            target_slot=target_slot,
            target_agent_id=target_agent_id,
        )
    return M6WaymaxEligibility(
        cohort_index=index,
        eligible=True,
        reason=None,
        scenario_id=scenario.scenario_id,
        target_index=target_index,
        target_agent_id=target_agent_id,
        target_slot=target_slot,
        rank_sha256=rank,
        source_binding_sha256=binding.source_state_sha256,
        primary_entry_sha256=primary_entry.entry_sha256,
        _issuance_capability=_ELIGIBILITY_ISSUER,
    )


@dataclass(frozen=True, slots=True)
class WaymaxEgoPlanView:
    """Immutable canonical 21-frame little-endian float32 Waymax ego-plan view."""

    scenario_id: str = field(repr=False)
    source_state_sha256: str = field(repr=False)
    ego_agent_id: int
    ego_slot: int
    current_index: int
    intervention_family: str
    intervention_version: str
    intervention_dose: float
    perturbation_identity: str
    canonical_plan_fingerprint: str = field(repr=False)
    canonical_plan_audit_fingerprint: str = field(repr=False)
    timestamps_micros: np.ndarray = field(repr=False, compare=False)
    valid: np.ndarray = field(repr=False, compare=False)
    applied: np.ndarray = field(repr=False, compare=False)
    x: np.ndarray = field(repr=False, compare=False)
    y: np.ndarray = field(repr=False, compare=False)
    heading: np.ndarray = field(repr=False, compare=False)
    vx: np.ndarray = field(repr=False, compare=False)
    vy: np.ndarray = field(repr=False, compare=False)
    local_mutation_sha256: str | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        for name in (
            "source_state_sha256",
            "canonical_plan_fingerprint",
            "canonical_plan_audit_fingerprint",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{name} must be lowercase SHA-256 hex")
        for name in ("ego_agent_id", "ego_slot", "current_index"):
            object.__setattr__(
                self,
                name,
                _strict_int(getattr(self, name), name),
            )
        if (
            not isinstance(self.perturbation_identity, str)
            or not self.perturbation_identity
        ):
            raise ValueError("perturbation_identity must be non-empty")
        if (
            not isinstance(self.intervention_family, str)
            or not self.intervention_family
            or not isinstance(self.intervention_version, str)
            or not self.intervention_version
        ):
            raise ValueError("intervention family/version must be non-empty")
        dose = float(self.intervention_dose)
        if not math.isfinite(dose) or dose < 0.0:
            raise ValueError("intervention_dose must be finite and non-negative")
        object.__setattr__(self, "intervention_dose", dose)
        object.__setattr__(
            self,
            "timestamps_micros",
            _readonly_vector(
                self.timestamps_micros,
                dtype=np.dtype("<i8"),
                name="timestamps_micros",
                size=M6_WAYMAX_FRAME_COUNT,
            ),
        )
        for name in ("valid", "applied"):
            object.__setattr__(
                self,
                name,
                _readonly_vector(
                    getattr(self, name),
                    dtype=np.bool_,
                    name=name,
                    size=M6_WAYMAX_FRAME_COUNT,
                ),
            )
        for name in _PLAN_FIELDS:
            object.__setattr__(
                self,
                name,
                _readonly_vector(
                    getattr(self, name),
                    dtype=np.dtype("<f4"),
                    name=name,
                    size=M6_WAYMAX_FRAME_COUNT,
                ),
            )
        if not bool(np.all(self.valid)):
            raise ValueError("the 21-frame Waymax ego plan must remain valid")
        if bool(self.applied[0]) or not bool(np.all(self.applied[1:])):
            raise ValueError("Waymax applied mask must be false then 20 true values")
        if not all(np.all(np.isfinite(getattr(self, name))) for name in _PLAN_FIELDS):
            raise ValueError("Waymax ego-plan floats must be finite")
        expected = hashlib.sha256(self.canonical_bytes).hexdigest()
        supplied = self.local_mutation_sha256
        if supplied is not None and supplied != expected:
            raise ValueError("local_mutation_sha256 differs from canonical view")
        object.__setattr__(self, "local_mutation_sha256", expected)

    @property
    def canonical_bytes(self) -> bytes:
        parts = [M6_WAYMAX_PLAN_VIEW_DOMAIN, b"\x00"]
        for text in (
            self.scenario_id,
            self.source_state_sha256,
            self.intervention_family,
            self.intervention_version,
            self.perturbation_identity,
            self.canonical_plan_fingerprint,
            self.canonical_plan_audit_fingerprint,
        ):
            raw = text.encode("utf-8")
            parts.extend((len(raw).to_bytes(8, "big"), raw))
        parts.extend(
            (
                struct.pack(">q", self.ego_agent_id),
                struct.pack(">I", self.ego_slot),
                struct.pack(">I", self.current_index),
                struct.pack("<d", self.intervention_dose),
                struct.pack(">I", M6_WAYMAX_FRAME_COUNT),
                np.asarray(
                    self.timestamps_micros,
                    dtype="<i8",
                ).tobytes(order="C"),
                self.valid.astype(np.uint8, copy=False).tobytes(order="C"),
                self.applied.astype(np.uint8, copy=False).tobytes(order="C"),
            )
        )
        parts.extend(
            np.asarray(getattr(self, name), dtype="<f4").tobytes(order="C")
            for name in _PLAN_FIELDS
        )
        return b"".join(parts)

    @property
    def future_action_data(self) -> np.ndarray:
        values = np.stack(
            (
                self.x[1:],
                self.y[1:],
                self.heading[1:],
                self.vx[1:],
                self.vy[1:],
            ),
            axis=-1,
        ).astype("<f4", copy=False)
        immutable = np.frombuffer(
            np.ascontiguousarray(values).tobytes(order="C"),
            dtype="<f4",
        ).reshape(M6_WAYMAX_TRANSITIONS, 5)
        immutable.setflags(write=False)
        return immutable

    def revalidate(self) -> None:
        if hashlib.sha256(self.canonical_bytes).hexdigest() != (
            self.local_mutation_sha256
        ):
            raise M6WaymaxError(
                "plan_view_mutated",
                "the immutable Waymax ego-plan view failed its local hash",
            )


def build_waymax_ego_plan_view(
    state: Any,
    scenario: Scenario,
    plan: EgoTrajectoryPlan,
) -> WaymaxEgoPlanView:
    """Cast one registered identity or primary-brake plan once to float32."""

    binding = _validate_source_binding(state, scenario)
    if not isinstance(plan, EgoTrajectoryPlan):
        raise TypeError("plan must be an EgoTrajectoryPlan")
    plan.revalidate()
    source_before = binding.source_state_sha256
    plan_before = plan.serialize()
    validate_registered_ego_plan(scenario, plan)
    is_identity = (
        plan.spec.family == IDENTITY_FAMILY
        and plan.spec.version == M6_INTERVENTION_VERSION
    )
    is_primary = (
        plan.spec.family == LONGITUDINAL_BRAKE_PULSE_FAMILY
        and plan.spec.version == M6_INTERVENTION_VERSION
        and plan.spec.dose == PRIMARY_BRAKE_MAGNITUDE_MPS2
    )
    if not (is_identity or is_primary):
        _fail(
            "plan_scope",
            "the bounded Waymax gate accepts only identity/v1 or primary b=2",
        )
    source_timestamps = np.asarray(
        state.log_trajectory.timestamp_micros,
    )[
        binding.ego_slot,
        binding.current_index : binding.current_index + M6_WAYMAX_FRAME_COUNT,
    ].astype("<i8", copy=True)
    if not np.array_equal(
        np.asarray(plan.timestamps[:M6_WAYMAX_FRAME_COUNT]),
        np.asarray(
            scenario.timestamps[
                binding.current_index : (
                    binding.current_index + M6_WAYMAX_FRAME_COUNT
                )
            ],
            dtype=np.float64,
        ),
    ):
        _fail("plan_time_binding", "canonical plan timestamps differ from source")
    if not np.all(
        np.diff(source_timestamps.astype(np.int64))
        == M6_WAYMAX_CADENCE_MICROS
    ):
        _fail("source_cadence", "Waymax plan view requires exact 100000us cadence")
    view = WaymaxEgoPlanView(
        scenario_id=scenario.scenario_id,
        source_state_sha256=source_before,
        ego_agent_id=int(scenario.ego.id),
        ego_slot=binding.ego_slot,
        current_index=binding.current_index,
        intervention_family=plan.spec.family,
        intervention_version=plan.spec.version,
        intervention_dose=plan.spec.dose,
        perturbation_identity=plan.perturbation_identity,
        canonical_plan_fingerprint=plan.plan_fingerprint,
        canonical_plan_audit_fingerprint=plan.audit_fingerprint,
        timestamps_micros=source_timestamps,
        valid=plan.valid[:M6_WAYMAX_FRAME_COUNT],
        applied=plan.applied[:M6_WAYMAX_FRAME_COUNT],
        x=plan.x[:M6_WAYMAX_FRAME_COUNT],
        y=plan.y[:M6_WAYMAX_FRAME_COUNT],
        heading=plan.heading[:M6_WAYMAX_FRAME_COUNT],
        vx=plan.vx[:M6_WAYMAX_FRAME_COUNT],
        vy=plan.vy[:M6_WAYMAX_FRAME_COUNT],
    )
    if source_state_mutation_sha256(state) != source_before:
        _fail("source_mutated", "plan-view construction mutated the source state")
    plan.revalidate()
    if plan.serialize() != plan_before:
        _fail("canonical_plan_mutated", "plan-view construction mutated float64 plan")
    return view


def _independently_rebuild_waymax_ego_plan_view(
    state: Any,
    scenario: Scenario,
    plan: EgoTrajectoryPlan,
) -> WaymaxEgoPlanView:
    """Rebuild the validation oracle without calling the production view builder."""

    if not isinstance(plan, EgoTrajectoryPlan):
        raise TypeError("plan must be an EgoTrajectoryPlan")
    binding = _validate_source_binding(state, scenario)
    source_before = binding.source_state_sha256
    serialized_before = plan.serialize()
    plan.revalidate()
    validate_registered_ego_plan(scenario, plan)
    timestamps_micros = np.asarray(
        state.log_trajectory.timestamp_micros,
    )[
        binding.ego_slot,
        binding.current_index : binding.current_index + M6_WAYMAX_FRAME_COUNT,
    ].astype("<i8", copy=True)
    expected_plan_timestamps = np.asarray(
        scenario.timestamps[
            binding.current_index : (
                binding.current_index + M6_WAYMAX_FRAME_COUNT
            )
        ],
        dtype=np.float64,
    )
    if not np.array_equal(
        np.asarray(plan.timestamps[:M6_WAYMAX_FRAME_COUNT]),
        expected_plan_timestamps,
    ):
        _fail(
            "plan_time_binding",
            "canonical plan timestamps differ from source during validation",
        )
    expected = WaymaxEgoPlanView(
        scenario_id=scenario.scenario_id,
        source_state_sha256=source_before,
        ego_agent_id=int(scenario.ego.id),
        ego_slot=binding.ego_slot,
        current_index=binding.current_index,
        intervention_family=plan.spec.family,
        intervention_version=plan.spec.version,
        intervention_dose=plan.spec.dose,
        perturbation_identity=plan.perturbation_identity,
        canonical_plan_fingerprint=plan.plan_fingerprint,
        canonical_plan_audit_fingerprint=plan.audit_fingerprint,
        timestamps_micros=timestamps_micros,
        valid=np.asarray(plan.valid[:M6_WAYMAX_FRAME_COUNT], dtype=bool),
        applied=np.asarray(plan.applied[:M6_WAYMAX_FRAME_COUNT], dtype=bool),
        x=np.asarray(plan.x[:M6_WAYMAX_FRAME_COUNT], dtype="<f4"),
        y=np.asarray(plan.y[:M6_WAYMAX_FRAME_COUNT], dtype="<f4"),
        heading=np.asarray(plan.heading[:M6_WAYMAX_FRAME_COUNT], dtype="<f4"),
        vx=np.asarray(plan.vx[:M6_WAYMAX_FRAME_COUNT], dtype="<f4"),
        vy=np.asarray(plan.vy[:M6_WAYMAX_FRAME_COUNT], dtype="<f4"),
    )
    if source_state_mutation_sha256(state) != source_before:
        _fail("source_mutated", "validation-oracle construction mutated source state")
    plan.revalidate()
    if plan.serialize() != serialized_before:
        _fail(
            "canonical_plan_mutated",
            "validation-oracle construction mutated the canonical plan",
        )
    return expected


def _validate_canonical_plan_view(
    *,
    state: Any,
    scenario: Scenario,
    plan: EgoTrajectoryPlan,
    view: WaymaxEgoPlanView,
) -> WaymaxEgoPlanView:
    """Bind a caller view to an independently rebuilt registered-plan view."""

    if not isinstance(plan, EgoTrajectoryPlan):
        raise TypeError("plan must be an EgoTrajectoryPlan")
    if not isinstance(view, WaymaxEgoPlanView):
        raise TypeError("view must be a WaymaxEgoPlanView")
    plan.revalidate()
    validate_registered_ego_plan(scenario, plan)
    if not (
        (
            plan.spec.family == IDENTITY_FAMILY
            and plan.spec.version == M6_INTERVENTION_VERSION
            and plan.spec.dose == 0.0
        )
        or (
            plan.spec.family == LONGITUDINAL_BRAKE_PULSE_FAMILY
            and plan.spec.version == M6_INTERVENTION_VERSION
            and plan.spec.dose == PRIMARY_BRAKE_MAGNITUDE_MPS2
        )
    ):
        _fail(
            "plan_scope",
            "Waymax validation accepts only registered identity or primary b=2",
        )
    expected = _independently_rebuild_waymax_ego_plan_view(
        state,
        scenario,
        plan,
    )
    scalar_fields = (
        "scenario_id",
        "source_state_sha256",
        "ego_agent_id",
        "ego_slot",
        "current_index",
        "intervention_family",
        "intervention_version",
        "intervention_dose",
        "perturbation_identity",
        "canonical_plan_fingerprint",
        "canonical_plan_audit_fingerprint",
        "local_mutation_sha256",
    )
    if any(getattr(view, name) != getattr(expected, name) for name in scalar_fields):
        _fail(
            "plan_view_binding",
            "caller view identity/audit fields differ from canonical reconstruction",
        )
    for name in (
        "timestamps_micros",
        "valid",
        "applied",
        *_PLAN_FIELDS,
    ):
        if not np.array_equal(
            np.asarray(getattr(view, name)),
            np.asarray(getattr(expected, name)),
        ):
            _fail(
                "plan_view_binding",
                f"caller view {name} differs from canonical reconstruction",
            )
    if (
        view.canonical_bytes != expected.canonical_bytes
        or not np.array_equal(
            view.future_action_data,
            expected.future_action_data,
        )
    ):
        _fail(
            "plan_view_binding",
            "caller view bytes differ from canonical reconstruction",
        )
    view.revalidate()
    return expected


class CompactM6WaymaxRollout(NamedTuple):
    """Twenty post-current compact frames plus complete actor accounting masks."""

    x: Any
    y: Any
    yaw: Any
    vx: Any
    vy: Any
    valid: Any
    timestamp_micros: Any
    timestep: Any
    requested_control: Any
    effective_control: Any
    lifecycle_fallback: Any
    initialized_overlap_excluded: Any


def _environment_config(config: Any) -> Any:
    return config.EnvironmentConfig(
        init_steps=M4_INIT_STEPS,
        max_num_objects=M4_MAX_OBJECTS,
        controlled_object=config.ObjectType.SDC,
        compute_reward=False,
        allow_new_objects_after_warmup=True,
        metrics=config.MetricsConfig(metrics_to_run=()),
    )


def _runtime_requested_mask(state: Any, datatypes: Any) -> Any:
    current_valid = datatypes.dynamic_index(
        state.log_trajectory.valid,
        state.timestep,
        axis=-1,
        keepdims=False,
    )
    next_valid = datatypes.dynamic_index(
        state.log_trajectory.valid,
        state.timestep + 1,
        axis=-1,
        keepdims=False,
    )
    return (
        ~state.object_metadata.is_sdc
        & (state.object_metadata.object_types == 1)
        & current_valid
        & next_valid
    )


def _make_runtime_environment(bundle: str) -> tuple[Any, ...]:
    if bundle not in M6_WAYMAX_BUNDLES:
        raise ValueError(f"bundle must be one of {M6_WAYMAX_BUNDLES}")
    jax, jnp, agents, config, datatypes, runtime = _require_runtime()
    dynamics, env = runtime
    actors: tuple[Any, ...] = ()
    params: tuple[Any, ...] = ()
    if bundle == M6_WAYMAX_PRIVILEGED_IDM:
        assert_waymax_idm_defaults()

        def is_controlled(candidate: Any) -> Any:
            return _runtime_requested_mask(candidate, datatypes)

        actors = (agents.IDMRoutePolicy(is_controlled_func=is_controlled),)
        params = ({},)
    environment = env.PlanningAgentEnvironment(
        dynamics_model=dynamics.StateDynamics(),
        config=_environment_config(config),
        sim_agent_actors=actors,
        sim_agent_params=params,
    )
    return jax, jnp, datatypes, environment


def _compact_runtime_frame(
    state: Any,
    *,
    requested: Any,
    effective: Any,
    lifecycle: Any,
    overlap_excluded: Any,
) -> CompactM6WaymaxRollout:
    trajectory = state.current_sim_trajectory
    return CompactM6WaymaxRollout(
        x=trajectory.x[..., 0],
        y=trajectory.y[..., 0],
        yaw=trajectory.yaw[..., 0],
        vx=trajectory.vel_x[..., 0],
        vy=trajectory.vel_y[..., 0],
        valid=trajectory.valid[..., 0],
        timestamp_micros=trajectory.timestamp_micros[..., 0],
        timestep=state.timestep,
        requested_control=requested,
        effective_control=effective,
        lifecycle_fallback=lifecycle,
        initialized_overlap_excluded=overlap_excluded,
    )


def _m6_waymax_kernel(
    state: Any,
    future_action_data: Any,
    *,
    bundle: str,
) -> CompactM6WaymaxRollout:
    jax, jnp, datatypes, environment = _make_runtime_environment(bundle)
    from waymax.utils import geometry

    reset = environment.reset(state)
    boxes = reset.log_trajectory.stack_fields(
        ["x", "y", "length", "width", "yaw"]
    )[:, 0, :]
    initialized_overlap = jnp.any(
        geometry.compute_pairwise_overlaps(boxes),
        axis=-1,
    )
    non_sdc_vehicle = (
        ~reset.object_metadata.is_sdc
        & (reset.object_metadata.object_types == 1)
    )

    def step(
        carry: Any,
        prescribed: Any,
    ) -> tuple[Any, CompactM6WaymaxRollout]:
        if bundle == M6_WAYMAX_PRIVILEGED_IDM:
            requested = _runtime_requested_mask(carry, datatypes)
        else:
            requested = jnp.zeros_like(non_sdc_vehicle, dtype=jnp.bool_)
        effective = requested & ~initialized_overlap
        lifecycle = non_sdc_vehicle & ~requested
        overlap_excluded = requested & initialized_overlap
        action = datatypes.Action(
            data=prescribed,
            valid=jnp.ones((1,), dtype=jnp.bool_),
        )
        next_state = environment.step(carry, action)
        frame = _compact_runtime_frame(
            next_state,
            requested=requested,
            effective=effective,
            lifecycle=lifecycle,
            overlap_excluded=overlap_excluded,
        )
        return next_state, frame

    _, compact = jax.lax.scan(
        step,
        reset,
        xs=future_action_data,
        length=M6_WAYMAX_TRANSITIONS,
    )
    return compact


def single_scene_m6_logged_world_kernel(
    state: Any,
    future_action_data: Any,
) -> CompactM6WaymaxRollout:
    """Fixed-shape one-scene logged-world kernel suitable for ``jax.jit``."""

    return _m6_waymax_kernel(
        state,
        future_action_data,
        bundle=M6_WAYMAX_LOGGED_WORLD,
    )


def single_scene_m6_idm_kernel(
    state: Any,
    future_action_data: Any,
) -> CompactM6WaymaxRollout:
    """Fixed-shape one-scene privileged-IDM kernel suitable for ``jax.jit``."""

    return _m6_waymax_kernel(
        state,
        future_action_data,
        bundle=M6_WAYMAX_PRIVILEGED_IDM,
    )


def compact_m6_waymax_rollout(
    state: Any,
    scenario: Scenario,
    plan: EgoTrajectoryPlan,
    *,
    bundle: str,
) -> tuple[CompactM6WaymaxRollout, WaymaxEgoPlanView]:
    """Execute one sequential 20-transition bundle with mutation checks."""

    if bundle not in M6_WAYMAX_BUNDLES:
        raise ValueError(f"bundle must be one of {M6_WAYMAX_BUNDLES}")
    source_before = source_state_mutation_sha256(state)
    plan.revalidate()
    plan_before = plan.serialize()
    view = build_waymax_ego_plan_view(state, scenario, plan)
    _, jnp, _, _, _, _ = _require_runtime()
    action_data = jnp.asarray(view.future_action_data)
    if bundle == M6_WAYMAX_LOGGED_WORLD:
        compact = single_scene_m6_logged_world_kernel(state, action_data)
    else:
        compact = single_scene_m6_idm_kernel(state, action_data)
    if source_state_mutation_sha256(state) != source_before:
        _fail("source_mutated", "Waymax execution mutated the source state")
    plan.revalidate()
    if plan.serialize() != plan_before:
        _fail("canonical_plan_mutated", "Waymax execution mutated float64 plan")
    view.revalidate()
    return compact, view


def compact_selected_m6_waymax_rollout(
    state: Any,
    scenario: Scenario,
    plan: EgoTrajectoryPlan,
    *,
    bundle: str,
    selection: M6WaymaxSelection,
    primary_domain: M6WaymaxPrimaryDomain,
    selection_position: int,
) -> tuple[CompactM6WaymaxRollout, WaymaxEgoPlanView]:
    """Guard the live kernel behind the exact canonical selected member.

    This is the official/live entry point.  All source-only selection checks run
    before delegating to :func:`compact_m6_waymax_rollout`, so unsupported
    selections and member/position mismatches execute zero Waymax kernels.
    """

    if not isinstance(selection, M6WaymaxSelection):
        raise TypeError("selection must be an M6WaymaxSelection")
    if not isinstance(primary_domain, M6WaymaxPrimaryDomain):
        raise TypeError("primary_domain must be an M6WaymaxPrimaryDomain")
    selection.revalidate(primary_domain=primary_domain)
    if not selection.supported:
        _fail(
            "selection_unsupported",
            "unsupported Waymax selection cannot execute a live kernel",
        )
    position = _strict_int(selection_position, "selection_position")
    if position >= len(selection.members):
        _fail(
            "selection_position",
            "selection_position lies outside the canonical selected subset",
        )
    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    member = selection.members[position]
    entry = primary_domain.entry_by_cohort_index.get(member.cohort_index)
    if entry is None:
        _fail("selection_member", "selected member is absent from primary domain")
    try:
        recomputed = _validate_primary_entry_against_scenario(
            entry,
            scenario,
            source_state_sha256=source_state_mutation_sha256(state),
        )
    except M6WaymaxError:
        _fail(
            "selection_member",
            "live scenario is not the exact member at selection_position",
        )
    if (
        member.primary_entry_sha256 != entry.entry_sha256
        or member.source_binding_sha256 != entry.source_state_sha256
        or member.scenario_id != scenario.scenario_id
        or member.target_index != recomputed.target_index
        or member.target_agent_id != entry.target_contract_id
    ):
        _fail(
            "selection_member",
            "live scenario is not the exact member at selection_position",
        )
    return compact_m6_waymax_rollout(
        state,
        scenario,
        plan,
        bundle=bundle,
    )


def tiny_m6_waymax_api_oracle(
    state: Any,
    view: WaymaxEgoPlanView,
    *,
    bundle: str,
    num_steps: int,
) -> CompactM6WaymaxRollout:
    """Independent stock-rollout oracle for tests only (one to three steps).

    This path intentionally does not call the production environment factory,
    control-mask helper, scan kernel, or compact-frame packer.  It constructs a
    prescribed-SDC actor, uses Waymax's stock :func:`env.rollout`, derives masks
    directly from raw states, and packs fields independently.
    """

    steps = _strict_int(num_steps, "num_steps", minimum=1)
    if steps > 3:
        raise ValueError("the stock/API oracle is intentionally limited to 3 steps")
    binding_hash = source_state_mutation_sha256(state)
    if binding_hash != view.source_state_sha256:
        _fail("source_binding", "plan view does not bind to the supplied source state")
    view.revalidate()
    if bundle not in M6_WAYMAX_BUNDLES:
        raise ValueError(f"bundle must be one of {M6_WAYMAX_BUNDLES}")
    try:
        import jax
        import jax.numpy as jnp
        from waymax import agents, config, datatypes, dynamics, env
    except ImportError as exc:
        raise M6WaymaxDependencyError(
            "the independent Waymax API oracle requires the optional runtime"
        ) from exc
    from waymax.utils import geometry

    environment_config = config.EnvironmentConfig(
        init_steps=11,
        max_num_objects=128,
        controlled_object=config.ObjectType.SDC,
        compute_reward=False,
        allow_new_objects_after_warmup=True,
        metrics=config.MetricsConfig(metrics_to_run=()),
    )

    def raw_requested(candidate: Any) -> Any:
        current_valid = datatypes.dynamic_index(
            candidate.log_trajectory.valid,
            candidate.timestep,
            axis=-1,
            keepdims=False,
        )
        next_valid = datatypes.dynamic_index(
            candidate.log_trajectory.valid,
            candidate.timestep + 1,
            axis=-1,
            keepdims=False,
        )
        return (
            ~candidate.object_metadata.is_sdc
            & (candidate.object_metadata.object_types == 1)
            & current_valid
            & next_valid
        )

    sim_actors: tuple[Any, ...] = ()
    sim_params: tuple[Any, ...] = ()
    if bundle == M6_WAYMAX_PRIVILEGED_IDM:
        sim_actors = (agents.IDMRoutePolicy(is_controlled_func=raw_requested),)
        sim_params = ({},)
    environment = env.PlanningAgentEnvironment(
        dynamics_model=dynamics.StateDynamics(),
        config=environment_config,
        sim_agent_actors=sim_actors,
        sim_agent_params=sim_params,
    )

    def actor_init(rng: Any, reset_state: Any) -> tuple[()]:
        del rng, reset_state
        return ()

    def actor_select(
        params: Any,
        candidate: Any,
        actor_state: tuple[()],
        rng: Any,
    ) -> Any:
        del rng
        offset = candidate.timestep - 10
        action = datatypes.Action(
            data=params[offset],
            valid=jnp.ones((1,), dtype=jnp.bool_),
        )
        return agents.WaymaxActorOutput(
            actor_state=actor_state,
            action=action,
            is_controlled=jnp.asarray(True, dtype=jnp.bool_),
        )

    prescribed_actor = agents.actor_core_factory(
        init=actor_init,
        select_action=actor_select,
        name="m6_test_only_prescribed_sdc",
    )
    stock = env.rollout(
        state,
        prescribed_actor,
        environment,
        rng=jax.random.PRNGKey(0),
        rollout_num_steps=steps,
        actor_params=jnp.asarray(view.future_action_data),
    )
    reset_state = jax.tree.map(lambda leaf: leaf[0], stock.state)
    boxes = reset_state.log_trajectory.stack_fields(
        ["x", "y", "length", "width", "yaw"]
    )[:, 0, :]
    initialized_overlap = jnp.any(
        geometry.compute_pairwise_overlaps(boxes),
        axis=-1,
    )
    non_sdc_vehicle = ~reset_state.object_metadata.is_sdc & (
        reset_state.object_metadata.object_types == 1
    )
    packed: dict[str, list[Any]] = {
        name: [] for name in CompactM6WaymaxRollout._fields
    }
    for offset in range(steps):
        before = jax.tree.map(lambda leaf: leaf[offset], stock.state)
        after = jax.tree.map(lambda leaf: leaf[offset + 1], stock.state)
        if bundle == M6_WAYMAX_PRIVILEGED_IDM:
            requested = raw_requested(before)
        else:
            requested = jnp.zeros_like(non_sdc_vehicle, dtype=jnp.bool_)
        effective = requested & ~initialized_overlap
        lifecycle = non_sdc_vehicle & ~requested
        overlap_excluded = requested & initialized_overlap
        current = after.current_sim_trajectory
        packed["x"].append(current.x[..., 0])
        packed["y"].append(current.y[..., 0])
        packed["yaw"].append(current.yaw[..., 0])
        packed["vx"].append(current.vel_x[..., 0])
        packed["vy"].append(current.vel_y[..., 0])
        packed["valid"].append(current.valid[..., 0])
        packed["timestamp_micros"].append(current.timestamp_micros[..., 0])
        packed["timestep"].append(after.timestep)
        packed["requested_control"].append(requested)
        packed["effective_control"].append(effective)
        packed["lifecycle_fallback"].append(lifecycle)
        packed["initialized_overlap_excluded"].append(overlap_excluded)
    return CompactM6WaymaxRollout(
        *(
            jnp.stack(packed[name], axis=0)
            for name in CompactM6WaymaxRollout._fields
        )
    )


@dataclass(frozen=True, slots=True)
class M6WaymaxComponentComparison:
    """Full evidence for one numeric or exact-binary comparison component."""

    denominator: int
    maximum_absolute_error: float
    tolerance_failure_count: int
    binary_mismatch_count: int

    def __post_init__(self) -> None:
        for name in (
            "denominator",
            "tolerance_failure_count",
            "binary_mismatch_count",
        ):
            object.__setattr__(
                self,
                name,
                _strict_int(getattr(self, name), name),
            )
        maximum = float(self.maximum_absolute_error)
        if not math.isfinite(maximum) or maximum < 0.0:
            raise ValueError("maximum_absolute_error must be finite and non-negative")
        object.__setattr__(self, "maximum_absolute_error", maximum)
        if (
            self.tolerance_failure_count > self.denominator
            or self.binary_mismatch_count > self.denominator
        ):
            raise ValueError("comparison failure counts exceed denominator")

    @property
    def passed(self) -> bool:
        return (
            self.tolerance_failure_count == 0
            and self.binary_mismatch_count == 0
        )


@dataclass(frozen=True, slots=True)
class M6WaymaxValidation:
    """Immutable complete component diagnostics; never an aggregate-only result."""

    components: Mapping[str, M6WaymaxComponentComparison]

    def __post_init__(self) -> None:
        if not isinstance(self.components, Mapping) or not self.components:
            raise ValueError("components must be a non-empty mapping")
        copied: dict[str, M6WaymaxComponentComparison] = {}
        for name, result in self.components.items():
            if not isinstance(name, str) or not name:
                raise ValueError("component names must be non-empty strings")
            if not isinstance(result, M6WaymaxComponentComparison):
                raise TypeError("component values must be comparisons")
            copied[name] = result
        object.__setattr__(self, "components", MappingProxyType(copied))

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.components.values())

    @property
    def failure_count(self) -> int:
        return sum(
            result.tolerance_failure_count + result.binary_mismatch_count
            for result in self.components.values()
        )

    def require_passed(self) -> None:
        if not self.passed:
            failed = {
                name: {
                    "denominator": result.denominator,
                    "maximum_absolute_error": result.maximum_absolute_error,
                    "tolerance_failure_count": result.tolerance_failure_count,
                    "binary_mismatch_count": result.binary_mismatch_count,
                }
                for name, result in self.components.items()
                if not result.passed
            }
            raise M6WaymaxError(
                "validation_failed",
                f"component diagnostics contain failures: {failed}",
            )


def _binary_comparison(actual: Any, expected: Any) -> M6WaymaxComponentComparison:
    left = np.asarray(actual)
    right = np.asarray(expected)
    if left.shape != right.shape:
        _fail("comparison_shape", "binary comparison shapes differ")
    mismatch = left != right
    count = int(np.count_nonzero(mismatch))
    maximum = 0.0
    if count:
        if np.issubdtype(left.dtype, np.number) and np.issubdtype(
            right.dtype,
            np.number,
        ):
            difference = np.abs(
                left.astype(np.float64) - right.astype(np.float64)
            )
            finite = difference[np.isfinite(difference)]
            maximum = (
                float(np.max(finite))
                if finite.size
                else float(np.finfo(np.float64).max)
            )
        else:
            maximum = 1.0
    return M6WaymaxComponentComparison(
        denominator=int(left.size),
        maximum_absolute_error=maximum,
        tolerance_failure_count=0,
        binary_mismatch_count=count,
    )


def _numeric_comparison(
    actual: Any,
    expected: Any,
    *,
    mask: Any,
    atol: float,
    rtol: float,
    circular: bool = False,
) -> M6WaymaxComponentComparison:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if left.shape != right.shape or left.shape != selected.shape:
        _fail("comparison_shape", "numeric comparison shapes differ")
    denominator = int(np.count_nonzero(selected))
    if denominator == 0:
        return M6WaymaxComponentComparison(0, 0.0, 0, 0)
    finite = np.isfinite(left) & np.isfinite(right)
    if circular:
        error = np.abs((left - right + np.pi) % (2.0 * np.pi) - np.pi)
        tolerance = np.full_like(error, atol)
    else:
        error = np.abs(left - right)
        tolerance = atol + rtol * np.abs(right)
    selected_error = np.where(finite, error, np.inf)[selected]
    failures = int(np.count_nonzero(selected_error > tolerance[selected]))
    finite_errors = selected_error[np.isfinite(selected_error)]
    maximum = float(np.max(finite_errors)) if finite_errors.size else 0.0
    if np.any(~np.isfinite(selected_error)):
        maximum = float(np.finfo(np.float64).max)
    return M6WaymaxComponentComparison(
        denominator=denominator,
        maximum_absolute_error=maximum,
        tolerance_failure_count=failures,
        binary_mismatch_count=0,
    )


def _compact_numpy(compact: CompactM6WaymaxRollout) -> dict[str, np.ndarray]:
    if not isinstance(compact, CompactM6WaymaxRollout):
        raise TypeError("compact must be a CompactM6WaymaxRollout")
    return {
        name: np.asarray(getattr(compact, name))
        for name in CompactM6WaymaxRollout._fields
    }


def _reject_nonfinite_compact_float_domain(
    arrays: Mapping[str, np.ndarray],
    *,
    code: str,
) -> None:
    """Reject non-finite values over every float field and every object slot."""

    for name in _FLOAT_FIELDS:
        values = np.asarray(arrays[name])
        if values.shape != (M6_WAYMAX_TRANSITIONS, M4_MAX_OBJECTS):
            _fail("compact_shape", f"{name} must have shape [20, 128]")
        if not bool(np.all(np.isfinite(values))):
            _fail(
                code,
                f"{name} contains a non-finite value in the full compact domain",
            )


def _validate_compact_schema(arrays: Mapping[str, np.ndarray]) -> None:
    object_shape = (M6_WAYMAX_TRANSITIONS, M4_MAX_OBJECTS)
    for name in (*_FLOAT_FIELDS, "valid", "timestamp_micros", *_MASK_FIELDS):
        if arrays[name].shape != object_shape:
            _fail("compact_shape", f"{name} must have shape {object_shape}")
    if arrays["timestep"].shape != (M6_WAYMAX_TRANSITIONS,):
        _fail("compact_shape", "timestep must have shape [20]")
    for name in _FLOAT_FIELDS:
        if arrays[name].dtype != np.float32:
            _fail("compact_dtype", f"{name} must be float32")
    # This check deliberately precedes validity, actor, lifecycle, and fallback
    # masks. Invalid padding is part of the structural float domain and cannot hide
    # NaN or infinity.
    _reject_nonfinite_compact_float_domain(
        arrays,
        code="compact_nonfinite",
    )
    for name in ("valid", *_MASK_FIELDS):
        if arrays[name].dtype != np.bool_:
            _fail("compact_dtype", f"{name} must be boolean")
    for name in ("timestamp_micros", "timestep"):
        if (
            np.issubdtype(arrays[name].dtype, np.bool_)
            or not np.issubdtype(arrays[name].dtype, np.integer)
        ):
            _fail("compact_dtype", f"{name} must be integer")


def _validate_frozen_qualification(
    qualification: M6WaymaxEligibility,
    *,
    state: Any,
    scenario: Scenario,
    binding: _SourceBinding,
    primary_domain: M6WaymaxPrimaryDomain,
) -> int:
    if not isinstance(qualification, M6WaymaxEligibility):
        raise TypeError("qualification must be an M6WaymaxEligibility")
    qualification.revalidate()
    if not isinstance(primary_domain, M6WaymaxPrimaryDomain):
        raise TypeError("primary_domain must be an M6WaymaxPrimaryDomain")
    primary_domain.revalidate()
    if not qualification.eligible:
        _fail("target_scope", "an ineligible Waymax member cannot be executed")
    if (
        qualification.scenario_id != scenario.scenario_id
        or qualification.source_binding_sha256 != binding.source_state_sha256
        or qualification.rank_sha256
        != m6_waymax_rank_sha256(qualification.cohort_index)
    ):
        _fail("target_scope", "qualification belongs to a different source state")
    entries_by_index = primary_domain.entry_by_cohort_index
    if qualification.cohort_index not in entries_by_index:
        _fail(
            "target_scope",
            "qualification cohort/source does not belong to the primary domain",
        )
    primary_entry = entries_by_index[qualification.cohort_index]
    recomputed = _validate_primary_entry_against_scenario(
        primary_entry,
        scenario,
        source_state_sha256=binding.source_state_sha256,
    )
    if (
        qualification.primary_entry_sha256 != primary_entry.entry_sha256
        or qualification.scenario_id != primary_entry.scenario_id
        or qualification.target_index != recomputed.target_index
        or qualification.target_agent_id != primary_entry.target_contract_id
    ):
        _fail(
            "target_scope",
            "qualification differs from its complete recomputed primary entry",
        )
    if (
        qualification.target_slot is None
        or qualification.target_index is None
        or qualification.target_agent_id is None
    ):
        _fail("target_scope", "eligible qualification lacks a frozen target")
    slot = qualification.target_slot
    if (
        slot >= M4_MAX_OBJECTS
        or qualification.target_index >= scenario.num_agents
        or int(binding.retained_slots[qualification.target_index]) != slot
        or int(scenario.agents[qualification.target_index].id)
        != qualification.target_agent_id
        or int(np.asarray(state.object_metadata.ids)[slot])
        != qualification.target_agent_id
        or bool(np.asarray(state.object_metadata.is_sdc)[slot])
    ):
        _fail(
            "target_scope",
            "qualification target index, identity, and source slot are inconsistent",
        )
    return slot


def validate_m6_waymax_compact(
    compact: CompactM6WaymaxRollout,
    *,
    state: Any,
    scenario: Scenario,
    plan: EgoTrajectoryPlan,
    view: WaymaxEgoPlanView,
    bundle: str,
    qualification: M6WaymaxEligibility,
    primary_domain: M6WaymaxPrimaryDomain,
) -> M6WaymaxValidation:
    """Validate one compact output with complete per-component diagnostics."""

    binding = _validate_source_binding(state, scenario)
    canonical_view = _validate_canonical_plan_view(
        state=state,
        scenario=scenario,
        plan=plan,
        view=view,
    )
    if (
        canonical_view.source_state_sha256 != binding.source_state_sha256
        or canonical_view.scenario_id != scenario.scenario_id
        or canonical_view.ego_slot != binding.ego_slot
        or canonical_view.ego_agent_id != int(scenario.ego.id)
        or canonical_view.current_index != binding.current_index
    ):
        _fail("plan_source_binding", "Waymax plan view differs from source binding")
    target = _validate_frozen_qualification(
        qualification,
        state=state,
        scenario=scenario,
        binding=binding,
        primary_domain=primary_domain,
    )
    arrays = _compact_numpy(compact)
    _validate_compact_schema(arrays)
    components: dict[str, M6WaymaxComponentComparison] = {}
    expected_steps = np.arange(
        binding.current_index + 1,
        binding.current_index + 1 + M6_WAYMAX_TRANSITIONS,
        dtype=arrays["timestep"].dtype,
    )
    components["identity.timestep"] = _binary_comparison(
        arrays["timestep"],
        expected_steps,
    )
    interval = slice(
        binding.current_index + 1,
        binding.current_index + 1 + M6_WAYMAX_TRANSITIONS,
    )
    logged = state.log_trajectory
    expected_valid = np.asarray(logged.valid, dtype=bool)[:, interval].T
    expected_time = np.asarray(logged.timestamp_micros)[:, interval].T
    components["lifecycle.valid"] = _binary_comparison(
        arrays["valid"],
        expected_valid,
    )
    components["identity.timestamp_micros"] = _binary_comparison(
        arrays["timestamp_micros"],
        expected_time,
    )
    expected_masks = _source_control_masks(
        state,
        current_index=binding.current_index,
        bundle=bundle,
    )
    for name in _MASK_FIELDS:
        components[f"actor.{name}"] = _binary_comparison(
            arrays[name],
            expected_masks[name],
        )

    ego_mask = np.ones(M6_WAYMAX_TRANSITIONS, dtype=bool)
    plan_expected = {
        "x": canonical_view.x[1:],
        "y": canonical_view.y[1:],
        "yaw": canonical_view.heading[1:],
        "vx": canonical_view.vx[1:],
        "vy": canonical_view.vy[1:],
    }
    for name, expected in plan_expected.items():
        components[f"ego_plan.{name}"] = _numeric_comparison(
            arrays[name][:, binding.ego_slot],
            expected,
            mask=ego_mask,
            atol=M6_WAYMAX_YAW_ATOL if name == "yaw" else M6_WAYMAX_FLOAT_ATOL,
            rtol=0.0 if name == "yaw" else M6_WAYMAX_FLOAT_RTOL,
            circular=name == "yaw",
        )

    source_fields = {
        "x": "x",
        "y": "y",
        "yaw": "yaw",
        "vx": "vel_x",
        "vy": "vel_y",
    }
    world_slots = np.ones(M4_MAX_OBJECTS, dtype=bool)
    world_slots[binding.ego_slot] = False
    if bundle == M6_WAYMAX_LOGGED_WORLD:
        comparison_mask = expected_valid & world_slots[np.newaxis, :]
    else:
        comparison_mask = (
            ~expected_masks["effective_control"]
            & world_slots[np.newaxis, :]
            & expected_valid
        )
    for name, source_name in source_fields.items():
        expected = np.asarray(getattr(logged, source_name))[:, interval].T
        components[f"logged_fallback.{name}"] = _numeric_comparison(
            arrays[name],
            expected,
            mask=comparison_mask,
            atol=M6_WAYMAX_YAW_ATOL if name == "yaw" else M6_WAYMAX_FLOAT_ATOL,
            rtol=0.0 if name == "yaw" else M6_WAYMAX_FLOAT_RTOL,
            circular=name == "yaw",
        )

    all_true = np.ones(M6_WAYMAX_TRANSITIONS, dtype=bool)
    all_false = np.zeros(M6_WAYMAX_TRANSITIONS, dtype=bool)
    if bundle == M6_WAYMAX_PRIVILEGED_IDM:
        components["target.requested_scope"] = _binary_comparison(
            arrays["requested_control"][:, target],
            all_true,
        )
        components["target.effective_scope"] = _binary_comparison(
            arrays["effective_control"][:, target],
            all_true,
        )
        components["target.lifecycle_fallback"] = _binary_comparison(
            arrays["lifecycle_fallback"][:, target],
            all_false,
        )
        components["target.overlap_excluded"] = _binary_comparison(
            arrays["initialized_overlap_excluded"][:, target],
            all_false,
        )
    return M6WaymaxValidation(components)


def validate_m6_waymax_pair(
    baseline: CompactM6WaymaxRollout,
    treatment: CompactM6WaymaxRollout,
    *,
    state: Any,
    scenario: Scenario,
    baseline_plan: EgoTrajectoryPlan,
    treatment_plan: EgoTrajectoryPlan,
    baseline_view: WaymaxEgoPlanView,
    treatment_view: WaymaxEgoPlanView,
    bundle: str,
    qualification: M6WaymaxEligibility,
    primary_domain: M6WaymaxPrimaryDomain,
) -> M6WaymaxValidation:
    """Enforce pair identity, logged-world nonresponse, and the t+2 floor."""

    baseline_validation = validate_m6_waymax_compact(
        baseline,
        state=state,
        scenario=scenario,
        plan=baseline_plan,
        view=baseline_view,
        bundle=bundle,
        qualification=qualification,
        primary_domain=primary_domain,
    )
    treatment_validation = validate_m6_waymax_compact(
        treatment,
        state=state,
        scenario=scenario,
        plan=treatment_plan,
        view=treatment_view,
        bundle=bundle,
        qualification=qualification,
        primary_domain=primary_domain,
    )
    left = _compact_numpy(baseline)
    right = _compact_numpy(treatment)
    binding = _validate_source_binding(state, scenario)
    baseline_plan.revalidate()
    treatment_plan.revalidate()
    validate_registered_ego_plan(scenario, baseline_plan)
    validate_registered_ego_plan(scenario, treatment_plan)
    if not (
        baseline_plan.spec.family == IDENTITY_FAMILY
        and baseline_plan.spec.version == M6_INTERVENTION_VERSION
        and baseline_plan.spec.dose == 0.0
    ):
        _fail("pair_condition", "baseline view must be the registered identity plan")
    if not (
        treatment_plan.spec.family == LONGITUDINAL_BRAKE_PULSE_FAMILY
        and treatment_plan.spec.version == M6_INTERVENTION_VERSION
        and treatment_plan.spec.dose == PRIMARY_BRAKE_MAGNITUDE_MPS2
    ):
        _fail(
            "pair_condition",
            "treatment view must be the registered primary b=2 plan",
        )
    components = {
        f"baseline.{name}": result
        for name, result in baseline_validation.components.items()
    }
    components.update(
        {
            f"treatment.{name}": result
            for name, result in treatment_validation.components.items()
        }
    )
    for name in ("valid", "timestamp_micros", "timestep", *_MASK_FIELDS):
        components[f"pair.{name}"] = _binary_comparison(left[name], right[name])
    world = np.ones_like(left["valid"], dtype=bool)
    world[:, binding.ego_slot] = False
    if bundle == M6_WAYMAX_LOGGED_WORLD:
        pair_mask = world & left["valid"] & right["valid"]
        prefix = "logged_world_no_response"
    else:
        pair_mask = np.zeros_like(world)
        pair_mask[0, :] = world[0, :] & left["valid"][0] & right["valid"][0]
        prefix = "synchronous_t_plus_2_floor"
    for name in _FLOAT_FIELDS:
        components[f"{prefix}.{name}"] = _numeric_comparison(
            left[name],
            right[name],
            mask=pair_mask,
            atol=0.0,
            rtol=0.0,
            circular=name == "yaw",
        )
        if bundle == M6_WAYMAX_LOGGED_WORLD:
            components[f"{prefix}_exact.{name}"] = _binary_comparison(
                left[name][world],
                right[name][world],
            )
        else:
            first_world = world[0]
            components[f"{prefix}_exact.{name}"] = _binary_comparison(
                left[name][0, first_world],
                right[name][0, first_world],
            )
    return M6WaymaxValidation(components)


def m6_waymax_to_rollout(
    compact: CompactM6WaymaxRollout,
    *,
    state: Any,
    scenario: Scenario,
    plan: EgoTrajectoryPlan,
    view: WaymaxEgoPlanView,
    bundle: str,
    qualification: M6WaymaxEligibility,
    primary_domain: M6WaymaxPrimaryDomain,
    seed: int = 0,
) -> tuple[Rollout, M6WaymaxValidation]:
    """Validate and reconstruct one source-neutral 31-frame M6 Waymax rollout."""

    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) != 0
    ):
        raise ValueError("the frozen M6 Waymax seed must equal 0")
    arrays = _compact_numpy(compact)
    # Reconstruction owns an independent full-domain finite gate; it executes
    # before and does not rely on the general compact validator.
    _reject_nonfinite_compact_float_domain(
        arrays,
        code="rollout_nonfinite",
    )
    validation = validate_m6_waymax_compact(
        compact,
        state=state,
        scenario=scenario,
        plan=plan,
        view=view,
        bundle=bundle,
        qualification=qualification,
        primary_domain=primary_domain,
    )
    validation.require_passed()
    binding = _validate_source_binding(state, scenario)
    target_slot = _validate_frozen_qualification(
        qualification,
        state=state,
        scenario=scenario,
        binding=binding,
        primary_domain=primary_domain,
    )
    output_frames = binding.current_index + 1 + M6_WAYMAX_TRANSITIONS
    agents: list[Agent] = []
    field_map = {
        "x": "x",
        "y": "y",
        "heading": "yaw",
        "vx": "vx",
        "vy": "vy",
    }
    for scenario_index, slot in enumerate(binding.retained_slots):
        source_agent = scenario.agents[scenario_index]
        valid = np.concatenate(
            (
                np.asarray(
                    source_agent.valid[: binding.current_index + 1],
                    dtype=bool,
                ),
                np.asarray(arrays["valid"][:, slot], dtype=bool),
            )
        )
        values: dict[str, np.ndarray] = {}
        for target_name, compact_name in field_map.items():
            future = np.asarray(arrays[compact_name][:, slot], dtype=np.float64)
            if compact_name == "yaw":
                future = (future + np.pi) % (2.0 * np.pi) - np.pi
            future = np.where(valid[binding.current_index + 1 :], future, 0.0)
            values[target_name] = np.concatenate(
                (
                    np.asarray(
                        getattr(source_agent, target_name)[
                            : binding.current_index + 1
                        ],
                        dtype=np.float64,
                    ),
                    future,
                )
            )
        agents.append(
            Agent(
                id=source_agent.id,
                type=source_agent.type,
                valid=valid,
                length=source_agent.length,
                width=source_agent.width,
                **values,
            )
        )
    rollout = Rollout(
        scenario_id=scenario.scenario_id,
        sim_name=bundle,
        sim_version=M6_WAYMAX_VERSION,
        seed=int(seed),
        timestamps=np.asarray(scenario.timestamps[:output_frames], dtype=np.float64),
        agents=agents,
        perturbation=view.perturbation_identity,
        metadata={
            "backend": "waymax",
            "backend_commit": WAYMAX_COMMIT,
            "ego_control": "typed_ego_plan",
            "horizon_transitions": M6_WAYMAX_TRANSITIONS,
            "policy_access_role": (
                "privileged_logged_trajectory_waypoint_following"
                if bundle == M6_WAYMAX_PRIVILEGED_IDM
                else "logged_world_fallback"
            ),
            "target_agent_id": int(
                np.asarray(state.object_metadata.ids)[target_slot]
            ),
            "control_accounting": {
                name: int(np.count_nonzero(arrays[name]))
                for name in _MASK_FIELDS
            },
        },
    )
    return rollout, validation


def m6_waymax_runtime_config() -> Mapping[str, Any]:
    """Return a recursively immutable pinned execution description.

    Tuples and mapping proxies are intentional; this object is not directly
    JSON-native.  A result-store boundary must create its own explicit plain-data
    copy before serialization.
    """

    return MappingProxyType(
        {
            "adapter_version": M6_WAYMAX_VERSION,
            "bundles": tuple(M6_WAYMAX_BUNDLES),
            "cadence_micros": M6_WAYMAX_CADENCE_MICROS,
            "float_atol": M6_WAYMAX_FLOAT_ATOL,
            "float_rtol": M6_WAYMAX_FLOAT_RTOL,
            "frame_count": M6_WAYMAX_FRAME_COUNT,
            "idm_defaults": MappingProxyType(dict(WAYMAX_IDM_DEFAULTS)),
            "rank_domain": M6_WAYMAX_RANK_DOMAIN,
            "qualification_ledger_domain": (
                M6_WAYMAX_QUALIFICATION_LEDGER_DOMAIN.decode("ascii")
            ),
            "selection_domain": M6_WAYMAX_SELECTION_DOMAIN.decode("ascii"),
            "live_entry_point": "compact_selected_m6_waymax_rollout",
            "transitions": M6_WAYMAX_TRANSITIONS,
            "waymax_commit": WAYMAX_COMMIT,
            "yaw_atol": M6_WAYMAX_YAW_ATOL,
        }
    )


__all__ = [
    "M6_WAYMAX_BUNDLES",
    "M6_WAYMAX_CADENCE_MICROS",
    "M6_WAYMAX_FLOAT_ATOL",
    "M6_WAYMAX_FLOAT_RTOL",
    "M6_WAYMAX_FRAME_COUNT",
    "M6_WAYMAX_LOGGED_WORLD",
    "M6_WAYMAX_MAX_SCENES",
    "M6_WAYMAX_MIN_SCENES",
    "M6_WAYMAX_PRIVILEGED_IDM",
    "M6_WAYMAX_RANK_DOMAIN",
    "M6_WAYMAX_TRANSITIONS",
    "M6_WAYMAX_VERSION",
    "M6_WAYMAX_YAW_ATOL",
    "CompactM6WaymaxRollout",
    "M6WaymaxComponentComparison",
    "M6WaymaxDependencyError",
    "M6WaymaxEligibility",
    "M6WaymaxError",
    "M6WaymaxPrimaryDomain",
    "M6WaymaxPrimaryDomainEntry",
    "M6WaymaxQualificationLedger",
    "M6WaymaxSelection",
    "M6WaymaxValidation",
    "WaymaxEgoPlanView",
    "build_m6_waymax_primary_domain_entry",
    "build_m6_waymax_qualification_ledger",
    "build_waymax_ego_plan_view",
    "compact_m6_waymax_rollout",
    "compact_selected_m6_waymax_rollout",
    "evaluate_m6_waymax_eligibility",
    "m6_waymax_rank_sha256",
    "m6_waymax_runtime_config",
    "m6_waymax_to_rollout",
    "select_m6_waymax_subset",
    "single_scene_m6_idm_kernel",
    "single_scene_m6_logged_world_kernel",
    "source_state_mutation_sha256",
    "tiny_m6_waymax_api_oracle",
    "validate_m6_waymax_compact",
    "validate_m6_waymax_pair",
]
