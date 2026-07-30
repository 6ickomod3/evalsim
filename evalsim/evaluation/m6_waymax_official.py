"""Runtime-injected official M6 Waymax source and evidence boundary.

The module is import-safe without JAX, TensorFlow, or Waymax.  Accepted-M4 reloads
enter one at a time through :class:`M6WaymaxOfficialCollector`; only the canonical
selected native states remain resident.  Execution is supplied by three callables so
the official command, rather than this pure seam, owns optional-runtime activation.

No public row issuer accepts mappings, digests, native identifiers, or source states.
Live scalar, field-comparison, and repeat/JIT evidence is derived from typed adapter
outputs and bound back to the complete primary-domain and qualification ledgers.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
import hashlib
import dataclasses
import json
import marshal
import resource
import math
import threading
import sys
import time
from types import MappingProxyType
import importlib.metadata
from typing import Any, Literal, Protocol

import numpy as np

from evalsim.contracts import EgoTrajectoryPlan, Scenario
from evalsim.contracts.counterfactual import ScenarioSnapshot
from evalsim.evaluation.m6 import (
    M6EligibilityLedger,
    M6EvaluationResult,
    M6PairedSceneResult,
)
from evalsim.evaluation.m6_official import (
    M6OfficialNumpyRows,
    m6_eligibility_rows,
)
from evalsim.metrics.m6 import (
    M6_PAIRED_METRIC_VERSION,
    M6_RESPONSE_ACCELERATION_THRESHOLD_MPS2,
    M6_RESPONSE_PERSISTENCE_S,
)
from evalsim.evaluation.m6_waymax_metrics import (
    M6WaymaxIssuedScalarTable,
    M6WaymaxLiveDeterminismExecution,
    M6WaymaxLiveDeterminismTable,
    M6WaymaxNoExecutionDeterminismTable,
    build_m6_waymax_live_determinism_table,
    build_m6_waymax_scene_scalar_table,
    build_m6_waymax_twenty_transition_pair_view,
    compute_m6_waymax_paired_measures,
    build_m6_waymax_unsupported_determinism_table,
)
from evalsim.perturb.m6 import (
    PRIMARY_BRAKE_MAGNITUDE_MPS2,
    compile_identity_plan,
    compile_longitudinal_brake_pulse_plan,
    evaluate_primary_brake_eligibility,
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
    CompactM6WaymaxRollout,
    M6WaymaxEligibility,
    M6WaymaxPrimaryDomain,
    build_waymax_ego_plan_view,
    compact_selected_m6_waymax_rollout,
    M6WaymaxQualificationLedger,
    M6WaymaxSelection,
    WaymaxEgoPlanView,
    single_scene_m6_idm_kernel,
    single_scene_m6_logged_world_kernel,
    build_m6_waymax_primary_domain_entry,
    build_m6_waymax_qualification_ledger,
    evaluate_m6_waymax_eligibility,
    m6_waymax_to_rollout,
    select_m6_waymax_subset,
    source_state_mutation_sha256,
    validate_m6_waymax_compact,
)
from evalsim.sources.m5_m4_reuse import (
    AcceptedM4Cohort,
    ReloadedM4Member,
    reverify_accepted_m4_run,
)
from evalsim.sources.waymax import WAYMAX_COMMIT


M6_WAYMAX_OFFICIAL_POPULATION_SIZE = 128
M6_WAYMAX_OFFICIAL_CONDITIONS = ("identity", "primary_brake")
M6_WAYMAX_OFFICIAL_COMPARISON_FIELDS = (
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
M6_WAYMAX_OFFICIAL_EXACT_FIELDS = frozenset(
    {
        "agent_identity",
        "timestamps",
        "validity",
        "actor_mask",
        "lifecycle_category",
    }
)
M6_WAYMAX_OFFICIAL_FIELD_ROW_COUNT = (
    M6_WAYMAX_MAX_SCENES
    * len(M6_WAYMAX_BUNDLES)
    * len(M6_WAYMAX_OFFICIAL_CONDITIONS)
    * len(M6_WAYMAX_OFFICIAL_COMPARISON_FIELDS)
)

M6_WAYMAX_NUMPY_COMPARISON_POLICIES = ("log_replay", "idm")
M6_WAYMAX_NUMPY_COMPARISON_ROW_COUNT = (
    M6_WAYMAX_MAX_SCENES
    * len(M6_WAYMAX_NUMPY_COMPARISON_POLICIES)
    * 4
)

_NUMPY_METRICS = (
    ("additional_target_braking_impulse_mps", M6_PAIRED_METRIC_VERSION, "m/s"),
    ("response_timeliness_s", M6_PAIRED_METRIC_VERSION, "s"),
    (
        "minimum_longitudinal_bumper_gap_change_m",
        M6_PAIRED_METRIC_VERSION,
        "m",
    ),
    ("target_progress_loss_m", M6_PAIRED_METRIC_VERSION, "m"),
)
_WAYMAX_CANONICAL_GIT_URL = "https://github.com/waymo-research/waymax.git"
_NUMPY_POLICY_ACCESS = {
    "log_replay": "privileged",
    "idm": "history_only",
}
_PINNED_RUNTIME_VERSIONS = {
    "jax": "0.4.38",
    "jaxlib": "0.4.38",
    "waymo-waymax": "0.1.0",
}
_SOURCE_AUTHORITY_DOMAIN = b"evalsim-m6-waymax-source-authority-v1"
_EXECUTION_AUTHORITY_DOMAIN = b"evalsim-m6-waymax-execution-authority-v1"
_NUMPY_VIEW_DOMAIN = b"evalsim-m6-waymax-numpy-20-transition-view-v1"
_NUMPY_ROW_DOMAIN = b"evalsim-m6-waymax-numpy-comparison-row-v1"
_NUMPY_TABLE_DOMAIN = b"evalsim-m6-waymax-numpy-comparison-table-v1"
_OFFICIAL_SOURCE_ISSUER = object()
_OFFICIAL_SOURCE_DOMAIN = b"evalsim-m6-waymax-official-source-v1"
_STORED_ELIGIBILITY_FIELDS = (
    "cohort_index",
    "primary_eligible",
    "rejection_reason",
    "secondary_b4_feasible",
)
_NUMPY_ISSUER = object()
_NONPROMOTABLE_ISSUER = object()

_FIELD_ROW_DOMAIN = b"evalsim-m6-waymax-official-field-row-v1"
_FIELD_TABLE_DOMAIN = b"evalsim-m6-waymax-official-field-table-v1"
_FIELD_ISSUER = object()
_RESULT_ISSUER = object()
_RESULT_DOMAIN = b"evalsim-m6-waymax-official-evidence-v1"
_SHA256 = frozenset("0123456789abcdef")


class M6WaymaxOfficialError(RuntimeError):
    """The runtime-injected official Waymax boundary failed closed."""


def _fail(code: str, detail: str) -> None:
    raise M6WaymaxOfficialError(f"{code}: {detail}")


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class _StrongIssuanceRecord:
    issued: Any
    binding_sha256: str
    content_sha256: str
    components: tuple[Any, ...]


_STRONG_ISSUANCE_LOCK = threading.Lock()
_STRONG_ISSUANCE_REGISTRY: dict[
    tuple[str, int], _StrongIssuanceRecord
] = {}


def _register_strong_issuance(
    kind: str,
    issued: Any,
    *,
    binding_sha256: str,
    content_sha256: str,
    components: Sequence[Any] = (),
) -> None:
    record = _StrongIssuanceRecord(
        issued=issued,
        binding_sha256=_sha256(binding_sha256, "binding_sha256"),
        content_sha256=_sha256(content_sha256, "content_sha256"),
        components=tuple(components),
    )
    key = (kind, id(issued))
    with _STRONG_ISSUANCE_LOCK:
        if key in _STRONG_ISSUANCE_REGISTRY:
            raise RuntimeError(f"{kind} issuance identity was reused")
        _STRONG_ISSUANCE_REGISTRY[key] = record


def _strong_issuance_record(
    kind: str,
    issued: Any,
    *,
    error_code: str,
) -> _StrongIssuanceRecord:
    with _STRONG_ISSUANCE_LOCK:
        record = _STRONG_ISSUANCE_REGISTRY.get((kind, id(issued)))
    if record is None or record.issued is not issued:
        _fail(error_code, f"{kind} was not issued by this process")
    return record


def _require_original_components(
    record: _StrongIssuanceRecord,
    components: Sequence[Any],
    *,
    error_code: str,
) -> None:
    current = tuple(components)
    if len(current) != len(record.components) or any(
        observed is not original
        for observed, original in zip(current, record.components, strict=True)
    ):
        _fail(error_code, "issued component identity changed")


def _scenario_snapshot(scenario: Scenario) -> ScenarioSnapshot:
    snapshot = ScenarioSnapshot.from_scenario(scenario)
    snapshot.revalidate()
    return snapshot


def _scenario_fingerprint(scenario: Scenario) -> str:
    # ScenarioSnapshot owns the complete source-neutral integrity fingerprint.  This
    # module is in the same package boundary and deliberately does not expose it.
    return _scenario_snapshot(scenario)._integrity_fingerprint


def _eligibility_ledger_sha256(ledger: M6EligibilityLedger) -> str:
    if not isinstance(ledger, M6EligibilityLedger):
        raise TypeError("eligibility_ledger must be M6EligibilityLedger")
    if ledger.input_n != M6_WAYMAX_OFFICIAL_POPULATION_SIZE:
        raise ValueError("official eligibility ledger must contain 128 entries")
    digest = hashlib.sha256()
    digest.update(b"evalsim-m6-waymax-numpy-eligibility-ledger-v1\x00")
    for expected_index, entry in enumerate(ledger.entries):
        entry.revalidate()
        if entry.cohort_index != expected_index:
            raise ValueError("official eligibility ledger must cover 0..127")
        digest.update(expected_index.to_bytes(4, "big"))
        digest.update(_canonical_json(entry.eligibility.to_dict()))
        digest.update(bytes.fromhex(entry.source_snapshot._integrity_fingerprint))
    return digest.hexdigest()


def m6_stored_eligibility_rows_sha256(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Hash an exact canonical 10- or 128-row eligibility store projection."""

    normalized = tuple(rows)
    if len(normalized) not in (10, M6_WAYMAX_OFFICIAL_POPULATION_SIZE):
        raise ValueError("stored eligibility projection must contain 10 or 128 rows")
    digest = hashlib.sha256()
    digest.update(b"evalsim-m6-stored-eligibility-projection-v1\x00")
    for expected_index, row in enumerate(normalized):
        if not isinstance(row, Mapping) or set(row) != set(
            _STORED_ELIGIBILITY_FIELDS
        ):
            raise TypeError("stored eligibility row schema is not exact")
        cohort_index = _strict_int(row["cohort_index"], "cohort_index")
        primary_eligible = row["primary_eligible"]
        rejection_reason = row["rejection_reason"]
        secondary_feasible = row["secondary_b4_feasible"]
        if cohort_index != expected_index or type(primary_eligible) is not bool:
            raise ValueError("stored eligibility rows are not canonical")
        if primary_eligible:
            if rejection_reason is not None or type(secondary_feasible) is not bool:
                raise ValueError("eligible stored row has invalid disposition")
        elif (
            not isinstance(rejection_reason, str)
            or not rejection_reason
            or secondary_feasible is not None
        ):
            raise ValueError("ineligible stored row has invalid disposition")
        canonical = {
            name: row[name] for name in _STORED_ELIGIBILITY_FIELDS
        }
        payload = _canonical_json(canonical)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _stored_eligibility_rows_sha256(
    evidence: M6OfficialNumpyRows,
) -> str:
    if not isinstance(evidence, M6OfficialNumpyRows):
        raise TypeError("numpy_evidence must be M6OfficialNumpyRows")
    evidence.revalidate()
    expected = m6_eligibility_rows(
        evidence.typed_result.eligibility_ledger,
        mode="official",
        secondary_plan_ledger=evidence.typed_result.secondary_plan_ledger,
    )
    if tuple(dict(row) for row in evidence.eligibility_rows) != tuple(
        dict(row) for row in expected
    ):
        _fail(
            "stored_eligibility_projection",
            "safe eligibility rows differ from typed NumPy result",
        )
    return m6_stored_eligibility_rows_sha256(evidence.eligibility_rows)


def _official_numpy_rows_sha256(evidence: M6OfficialNumpyRows) -> str:
    if not isinstance(evidence, M6OfficialNumpyRows):
        raise TypeError("numpy_evidence must be M6OfficialNumpyRows")
    evidence.revalidate()
    collections = (
        ("eligibility", evidence.eligibility_rows),
        ("primary", evidence.primary_scene_scalar_rows),
        ("primary_repeat", evidence.primary_repeat_scene_scalar_rows),
        ("secondary", evidence.secondary_scene_scalar_rows),
        ("negative", evidence.negative_timing_observation_rows),
    )
    digest = hashlib.sha256()
    digest.update(b"evalsim-m6-official-numpy-safe-rows-v1\x00")
    metadata = _canonical_json(
        {
            "adapter_version": evidence.adapter_version,
            "phase_durations_ms": dict(evidence.phase_durations_ms),
        }
    )
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    for label, rows in collections:
        encoded_label = label.encode("ascii")
        digest.update(len(encoded_label).to_bytes(4, "big"))
        digest.update(encoded_label)
        digest.update(len(rows).to_bytes(8, "big"))
        for row in rows:
            payload = _canonical_json(row)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def _source_authority_sha256(
    kind: str,
    promotable: bool,
    bindings: Sequence[tuple[int, str, int, str, str, str]],
) -> str:
    payload = {
        "bindings": [list(binding) for binding in bindings],
        "kind": kind,
        "promotable": promotable,
    }
    return hashlib.sha256(
        _SOURCE_AUTHORITY_DOMAIN + b"\x00" + _canonical_json(payload)
    ).hexdigest()


def _initialize_source_authority_api(
    cohort_type: type[Any],
    cohort_verifier: Callable[[Any], None],
    authority_hash: Callable[[str, bool, Sequence[Any]], str],
    fail_function: Callable[[str, str], None],
    error_type: type[Exception],
    member_type: type[Any],
):
    """Create the source authority API with lexical, one-use trust state."""

    # Ordinary rebinding of module names is inside the integrity boundary.
    # Mutation of closure cells/registries, class methods, descriptors, or code
    # objects is explicitly outside this local same-process boundary.
    receipt_lock = threading.RLock()
    receipt_records: dict[int, tuple[Any, ...]] = {}
    issuance_lock = threading.Lock()

    @dataclass(frozen=True, slots=True)
    class _SourceIssuance:
        authority: Any
        kind: Literal["verified_accepted_m4", "test_only"]
        promotable: bool
        member_bindings: tuple[tuple[int, str, int, str, str, str], ...]
        member_bindings_content: tuple[
            tuple[int, str, int, str, str, str], ...
        ]
        authority_sha256: str
        issued_original_sha256: str

    issuances: dict[int, _SourceIssuance] = {}
    dependencies = (
        cohort_type,
        cohort_verifier,
        authority_hash,
        fail_function,
        error_type,
        member_type,
    )

    def consume(
        receipt: object,
        kind: Literal["verified_accepted_m4", "test_only"],
        promotable: bool,
        bindings: tuple[tuple[int, str, int, str, str, str], ...],
    ) -> None:
        with receipt_lock:
            record = receipt_records.pop(id(receipt), None)
        verified_proof_invalid = (
            kind == "verified_accepted_m4"
            and (
                type(record[5]) is not dependencies[0]
                or record[5].members is not record[6]
                or record[6] != record[7]
            )
        ) if record is not None and len(record) == 8 else True
        test_proof_invalid = (
            kind == "test_only"
            and record is not None
            and tuple(record[5:]) != (None, None, None)
        )
        if (
            record is None
            or len(record) != 8
            or record[0] is not receipt
            or record[1] != kind
            or record[2] is not promotable
            or record[3] is not bindings
            or record[4] != bindings
            or verified_proof_invalid
            or test_proof_invalid
        ):
            raise TypeError(
                "M6WaymaxSourceAuthority is factory-issued only"
            )

    @dataclass(frozen=True, slots=True)
    class SourceAuthority:
        """Factory-issued accepted-M4 source authority or test-only non-authority."""

        kind: Literal["verified_accepted_m4", "test_only"]
        promotable: bool
        _member_bindings: tuple[
            tuple[int, str, int, str, str, str], ...
        ] = field(
            repr=False,
            compare=False,
        )
        authority_sha256: str | None = field(default=None, repr=False)
        _issued_original_sha256: str = field(
            init=False,
            repr=False,
            compare=False,
        )
        _issuance_receipt: InitVar[object] = None

        def __post_init__(self, _issuance_receipt: object) -> None:
            bindings = tuple(self._member_bindings)
            consume(
                _issuance_receipt,
                self.kind,
                self.promotable,
                bindings,
            )
            if self.kind == "verified_accepted_m4":
                if self.promotable is not True or len(bindings) != 128:
                    raise ValueError(
                        "verified source authority must bind 128 members"
                    )
            elif self.kind == "test_only":
                if self.promotable is not False or bindings:
                    raise ValueError(
                        "test source authority is permanently non-promotable"
                    )
            else:
                raise ValueError("source authority kind is not registered")
            if bindings and tuple(item[0] for item in bindings) != tuple(
                range(128)
            ):
                raise ValueError(
                    "source authority member bindings must cover 0..127"
                )
            object.__setattr__(self, "_member_bindings", bindings)
            expected = dependencies[2](
                self.kind,
                self.promotable,
                bindings,
            )
            if (
                self.authority_sha256 is not None
                and self.authority_sha256 != expected
            ):
                raise ValueError(
                    "authority_sha256 does not bind source authority"
                )
            object.__setattr__(self, "authority_sha256", expected)
            object.__setattr__(self, "_issued_original_sha256", expected)
            content = tuple(
                tuple(value for value in binding)
                for binding in self._member_bindings
            )
            record = _SourceIssuance(
                authority=self,
                kind=self.kind,
                promotable=self.promotable,
                member_bindings=self._member_bindings,
                member_bindings_content=content,
                authority_sha256=expected,
                issued_original_sha256=expected,
            )
            with issuance_lock:
                if id(self) in issuances:
                    raise RuntimeError(
                        "source authority issuance identity was reused"
                    )
                issuances[id(self)] = record

        def revalidate(self) -> None:
            with issuance_lock:
                record = issuances.get(id(self))
            if (
                record is None
                or record.authority is not self
                or self.kind != record.kind
                or self.promotable is not record.promotable
                or self._member_bindings is not record.member_bindings
                or self._member_bindings != record.member_bindings_content
                or self.authority_sha256 != record.authority_sha256
                or self._issued_original_sha256
                != record.issued_original_sha256
            ):
                dependencies[3](
                    "source_authority_mutated",
                    "source authority original issuance changed",
                )
            expected = dependencies[2](
                self.kind,
                self.promotable,
                self._member_bindings,
            )
            if (
                expected != record.authority_sha256
                or expected != self.authority_sha256
                or expected != self._issued_original_sha256
            ):
                dependencies[3](
                    "source_authority_mutated",
                    "source authority binding changed",
                )

        def verify_member(self, member: Any) -> None:
            self.revalidate()
            if type(member) is not dependencies[5]:
                raise TypeError("member must be ReloadedM4Member")
            if not self.promotable:
                return
            expected = self._member_bindings[member.cohort_index]
            record = member.record
            actual = (
                member.cohort_index,
                record.shard_suffix,
                record.record_ordinal,
                record.scenario_id,
                record.shard_sha256,
                record.dataset_config_fingerprint,
            )
            if (
                actual != expected
                or member.scenario.scenario_id != expected[3]
            ):
                dependencies[3](
                    "source_authority_member",
                    "reloaded member differs from accepted-M4 verifier binding",
                )

    SourceAuthority.__name__ = "M6WaymaxSourceAuthority"
    SourceAuthority.__qualname__ = "M6WaymaxSourceAuthority"
    factory_dependencies = (
        SourceAuthority,
        cohort_type,
        cohort_verifier,
        error_type,
    )

    def build_test():
        bindings: tuple[tuple[int, str, int, str, str, str], ...] = ()
        receipt = object()
        with receipt_lock:
            receipt_records[id(receipt)] = (
                receipt,
                "test_only",
                False,
                bindings,
                bindings,
                None,
                None,
                None,
            )
            try:
                authority = factory_dependencies[0](
                    kind="test_only",
                    promotable=False,
                    _member_bindings=bindings,
                    _issuance_receipt=receipt,
                )
                if type(authority) is not factory_dependencies[0]:
                    raise TypeError(
                        "source authority factory returned wrong type"
                    )
                authority.revalidate()
                return authority
            finally:
                receipt_records.pop(id(receipt), None)

    def build_verified(cohort: Any):
        if type(cohort) is not factory_dependencies[1]:
            raise TypeError("cohort must be exact AcceptedM4Cohort")
        factory_dependencies[2](cohort)
        members = cohort.members
        if type(members) is not tuple:
            raise TypeError("verified cohort members must be an immutable tuple")
        bindings = tuple(
            (
                member.cohort_index,
                member.expectation.locator.shard_suffix,
                member.expectation.locator.record_ordinal,
                member.expectation.expected_scenario_id,
                member.expectation.expected_shard_sha256,
                member.expectation.expected_dataset_config_fingerprint,
            )
            for member in members
        )
        member_content = tuple(member for member in members)
        factory_dependencies[2](cohort)
        rebound = tuple(
            (
                member.cohort_index,
                member.expectation.locator.shard_suffix,
                member.expectation.locator.record_ordinal,
                member.expectation.expected_scenario_id,
                member.expectation.expected_shard_sha256,
                member.expectation.expected_dataset_config_fingerprint,
            )
            for member in cohort.members
        )
        if (
            cohort.members is not members
            or tuple(cohort.members) != member_content
            or rebound != bindings
        ):
            raise factory_dependencies[3](
                "source_authority_verifier_drift: accepted M4 cohort changed"
            )
        content = tuple(tuple(value for value in row) for row in bindings)
        receipt = object()
        with receipt_lock:
            receipt_records[id(receipt)] = (
                receipt,
                "verified_accepted_m4",
                True,
                bindings,
                content,
                cohort,
                members,
                member_content,
            )
            try:
                authority = factory_dependencies[0](
                    kind="verified_accepted_m4",
                    promotable=True,
                    _member_bindings=bindings,
                    _issuance_receipt=receipt,
                )
                if type(authority) is not factory_dependencies[0]:
                    raise TypeError(
                        "source authority factory returned wrong type"
                    )
                authority.revalidate()
                return authority
            finally:
                receipt_records.pop(id(receipt), None)

    return SourceAuthority, build_test, build_verified


(
    M6WaymaxSourceAuthority,
    build_m6_waymax_test_source_authority,
    build_m6_waymax_verified_source_authority,
) = _initialize_source_authority_api(
    AcceptedM4Cohort,
    reverify_accepted_m4_run,
    _source_authority_sha256,
    _fail,
    M6WaymaxOfficialError,
    ReloadedM4Member,
)
del _initialize_source_authority_api




@dataclass(frozen=True, slots=True)
class _ResidentCandidate:
    cohort_index: int
    rank_sha256: str
    state: Any = field(repr=False, compare=False)
    scenario: ScenarioSnapshot = field(repr=False, compare=False)
    qualification: M6WaymaxEligibility = field(repr=False, compare=False)

    def revalidate(self) -> None:
        self.scenario.revalidate()
        self.qualification.revalidate()
        if (
            self.cohort_index != self.qualification.cohort_index
            or self.rank_sha256 != self.qualification.rank_sha256
            or not self.qualification.eligible
            or source_state_mutation_sha256(self.state)
            != self.qualification.source_binding_sha256
        ):
            _fail(
                "resident_candidate_mutated",
                "resident state/snapshot no longer matches its qualification",
            )


@dataclass(frozen=True, slots=True)
class _M6WaymaxOfficialSourceIssuance:
    source: Any
    source_authority: M6WaymaxSourceAuthority
    numpy_eligibility_ledger: M6EligibilityLedger
    primary_domain: M6WaymaxPrimaryDomain
    qualification_ledger: M6WaymaxQualificationLedger
    selection: M6WaymaxSelection
    residents: tuple[_ResidentCandidate, ...]
    member_bindings: tuple[tuple[int, str, int, str, str, str], ...]
    source_scenarios: tuple[ScenarioSnapshot, ...]
    source_binding_sha256: str


_OFFICIAL_SOURCE_ISSUANCE_LOCK = threading.Lock()
_OFFICIAL_SOURCE_ISSUANCE_REGISTRY: dict[
    int, _M6WaymaxOfficialSourceIssuance
] = {}


def _official_source_sha256(source: "M6WaymaxOfficialSource") -> str:
    payload = {
        "source_authority_identity": id(source._source_authority),
        "source_authority_sha256": source._source_authority.authority_sha256,
        "source_authority_kind": source._source_authority.kind,
        "member_bindings": [list(item) for item in source._member_bindings],
        "source_scenarios": [
            snapshot._integrity_fingerprint
            for snapshot in source._source_scenarios
        ],
        "numpy_eligibility_sha256": source._numpy_eligibility_sha256,
        "numpy_eligibility_identity": id(source._numpy_eligibility_ledger),
        "primary_domain_sha256": source._primary_domain.domain_sha256,
        "primary_domain_identity": id(source._primary_domain),
        "qualification_ledger_sha256": (
            source._qualification_ledger.ledger_sha256
        ),
        "qualification_ledger_identity": id(source._qualification_ledger),
        "selection_sha256": source._selection.selection_sha256,
        "selection_identity": id(source._selection),
        "residents": [
            {
                "cohort_index": resident.cohort_index,
                "qualification_binding_sha256": (
                    resident.qualification.qualification_binding_sha256
                ),
                "rank_sha256": resident.rank_sha256,
                "scenario_sha256": resident.scenario._integrity_fingerprint,
                "source_binding_sha256": (
                    resident.qualification.source_binding_sha256
                ),
                "state_identity": id(resident.state),
            }
            for resident in source._residents
        ],
    }
    return hashlib.sha256(
        _OFFICIAL_SOURCE_DOMAIN + b"\x00" + _canonical_json(payload)
    ).hexdigest()


class M6WaymaxOfficialSource:
    """Collector-issued immutable ledgers plus selected resident candidates."""

    _AUTHORITY_TYPES = (M6WaymaxSourceAuthority,)

    __slots__ = (
        "_source_authority",
        "_numpy_eligibility_ledger",
        "_numpy_eligibility_sha256",
        "_primary_domain",
        "_qualification_ledger",
        "_selection",
        "_residents",
        "_member_bindings",
        "_source_scenarios",
        "_source_binding_sha256",
        "_issued_original_sha256",
    )

    def __init__(
        self,
        *,
        source_authority: M6WaymaxSourceAuthority,
        numpy_eligibility_ledger: M6EligibilityLedger,
        primary_domain: M6WaymaxPrimaryDomain,
        qualification_ledger: M6WaymaxQualificationLedger,
        selection: M6WaymaxSelection,
        residents: Sequence[_ResidentCandidate],
        member_bindings: Sequence[
            tuple[int, str, int, str, str, str]
        ],
        source_scenarios: Sequence[ScenarioSnapshot],
        _issuance_capability: object = None,
    ) -> None:
        if _issuance_capability is not _OFFICIAL_SOURCE_ISSUER:
            raise TypeError("M6WaymaxOfficialSource is collector-issued only")
        if type(source_authority) is not self._AUTHORITY_TYPES[0]:
            raise TypeError(
                "source_authority must be exact M6WaymaxSourceAuthority"
            )
        self._source_authority = source_authority
        self._numpy_eligibility_ledger = numpy_eligibility_ledger
        self._numpy_eligibility_sha256 = _eligibility_ledger_sha256(
            numpy_eligibility_ledger
        )
        self._primary_domain = primary_domain
        self._qualification_ledger = qualification_ledger
        self._selection = selection
        self._residents = tuple(residents)
        self._member_bindings = tuple(member_bindings)
        self._source_scenarios = tuple(source_scenarios)
        self._validate_semantics()
        expected = _official_source_sha256(self)
        self._source_binding_sha256 = expected
        self._issued_original_sha256 = expected
        record = _M6WaymaxOfficialSourceIssuance(
            source=self,
            source_authority=self._source_authority,
            numpy_eligibility_ledger=self._numpy_eligibility_ledger,
            primary_domain=self._primary_domain,
            qualification_ledger=self._qualification_ledger,
            selection=self._selection,
            residents=self._residents,
            member_bindings=self._member_bindings,
            source_scenarios=self._source_scenarios,
            source_binding_sha256=expected,
        )
        with _OFFICIAL_SOURCE_ISSUANCE_LOCK:
            if id(self) in _OFFICIAL_SOURCE_ISSUANCE_REGISTRY:
                raise RuntimeError("official source issuance identity was reused")
            _OFFICIAL_SOURCE_ISSUANCE_REGISTRY[id(self)] = record

    @property
    def primary_domain(self) -> M6WaymaxPrimaryDomain:
        return self._primary_domain

    @property
    def qualification_ledger(self) -> M6WaymaxQualificationLedger:
        return self._qualification_ledger

    @property
    def selection(self) -> M6WaymaxSelection:
        return self._selection

    @property
    def numpy_eligibility_ledger(self) -> M6EligibilityLedger:
        return self._numpy_eligibility_ledger

    @property
    def source_binding_sha256(self) -> str:
        return self._source_binding_sha256

    @property
    def promotable(self) -> bool:
        return self._source_authority.promotable

    @property
    def resident_candidate_count(self) -> int:
        return len(self._residents)

    def _validate_semantics(self) -> None:
        if type(self._source_authority) is not self._AUTHORITY_TYPES[0]:
            _fail("official_source_mutated", "source authority type changed")
        if not isinstance(self._numpy_eligibility_ledger, M6EligibilityLedger):
            _fail("official_source_mutated", "NumPy ledger type changed")
        if not isinstance(self._primary_domain, M6WaymaxPrimaryDomain):
            _fail("official_source_mutated", "primary domain type changed")
        if not isinstance(
            self._qualification_ledger,
            M6WaymaxQualificationLedger,
        ):
            _fail("official_source_mutated", "qualification ledger type changed")
        if not isinstance(self._selection, M6WaymaxSelection):
            _fail("official_source_mutated", "selection type changed")
        self._primary_domain.revalidate()
        self._qualification_ledger.revalidate()
        self._source_authority.revalidate()
        if (
            _eligibility_ledger_sha256(self._numpy_eligibility_ledger)
            != self._numpy_eligibility_sha256
        ):
            _fail(
                "official_source_numpy_ledger",
                "frozen NumPy eligibility ledger changed",
            )
        self._selection.revalidate(primary_domain=self._primary_domain)
        if (
            self._qualification_ledger.ledger_sha256
            != self._selection.qualification_ledger_sha256
            or self._qualification_ledger.primary_domain_sha256
            != self._primary_domain.domain_sha256
            or self._selection.qualification_ledger
            is not self._qualification_ledger
        ):
            _fail(
                "official_source_ledger_mismatch",
                "source ledgers no longer share one primary domain",
            )
        if (
            len(self._member_bindings) != M6_WAYMAX_OFFICIAL_POPULATION_SIZE
            or tuple(item[0] for item in self._member_bindings)
            != tuple(range(M6_WAYMAX_OFFICIAL_POPULATION_SIZE))
            or len(self._source_scenarios)
            != M6_WAYMAX_OFFICIAL_POPULATION_SIZE
        ):
            _fail(
                "official_source_complete_domain",
                "source seal must bind every cohort member 0..127",
            )
        for index, snapshot in enumerate(self._source_scenarios):
            if not isinstance(snapshot, ScenarioSnapshot):
                _fail("official_source_mutated", "source snapshot type changed")
            snapshot.revalidate()
            numpy_entry = self._numpy_eligibility_ledger.entry_for(index)
            if (
                snapshot._integrity_fingerprint
                != numpy_entry.source_snapshot._integrity_fingerprint
            ):
                _fail(
                    "official_source_numpy_source",
                    "collected and NumPy source snapshots differ",
                )
        if self._source_authority.promotable:
            if self._member_bindings != self._source_authority._member_bindings:
                _fail(
                    "official_source_authority_domain",
                    "collected members differ from verified authority",
                )
            for binding, snapshot in zip(
                self._member_bindings,
                self._source_scenarios,
                strict=True,
            ):
                if binding[3] != snapshot.scenario_id:
                    _fail(
                        "official_source_authority_domain",
                        "authority scenario differs from collected snapshot",
                    )
        elif self._source_authority._member_bindings:
            _fail(
                "official_source_authority_domain",
                "test source authority unexpectedly carries member bindings",
            )
        entries = self._primary_domain.entry_by_cohort_index
        if tuple(entries) != self._numpy_eligibility_ledger.eligible_indices:
            _fail(
                "official_source_primary_domain",
                "Waymax primary domain differs from NumPy eligibility ledger",
            )
        if tuple(row.cohort_index for row in self._qualification_ledger.rows) != tuple(
            entries
        ):
            _fail(
                "official_source_qualification_domain",
                "qualification ledger differs from primary domain",
            )
        for cohort_index, primary_entry in entries.items():
            numpy_entry = self._numpy_eligibility_ledger.entry_for(cohort_index)
            source = numpy_entry.source_snapshot.to_scenario()
            target_index = numpy_entry.target_index
            if (
                target_index is None
                or primary_entry.scenario_id != source.scenario_id
                or primary_entry.upstream_eligibility.to_dict()
                != numpy_entry.eligibility.to_dict()
                or primary_entry.target_contract_id
                != source.agents[target_index].id
            ):
                _fail(
                    "official_source_target",
                    "Waymax primary target differs from frozen NumPy target",
                )
        expected = self._selection.members if self._selection.supported else ()
        if len(self._residents) != len(expected):
            _fail(
                "official_source_resident_mismatch",
                "resident candidates differ from the supported selection",
            )
        for resident, member in zip(self._residents, expected, strict=True):
            if not isinstance(resident, _ResidentCandidate):
                _fail("official_source_mutated", "resident type changed")
            resident.revalidate()
            if (
                resident.cohort_index != member.cohort_index
                or resident.qualification is not member
                or resident.scenario
                is not self._source_scenarios[resident.cohort_index]
                or resident.qualification.qualification_binding_sha256
                != member.qualification_binding_sha256
            ):
                _fail(
                    "official_source_resident_order",
                    "resident candidates are not exact selected members",
                )

    def revalidate(self) -> None:
        with _OFFICIAL_SOURCE_ISSUANCE_LOCK:
            record = _OFFICIAL_SOURCE_ISSUANCE_REGISTRY.get(id(self))
        if (
            record is None
            or record.source is not self
            or self._source_authority is not record.source_authority
            or self._numpy_eligibility_ledger
            is not record.numpy_eligibility_ledger
            or self._primary_domain is not record.primary_domain
            or self._qualification_ledger is not record.qualification_ledger
            or self._selection is not record.selection
            or self._residents is not record.residents
            or self._member_bindings is not record.member_bindings
            or self._source_scenarios is not record.source_scenarios
        ):
            _fail(
                "official_source_mutated",
                "official source component identity changed",
            )
        self._validate_semantics()
        expected = _official_source_sha256(self)
        if (
            expected != record.source_binding_sha256
            or self._source_binding_sha256 != expected
            or self._issued_original_sha256 != expected
        ):
            _fail("official_source_mutated", "official source seal changed")

    def _materialize(
        self,
        selection_position: int,
    ) -> tuple[Any, Scenario, M6WaymaxEligibility]:
        self.revalidate()
        position = _strict_int(selection_position, "selection_position")
        if position >= len(self._residents):
            raise IndexError("selection_position is outside resident candidates")
        resident = self._residents[position]
        scenario = resident.scenario.to_scenario()
        if _scenario_fingerprint(scenario) != resident.scenario._integrity_fingerprint:
            _fail(
                "official_source_snapshot_materialization",
                "materialized scenario differs from its immutable snapshot",
            )
        return resident.state, scenario, resident.qualification



class M6WaymaxOfficialCollector:
    """Shard-order-independent accepted-M4 visitor with a top-16 native-state cap."""

    _AUTHORITY_TYPES = (M6WaymaxSourceAuthority,)

    __slots__ = (
        "_source_authority",
        "_source_scenarios",
        "_seen",
        "_member_bindings",
        "_primary_entries",
        "_qualifications",
        "_primary_scenarios",
        "_residents",
        "_finalized",
    )

    def __init__(self, source_authority: M6WaymaxSourceAuthority) -> None:
        if type(source_authority) is not self._AUTHORITY_TYPES[0]:
            raise TypeError(
                "source_authority must be exact M6WaymaxSourceAuthority"
            )
        source_authority.revalidate()
        self._source_authority = source_authority
        self._source_scenarios: dict[int, ScenarioSnapshot] = {}
        self._seen: set[int] = set()
        self._member_bindings: dict[
            int, tuple[int, str, int, str, str, str]
        ] = {}
        self._primary_entries: dict[int, Any] = {}
        self._qualifications: dict[int, M6WaymaxEligibility] = {}
        self._primary_scenarios: dict[int, ScenarioSnapshot] = {}
        self._residents: dict[int, _ResidentCandidate] = {}
        self._finalized = False

    @property
    def count(self) -> int:
        return len(self._seen)

    @property
    def primary_member_count(self) -> int:
        return len(self._primary_entries)

    @property
    def qualified_count(self) -> int:
        return sum(row.eligible for row in self._qualifications.values())

    @property
    def resident_candidate_count(self) -> int:
        return len(self._residents)

    def __call__(self, member: ReloadedM4Member) -> None:
        if self._finalized:
            _fail("official_collector_finalized", "collector cannot be reused")
        if not isinstance(member, ReloadedM4Member):
            raise TypeError("member must be a ReloadedM4Member")
        self._source_authority.verify_member(member)
        index = member.cohort_index
        if index in self._seen:
            _fail("official_collector_duplicate", "cohort index arrived twice")
        record = member.record
        actual_binding = (
            index,
            record.shard_suffix,
            record.record_ordinal,
            record.scenario_id,
            record.shard_sha256,
            record.dataset_config_fingerprint,
        )

        before_scenario = _scenario_snapshot(member.scenario)
        before_fingerprint = before_scenario._integrity_fingerprint
        primary = evaluate_primary_brake_eligibility(member.scenario)
        if primary.eligible:
            before_state = source_state_mutation_sha256(member.record.state)
            entry = build_m6_waymax_primary_domain_entry(
                member.record.state,
                member.scenario,
                primary,
                cohort_index=index,
            )
            qualification = evaluate_m6_waymax_eligibility(
                member.record.state,
                member.scenario,
                primary,
                cohort_index=index,
                primary_entry=entry,
            )
            if source_state_mutation_sha256(member.record.state) != before_state:
                _fail(
                    "official_collector_state_mutated",
                    "source qualification mutated the native state",
                )
            self._primary_entries[index] = entry
            self._qualifications[index] = qualification
            self._primary_scenarios[index] = before_scenario
            if qualification.eligible:
                self._residents[index] = _ResidentCandidate(
                    cohort_index=index,
                    rank_sha256=qualification.rank_sha256,
                    state=member.record.state,
                    scenario=before_scenario,
                    qualification=qualification,
                )
                if len(self._residents) > M6_WAYMAX_MAX_SCENES:
                    worst = max(
                        self._residents.values(),
                        key=lambda candidate: (
                            bytes.fromhex(candidate.rank_sha256),
                            candidate.cohort_index,
                        ),
                    )
                    del self._residents[worst.cohort_index]
        if _scenario_fingerprint(member.scenario) != before_fingerprint:
            _fail(
                "official_collector_scenario_mutated",
                "source qualification mutated the contract scenario",
            )
        self._seen.add(index)
        self._source_scenarios[index] = before_scenario

        self._member_bindings[index] = actual_binding
    def finalize(self, eligibility_ledger: M6EligibilityLedger) -> M6WaymaxOfficialSource:
        if self._finalized:
            _fail("official_collector_finalized", "collector cannot be finalized twice")
        expected = set(range(M6_WAYMAX_OFFICIAL_POPULATION_SIZE))
        if self._seen != expected:
            _fail(
                "official_collector_incomplete",
                "collector must cover every opaque cohort index 0..127",
            )
        _eligibility_ledger_sha256(eligibility_ledger)
        for cohort_index in range(M6_WAYMAX_OFFICIAL_POPULATION_SIZE):
            snapshot = self._source_scenarios[cohort_index]
            numpy_entry = eligibility_ledger.entry_for(cohort_index)
            if (
                snapshot._integrity_fingerprint
                != numpy_entry.source_snapshot._integrity_fingerprint
            ):
                _fail(
                    "official_collector_numpy_source",
                    "NumPy and Waymax collectors retained different source scenes",
                )
            recomputed = evaluate_primary_brake_eligibility(
                snapshot.to_scenario()
            )
            if recomputed.to_dict() != numpy_entry.eligibility.to_dict():
                _fail(
                    "official_collector_numpy_eligibility",
                    "NumPy and Waymax primary eligibility differs",
                )
        if tuple(sorted(self._primary_entries)) != eligibility_ledger.eligible_indices:
            _fail(
                "official_collector_numpy_domain",
                "Waymax primary domain differs from frozen NumPy ledger",
            )
        primary_domain = M6WaymaxPrimaryDomain(
            tuple(self._primary_entries.values())
        )
        primary_scenarios = {
            index: snapshot.to_scenario()
            for index, snapshot in self._primary_scenarios.items()
        }
        ledger = build_m6_waymax_qualification_ledger(
            tuple(self._qualifications.values()),
            primary_domain=primary_domain,
            primary_scenarios=primary_scenarios,
        )
        selection = select_m6_waymax_subset(
            ledger,
            primary_domain=primary_domain,
        )
        resident_by_index = self._residents
        if selection.supported:
            selected_indices = tuple(
                member.cohort_index for member in selection.members
            )
            if set(resident_by_index) != set(selected_indices):
                _fail(
                    "official_collector_resident_selection",
                    "resident top-16 differs from canonical ledger selection",
                )
            residents = tuple(resident_by_index[index] for index in selected_indices)
        else:
            residents = ()
        member_bindings = tuple(
            self._member_bindings[index]
            for index in range(M6_WAYMAX_OFFICIAL_POPULATION_SIZE)
        )
        source_scenarios = tuple(
            self._source_scenarios[index]
            for index in range(M6_WAYMAX_OFFICIAL_POPULATION_SIZE)
        )
        source = M6WaymaxOfficialSource(
            primary_domain=primary_domain,
            source_authority=self._source_authority,
            numpy_eligibility_ledger=eligibility_ledger,
            qualification_ledger=ledger,
            selection=selection,
            residents=residents,
            member_bindings=member_bindings,
            source_scenarios=source_scenarios,
            _issuance_capability=_OFFICIAL_SOURCE_ISSUER,
        )
        self._residents = {}
        self._source_scenarios = {}
        self._member_bindings = {}
        self._finalized = True
        return source


# A compatibility spelling for callers that prefer the noun before its role.
M6WaymaxOfficialSourceCollector = M6WaymaxOfficialCollector


class M6WaymaxCompactExecutor(Protocol):
    """One independently invoked eager or JIT compact execution entry point."""

    def __call__(
        self,
        state: Any,
        scenario: Scenario,
        plan: EgoTrajectoryPlan,
        *,
        bundle: str,
        selection: M6WaymaxSelection,
        primary_domain: M6WaymaxPrimaryDomain,
        selection_position: int,
    ) -> tuple[CompactM6WaymaxRollout, WaymaxEgoPlanView]: ...


@dataclass(frozen=True, slots=True)
class M6WaymaxOfficialExecutors:
    """Runtime-owned eager and position-zero JIT execution callables."""

    eager: M6WaymaxCompactExecutor = field(repr=False)
    jit_eager: M6WaymaxCompactExecutor = field(repr=False)
    jit_compiled: M6WaymaxCompactExecutor = field(repr=False)

    def __post_init__(self) -> None:
        for name in ("eager", "jit_eager", "jit_compiled"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")


_CALLABLE_BEHAVIOR_MISSING = object()
_EMPTY_CLOSURE_CELL = object()


def _code_behavior_sha256(value: Any) -> str | None:
    if value is _CALLABLE_BEHAVIOR_MISSING or value is None:
        return None
    try:
        payload = marshal.dumps(value)
    except (TypeError, ValueError):
        payload = repr(value).encode("utf-8", errors="backslashreplace")
    return hashlib.sha256(payload).hexdigest()


def _default_behavior_payload(value: Any, seen: set[int]) -> Any:
    if value is _CALLABLE_BEHAVIOR_MISSING:
        return {"missing": True}
    if value is None or type(value) in (bool, int, str):
        return {"literal_type": type(value).__name__, "value": value}
    if type(value) is float:
        return {"literal_type": "float", "value": value.hex()}
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    identity = id(value)
    if identity in seen:
        return {"recursive_identity": identity}
    if type(value) in (tuple, list):
        seen.add(identity)
        try:
            return {
                type(value).__name__: [
                    _default_behavior_payload(item, seen) for item in value
                ]
            }
        finally:
            seen.remove(identity)
    if isinstance(value, (frozenset, set)):
        seen.add(identity)
        try:
            items = [
                _default_behavior_payload(item, seen) for item in value
            ]
            items.sort(
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return {"set": items, "frozen": isinstance(value, frozenset)}
        finally:
            seen.remove(identity)
    if isinstance(value, Mapping):
        seen.add(identity)
        try:
            items = [
                (
                    _default_behavior_payload(key, seen),
                    _default_behavior_payload(item, seen),
                )
                for key, item in value.items()
            ]
            items.sort(
                key=lambda item: json.dumps(
                    item[0],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return {"mapping": items}
        finally:
            seen.remove(identity)
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        payload = np.ascontiguousarray(array).tobytes(order="C")
        return {
            "array_dtype": array.dtype.str,
            "array_shape": list(array.shape),
            "array_sha256": hashlib.sha256(payload).hexdigest(),
        }
    return {
        "identity": identity,
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
    }


def _global_content_behavior_payload(value: Any, seen: set[int]) -> Any:
    if value is _CALLABLE_BEHAVIOR_MISSING:
        return {"missing": True}
    if value is None or type(value) in (bool, int, float, str, bytes):
        return _default_behavior_payload(value, seen)
    code = getattr(value, "__code__", _CALLABLE_BEHAVIOR_MISSING)
    if callable(value) and code is not _CALLABLE_BEHAVIOR_MISSING:
        return {"callable": _callable_behavior_payload(value, seen)}

    identity = id(value)
    if identity in seen:
        return {"recursive_dependency_identity": identity}
    seen.add(identity)
    try:
        if type(value) in (tuple, list):
            return {
                "sequence_type": type(value).__name__,
                "items": [
                    _global_content_behavior_payload(item, seen)
                    for item in value
                ],
            }
        if type(value) in (frozenset, set):
            items = [
                _global_content_behavior_payload(item, seen)
                for item in value
            ]
            items.sort(
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return {
                "set_type": type(value).__name__,
                "items": items,
            }
        if type(value) is dict or isinstance(value, MappingProxyType):
            items = [
                (
                    _global_content_behavior_payload(key, seen),
                    _global_content_behavior_payload(item, seen),
                )
                for key, item in value.items()
            ]
            items.sort(
                key=lambda item: json.dumps(
                    item[0],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return {"mapping": items}
        if isinstance(value, np.ndarray):
            array = np.asarray(value)
            if array.dtype.hasobject:
                return {
                    "array_dtype": array.dtype.str,
                    "array_shape": list(array.shape),
                    "array_objects": [
                        _global_content_behavior_payload(item, seen)
                        for item in array.flat
                    ],
                }
            payload = np.ascontiguousarray(array).tobytes(order="C")
            return {
                "array_dtype": array.dtype.str,
                "array_shape": list(array.shape),
                "array_sha256": hashlib.sha256(payload).hexdigest(),
            }
        # Modules, types, and runtime handles form the explicit opaque domain.
        # They are sealed by exact identity; mutable containers are never opaque.
        return {
            "domain": "opaque_identity",
            "identity": identity,
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
        }
    finally:
        seen.remove(identity)


def _closure_content_behavior_payload(value: Any, seen: set[int]) -> Any:
    if value is _EMPTY_CLOSURE_CELL:
        return {"empty_cell": True}
    if callable(value) and getattr(
        value,
        "__code__",
        _CALLABLE_BEHAVIOR_MISSING,
    ) is not _CALLABLE_BEHAVIOR_MISSING:
        return {"callable": _callable_behavior_payload(value, seen)}
    if (
        value is None
        or type(value)
        in (bool, int, float, str, bytes, tuple, list, frozenset, set, dict)
        or isinstance(value, (MappingProxyType, np.ndarray))
    ):
        return _global_content_behavior_payload(value, seen)
    # Exact opaque closure identity is reserved for runtime handles and
    # dedicated mutable cache objects, never ordinary containers.
    return {
        "domain": "opaque_closure_identity",
        "identity": id(value),
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
    }


def _callable_behavior_payload(value: Any, seen: set[int]) -> Mapping[str, Any]:
    identity = id(value)
    if identity in seen:
        return {"recursive_callable_identity": identity}
    seen.add(identity)
    try:
        code = getattr(value, "__code__", _CALLABLE_BEHAVIOR_MISSING)
        defaults = getattr(
            value,
            "__defaults__",
            _CALLABLE_BEHAVIOR_MISSING,
        )
        kwdefaults = getattr(
            value,
            "__kwdefaults__",
            _CALLABLE_BEHAVIOR_MISSING,
        )
        closure = getattr(value, "__closure__", _CALLABLE_BEHAVIOR_MISSING)
        cells = tuple(closure) if isinstance(closure, tuple) else ()
        global_mapping = getattr(
            value, "__globals__", _CALLABLE_BEHAVIOR_MISSING
        )
        code_names = getattr(code, "co_names", ())
        global_names = tuple(
            sorted(
                name
                for name in code_names
                if isinstance(global_mapping, Mapping)
                and name in global_mapping
            )
        )
        global_payload = [
            {
                "name": name,
                "identity": id(global_mapping[name]),
                "content": _global_content_behavior_payload(
                    global_mapping[name], seen
                ),
            }
            for name in global_names
        ]
        closure_payload = []
        for cell in cells:
            try:
                content = cell.cell_contents
            except ValueError:
                content = _EMPTY_CLOSURE_CELL
            closure_payload.append(
                {
                    "cell_identity": id(cell),
                    "content_identity": id(content),
                    "content": _closure_content_behavior_payload(
                        content,
                        seen,
                    ),
                }
            )
        return {
            "callable_identity": identity,
            "type_module": type(value).__module__,
            "type_qualname": type(value).__qualname__,
            "module": _default_behavior_payload(
                getattr(value, "__module__", None),
                seen,
            ),
            "qualname": _default_behavior_payload(
                getattr(value, "__qualname__", None),
                seen,
            ),
            "code_identity": id(code),
            "code_sha256": _code_behavior_sha256(code),
            "defaults_identity": id(defaults),
            "defaults": _default_behavior_payload(defaults, seen),
            "kwdefaults_identity": id(kwdefaults),
            "kwdefaults": _default_behavior_payload(kwdefaults, seen),
            "closure_identity": id(closure),
            "closure": closure_payload,
            "globals_identity": id(global_mapping),
            "globals": global_payload,
        }
    finally:
        seen.remove(identity)


@dataclass(frozen=True, slots=True)
class _CallableBehaviorSeal:
    callable: Any
    type_module: str
    type_qualname: str
    module: Any
    qualname: Any
    code: Any
    defaults: Any
    kwdefaults: Any
    closure: Any
    closure_cells: tuple[Any, ...]
    closure_contents: tuple[Any, ...]
    globals_mapping: Any
    global_names: tuple[str, ...]
    global_values: tuple[Any, ...]
    behavior_sha256: str


def _capture_callable_behavior(value: Any) -> _CallableBehaviorSeal:
    code = getattr(value, "__code__", _CALLABLE_BEHAVIOR_MISSING)
    defaults = getattr(value, "__defaults__", _CALLABLE_BEHAVIOR_MISSING)
    kwdefaults = getattr(
        value,
        "__kwdefaults__",
        _CALLABLE_BEHAVIOR_MISSING,
    )
    closure = getattr(value, "__closure__", _CALLABLE_BEHAVIOR_MISSING)
    cells = tuple(closure) if isinstance(closure, tuple) else ()
    contents = []
    for cell in cells:
        try:
            contents.append(cell.cell_contents)
        except ValueError:
            contents.append(_EMPTY_CLOSURE_CELL)
    globals_mapping = getattr(
        value,
        "__globals__",
        _CALLABLE_BEHAVIOR_MISSING,
    )
    code_names = getattr(code, "co_names", ())
    global_names = tuple(
        sorted(
            name
            for name in code_names
            if isinstance(globals_mapping, Mapping)
            and name in globals_mapping
        )
    )
    global_values = tuple(globals_mapping[name] for name in global_names)
    payload = _callable_behavior_payload(value, set())
    behavior_sha256 = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return _CallableBehaviorSeal(
        callable=value,
        type_module=type(value).__module__,
        type_qualname=type(value).__qualname__,
        module=getattr(value, "__module__", None),
        qualname=getattr(value, "__qualname__", None),
        code=code,
        defaults=defaults,
        kwdefaults=kwdefaults,
        globals_mapping=globals_mapping,
        global_names=global_names,
        global_values=global_values,
        closure=closure,
        closure_cells=cells,
        closure_contents=tuple(contents),
        behavior_sha256=behavior_sha256,
    )


def _callable_behavior_matches(
    value: Any,
    original: _CallableBehaviorSeal,
) -> bool:
    current = _capture_callable_behavior(value)
    return (
        current.callable is original.callable
        and current.type_module == original.type_module
        and current.type_qualname == original.type_qualname
        and current.module == original.module
        and current.qualname == original.qualname
        and current.code is original.code
        and current.globals_mapping is original.globals_mapping
        and current.global_names == original.global_names
        and len(current.global_values) == len(original.global_values)
        and all(
            observed is expected
            for observed, expected in zip(
                current.global_values,
                original.global_values,
                strict=True,
            )
        )
        and current.defaults is original.defaults
        and current.kwdefaults is original.kwdefaults
        and current.closure is original.closure
        and len(current.closure_cells) == len(original.closure_cells)
        and all(
            observed is expected
            for observed, expected in zip(
                current.closure_cells,
                original.closure_cells,
                strict=True,
            )
        )
        and len(current.closure_contents) == len(original.closure_contents)
        and all(
            observed is expected
            for observed, expected in zip(
                current.closure_contents,
                original.closure_contents,
                strict=True,
            )
        )
        and current.behavior_sha256 == original.behavior_sha256
    )


def _execution_authority_sha256(
    kind: str,
    promotable: bool,
    runtime_facts: Sequence[tuple[str, str]],
    executors: M6WaymaxOfficialExecutors,
) -> str:
    payload = {
        "kind": kind,
        "promotable": promotable,
        "runtime_facts": [list(item) for item in runtime_facts],
        "executors": [
            {
                "name": name,
                "identity": id(getattr(executors, name)),
                "behavior_sha256": _capture_callable_behavior(
                    getattr(executors, name)
                ).behavior_sha256,
                "type": (
                    f"{type(getattr(executors, name)).__module__}."
                    f"{type(getattr(executors, name)).__qualname__}"
                ),
                "module": getattr(getattr(executors, name), "__module__", None),
                "qualname": getattr(
                    getattr(executors, name),
                    "__qualname__",
                    None,
                ),
            }
            for name in ("eager", "jit_eager", "jit_compiled")
        ],
    }
    return hashlib.sha256(
        _EXECUTION_AUTHORITY_DOMAIN + b"\x00" + _canonical_json(payload)
    ).hexdigest()


def _initialize_execution_authority_api(
    executor_type: type[Any],
    material_resolver: Callable[[], tuple[Any, tuple[tuple[str, str], ...]]],
    authority_hash: Callable[[str, bool, Sequence[Any], Any], str],
    capture_behavior: Callable[[Any], _CallableBehaviorSeal],
    behavior_matches: Callable[[Any, _CallableBehaviorSeal], bool],
    fail_function: Callable[[str, str], None],
):
    """Create the execution authority API with lexical, one-use trust state."""

    # Module-name rebinding cannot replace the captured runtime verifier,
    # receipt consumer, authority type, or issuance state. Direct closure,
    # registry, class-method, descriptor, and code-object mutation is outside
    # this local same-process integrity boundary.
    receipt_lock = threading.RLock()
    receipt_records: dict[int, tuple[Any, ...]] = {}
    issuance_lock = threading.Lock()
    consumed_capabilities: set[object] = set()

    @dataclass(frozen=True, slots=True)
    class _ExecutionIssuance:
        authority: Any
        executors: Any
        eager: M6WaymaxCompactExecutor
        jit_eager: M6WaymaxCompactExecutor
        jit_compiled: M6WaymaxCompactExecutor
        runtime_facts: tuple[tuple[str, str], ...]
        eager_behavior: _CallableBehaviorSeal
        jit_eager_behavior: _CallableBehaviorSeal
        jit_compiled_behavior: _CallableBehaviorSeal
        authority_sha256: str
        one_shot_capability: object

    issuances: dict[int, _ExecutionIssuance] = {}
    dependencies = (
        executor_type,
        material_resolver,
        authority_hash,
        capture_behavior,
        behavior_matches,
        fail_function,
    )

    def consume(
        receipt: object,
        kind: Literal["pinned_cpu_waymax_jax", "test_only"],
        promotable: bool,
        executors: Any,
        facts: tuple[tuple[str, str], ...],
    ) -> None:
        with receipt_lock:
            record = receipt_records.pop(id(receipt), None)
        pinned_proof_invalid = (
            kind == "pinned_cpu_waymax_jax"
            and (
                type(record[6]) is not tuple
                or len(record[6]) != 2
                or record[6][0] is not executors
                or record[6][1] is not facts
            )
        ) if record is not None and len(record) == 7 else True
        test_proof_invalid = (
            kind == "test_only"
            and record is not None
            and len(record) == 7
            and record[6] is not None
        )
        if (
            record is None
            or len(record) != 7
            or record[0] is not receipt
            or record[1] != kind
            or record[2] is not promotable
            or record[3] is not executors
            or record[4] is not facts
            or record[5] != facts
            or pinned_proof_invalid
            or test_proof_invalid
        ):
            raise TypeError(
                "M6WaymaxExecutionAuthority is factory-issued only"
            )

    @dataclass(frozen=True, slots=True)
    class ExecutionAuthority:
        """Factory-issued pinned-runtime or test-only execution authority."""

        kind: Literal["pinned_cpu_waymax_jax", "test_only"]
        promotable: bool
        _executors: Any = field(repr=False, compare=False)
        _runtime_facts: tuple[tuple[str, str], ...] = field(
            repr=False,
            compare=False,
        )
        authority_sha256: str | None = field(default=None, repr=False)
        _issued_original_sha256: str = field(
            init=False,
            repr=False,
            compare=False,
        )
        _issued_executors: Any = field(
            init=False,
            repr=False,
            compare=False,
        )
        _one_shot_capability: object = field(
            init=False,
            repr=False,
            compare=False,
        )
        _issuance_receipt: InitVar[object] = None
        _consumed: bool = field(
            init=False,
            default=False,
            repr=False,
            compare=False,
        )

        def __post_init__(self, _issuance_receipt: object) -> None:
            facts = tuple(self._runtime_facts)
            consume(
                _issuance_receipt,
                self.kind,
                self.promotable,
                self._executors,
                facts,
            )
            if type(self._executors) is not dependencies[0]:
                raise TypeError(
                    "execution authority requires exact typed executors"
                )
            if self.kind == "pinned_cpu_waymax_jax":
                if self.promotable is not True or not facts:
                    raise ValueError(
                        "pinned runtime authority must be promotable"
                    )
            elif self.kind == "test_only":
                if self.promotable is not False or facts:
                    raise ValueError(
                        "test runtime authority is permanently non-promotable"
                    )
            else:
                raise ValueError("execution authority kind is not registered")
            object.__setattr__(self, "_runtime_facts", facts)
            object.__setattr__(self, "_issued_executors", self._executors)
            one_shot_capability = object()
            object.__setattr__(
                self,
                "_one_shot_capability",
                one_shot_capability,
            )
            expected = dependencies[2](
                self.kind,
                self.promotable,
                facts,
                self._executors,
            )
            if (
                self.authority_sha256 is not None
                and self.authority_sha256 != expected
            ):
                raise ValueError(
                    "authority_sha256 does not bind execution authority"
                )
            object.__setattr__(self, "authority_sha256", expected)
            object.__setattr__(self, "_issued_original_sha256", expected)
            record = _ExecutionIssuance(
                authority=self,
                executors=self._executors,
                eager=self._executors.eager,
                jit_eager=self._executors.jit_eager,
                jit_compiled=self._executors.jit_compiled,
                runtime_facts=self._runtime_facts,
                eager_behavior=dependencies[3](self._executors.eager),
                jit_eager_behavior=dependencies[3](
                    self._executors.jit_eager
                ),
                jit_compiled_behavior=dependencies[3](
                    self._executors.jit_compiled
                ),
                authority_sha256=expected,
                one_shot_capability=one_shot_capability,
            )
            with issuance_lock:
                if id(self) in issuances:
                    raise RuntimeError(
                        "execution authority issuance identity was reused"
                    )
                issuances[id(self)] = record

        def revalidate(self) -> None:
            with issuance_lock:
                record = issuances.get(id(self))
                consumed = (
                    record is not None
                    and record.one_shot_capability
                    in consumed_capabilities
                )
            if (
                record is None
                or record.authority is not self
                or self._executors is not record.executors
                or self._issued_executors is not record.executors
                or self._executors.eager is not record.eager
                or self._executors.jit_eager is not record.jit_eager
                or self._executors.jit_compiled is not record.jit_compiled
                or not dependencies[4](
                    self._executors.eager,
                    record.eager_behavior,
                )
                or not dependencies[4](
                    self._executors.jit_eager,
                    record.jit_eager_behavior,
                )
                or not dependencies[4](
                    self._executors.jit_compiled,
                    record.jit_compiled_behavior,
                )
                or self._runtime_facts is not record.runtime_facts
                or self._one_shot_capability
                is not record.one_shot_capability
                or type(self._consumed) is not bool
                or self._consumed is not consumed
            ):
                dependencies[5](
                    "execution_authority_mutated",
                    "execution authority identity changed",
                )
            expected = dependencies[2](
                self.kind,
                self.promotable,
                self._runtime_facts,
                self._executors,
            )
            if (
                expected != self.authority_sha256
                or expected != self._issued_original_sha256
                or expected != record.authority_sha256
            ):
                dependencies[5](
                    "execution_authority_mutated",
                    "execution authority binding changed",
                )

        def _claim(self) -> None:
            self.revalidate()
            with issuance_lock:
                record = issuances.get(id(self))
                if record is None or record.authority is not self:
                    dependencies[5](
                        "execution_authority_mutated",
                        "authority issuance disappeared",
                    )
                capability = record.one_shot_capability
                if capability in consumed_capabilities:
                    dependencies[5](
                        "execution_authority_consumed",
                        "execution authority can authorize exactly one run",
                    )
                consumed_capabilities.add(capability)
                object.__setattr__(self, "_consumed", True)

        def _get_executors(self) -> Any:
            self.revalidate()
            return self._executors

    ExecutionAuthority.__name__ = "M6WaymaxExecutionAuthority"
    ExecutionAuthority.__qualname__ = "M6WaymaxExecutionAuthority"
    factory_dependencies = (
        ExecutionAuthority,
        executor_type,
        material_resolver,
    )

    def build_test(executors: Any):
        """Issue exact nonpromotable authority for invented executors."""

        if type(executors) is not factory_dependencies[1]:
            raise TypeError(
                "executors must be exact M6WaymaxOfficialExecutors"
            )
        facts: tuple[tuple[str, str], ...] = ()
        receipt = object()
        with receipt_lock:
            receipt_records[id(receipt)] = (
                receipt,
                "test_only",
                False,
                executors,
                facts,
                facts,
                None,
            )
            try:
                authority = factory_dependencies[0](
                    kind="test_only",
                    promotable=False,
                    _executors=executors,
                    _runtime_facts=facts,
                    _issuance_receipt=receipt,
                )
                if type(authority) is not factory_dependencies[0]:
                    raise TypeError(
                        "execution authority factory returned wrong type"
                    )
                authority.revalidate()
                return authority
            finally:
                receipt_records.pop(id(receipt), None)

    def build_pinned():
        """Verify pinned runtime material before issuing production authority."""

        material = factory_dependencies[2]()
        if type(material) is not tuple or len(material) != 2:
            raise TypeError("verified runtime returned invalid material")
        executors, facts = material
        if type(executors) is not factory_dependencies[1]:
            raise TypeError("verified runtime returned invalid executors")
        if type(facts) is not tuple:
            raise TypeError("verified runtime returned invalid facts")
        content = tuple(tuple(value for value in row) for row in facts)
        receipt = object()
        with receipt_lock:
            receipt_records[id(receipt)] = (
                receipt,
                "pinned_cpu_waymax_jax",
                True,
                executors,
                facts,
                content,
                material,
            )
            try:
                authority = factory_dependencies[0](
                    kind="pinned_cpu_waymax_jax",
                    promotable=True,
                    _executors=executors,
                    _runtime_facts=facts,
                    _issuance_receipt=receipt,
                )
                if type(authority) is not factory_dependencies[0]:
                    raise TypeError(
                        "execution authority factory returned wrong type"
                    )
                authority.revalidate()
                return authority
            finally:
                receipt_records.pop(id(receipt), None)

    return ExecutionAuthority, build_test, build_pinned




def _block_compact(jax: Any, compact: Any) -> CompactM6WaymaxRollout:
    leaves = []
    for name in CompactM6WaymaxRollout._fields:
        value = jax.block_until_ready(getattr(compact, name))
        leaves.append(np.array(value, copy=True, order="C"))
    return CompactM6WaymaxRollout(*leaves)


def _validate_runtime_execution_member(
    state: Any,
    scenario: Scenario,
    selection: M6WaymaxSelection,
    primary_domain: M6WaymaxPrimaryDomain,
    selection_position: int,
) -> None:
    selection.revalidate(primary_domain=primary_domain)
    position = _strict_int(selection_position, "selection_position")
    if not selection.supported or position >= len(selection.members):
        _fail("runtime_selection", "runtime execution is outside supported selection")
    member = selection.members[position]
    if (
        member.scenario_id != scenario.scenario_id
        or member.source_binding_sha256 != source_state_mutation_sha256(state)
    ):
        _fail("runtime_selection", "runtime source differs from selected position")


def _validate_waymax_direct_url(
    direct_url: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    try:
        url = direct_url["url"]
        vcs_info = direct_url["vcs_info"]
        vcs = vcs_info["vcs"]
        requested_revision = vcs_info["requested_revision"]
        commit = vcs_info["commit_id"]
    except (KeyError, TypeError) as exc:
        raise M6WaymaxOfficialError(
            "runtime_waymax_pin: Waymax direct-url provenance is unavailable"
        ) from exc
    if (
        url != _WAYMAX_CANONICAL_GIT_URL
        or vcs != "git"
        or requested_revision != WAYMAX_COMMIT
        or commit != WAYMAX_COMMIT
    ):
        _fail(
            "runtime_waymax_pin",
            "installed Waymax direct-url provenance differs from exact pin",
        )
    return url, vcs, requested_revision, commit


class _M6WaymaxJitKernelCache:
    """Explicit identity-bound mutable state for verified JIT compilation."""

    __slots__ = ("kernels",)

    def __init__(self) -> None:
        self.kernels: dict[tuple[str, str], Any] = {}


def _verified_pinned_m6_waymax_execution_material() -> tuple[
    M6WaymaxOfficialExecutors, tuple[tuple[str, str], ...]
]:
    """Verify the exact CPU runtime and return its executor material."""

    try:
        import jax
        import jax.numpy as jnp
        import jaxlib
        import waymax  # noqa: F401
    except ImportError as exc:
        raise M6WaymaxOfficialError(
            "runtime_unavailable: pinned Waymax/JAX runtime is unavailable"
        ) from exc
    observed = {
        "jax": str(jax.__version__),
        "jaxlib": str(jaxlib.__version__),
        "waymo-waymax": importlib.metadata.version("waymo-waymax"),
    }
    if observed != _PINNED_RUNTIME_VERSIONS:
        _fail("runtime_version", "optional runtime versions differ from pins")
    devices = tuple(jax.devices())
    if (
        jax.default_backend() != "cpu"
        or not devices
        or any(getattr(device, "platform", None) != "cpu" for device in devices)
        or bool(jax.config.jax_disable_jit)
    ):
        _fail("runtime_cpu", "promotable execution requires enabled JIT on CPU")
    distribution = importlib.metadata.distribution("waymo-waymax")
    try:
        direct_url = json.loads(distribution.read_text("direct_url.json") or "")
    except Exception as exc:
        raise M6WaymaxOfficialError(
            "runtime_waymax_pin: Waymax direct-url provenance is unavailable"
        ) from exc
    url, vcs, requested_revision, commit = _validate_waymax_direct_url(
        direct_url
    )

    cache = _M6WaymaxJitKernelCache()

    def eager(
        state: Any,
        scenario: Scenario,
        plan: EgoTrajectoryPlan,
        *,
        bundle: str,
        selection: M6WaymaxSelection,
        primary_domain: M6WaymaxPrimaryDomain,
        selection_position: int,
    ) -> tuple[CompactM6WaymaxRollout, WaymaxEgoPlanView]:
        return compact_selected_m6_waymax_rollout(
            state,
            scenario,
            plan,
            bundle=bundle,
            selection=selection,
            primary_domain=primary_domain,
            selection_position=selection_position,
        )

    def execute_jit(
        trace: bool,
        state: Any,
        scenario: Scenario,
        plan: EgoTrajectoryPlan,
        *,
        bundle: str,
        selection: M6WaymaxSelection,
        primary_domain: M6WaymaxPrimaryDomain,
        selection_position: int,
    ) -> tuple[CompactM6WaymaxRollout, WaymaxEgoPlanView]:
        _validate_runtime_execution_member(
            state,
            scenario,
            selection,
            primary_domain,
            selection_position,
        )
        plan.revalidate()
        view = build_waymax_ego_plan_view(state, scenario, plan)
        key = (bundle, plan.configuration_fingerprint)
        kernel = (
            single_scene_m6_logged_world_kernel
            if bundle == M6_WAYMAX_LOGGED_WORLD
            else single_scene_m6_idm_kernel
        )
        if bundle not in M6_WAYMAX_BUNDLES:
            raise ValueError("bundle is not registered")
        if trace:
            if key in cache.kernels:
                _fail("runtime_jit_trace", "JIT trace key was reused")
            cache.kernels[key] = jax.jit(kernel)
        elif key not in cache.kernels:
            _fail("runtime_jit_compiled", "compiled JIT called before trace")
        compact = cache.kernels[key](state, jnp.asarray(view.future_action_data))
        return _block_compact(jax, compact), view

    def jit_eager(*args: Any, **kwargs: Any):
        return execute_jit(True, *args, **kwargs)

    def jit_compiled(*args: Any, **kwargs: Any):
        return execute_jit(False, *args, **kwargs)

    executors = M6WaymaxOfficialExecutors(
        eager=eager,
        jit_eager=jit_eager,
        jit_compiled=jit_compiled,
    )
    facts = tuple(
        sorted(
            {
                **observed,
                "backend": "cpu",
                "device_count": str(len(devices)),
                "waymax_requested_revision": requested_revision,
                "waymax_url": url,
                "waymax_vcs": vcs,
                "waymax_commit": commit,
            }.items()
        )
    )
    return executors, facts


@dataclass(frozen=True, slots=True)
class _Comparison:
    denominator: int
    maximum_absolute_error: float
    maximum_normalized_error: float | None
    tolerance_failures: int
    binary_mismatches: int


(
    M6WaymaxExecutionAuthority,
    build_m6_waymax_test_execution_authority,
    build_pinned_m6_waymax_execution_authority,
) = _initialize_execution_authority_api(
    M6WaymaxOfficialExecutors,
    _verified_pinned_m6_waymax_execution_material,
    _execution_authority_sha256,
    _capture_callable_behavior,
    _callable_behavior_matches,
    _fail,
)
del _initialize_execution_authority_api


def _binary_comparison(actual: Any, expected: Any) -> _Comparison:
    left = np.asarray(actual)
    right = np.asarray(expected)
    if left.shape != right.shape:
        _fail("field_comparison_shape", "exact comparison shapes differ")
    mismatch = left != right
    mismatch_count = int(np.count_nonzero(mismatch))
    maximum = 0.0
    if mismatch_count:
        if np.issubdtype(left.dtype, np.number) and np.issubdtype(
            right.dtype, np.number
        ):
            differences = np.abs(
                left.astype(np.float64) - right.astype(np.float64)
            )
            finite = differences[np.isfinite(differences)]
            maximum = (
                float(np.max(finite))
                if finite.size
                else float(np.finfo(np.float64).max)
            )
        else:
            maximum = 1.0
    return _Comparison(
        denominator=int(left.size),
        maximum_absolute_error=maximum,
        maximum_normalized_error=None,
        tolerance_failures=0,
        binary_mismatches=mismatch_count,
    )


def _merge_exact(*comparisons: _Comparison) -> _Comparison:
    return _Comparison(
        denominator=sum(item.denominator for item in comparisons),
        maximum_absolute_error=max(
            (item.maximum_absolute_error for item in comparisons),
            default=0.0,
        ),
        maximum_normalized_error=None,
        tolerance_failures=0,
        binary_mismatches=sum(item.binary_mismatches for item in comparisons),
    )


def _typed_exact(value: Any) -> _Comparison:
    return _Comparison(
        denominator=value.denominator,
        maximum_absolute_error=value.maximum_absolute_error,
        maximum_normalized_error=None,
        tolerance_failures=value.tolerance_failure_count,
        binary_mismatches=value.binary_mismatch_count,
    )


def _numeric_comparison(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    circular: bool,
) -> _Comparison:
    left = np.asarray(actual, dtype=np.float64).reshape(-1)
    right = np.asarray(expected, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < M6_WAYMAX_TRANSITIONS:
        _fail(
            "field_comparison_shape",
            "tolerance comparison must cover at least 20 values",
        )
    finite = np.isfinite(left) & np.isfinite(right)
    if circular:
        error = np.abs((left - right + np.pi) % (2.0 * np.pi) - np.pi)
        tolerance = np.full_like(error, M6_WAYMAX_YAW_ATOL)
    else:
        error = np.abs(left - right)
        tolerance = M6_WAYMAX_FLOAT_ATOL + (
            M6_WAYMAX_FLOAT_RTOL * np.abs(right)
        )
    normalized = np.where(finite, error / tolerance, np.inf)
    failures = int(np.count_nonzero(normalized > 1.0))
    finite_error = error[np.isfinite(error) & finite]
    finite_normalized = normalized[np.isfinite(normalized)]
    maximum = (
        float(np.max(finite_error))
        if finite_error.size
        else float(np.finfo(np.float64).max)
    )
    maximum_normalized = (
        float(np.max(finite_normalized))
        if finite_normalized.size
        else float(np.finfo(np.float64).max)
    )
    return _Comparison(
        denominator=int(left.size),
        maximum_absolute_error=maximum,
        maximum_normalized_error=maximum_normalized,
        tolerance_failures=failures,
        binary_mismatches=0,
    )


def _compact_arrays(compact: CompactM6WaymaxRollout) -> dict[str, np.ndarray]:
    if not isinstance(compact, CompactM6WaymaxRollout):
        raise TypeError("executor must return CompactM6WaymaxRollout")
    return {
        name: np.asarray(getattr(compact, name))
        for name in CompactM6WaymaxRollout._fields
    }


def _clone_compact(compact: CompactM6WaymaxRollout) -> CompactM6WaymaxRollout:
    arrays = _compact_arrays(compact)
    copied: list[np.ndarray] = []
    for name in CompactM6WaymaxRollout._fields:
        value = np.array(arrays[name], copy=True, order="C")
        value.setflags(write=False)
        copied.append(value)
    return CompactM6WaymaxRollout(*copied)


def _compact_value_sha256(compact: CompactM6WaymaxRollout) -> str:
    digest = hashlib.sha256()
    digest.update(b"evalsim-m6-waymax-pilot-compact-v1\x00")
    arrays = _compact_arrays(compact)
    for name in CompactM6WaymaxRollout._fields:
        _hash_named_array(digest, name, arrays[name])
    return digest.hexdigest()


def _tree_array_leaves(value: Any) -> tuple[Any, ...]:
    leaves: list[Any] = []
    seen: set[int] = set()

    def visit(candidate: Any) -> None:
        identity = id(candidate)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(candidate, (str, bytes, int, float, bool, type(None))):
            return
        if isinstance(candidate, np.ndarray) or (
            hasattr(candidate, "shape")
            and hasattr(candidate, "dtype")
            and not dataclasses.is_dataclass(candidate)
        ):
            leaves.append(candidate)
            return
        if dataclasses.is_dataclass(candidate):
            for item in dataclasses.fields(candidate):
                visit(getattr(candidate, item.name))
            return
        if isinstance(candidate, Mapping):
            for item in candidate.values():
                visit(item)
            return
        if isinstance(candidate, (tuple, list)):
            for item in candidate:
                visit(item)

    visit(value)
    return tuple(leaves)


def _storage_tokens(value: Any) -> set[tuple[str, int]]:
    tokens = {("object", id(value))}
    candidate = value
    seen: set[int] = set()
    while isinstance(candidate, np.ndarray) and id(candidate) not in seen:
        seen.add(id(candidate))
        tokens.add(("base", id(candidate)))
        candidate = candidate.base
    pointer = getattr(value, "unsafe_buffer_pointer", None)
    if callable(pointer):
        try:
            tokens.add(("pointer", int(pointer())))
        except Exception:
            pass
    try:
        array = np.asarray(value)
        data = array.__array_interface__.get("data")
        if data is not None:
            tokens.add(("pointer", int(data[0])))
    except Exception:
        pass
    return tokens


def _leaves_share_storage(left: Any, right: Any) -> bool:
    if _storage_tokens(left) & _storage_tokens(right):
        return True
    try:
        return bool(np.shares_memory(np.asarray(left), np.asarray(right)))
    except Exception:
        return False


def _trees_share_storage(left: Any, right: Any) -> bool:
    left_leaves = _tree_array_leaves(left)
    right_leaves = _tree_array_leaves(right)
    return any(
        _leaves_share_storage(left_leaf, right_leaf)
        for left_leaf in left_leaves
        for right_leaf in right_leaves
    )


def _compacts_share_storage(
    left: CompactM6WaymaxRollout,
    right: CompactM6WaymaxRollout,
) -> bool:
    return _trees_share_storage(left, right)


def _field_comparisons(
    compact: CompactM6WaymaxRollout,
    *,
    state: Any,
    scenario: Scenario,
    plan: EgoTrajectoryPlan,
    view: WaymaxEgoPlanView,
    bundle: str,
    qualification: M6WaymaxEligibility,
    primary_domain: M6WaymaxPrimaryDomain,
) -> Mapping[str, _Comparison]:
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
    rollout, rollout_validation = m6_waymax_to_rollout(
        compact,
        state=state,
        scenario=scenario,
        plan=plan,
        view=view,
        bundle=bundle,
        qualification=qualification,
        primary_domain=primary_domain,
    )
    rollout_validation.require_passed()
    retained_slots = np.flatnonzero(
        np.any(np.asarray(state.log_trajectory.valid, dtype=bool), axis=1)
    )
    source_ids = np.asarray(state.object_metadata.ids)[retained_slots]
    scenario_ids = np.asarray(
        [agent.id for agent in scenario.agents],
        dtype=source_ids.dtype,
    )
    identity_mismatches = int(np.count_nonzero(source_ids != scenario_ids))
    if any(
        source.id != output.id
        or source.type != output.type
        or source.length != output.length
        or source.width != output.width
        for source, output in zip(scenario.agents, rollout.agents, strict=True)
    ):
        _fail(
            "field_agent_identity",
            "reconstructed rollout changed source-neutral agent identity",
        )
    identity = _Comparison(
        denominator=max(
            M6_WAYMAX_TRANSITIONS,
            M6_WAYMAX_TRANSITIONS * len(scenario.agents),
        ),
        maximum_absolute_error=1.0 if identity_mismatches else 0.0,
        maximum_normalized_error=None,
        tolerance_failures=0,
        binary_mismatches=M6_WAYMAX_TRANSITIONS * identity_mismatches,
    )
    components = validation.components
    timestamps = _merge_exact(
        _typed_exact(components["identity.timestep"]),
        _typed_exact(components["identity.timestamp_micros"]),
    )
    validity = _merge_exact(_typed_exact(components["lifecycle.valid"]))
    actor = _merge_exact(
        _typed_exact(components["actor.requested_control"]),
        _typed_exact(components["actor.effective_control"]),
    )
    lifecycle = _merge_exact(
        _typed_exact(components["actor.lifecycle_fallback"]),
        _typed_exact(components["actor.initialized_overlap_excluded"]),
    )

    arrays = _compact_arrays(compact)
    current = view.current_index
    interval = slice(
        current + 1,
        current + 1 + M6_WAYMAX_TRANSITIONS,
    )
    logged = state.log_trajectory
    expected_valid = np.asarray(logged.valid, dtype=bool)[:, interval].T
    world_slots = np.ones(expected_valid.shape[1], dtype=bool)
    world_slots[view.ego_slot] = False
    if bundle == M6_WAYMAX_LOGGED_WORLD:
        fallback_mask = expected_valid & world_slots[np.newaxis, :]
    elif bundle == M6_WAYMAX_PRIVILEGED_IDM:
        fallback_mask = (
            expected_valid
            & world_slots[np.newaxis, :]
            & ~np.asarray(arrays["effective_control"], dtype=bool)
        )
    else:
        raise ValueError("bundle is not registered")
    source_fields = {
        "x": "x",
        "y": "y",
        "vx": "vel_x",
        "vy": "vel_y",
        "heading": "yaw",
    }
    view_fields = {
        "x": "x",
        "y": "y",
        "vx": "vx",
        "vy": "vy",
        "heading": "heading",
    }
    compact_fields = {
        **{name: name for name in ("x", "y", "vx", "vy")},
        "heading": "yaw",
    }
    numeric: dict[str, _Comparison] = {}
    for field_name in ("x", "y", "vx", "vy", "heading"):
        compact_name = compact_fields[field_name]
        expected_logged = np.asarray(
            getattr(logged, source_fields[field_name]),
            dtype=np.float64,
        )[:, interval].T
        ego_actual = np.asarray(arrays[compact_name])[:, view.ego_slot]
        ego_expected = np.asarray(
            getattr(view, view_fields[field_name]),
            dtype=np.float64,
        )[1:]
        actual = np.concatenate((ego_actual, arrays[compact_name][fallback_mask]))
        expected = np.concatenate(
            (ego_expected, expected_logged[fallback_mask])
        )
        comparison = _numeric_comparison(
            actual,
            expected,
            circular=field_name == "heading",
        )
        component_name = "yaw" if field_name == "heading" else field_name
        typed_ego = components[f"ego_plan.{component_name}"]
        typed_fallback = components[f"logged_fallback.{component_name}"]
        if (
            comparison.denominator
            != typed_ego.denominator + typed_fallback.denominator
            or comparison.tolerance_failures
            != typed_ego.tolerance_failure_count
            + typed_fallback.tolerance_failure_count
            or not math.isclose(
                comparison.maximum_absolute_error,
                max(
                    typed_ego.maximum_absolute_error,
                    typed_fallback.maximum_absolute_error,
                ),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            _fail(
                "field_comparison_crosscheck",
                "elementwise tolerance evidence differs from adapter validation",
            )
        numeric[field_name] = comparison
    return MappingProxyType(
        {
            "agent_identity": identity,
            "timestamps": timestamps,
            "validity": validity,
            "actor_mask": actor,
            "lifecycle_category": lifecycle,
            **numeric,
        }
    )


@dataclass(frozen=True, slots=True)
class M6WaymaxOfficialFieldComparisonRow:
    """Factory-issued safe projection of one validated field comparison."""

    selection_position: int
    bundle: str
    condition: str
    field_name: str
    cohort_index: int | None
    qualification_binding_sha256: str | None
    comparison_kind: Literal["exact", "tolerance"]
    denominator: int | None
    max_abs_error: float | None
    max_normalized_error: float | None
    tolerance_failures: int | None
    binary_mismatches: int | None
    status: Literal["passed", "failed", "not_applicable"]
    row_binding_sha256: str | None = field(default=None, repr=False)
    _issued_original_binding_sha256: str = field(
        init=False, repr=False, compare=False
    )
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _FIELD_ISSUER:
            raise TypeError(
                "M6WaymaxOfficialFieldComparisonRow is factory-issued only"
            )
        object.__setattr__(
            self,
            "selection_position",
            _strict_int(self.selection_position, "selection_position"),
        )
        self._validate_semantics()
        expected = self._binding()
        if self.row_binding_sha256 is not None and self.row_binding_sha256 != expected:
            raise ValueError("row_binding_sha256 does not bind this row")
        object.__setattr__(self, "row_binding_sha256", expected)
        object.__setattr__(self, "_issued_original_binding_sha256", expected)
        _register_strong_issuance(
            "field_comparison_row",
            self,
            binding_sha256=expected,
            content_sha256=expected,
        )

    def _validate_semantics(self) -> None:
        if self.selection_position >= M6_WAYMAX_MAX_SCENES:
            raise ValueError("selection_position must lie in [0, 15]")
        if self.bundle not in M6_WAYMAX_BUNDLES:
            raise ValueError("bundle is not registered")
        if self.condition not in M6_WAYMAX_OFFICIAL_CONDITIONS:
            raise ValueError("condition is not registered")
        if self.field_name not in M6_WAYMAX_OFFICIAL_COMPARISON_FIELDS:
            raise ValueError("field_name is not registered")
        expected_kind = (
            "exact"
            if self.field_name in M6_WAYMAX_OFFICIAL_EXACT_FIELDS
            else "tolerance"
        )
        if self.comparison_kind != expected_kind:
            raise ValueError("comparison_kind differs from field registration")
        evidence = (
            self.denominator,
            self.max_abs_error,
            self.max_normalized_error,
            self.tolerance_failures,
            self.binary_mismatches,
        )
        if self.status == "not_applicable":
            if (
                self.cohort_index is not None
                or self.qualification_binding_sha256 is not None
                or any(value is not None for value in evidence)
            ):
                raise ValueError("not-applicable field rows must be exact NA")
            return
        if self.status not in ("passed", "failed") or self.cohort_index is None:
            raise ValueError("executed field rows require selected provenance")
        _strict_int(self.cohort_index, "cohort_index")
        _sha256(
            self.qualification_binding_sha256,
            "qualification_binding_sha256",
        )
        denominator = _strict_int(self.denominator, "denominator")
        tolerance = _strict_int(self.tolerance_failures, "tolerance_failures")
        binary = _strict_int(self.binary_mismatches, "binary_mismatches")
        if denominator < M6_WAYMAX_TRANSITIONS:
            raise ValueError("executed field denominator must be at least 20")
        if tolerance > denominator or binary > denominator:
            raise ValueError("field failures exceed denominator")
        maximum = float(self.max_abs_error)
        if not math.isfinite(maximum) or maximum < 0.0:
            raise ValueError("max_abs_error must be finite and non-negative")
        if self.comparison_kind == "exact":
            if tolerance != 0 or self.max_normalized_error is not None:
                raise ValueError(
                    "exact rows cannot carry normalized/tolerance evidence"
                )
            if (binary == 0) != (maximum == 0.0):
                raise ValueError("exact mismatch and maximum contradict")
        else:
            normalized = float(self.max_normalized_error)
            if binary != 0 or not math.isfinite(normalized) or normalized < 0.0:
                raise ValueError("tolerance row evidence is invalid")
            if (maximum == 0.0) != (normalized == 0.0):
                raise ValueError("absolute and normalized maxima contradict")
            if (tolerance == 0 and normalized > 1.0) or (
                tolerance > 0 and normalized <= 1.0
            ):
                raise ValueError("normalized maximum contradicts failures")
        expected_status = "passed" if tolerance == 0 and binary == 0 else "failed"
        if self.status != expected_status:
            raise ValueError("status differs from failure counts")

    def to_store_dict(self) -> dict[str, Any]:
        return {
            "selection_position": self.selection_position,
            "bundle": self.bundle,
            "condition": self.condition,
            "field_name": self.field_name,
            "cohort_index": self.cohort_index,
            "qualification_binding_sha256": self.qualification_binding_sha256,
            "comparison_kind": self.comparison_kind,
            "denominator": self.denominator,
            "max_abs_error": self.max_abs_error,
            "max_normalized_error": self.max_normalized_error,
            "tolerance_failures": self.tolerance_failures,
            "binary_mismatches": self.binary_mismatches,
            "status": self.status,
        }

    def _binding(self) -> str:
        return hashlib.sha256(
            _FIELD_ROW_DOMAIN + b"\x00" + _canonical_json(self.to_store_dict())
        ).hexdigest()

    def revalidate(self) -> None:
        record = _strong_issuance_record(
            "field_comparison_row",
            self,
            error_code="field_row_mutated",
        )
        try:
            self._validate_semantics()
            expected = self._binding()
            if (
                expected != self.row_binding_sha256
                or expected != self._issued_original_binding_sha256
                or expected != record.binding_sha256
                or expected != record.content_sha256
            ):
                raise ValueError("field row digest changed")
        except (TypeError, ValueError) as exc:
            raise M6WaymaxOfficialError(
                "field_row_mutated: field row failed its local binding"
            ) from exc


def _field_keys() -> tuple[tuple[int, str, str, str], ...]:
    return tuple(
        (position, bundle, condition, field_name)
        for position in range(M6_WAYMAX_MAX_SCENES)
        for bundle in M6_WAYMAX_BUNDLES
        for condition in M6_WAYMAX_OFFICIAL_CONDITIONS
        for field_name in M6_WAYMAX_OFFICIAL_COMPARISON_FIELDS
    )


@dataclass(frozen=True, slots=True)
class M6WaymaxOfficialFieldComparisonTable(
    Sequence[M6WaymaxOfficialFieldComparisonRow]
):
    """Selection-bound fixed 640-row live or unsupported field table."""

    selection_supported: bool
    selected_member_count: int
    selection_sha256: str
    primary_domain_sha256: str
    rows: tuple[M6WaymaxOfficialFieldComparisonRow, ...]
    promotable: bool
    table_binding_sha256: str | None = field(default=None, repr=False)
    _issued_original_binding_sha256: str = field(
        init=False, repr=False, compare=False
    )
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _FIELD_ISSUER:
            raise TypeError(
                "M6WaymaxOfficialFieldComparisonTable is factory-issued only"
            )
        if (
            type(self.selection_supported) is not bool
            or type(self.promotable) is not bool
        ):
            raise TypeError("selection_supported/promotable must be booleans")
        count = _strict_int(self.selected_member_count, "selected_member_count")
        object.__setattr__(self, "selected_member_count", count)
        _sha256(self.selection_sha256, "selection_sha256")
        _sha256(self.primary_domain_sha256, "primary_domain_sha256")
        rows = tuple(self.rows)
        if len(rows) != M6_WAYMAX_OFFICIAL_FIELD_ROW_COUNT or any(
            not isinstance(row, M6WaymaxOfficialFieldComparisonRow) for row in rows
        ):
            raise ValueError("field table must contain 640 factory-issued rows")
        for row in rows:
            row.revalidate()
        keys = tuple(
            (
                row.selection_position,
                row.bundle,
                row.condition,
                row.field_name,
            )
            for row in rows
        )
        if keys != _field_keys():
            raise ValueError("field rows are not in canonical 16x2x2x10 order")
        if self.selection_supported:
            if not 8 <= count <= M6_WAYMAX_MAX_SCENES:
                raise ValueError("supported field tables require 8..16 members")
        elif count != 0 or self.promotable:
            raise ValueError(
                "unsupported field tables must be non-promotable and empty"
            )
        for row in rows:
            selected = row.selection_position < count
            if selected != (row.status != "not_applicable"):
                raise ValueError("field selected/NA layout differs from selection")
        object.__setattr__(self, "rows", rows)
        expected = self._binding(rows)
        if (
            self.table_binding_sha256 is not None
            and self.table_binding_sha256 != expected
        ):
            raise ValueError("table_binding_sha256 does not bind rows")
        object.__setattr__(self, "table_binding_sha256", expected)
        object.__setattr__(self, "_issued_original_binding_sha256", expected)
        _register_strong_issuance(
            "field_comparison_table",
            self,
            binding_sha256=expected,
            content_sha256=expected,
            components=(self.rows, *self.rows),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int | slice):
        return self.rows[index]

    def __iter__(self) -> Iterator[M6WaymaxOfficialFieldComparisonRow]:
        return iter(self.rows)

    def _binding(
        self,
        rows: Sequence[M6WaymaxOfficialFieldComparisonRow],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(_FIELD_TABLE_DOMAIN)
        digest.update(b"\x00")
        digest.update(b"\x01" if self.selection_supported else b"\x00")
        digest.update(b"\x01" if self.promotable else b"\x00")
        digest.update(self.selected_member_count.to_bytes(4, "big"))
        digest.update(bytes.fromhex(self.selection_sha256))
        digest.update(bytes.fromhex(self.primary_domain_sha256))
        for row in rows:
            assert row.row_binding_sha256 is not None
            digest.update(bytes.fromhex(row.row_binding_sha256))
        return digest.hexdigest()

    def to_store_rows(self) -> tuple[dict[str, Any], ...]:
        self.revalidate()
        return tuple(row.to_store_dict() for row in self.rows)

    def revalidate(
        self,
        *,
        selection: M6WaymaxSelection | None = None,
        primary_domain: M6WaymaxPrimaryDomain | None = None,
    ) -> None:
        record = _strong_issuance_record(
            "field_comparison_table",
            self,
            error_code="field_table_mutated",
        )
        _require_original_components(
            record,
            (self.rows, *self.rows),
            error_code="field_table_mutated",
        )
        if (
            type(self.selection_supported) is not bool
            or type(self.promotable) is not bool
        ):
            _fail(
                "field_table_mutated",
                "field table support/promotability flags changed",
            )
        if self.selection_supported:
            if not 8 <= self.selected_member_count <= M6_WAYMAX_MAX_SCENES:
                _fail(
                    "field_table_mutated",
                    "supported field table layout changed",
                )
        elif self.selected_member_count != 0 or self.promotable:
            _fail(
                "field_table_mutated",
                "unsupported field table layout changed",
            )
        for row in self.rows:
            row.revalidate()
            selected = row.selection_position < self.selected_member_count
            if selected != (row.status != "not_applicable"):
                _fail(
                    "field_table_mutated",
                    "field selected/NA layout changed",
                )
        expected = self._binding(self.rows)
        if (
            expected != self.table_binding_sha256
            or expected != self._issued_original_binding_sha256
            or expected != record.binding_sha256
            or expected != record.content_sha256
        ):
            _fail("field_table_mutated", "field table failed its local binding")
        if (selection is None) != (primary_domain is None):
            raise ValueError("selection and primary_domain must be supplied together")
        if selection is None:
            return
        assert primary_domain is not None
        selection.revalidate(primary_domain=primary_domain)
        if (
            selection.supported != self.selection_supported
            or (len(selection.members) if selection.supported else 0)
            != self.selected_member_count
            or selection.selection_sha256 != self.selection_sha256
            or primary_domain.domain_sha256 != self.primary_domain_sha256
        ):
            _fail("field_table_selection", "field table belongs to another selection")
        rows_per_position = (
            len(M6_WAYMAX_BUNDLES)
            * len(M6_WAYMAX_OFFICIAL_CONDITIONS)
            * len(M6_WAYMAX_OFFICIAL_COMPARISON_FIELDS)
        )
        for position, member in enumerate(selection.members):
            group = self.rows[
                position * rows_per_position : (position + 1) * rows_per_position
            ]
            if any(
                row.cohort_index != member.cohort_index
                or row.qualification_binding_sha256
                != member.qualification_binding_sha256
                for row in group
            ):
                _fail(
                    "field_table_selection",
                    "field row provenance differs from selected member",
                )


def _issue_field_table(
    evidence: Mapping[tuple[int, str, str, str], _Comparison],
    *,
    selection: M6WaymaxSelection,
    promotable: bool,
    primary_domain: M6WaymaxPrimaryDomain,
) -> M6WaymaxOfficialFieldComparisonTable:
    if type(promotable) is not bool:
        raise TypeError("promotable must be a boolean")
    selection.revalidate(primary_domain=primary_domain)
    expected_live_keys = {
        key for key in _field_keys() if key[0] < len(selection.members)
    }
    if set(evidence) != expected_live_keys:
        raise ValueError("field evidence does not cover exact selected grid")
    rows: list[M6WaymaxOfficialFieldComparisonRow] = []
    for key in _field_keys():
        position, bundle, condition, field_name = key
        comparison = evidence.get(key)
        kind = (
            "exact"
            if field_name in M6_WAYMAX_OFFICIAL_EXACT_FIELDS
            else "tolerance"
        )
        if comparison is None:
            row = M6WaymaxOfficialFieldComparisonRow(
                selection_position=position,
                bundle=bundle,
                condition=condition,
                field_name=field_name,
                cohort_index=None,
                qualification_binding_sha256=None,
                comparison_kind=kind,
                denominator=None,
                max_abs_error=None,
                max_normalized_error=None,
                tolerance_failures=None,
                binary_mismatches=None,
                status="not_applicable",
                _issuance_capability=_FIELD_ISSUER,
            )
        else:
            member = selection.members[position]
            status = (
                "passed"
                if comparison.tolerance_failures == 0
                and comparison.binary_mismatches == 0
                else "failed"
            )
            row = M6WaymaxOfficialFieldComparisonRow(
                selection_position=position,
                bundle=bundle,
                condition=condition,
                field_name=field_name,
                cohort_index=member.cohort_index,
                qualification_binding_sha256=(
                    member.qualification_binding_sha256
                ),
                comparison_kind=kind,
                denominator=comparison.denominator,
                max_abs_error=comparison.maximum_absolute_error,
                max_normalized_error=comparison.maximum_normalized_error,
                tolerance_failures=comparison.tolerance_failures,
                binary_mismatches=comparison.binary_mismatches,
                status=status,
                _issuance_capability=_FIELD_ISSUER,
            )
        rows.append(row)
    table = M6WaymaxOfficialFieldComparisonTable(
        selection_supported=selection.supported,
        selected_member_count=(len(selection.members) if selection.supported else 0),
        selection_sha256=selection.selection_sha256,
        primary_domain_sha256=primary_domain.domain_sha256,
        rows=tuple(rows),
        promotable=promotable,
        _issuance_capability=_FIELD_ISSUER,
    )
    table.revalidate(selection=selection, primary_domain=primary_domain)
    return table


def build_m6_waymax_unsupported_field_comparison_table(
    *,
    selection: M6WaymaxSelection,
    primary_domain: M6WaymaxPrimaryDomain,
) -> M6WaymaxOfficialFieldComparisonTable:
    """Issue strict NA only for an authentic unsupported canonical selection."""

    selection.revalidate(primary_domain=primary_domain)
    if selection.supported:
        raise ValueError("supported selection requires live field evidence")
    return _issue_field_table(
        {},
        selection=selection,
        primary_domain=primary_domain,
        promotable=False,
    )


def m6_waymax_selection_binding_sha256(selection: M6WaymaxSelection) -> str:
    selection.revalidate()
    payload = {
        "eligible_count": selection.eligible_count,
        "qualification_ledger_sha256": selection.qualification_ledger_sha256,
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
        "primary_domain_member_count": selection.primary_domain_member_count,
        "primary_domain_sha256": selection.primary_domain_sha256,
        "supported": selection.supported,
        "selector_selection_sha256": selection.selection_sha256,
    }
    return hashlib.sha256(
        b"evalsim-m6-waymax-selection-binding-v1\x00"
        + _canonical_json(payload)
    ).hexdigest()


def _hash_named_array(digest: Any, name: str, value: Any) -> None:
    array = np.ascontiguousarray(value)
    encoded_name = name.encode("ascii")
    encoded_dtype = array.dtype.str.encode("ascii")
    digest.update(len(encoded_name).to_bytes(4, "big"))
    digest.update(encoded_name)
    digest.update(len(encoded_dtype).to_bytes(4, "big"))
    digest.update(encoded_dtype)
    digest.update(array.ndim.to_bytes(4, "big"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    payload = array.tobytes(order="C")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _numpy_pair_view(
    scene_result: M6PairedSceneResult,
    *,
    selection_position: int,
    qualification: M6WaymaxEligibility,
    primary_domain: M6WaymaxPrimaryDomain,
    selection_binding_sha256: str,
    numpy_eligibility_sha256: str,
) -> tuple[str, Mapping[str, np.ndarray]]:
    scene_result.revalidate()
    pair = scene_result.pair
    pair.revalidate()
    source = pair.scenario
    source.revalidate()
    if (
        scene_result.cohort_index != qualification.cohort_index
        or scene_result.policy_name not in M6_WAYMAX_NUMPY_COMPARISON_POLICIES
        or scene_result.policy_access_role
        != _NUMPY_POLICY_ACCESS[scene_result.policy_name]
        or pair.eligibility.to_dict()
        != primary_domain.entry_by_cohort_index[
            qualification.cohort_index
        ].upstream_eligibility.to_dict()
        or pair.eligibility.target_index != qualification.target_index
        or qualification.target_agent_id
        != source.agents[pair.eligibility.target_index].id
    ):
        _fail(
            "numpy_comparison_pairing",
            "NumPy pair differs from selected Waymax source/target",
        )
    current = pair.eligibility.current_index
    start = current
    stop = current + M6_WAYMAX_TRANSITIONS + 1
    baseline = pair.baseline
    treatment = pair.intervention
    baseline.revalidate()
    treatment.revalidate()
    timestamps = np.rint(
        np.asarray(baseline.timestamps[start:stop]) * 1_000_000.0
    ).astype("<i8")
    if (
        timestamps.shape != (M6_WAYMAX_TRANSITIONS + 1,)
        or not np.array_equal(
            np.asarray(baseline.timestamps[start:stop]),
            np.asarray(treatment.timestamps[start:stop]),
        )
    ):
        _fail("numpy_comparison_timeline", "NumPy pair timeline differs")

    arrays: dict[str, np.ndarray] = {"timestamps_micros": timestamps}
    for side_name, rollout in (
        ("baseline", baseline),
        ("treatment", treatment),
    ):
        for field_name in ("valid", "x", "y", "heading", "vx", "vy"):
            arrays[f"{side_name}_{field_name}"] = np.stack(
                [
                    np.asarray(getattr(agent, field_name)[start:stop])
                    for agent in rollout.agents
                ],
                axis=0,
            )
    world_mask = np.ones(source.num_agents, dtype=bool)
    world_mask[source.ego_index] = False
    target_mask = np.zeros(source.num_agents, dtype=bool)
    target_mask[pair.eligibility.target_index] = True
    arrays["world_mask"] = world_mask
    arrays["target_mask"] = target_mask
    arrays["agent_ids"] = np.asarray(
        [agent.id for agent in source.agents],
        dtype="<i8",
    )
    world_indices = np.flatnonzero(world_mask)
    fields = ("valid", "x", "y", "heading", "vx", "vy")
    if scene_result.policy_name == "log_replay":
        gate_offset = slice(None)
    else:
        gate_offset = slice(1, 2)
    for field_name in fields:
        left = arrays[f"baseline_{field_name}"][world_indices, gate_offset]
        right = arrays[f"treatment_{field_name}"][world_indices, gate_offset]
        if not np.array_equal(left, right):
            _fail(
                "numpy_comparison_world_gate",
                "NumPy log-replay or synchronous-order gate failed",
            )

    digest = hashlib.sha256()
    digest.update(_NUMPY_VIEW_DOMAIN)
    digest.update(b"\x00")
    digest.update(selection_position.to_bytes(4, "big"))
    digest.update(qualification.cohort_index.to_bytes(4, "big"))
    digest.update(bytes.fromhex(qualification.qualification_binding_sha256))
    digest.update(bytes.fromhex(primary_domain.domain_sha256))
    digest.update(bytes.fromhex(selection_binding_sha256))
    digest.update(bytes.fromhex(numpy_eligibility_sha256))
    digest.update(scene_result.policy_name.encode("ascii"))
    digest.update(scene_result.policy_access_role.encode("ascii"))
    digest.update(bytes.fromhex(source._integrity_fingerprint))
    for name in sorted(arrays):
        _hash_named_array(digest, name, arrays[name])
    return digest.hexdigest(), MappingProxyType(arrays)


def _numpy_measures(
    arrays: Mapping[str, np.ndarray],
    *,
    ego_index: int,
    target_index: int,
    target_length: float,
    ego_length: float,
) -> tuple[tuple[str, str, str, float, bool | None, float | None], ...]:
    timestamps = arrays["timestamps_micros"]
    dt_s = np.diff(timestamps.astype(np.int64)) * 1e-6
    if (
        dt_s.shape != (M6_WAYMAX_TRANSITIONS,)
        or not np.all(np.isfinite(dt_s))
        or np.any(dt_s <= 0.0)
    ):
        _fail("numpy_comparison_timeline", "NumPy comparison cadence is invalid")
    baseline_speed = np.hypot(
        arrays["baseline_vx"][target_index],
        arrays["baseline_vy"][target_index],
    )
    treatment_speed = np.hypot(
        arrays["treatment_vx"][target_index],
        arrays["treatment_vy"][target_index],
    )
    baseline_acceleration = np.diff(baseline_speed) / dt_s
    treatment_acceleration = np.diff(treatment_speed) / dt_s
    impulse = math.fsum(
        float(value)
        for value in np.maximum(
            0.0,
            baseline_acceleration - treatment_acceleration,
        )
        * dt_s
    )

    delta = treatment_acceleration - baseline_acceleration
    run_start: int | None = None
    response_end: int | None = None
    for transition in range(1, M6_WAYMAX_TRANSITIONS):
        if float(delta[transition]) <= M6_RESPONSE_ACCELERATION_THRESHOLD_MPS2:
            if run_start is None:
                run_start = transition
            elapsed = (
                int(timestamps[transition + 1]) - int(timestamps[run_start])
            ) * 1e-6
            if elapsed >= M6_RESPONSE_PERSISTENCE_S:
                response_end = transition
                break
        else:
            run_start = None
    window_s = (int(timestamps[-1]) - int(timestamps[0])) * 1e-6
    responded = response_end is not None
    if responded:
        assert response_end is not None
        latency = (
            int(timestamps[response_end + 1]) - int(timestamps[0])
        ) * 1e-6
        timeliness = window_s - min(latency, window_s)
    else:
        latency = None
        timeliness = 0.0

    target_vx = float(arrays["baseline_vx"][target_index, 0])
    target_vy = float(arrays["baseline_vy"][target_index, 0])
    target_speed = math.hypot(target_vx, target_vy)
    if target_speed > 1e-12:
        gap_hx, gap_hy = target_vx / target_speed, target_vy / target_speed
    else:
        heading = float(arrays["baseline_heading"][target_index, 0])
        gap_hx, gap_hy = math.cos(heading), math.sin(heading)
    half_length = 0.5 * (target_length + ego_length)
    baseline_gap = (
        (
            arrays["baseline_x"][ego_index, 1:]
            - arrays["baseline_x"][target_index, 1:]
        )
        * gap_hx
        + (
            arrays["baseline_y"][ego_index, 1:]
            - arrays["baseline_y"][target_index, 1:]
        )
        * gap_hy
        - half_length
    )
    treatment_gap = (
        (
            arrays["treatment_x"][ego_index, 1:]
            - arrays["treatment_x"][target_index, 1:]
        )
        * gap_hx
        + (
            arrays["treatment_y"][ego_index, 1:]
            - arrays["treatment_y"][target_index, 1:]
        )
        * gap_hy
        - half_length
    )
    gap_change = float(np.min(treatment_gap)) - float(np.min(baseline_gap))

    heading = float(arrays["baseline_heading"][target_index, 0])
    progress_hx, progress_hy = math.cos(heading), math.sin(heading)
    origin_x = float(arrays["baseline_x"][target_index, 0])
    origin_y = float(arrays["baseline_y"][target_index, 0])
    baseline_progress = (
        (float(arrays["baseline_x"][target_index, -1]) - origin_x)
        * progress_hx
        + (float(arrays["baseline_y"][target_index, -1]) - origin_y)
        * progress_hy
    )
    treatment_progress = (
        (float(arrays["treatment_x"][target_index, -1]) - origin_x)
        * progress_hx
        + (float(arrays["treatment_y"][target_index, -1]) - origin_y)
        * progress_hy
    )
    progress_loss = baseline_progress - treatment_progress
    return (
        (*_NUMPY_METRICS[0], impulse, None, None),
        (*_NUMPY_METRICS[1], timeliness, responded, latency),
        (*_NUMPY_METRICS[2], gap_change, None, None),
        (*_NUMPY_METRICS[3], progress_loss, None, None),
    )


@dataclass(frozen=True, slots=True)
class M6WaymaxNumpyComparisonRow:
    """Factory-issued local-only scalar from one exact NumPy 20-step view."""

    selection_position: int
    cohort_index: int | None
    qualification_binding_sha256: str | None
    primary_domain_sha256: str
    selection_binding_sha256: str
    numpy_eligibility_ledger_sha256: str
    stored_eligibility_rows_sha256: str
    policy_name: Literal["log_replay", "idm"]
    policy_access_role: str
    metric_name: str
    metric_version: str
    value_unit: str
    value: float | None
    responded: bool | None
    responder_latency_s: float | None
    view_binding_sha256: str | None
    source_pairing_complete: bool
    status: Literal["selected", "not_selected"]
    row_binding_sha256: str | None = field(default=None, repr=False)
    _issued_original_sha256: str = field(init=False, repr=False, compare=False)
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _NUMPY_ISSUER:
            raise TypeError("M6WaymaxNumpyComparisonRow is factory-issued only")
        object.__setattr__(
            self,
            "selection_position",
            _strict_int(self.selection_position, "selection_position"),
        )
        if self.selection_position >= M6_WAYMAX_MAX_SCENES:
            raise ValueError("selection_position must lie in [0, 15]")
        if self.policy_name not in M6_WAYMAX_NUMPY_COMPARISON_POLICIES:
            raise ValueError("NumPy comparison policy is not registered")
        if self.policy_access_role != _NUMPY_POLICY_ACCESS[self.policy_name]:
            raise ValueError("NumPy policy access role differs from registration")
        metric = next(
            (item for item in _NUMPY_METRICS if item[0] == self.metric_name),
            None,
        )
        if metric != (self.metric_name, self.metric_version, self.value_unit):
            raise ValueError("NumPy comparison metric identity is not registered")
        for name in (
            "primary_domain_sha256",
            "selection_binding_sha256",
            "numpy_eligibility_ledger_sha256",
            "stored_eligibility_rows_sha256",
        ):
            _sha256(getattr(self, name), name)
        if self.status == "not_selected":
            if (
                self.cohort_index is not None
                or self.qualification_binding_sha256 is not None
                or self.value is not None
                or self.responded is not None
                or self.responder_latency_s is not None
                or self.view_binding_sha256 is not None
                or self.source_pairing_complete
            ):
                raise ValueError("not-selected NumPy comparison rows must be exact NA")
        elif self.status == "selected":
            _strict_int(self.cohort_index, "cohort_index")
            _sha256(
                self.qualification_binding_sha256,
                "qualification_binding_sha256",
            )
            _sha256(self.view_binding_sha256, "view_binding_sha256")
            value = float(self.value)
            if not math.isfinite(value) or self.source_pairing_complete is not True:
                raise ValueError("selected NumPy comparison row is incomplete")
            object.__setattr__(self, "value", value)
            if self.metric_name == "response_timeliness_s":
                if type(self.responded) is not bool:
                    raise ValueError("timeliness row requires responded")
                if self.responded:
                    latency = float(self.responder_latency_s)
                    if not math.isfinite(latency) or latency < 0.0:
                        raise ValueError("responder latency is invalid")
                    object.__setattr__(self, "responder_latency_s", latency)
                elif self.responder_latency_s is not None:
                    raise ValueError("censored timeliness cannot carry latency")
            elif self.responded is not None or self.responder_latency_s is not None:
                raise ValueError("response fields belong only to timeliness")
        else:
            raise ValueError("NumPy comparison status is not registered")
        expected = self._binding()
        if self.row_binding_sha256 is not None and self.row_binding_sha256 != expected:
            raise ValueError("row_binding_sha256 does not bind NumPy comparison")
        object.__setattr__(self, "row_binding_sha256", expected)
        object.__setattr__(self, "_issued_original_sha256", expected)
        _register_strong_issuance(
            "numpy_comparison_row",
            self,
            binding_sha256=expected,
            content_sha256=expected,
        )

    def to_store_dict(self) -> dict[str, Any]:
        return {
            "selection_position": self.selection_position,
            "cohort_index": self.cohort_index,
            "qualification_binding_sha256": self.qualification_binding_sha256,
            "primary_domain_sha256": self.primary_domain_sha256,
            "selection_binding_sha256": self.selection_binding_sha256,
            "numpy_eligibility_ledger_sha256": (
                self.numpy_eligibility_ledger_sha256
            ),
            "stored_eligibility_rows_sha256": (
                self.stored_eligibility_rows_sha256
            ),
            "policy_name": self.policy_name,
            "policy_access_role": self.policy_access_role,
            "metric_name": self.metric_name,
            "metric_version": self.metric_version,
            "value_unit": self.value_unit,
            "value": self.value,
            "responded": self.responded,
            "responder_latency_s": self.responder_latency_s,
            "view_binding_sha256": self.view_binding_sha256,
            "source_pairing_complete": self.source_pairing_complete,
            "status": self.status,
        }

    def _binding(self) -> str:
        return hashlib.sha256(
            _NUMPY_ROW_DOMAIN + b"\x00" + _canonical_json(self.to_store_dict())
        ).hexdigest()

    def revalidate(self) -> None:
        record = _strong_issuance_record(
            "numpy_comparison_row",
            self,
            error_code="numpy_comparison_row_mutated",
        )
        expected = self._binding()
        if (
            expected != self.row_binding_sha256
            or expected != self._issued_original_sha256
            or expected != record.binding_sha256
            or expected != record.content_sha256
        ):
            _fail("numpy_comparison_row_mutated", "NumPy comparison row changed")


def _numpy_comparison_keys() -> tuple[tuple[int, str, str], ...]:
    return tuple(
        (position, policy, metric[0])
        for position in range(M6_WAYMAX_MAX_SCENES)
        for policy in M6_WAYMAX_NUMPY_COMPARISON_POLICIES
        for metric in _NUMPY_METRICS
    )


@dataclass(frozen=True, slots=True)
class M6WaymaxNumpyComparisonTable(Sequence[M6WaymaxNumpyComparisonRow]):
    """Fixed 128-row local-only NumPy comparison evidence."""

    selected_member_count: int
    primary_domain_sha256: str
    selection_binding_sha256: str
    numpy_eligibility_ledger_sha256: str
    rows: tuple[M6WaymaxNumpyComparisonRow, ...]
    stored_eligibility_rows_sha256: str
    promotable: bool
    table_binding_sha256: str | None = field(default=None, repr=False)
    _issued_original_sha256: str = field(init=False, repr=False, compare=False)
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _NUMPY_ISSUER:
            raise TypeError("M6WaymaxNumpyComparisonTable is factory-issued only")
        count = _strict_int(self.selected_member_count, "selected_member_count")
        if type(self.promotable) is not bool:
            raise TypeError("promotable must be a boolean")
        if count not in (0, *range(8, M6_WAYMAX_MAX_SCENES + 1)):
            raise ValueError("NumPy comparison selected count is invalid")
        object.__setattr__(self, "selected_member_count", count)
        for name in (
            "primary_domain_sha256",
            "selection_binding_sha256",
            "stored_eligibility_rows_sha256",
            "numpy_eligibility_ledger_sha256",
        ):
            _sha256(getattr(self, name), name)
        rows = tuple(self.rows)
        if len(rows) != M6_WAYMAX_NUMPY_COMPARISON_ROW_COUNT:
            raise ValueError("NumPy comparison table must contain 128 rows")
        for row in rows:
            if not isinstance(row, M6WaymaxNumpyComparisonRow):
                raise TypeError("NumPy comparison rows must be factory-issued")
            if (
                row.primary_domain_sha256 != self.primary_domain_sha256
                or row.selection_binding_sha256 != self.selection_binding_sha256
                or row.numpy_eligibility_ledger_sha256
                != self.numpy_eligibility_ledger_sha256
                or row.stored_eligibility_rows_sha256
                != self.stored_eligibility_rows_sha256
            ):
                raise ValueError("NumPy row/table provenance differs")
            row.revalidate()
        keys = tuple(
            (row.selection_position, row.policy_name, row.metric_name)
            for row in rows
        )
        if keys != _numpy_comparison_keys():
            raise ValueError("NumPy comparison rows are not in canonical order")
        if count == 0 and self.promotable:
            raise ValueError("empty NumPy comparison table is non-promotable")
        for row in rows:
            if (row.selection_position < count) != (row.status == "selected"):
                raise ValueError("NumPy selected/NA layout differs from selection")
        object.__setattr__(self, "rows", rows)
        expected = self._binding()
        if self.table_binding_sha256 is not None and (
            self.table_binding_sha256 != expected
        ):
            raise ValueError("table_binding_sha256 does not bind NumPy table")
        object.__setattr__(self, "table_binding_sha256", expected)
        object.__setattr__(self, "_issued_original_sha256", expected)
        _register_strong_issuance(
            "numpy_comparison_table",
            self,
            binding_sha256=expected,
            content_sha256=expected,
            components=(self.rows, *self.rows),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int | slice):
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)

    def _binding(self) -> str:
        digest = hashlib.sha256()
        digest.update(_NUMPY_TABLE_DOMAIN)
        digest.update(b"\x00")
        digest.update(self.selected_member_count.to_bytes(4, "big"))
        digest.update(bytes.fromhex(self.primary_domain_sha256))
        digest.update(bytes.fromhex(self.selection_binding_sha256))
        digest.update(bytes.fromhex(self.numpy_eligibility_ledger_sha256))
        digest.update(b"\x01" if self.promotable else b"\x00")
        digest.update(bytes.fromhex(self.stored_eligibility_rows_sha256))
        for row in self.rows:
            digest.update(bytes.fromhex(row.row_binding_sha256))
        return digest.hexdigest()

    def to_store_rows(self) -> tuple[dict[str, Any], ...]:
        self.revalidate()
        return tuple(row.to_store_dict() for row in self.rows)

    def revalidate(
        self,
        *,
        selection: M6WaymaxSelection | None = None,
        primary_domain: M6WaymaxPrimaryDomain | None = None,
        eligibility_ledger: M6EligibilityLedger | None = None,
    ) -> None:
        record = _strong_issuance_record(
            "numpy_comparison_table",
            self,
            error_code="numpy_comparison_table_mutated",
        )
        _require_original_components(
            record,
            (self.rows, *self.rows),
            error_code="numpy_comparison_table_mutated",
        )
        if type(self.promotable) is not bool:
            _fail("numpy_comparison_table_mutated", "promotable flag changed")
        for row in self.rows:
            if (
                row.primary_domain_sha256 != self.primary_domain_sha256
                or row.selection_binding_sha256 != self.selection_binding_sha256
                or row.numpy_eligibility_ledger_sha256
                != self.numpy_eligibility_ledger_sha256
                or row.stored_eligibility_rows_sha256
                != self.stored_eligibility_rows_sha256
            ):
                _fail(
                    "numpy_comparison_table_mutated",
                    "NumPy row/table provenance changed",
                )
        for row in self.rows:
            row.revalidate()
        expected = self._binding()
        if (
            expected != self.table_binding_sha256
            or expected != self._issued_original_sha256
            or expected != record.binding_sha256
            or expected != record.content_sha256
        ):
            _fail("numpy_comparison_table_mutated", "NumPy comparison table changed")
        supplied = (selection, primary_domain, eligibility_ledger)
        if all(value is None for value in supplied):
            return
        if any(value is None for value in supplied):
            raise ValueError("selection/domain/eligibility must be supplied together")
        assert selection is not None
        assert primary_domain is not None
        assert eligibility_ledger is not None
        selection.revalidate(primary_domain=primary_domain)
        if (
            self.selected_member_count
            != (len(selection.members) if selection.supported else 0)
            or self.primary_domain_sha256 != primary_domain.domain_sha256
            or self.selection_binding_sha256
            != m6_waymax_selection_binding_sha256(selection)
            or self.numpy_eligibility_ledger_sha256
            != _eligibility_ledger_sha256(eligibility_ledger)
        ):
            _fail(
                "numpy_comparison_selection",
                "NumPy comparison table belongs to another selection/domain",
            )
        rows_per_position = len(M6_WAYMAX_NUMPY_COMPARISON_POLICIES) * len(
            _NUMPY_METRICS
        )
        for position, member in enumerate(selection.members):
            group = self.rows[
                position * rows_per_position : (position + 1) * rows_per_position
            ]
            if any(
                row.cohort_index != member.cohort_index
                or row.qualification_binding_sha256
                != member.qualification_binding_sha256
                for row in group
            ):
                _fail(
                    "numpy_comparison_selection",
                    "NumPy row provenance differs from selected member",
                )


def _build_numpy_comparison_table(
    numpy_evidence: M6OfficialNumpyRows,
    *,
    selection: M6WaymaxSelection,
    primary_domain: M6WaymaxPrimaryDomain,
    eligibility_ledger: M6EligibilityLedger,
    promotable: bool,
) -> M6WaymaxNumpyComparisonTable:
    if not isinstance(numpy_evidence, M6OfficialNumpyRows):
        raise TypeError("numpy_evidence must be M6OfficialNumpyRows")
    numpy_result = numpy_evidence.typed_result
    numpy_result.revalidate()
    stored_rows_sha256 = _stored_eligibility_rows_sha256(numpy_evidence)
    ledger_sha256 = _eligibility_ledger_sha256(eligibility_ledger)
    if _eligibility_ledger_sha256(numpy_result.eligibility_ledger) != ledger_sha256:
        _fail(
            "numpy_comparison_ledger",
            "NumPy outcome result differs from frozen source-only ledger",
        )
    selection.revalidate(primary_domain=primary_domain)
    selection_binding = m6_waymax_selection_binding_sha256(selection)
    results = {
        (result.cohort_index, result.policy_name): result
        for result in numpy_result.primary_scene_results
        if result.policy_name in M6_WAYMAX_NUMPY_COMPARISON_POLICIES
    }
    issued: dict[tuple[int, str, str], M6WaymaxNumpyComparisonRow] = {}
    for position, qualification in enumerate(selection.members):
        numpy_entry = eligibility_ledger.entry_for(qualification.cohort_index)
        for policy_name in M6_WAYMAX_NUMPY_COMPARISON_POLICIES:
            scene_result = results[(qualification.cohort_index, policy_name)]
            view_binding, arrays = _numpy_pair_view(
                scene_result,
                selection_position=position,
                qualification=qualification,
                primary_domain=primary_domain,
                selection_binding_sha256=selection_binding,
                numpy_eligibility_sha256=ledger_sha256,
            )
            source = numpy_entry.source_snapshot
            target_index = numpy_entry.target_index
            assert target_index is not None
            measures = _numpy_measures(
                arrays,
                ego_index=source.ego_index,
                target_index=target_index,
                target_length=source.agents[target_index].length,
                ego_length=source.ego.length,
            )
            for (
                metric_name,
                metric_version,
                value_unit,
                value,
                responded,
                latency,
            ) in measures:
                key = (position, policy_name, metric_name)
                issued[key] = M6WaymaxNumpyComparisonRow(
                    selection_position=position,
                    cohort_index=qualification.cohort_index,
                    qualification_binding_sha256=(
                        qualification.qualification_binding_sha256
                    ),
                    primary_domain_sha256=primary_domain.domain_sha256,
                    selection_binding_sha256=selection_binding,
                    numpy_eligibility_ledger_sha256=ledger_sha256,
                    policy_name=policy_name,
                    policy_access_role=_NUMPY_POLICY_ACCESS[policy_name],
                    stored_eligibility_rows_sha256=stored_rows_sha256,
                    metric_name=metric_name,
                    metric_version=metric_version,
                    value_unit=value_unit,
                    value=value,
                    responded=responded,
                    responder_latency_s=latency,
                    view_binding_sha256=view_binding,
                    source_pairing_complete=True,
                    status="selected",
                    _issuance_capability=_NUMPY_ISSUER,
                )
    rows = []
    metric_by_name = {metric[0]: metric for metric in _NUMPY_METRICS}
    for key in _numpy_comparison_keys():
        selected = issued.get(key)
        if selected is not None:
            rows.append(selected)
            continue
        position, policy_name, metric_name = key
        metric_version, value_unit = metric_by_name[metric_name][1:]
        rows.append(
            M6WaymaxNumpyComparisonRow(
                selection_position=position,
                cohort_index=None,
                qualification_binding_sha256=None,
                primary_domain_sha256=primary_domain.domain_sha256,
                selection_binding_sha256=selection_binding,
                numpy_eligibility_ledger_sha256=ledger_sha256,
                policy_name=policy_name,
                policy_access_role=_NUMPY_POLICY_ACCESS[policy_name],
                stored_eligibility_rows_sha256=stored_rows_sha256,
                metric_name=metric_name,
                metric_version=metric_version,
                value_unit=value_unit,
                value=None,
                responded=None,
                responder_latency_s=None,
                view_binding_sha256=None,
                source_pairing_complete=False,
                status="not_selected",
                _issuance_capability=_NUMPY_ISSUER,
            )
        )
    table = M6WaymaxNumpyComparisonTable(
        selected_member_count=(len(selection.members) if selection.supported else 0),
        primary_domain_sha256=primary_domain.domain_sha256,
        selection_binding_sha256=selection_binding,
        numpy_eligibility_ledger_sha256=ledger_sha256,
        rows=tuple(rows),
        promotable=promotable,
        stored_eligibility_rows_sha256=stored_rows_sha256,
        _issuance_capability=_NUMPY_ISSUER,
    )
    table.revalidate(
        selection=selection,
        primary_domain=primary_domain,
        eligibility_ledger=eligibility_ledger,
    )
    return table


def validate_m6_waymax_numpy_comparison_table(
    value: Any,
    *,
    selection: M6WaymaxSelection,
    primary_domain: M6WaymaxPrimaryDomain,
    eligibility_ledger: M6EligibilityLedger,
) -> M6WaymaxNumpyComparisonTable:
    if not isinstance(value, M6WaymaxNumpyComparisonTable):
        raise TypeError(
            "NumPy comparison evidence must be a factory-issued typed table"
        )
    value.revalidate(
        selection=selection,
        primary_domain=primary_domain,
        eligibility_ledger=eligibility_ledger,
    )
    return value


def _execute_once(
    executor: M6WaymaxCompactExecutor,
    *,
    state: Any,
    scenario: Scenario,
    plan: EgoTrajectoryPlan,
    bundle: str,
    selection: M6WaymaxSelection,
    primary_domain: M6WaymaxPrimaryDomain,
    selection_position: int,
    qualification: M6WaymaxEligibility,
    prior_raw: list[CompactM6WaymaxRollout],
    prior_views: list[WaymaxEgoPlanView],
    prior_plans: list[EgoTrajectoryPlan],
    prior_sources: list[tuple[Any, Scenario]],
) -> tuple[CompactM6WaymaxRollout, WaymaxEgoPlanView]:
    state_before = source_state_mutation_sha256(state)
    scenario_before = _scenario_fingerprint(scenario)
    plan_copy = EgoTrajectoryPlan.deserialize(plan.serialize())
    plan_before = plan_copy.serialize()
    result = executor(
        state,
        scenario,
        plan_copy,
        bundle=bundle,
        selection=selection,
        primary_domain=primary_domain,
        selection_position=selection_position,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError(
            "executor must return "
            "(CompactM6WaymaxRollout, WaymaxEgoPlanView)"
        )
    compact, view = result
    if not isinstance(compact, CompactM6WaymaxRollout) or not isinstance(
        view, WaymaxEgoPlanView
    ):
        raise TypeError("executor returned untyped compact or plan view")

    current_items = (
        ("compact", compact),
        ("view", view),
        ("plan", plan_copy),
        ("plan", plan),
        ("source", state),
        ("source", scenario),
    )
    for left_index, (left_category, left_value) in enumerate(current_items):
        for right_category, right_value in current_items[left_index + 1 :]:
            if left_category != right_category and _trees_share_storage(
                left_value,
                right_value,
            ):
                _fail(
                    "official_cross_category_alias",
                    f"current {left_category} aliases {right_category} storage",
                )

    prior_items: list[tuple[str, Any]] = [
        *(("compact", prior) for prior in prior_raw),
        *(("view", prior) for prior in prior_views),
        *(("plan", prior) for prior in prior_plans),
    ]
    for prior_state, prior_scenario in prior_sources:
        prior_items.extend(
            (("source", prior_state), ("source", prior_scenario))
        )
    for current_category, current_value in current_items:
        for prior_category, prior_value in prior_items:
            if current_category != prior_category and _trees_share_storage(
                current_value,
                prior_value,
            ):
                _fail(
                    "official_cross_category_replay",
                    f"current {current_category} aliases prior "
                    f"{prior_category} storage",
                )
    if any(_trees_share_storage(view, prior) for prior in prior_views):
        _fail(
            "official_view_replay",
            "executor replayed plan-view storage across independent calls",
        )
    if any(_compacts_share_storage(compact, prior) for prior in prior_raw):
        _fail(
            "official_execution_replay",
            "executor replayed compact storage across independent calls",
        )
    if any(_trees_share_storage(plan_copy, prior) for prior in prior_plans):
        _fail(
            "official_plan_replay",
            "executor input plan replayed storage across independent calls",
        )
    prior_raw.append(compact)
    prior_views.append(view)
    prior_plans.append(plan_copy)
    if not any(
        prior_state is state and prior_scenario is scenario
        for prior_state, prior_scenario in prior_sources
    ):
        prior_sources.append((state, scenario))
    validation = validate_m6_waymax_compact(
        compact,
        state=state,
        scenario=scenario,
        plan=plan_copy,
        view=view,
        bundle=bundle,
        qualification=qualification,
        primary_domain=primary_domain,
    )
    validation.require_passed()
    plan_copy.revalidate()
    view.revalidate()
    if plan_copy.serialize() != plan_before:
        _fail("official_plan_mutated", "executor mutated its canonical plan")
    if source_state_mutation_sha256(state) != state_before:
        _fail("official_source_mutated", "executor mutated its source state")
    if _scenario_fingerprint(scenario) != scenario_before:
        _fail("official_scenario_mutated", "executor mutated its scenario")
    frozen = _clone_compact(compact)
    frozen_validation = validate_m6_waymax_compact(
        frozen,
        state=state,
        scenario=scenario,
        plan=plan,
        view=view,
        bundle=bundle,
        qualification=qualification,
        primary_domain=primary_domain,
    )
    frozen_validation.require_passed()
    return frozen, view


@dataclass(frozen=True, slots=True)
class M6WaymaxNonPromotableScalarEvidence(
    Sequence[Any]
):
    """Test-authority scalar evidence that cannot enter the live store API."""

    _issued: M6WaymaxIssuedScalarTable = field(repr=False)
    promotable: bool = False
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _NONPROMOTABLE_ISSUER:
            raise TypeError("non-promotable scalar evidence is runner-issued only")
        if not isinstance(self._issued, M6WaymaxIssuedScalarTable):
            raise TypeError("non-promotable scalar evidence requires issued rows")
        if self.promotable is not False:
            raise ValueError("test scalar evidence is permanently non-promotable")

    def __len__(self) -> int:
        return len(self._issued)

    def __getitem__(self, index: int | slice):
        return self._issued[index]

    def __iter__(self):
        return iter(self._issued)

    def revalidate(self, *, selection: M6WaymaxSelection) -> None:
        if self.promotable is not False:
            _fail("test_scalar_promoted", "test scalar evidence was promoted")
        self._issued.revalidate(selection=selection)


@dataclass(frozen=True, slots=True)
class M6WaymaxNonPromotableDeterminismEvidence(Sequence[Any]):
    """Test-authority live hashes that cannot satisfy the store's live type gate."""

    _issued: M6WaymaxLiveDeterminismTable = field(repr=False)
    promotable: bool = False
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _NONPROMOTABLE_ISSUER:
            raise TypeError(
                "non-promotable determinism evidence is runner-issued only"
            )
        if not isinstance(self._issued, M6WaymaxLiveDeterminismTable):
            raise TypeError("test determinism evidence requires live typed hashes")
        if self.promotable is not False:
            raise ValueError("test determinism is permanently non-promotable")

    def __len__(self) -> int:
        return len(self._issued)

    def __getitem__(self, index: int | slice):
        return self._issued[index]

    def __iter__(self):
        return iter(self._issued)

    def to_store_rows(self) -> tuple[dict[str, Any], ...]:
        self.revalidate()
        return self._issued.to_store_rows()

    def revalidate(
        self,
        *,
        selection: M6WaymaxSelection | None = None,
        primary_domain: M6WaymaxPrimaryDomain | None = None,
    ) -> None:
        if self.promotable is not False:
            _fail("test_determinism_promoted", "test determinism was promoted")
        if selection is None and primary_domain is None:
            self._issued.revalidate()
            return
        if selection is None or primary_domain is None:
            raise ValueError("selection and primary_domain must be supplied together")
        self._issued.revalidate(
            selection=selection,
            primary_domain=primary_domain,
        )


@dataclass(frozen=True, slots=True)
class M6WaymaxOfficialEvidence:
    """Complete authority-bound official Waymax and NumPy comparison evidence."""

    _AUTHORITY_TYPES = (
        M6WaymaxSourceAuthority,
        M6WaymaxExecutionAuthority,
    )

    primary_domain: M6WaymaxPrimaryDomain = field(repr=False)
    qualification_ledger: M6WaymaxQualificationLedger = field(repr=False)
    selection: M6WaymaxSelection = field(repr=False)
    scene_scalars: (
        M6WaymaxIssuedScalarTable | M6WaymaxNonPromotableScalarEvidence
    )
    field_comparisons: M6WaymaxOfficialFieldComparisonTable
    determinism: (
        M6WaymaxLiveDeterminismTable
        | M6WaymaxNoExecutionDeterminismTable
        | M6WaymaxNonPromotableDeterminismEvidence
    )
    numpy_comparisons: M6WaymaxNumpyComparisonTable
    promotable: bool
    production_authoritative: bool = field(init=False)
    _source_authority: M6WaymaxSourceAuthority = field(
        repr=False,
        compare=False,
    )
    _execution_authority: M6WaymaxExecutionAuthority = field(
        repr=False,
        compare=False,
    )
    _numpy_evidence: M6OfficialNumpyRows = field(repr=False, compare=False)
    _issuance_capability: InitVar[object] = None
    evidence_binding_sha256: str | None = field(default=None, repr=False)
    _issued_original_binding_sha256: str = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _RESULT_ISSUER:
            raise TypeError("M6WaymaxOfficialEvidence is runner-issued only")
        self._validate_authorities()
        object.__setattr__(
            self,
            "production_authoritative",
            self._production_authority_fact(),
        )
        expected = self._binding_sha256()
        if (
            self.evidence_binding_sha256 is not None
            and self.evidence_binding_sha256 != expected
        ):
            raise ValueError("evidence_binding_sha256 does not bind evidence")
        object.__setattr__(self, "evidence_binding_sha256", expected)
        object.__setattr__(self, "_issued_original_binding_sha256", expected)
        _register_strong_issuance(
            "official_evidence",
            self,
            binding_sha256=expected,
            content_sha256=expected,
            components=self._issuance_components(),
        )
        self.revalidate()

    @property
    def supported(self) -> bool:
        return self.selection.supported

    def _validate_authorities(self) -> None:
        if type(self._source_authority) is not self._AUTHORITY_TYPES[0]:
            raise TypeError("official evidence requires exact source authority")
        if type(self._execution_authority) is not self._AUTHORITY_TYPES[1]:
            raise TypeError(
                "official evidence requires exact execution authority"
            )
        self._source_authority.revalidate()
        self._execution_authority.revalidate()

    def _production_authority_fact(self) -> bool:
        return (
            self._source_authority.kind == "verified_accepted_m4"
            and self._source_authority.promotable is True
            and self._execution_authority.kind
            == "pinned_cpu_waymax_jax"
            and self._execution_authority.promotable is True
        )

    def _issuance_components(self) -> tuple[Any, ...]:
        scalar_table = (
            self.scene_scalars._issued
            if isinstance(
                self.scene_scalars,
                M6WaymaxNonPromotableScalarEvidence,
            )
            else self.scene_scalars
        )
        determinism_table = (
            self.determinism._issued
            if isinstance(
                self.determinism,
                M6WaymaxNonPromotableDeterminismEvidence,
            )
            else self.determinism
        )
        return (
            self.primary_domain,
            self.qualification_ledger,
            self.selection,
            self.scene_scalars,
            scalar_table,
            self.field_comparisons,
            self.determinism,
            determinism_table,
            self.numpy_comparisons,
            self._source_authority,
            self._execution_authority,
            self._numpy_evidence,
            self._numpy_evidence.typed_result,
        )

    def _binding_sha256(self) -> str:
        scalar_table = (
            self.scene_scalars._issued
            if isinstance(
                self.scene_scalars,
                M6WaymaxNonPromotableScalarEvidence,
            )
            else self.scene_scalars
        )
        determinism_table = (
            self.determinism._issued
            if isinstance(
                self.determinism,
                M6WaymaxNonPromotableDeterminismEvidence,
            )
            else self.determinism
        )
        payload = {
            "source_authority_kind": self._source_authority.kind,
            "source_authority_sha256": self._source_authority.authority_sha256,
            "execution_authority_kind": self._execution_authority.kind,
            "execution_authority_sha256": (
                self._execution_authority.authority_sha256
            ),
            "primary_domain_sha256": self.primary_domain.domain_sha256,
            "qualification_ledger_sha256": (
                self.qualification_ledger.ledger_sha256
            ),
            "selection_sha256": self.selection.selection_sha256,
            "selection_binding_sha256": (
                m6_waymax_selection_binding_sha256(self.selection)
            ),
            "scene_scalar_type": type(self.scene_scalars).__name__,
            "scene_scalar_table_sha256": scalar_table.table_binding_sha256,
            "field_table_sha256": (
                self.field_comparisons.table_binding_sha256
            ),
            "determinism_type": type(self.determinism).__name__,
            "determinism_table_sha256": (
                determinism_table.table_binding_sha256
            ),
            "numpy_comparison_table_sha256": (
                self.numpy_comparisons.table_binding_sha256
            ),
            "numpy_safe_rows_sha256": (
                _official_numpy_rows_sha256(self._numpy_evidence)
            ),
            "promotable": self.promotable,
            "production_authoritative": self.production_authoritative,
        }
        return hashlib.sha256(
            _RESULT_DOMAIN + b"\x00" + _canonical_json(payload)
        ).hexdigest()

    def revalidate(self) -> None:
        record = _strong_issuance_record(
            "official_evidence",
            self,
            error_code="official_evidence_binding",
        )
        _require_original_components(
            record,
            self._issuance_components(),
            error_code="official_evidence_binding",
        )
        if type(self.promotable) is not bool:
            _fail("official_evidence_promotable", "promotable flag changed")
        self._validate_authorities()
        if not isinstance(self._numpy_evidence, M6OfficialNumpyRows):
            raise TypeError("official evidence requires factory-issued NumPy rows")
        if (
            type(self.production_authoritative) is not bool
            or self.production_authoritative
            is not self._production_authority_fact()
        ):
            _fail(
                "official_evidence_production_authority",
                "production authority fact differs from its authorities",
            )
        self._numpy_evidence.revalidate()
        self.primary_domain.revalidate()
        self.qualification_ledger.revalidate()
        self.selection.revalidate(primary_domain=self.primary_domain)
        if (
            self.qualification_ledger.ledger_sha256
            != self.selection.qualification_ledger_sha256
            or self.qualification_ledger.primary_domain_sha256
            != self.primary_domain.domain_sha256
        ):
            _fail("official_evidence_ledger", "qualification ledger drifted")
        numpy_ledger = self._numpy_evidence.typed_result.eligibility_ledger
        expected_promotable = (
            self.selection.supported
            and self._source_authority.promotable
            and self._execution_authority.promotable
        )
        if self.promotable is not expected_promotable:
            _fail(
                "official_evidence_promotable",
                "evidence promotability differs from its authorities",
            )
        if (
            self.numpy_comparisons.promotable is not expected_promotable
            or self.field_comparisons.promotable is not expected_promotable
            or self.numpy_comparisons.stored_eligibility_rows_sha256
            != _stored_eligibility_rows_sha256(self._numpy_evidence)
        ):
            _fail(
                "official_evidence_promotable",
                "typed evidence authority or eligibility binding differs",
            )
        self.numpy_comparisons.revalidate(
            selection=self.selection,
            primary_domain=self.primary_domain,
            eligibility_ledger=numpy_ledger,
        )
        self.field_comparisons.revalidate(
            selection=self.selection,
            primary_domain=self.primary_domain,
        )
        if expected_promotable:
            if not isinstance(self.scene_scalars, M6WaymaxIssuedScalarTable):
                _fail(
                    "official_evidence_scalar_type",
                    "promotable scalar evidence lost its live type",
                )
            if not isinstance(
                self.determinism,
                M6WaymaxLiveDeterminismTable,
            ):
                _fail(
                    "official_evidence_determinism_type",
                    "promotable determinism lost its live type",
                )
        elif self.selection.supported:
            if not isinstance(
                self.scene_scalars,
                M6WaymaxNonPromotableScalarEvidence,
            ) or not isinstance(
                self.determinism,
                M6WaymaxNonPromotableDeterminismEvidence,
            ):
                _fail(
                    "official_evidence_test_type",
                    "test-authority evidence escaped its non-promotable wrapper",
                )
        elif not isinstance(
            self.scene_scalars,
            M6WaymaxIssuedScalarTable,
        ) or not isinstance(
            self.determinism,
            M6WaymaxNoExecutionDeterminismTable,
        ):
            _fail(
                "official_evidence_unsupported_type",
                "unsupported evidence is not exact non-promotable NA",
            )
        self.scene_scalars.revalidate(selection=self.selection)
        self.determinism.revalidate(
            selection=self.selection,
            primary_domain=self.primary_domain,
        )
        expected_binding = self._binding_sha256()
        if (
            self.evidence_binding_sha256 != expected_binding
            or self._issued_original_binding_sha256 != expected_binding
            or record.binding_sha256 != expected_binding
            or record.content_sha256 != expected_binding
        ):
            _fail(
                "official_evidence_binding",
                "official evidence bundle binding changed",
            )


def build_m6_waymax_unsupported_numpy_comparison_table(
    numpy_evidence: M6OfficialNumpyRows,
    *,
    selection: M6WaymaxSelection,
    primary_domain: M6WaymaxPrimaryDomain,
    eligibility_ledger: M6EligibilityLedger,
) -> M6WaymaxNumpyComparisonTable:
    """Issue exact fixed-grid NA NumPy rows for an authentic unsupported selection."""

    selection.revalidate(primary_domain=primary_domain)
    if selection.supported:
        raise ValueError("supported selection requires live NumPy comparisons")
    return _build_numpy_comparison_table(
        numpy_evidence,
        selection=selection,
        primary_domain=primary_domain,
        eligibility_ledger=eligibility_ledger,
        promotable=False,
    )


def _validate_official_inputs(
    source: M6WaymaxOfficialSource,
    execution_authority: M6WaymaxExecutionAuthority,
    numpy_evidence: M6OfficialNumpyRows,
    _authority_types=(
        M6WaymaxOfficialSource,
        M6WaymaxExecutionAuthority,
        M6OfficialNumpyRows,
    ),
) -> None:
    if type(source) is not _authority_types[0]:
        raise TypeError("source must be exact M6WaymaxOfficialSource")
    if type(execution_authority) is not _authority_types[1]:
        raise TypeError(
            "execution_authority must be exact M6WaymaxExecutionAuthority"
        )
    if type(numpy_evidence) is not _authority_types[2]:
        raise TypeError("numpy_evidence must be exact M6OfficialNumpyRows")
    source.revalidate()
    execution_authority.revalidate()
    numpy_evidence.revalidate()
    if (
        _eligibility_ledger_sha256(
            numpy_evidence.typed_result.eligibility_ledger
        )
        != _eligibility_ledger_sha256(source.numpy_eligibility_ledger)
    ):
        _fail(
            "official_numpy_source",
            "official NumPy result and Waymax source use different ledgers",
        )
    _stored_eligibility_rows_sha256(numpy_evidence)


def _wrap_nonpromotable_scalars(
    issued: M6WaymaxIssuedScalarTable,
) -> M6WaymaxNonPromotableScalarEvidence:
    return M6WaymaxNonPromotableScalarEvidence(
        _issued=issued,
        _issuance_capability=_NONPROMOTABLE_ISSUER,
    )


def _wrap_nonpromotable_determinism(
    issued: M6WaymaxLiveDeterminismTable,
) -> M6WaymaxNonPromotableDeterminismEvidence:
    return M6WaymaxNonPromotableDeterminismEvidence(
        _issued=issued,
        _issuance_capability=_NONPROMOTABLE_ISSUER,
    )


def _initialize_official_runner(
    source_type,
    execution_type,
    numpy_type,
    input_validator,
):
    dependencies = (
        source_type,
        execution_type,
        numpy_type,
        input_validator,
    )

    def run(
        source: M6WaymaxOfficialSource,
        execution_authority: M6WaymaxExecutionAuthority,
        numpy_evidence: M6OfficialNumpyRows,
    ) -> M6WaymaxOfficialEvidence:
        """Execute the exact selected 2x2 matrix through one factory-issued authority."""
        if type(source) is not dependencies[0]:
            raise TypeError(
                "source must be exact M6WaymaxOfficialSource"
            )
        if type(execution_authority) is not dependencies[1]:
            raise TypeError(
                "execution_authority must be exact "
                "M6WaymaxExecutionAuthority"
            )
        if type(numpy_evidence) is not dependencies[2]:
            raise TypeError(
                "numpy_evidence must be exact M6OfficialNumpyRows"
            )
        dependencies[3](
            source,
            execution_authority,
            numpy_evidence,
        )

        execution_authority._claim()
        selection = source.selection
        primary_domain = source.primary_domain
        numpy_ledger = source.numpy_eligibility_ledger
        promotable = (
            selection.supported
            and source.promotable
            and execution_authority.promotable
        )
        numpy_comparisons = _build_numpy_comparison_table(
            numpy_evidence,
            selection=selection,
            primary_domain=primary_domain,
            eligibility_ledger=numpy_ledger,
            promotable=promotable,
        )
        if not selection.supported:
            issued_scalars = build_m6_waymax_scene_scalar_table(
                (),
                selection=selection,
            )
            fields = build_m6_waymax_unsupported_field_comparison_table(
                selection=selection,
                primary_domain=primary_domain,
            )
            determinism = build_m6_waymax_unsupported_determinism_table(
                selection=selection,
                primary_domain=primary_domain,
            )
            evidence = M6WaymaxOfficialEvidence(
                primary_domain=primary_domain,
                qualification_ledger=source.qualification_ledger,
                selection=selection,
                scene_scalars=issued_scalars,
                field_comparisons=fields,
                determinism=determinism,
                numpy_comparisons=numpy_comparisons,
                promotable=False,
                _source_authority=source._source_authority,
                _execution_authority=execution_authority,
                _numpy_evidence=numpy_evidence,
                _issuance_capability=_RESULT_ISSUER,
            )
            source.revalidate()
            return evidence

        executors = execution_authority._get_executors()
        pair_views = []
        determinism_inputs: list[M6WaymaxLiveDeterminismExecution] = []
        field_evidence: dict[tuple[int, str, str, str], _Comparison] = {}
        prior_raw: list[CompactM6WaymaxRollout] = []
        prior_views: list[WaymaxEgoPlanView] = []
        prior_plans: list[EgoTrajectoryPlan] = []
        prior_sources: list[tuple[Any, Scenario]] = []
        for position, expected_member in enumerate(selection.members):
            state, scenario, qualification = source._materialize(position)
            if (
                qualification.cohort_index != expected_member.cohort_index
                or qualification.qualification_binding_sha256
                != expected_member.qualification_binding_sha256
            ):
                _fail(
                    "official_execution_order",
                    "resident member differs from exact selected order",
                )
            plans = {
                "identity": compile_identity_plan(scenario),
                "primary_brake": compile_longitudinal_brake_pulse_plan(
                    scenario,
                    PRIMARY_BRAKE_MAGNITUDE_MPS2,
                ),
            }
            for plan in plans.values():
                plan.revalidate()
            by_bundle_condition: dict[
                tuple[str, str],
                tuple[CompactM6WaymaxRollout, WaymaxEgoPlanView],
            ] = {}
            for bundle in M6_WAYMAX_BUNDLES:
                for condition in M6_WAYMAX_OFFICIAL_CONDITIONS:
                    plan = plans[condition]
                    first, first_view = _execute_once(
                        executors.eager,
                        state=state,
                        scenario=scenario,
                        plan=plan,
                        bundle=bundle,
                        selection=selection,
                        primary_domain=primary_domain,
                        selection_position=position,
                        qualification=qualification,
                        prior_raw=prior_raw,
                        prior_views=prior_views,
                        prior_plans=prior_plans,
                        prior_sources=prior_sources,
                    )
                    second, _second_view = _execute_once(
                        executors.eager,
                        state=state,
                        scenario=scenario,
                        plan=plan,
                        bundle=bundle,
                        selection=selection,
                        primary_domain=primary_domain,
                        selection_position=position,
                        qualification=qualification,
                        prior_raw=prior_raw,
                        prior_views=prior_views,
                        prior_plans=prior_plans,
                        prior_sources=prior_sources,
                    )
                    jit_eager = None
                    jit_compiled = None
                    if position == 0:
                        jit_eager, _ = _execute_once(
                            executors.jit_eager,
                            state=state,
                            scenario=scenario,
                            plan=plan,
                            bundle=bundle,
                            selection=selection,
                            primary_domain=primary_domain,
                            selection_position=position,
                            qualification=qualification,
                            prior_raw=prior_raw,
                            prior_views=prior_views,
                            prior_plans=prior_plans,
                            prior_sources=prior_sources,
                        )
                        jit_compiled, _ = _execute_once(
                            executors.jit_compiled,
                            state=state,
                            scenario=scenario,
                            plan=plan,
                            bundle=bundle,
                            selection=selection,
                            primary_domain=primary_domain,
                            selection_position=position,
                            qualification=qualification,
                            prior_raw=prior_raw,
                            prior_views=prior_views,
                            prior_plans=prior_plans,
                            prior_sources=prior_sources,
                        )
                    determinism_inputs.append(
                        M6WaymaxLiveDeterminismExecution(
                            selection_position=position,
                            bundle=bundle,
                            condition=condition,
                            qualification=qualification,
                            eager_pass_1=first,
                            eager_pass_2=second,
                            jit_eager=jit_eager,
                            jit_compiled=jit_compiled,
                        )
                    )
                    comparisons = _field_comparisons(
                        first,
                        state=state,
                        scenario=scenario,
                        plan=plan,
                        view=first_view,
                        bundle=bundle,
                        qualification=qualification,
                        primary_domain=primary_domain,
                    )
                    for field_name, comparison in comparisons.items():
                        field_evidence[
                            (position, bundle, condition, field_name)
                        ] = comparison
                    by_bundle_condition[(bundle, condition)] = (
                        first,
                        first_view,
                    )
                baseline, baseline_view = by_bundle_condition[
                    (bundle, "identity")
                ]
                treatment, treatment_view = by_bundle_condition[
                    (bundle, "primary_brake")
                ]
                pair_views.append(
                    build_m6_waymax_twenty_transition_pair_view(
                        baseline,
                        treatment,
                        selection_position=position,
                        state=state,
                        scenario=scenario,
                        baseline_plan=plans["identity"],
                        treatment_plan=plans["primary_brake"],
                        baseline_view=baseline_view,
                        treatment_view=treatment_view,
                        bundle=bundle,
                        qualification=qualification,
                        primary_domain=primary_domain,
                        selection=selection,
                    )
                )
            if (
                source_state_mutation_sha256(state)
                != qualification.source_binding_sha256
            ):
                _fail(
                    "official_source_mutated",
                    "scene execution mutated source state",
                )
        issued_scalars = build_m6_waymax_scene_scalar_table(
            tuple(pair_views),
            selection=selection,
        )
        fields = _issue_field_table(
            field_evidence,
            selection=selection,
            primary_domain=primary_domain,
            promotable=promotable,
        )
        live_determinism = build_m6_waymax_live_determinism_table(
            tuple(determinism_inputs),
            selection=selection,
            primary_domain=primary_domain,
        )
        if promotable:
            scene_scalars = issued_scalars
            determinism_evidence = live_determinism
        else:
            scene_scalars = _wrap_nonpromotable_scalars(issued_scalars)
            determinism_evidence = _wrap_nonpromotable_determinism(
                live_determinism
            )
        evidence = M6WaymaxOfficialEvidence(
            primary_domain=primary_domain,
            qualification_ledger=source.qualification_ledger,
            selection=selection,
            scene_scalars=scene_scalars,
            field_comparisons=fields,
            determinism=determinism_evidence,
            numpy_comparisons=numpy_comparisons,
            promotable=promotable,
            _source_authority=source._source_authority,
            _execution_authority=execution_authority,
            _numpy_evidence=numpy_evidence,
            _issuance_capability=_RESULT_ISSUER,
        )
        source.revalidate()
        evidence.revalidate()
        return evidence

    run.__name__ = "run_m6_waymax_official"
    run.__qualname__ = "run_m6_waymax_official"
    return run


run_m6_waymax_official = _initialize_official_runner(
    M6WaymaxOfficialSource,
    M6WaymaxExecutionAuthority,
    M6OfficialNumpyRows,
    _validate_official_inputs,
)
del _initialize_official_runner


M6_WAYMAX_PILOT_SCHEMA_VERSION = "m6-waymax-pilot-1.2.0"
_PILOT_ISSUER = object()
_PILOT_RUNNER_BINDING_DOMAIN = b"evalsim-m6-waymax-pilot-runner-v2"
_PILOT_OBSERVATION_DOMAIN = b"evalsim-m6-waymax-pilot-observation-v2"


def _pilot_runner_binding_sha256(
    *,
    source_binding_sha256: str,
    selection_binding_sha256: str,
    selected_cohort_indices_sha256: str,
    execution_authority_sha256: str,
) -> str:
    payload = {
        "schema_version": M6_WAYMAX_PILOT_SCHEMA_VERSION,
        "source_binding_sha256": _sha256(
            source_binding_sha256,
            "source_binding_sha256",
        ),
        "selection_binding_sha256": _sha256(
            selection_binding_sha256,
            "selection_binding_sha256",
        ),
        "selected_cohort_indices_sha256": _sha256(
            selected_cohort_indices_sha256,
            "selected_cohort_indices_sha256",
        ),
        "execution_authority_sha256": _sha256(
            execution_authority_sha256,
            "execution_authority_sha256",
        ),
    }
    return hashlib.sha256(
        _PILOT_RUNNER_BINDING_DOMAIN
        + b"\x00"
        + _canonical_json(payload)
    ).hexdigest()


def _pilot_binding_fields(
    source: M6WaymaxOfficialSource,
    execution_authority: M6WaymaxExecutionAuthority,
) -> dict[str, str]:
    source.revalidate()
    execution_authority.revalidate()
    source_binding = source.source_binding_sha256
    selection_binding = m6_waymax_selection_binding_sha256(source.selection)
    selection = source.selection
    if selection.supported:
        selected_indices = tuple(
            member.cohort_index for member in selection.members[:8]
        )
    else:
        selected_indices = tuple(
            row.cohort_index
            for row in sorted(
                source.qualification_ledger.rows,
                key=lambda row: (
                    bytes.fromhex(row.rank_sha256),
                    row.cohort_index,
                ),
            )[:8]
        )
    from evalsim.evaluation.m6_pilot import (
        m6_numpy_pilot_selected_cohort_indices_sha256,
    )

    selected_indices_binding = (
        m6_numpy_pilot_selected_cohort_indices_sha256(selected_indices)
    )
    authority_binding = _sha256(
        execution_authority.authority_sha256,
        "execution_authority_sha256",
    )
    return {
        "source_binding_sha256": source_binding,
        "selection_binding_sha256": selection_binding,
        "selected_cohort_indices_sha256": selected_indices_binding,
        "execution_authority_sha256": authority_binding,
        "runner_binding_sha256": _pilot_runner_binding_sha256(
            source_binding_sha256=source_binding,
            selection_binding_sha256=selection_binding,
            selected_cohort_indices_sha256=selected_indices_binding,
            execution_authority_sha256=authority_binding,
        ),
    }


def _pilot_observation_content_sha256(
    value: "M6WaymaxPilotObservation",
) -> str:
    payload = {
        "status": value.status,
        "scene_count": value.scene_count,
        "validation_ms": value.validation_ms,
        "scene_durations_ms": list(value.scene_durations_ms),
        "execution_ms": value.execution_ms,
        "total_wall_ms": value.total_wall_ms,
        "max_scene_ms": value.max_scene_ms,
        "peak_process_rss_bytes": value.peak_process_rss_bytes,
        "source_binding_sha256": value.source_binding_sha256,
        "selection_binding_sha256": value.selection_binding_sha256,
        "selected_cohort_indices_sha256": (
            value.selected_cohort_indices_sha256
        ),
        "execution_authority_sha256": value.execution_authority_sha256,
        "runner_binding_sha256": value.runner_binding_sha256,
        "schema_version": value.schema_version,
    }
    return hashlib.sha256(
        _PILOT_OBSERVATION_DOMAIN + b"\x00" + _canonical_json(payload)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class M6WaymaxPilotObservation:
    """Outcome-suppressed integer timing facts from a production-only pilot."""

    status: Literal["completed", "unsupported"]
    scene_count: int
    validation_ms: int
    scene_durations_ms: tuple[int, ...]
    execution_ms: int
    total_wall_ms: int
    max_scene_ms: int
    peak_process_rss_bytes: int
    source_binding_sha256: str
    selection_binding_sha256: str
    selected_cohort_indices_sha256: str
    execution_authority_sha256: str
    runner_binding_sha256: str
    schema_version: str = M6_WAYMAX_PILOT_SCHEMA_VERSION
    _issuance_capability: InitVar[object] = None

    def __post_init__(self, _issuance_capability: object) -> None:
        if _issuance_capability is not _PILOT_ISSUER:
            raise TypeError("M6WaymaxPilotObservation is runner-issued only")
        durations = tuple(self.scene_durations_ms)
        object.__setattr__(self, "scene_durations_ms", durations)
        self._validate_semantics()
        content_sha256 = _pilot_observation_content_sha256(self)
        _register_strong_issuance(
            "pilot_observation",
            self,
            binding_sha256=self.runner_binding_sha256,
            content_sha256=content_sha256,
            components=(self.scene_durations_ms,),
        )

    def _validate_semantics(self) -> None:
        if not isinstance(self.scene_durations_ms, tuple):
            raise TypeError("pilot scene durations must be an immutable tuple")
        integers = (
            self.scene_count,
            self.validation_ms,
            self.execution_ms,
            self.total_wall_ms,
            self.max_scene_ms,
            self.peak_process_rss_bytes,
        )
        if any(type(value) is not int or value < 0 for value in integers):
            raise ValueError("pilot facts must be non-negative integers")
        if self.validation_ms <= 0:
            raise ValueError("pilot validation timing must be positive")
        if self.status == "unsupported":
            if (
                self.scene_count != 0
                or self.scene_durations_ms
                or self.execution_ms != 0
                or self.total_wall_ms != 0
                or self.max_scene_ms != 0
                or self.peak_process_rss_bytes != 0
            ):
                raise ValueError(
                    "unsupported pilot can retain only validation timing"
                )
        elif self.status == "completed":
            if (
                self.scene_count != 8
                or len(self.scene_durations_ms) != self.scene_count
                or any(
                    type(value) is not int or value <= 0
                    for value in self.scene_durations_ms
                )
                or self.execution_ms != sum(self.scene_durations_ms)
                or self.max_scene_ms != max(self.scene_durations_ms)
                or self.total_wall_ms
                != self.validation_ms + self.execution_ms
                or self.peak_process_rss_bytes <= 0
            ):
                raise ValueError("completed pilot timing facts are incomplete")
        else:
            raise ValueError("pilot status is not registered")
        for name in (
            "source_binding_sha256",
            "selection_binding_sha256",
            "selected_cohort_indices_sha256",
            "execution_authority_sha256",
            "runner_binding_sha256",
        ):
            _sha256(getattr(self, name), name)
        expected_runner = _pilot_runner_binding_sha256(
            source_binding_sha256=self.source_binding_sha256,
            selection_binding_sha256=self.selection_binding_sha256,
            selected_cohort_indices_sha256=(
                self.selected_cohort_indices_sha256
            ),
            execution_authority_sha256=self.execution_authority_sha256,
        )
        if self.runner_binding_sha256 != expected_runner:
            raise ValueError("pilot runner binding is inconsistent")
        if self.schema_version != M6_WAYMAX_PILOT_SCHEMA_VERSION:
            raise ValueError("pilot schema version is not supported")

    def revalidate(self) -> None:
        record = _strong_issuance_record(
            "pilot_observation",
            self,
            error_code="pilot_observation_mutated",
        )
        _require_original_components(
            record,
            (self.scene_durations_ms,),
            error_code="pilot_observation_mutated",
        )
        try:
            self._validate_semantics()
            content_sha256 = _pilot_observation_content_sha256(self)
        except (TypeError, ValueError) as exc:
            raise M6WaymaxOfficialError(
                "pilot_observation_mutated: pilot observation changed"
            ) from exc
        if (
            self.runner_binding_sha256 != record.binding_sha256
            or content_sha256 != record.content_sha256
        ):
            _fail(
                "pilot_observation_mutated",
                "pilot observation differs from its issued original",
            )

    @property
    def observation_binding_sha256(self) -> str:
        """Registry-held digest of all outcome-free pilot observation content."""

        self.revalidate()
        record = _strong_issuance_record(
            "pilot_observation",
            self,
            error_code="pilot_observation_mutated",
        )
        return record.content_sha256

    def to_summary_fields(self) -> dict[str, int]:
        self.revalidate()
        return {
            "pilot_scene_n": self.scene_count,
            "waymax_validation_ms": self.validation_ms,
            "waymax_ms": self.validation_ms + self.execution_ms,
            "waymax_total_wall_ms": self.total_wall_ms,
            "waymax_max_scene_ms": self.max_scene_ms,
            "peak_process_rss_bytes": self.peak_process_rss_bytes,
        }


def _positive_elapsed_ms(start_ns: int, stop_ns: int) -> int:
    if stop_ns <= start_ns:
        _fail("pilot_clock", "monotonic pilot clock did not advance")
    return (stop_ns - start_ns + 999_999) // 1_000_000


def _peak_process_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if raw <= 0:
        _fail("pilot_rss", "peak process RSS is unavailable")
    return raw if sys.platform == "darwin" else raw * 1024


def _initialize_pilot_runner(
    source_type,
    execution_type,
):
    authority_types = (source_type, execution_type)

    def run(
        source: M6WaymaxOfficialSource,
        execution_authority: M6WaymaxExecutionAuthority,
        *,
        clock_ns: Callable[[], int] | None = None,
    ) -> M6WaymaxPilotObservation:
        """Run the official call grid on the first eight scenes and discard outcomes."""
        if type(source) is not authority_types[0]:
            raise TypeError(
                "source must be exact M6WaymaxOfficialSource"
            )
        if type(execution_authority) is not authority_types[1]:
            raise TypeError(
                "execution_authority must be exact "
                "M6WaymaxExecutionAuthority"
            )

        clock = time.monotonic_ns if clock_ns is None else clock_ns
        if not callable(clock):
            raise TypeError("clock_ns must be callable")

        def now_ns() -> int:
            value = clock()
            if type(value) is not int or value < 0:
                _fail("pilot_clock", "pilot clock must return non-negative integers")
            return value

        validation_start = now_ns()
        source.revalidate()
        execution_authority.revalidate()
        if (
            not source.promotable
            or execution_authority.kind != "pinned_cpu_waymax_jax"
            or not execution_authority.promotable
        ):
            _fail(
                "pilot_authority",
                "Waymax pilot requires verified source and pinned production runtime",
            )
        execution_authority._claim()
        validation_ms = _positive_elapsed_ms(validation_start, now_ns())
        selection = source.selection
        pilot_bindings = _pilot_binding_fields(source, execution_authority)
        if not selection.supported:
            return M6WaymaxPilotObservation(
                status="unsupported",
                scene_count=0,
                validation_ms=validation_ms,
                scene_durations_ms=(),
                execution_ms=0,
                total_wall_ms=0,
                max_scene_ms=0,
                peak_process_rss_bytes=0,
                **pilot_bindings,
                _issuance_capability=_PILOT_ISSUER,
            )

        executors = execution_authority._get_executors()
        primary_domain = source.primary_domain
        durations: list[int] = []
        prior_raw: list[CompactM6WaymaxRollout] = []
        prior_views: list[WaymaxEgoPlanView] = []
        prior_plans: list[EgoTrajectoryPlan] = []
        prior_sources: list[tuple[Any, Scenario]] = []
        for position, expected_member in enumerate(selection.members[:8]):
            scene_start = now_ns()
            state, scenario, qualification = source._materialize(position)
            if (
                qualification.cohort_index != expected_member.cohort_index
                or qualification.qualification_binding_sha256
                != expected_member.qualification_binding_sha256
            ):
                _fail(
                    "pilot_execution_order",
                    "pilot resident differs from canonical selection",
                )
            plans = {
                "identity": compile_identity_plan(scenario),
                "primary_brake": compile_longitudinal_brake_pulse_plan(
                    scenario,
                    PRIMARY_BRAKE_MAGNITUDE_MPS2,
                ),
            }
            for plan in plans.values():
                plan.revalidate()
            by_bundle_condition: dict[
                tuple[str, str],
                tuple[CompactM6WaymaxRollout, WaymaxEgoPlanView],
            ] = {}
            for bundle in M6_WAYMAX_BUNDLES:
                for condition in M6_WAYMAX_OFFICIAL_CONDITIONS:
                    plan = plans[condition]
                    first, first_view = _execute_once(
                        executors.eager,
                        state=state,
                        scenario=scenario,
                        plan=plan,
                        bundle=bundle,
                        selection=selection,
                        primary_domain=primary_domain,
                        selection_position=position,
                        qualification=qualification,
                        prior_raw=prior_raw,
                        prior_views=prior_views,
                        prior_plans=prior_plans,
                        prior_sources=prior_sources,
                    )
                    second, _second_view = _execute_once(
                        executors.eager,
                        state=state,
                        scenario=scenario,
                        plan=plan,
                        bundle=bundle,
                        selection=selection,
                        primary_domain=primary_domain,
                        selection_position=position,
                        qualification=qualification,
                        prior_raw=prior_raw,
                        prior_views=prior_views,
                        prior_plans=prior_plans,
                        prior_sources=prior_sources,
                    )
                    jit_eager = None
                    jit_compiled = None
                    if position == 0:
                        jit_eager, _ = _execute_once(
                            executors.jit_eager,
                            state=state,
                            scenario=scenario,
                            plan=plan,
                            bundle=bundle,
                            selection=selection,
                            primary_domain=primary_domain,
                            selection_position=position,
                            qualification=qualification,
                            prior_raw=prior_raw,
                            prior_views=prior_views,
                            prior_plans=prior_plans,
                            prior_sources=prior_sources,
                        )
                        jit_compiled, _ = _execute_once(
                            executors.jit_compiled,
                            state=state,
                            scenario=scenario,
                            plan=plan,
                            bundle=bundle,
                            selection=selection,
                            primary_domain=primary_domain,
                            selection_position=position,
                            qualification=qualification,
                            prior_raw=prior_raw,
                            prior_views=prior_views,
                            prior_plans=prior_plans,
                            prior_sources=prior_sources,
                        )
                    determinism = M6WaymaxLiveDeterminismExecution(
                        selection_position=position,
                        bundle=bundle,
                        condition=condition,
                        qualification=qualification,
                        eager_pass_1=first,
                        eager_pass_2=second,
                        jit_eager=jit_eager,
                        jit_compiled=jit_compiled,
                    )
                    determinism.revalidate()
                    exact_outputs = [first, second]
                    if jit_eager is not None:
                        exact_outputs.extend((jit_eager, jit_compiled))
                    if len(
                        {_compact_value_sha256(value) for value in exact_outputs}
                    ) != 1:
                        _fail(
                            "pilot_determinism",
                            "pilot eager/JIT compact outputs disagree",
                        )
                    comparisons = _field_comparisons(
                        first,
                        state=state,
                        scenario=scenario,
                        plan=plan,
                        view=first_view,
                        bundle=bundle,
                        qualification=qualification,
                        primary_domain=primary_domain,
                    )
                    if any(
                        comparison.tolerance_failures
                        or comparison.binary_mismatches
                        for comparison in comparisons.values()
                    ):
                        _fail(
                            "pilot_field_comparison",
                            "pilot adapter field comparison failed",
                        )
                    by_bundle_condition[(bundle, condition)] = (
                        first,
                        first_view,
                    )
                baseline, baseline_view = by_bundle_condition[
                    (bundle, "identity")
                ]
                treatment, treatment_view = by_bundle_condition[
                    (bundle, "primary_brake")
                ]
                pair_view = build_m6_waymax_twenty_transition_pair_view(
                    baseline,
                    treatment,
                    selection_position=position,
                    state=state,
                    scenario=scenario,
                    baseline_plan=plans["identity"],
                    treatment_plan=plans["primary_brake"],
                    baseline_view=baseline_view,
                    treatment_view=treatment_view,
                    bundle=bundle,
                    qualification=qualification,
                    primary_domain=primary_domain,
                    selection=selection,
                )
                measures = compute_m6_waymax_paired_measures(pair_view)
                if tuple(result.metric_name for result in measures) != tuple(
                    metric[0] for metric in _NUMPY_METRICS
                ):
                    _fail(
                        "pilot_measure_domain",
                        "pilot paired-measure domain differs from official grid",
                    )
                for result in measures:
                    result.revalidate()
                    if (
                        result.selection_position != position
                        or result.cohort_index != qualification.cohort_index
                        or result.bundle != bundle
                        or result.qualification_binding_sha256
                        != qualification.qualification_binding_sha256
                        or result.view_binding_sha256
                        != pair_view.view_binding_sha256
                    ):
                        _fail(
                            "pilot_measure_binding",
                            "pilot measure differs from its discarded pair view",
                        )
            if (
                source_state_mutation_sha256(state)
                != qualification.source_binding_sha256
            ):
                _fail("pilot_source_mutated", "pilot execution mutated source state")
            durations.append(_positive_elapsed_ms(scene_start, now_ns()))
        source.revalidate()
        execution_ms = sum(durations)
        return M6WaymaxPilotObservation(
            status="completed",
            scene_count=len(durations),
            validation_ms=validation_ms,
            scene_durations_ms=tuple(durations),
            execution_ms=execution_ms,
            total_wall_ms=validation_ms + execution_ms,
            max_scene_ms=max(durations),
            peak_process_rss_bytes=_peak_process_rss_bytes(),
            **pilot_bindings,
            _issuance_capability=_PILOT_ISSUER,
        )

    run.__name__ = "run_m6_waymax_outcome_suppressed_pilot"
    run.__qualname__ = "run_m6_waymax_outcome_suppressed_pilot"
    return run


run_m6_waymax_outcome_suppressed_pilot = _initialize_pilot_runner(
    M6WaymaxOfficialSource,
    M6WaymaxExecutionAuthority,
)
del _initialize_pilot_runner



__all__ = [
    "M6_WAYMAX_NUMPY_COMPARISON_POLICIES",
    "M6_WAYMAX_NUMPY_COMPARISON_ROW_COUNT",
    "M6_WAYMAX_OFFICIAL_COMPARISON_FIELDS",
    "M6_WAYMAX_OFFICIAL_CONDITIONS",
    "M6_WAYMAX_OFFICIAL_EXACT_FIELDS",
    "M6_WAYMAX_OFFICIAL_FIELD_ROW_COUNT",
    "M6_WAYMAX_OFFICIAL_POPULATION_SIZE",
    "M6_WAYMAX_PILOT_SCHEMA_VERSION",
    "M6WaymaxCompactExecutor",
    "M6WaymaxExecutionAuthority",
    "M6WaymaxNonPromotableDeterminismEvidence",
    "M6WaymaxNonPromotableScalarEvidence",
    "M6WaymaxNumpyComparisonRow",
    "M6WaymaxNumpyComparisonTable",
    "M6WaymaxOfficialCollector",
    "M6WaymaxOfficialError",
    "M6WaymaxOfficialEvidence",
    "M6WaymaxOfficialExecutors",
    "M6WaymaxOfficialFieldComparisonRow",
    "M6WaymaxOfficialFieldComparisonTable",
    "M6WaymaxOfficialSource",
    "M6WaymaxOfficialSourceCollector",
    "M6WaymaxPilotObservation",
    "M6WaymaxSourceAuthority",
    "build_m6_waymax_test_execution_authority",
    "build_m6_waymax_test_source_authority",
    "build_m6_waymax_unsupported_field_comparison_table",
    "build_m6_waymax_unsupported_numpy_comparison_table",
    "build_m6_waymax_verified_source_authority",
    "build_pinned_m6_waymax_execution_authority",
    "m6_stored_eligibility_rows_sha256",
    "m6_waymax_selection_binding_sha256",
    "run_m6_waymax_official",
    "run_m6_waymax_outcome_suppressed_pilot",
    "validate_m6_waymax_numpy_comparison_table",
]
