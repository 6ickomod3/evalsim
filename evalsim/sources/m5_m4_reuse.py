"""Fail-closed reuse of the explicitly selected, accepted local M4 run.

The user-selected ignored M4 run directory is the local trust root.  This module
proves that the supplied artifacts are canonical, internally consistent, bound to
the accepted M4 code snapshot, and still agree with the immutable local shards.  It
cannot authenticate the historical review event because the privacy-safe M4
aggregate intentionally did not hash-bind its private manifest or provenance.
Supplying :class:`M4AcceptanceReceipt` strengthens the boundary by pinning the exact
artifact bytes independently.

Nothing in this module scans the ten-shard population or selects a new cohort.
Artifact verification is data-free; optional Waymax imports remain lazy inside the
existing bounded reload functions.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import subprocess
from types import MappingProxyType
from typing import Any, NoReturn

from evalsim.contracts import Scenario

from .waymax import (
    WAYMAX_ADAPTER_SCHEMA_VERSION,
    WAYMAX_ADAPTER_VERSION,
    WAYMAX_COMMIT,
    WOMD_DATASET_VERSION,
)
from .waymax_cohort import (
    CohortInvariantError,
    M4_COHORT_TARGET,
    M4_MANIFEST_SCHEMA_VERSION,
    M4_SELECTOR_CONFIG_FINGERPRINT,
    M4_SELECTOR_VERSION,
    M4_SHARD_SUFFIXES,
    SOURCE_REJECTION_CODES,
    ScanEvent,
    WaymaxCohortManifest,
)
from .waymax_loader import (
    M4ReloadExpectation,
    M4ShardLocator,
    M4StreamRecord,
    WaymaxDataError,
    WaymaxRecord,
    m4_shard_sha256,
    reload_m4_waymax_records,
    resolve_validation_shard,
    runtime_summary,
    verify_m4_stream_record,
)


M5_M4_SELECTED_ORDER_VERSION = "m4-selected-order-1"
M5_M4_REUSE_SCHEMA_VERSION = "m5-m4-reuse-1.0.0"
M4_ACCEPTED_GIT_COMMIT = "a7a20e5de89c9c988f36a4b2f10ff4acc49246f0"
M4_ACCEPTED_GIT_TREE = "0d45328baf57487ed4807ee2f378ee84fe2221a2"
M4_ACCEPTED_REFERENCE_CONFIG_FINGERPRINT = (
    "06f21c37f3b4d950811a60d7b9f67e57515d1f0e69c3fd6ebd57683bf781489e"
)
M4_ACCEPTED_DATASET_CONFIG_FINGERPRINT = (
    "bd535784b399547faf02bf40ad7ea0685627036df5ec2895a3e16c99ae9bf534"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SELECTED_ORDER_DOMAIN = b"evalsim-m5-m4-selected-order-v1\0"
_LOCAL_TRUST_ASSUMPTION = "explicit_ignored_m4_run_is_local_trust_root"
_MAX_SMALL_JSON_BYTES = 1024 * 1024
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024

_M4_ARTIFACT_LIMITS = MappingProxyType(
    {
        "aggregate-summary.json": _MAX_SMALL_JSON_BYTES,
        "cohort/manifest-pass-1.json": _MAX_MANIFEST_BYTES,
        "cohort/manifest-pass-2.json": _MAX_MANIFEST_BYTES,
        "execution-provenance.json": _MAX_SMALL_JSON_BYTES,
        "terminal-output.bin": 0,
    }
)

_M4_ACCEPTED_EXECUTABLE_SHA256 = MappingProxyType(
    {
        "docs/plans/2026-07-28-m4-womd-cohort-waymax-parity.md": (
            "9410fd17ba410ff9d0c3c062e0f05f9e745ec46f450edca8eb4754cd120e8849"
        ),
        "evalsim/rollout/engine.py": (
            "c7a01ed201411c299924aee7732e7fdfafd19703aa7cac0c32266a41be091ee3"
        ),
        "evalsim/simulators/constant_velocity.py": (
            "5178064e0d3faf1a2360a8521c838f19f5f712f2f0b58a1ca3acbeee4bc44212"
        ),
        "evalsim/simulators/idm.py": (
            "952cec7a8ada9a306e780825c77b02242fafb2fdc795e6c54257b6f353e17580"
        ),
        "evalsim/simulators/log_replay.py": (
            "2859ca4d81f1d68d8768f2043db25850c370147f99cc28f151bd25a42cad6d2a"
        ),
        "evalsim/simulators/waymax_reference.py": (
            "e17069aa2c08842f38665504aaabaf902ecb4355a65cba1e587ba04677279c6b"
        ),
        "evalsim/sources/waymax.py": (
            "ecceec2a2b6962555c33e0f9fcb3fdf468afb860e951e37b70686218766f640e"
        ),
        "evalsim/sources/waymax_cohort.py": (
            "01630a52fa953962e65c3dacecbce710ad30be8322f3a6fd12bead90fbaaab76"
        ),
        "evalsim/sources/waymax_loader.py": (
            "8c24d23e33d75d8f230d2c9d419bd4a15045e25b7bd6fa3787d6d536a2b52229"
        ),
        "evalsim/sources/waymax_m4_cli.py": (
            "21e65f9e86efc60b522f8fe08312353291400971cb94d8e101d9e5717313bbc8"
        ),
        "pyproject.toml": (
            "b796b5d3d185a8f11967b4e503d813523ee7885dca6033cd0e4dc39d7c6dd801"
        ),
        "uv.lock": (
            "95500e4dcf00b744124d6364b691587b62cd890e5d4b1f26db5c0fc308b52f57"
        ),
    }
)

_M4_ACCEPTED_RUNTIME = MappingProxyType(
    {
        "flax": "0.10.4",
        "jax": "0.4.38",
        "jaxlib": "0.4.38",
        "numpy": "1.26.4",
        "python": "3.11.5",
        "tensorflow": "2.18.1",
        "waymo_waymax": "0.1.0",
    }
)

_M4_ACCEPTED_COUNTS = (
    ("00000", 286, 148, 138, 13),
    ("00001", 309, 165, 144, 13),
    ("00002", 306, 158, 148, 13),
    ("00003", 307, 151, 156, 13),
    ("00004", 269, 133, 136, 13),
    ("00005", 278, 150, 128, 13),
    ("00006", 285, 154, 131, 13),
    ("00007", 297, 171, 126, 13),
    ("00008", 284, 143, 141, 12),
    ("00009", 295, 154, 141, 12),
)

_M4_ACCEPTED_IDM = MappingProxyType(
    {
        "effective_controlled_transitions": 7_134,
        "horizon_transitions": 20,
        "initialized_overlap_fallback_transitions": 1_333,
        "initialized_overlap_fallback_vehicles": 112,
        "initialized_overlap_vehicle_exclusions_full_cohort": 3_308,
        "lifecycle_fallback_transitions": 6_673,
        "minimum_qualifying_vehicle_effective_transitions": 20,
        "nonfallback_motion_observed": True,
        "qualifying_scenarios": 128,
        "requested_controlled_transitions": 8_467,
        "subset_scenarios": 16,
    }
)

_AGGREGATE_CHECKS = frozenset(
    {
        "adapter_and_independent_parity_full_cohort",
        "evalsim_cv_full_80",
        "evalsim_idm_full_80",
        "evalsim_log_replay_full_80",
        "exact_log_direct_oracle_full_80",
        "exact_log_rollout_conversion_full_cohort",
        "manifest_repeat_byte_identical",
        "selected_locator_reload_complete",
        "stock_waymax_first_selected_one_step",
        "waymax_idm_jit_one_scene",
        "waymax_idm_repeat_byte_identical",
    }
)
_AGGREGATE_PRIVACY = frozenset(
    {
        "absolute_local_values_absent",
        "motion_samples_absent",
        "private_manifests_remain_local",
        "source_hashes_absent",
        "source_identifiers_absent",
    }
)
_BENCHMARK_FIELDS = frozenset(
    {
        "batch_size",
        "compile_seconds",
        "device_transfer_before_timing",
        "eager_sequential_parity",
        "fresh_worker_process",
        "horizon_transitions",
        "jit_vmap",
        "memory_measurement",
        "median_seconds",
        "nearest_rank_p95_seconds",
        "peak_rss_bytes",
        "permutation_invariance",
        "runs",
        "scenarios_per_second_at_median",
        "warm_durations_seconds",
    }
)
_RUNTIME_FIELDS = frozenset(
    {
        "flax",
        "jax",
        "jax_backend",
        "jax_devices",
        "jaxlib",
        "numpy",
        "platform",
        "python",
        "tensorflow",
        "waymo_waymax",
    }
)

_M4_REUSE_ERROR_CODES = frozenset(
    {
        "m4_reuse_adapter_parity_drift",
        "m4_reuse_aggregate_invalid",
        "m4_reuse_artifact_changed",
        "m4_reuse_artifact_layout",
        "m4_reuse_artifact_unsafe",
        "m4_reuse_cohort_mismatch",
        "m4_reuse_incomplete",
        "m4_reuse_json_invalid",
        "m4_reuse_json_noncanonical",
        "m4_reuse_manifest_invalid",
        "m4_reuse_manifest_repeat_mismatch",
        "m4_reuse_not_accepted",
        "m4_reuse_path_invalid",
        "m4_reuse_provenance_invalid",
        "m4_reuse_receipt_mismatch",
        "m4_reuse_reload_mismatch",
        "m4_reuse_runtime_mismatch",
        "m4_reuse_scenario_contract_drift",
        "m4_reuse_snapshot_mismatch",
        "m4_reuse_source_predicate_drift",
    }
)


class M4ReuseError(RuntimeError):
    """Privacy-safe accepted-M4 reuse failure with a stable reason code."""

    def __init__(self, code: str, message: str) -> None:
        if code not in _M4_REUSE_ERROR_CODES:
            raise ValueError("M4ReuseError code is not registered")
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class M4AcceptanceReceipt:
    """Optional independently retained hashes for the accepted local artifacts."""

    aggregate_summary_sha256: str = field(repr=False)
    execution_provenance_sha256: str = field(repr=False)
    manifest_sha256: str = field(repr=False)

    def __post_init__(self) -> None:
        for name in (
            "aggregate_summary_sha256",
            "execution_provenance_sha256",
            "manifest_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be lowercase SHA-256 hex")


@dataclass(frozen=True, slots=True)
class M4ReuseEvidence:
    """Non-payload evidence carried into the M5 evaluation manifest."""

    aggregate_summary_sha256: str = field(repr=False)
    execution_provenance_sha256: str = field(repr=False)
    manifest_sha256: str = field(repr=False)
    selected_order_version: str
    selected_order_fingerprint_sha256: str = field(repr=False)
    accepted_git_commit: str
    accepted_git_tree: str
    runtime_versions: Mapping[str, str]
    integrity_assumption: str = _LOCAL_TRUST_ASSUMPTION
    schema_version: str = M5_M4_REUSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "aggregate_summary_sha256",
            "execution_provenance_sha256",
            "manifest_sha256",
            "selected_order_fingerprint_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be lowercase SHA-256 hex")
        if self.selected_order_version != M5_M4_SELECTED_ORDER_VERSION:
            raise ValueError("selected_order_version is not the frozen M5 value")
        if self.accepted_git_commit != M4_ACCEPTED_GIT_COMMIT:
            raise ValueError("accepted_git_commit is not the frozen M4 commit")
        if self.accepted_git_tree != M4_ACCEPTED_GIT_TREE:
            raise ValueError("accepted_git_tree is not the frozen M4 tree")
        if self.integrity_assumption != _LOCAL_TRUST_ASSUMPTION:
            raise ValueError("integrity_assumption is not the documented local rule")
        if self.schema_version != M5_M4_REUSE_SCHEMA_VERSION:
            raise ValueError("schema_version is not supported")
        runtime = dict(self.runtime_versions)
        if runtime != dict(_M4_ACCEPTED_RUNTIME):
            raise ValueError("runtime_versions differ from accepted M4 evidence")
        object.__setattr__(
            self,
            "runtime_versions",
            MappingProxyType(runtime),
        )


@dataclass(frozen=True, slots=True)
class _ArtifactIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True, slots=True)
class AcceptedM4MemberRef:
    """One private selected member mapped to its public M5 cohort index."""

    cohort_index: int
    event: ScanEvent = field(repr=False)
    expectation: M4ReloadExpectation = field(repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.cohort_index, bool)
            or not isinstance(self.cohort_index, int)
            or not 0 <= self.cohort_index < M4_COHORT_TARGET
        ):
            raise ValueError("cohort_index must lie in the accepted M4 range")
        if not isinstance(self.event, ScanEvent) or self.event.selected is not True:
            raise TypeError("event must be one selected ScanEvent")
        if not isinstance(self.expectation, M4ReloadExpectation):
            raise TypeError("expectation must be an M4ReloadExpectation")


@dataclass(frozen=True, slots=True)
class AcceptedM4Cohort:
    """Verified local reuse contract; private fields are intentionally hidden."""

    evidence: M4ReuseEvidence = field(repr=False)
    members: tuple[AcceptedM4MemberRef, ...] = field(repr=False)
    manifest: WaymaxCohortManifest = field(repr=False)
    project_root: Path = field(repr=False)
    run_dir: Path = field(repr=False)
    artifacts: Mapping[str, _ArtifactIdentity] = field(repr=False)
    receipt_verified: bool = False

    def __post_init__(self) -> None:
        members = tuple(self.members)
        if (
            len(members) != M4_COHORT_TARGET
            or tuple(item.cohort_index for item in members)
            != tuple(range(M4_COHORT_TARGET))
        ):
            raise ValueError("members must cover canonical cohort indices 0..127")
        if not isinstance(self.manifest, WaymaxCohortManifest):
            raise TypeError("manifest must be a WaymaxCohortManifest")
        if type(self.receipt_verified) is not bool:
            raise TypeError("receipt_verified must be a boolean")
        object.__setattr__(self, "members", members)
        object.__setattr__(
            self,
            "artifacts",
            MappingProxyType(dict(self.artifacts)),
        )


@dataclass(frozen=True, slots=True)
class ReloadedM4Member:
    """Transient verified source objects for one opaque M5 cohort index."""

    cohort_index: int
    scenario: Scenario = field(repr=False)
    record: WaymaxRecord = field(repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.cohort_index, bool)
            or not isinstance(self.cohort_index, int)
            or not 0 <= self.cohort_index < M4_COHORT_TARGET
        ):
            raise ValueError("cohort_index must lie in the accepted M4 range")
        if not isinstance(self.scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        if not isinstance(self.record, WaymaxRecord):
            raise TypeError("record must be a WaymaxRecord")


def verify_accepted_m4_run(
    project_root: str | Path,
    m4_run_dir: str | Path,
    *,
    expected_receipt: M4AcceptanceReceipt | None = None,
) -> AcceptedM4Cohort:
    """Verify an explicit accepted ignored M4 run without touching WOMD data."""

    if expected_receipt is not None and not isinstance(
        expected_receipt,
        M4AcceptanceReceipt,
    ):
        raise TypeError("expected_receipt must be M4AcceptanceReceipt or None")
    root = _validated_project_root(project_root)
    run_dir = _validated_run_dir(root, m4_run_dir)
    _validate_artifact_layout(run_dir)

    encoded: dict[str, bytes] = {}
    identities: dict[str, _ArtifactIdentity] = {}
    for relative, maximum in _M4_ARTIFACT_LIMITS.items():
        payload, identity = _guarded_read(
            run_dir.joinpath(*Path(relative).parts),
            maximum_bytes=maximum,
            exact_size=0 if relative == "terminal-output.bin" else None,
        )
        encoded[relative] = payload
        identities[relative] = identity

    provenance = _parse_pretty_json(
        encoded["execution-provenance.json"],
    )
    _validate_execution_provenance(root, provenance)

    first_manifest = _parse_manifest(
        encoded["cohort/manifest-pass-1.json"]
    )
    second_manifest = _parse_manifest(
        encoded["cohort/manifest-pass-2.json"]
    )
    if (
        encoded["cohort/manifest-pass-1.json"]
        != encoded["cohort/manifest-pass-2.json"]
        or first_manifest != second_manifest
    ):
        _fail(
            "m4_reuse_manifest_repeat_mismatch",
            "the two accepted M4 manifest passes are not byte-identical",
        )
    _validate_accepted_cohort(first_manifest)

    aggregate = _parse_pretty_json(encoded["aggregate-summary.json"])
    runtime_versions = _validate_aggregate(aggregate, first_manifest)

    manifest_sha256 = hashlib.sha256(
        encoded["cohort/manifest-pass-1.json"]
    ).hexdigest()
    provenance_sha256 = hashlib.sha256(
        encoded["execution-provenance.json"]
    ).hexdigest()
    aggregate_sha256 = hashlib.sha256(
        encoded["aggregate-summary.json"]
    ).hexdigest()
    if expected_receipt is not None and (
        expected_receipt.aggregate_summary_sha256 != aggregate_sha256
        or expected_receipt.execution_provenance_sha256
        != provenance_sha256
        or expected_receipt.manifest_sha256 != manifest_sha256
    ):
        _fail(
            "m4_reuse_receipt_mismatch",
            "the accepted M4 artifacts differ from the supplied local receipt",
        )

    selected = first_manifest.selected_events
    selected_fingerprint = _selected_order_fingerprint(selected)
    members: list[AcceptedM4MemberRef] = []
    try:
        for cohort_index, event in enumerate(selected):
            expectation = M4ReloadExpectation(
                locator=M4ShardLocator(
                    shard_suffix=event.shard_suffix,
                    record_ordinal=event.record_ordinal,
                ),
                expected_scenario_id=event.native_scenario_id,
                expected_shard_sha256=event.shard_sha256,
                expected_dataset_config_fingerprint=(
                    event.dataset_config_fingerprint
                ),
            )
            members.append(
                AcceptedM4MemberRef(
                    cohort_index=cohort_index,
                    event=event,
                    expectation=expectation,
                )
            )
    except (TypeError, ValueError) as exc:
        raise M4ReuseError(
            "m4_reuse_cohort_mismatch",
            "a selected M4 member cannot form an exact reload expectation",
        ) from exc

    evidence = M4ReuseEvidence(
        aggregate_summary_sha256=aggregate_sha256,
        execution_provenance_sha256=provenance_sha256,
        manifest_sha256=manifest_sha256,
        selected_order_version=M5_M4_SELECTED_ORDER_VERSION,
        selected_order_fingerprint_sha256=selected_fingerprint,
        accepted_git_commit=M4_ACCEPTED_GIT_COMMIT,
        accepted_git_tree=M4_ACCEPTED_GIT_TREE,
        runtime_versions=runtime_versions,
    )
    return AcceptedM4Cohort(
        evidence=evidence,
        members=tuple(members),
        manifest=first_manifest,
        project_root=root,
        run_dir=run_dir,
        artifacts=identities,
        receipt_verified=expected_receipt is not None,
    )


def visit_accepted_m4_cohort(
    cohort: AcceptedM4Cohort,
    data_dir: str | Path,
    visitor: Callable[[ReloadedM4Member], None],
) -> None:
    """Reload and visit all 128 accepted members without scanning or replacement."""

    if not isinstance(cohort, AcceptedM4Cohort):
        raise TypeError("cohort must be an AcceptedM4Cohort")
    if not callable(visitor):
        raise TypeError("visitor must be callable")
    _validate_current_runtime()

    members_by_suffix: dict[str, list[AcceptedM4MemberRef]] = {
        suffix: [] for suffix in M4_SHARD_SUFFIXES
    }
    for member in cohort.members:
        members_by_suffix[member.event.shard_suffix].append(member)

    seen_indices: set[int] = set()
    seen_locators: set[tuple[str, int]] = set()
    for suffix in M4_SHARD_SUFFIXES:
        expected_members = members_by_suffix[suffix]
        if not expected_members:
            _fail(
                "m4_reuse_incomplete",
                "the accepted cohort does not cover every frozen M4 shard",
            )
        try:
            records = reload_m4_waymax_records(
                data_dir,
                tuple(item.expectation for item in expected_members),
            )
        except Exception as exc:
            raise M4ReuseError(
                "m4_reuse_reload_mismatch",
                "a bounded selected-record reload failed its exact source checks",
            ) from exc
        if len(records) != len(expected_members):
            _fail(
                "m4_reuse_reload_mismatch",
                "a bounded selected-record reload returned the wrong count",
            )
        for member, stream_record in zip(
            expected_members,
            records,
            strict=True,
        ):
            _visit_reloaded_member(
                member,
                stream_record,
                visitor,
                seen_indices,
                seen_locators,
            )

    if (
        seen_indices != set(range(M4_COHORT_TARGET))
        or len(seen_locators) != M4_COHORT_TARGET
    ):
        _fail(
            "m4_reuse_incomplete",
            "the bounded reload did not visit every accepted cohort member once",
        )
    _verify_shards_unchanged(cohort.manifest, data_dir)
    reverify_accepted_m4_run(cohort)


def reverify_accepted_m4_run(cohort: AcceptedM4Cohort) -> None:
    """Prove the ignored acceptance artifacts have not changed since preflight."""

    if not isinstance(cohort, AcceptedM4Cohort):
        raise TypeError("cohort must be an AcceptedM4Cohort")
    receipt = M4AcceptanceReceipt(
        aggregate_summary_sha256=(
            cohort.evidence.aggregate_summary_sha256
        ),
        execution_provenance_sha256=(
            cohort.evidence.execution_provenance_sha256
        ),
        manifest_sha256=cohort.evidence.manifest_sha256,
    )
    fresh = verify_accepted_m4_run(
        cohort.project_root,
        cohort.run_dir,
        expected_receipt=receipt,
    )
    if (
        fresh.evidence != cohort.evidence
        or fresh.members != cohort.members
        or dict(fresh.artifacts) != dict(cohort.artifacts)
    ):
        _fail(
            "m4_reuse_artifact_changed",
            "the accepted M4 artifact identities changed after preflight",
        )


def _visit_reloaded_member(
    member: AcceptedM4MemberRef,
    stream_record: M4StreamRecord,
    visitor: Callable[[ReloadedM4Member], None],
    seen_indices: set[int],
    seen_locators: set[tuple[str, int]],
) -> None:
    if not isinstance(stream_record, M4StreamRecord):
        _fail(
            "m4_reuse_reload_mismatch",
            "the bounded reload returned an invalid record type",
        )
    if stream_record.locator != member.expectation.locator:
        _fail(
            "m4_reuse_reload_mismatch",
            "the bounded reload changed canonical member order",
        )
    try:
        event, scenario = verify_m4_stream_record(stream_record)
    except CohortInvariantError as exc:
        raise M4ReuseError(
            "m4_reuse_source_predicate_drift",
            "the reloaded source predicate no longer reproduces the manifest",
        ) from exc
    except WaymaxDataError as exc:
        raise M4ReuseError(
            "m4_reuse_adapter_parity_drift",
            "the reloaded adapter or independent parity gate failed",
        ) from exc
    except Exception as exc:
        raise M4ReuseError(
            "m4_reuse_adapter_parity_drift",
            "the reloaded record failed conversion or parity validation",
        ) from exc

    expected_event = replace(member.event, selected=False)
    if event.outcome != "eligible" or event != expected_event or scenario is None:
        _fail(
            "m4_reuse_source_predicate_drift",
            "the reloaded event no longer matches the accepted selected event",
        )
    if (
        scenario.num_steps != 91
        or scenario.metadata.get("current_index") != 10
        or scenario.scenario_id != stream_record.record.scenario_id
    ):
        _fail(
            "m4_reuse_scenario_contract_drift",
            "the reloaded Scenario changed the frozen M4 temporal or identity contract",
        )

    locator = member.event.locator
    if member.cohort_index in seen_indices or locator in seen_locators:
        _fail(
            "m4_reuse_incomplete",
            "the bounded reload visited an accepted member more than once",
        )
    visitor(
        ReloadedM4Member(
            cohort_index=member.cohort_index,
            scenario=scenario,
            record=stream_record.record,
        )
    )
    seen_indices.add(member.cohort_index)
    seen_locators.add(locator)


def _verify_shards_unchanged(
    manifest: WaymaxCohortManifest,
    data_dir: str | Path,
) -> None:
    expected_by_suffix: dict[str, str] = {}
    for event in manifest.events:
        expected_by_suffix.setdefault(event.shard_suffix, event.shard_sha256)
    try:
        for suffix in M4_SHARD_SUFFIXES:
            shard_path = resolve_validation_shard(
                data_dir,
                shard_index=int(suffix),
            )
            if m4_shard_sha256(shard_path) != expected_by_suffix[suffix]:
                _fail(
                    "m4_reuse_reload_mismatch",
                    "an immutable M4 shard digest changed after bounded reload",
                )
    except M4ReuseError:
        raise
    except Exception as exc:
        raise M4ReuseError(
            "m4_reuse_reload_mismatch",
            "the immutable M4 shards failed end-of-reload verification",
        ) from exc


def _validated_project_root(candidate: str | Path) -> Path:
    try:
        lexical = Path(candidate)
    except TypeError as exc:
        raise M4ReuseError(
            "m4_reuse_path_invalid",
            "the explicit project root is invalid",
        ) from exc
    try:
        root = lexical.resolve(strict=True)
    except OSError as exc:
        raise M4ReuseError(
            "m4_reuse_path_invalid",
            "the explicit project root does not exist",
        ) from exc
    if (
        not root.is_dir()
        or not (root / "pyproject.toml").is_file()
        or not (root / ".gitignore").is_file()
    ):
        _fail(
            "m4_reuse_path_invalid",
            "the explicit project root is not an EvalSim checkout",
        )
    top_level = _git_text(root, "rev-parse", "--show-toplevel")
    try:
        if Path(top_level).resolve(strict=True) != root:
            _fail(
                "m4_reuse_path_invalid",
                "the explicit project root is not the Git worktree root",
            )
    except OSError as exc:
        raise M4ReuseError(
            "m4_reuse_path_invalid",
            "the Git worktree root cannot be resolved",
        ) from exc
    return root


def _validated_run_dir(root: Path, candidate: str | Path) -> Path:
    try:
        raw = Path(candidate)
    except TypeError as exc:
        raise M4ReuseError(
            "m4_reuse_path_invalid",
            "the explicit M4 run directory is invalid",
        ) from exc
    lexical = raw if raw.is_absolute() else root / raw
    lexical = Path(os.path.abspath(os.fspath(lexical)))
    try:
        run_dir = lexical.resolve(strict=True)
        allowed = (root / "outputs" / "m4").resolve(strict=True)
    except OSError as exc:
        raise M4ReuseError(
            "m4_reuse_path_invalid",
            "the explicit M4 run directory does not exist",
        ) from exc
    try:
        relative = run_dir.relative_to(allowed)
    except ValueError as exc:
        raise M4ReuseError(
            "m4_reuse_path_invalid",
            "the explicit M4 run directory is outside outputs/m4",
        ) from exc
    if (
        lexical != run_dir
        or run_dir == allowed
        or not run_dir.is_dir()
        or not relative.parts
        or any(
            _SAFE_COMPONENT.fullmatch(part) is None
            for part in relative.parts
        )
    ):
        _fail(
            "m4_reuse_path_invalid",
            "the explicit M4 run directory is linked or noncanonical",
        )
    status = _git_process(
        root,
        "check-ignore",
        "--quiet",
        "--no-index",
        "--",
        run_dir.relative_to(root).as_posix(),
    )
    if status.returncode != 0:
        _fail(
            "m4_reuse_path_invalid",
            "the explicit M4 run directory is not ignored by Git",
        )
    relative_run = run_dir.relative_to(root).as_posix()
    tracked = _git_process(
        root,
        "ls-files",
        "-z",
        "--",
        relative_run,
    )
    if tracked.returncode != 0:
        _fail(
            "m4_reuse_path_invalid",
            "Git cannot inspect the explicit M4 run tracking boundary",
        )
    if tracked.stdout:
        _fail(
            "m4_reuse_path_invalid",
            "the explicit M4 run directory contains tracked files",
        )
    visible = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored=no",
        "--",
        run_dir.relative_to(root).as_posix(),
    )
    if visible:
        _fail(
            "m4_reuse_path_invalid",
            "the explicit M4 run directory is visible to Git",
        )
    return run_dir


def _validate_artifact_layout(run_dir: Path) -> None:
    try:
        root_entries = {entry.name: entry for entry in os.scandir(run_dir)}
    except OSError as exc:
        raise M4ReuseError(
            "m4_reuse_artifact_layout",
            "the accepted M4 run directory cannot be enumerated",
        ) from exc
    expected_root = {
        "aggregate-summary.json",
        "cohort",
        "execution-provenance.json",
        "terminal-output.bin",
    }
    if set(root_entries) != expected_root:
        _fail(
            "m4_reuse_artifact_layout",
            "the accepted M4 run has missing or unexpected members",
        )
    cohort_entry = root_entries["cohort"]
    if cohort_entry.is_symlink() or not cohort_entry.is_dir(follow_symlinks=False):
        _fail(
            "m4_reuse_artifact_unsafe",
            "the accepted M4 cohort member is not a real directory",
        )
    cohort_dir = run_dir / "cohort"
    if cohort_dir.resolve(strict=True) != cohort_dir:
        _fail(
            "m4_reuse_artifact_unsafe",
            "the accepted M4 cohort directory is linked",
        )
    try:
        cohort_entries = {
            entry.name: entry for entry in os.scandir(cohort_dir)
        }
    except OSError as exc:
        raise M4ReuseError(
            "m4_reuse_artifact_layout",
            "the accepted M4 cohort directory cannot be enumerated",
        ) from exc
    if set(cohort_entries) != {
        "manifest-pass-1.json",
        "manifest-pass-2.json",
    }:
        _fail(
            "m4_reuse_artifact_layout",
            "the accepted M4 cohort directory has unexpected members",
        )
    for name, entry in root_entries.items():
        if name == "cohort":
            continue
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            _fail(
                "m4_reuse_artifact_unsafe",
                "an accepted M4 artifact is not a real regular file",
            )
    for entry in cohort_entries.values():
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            _fail(
                "m4_reuse_artifact_unsafe",
                "an accepted M4 manifest is not a real regular file",
            )


def _guarded_read(
    path: Path,
    *,
    maximum_bytes: int,
    exact_size: int | None = None,
) -> tuple[bytes, _ArtifactIdentity]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(
                "m4_reuse_artifact_unsafe",
                "an accepted M4 artifact is not a regular file",
            )
        if (
            before.st_size > maximum_bytes
            or (exact_size is not None and before.st_size != exact_size)
        ):
            _fail(
                "m4_reuse_artifact_unsafe",
                "an accepted M4 artifact has an invalid bounded size",
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                _fail(
                    "m4_reuse_artifact_changed",
                    "an accepted M4 artifact changed while it was read",
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        latest = os.lstat(path)
    except M4ReuseError:
        raise
    except OSError as exc:
        raise M4ReuseError(
            "m4_reuse_artifact_unsafe",
            "an accepted M4 artifact cannot be read safely",
        ) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
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
    latest_identity = (
        latest.st_dev,
        latest.st_ino,
        latest.st_size,
        latest.st_mtime_ns,
    )
    if (
        before_identity != after_identity
        or after_identity != latest_identity
        or len(payload) != before.st_size
    ):
        _fail(
            "m4_reuse_artifact_changed",
            "an accepted M4 artifact changed while it was read",
        )
    return payload, _ArtifactIdentity(
        device=int(before.st_dev),
        inode=int(before.st_ino),
        size=int(before.st_size),
        mtime_ns=int(before.st_mtime_ns),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _parse_pretty_json(encoded: bytes) -> dict[str, Any]:
    try:
        text = encoded.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise M4ReuseError(
            "m4_reuse_json_invalid",
            "an accepted M4 JSON artifact is not strict UTF-8",
        ) from exc
    payload = _strict_json_loads(text)
    if type(payload) is not dict:
        _fail(
            "m4_reuse_json_invalid",
            "an accepted M4 JSON artifact must contain one object",
        )
    if _pretty_json_bytes(payload) != encoded:
        _fail(
            "m4_reuse_json_noncanonical",
            "an accepted M4 JSON artifact is not canonically encoded",
        )
    return payload


def _strict_json_loads(text: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(
                    "m4_reuse_json_invalid",
                    "an accepted M4 JSON artifact contains a duplicate key",
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        del value
        _fail(
            "m4_reuse_json_invalid",
            "an accepted M4 JSON artifact contains a non-finite constant",
        )

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except M4ReuseError:
        raise
    except (RecursionError, json.JSONDecodeError) as exc:
        raise M4ReuseError(
            "m4_reuse_json_invalid",
            "an accepted M4 JSON artifact is invalid",
        ) from exc


def _pretty_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8", errors="strict")
    except (RecursionError, TypeError, ValueError, UnicodeEncodeError) as exc:
        raise M4ReuseError(
            "m4_reuse_json_invalid",
            "an accepted M4 JSON artifact cannot be canonically encoded",
        ) from exc


def _parse_manifest(encoded: bytes) -> WaymaxCohortManifest:
    try:
        text = encoded.decode("utf-8", errors="strict")
        if not text.endswith("\n"):
            _fail(
                "m4_reuse_json_noncanonical",
                "an accepted M4 manifest lacks its canonical newline",
            )
        manifest = WaymaxCohortManifest.from_json(text[:-1])
        if manifest.canonical_bytes() != encoded:
            _fail(
                "m4_reuse_json_noncanonical",
                "an accepted M4 manifest differs from canonical bytes",
            )
        return manifest
    except M4ReuseError:
        raise
    except (UnicodeDecodeError, CohortInvariantError) as exc:
        raise M4ReuseError(
            "m4_reuse_manifest_invalid",
            "an accepted M4 manifest failed its frozen schema",
        ) from exc


def _validate_execution_provenance(
    root: Path,
    payload: Mapping[str, Any],
) -> None:
    required = {
        "files",
        "git_commit",
        "git_tree",
        "reference_config_fingerprint",
        "schema_version",
        "selector_config_fingerprint",
    }
    files = payload.get("files")
    if (
        type(payload) is not dict
        or set(payload) != required
        or type(files) is not dict
        or set(files) != set(_M4_ACCEPTED_EXECUTABLE_SHA256)
        or any(
            not isinstance(value, str)
            or _SHA256.fullmatch(value) is None
            for value in files.values()
        )
    ):
        _fail(
            "m4_reuse_provenance_invalid",
            "execution provenance has an invalid frozen schema",
        )
    expected = {
        "files": dict(_M4_ACCEPTED_EXECUTABLE_SHA256),
        "git_commit": M4_ACCEPTED_GIT_COMMIT,
        "git_tree": M4_ACCEPTED_GIT_TREE,
        "reference_config_fingerprint": (
            M4_ACCEPTED_REFERENCE_CONFIG_FINGERPRINT
        ),
        "schema_version": "1",
        "selector_config_fingerprint": M4_SELECTOR_CONFIG_FINGERPRINT,
    }
    if payload != expected:
        _fail(
            "m4_reuse_snapshot_mismatch",
            "execution provenance differs from the accepted M4 snapshot",
        )
    _verify_historical_snapshot(root)


def _verify_historical_snapshot(root: Path) -> None:
    commit = _git_text(
        root,
        "rev-parse",
        "--verify",
        f"{M4_ACCEPTED_GIT_COMMIT}^{{commit}}",
    )
    tree = _git_text(
        root,
        "rev-parse",
        "--verify",
        f"{M4_ACCEPTED_GIT_COMMIT}^{{tree}}",
    )
    if commit != M4_ACCEPTED_GIT_COMMIT or tree != M4_ACCEPTED_GIT_TREE:
        _fail(
            "m4_reuse_snapshot_mismatch",
            "the accepted historical M4 Git object differs from its frozen identity",
        )
    for relative, expected_sha256 in _M4_ACCEPTED_EXECUTABLE_SHA256.items():
        blob = _git_bytes(
            root,
            "show",
            f"{M4_ACCEPTED_GIT_COMMIT}:{relative}",
        )
        if hashlib.sha256(blob).hexdigest() != expected_sha256:
            _fail(
                "m4_reuse_snapshot_mismatch",
                "an accepted historical M4 executable blob changed",
            )


def _validate_accepted_cohort(manifest: WaymaxCohortManifest) -> None:
    if (
        manifest.schema_version != M4_MANIFEST_SCHEMA_VERSION
        or manifest.selector_version != M4_SELECTOR_VERSION
        or manifest.selector_config_fingerprint
        != M4_SELECTOR_CONFIG_FINGERPRINT
        or manifest.selection.fallback_used is not False
        or manifest.selection.redistributed_count != 0
        or any(count != 0 for _, count in manifest.selection.quota_deficits)
        or manifest.selection.actual_count != M4_COHORT_TARGET
    ):
        _fail(
            "m4_reuse_cohort_mismatch",
            "the manifest does not describe the accepted fixed M4 cohort",
        )
    selected_by_shard = Counter(
        event.shard_suffix for event in manifest.selected_events
    )
    observed_counts = tuple(
        (
            count.shard_suffix,
            count.raw_seen,
            count.eligible,
            count.rejected,
            selected_by_shard[count.shard_suffix],
        )
        for count in manifest.shard_counts
    )
    if observed_counts != _M4_ACCEPTED_COUNTS:
        _fail(
            "m4_reuse_cohort_mismatch",
            "the manifest population accounting differs from accepted M4 evidence",
        )
    rejection_counts = Counter(
        event.rejection_code
        for event in manifest.events
        if event.outcome == "rejected"
    )
    expected_rejections = {
        code: 1_389 if code == "source_no_supported_map" else 0
        for code in SOURCE_REJECTION_CODES
    }
    if {
        code: int(rejection_counts[code]) for code in SOURCE_REJECTION_CODES
    } != expected_rejections:
        _fail(
            "m4_reuse_cohort_mismatch",
            "the source rejection accounting differs from accepted M4 evidence",
        )
    for event in manifest.events:
        if (
            event.dataset_config_fingerprint
            != M4_ACCEPTED_DATASET_CONFIG_FINGERPRINT
            or event.womd_dataset_version != WOMD_DATASET_VERSION
            or event.waymax_commit != WAYMAX_COMMIT
            or event.adapter_version != WAYMAX_ADAPTER_VERSION
            or event.adapter_schema_version != WAYMAX_ADAPTER_SCHEMA_VERSION
        ):
            _fail(
                "m4_reuse_cohort_mismatch",
                "an M4 event differs from accepted dataset or adapter provenance",
            )


def _validate_aggregate(
    payload: Mapping[str, Any],
    manifest: WaymaxCohortManifest,
) -> dict[str, str]:
    _require_exact_keys(
        payload,
        {
            "accepted",
            "benchmark",
            "checks",
            "cohort",
            "idm",
            "privacy",
            "purpose",
            "runtime",
            "schema_version",
            "shared_decode_limitation",
        },
        "m4_reuse_aggregate_invalid",
    )
    if payload["accepted"] is not True:
        _fail(
            "m4_reuse_not_accepted",
            "the selected M4 run is not marked accepted",
        )
    if (
        payload["schema_version"] != "1"
        or payload["purpose"]
        != "personal_non_commercial_interview_preparation"
        or payload["shared_decode_limitation"]
        != "EvalSim and Waymax reference paths share the pinned Waymax WOMD decode"
    ):
        _fail(
            "m4_reuse_not_accepted",
            "the selected M4 run has a different schema, purpose, or limitation",
        )
    checks = payload["checks"]
    privacy = payload["privacy"]
    if (
        type(checks) is not dict
        or set(checks) != _AGGREGATE_CHECKS
        or any(value is not True for value in checks.values())
        or type(privacy) is not dict
        or set(privacy) != _AGGREGATE_PRIVACY
        or any(value is not True for value in privacy.values())
    ):
        _fail(
            "m4_reuse_not_accepted",
            "the selected M4 run does not retain every acceptance/privacy gate",
        )
    if not _exact_json_equal(
        payload["cohort"],
        _cohort_summary(manifest),
    ):
        _fail(
            "m4_reuse_aggregate_invalid",
            "the accepted aggregate contradicts its private canonical manifest",
        )
    _validate_benchmark(payload["benchmark"])
    if not _exact_json_equal(payload["idm"], dict(_M4_ACCEPTED_IDM)):
        _fail(
            "m4_reuse_aggregate_invalid",
            "the accepted aggregate IDM accounting differs from tracked evidence",
        )
    return _validate_runtime(payload["runtime"])


def _validate_benchmark(value: Any) -> None:
    if type(value) is not dict or set(value) != _BENCHMARK_FIELDS:
        _fail(
            "m4_reuse_aggregate_invalid",
            "the accepted benchmark report has an invalid schema",
        )
    if (
        type(value["batch_size"]) is not int
        or value["batch_size"] != 2
        or type(value["horizon_transitions"]) is not int
        or value["horizon_transitions"] != 80
        or type(value["runs"]) is not int
        or value["runs"] != 20
        or value["memory_measurement"]
        != "process_high_water_rss_not_jax_device_memory"
        or any(
            value[name] is not True
            for name in (
                "device_transfer_before_timing",
                "eager_sequential_parity",
                "fresh_worker_process",
                "jit_vmap",
                "permutation_invariance",
            )
        )
    ):
        _fail(
            "m4_reuse_aggregate_invalid",
            "the accepted benchmark report lost a frozen execution gate",
        )
    durations = value["warm_durations_seconds"]
    if (
        type(durations) is not list
        or len(durations) != 20
        or any(not _positive_builtin_float(item) for item in durations)
        or not _positive_builtin_float(value["compile_seconds"])
        or not _positive_builtin_float(value["median_seconds"])
        or not _positive_builtin_float(value["nearest_rank_p95_seconds"])
        or not _positive_builtin_float(
            value["scenarios_per_second_at_median"]
        )
        or isinstance(value["peak_rss_bytes"], bool)
        or type(value["peak_rss_bytes"]) is not int
        or value["peak_rss_bytes"] <= 0
    ):
        _fail(
            "m4_reuse_aggregate_invalid",
            "the accepted benchmark report has invalid finite measurements",
        )
    ordered = sorted(durations)
    median = float((ordered[9] + ordered[10]) / 2.0)
    p95 = float(ordered[18])
    if (
        value["median_seconds"] != median
        or value["nearest_rank_p95_seconds"] != p95
        or value["scenarios_per_second_at_median"] != 2 / median
    ):
        _fail(
            "m4_reuse_aggregate_invalid",
            "the accepted benchmark summaries contradict retained durations",
        )


def _validate_runtime(value: Any) -> dict[str, str]:
    if type(value) is not dict or set(value) != _RUNTIME_FIELDS:
        _fail(
            "m4_reuse_runtime_mismatch",
            "the accepted M4 runtime report has an invalid schema",
        )
    if (
        type(value["platform"]) is not str
        or not value["platform"]
        or Path(value["platform"]).is_absolute()
        or PureWindowsPath(value["platform"]).is_absolute()
        or value["jax_backend"] != "cpu"
        or not _exact_json_equal(value["jax_devices"], ["cpu"])
    ):
        _fail(
            "m4_reuse_runtime_mismatch",
            "the accepted M4 runtime is not the frozen CPU environment",
        )
    versions = {name: value[name] for name in _M4_ACCEPTED_RUNTIME}
    if not _exact_json_equal(versions, dict(_M4_ACCEPTED_RUNTIME)):
        _fail(
            "m4_reuse_runtime_mismatch",
            "the accepted M4 dependency versions differ from tracked evidence",
        )
    return versions


def _validate_current_runtime() -> None:
    try:
        _validate_runtime(runtime_summary())
        return
    except M4ReuseError:
        raise
    except Exception:
        # Drop arbitrary optional-runtime exception text at this boundary.  The
        # official caller may expose this error, while KeyboardInterrupt and
        # SystemExit remain outside Exception and still propagate.
        pass
    raise M4ReuseError(
        "m4_reuse_runtime_mismatch",
        "the current optional runtime cannot reproduce accepted M4",
    )


def _cohort_summary(manifest: WaymaxCohortManifest) -> dict[str, Any]:
    rejections = Counter(
        event.rejection_code
        for event in manifest.events
        if event.outcome == "rejected"
    )
    selected_by_shard = Counter(
        event.shard_suffix for event in manifest.selected_events
    )
    return {
        "cohort_label": (
            "complete-case conditional sample from exactly the first ten "
            "WOMD validation shards; not random or representative"
        ),
        "fallback_used": manifest.selection.fallback_used,
        "per_shard": [
            {
                "eligible": count.eligible,
                "raw": count.raw_seen,
                "rejected": count.rejected,
                "selected": selected_by_shard[count.shard_suffix],
                "shard_suffix": count.shard_suffix,
            }
            for count in manifest.shard_counts
        ],
        "quota_deficits": {
            suffix: deficit
            for suffix, deficit in manifest.selection.quota_deficits
        },
        "redistributed_count": manifest.selection.redistributed_count,
        "rejection_counts": {
            code: int(rejections[code]) for code in SOURCE_REJECTION_CODES
        },
        "selected": len(manifest.selected_events),
        "total_eligible": sum(
            count.eligible for count in manifest.shard_counts
        ),
        "total_raw": sum(count.raw_seen for count in manifest.shard_counts),
        "total_rejected": sum(
            count.rejected for count in manifest.shard_counts
        ),
    }


def _selected_order_fingerprint(events: tuple[ScanEvent, ...]) -> str:
    payload = {
        "members": [
            {
                "cohort_index": cohort_index,
                **event.to_dict(),
            }
            for cohort_index, event in enumerate(events)
        ],
        "version": M5_M4_SELECTED_ORDER_VERSION,
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
        raise M4ReuseError(
            "m4_reuse_cohort_mismatch",
            "the selected M4 order cannot be canonically fingerprinted",
        ) from exc
    digest = hashlib.sha256(_SELECTED_ORDER_DOMAIN)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    return digest.hexdigest()


def _require_exact_keys(
    value: Any,
    expected: set[str],
    code: str,
) -> None:
    if type(value) is not dict or set(value) != expected:
        _fail(code, "an accepted M4 artifact has an unexpected field set")


def _positive_builtin_float(value: Any) -> bool:
    return type(value) is float and math.isfinite(value) and value > 0.0


def _exact_json_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON values without Python's bool/int/float coercions."""

    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        if set(actual) != set(expected):
            return False
        return all(
            _exact_json_equal(actual[key], expected[key])
            for key in expected
        )
    if type(actual) is list:
        return len(actual) == len(expected) and all(
            _exact_json_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return bool(actual == expected)


def _git_process(
    root: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ("git", "-C", os.fspath(root), *arguments),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise M4ReuseError(
            "m4_reuse_path_invalid",
            "Git cannot inspect the local M4 trust boundary",
        ) from exc


def _git_bytes(root: Path, *arguments: str) -> bytes:
    completed = _git_process(root, *arguments)
    if completed.returncode != 0:
        _fail(
            "m4_reuse_snapshot_mismatch",
            "Git cannot verify the accepted historical M4 snapshot",
        )
    return completed.stdout


def _git_text(root: Path, *arguments: str) -> str:
    try:
        value = _git_bytes(root, *arguments).decode(
            "utf-8",
            errors="strict",
        ).strip()
    except UnicodeDecodeError as exc:
        raise M4ReuseError(
            "m4_reuse_snapshot_mismatch",
            "Git returned invalid historical M4 provenance",
        ) from exc
    if not value or "\n" in value or "\r" in value:
        _fail(
            "m4_reuse_snapshot_mismatch",
            "Git returned invalid historical M4 provenance",
        )
    return value


def _fail(code: str, message: str) -> NoReturn:
    raise M4ReuseError(code, message)


__all__ = [
    "AcceptedM4Cohort",
    "AcceptedM4MemberRef",
    "M4_ACCEPTED_DATASET_CONFIG_FINGERPRINT",
    "M4_ACCEPTED_GIT_COMMIT",
    "M4_ACCEPTED_GIT_TREE",
    "M4AcceptanceReceipt",
    "M4ReuseError",
    "M4ReuseEvidence",
    "M5_M4_REUSE_SCHEMA_VERSION",
    "M5_M4_SELECTED_ORDER_VERSION",
    "ReloadedM4Member",
    "reverify_accepted_m4_run",
    "verify_accepted_m4_run",
    "visit_accepted_m4_cohort",
]
