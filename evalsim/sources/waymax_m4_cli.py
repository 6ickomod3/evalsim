"""Opt-in, local-only M4 cohort and Waymax-reference acceptance command.

The command is intentionally strict:

* it runs only from an explicit, clean EvalSim Git worktree;
* it resolves and opens only WOMD validation shards 00000 through 00009;
* it writes exclusively into a new, ignored ``outputs/m4`` descendant;
* it scans every raw record twice before using the frozen selection;
* it keeps native identities, locators, shard digests, and manifests local; and
* it prints only a stable pass/fail status and a relative ignored report path.

Waymax, JAX, and TensorFlow remain lazy optional dependencies. Importing this module
does not import any of them.
"""
from __future__ import annotations

import argparse
import ctypes
import dataclasses
import hashlib
import importlib
import json
import math
import multiprocessing
import os
import re
import resource
import stat
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

from evalsim.contracts import Rollout, Scenario
from evalsim.rollout import RolloutEngine
from evalsim.simulators import ConstantVelocityPolicy, IDMPolicy, LogReplayPolicy
from evalsim.simulators.waymax_reference import (
    CompactWaymaxIDMRollout,
    CompactWaymaxRollout,
    M4_EXACT_LOG_TRANSITIONS,
    M4_FLOAT_ATOL,
    M4_IDM_TRANSITIONS,
    WAYMAX_EXACT_LOG_NAME,
    WAYMAX_IDM_NAME,
    WAYMAX_REFERENCE_VERSION,
    assert_waymax_idm_defaults,
    compact_exact_log_rollout,
    compact_rollout_bytes,
    compact_stock_exact_log_rollout,
    compact_waymax_idm_rollout,
    compact_waymax_to_rollout,
    initialized_overlap_mask_numpy,
    reference_config_fingerprint,
    single_scene_exact_log_kernel,
    single_scene_idm_kernel,
    validate_exact_log_compact,
    validate_stock_equivalence,
)

from .waymax import (
    WAYMAX_COMMIT,
    scenario_from_waymax_state,
)
from .waymax_cohort import (
    M4_SELECTOR_CONFIG_FINGERPRINT,
    M4_SHARD_SUFFIXES,
    SOURCE_REJECTION_CODES,
    ScanEvent,
    ShardScanCounts,
    WaymaxCohortManifest,
    select_idm_subset,
    select_vmap_pair,
    source_rejection_code,
)
from .waymax_loader import (
    LOCAL_WAYMO_ENV_FLAG,
    M4ReloadExpectation,
    M4ShardLocator,
    M4StreamCounters,
    M4StreamRecord,
    WaymaxRecord,
    iter_m4_waymax_records,
    reload_m4_waymax_records,
    resolve_m4_validation_shards,
    runtime_summary,
    validate_record_parity,
)

M4_COMMAND_SCHEMA_VERSION = "1"
M4_RANDOM_SEED = 2026
M4_BENCHMARK_RUNS = 20
M4_BENCHMARK_BATCH_SIZE = 2
M4_BENCHMARK_TIMEOUT_SECONDS = 3600.0

_PLAN_PATH = "docs/plans/2026-07-28-m4-womd-cohort-waymax-parity.md"
_EXECUTABLE_PATHS = (
    "pyproject.toml",
    "uv.lock",
    _PLAN_PATH,
    "evalsim/sources/waymax.py",
    "evalsim/sources/waymax_cohort.py",
    "evalsim/sources/waymax_loader.py",
    "evalsim/sources/waymax_m4_cli.py",
    "evalsim/rollout/engine.py",
    "evalsim/simulators/log_replay.py",
    "evalsim/simulators/constant_velocity.py",
    "evalsim/simulators/idm.py",
    "evalsim/simulators/waymax_reference.py",
)
_HEX_DIGEST = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")
_GIT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SAFE_OUTPUT_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_TRUSTED_COMMAND_CODES = frozenset(
    """
    adapter_parity_false
    aggregate_json
    aggregate_privacy_key
    aggregate_privacy_value
    argument_error
    artifact_exists
    artifact_write_failed
    benchmark_clock
    benchmark_timeout
    benchmark_worker_eof
    benchmark_worker_exit
    benchmark_worker_failure
    benchmark_worker_protocol
    benchmark_worker_start
    data_directory_missing
    dirty_worktree
    evalsim_horizon
    evalsim_nondeterministic
    evalsim_rollout_contract
    evalsim_rollout_identity
    evalsim_rollout_lifecycle
    evalsim_rollout_nonfinite
    exact_log_rollout_mismatch
    execution_source_changed
    full_acceptance_accounting
    git_identity_invalid
    git_preflight_failed
    idm_accounting_shape
    idm_control_mask_drift
    idm_conversion_horizon
    idm_conversion_nonfinite
    idm_effective_control_floor
    idm_fallback_motion
    idm_fallback_validity
    idm_jit_mismatch
    idm_lifecycle_drift
    idm_nondeterministic
    idm_nonvacuity
    idm_qualification_contract
    idm_qualification_drift
    import_checkout_mismatch
    import_source_unresolved
    initialized_overlap_oracle
    jax_backend
    jax_dependency_missing
    jax_platform_override
    local_opt_in_required
    manifest_exists
    manifest_repeat_mismatch
    manifest_round_trip
    manifest_write_failed
    manifest_write_mismatch
    output_create_failed
    output_data_overlap
    output_exists
    output_name_invalid
    output_not_ignored
    output_root_escape
    output_scope_invalid
    output_validation_failed
    output_visible_to_git
    overlap_box_contract
    project_root_invalid
    reload_target_duplicate
    reload_target_manifest
    reload_target_unselected
    scan_event_type
    selected_reload_count
    selected_reload_duplicate
    selected_reload_mismatch
    selected_reload_missing
    shard_count_invalid
    source_changed
    source_fingerprint_failed
    source_not_tracked
    source_path_invalid
    terminal_capture_failed
    terminal_output_detected
    unexpected_failure
    vmap_pair_size
    vmap_parity
    vmap_schema
    waymax_conversion_agent
    waymax_conversion_identity
    waymax_conversion_lifecycle
    waymax_conversion_mapping
    waymax_conversion_provenance
    waymax_conversion_shape
    waymax_dependency_missing
    """.split()
)
_IMPORTED_MODULE_PATHS = {
    "evalsim": "evalsim/__init__.py",
    "evalsim.rollout.engine": "evalsim/rollout/engine.py",
    "evalsim.simulators.constant_velocity": (
        "evalsim/simulators/constant_velocity.py"
    ),
    "evalsim.simulators.idm": "evalsim/simulators/idm.py",
    "evalsim.simulators.log_replay": "evalsim/simulators/log_replay.py",
    "evalsim.simulators.waymax_reference": (
        "evalsim/simulators/waymax_reference.py"
    ),
    "evalsim.sources.waymax": "evalsim/sources/waymax.py",
    "evalsim.sources.waymax_cohort": "evalsim/sources/waymax_cohort.py",
    "evalsim.sources.waymax_loader": "evalsim/sources/waymax_loader.py",
    "evalsim.sources.waymax_m4_cli": "evalsim/sources/waymax_m4_cli.py",
}


class M4CommandError(RuntimeError):
    """A local-command failure with a stable, privacy-safe reason code."""

    def __init__(self, code: str, message: str) -> None:
        if code not in _TRUSTED_COMMAND_CODES:
            raise ValueError("M4CommandError code is not in the trusted registry")
        self.code = code
        super().__init__(f"{code}: {message}")


class _PrivacySafeParser(argparse.ArgumentParser):
    """Suppress user/path-bearing argparse diagnostics at the CLI boundary."""

    def error(self, message: str) -> None:
        del message
        self.exit(2, "M4 local acceptance: FAIL (argument_error)\n")


@dataclass(frozen=True, slots=True)
class _RunResult:
    """Only the relative ignored report path may leave the command boundary."""

    report_relative: Path
    terminal_status: _TerminalStatus | None = None


@dataclass(frozen=True, slots=True)
class _PendingAcceptance:
    """An accepted in-memory aggregate that has not yet been published."""

    aggregate: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _TerminalStatus:
    """Non-inheritable duplicates of the original command status streams."""

    stdout_fd: int
    stderr_fd: int

    def close_best_effort(self) -> None:
        for descriptor in (self.stdout_fd, self.stderr_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass


class _TerminalizedFailure(RuntimeError):
    """A failure whose only safe output path is a preserved status descriptor."""

    def __init__(
        self,
        primary: BaseException,
        terminal_status: _TerminalStatus,
    ) -> None:
        self.primary = primary
        self.terminal_status = terminal_status
        super().__init__("M4 terminalized failure")


@dataclass(slots=True)
class _TerminalCapture:
    """Mutable descriptor state for one optional-runtime capture."""

    transcript_path: Path
    transcript_fd: int
    transcript_identity: tuple[int, int]
    restore_stdout_fd: int
    restore_stderr_fd: int
    terminal_status: _TerminalStatus
    stdout_inheritable: bool
    stderr_inheritable: bool
    stdout_redirected: bool = False
    stderr_redirected: bool = False


@dataclass(frozen=True, slots=True)
class _IDMQualification:
    """Result-independent nested-subset facts for one selected state."""

    qualifying_vehicle_mask: np.ndarray
    initialized_overlap_mask: np.ndarray
    initialized_overlap_vehicle_exclusions: int

    def __post_init__(self) -> None:
        mask = np.array(self.qualifying_vehicle_mask, dtype=bool, copy=True)
        overlap = np.array(self.initialized_overlap_mask, dtype=bool, copy=True)
        if mask.shape != (128,) or overlap.shape != (128,):
            raise M4CommandError(
                "idm_qualification_contract",
                "qualification and overlap masks must have 128 object slots",
            )
        mask.setflags(write=False)
        overlap.setflags(write=False)
        object.__setattr__(self, "qualifying_vehicle_mask", mask)
        object.__setattr__(self, "initialized_overlap_mask", overlap)
        count = self.initialized_overlap_vehicle_exclusions
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or int(count) < 0
        ):
            raise M4CommandError(
                "idm_qualification_contract",
                "overlap vehicle exclusions must be a non-negative count",
            )
        object.__setattr__(
            self,
            "initialized_overlap_vehicle_exclusions",
            int(count),
        )

    @property
    def qualifies(self) -> bool:
        return bool(np.any(self.qualifying_vehicle_mask))


def _parser() -> argparse.ArgumentParser:
    parser = _PrivacySafeParser(
        prog="evalsim-waymax-m4",
        description=(
            "Run the opt-in local M4 ten-shard WOMD/Waymax acceptance. "
            "Native IDs, locators, digests, coordinates, and absolute paths "
            "are never printed."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Explicit EvalSim Git worktree root.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Explicit local WOMD v1.3.1 TFExample validation directory. "
            "Only exact suffixes 00000 through 00009 are resolved."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "New ignored run directory strictly below outputs/m4. "
            "It must not already exist."
        ),
    )
    return parser


def _run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise M4CommandError(
            "git_preflight_failed",
            "a required read-only Git inspection failed",
        )
    return result.stdout


def _project_root(candidate: Path) -> Path:
    """Resolve and verify the exact explicit EvalSim worktree root."""

    if not isinstance(candidate, Path):
        candidate = Path(candidate)
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise M4CommandError(
            "project_root_invalid",
            "the explicit project root does not exist",
        ) from exc
    if not root.is_dir():
        raise M4CommandError(
            "project_root_invalid",
            "the explicit project root is not a directory",
        )
    if not (root / "pyproject.toml").is_file() or not (
        root / ".gitignore"
    ).is_file():
        raise M4CommandError(
            "project_root_invalid",
            "the explicit root is not an EvalSim checkout",
        )
    top_level = _run_git(root, "rev-parse", "--show-toplevel").strip()
    try:
        resolved_top_level = Path(top_level).resolve(strict=True)
    except OSError as exc:
        raise M4CommandError(
            "project_root_invalid",
            "the Git worktree root could not be verified",
        ) from exc
    if resolved_top_level != root:
        raise M4CommandError(
            "project_root_invalid",
            "--project-root must name the Git worktree root exactly",
        )
    return root


def _resolve_project_path(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _assert_running_checkout(root: Path) -> None:
    """Prove imported execution code comes from the Git root being bound."""

    for module_name, relative_text in _IMPORTED_MODULE_PATHS.items():
        module = importlib.import_module(module_name)
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            raise M4CommandError(
                "import_source_unresolved",
                "an imported M4 execution module has no filesystem source",
            )
        try:
            actual = Path(raw_path).resolve(strict=True)
            expected = (root / relative_text).resolve(strict=True)
        except OSError as exc:
            raise M4CommandError(
                "import_source_unresolved",
                "an imported M4 execution module could not be resolved",
            ) from exc
        if actual != expected:
            raise M4CommandError(
                "import_checkout_mismatch",
                "running M4 code does not come from the bound Git checkout",
            )


def _assert_clean_worktree(root: Path) -> None:
    """Reject tracked, staged, and non-ignored untracked changes."""

    status = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored=no",
    )
    if status:
        raise M4CommandError(
            "dirty_worktree",
            "M4 requires a clean tracked/staged/non-ignored worktree",
        )


def _is_git_ignored(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            relative.as_posix(),
        ],
        cwd=root,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def _prepare_output_directory(
    candidate: Path,
    *,
    root: Path,
    data_dir: Path,
) -> Path:
    """Create one new ignored run directory without allowing path escape."""

    output = _resolve_project_path(candidate, root)
    allowed = (root / "outputs" / "m4").resolve()
    if root != allowed and root not in allowed.parents:
        raise M4CommandError(
            "output_root_escape",
            "the resolved outputs/m4 root escapes the checkout",
        )
    if output == allowed or allowed not in output.parents:
        raise M4CommandError(
            "output_scope_invalid",
            "the output must be a new directory strictly below outputs/m4",
        )
    relative_run = output.relative_to(allowed)
    if any(
        _SAFE_OUTPUT_COMPONENT.fullmatch(component) is None
        for component in relative_run.parts
    ):
        raise M4CommandError(
            "output_name_invalid",
            "M4 output path components must use safe portable characters",
        )
    if output == data_dir or output in data_dir.parents or data_dir in output.parents:
        raise M4CommandError(
            "output_data_overlap",
            "the output and immutable dataset roots must be disjoint",
        )
    if not _is_git_ignored(root, output):
        raise M4CommandError(
            "output_not_ignored",
            "the requested M4 output directory is not ignored by Git",
        )
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise M4CommandError(
            "output_exists",
            "the M4 run directory already exists; choose a new one",
        ) from exc
    except OSError as exc:
        raise M4CommandError(
            "output_create_failed",
            "the ignored M4 run directory could not be created",
        ) from exc
    if output.resolve(strict=True) != output or not _is_git_ignored(root, output):
        raise M4CommandError(
            "output_validation_failed",
            "the created M4 output directory failed its containment/ignore gate",
        )
    return output


def _guarded_file_sha256(path: Path) -> str:
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise M4CommandError(
            "source_fingerprint_failed",
            "an executable input could not be read for provenance",
        ) from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise M4CommandError(
            "source_changed",
            "an executable input changed while it was fingerprinted",
        )
    return digest.hexdigest()


def _execution_provenance(root: Path) -> dict[str, Any]:
    """Bind execution to the clean commit/tree and every executable input."""

    commit = _run_git(root, "rev-parse", "--verify", "HEAD^{commit}").strip()
    tree = _run_git(root, "rev-parse", "--verify", "HEAD^{tree}").strip()
    if (
        _GIT_OBJECT_ID.fullmatch(commit) is None
        or _GIT_OBJECT_ID.fullmatch(tree) is None
    ):
        raise M4CommandError(
            "git_identity_invalid",
            "Git returned a noncanonical commit or tree identity",
        )
    files: dict[str, str] = {}
    for relative_text in _EXECUTABLE_PATHS:
        tracked = _run_git(
            root,
            "ls-files",
            "--error-unmatch",
            "--",
            relative_text,
        ).strip()
        if tracked != relative_text:
            raise M4CommandError(
                "source_not_tracked",
                "an M4 executable input is not tracked at its locked path",
            )
        path = (root / relative_text).resolve(strict=True)
        if root not in path.parents or not path.is_file():
            raise M4CommandError(
                "source_path_invalid",
                "an M4 executable input escapes the checkout or is not a file",
            )
        files[relative_text] = _guarded_file_sha256(path)
    return {
        "schema_version": M4_COMMAND_SCHEMA_VERSION,
        "git_commit": commit,
        "git_tree": tree,
        "files": files,
        "selector_config_fingerprint": M4_SELECTOR_CONFIG_FINGERPRINT,
        "reference_config_fingerprint": reference_config_fingerprint(),
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        encoded = (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        descriptor, temporary_text = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".pending",
            dir=path.parent,
        )
        temporary = Path(temporary_text)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link is an atomic exclusive publication on the same filesystem:
        # unlike replace/rename, it can never overwrite an existing evidence file.
        os.link(temporary, path)
    except FileExistsError as exc:
        raise M4CommandError(
            "artifact_exists",
            "an M4 artifact already exists and will not be overwritten",
        ) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise M4CommandError(
            "artifact_write_failed",
            "an M4 artifact could not be written canonically",
        ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # Once the canonical hard link exists it is valid evidence. A
                # best-effort pending-file cleanup must not turn success into a
                # failed command that nevertheless left accepted evidence.
                pass


def _record_provenance(record: WaymaxRecord) -> dict[str, Any]:
    return {
        "shard_suffix": record.shard_suffix,
        "record_ordinal": record.record_ordinal,
        "shard_sha256": record.shard_sha256,
        "dataset_config_fingerprint": record.dataset_config_fingerprint,
    }


def _classify_stream_record(
    stream_record: M4StreamRecord,
) -> tuple[ScanEvent, Scenario | None]:
    """Source-classify, then convert and independently check every eligible record."""

    record = stream_record.record
    rejection = source_rejection_code(record.audit)
    if rejection is not None:
        return (
            ScanEvent.rejected_event(
                shard_suffix=stream_record.locator.shard_suffix,
                record_ordinal=stream_record.locator.record_ordinal,
                native_scenario_id=record.scenario_id,
                shard_sha256=record.shard_sha256,
                dataset_config_fingerprint=record.dataset_config_fingerprint,
                rejection_code=rejection,
            ),
            None,
        )

    scenario = scenario_from_waymax_state(
        record.state,
        scenario_id=record.scenario_id,
        provenance=_record_provenance(record),
    )
    parity = validate_record_parity(record, scenario)
    if not parity or any(value is not True for value in parity.values()):
        raise M4CommandError(
            "adapter_parity_false",
            "the independent adapter parity gate did not return all true",
        )
    return (
        ScanEvent.eligible_event(
            shard_suffix=stream_record.locator.shard_suffix,
            record_ordinal=stream_record.locator.record_ordinal,
            native_scenario_id=record.scenario_id,
            shard_sha256=record.shard_sha256,
            dataset_config_fingerprint=record.dataset_config_fingerprint,
        ),
        scenario,
    )


def _scan_population(shard_paths: Sequence[Path]) -> WaymaxCohortManifest:
    """Read all ten raw streams through clean EOF and build one manifest."""

    if len(shard_paths) != len(M4_SHARD_SUFFIXES):
        raise M4CommandError(
            "shard_count_invalid",
            "M4 requires exactly ten resolved shard paths",
        )
    events: list[ScanEvent] = []
    counts: list[ShardScanCounts] = []
    for suffix, path in zip(M4_SHARD_SUFFIXES, shard_paths, strict=True):
        counters = M4StreamCounters(suffix)
        shard_events: list[ScanEvent] = []
        for event in iter_m4_waymax_records(
            path,
            counters=counters,
        ):
            if not isinstance(event, ScanEvent):
                raise M4CommandError(
                    "scan_event_type",
                    "the stream did not emit a ScanEvent",
                )
            shard_events.append(event)
        eligible = sum(event.outcome == "eligible" for event in shard_events)
        rejected = sum(event.outcome == "rejected" for event in shard_events)
        counts.append(
            ShardScanCounts(
                shard_suffix=suffix,
                raw_seen=counters.raw_seen,
                decode_attempted=counters.decode_attempted,
                event_emitted=counters.event_emitted,
                eligible=eligible,
                rejected=rejected,
                clean_eof=counters.clean_eof,
            )
        )
        events.extend(shard_events)
    return WaymaxCohortManifest.build(events=events, shard_counts=counts)


def _write_manifest_exclusive(
    manifest: WaymaxCohortManifest,
    path: Path,
) -> None:
    try:
        manifest.to_file(path)
        actual = path.read_bytes()
    except FileExistsError as exc:
        raise M4CommandError(
            "manifest_exists",
            "a local M4 manifest already exists and will not be overwritten",
        ) from exc
    except OSError as exc:
        raise M4CommandError(
            "manifest_write_failed",
            "the local M4 manifest could not be written",
        ) from exc
    if actual != manifest.canonical_bytes():
        raise M4CommandError(
            "manifest_write_mismatch",
            "written manifest bytes differ from the canonical manifest",
        )
    reloaded = WaymaxCohortManifest.from_file(path)
    if reloaded.canonical_bytes() != manifest.canonical_bytes():
        raise M4CommandError(
            "manifest_round_trip",
            "the local M4 manifest did not round-trip byte-for-byte",
        )


def _expected_unselected(event: ScanEvent) -> ScanEvent:
    return dataclasses.replace(event, selected=False)


def _visit_selected_records(
    *,
    data_dir: Path,
    manifest: WaymaxCohortManifest,
    selected_events: Sequence[ScanEvent],
    visitor: Callable[[ScanEvent, WaymaxRecord, Scenario], None],
) -> None:
    """Reload every frozen locator through the loader's exact ordinal API."""

    targets = {event.locator: event for event in selected_events}
    if len(targets) != len(selected_events):
        raise M4CommandError(
            "reload_target_duplicate",
            "selected reload targets must have unique exact locators",
        )
    if any(not event.selected for event in targets.values()):
        raise M4CommandError(
            "reload_target_unselected",
            "every reload target must belong to the frozen cohort",
        )
    manifest_selected = {
        event.locator: event for event in manifest.selected_events
    }
    if any(
        manifest_selected.get(locator) != event
        for locator, event in targets.items()
    ):
        raise M4CommandError(
            "reload_target_manifest",
            "a reload target differs from the frozen manifest selection",
        )

    by_suffix: dict[str, list[ScanEvent]] = {
        suffix: [] for suffix in M4_SHARD_SUFFIXES
    }
    for event in selected_events:
        by_suffix[event.shard_suffix].append(event)

    seen: set[tuple[str, int]] = set()
    for suffix in M4_SHARD_SUFFIXES:
        shard_events = by_suffix[suffix]
        if not shard_events:
            continue
        expectations = tuple(
            M4ReloadExpectation(
                locator=M4ShardLocator(
                    shard_suffix=expected.shard_suffix,
                    record_ordinal=expected.record_ordinal,
                ),
                expected_scenario_id=expected.native_scenario_id,
                expected_shard_sha256=expected.shard_sha256,
                expected_dataset_config_fingerprint=(
                    expected.dataset_config_fingerprint
                ),
            )
            for expected in shard_events
        )
        stream_records = reload_m4_waymax_records(data_dir, expectations)
        if len(stream_records) != len(shard_events):
            raise M4CommandError(
                "selected_reload_count",
                "the grouped exact reload returned the wrong record count",
            )
        for expected, stream_record in zip(
            shard_events,
            stream_records,
            strict=True,
        ):
            event, scenario = _classify_stream_record(stream_record)
            if event != _expected_unselected(expected) or scenario is None:
                raise M4CommandError(
                    "selected_reload_mismatch",
                    "a selected exact locator changed identity or provenance",
                )
            if event.locator in seen:
                raise M4CommandError(
                    "selected_reload_duplicate",
                    "a selected exact locator was reloaded more than once",
                )
            visitor(expected, stream_record.record, scenario)
            seen.add(event.locator)
    if seen != set(targets):
        raise M4CommandError(
            "selected_reload_missing",
            "one or more frozen selected locators were not reloaded",
        )


def _assert_rollout_contract(rollout: Rollout, scenario: Scenario) -> None:
    if (
        rollout.scenario_id != scenario.scenario_id
        or rollout.num_steps != scenario.num_steps
        or rollout.num_agents != scenario.num_agents
        or not np.array_equal(rollout.timestamps, scenario.timestamps)
    ):
        raise M4CommandError(
            "evalsim_rollout_contract",
            "an EvalSim rollout changed scalar, time, or shape semantics",
        )
    if [agent.id for agent in rollout.agents] != [
        agent.id for agent in scenario.agents
    ]:
        raise M4CommandError(
            "evalsim_rollout_identity",
            "an EvalSim rollout changed agent identity/order",
        )
    for source, actual in zip(scenario.agents, rollout.agents, strict=True):
        if (
            source.type != actual.type
            or source.length != actual.length
            or source.width != actual.width
            or not np.array_equal(source.valid, actual.valid)
        ):
            raise M4CommandError(
                "evalsim_rollout_lifecycle",
                "an EvalSim rollout changed type, dimensions, or lifecycle masks",
            )
        for field in ("x", "y", "heading", "vx", "vy"):
            if not np.all(np.isfinite(getattr(actual, field))):
                raise M4CommandError(
                    "evalsim_rollout_nonfinite",
                    "an EvalSim rollout emitted a non-finite motion value",
                )


def _assert_rollouts_equal(left: Rollout, right: Rollout) -> None:
    if (
        left.scenario_id != right.scenario_id
        or left.sim_name != right.sim_name
        or left.sim_version != right.sim_version
        or left.seed != right.seed
        or left.perturbation != right.perturbation
        or left.metadata != right.metadata
        or not np.array_equal(left.timestamps, right.timestamps)
        or len(left.agents) != len(right.agents)
    ):
        raise M4CommandError(
            "evalsim_nondeterministic",
            "repeated EvalSim rollout metadata or structure changed",
        )
    for first, second in zip(left.agents, right.agents, strict=True):
        if (
            first.id != second.id
            or first.type != second.type
            or first.length != second.length
            or first.width != second.width
        ):
            raise M4CommandError(
                "evalsim_nondeterministic",
                "repeated EvalSim rollout agent metadata changed",
            )
        for field in ("valid", "x", "y", "heading", "vx", "vy"):
            if not np.array_equal(
                getattr(first, field),
                getattr(second, field),
            ):
                raise M4CommandError(
                    "evalsim_nondeterministic",
                    "repeated EvalSim rollout arrays changed",
                )


def _assert_rollout_matches_scenario(
    rollout: Rollout,
    scenario: Scenario,
    *,
    atol: float,
) -> None:
    _assert_rollout_contract(rollout, scenario)
    for source, actual in zip(scenario.agents, rollout.agents, strict=True):
        for field in ("x", "y", "heading", "vx", "vy"):
            source_values = np.asarray(getattr(source, field))
            actual_values = np.asarray(getattr(actual, field))
            if field == "heading":
                delta = (
                    actual_values - source_values + np.pi
                ) % (2.0 * np.pi) - np.pi
                equal = np.allclose(delta, 0.0, rtol=0.0, atol=atol)
            else:
                equal = np.allclose(
                    actual_values,
                    source_values,
                    rtol=0.0,
                    atol=atol,
                )
            if not equal:
                raise M4CommandError(
                    "exact_log_rollout_mismatch",
                    "exact log playback changed a supported trajectory field",
                )


def _run_evalsims(scenario: Scenario) -> None:
    if (
        scenario.metadata.get("current_index") != 10
        or scenario.num_steps != 91
    ):
        raise M4CommandError(
            "evalsim_horizon",
            "the selected Scenario does not preserve the 10/1/80 boundary",
        )
    engine = RolloutEngine()
    replay = engine.run(scenario, LogReplayPolicy(), seed=M4_RANDOM_SEED)
    _assert_rollout_matches_scenario(replay, scenario, atol=0.0)
    for policy in (ConstantVelocityPolicy(), IDMPolicy()):
        first = engine.run(scenario, policy, seed=M4_RANDOM_SEED)
        second = engine.run(scenario, policy, seed=M4_RANDOM_SEED)
        _assert_rollout_contract(first, scenario)
        _assert_rollout_contract(second, scenario)
        _assert_rollouts_equal(first, second)


def _block_tree(tree: Any) -> Any:
    try:
        import jax
    except ImportError as exc:
        raise M4CommandError(
            "jax_dependency_missing",
            "the optional JAX runtime is unavailable",
        ) from exc
    return jax.tree.map(
        lambda value: (
            value.block_until_ready()
            if hasattr(value, "block_until_ready")
            else value
        ),
        tree,
    )


def _slice_compact(
    compact: CompactWaymaxRollout,
    count: int,
) -> CompactWaymaxRollout:
    return CompactWaymaxRollout(
        x=compact.x[:count],
        y=compact.y[:count],
        yaw=compact.yaw[:count],
        vx=compact.vx[:count],
        vy=compact.vy[:count],
        valid=compact.valid[:count],
        timestamp_micros=compact.timestamp_micros[:count],
        timestep=compact.timestep[:count],
    )


def _run_exact_log_reference(
    record: WaymaxRecord,
    scenario: Scenario,
    *,
    stock_gate: bool,
) -> None:
    compact = _block_tree(
        compact_exact_log_rollout(
            record.state,
            num_steps=M4_EXACT_LOG_TRANSITIONS,
        )
    )
    validate_exact_log_compact(record.state, compact)
    converted = compact_waymax_to_rollout(
        compact,
        state=record.state,
        scenario=scenario,
        sim_name=WAYMAX_EXACT_LOG_NAME,
        seed=M4_RANDOM_SEED,
    )
    _assert_waymax_conversion_mapping(
        converted,
        compact=compact,
        state=record.state,
        scenario=scenario,
        expected_sim_name=WAYMAX_EXACT_LOG_NAME,
        expected_control_accounting={},
    )
    _assert_rollout_matches_scenario(
        converted,
        scenario,
        atol=M4_FLOAT_ATOL,
    )
    if stock_gate:
        stock = _block_tree(
            compact_stock_exact_log_rollout(record.state, num_steps=1)
        )
        validate_stock_equivalence(_slice_compact(compact, 1), stock)


def _assert_waymax_conversion_mapping(
    rollout: Rollout,
    *,
    compact: CompactWaymaxRollout | CompactWaymaxIDMRollout,
    state: Any,
    scenario: Scenario,
    expected_sim_name: str,
    expected_control_accounting: Mapping[str, int],
) -> None:
    """Independently verify compact-slot mapping and conversion provenance."""

    compact_steps = int(np.asarray(compact.timestep).shape[0])
    horizon = 11 + compact_steps
    expected_metadata = {
        "backend": "waymax",
        "backend_commit": WAYMAX_COMMIT,
        "compact_reference_version": WAYMAX_REFERENCE_VERSION,
        "control_accounting": dict(expected_control_accounting),
        "horizon_transitions": compact_steps,
        "init_steps": 11,
        "invalid_fill": "finite_zero_where_invalid",
        "reference_config_fingerprint": reference_config_fingerprint(),
        "rollout_start_index": 10,
        "scenario_source": scenario.metadata.get("source", "unknown"),
        "scenario_source_fingerprint": scenario.metadata.get(
            "source_fingerprint"
        ),
        "time_source": "direct_waymax_emission_checked_against_log",
    }
    if (
        rollout.scenario_id != scenario.scenario_id
        or rollout.sim_name != expected_sim_name
        or rollout.sim_version != WAYMAX_REFERENCE_VERSION
        or rollout.seed != M4_RANDOM_SEED
        or rollout.perturbation is not None
        or rollout.metadata != expected_metadata
        or rollout.num_steps != horizon
        or rollout.num_agents != scenario.num_agents
        or not np.array_equal(
            rollout.timestamps,
            scenario.timestamps[:horizon],
        )
    ):
        raise M4CommandError(
            "waymax_conversion_provenance",
            "the converted Waymax rollout changed structure or provenance",
        )

    log_valid = np.asarray(state.log_trajectory.valid, dtype=bool)
    retained = np.flatnonzero(np.any(log_valid, axis=1))
    source_ids = np.asarray(state.object_metadata.ids)[retained]
    if retained.size != scenario.num_agents or not np.array_equal(
        source_ids,
        np.asarray(
            [agent.id for agent in scenario.agents],
            dtype=source_ids.dtype,
        ),
    ):
        raise M4CommandError(
            "waymax_conversion_identity",
            "the source slot-to-agent identity mapping changed",
        )

    compact_valid = np.asarray(compact.valid, dtype=bool)
    if compact_valid.shape != (compact_steps, 128):
        raise M4CommandError(
            "waymax_conversion_shape",
            "compact validity does not have the fixed step/object shape",
        )
    source_future_valid = np.asarray(
        state.log_trajectory.valid[:, 11:horizon],
        dtype=bool,
    ).T
    scenario_future_valid = np.stack(
        [
            np.asarray(agent.valid[11:horizon], dtype=bool)
            for agent in scenario.agents
        ],
        axis=1,
    )
    if (
        not np.array_equal(
            compact_valid[:, retained],
            source_future_valid[:, retained],
        )
        or not np.array_equal(
            compact_valid[:, retained],
            scenario_future_valid,
        )
    ):
        raise M4CommandError(
            "waymax_conversion_lifecycle",
            "compact retained validity differs from source/Scenario lifecycle",
        )
    expected_timestep = np.arange(
        11,
        11 + compact_steps,
        dtype=np.asarray(compact.timestep).dtype,
    )
    expected_micros = np.asarray(
        state.log_trajectory.timestamp_micros[:, 11:horizon]
    ).T
    if (
        not np.array_equal(np.asarray(compact.timestep), expected_timestep)
        or not np.array_equal(
            np.asarray(compact.timestamp_micros),
            expected_micros,
        )
    ):
        raise M4CommandError(
            "waymax_conversion_provenance",
            "compact timestep/timestamps differ from the direct source timeline",
        )
    compact_fields = {
        "x": np.asarray(compact.x, dtype=np.float64),
        "y": np.asarray(compact.y, dtype=np.float64),
        "heading": np.asarray(compact.yaw, dtype=np.float64),
        "vx": np.asarray(compact.vx, dtype=np.float64),
        "vy": np.asarray(compact.vy, dtype=np.float64),
    }
    for source_agent, actual_agent, slot in zip(
        scenario.agents,
        rollout.agents,
        retained,
        strict=True,
    ):
        expected_valid = np.concatenate(
            (
                np.asarray(source_agent.valid[:11], dtype=bool),
                compact_valid[:, slot],
            )
        )
        if (
            actual_agent.id != source_agent.id
            or actual_agent.type != source_agent.type
            or actual_agent.length != source_agent.length
            or actual_agent.width != source_agent.width
            or not np.array_equal(actual_agent.valid, expected_valid)
        ):
            raise M4CommandError(
                "waymax_conversion_agent",
                "converted agent metadata or validity differs from source slots",
            )
        for field, compact_values in compact_fields.items():
            history = np.asarray(getattr(source_agent, field)[:11])
            future = np.asarray(compact_values[:, slot], dtype=np.float64)
            if field == "heading":
                future = (future + np.pi) % (2.0 * np.pi) - np.pi
            future = np.where(compact_valid[:, slot], future, 0.0)
            expected = np.concatenate((history, future))
            actual = np.asarray(getattr(actual_agent, field))
            if not np.allclose(
                actual,
                expected,
                rtol=0.0,
                atol=M4_FLOAT_ATOL,
            ):
                raise M4CommandError(
                    "waymax_conversion_mapping",
                    "converted motion differs from the identity-aligned compact field",
                )


def _frame_zero_boxes(state: Any) -> np.ndarray:
    trajectory = state.log_trajectory
    boxes = np.stack(
        (
            np.asarray(trajectory.x)[:, 0],
            np.asarray(trajectory.y)[:, 0],
            np.asarray(trajectory.length)[:, 0],
            np.asarray(trajectory.width)[:, 0],
            np.asarray(trajectory.yaw)[:, 0],
        ),
        axis=-1,
    )
    if boxes.shape != (128, 5) or not np.all(np.isfinite(boxes)):
        raise M4CommandError(
            "overlap_box_contract",
            "frame-zero Waymax boxes do not satisfy the pinned [128,5] contract",
        )
    return boxes


def _overlap_and_qualification(state: Any) -> _IDMQualification:
    """Cross-check the no-validity-mask overlap oracle before IDM output."""

    boxes = _frame_zero_boxes(state)
    independent = initialized_overlap_mask_numpy(boxes)
    try:
        import jax.numpy as jnp
        from waymax.utils import geometry
    except ImportError as exc:
        raise M4CommandError(
            "waymax_dependency_missing",
            "the optional pinned Waymax runtime is unavailable",
        ) from exc
    upstream = np.any(
        np.asarray(geometry.compute_pairwise_overlaps(jnp.asarray(boxes))),
        axis=-1,
    )
    if not np.array_equal(independent, upstream):
        raise M4CommandError(
            "initialized_overlap_oracle",
            "the independent frame-zero overlap mask differs from Waymax",
        )
    metadata = state.object_metadata
    is_sdc = np.asarray(metadata.is_sdc, dtype=bool)
    object_types = np.asarray(metadata.object_types)
    valid = np.asarray(state.log_trajectory.valid, dtype=bool)
    if (
        is_sdc.shape != (128,)
        or object_types.shape != (128,)
        or valid.shape != (128, 91)
    ):
        raise M4CommandError(
            "idm_qualification_contract",
            "the state does not satisfy the fixed IDM qualification shapes",
        )
    non_sdc_vehicle = ~is_sdc & (object_types == 1)
    continuous = np.all(valid[:, 10:31], axis=1)
    qualifying_vehicle = non_sdc_vehicle & continuous & ~independent
    overlap_vehicle_exclusions = int(
        np.count_nonzero(non_sdc_vehicle & independent)
    )
    return _IDMQualification(
        qualifying_vehicle_mask=qualifying_vehicle,
        initialized_overlap_mask=independent,
        initialized_overlap_vehicle_exclusions=overlap_vehicle_exclusions,
    )


def _idm_has_nonfallback_motion(
    compact: Any,
    state: Any,
) -> bool:
    effective = np.asarray(compact.effective_control, dtype=bool)
    valid = np.asarray(compact.valid, dtype=bool)
    if effective.shape != (M4_IDM_TRANSITIONS, 128) or valid.shape != (
        M4_IDM_TRANSITIONS,
        128,
    ):
        raise M4CommandError(
            "idm_accounting_shape",
            "Waymax IDM control masks do not match the fixed horizon/slots",
        )
    mask = effective & valid
    interval = slice(11, 11 + M4_IDM_TRANSITIONS)
    fields = {
        "x": "x",
        "y": "y",
        "yaw": "yaw",
        "vx": "vel_x",
        "vy": "vel_y",
    }
    for output_name, source_name in fields.items():
        actual = np.asarray(getattr(compact, output_name), dtype=np.float64)
        expected = np.asarray(
            getattr(state.log_trajectory, source_name)[:, interval],
            dtype=np.float64,
        ).T
        if output_name == "yaw":
            difference = np.abs(
                (actual - expected + np.pi) % (2.0 * np.pi) - np.pi
            )
        else:
            difference = np.abs(actual - expected)
        if np.any(difference[mask] > 1e-6):
            return True
    return False


def _independent_idm_accounting(
    compact: CompactWaymaxIDMRollout,
    *,
    state: Any,
    qualification: _IDMQualification,
) -> dict[str, int]:
    """Rebuild all IDM masks and fallback claims without trusting output labels."""

    masks: dict[str, np.ndarray] = {}
    for field in (
        "requested_control",
        "effective_control",
        "lifecycle_fallback",
        "initialized_overlap_excluded",
    ):
        value = np.asarray(getattr(compact, field))
        if value.dtype != np.bool_ or value.shape != (M4_IDM_TRANSITIONS, 128):
            raise M4CommandError(
                "idm_accounting_shape",
                "every IDM accounting mask must be boolean [20,128]",
            )
        masks[field] = value

    metadata = state.object_metadata
    is_sdc = np.asarray(metadata.is_sdc, dtype=bool)
    object_types = np.asarray(metadata.object_types)
    logged_valid = np.asarray(state.log_trajectory.valid, dtype=bool)
    non_sdc_vehicle = ~is_sdc & (object_types == 1)
    expected_requested = np.stack(
        [
            non_sdc_vehicle
            & logged_valid[:, current]
            & logged_valid[:, current + 1]
            for current in range(10, 10 + M4_IDM_TRANSITIONS)
        ],
        axis=0,
    )
    expected_masks = {
        "requested_control": expected_requested,
        "effective_control": (
            expected_requested
            & ~qualification.initialized_overlap_mask[np.newaxis, :]
        ),
        "lifecycle_fallback": (
            non_sdc_vehicle[np.newaxis, :] & ~expected_requested
        ),
        "initialized_overlap_excluded": (
            expected_requested
            & qualification.initialized_overlap_mask[np.newaxis, :]
        ),
    }
    for field, expected in expected_masks.items():
        if not np.array_equal(masks[field], expected):
            raise M4CommandError(
                "idm_control_mask_drift",
                "an IDM control/fallback mask differs from source semantics",
            )

    # Prove both declared fallback classes used the next direct logged state.
    fallback = (
        masks["lifecycle_fallback"]
        | masks["initialized_overlap_excluded"]
    )
    interval = slice(11, 11 + M4_IDM_TRANSITIONS)
    actual_valid = np.asarray(compact.valid, dtype=bool)
    expected_valid = np.asarray(
        state.log_trajectory.valid[:, interval],
        dtype=bool,
    ).T
    if not np.array_equal(actual_valid, expected_valid):
        raise M4CommandError(
            "idm_lifecycle_drift",
            "Waymax IDM validity differs from the full logged lifecycle",
        )
    if not np.array_equal(actual_valid[fallback], expected_valid[fallback]):
        raise M4CommandError(
            "idm_fallback_validity",
            "declared IDM fallback validity differs from direct log",
        )
    for output_name, source_name in {
        "x": "x",
        "y": "y",
        "yaw": "yaw",
        "vx": "vel_x",
        "vy": "vel_y",
    }.items():
        actual = np.asarray(getattr(compact, output_name), dtype=np.float64)
        expected = np.asarray(
            getattr(state.log_trajectory, source_name)[:, interval],
            dtype=np.float64,
        ).T
        difference = (
            np.abs((actual - expected + np.pi) % (2.0 * np.pi) - np.pi)
            if output_name == "yaw"
            else np.abs(actual - expected)
        )
        if np.any(difference[fallback] > M4_FLOAT_ATOL):
            raise M4CommandError(
                "idm_fallback_motion",
                "declared IDM fallback motion differs from direct log",
            )

    per_qualifying_vehicle = np.count_nonzero(
        masks["effective_control"][:, qualification.qualifying_vehicle_mask],
        axis=0,
    )
    qualifying_vehicle_effective = int(
        np.max(per_qualifying_vehicle, initial=0)
    )
    if qualifying_vehicle_effective < M4_IDM_TRANSITIONS:
        raise M4CommandError(
            "idm_effective_control_floor",
            "no continuously valid qualifying vehicle has 20 effective transitions",
        )
    return {
        "effective_controlled_transitions": int(
            np.count_nonzero(masks["effective_control"])
        ),
        "initialized_overlap_excluded_transitions": int(
            np.count_nonzero(masks["initialized_overlap_excluded"])
        ),
        "initialized_overlap_excluded_vehicles": int(
            np.count_nonzero(
                np.any(masks["initialized_overlap_excluded"], axis=0)
            )
        ),
        "lifecycle_fallbacks": int(
            np.count_nonzero(masks["lifecycle_fallback"])
        ),
        "qualifying_vehicle_effective_transitions": (
            qualifying_vehicle_effective
        ),
        "requested_control_transitions": int(
            np.count_nonzero(masks["requested_control"])
        ),
    }


def _run_idm_reference(
    record: WaymaxRecord,
    scenario: Scenario,
    *,
    expected_qualification: _IDMQualification,
    jit_gate: bool,
) -> dict[str, int | bool]:
    qualification = _overlap_and_qualification(record.state)
    if (
        not qualification.qualifies
        or not np.array_equal(
            qualification.qualifying_vehicle_mask,
            expected_qualification.qualifying_vehicle_mask,
        )
        or not np.array_equal(
            qualification.initialized_overlap_mask,
            expected_qualification.initialized_overlap_mask,
        )
        or qualification.initialized_overlap_vehicle_exclusions
        != expected_qualification.initialized_overlap_vehicle_exclusions
    ):
        raise M4CommandError(
            "idm_qualification_drift",
            "a frozen nested-subset scene no longer qualifies",
        )
    assert_waymax_idm_defaults()
    first = _block_tree(
        compact_waymax_idm_rollout(
            record.state,
            num_steps=M4_IDM_TRANSITIONS,
        )
    )
    second = _block_tree(
        compact_waymax_idm_rollout(
            record.state,
            num_steps=M4_IDM_TRANSITIONS,
        )
    )
    if compact_rollout_bytes(first) != compact_rollout_bytes(second):
        raise M4CommandError(
            "idm_nondeterministic",
            "repeated Waymax IDM compact output bytes changed",
        )
    if jit_gate:
        try:
            import jax
        except ImportError as exc:
            raise M4CommandError(
                "jax_dependency_missing",
                "the optional JAX runtime is unavailable",
            ) from exc
        compiled = _block_tree(jax.jit(single_scene_idm_kernel)(record.state))
        if compact_rollout_bytes(first) != compact_rollout_bytes(compiled):
            raise M4CommandError(
                "idm_jit_mismatch",
                "compiled and eager Waymax IDM outputs differ",
            )

    accounting = _independent_idm_accounting(
        first,
        state=record.state,
        qualification=qualification,
    )
    nonfallback = _idm_has_nonfallback_motion(first, record.state)
    converted = compact_waymax_to_rollout(
        first,
        state=record.state,
        scenario=scenario,
        sim_name=WAYMAX_IDM_NAME,
        seed=M4_RANDOM_SEED,
    )
    expected_converter_accounting = {
        "effective_controlled_transitions": accounting[
            "effective_controlled_transitions"
        ],
        "initialized_overlap_excluded_transitions": accounting[
            "initialized_overlap_excluded_transitions"
        ],
        "initialized_overlap_excluded_vehicles": accounting[
            "initialized_overlap_excluded_vehicles"
        ],
        "lifecycle_fallbacks": accounting["lifecycle_fallbacks"],
        "requested_control_transitions": accounting[
            "requested_control_transitions"
        ],
    }
    _assert_waymax_conversion_mapping(
        converted,
        compact=first,
        state=record.state,
        scenario=scenario,
        expected_sim_name=WAYMAX_IDM_NAME,
        expected_control_accounting=expected_converter_accounting,
    )
    if converted.num_steps != 11 + M4_IDM_TRANSITIONS:
        raise M4CommandError(
            "idm_conversion_horizon",
            "the converted IDM rollout has the wrong fixed horizon",
        )
    for agent in converted.agents:
        for field in ("x", "y", "heading", "vx", "vy"):
            if not np.all(np.isfinite(getattr(agent, field))):
                raise M4CommandError(
                    "idm_conversion_nonfinite",
                    "the converted IDM rollout contains non-finite values",
                )
    return {
        "requested": accounting["requested_control_transitions"],
        "effective": accounting["effective_controlled_transitions"],
        "lifecycle_fallback": accounting["lifecycle_fallbacks"],
        "overlap_fallback": accounting[
            "initialized_overlap_excluded_transitions"
        ],
        "overlap_excluded_vehicles": accounting[
            "initialized_overlap_excluded_vehicles"
        ],
        "qualifying_vehicle_effective": accounting[
            "qualifying_vehicle_effective_transitions"
        ],
        "nonfallback_motion": nonfallback,
    }


def _assert_batched_compact_equal(left: Any, right: Any) -> None:
    exact_fields = {"valid", "timestamp_micros", "timestep"}
    if tuple(left._fields) != tuple(right._fields):
        raise M4CommandError(
            "vmap_schema",
            "batched compact outputs have different schemas",
        )
    left_valid = np.asarray(left.valid, dtype=bool)
    right_valid = np.asarray(right.valid, dtype=bool)
    if not np.array_equal(left_valid, right_valid):
        raise M4CommandError(
            "vmap_parity",
            "sequential/JIT/vmap validity differs",
        )
    for field in left._fields:
        first = np.asarray(getattr(left, field))
        second = np.asarray(getattr(right, field))
        if field in {"x", "y", "yaw", "vx", "vy"}:
            first = np.where(left_valid, first, 0.0)
            second = np.where(right_valid, second, 0.0)
        equal = (
            np.array_equal(first, second)
            if field in exact_fields
            else np.allclose(first, second, rtol=0.0, atol=M4_FLOAT_ATOL)
        )
        if not equal:
            raise M4CommandError(
                "vmap_parity",
                "sequential/JIT/vmap/permutation output parity failed",
            )


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _benchmark_in_worker(states: tuple[Any, Any]) -> dict[str, Any]:
    """Run the explicit two-scene JIT/vmap gate in the fresh worker process."""

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise M4CommandError(
            "jax_dependency_missing",
            "the optional JAX runtime is unavailable",
        ) from exc
    if len(states) != M4_BENCHMARK_BATCH_SIZE:
        raise M4CommandError(
            "vmap_pair_size",
            "the benchmark requires exactly two frozen states",
        )
    host_states = tuple(jax.device_get(state) for state in states)
    sequential_items = tuple(
        _block_tree(single_scene_exact_log_kernel(state))
        for state in host_states
    )
    sequential = jax.tree.map(
        lambda first, second: jnp.stack((first, second), axis=0),
        *sequential_items,
    )
    host_batch = jax.tree.map(
        lambda first, second: np.stack(
            (np.asarray(first), np.asarray(second)),
            axis=0,
        ),
        *host_states,
    )
    device_batch = jax.device_put(host_batch)
    _block_tree(device_batch)
    compiled_function = jax.jit(jax.vmap(single_scene_exact_log_kernel))
    compile_started = time.perf_counter_ns()
    executable = compiled_function.lower(device_batch).compile()
    compile_seconds = (time.perf_counter_ns() - compile_started) / 1e9
    warmup = _block_tree(executable(device_batch))
    _assert_batched_compact_equal(sequential, warmup)

    durations: list[float] = []
    for _ in range(M4_BENCHMARK_RUNS):
        started = time.perf_counter_ns()
        _block_tree(executable(device_batch))
        durations.append((time.perf_counter_ns() - started) / 1e9)

    reversed_batch = jax.tree.map(lambda value: value[::-1], device_batch)
    reversed_output = _block_tree(executable(reversed_batch))
    restored = jax.tree.map(lambda value: value[::-1], reversed_output)
    _assert_batched_compact_equal(warmup, restored)

    ordered = sorted(durations)
    median = float(np.median(np.asarray(durations, dtype=np.float64)))
    p95_rank = math.ceil(0.95 * len(ordered)) - 1
    p95 = float(ordered[p95_rank])
    if median <= 0.0 or p95 <= 0.0:
        raise M4CommandError(
            "benchmark_clock",
            "the synchronized benchmark produced a non-positive duration",
        )
    return {
        "batch_size": M4_BENCHMARK_BATCH_SIZE,
        "compile_seconds": float(compile_seconds),
        "device_transfer_before_timing": True,
        "eager_sequential_parity": True,
        "fresh_worker_process": True,
        "horizon_transitions": M4_EXACT_LOG_TRANSITIONS,
        "jit_vmap": True,
        "memory_measurement": "process_high_water_rss_not_jax_device_memory",
        "median_seconds": median,
        "nearest_rank_p95_seconds": p95,
        "peak_rss_bytes": _peak_rss_bytes(),
        "permutation_invariance": True,
        "runs": M4_BENCHMARK_RUNS,
        "scenarios_per_second_at_median": (
            M4_BENCHMARK_BATCH_SIZE / median
        ),
        "warm_durations_seconds": durations,
    }


def _benchmark_worker(connection: Any, states: tuple[Any, Any]) -> None:
    """Multiprocessing target that never sends source data or exception text."""

    try:
        report = _benchmark_in_worker(states)
        connection.send({"ok": True, "report": report})
    except BaseException as exc:  # pragma: no cover - exercised in child process.
        connection.send({"ok": False, "code": _failure_code(exc)})
    finally:
        connection.close()


def _close_benchmark_endpoint(endpoint: Any) -> bool:
    try:
        endpoint.close()
    except BaseException:
        return False
    return True


def _benchmark_process_alive(process: Any) -> bool | None:
    try:
        return bool(process.is_alive())
    except BaseException:
        try:
            return False if process.exitcode is not None else None
        except BaseException:
            return None


def _reap_benchmark_worker(process: Any, *, force_stop: bool) -> bool:
    """Join one started worker, escalating terminate to kill when required."""

    cleanup_ok = True
    if force_stop and _benchmark_process_alive(process) is not False:
        try:
            process.terminate()
        except BaseException:
            cleanup_ok = False

    try:
        process.join(timeout=10.0)
    except BaseException:
        cleanup_ok = False
    alive = _benchmark_process_alive(process)

    if alive is not False:
        try:
            process.terminate()
        except BaseException:
            cleanup_ok = False
        try:
            process.join(timeout=10.0)
        except BaseException:
            cleanup_ok = False
        alive = _benchmark_process_alive(process)

    while alive is not False:
        try:
            process.kill()
        except BaseException:
            cleanup_ok = False
        try:
            # After an unconditional kill, wait without another timeout. Returning
            # while the child still holds inherited capture descriptors would make
            # terminal finalization unsafe.
            process.join()
        except BaseException:
            cleanup_ok = False
        alive = _benchmark_process_alive(process)

    return cleanup_ok


def _fresh_worker_benchmark(states: Sequence[Any]) -> dict[str, Any]:
    """Spawn one clean process and return only its sanitized benchmark facts."""

    if len(states) != M4_BENCHMARK_BATCH_SIZE:
        raise M4CommandError(
            "vmap_pair_size",
            "the fresh benchmark requires exactly two states",
        )
    try:
        import jax
    except ImportError as exc:
        raise M4CommandError(
            "jax_dependency_missing",
            "the optional JAX runtime is unavailable",
        ) from exc
    host_states = tuple(jax.device_get(state) for state in states)
    context = multiprocessing.get_context("spawn")
    receive: Any | None = None
    send: Any | None = None
    try:
        receive, send = context.Pipe(duplex=False)
        process = context.Process(
            target=_benchmark_worker,
            args=(send, host_states),
            name="evalsim-m4-vmap-benchmark",
        )
    except Exception as exc:
        for endpoint in (receive, send):
            if endpoint is not None:
                _close_benchmark_endpoint(endpoint)
        raise M4CommandError(
            "benchmark_worker_start",
            "the fresh benchmark worker could not be constructed",
        ) from exc

    try:
        process.start()
    except Exception as exc:
        _close_benchmark_endpoint(receive)
        _close_benchmark_endpoint(send)
        raise M4CommandError(
            "benchmark_worker_start",
            "the fresh benchmark worker could not start",
        ) from exc

    message: Any = None
    primary: BaseException | None = None
    endpoints_closed = _close_benchmark_endpoint(send)
    try:
        if not endpoints_closed:
            raise M4CommandError(
                "benchmark_worker_exit",
                "the benchmark pipe endpoint could not be closed",
            )
        if not receive.poll(M4_BENCHMARK_TIMEOUT_SECONDS):
            raise M4CommandError(
                "benchmark_timeout",
                "the fresh benchmark worker exceeded the resource gate",
            )
        message = receive.recv()
    except EOFError as exc:
        primary = M4CommandError(
            "benchmark_worker_eof",
            "the fresh benchmark worker exited without a result",
        )
        primary.__cause__ = exc
    except BaseException as exc:
        primary = exc

    endpoints_closed = (
        _close_benchmark_endpoint(receive) and endpoints_closed
    )
    worker_reaped = _reap_benchmark_worker(
        process,
        force_stop=primary is not None,
    )
    if not endpoints_closed or not worker_reaped:
        raise M4CommandError(
            "benchmark_worker_exit",
            "the fresh benchmark worker could not be safely reaped",
        ) from primary
    if primary is not None:
        raise primary
    try:
        exitcode = process.exitcode
    except BaseException as exc:
        raise M4CommandError(
            "benchmark_worker_exit",
            "the fresh benchmark worker exit status is unavailable",
        ) from exc
    if exitcode != 0 or not isinstance(message, Mapping):
        raise M4CommandError(
            "benchmark_worker_exit",
            "the fresh benchmark worker failed",
        )
    if message.get("ok") is not True:
        code = message.get("code")
        if not isinstance(code, str) or code not in _TRUSTED_COMMAND_CODES:
            code = "benchmark_worker_failure"
        raise M4CommandError(
            code,
            "the fresh benchmark worker rejected the batching gate",
        )
    report = message.get("report")
    if not isinstance(report, Mapping):
        raise M4CommandError(
            "benchmark_worker_protocol",
            "the fresh benchmark worker returned an invalid report",
        )
    return dict(report)


def _construction_report(
    manifest: WaymaxCohortManifest,
) -> dict[str, Any]:
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


def _assert_sanitized_aggregate(payload: Mapping[str, Any]) -> None:
    """Fail closed if a public-safe aggregate accidentally contains private fields."""

    forbidden_key_fragments = (
        "coordinate",
        "digest",
        "locator",
        "native_id",
        "ordinal",
        "rank",
        "scenario_id",
        "sha256",
        "trajectory",
    )

    def inspect(value: Any, *, key: str = "") -> None:
        lowered = key.lower()
        if any(fragment in lowered for fragment in forbidden_key_fragments):
            raise M4CommandError(
                "aggregate_privacy_key",
                "the aggregate report contains a forbidden private field",
            )
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                if not isinstance(child_key, str):
                    raise M4CommandError(
                        "aggregate_privacy_key",
                        "aggregate report keys must be strings",
                    )
                inspect(child_value, key=child_key)
        elif isinstance(value, (list, tuple)):
            for child in value:
                inspect(child, key=key)
        elif isinstance(value, str):
            if (
                Path(value).is_absolute()
                or PureWindowsPath(value).is_absolute()
                or _HEX_DIGEST.search(value) is not None
            ):
                raise M4CommandError(
                    "aggregate_privacy_value",
                    "the aggregate report contains a path or digest",
                )

    inspect(payload)
    try:
        json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise M4CommandError(
            "aggregate_json",
            "the aggregate report is not finite JSON-native data",
        ) from exc


def _assert_output_ignored(root: Path, output: Path) -> None:
    if not _is_git_ignored(root, output):
        raise M4CommandError(
            "output_not_ignored",
            "the M4 output directory is no longer ignored",
        )
    relative = output.relative_to(root).as_posix()
    visible = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored=no",
        "--",
        relative,
    )
    if visible:
        raise M4CommandError(
            "output_visible_to_git",
            "an M4 local artifact is visible to Git",
        )


def _publish_accepted_aggregate(
    *,
    root: Path,
    output: Path,
    provenance: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> Path:
    """Create the accepted report only after every final local gate passes."""

    if aggregate.get("accepted") is not True:
        raise M4CommandError(
            "aggregate_json",
            "the final aggregate must explicitly record accepted true",
        )
    _assert_sanitized_aggregate(aggregate)
    report_path = output / "aggregate-summary.json"
    _assert_clean_worktree(root)
    if _execution_provenance(root) != provenance:
        raise M4CommandError(
            "execution_source_changed",
            "the reviewed executable source changed during local acceptance",
        )
    _assert_output_ignored(root, output)
    if not _is_git_ignored(root, report_path):
        raise M4CommandError(
            "output_not_ignored",
            "the final aggregate report path is not ignored by Git",
        )
    _write_json_exclusive(report_path, aggregate)
    return report_path


_TERMINAL_TRANSCRIPT_NAME = "terminal-output.bin"


def _terminal_open(path: Path, flags: int, mode: int) -> int:
    return os.open(path, flags, mode)


def _terminal_dup(descriptor: int) -> int:
    return os.dup(descriptor)


def _terminal_dup2(
    source: int,
    target: int,
    *,
    inheritable: bool,
) -> int:
    return os.dup2(source, target, inheritable=inheritable)


def _terminal_close(descriptor: int) -> None:
    os.close(descriptor)


def _terminal_fstat(descriptor: int) -> os.stat_result:
    return os.fstat(descriptor)


def _terminal_lstat(path: Path) -> os.stat_result:
    return path.stat(follow_symlinks=False)


def _terminal_fsync(descriptor: int) -> None:
    os.fsync(descriptor)


def _flush_python_streams() -> None:
    sys.stdout.flush()
    sys.stderr.flush()


def _flush_native_stdio() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    fflush = libc.fflush
    fflush.argtypes = [ctypes.c_void_p]
    fflush.restype = ctypes.c_int
    if fflush(None) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "native stdio flush failed")


def _terminal_capture_error() -> M4CommandError:
    return M4CommandError(
        "terminal_capture_failed",
        "the local terminal privacy boundary could not be established",
    )


def _close_descriptors_best_effort(descriptors: Sequence[int]) -> None:
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _open_terminal_transcript(
    *,
    root: Path,
    output: Path,
) -> tuple[Path, int, tuple[int, int]]:
    """Exclusively create and validate the fixed ignored local transcript."""

    transcript_path = output / _TERMINAL_TRANSCRIPT_NAME
    try:
        if output.resolve(strict=True) != output:
            raise OSError("the output directory identity changed")
        if transcript_path.parent != output:
            raise OSError("the transcript escaped its output directory")
        if not _is_git_ignored(root, transcript_path):
            raise OSError("the transcript is not ignored")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        descriptor = _terminal_open(transcript_path, flags, 0o600)
    except (OSError, ValueError) as exc:
        raise _terminal_capture_error() from exc

    try:
        os.set_inheritable(descriptor, False)
        os.fchmod(descriptor, 0o600)
        descriptor_stat = _terminal_fstat(descriptor)
        path_stat = _terminal_lstat(transcript_path)
        identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or not stat.S_ISREG(path_stat.st_mode)
            or identity != (path_stat.st_dev, path_stat.st_ino)
            or descriptor_stat.st_nlink != 1
            or stat.S_IMODE(descriptor_stat.st_mode) != 0o600
            or not _is_git_ignored(root, transcript_path)
        ):
            raise OSError("the transcript identity or permissions are invalid")
    except (OSError, ValueError) as exc:
        _close_descriptors_best_effort((descriptor,))
        raise _terminal_capture_error() from exc
    return transcript_path, descriptor, identity


def _duplicate_terminal_descriptors(
    transcript_path: Path,
    transcript_fd: int,
    transcript_identity: tuple[int, int],
) -> _TerminalCapture:
    """Create non-inheritable restoration and status descriptors."""

    descriptors: list[int] = []
    try:
        stdout_inheritable = os.get_inheritable(1)
        stderr_inheritable = os.get_inheritable(2)
        for source in (1, 2, 1, 2):
            duplicate = _terminal_dup(source)
            descriptors.append(duplicate)
            os.set_inheritable(duplicate, False)
    except OSError as exc:
        _close_descriptors_best_effort(tuple(descriptors))
        _close_descriptors_best_effort((transcript_fd,))
        raise _terminal_capture_error() from exc

    return _TerminalCapture(
        transcript_path=transcript_path,
        transcript_fd=transcript_fd,
        transcript_identity=transcript_identity,
        restore_stdout_fd=descriptors[0],
        restore_stderr_fd=descriptors[1],
        terminal_status=_TerminalStatus(
            stdout_fd=descriptors[2],
            stderr_fd=descriptors[3],
        ),
        stdout_inheritable=stdout_inheritable,
        stderr_inheritable=stderr_inheritable,
    )


def _begin_terminal_capture(
    *,
    root: Path,
    output: Path,
) -> tuple[_TerminalCapture, BaseException | None]:
    """Set up fd-level capture, returning partial-state failure for rollback."""

    transcript_path, transcript_fd, identity = _open_terminal_transcript(
        root=root,
        output=output,
    )
    try:
        _flush_python_streams()
    except BaseException as exc:
        _close_descriptors_best_effort((transcript_fd,))
        raise _terminal_capture_error() from exc

    capture = _duplicate_terminal_descriptors(
        transcript_path,
        transcript_fd,
        identity,
    )
    try:
        _terminal_dup2(
            capture.transcript_fd,
            1,
            inheritable=True,
        )
        capture.stdout_redirected = True
        _terminal_dup2(
            capture.transcript_fd,
            2,
            inheritable=True,
        )
        capture.stderr_redirected = True
    except BaseException as exc:
        return capture, exc
    return capture, None


def _same_transcript_identity(
    capture: _TerminalCapture,
    descriptor_stat: os.stat_result,
) -> bool:
    try:
        path_stat = _terminal_lstat(capture.transcript_path)
    except OSError:
        return False
    return (
        stat.S_ISREG(descriptor_stat.st_mode)
        and stat.S_ISREG(path_stat.st_mode)
        and capture.transcript_identity
        == (descriptor_stat.st_dev, descriptor_stat.st_ino)
        == (path_stat.st_dev, path_stat.st_ino)
        and descriptor_stat.st_nlink == 1
        and stat.S_IMODE(descriptor_stat.st_mode) == 0o600
    )


def _finalize_terminal_capture(
    capture: _TerminalCapture,
    *,
    setup_failure: BaseException | None,
) -> tuple[int | None, bool]:
    """Restore terminals and close every acceptance-critical descriptor."""

    failed = setup_failure is not None
    transcript_size: int | None = None

    restorations = (
        (
            capture.restore_stdout_fd,
            1,
            capture.stdout_inheritable,
        ),
        (
            capture.restore_stderr_fd,
            2,
            capture.stderr_inheritable,
        ),
    )

    # A partial setup never starts the optional-runtime callback. Restore both
    # standard descriptors before any native flush so an unredirected descriptor
    # cannot receive bytes from a process-global C stdio buffer.
    if setup_failure is None:
        for operation in (_flush_python_streams, _flush_native_stdio):
            try:
                operation()
            except BaseException:
                failed = True
        try:
            _terminal_fsync(capture.transcript_fd)
        except OSError:
            failed = True

    for source, target, inheritable in restorations:
        try:
            _terminal_dup2(
                source,
                target,
                inheritable=inheritable,
            )
        except BaseException:
            failed = True

    if setup_failure is not None:
        try:
            _terminal_fsync(capture.transcript_fd)
        except OSError:
            failed = True

    try:
        descriptor_stat = _terminal_fstat(capture.transcript_fd)
        transcript_size = int(descriptor_stat.st_size)
        if not _same_transcript_identity(capture, descriptor_stat):
            failed = True
    except (OSError, ValueError, OverflowError):
        failed = True

    for descriptor in (
        capture.transcript_fd,
        capture.restore_stdout_fd,
        capture.restore_stderr_fd,
    ):
        try:
            _terminal_close(descriptor)
        except OSError:
            failed = True

    return transcript_size, failed


def _run_captured_phase(
    *,
    root: Path,
    output: Path,
    callback: Callable[[], _PendingAcceptance],
) -> tuple[_PendingAcceptance, _TerminalStatus]:
    """Execute optional-runtime work without allowing terminal bytes to escape."""

    capture, setup_failure = _begin_terminal_capture(
        root=root,
        output=output,
    )
    pending: _PendingAcceptance | None = None
    primary: BaseException | None = None
    if setup_failure is None:
        try:
            pending = callback()
        except BaseException as exc:
            primary = exc

    transcript_size, capture_failed = _finalize_terminal_capture(
        capture,
        setup_failure=setup_failure,
    )
    if primary is not None:
        raise _TerminalizedFailure(primary, capture.terminal_status)
    if capture_failed or pending is None or transcript_size is None:
        raise _TerminalizedFailure(
            _terminal_capture_error(),
            capture.terminal_status,
        )
    if transcript_size != 0:
        raise _TerminalizedFailure(
            M4CommandError(
                "terminal_output_detected",
                "optional runtime emitted unexpected terminal output",
            ),
            capture.terminal_status,
        )
    return pending, capture.terminal_status


def _execute_captured_acceptance(
    *,
    root: Path,
    data_dir: Path,
    output: Path,
    provenance: Mapping[str, Any],
) -> _PendingAcceptance:
    """Execute optional-runtime M4 work and return an unpublished aggregate."""

    # The exact resolver is called only after all environment, Git, and output
    # safety gates. Its implementation resolves suffixes 00000 through 00009 only.
    shard_paths = resolve_m4_validation_shards(data_dir)

    first_manifest = _scan_population(shard_paths)
    cohort_dir = output / "cohort"
    cohort_dir.mkdir(exist_ok=False)
    first_path = cohort_dir / "manifest-pass-1.json"
    _write_manifest_exclusive(first_manifest, first_path)

    second_manifest = _scan_population(shard_paths)
    second_path = cohort_dir / "manifest-pass-2.json"
    _write_manifest_exclusive(second_manifest, second_path)
    if first_manifest.canonical_bytes() != second_manifest.canonical_bytes():
        raise M4CommandError(
            "manifest_repeat_mismatch",
            "the independent repeat scan changed canonical manifest bytes",
        )

    selected = first_manifest.selected_events
    first_selected_locator = selected[0].locator
    vmap_events = select_vmap_pair(selected)
    vmap_locators = {event.locator for event in vmap_events}
    vmap_states: dict[tuple[str, int], Any] = {}
    qualification: dict[tuple[str, int], _IDMQualification] = {}
    initialized_overlap_vehicle_exclusions = 0

    # Freeze the nested subset before executing either EvalSim IDM or Waymax IDM.
    # This pass inspects source validity and frame-zero overlap only.
    def qualification_acceptance(
        event: ScanEvent,
        record: WaymaxRecord,
        scenario: Scenario,
    ) -> None:
        del scenario
        nonlocal initialized_overlap_vehicle_exclusions
        facts = _overlap_and_qualification(record.state)
        qualification[event.locator] = facts
        initialized_overlap_vehicle_exclusions += (
            facts.initialized_overlap_vehicle_exclusions
        )
        if event.locator in vmap_locators:
            vmap_states[event.locator] = record.state

    _visit_selected_records(
        data_dir=data_dir,
        manifest=first_manifest,
        selected_events=selected,
        visitor=qualification_acceptance,
    )
    if (
        len(qualification) != len(selected)
        or set(vmap_states) != vmap_locators
    ):
        raise M4CommandError(
            "full_acceptance_accounting",
            "qualification did not cover every selected scene and vmap state",
        )
    idm_events = select_idm_subset(
        selected,
        {
            locator: facts.qualifies
            for locator, facts in qualification.items()
        },
    )

    full_scenarios = 0
    stock_gate_count = 0

    def full_acceptance(
        event: ScanEvent,
        record: WaymaxRecord,
        scenario: Scenario,
    ) -> None:
        nonlocal full_scenarios
        nonlocal stock_gate_count
        _run_evalsims(scenario)
        stock_gate = event.locator == first_selected_locator
        _run_exact_log_reference(record, scenario, stock_gate=stock_gate)
        if stock_gate:
            stock_gate_count += 1
        full_scenarios += 1

    _visit_selected_records(
        data_dir=data_dir,
        manifest=first_manifest,
        selected_events=selected,
        visitor=full_acceptance,
    )
    if (
        full_scenarios != len(selected)
        or stock_gate_count != 1
    ):
        raise M4CommandError(
            "full_acceptance_accounting",
            "full-cohort acceptance did not cover every frozen selected scene",
        )

    idm_totals: Counter[str] = Counter()
    idm_nonfallback = False
    idm_scenarios = 0
    qualifying_vehicle_effective: list[int] = []
    first_idm_locator = idm_events[0].locator

    def idm_acceptance(
        event: ScanEvent,
        record: WaymaxRecord,
        scenario: Scenario,
    ) -> None:
        nonlocal idm_nonfallback
        nonlocal idm_scenarios
        facts = _run_idm_reference(
            record,
            scenario,
            expected_qualification=qualification[event.locator],
            jit_gate=event.locator == first_idm_locator,
        )
        for name in (
            "requested",
            "effective",
            "lifecycle_fallback",
            "overlap_fallback",
            "overlap_excluded_vehicles",
        ):
            idm_totals[name] += int(facts[name])
        qualifying_vehicle_effective.append(
            int(facts["qualifying_vehicle_effective"])
        )
        idm_nonfallback = idm_nonfallback or bool(
            facts["nonfallback_motion"]
        )
        idm_scenarios += 1

    _visit_selected_records(
        data_dir=data_dir,
        manifest=first_manifest,
        selected_events=idm_events,
        visitor=idm_acceptance,
    )
    if (
        idm_scenarios != len(idm_events)
        or not idm_nonfallback
        or len(qualifying_vehicle_effective) != len(idm_events)
        or min(qualifying_vehicle_effective, default=0)
        < M4_IDM_TRANSITIONS
    ):
        raise M4CommandError(
            "idm_nonvacuity",
            "the nested IDM run lacks complete positive non-fallback evidence",
        )

    ordered_vmap_states = tuple(
        vmap_states[event.locator] for event in vmap_events
    )
    benchmark = _fresh_worker_benchmark(ordered_vmap_states)

    runtime = runtime_summary()
    if runtime.get("jax_backend") != "cpu":
        raise M4CommandError(
            "jax_backend",
            "M4 local acceptance is pre-registered to the JAX CPU backend",
        )

    aggregate = {
        "accepted": True,
        "benchmark": benchmark,
        "checks": {
            "adapter_and_independent_parity_full_cohort": True,
            "evalsim_cv_full_80": True,
            "evalsim_idm_full_80": True,
            "evalsim_log_replay_full_80": True,
            "exact_log_direct_oracle_full_80": True,
            "exact_log_rollout_conversion_full_cohort": True,
            "manifest_repeat_byte_identical": True,
            "selected_locator_reload_complete": True,
            "stock_waymax_first_selected_one_step": True,
            "waymax_idm_jit_one_scene": True,
            "waymax_idm_repeat_byte_identical": True,
        },
        "cohort": _construction_report(first_manifest),
        "idm": {
            "effective_controlled_transitions": idm_totals["effective"],
            "horizon_transitions": M4_IDM_TRANSITIONS,
            "initialized_overlap_fallback_transitions": idm_totals[
                "overlap_fallback"
            ],
            "initialized_overlap_fallback_vehicles": idm_totals[
                "overlap_excluded_vehicles"
            ],
            "initialized_overlap_vehicle_exclusions_full_cohort": (
                initialized_overlap_vehicle_exclusions
            ),
            "lifecycle_fallback_transitions": idm_totals[
                "lifecycle_fallback"
            ],
            "nonfallback_motion_observed": idm_nonfallback,
            "minimum_qualifying_vehicle_effective_transitions": min(
                qualifying_vehicle_effective
            ),
            "qualifying_scenarios": sum(
                facts.qualifies for facts in qualification.values()
            ),
            "requested_controlled_transitions": idm_totals["requested"],
            "subset_scenarios": len(idm_events),
        },
        "privacy": {
            "absolute_local_values_absent": True,
            "motion_samples_absent": True,
            "private_manifests_remain_local": True,
            "source_hashes_absent": True,
            "source_identifiers_absent": True,
        },
        "purpose": "personal_non_commercial_interview_preparation",
        "runtime": runtime,
        "schema_version": M4_COMMAND_SCHEMA_VERSION,
        "shared_decode_limitation": (
            "EvalSim and Waymax reference paths share the pinned Waymax WOMD decode"
        ),
    }
    return _PendingAcceptance(aggregate=aggregate)


def run_acceptance(args: argparse.Namespace) -> _RunResult:
    """Execute the complete pre-registered M4 local acceptance."""

    if os.environ.get(LOCAL_WAYMO_ENV_FLAG) != "1":
        raise M4CommandError(
            "local_opt_in_required",
            f"set {LOCAL_WAYMO_ENV_FLAG}=1 to opt in to local WOMD access",
        )

    # Bind the pre-registered Apple-CPU runtime before its first optional import.
    configured_jax_platforms = os.environ.get("JAX_PLATFORMS")
    if configured_jax_platforms not in {None, "cpu"}:
        raise M4CommandError(
            "jax_platform_override",
            "M4 requires JAX_PLATFORMS=cpu",
        )
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    root = _project_root(args.project_root)
    _assert_running_checkout(root)
    _assert_clean_worktree(root)
    data_dir = _resolve_project_path(args.data_dir, root)
    if not data_dir.is_dir():
        raise M4CommandError(
            "data_directory_missing",
            "the explicit local validation directory does not exist",
        )
    output = _prepare_output_directory(
        args.output_dir,
        root=root,
        data_dir=data_dir,
    )
    provenance = _execution_provenance(root)
    _write_json_exclusive(output / "execution-provenance.json", provenance)

    pending, terminal_status = _run_captured_phase(
        root=root,
        output=output,
        callback=lambda: _execute_captured_acceptance(
            root=root,
            data_dir=data_dir,
            output=output,
            provenance=provenance,
        ),
    )
    try:
        # The accepted report is created only after terminal capture has been
        # restored, finalized, closed, identity-checked, and proven empty.
        report_path = _publish_accepted_aggregate(
            root=root,
            output=output,
            provenance=provenance,
            aggregate=pending.aggregate,
        )
        report_relative = report_path.relative_to(root)
    except BaseException as exc:
        raise _TerminalizedFailure(exc, terminal_status) from None
    return _RunResult(
        report_relative=report_relative,
        terminal_status=terminal_status,
    )


def _failure_code(exc: BaseException) -> str:
    if type(exc) is M4CommandError and exc.code in _TRUSTED_COMMAND_CODES:
        return exc.code
    return "unexpected_failure"


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        try:
            written = os.write(descriptor, remaining)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("status descriptor accepted no bytes")
        remaining = remaining[written:]


def _failure_line(code: str) -> bytes:
    return f"M4 local acceptance: FAIL ({code})\n".encode("ascii")


def _success_output(result: _RunResult) -> bytes:
    return (
        "M4 local acceptance: PASS\n"
        "Native WOMD identities, locators, digests, and coordinates were not printed.\n"
        f"Ignored aggregate report: {result.report_relative.as_posix()}\n"
    ).encode("ascii")


def _emit_terminalized_failure(failure: _TerminalizedFailure) -> None:
    try:
        _write_all(
            failure.terminal_status.stderr_fd,
            _failure_line(_failure_code(failure.primary)),
        )
    except OSError:
        # The preserved status channel itself is unavailable. Fail without a
        # traceback, which could disclose this source path or exception details.
        pass
    finally:
        failure.terminal_status.close_best_effort()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = run_acceptance(args)
    except _TerminalizedFailure as exc:
        _emit_terminalized_failure(exc)
        raise SystemExit(1) from None
    except BaseException as exc:
        # Third-party data/runtime exceptions may embed native values or local paths.
        # Deliberately emit only a stable code/type and keep details local.
        parser.exit(1, f"M4 local acceptance: FAIL ({_failure_code(exc)})\n")
    if result.terminal_status is None:
        sys.stdout.write(_success_output(result).decode("ascii"))
        sys.stdout.flush()
    else:
        try:
            _write_all(
                result.terminal_status.stdout_fd,
                _success_output(result),
            )
        except OSError:
            raise SystemExit(1) from None
        finally:
            result.terminal_status.close_best_effort()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
