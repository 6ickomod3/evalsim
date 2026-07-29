"""Data-free M5 end-to-end acceptance command.

This command deliberately has no WOMD, accepted-M4, or Waymax execution mode.  It
exercises the contract-first policy, metric, slice, statistics, immutable-store, and
scorecard path over five fixed synthetic scenarios.  A future official data-backed
command remains a separate, opt-in boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any, TypeVar

import pyarrow as pa

from evalsim.results import (
    ExpectedRowCounts,
    M5_RESULT_STORE_SCHEMA_VERSION,
    M5RunProvenance,
    M5ResultStore,
    PreparedM5Finalization,
    scorecard_row_from_result,
)
from evalsim.simulators import (
    ConstantVelocityPolicy,
    IDMPolicy,
    LogReplayPolicy,
)
from evalsim.simulators.waymax_reference import (
    WAYMAX_EXACT_LOG_NAME,
    WAYMAX_REFERENCE_VERSION,
)

from ._terminal import (
    TerminalBoundaryError,
    TerminalStatus,
    TerminalizedFailure,
    capture_terminal,
    write_all,
)


M5_SYNTHETIC_STATUS_SCHEMA_VERSION = "m5-cli-status-1.0.0"
M5_SYNTHETIC_PROFILE = "data_free_test"
M5_SYNTHETIC_SCENARIO_COUNT = 5
M5_SYNTHETIC_SEED = 20260728
M5_SYNTHETIC_POLICY_SEED = 0
M5_SYNTHETIC_EXPECTED_ROWS = ExpectedRowCounts(
    metric_results=195,
    slice_membership=40,
    scorecards=312,
    waymax_parity_summary=0,
)
M5_SYNTHETIC_STAGE_NAMES = (
    "preflight",
    "cohort_evaluation",
    "statistics",
    "artifact_publication",
    "finalization",
)

_RUN_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_GIT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_MAX_DIAGNOSTIC_BYTES = 2 * 1024 * 1024
_DATA_FREE_SELECTED_ORDER_VERSION = "m5-synthetic-order-1"
_NO_M4_MANIFEST_SHA256 = hashlib.sha256(
    b"evalsim-m5-data-free-no-m4-manifest-v1"
).hexdigest()
_NO_M4_PROVENANCE_SHA256 = hashlib.sha256(
    b"evalsim-m5-data-free-no-m4-execution-provenance-v1"
).hexdigest()
_SOURCE_ROOTS = (
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
_SOURCE_FILES = (
    "AGENTS.md",
    "NOTICE.md",
    "docs/data/womd-waymax-m5-metric-crosswalk.md",
    "docs/plans/2026-07-28-m5-real-womd-metrics-scorecards.md",
    "pyproject.toml",
    "uv.lock",
)
_IMPORTED_MODULE_PATHS = {
    "evalsim": "evalsim/__init__.py",
    "evalsim.cli._terminal": "evalsim/cli/_terminal.py",
    "evalsim.cli.m5_synthetic": "evalsim/cli/m5_synthetic.py",
    "evalsim.evaluation.m5": "evalsim/evaluation/m5.py",
    "evalsim.metrics.m5": "evalsim/metrics/m5.py",
    "evalsim.report.m5": "evalsim/report/m5.py",
    "evalsim.results.m5": "evalsim/results/m5.py",
    "evalsim.rollout.engine": "evalsim/rollout/engine.py",
    "evalsim.simulators.constant_velocity": (
        "evalsim/simulators/constant_velocity.py"
    ),
    "evalsim.simulators.idm": "evalsim/simulators/idm.py",
    "evalsim.simulators.log_replay": "evalsim/simulators/log_replay.py",
    "evalsim.slices.m5": "evalsim/slices/m5.py",
    "evalsim.sources.synthetic": "evalsim/sources/synthetic.py",
    "evalsim.stats.m5": "evalsim/stats/m5.py",
}
_TRUSTED_CODES = frozenset(
    {
        "argument_error",
        "artifact_exists",
        "artifact_publication_failed",
        "cohort_evaluation_failed",
        "dirty_worktree",
        "finalization_failed",
        "metric_results_write_failed",
        "output_exists",
        "output_not_ignored",
        "output_visible_to_git",
        "preflight_failed",
        "project_root_invalid",
        "result_contract_failed",
        "result_store_failed",
        "scorecard_inputs_incomplete",
        "scorecard_report_exists",
        "scorecard_report_inputs_incomplete",
        "scorecard_report_write_failed",
        "scorecards_write_failed",
        "slice_membership_write_failed",
        "statistics_failed",
        "terminal_capture_failed",
        "terminal_output_detected",
        "unexpected_failure",
        "waymax_parity_summary_write_failed",
    }
)
_T = TypeVar("_T")


class M5SyntheticCommandError(RuntimeError):
    """A command failure carrying a stable terminal-safe reason code."""

    def __init__(self, code: str, message: str) -> None:
        if code not in _TRUSTED_CODES:
            raise ValueError("unregistered M5 synthetic command error code")
        self.code = code
        super().__init__(f"{code}: {message}")


class _PrivacySafeParser(argparse.ArgumentParser):
    """Suppress path-bearing argparse diagnostics."""

    def error(self, message: str) -> None:
        del message
        self.exit(2, _rejection_output("argument_error").decode("ascii"))


@dataclass(frozen=True, slots=True)
class _RunResult:
    result_relative: Path
    row_counts: Mapping[str, int]
    elapsed_ms: Mapping[str, int]
    success_payload: bytes | None = field(default=None, repr=False)
    terminal_status: TerminalStatus | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class _PendingRun:
    prepared: PreparedM5Finalization = field(repr=False)
    result: _RunResult = field(repr=False)


@dataclass(slots=True)
class _PreparedHolder:
    prepared: PreparedM5Finalization | None = field(default=None, repr=False)


class _CommandFailure(RuntimeError):
    def __init__(
        self,
        *,
        primary: BaseException,
        failure_relative: Path | None,
        reason_code: str,
        terminal_status: TerminalStatus | None = None,
    ) -> None:
        self.primary = primary
        self.failure_relative = failure_relative
        self.reason_code = (
            reason_code
            if reason_code in _TRUSTED_CODES
            else "unexpected_failure"
        )
        self.terminal_status = terminal_status
        super().__init__("M5 synthetic command failure")


def _parser() -> argparse.ArgumentParser:
    parser = _PrivacySafeParser(
        prog="evalsim-m5-synthetic",
        description=(
            "Run the fixed five-scene, data-free M5 end-to-end acceptance. "
            "This command cannot access WOMD or run Waymax."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Exact EvalSim Git worktree root.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="New lowercase run name beneath ignored outputs/m5.",
    )
    return parser


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _rejection_output(code: str) -> bytes:
    if code not in _TRUSTED_CODES:
        code = "unexpected_failure"
    return _canonical_json_bytes(
        {
            "reason_code": code,
            "schema_version": M5_SYNTHETIC_STATUS_SCHEMA_VERSION,
            "status": "rejected",
        }
    )


def _failure_output(code: str, failure_relative: Path | None) -> bytes:
    if code not in _TRUSTED_CODES:
        code = "unexpected_failure"
    payload: dict[str, Any] = {
        "failure_record": (
            _safe_failure_relative(failure_relative)
        ),
        "reason_code": code,
        "schema_version": M5_SYNTHETIC_STATUS_SCHEMA_VERSION,
        "status": "failure",
    }
    return _canonical_json_bytes(payload)


def _success_output(result: _RunResult) -> bytes:
    elapsed = dict(result.elapsed_ms)
    if (
        set(elapsed) != set(M5_SYNTHETIC_STAGE_NAMES)
        or any(
            type(value) is not int or value < 0
            for value in elapsed.values()
        )
    ):
        raise M5SyntheticCommandError(
            "result_contract_failed",
            "the completed run omitted a fixed stage label",
        )
    rows = dict(result.row_counts)
    if rows != M5_SYNTHETIC_EXPECTED_ROWS.to_dict():
        raise M5SyntheticCommandError(
            "result_contract_failed",
            "the completed run has unexpected row accounting",
        )
    result_path = _safe_result_relative(result.result_relative)
    return _canonical_json_bytes(
        {
            "elapsed_ms": elapsed,
            "profile": M5_SYNTHETIC_PROFILE,
            "result_path": result_path,
            "row_counts": rows,
            "schema_version": M5_SYNTHETIC_STATUS_SCHEMA_VERSION,
            "status": "success",
        }
    )


def _safe_result_relative(value: Any) -> str:
    if not isinstance(value, Path):
        raise M5SyntheticCommandError(
            "result_contract_failed",
            "the completed run has an invalid result path",
        )
    text = value.as_posix()
    windows = PureWindowsPath(text)
    if (
        value.is_absolute()
        or windows.is_absolute()
        or value.parts[:2] != ("outputs", "m5")
        or len(value.parts) != 3
        or _RUN_NAME.fullmatch(value.parts[2]) is None
        or value.parts[2] in {".", ".."}
        or text != f"outputs/m5/{value.parts[2]}"
    ):
        raise M5SyntheticCommandError(
            "result_contract_failed",
            "the completed run path is outside ignored outputs/m5",
        )
    return text


def _safe_failure_relative(value: Path | None) -> str | None:
    if not isinstance(value, Path):
        return None
    parent = value.parent
    if value.name != "FAILURE.json":
        return None
    try:
        parent_text = _safe_result_relative(parent)
    except M5SyntheticCommandError:
        return None
    return f"{parent_text}/FAILURE.json"


def _git_process(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ("git", "-C", os.fspath(root), *arguments),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise M5SyntheticCommandError(
            "project_root_invalid",
            "Git cannot inspect the explicit project root",
        ) from exc


def _git_bytes(root: Path, *arguments: str) -> bytes:
    completed = _git_process(root, *arguments)
    if completed.returncode != 0:
        raise M5SyntheticCommandError(
            "preflight_failed",
            "a required read-only Git inspection failed",
        )
    return completed.stdout


def _git_text(root: Path, *arguments: str) -> str:
    try:
        value = _git_bytes(root, *arguments).decode(
            "utf-8",
            errors="strict",
        ).strip()
    except UnicodeDecodeError as exc:
        raise M5SyntheticCommandError(
            "preflight_failed",
            "Git returned non-UTF-8 provenance",
        ) from exc
    if not value or "\n" in value or "\r" in value:
        raise M5SyntheticCommandError(
            "preflight_failed",
            "Git returned invalid provenance",
        )
    return value


def _validated_root(candidate: Path) -> Path:
    try:
        lexical = Path(os.path.abspath(os.fspath(candidate)))
        root = lexical.resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise M5SyntheticCommandError(
            "project_root_invalid",
            "the explicit project root does not exist",
        ) from exc
    if (
        lexical != root
        or not root.is_dir()
        or not (root / "pyproject.toml").is_file()
        or not (root / ".gitignore").is_file()
    ):
        raise M5SyntheticCommandError(
            "project_root_invalid",
            "the explicit project root is not a canonical EvalSim checkout",
        )
    top = _git_text(root, "rev-parse", "--show-toplevel")
    try:
        git_root = Path(top).resolve(strict=True)
    except OSError as exc:
        raise M5SyntheticCommandError(
            "project_root_invalid",
            "the Git worktree root cannot be resolved",
        ) from exc
    if git_root != root:
        raise M5SyntheticCommandError(
            "project_root_invalid",
            "the explicit project root is not the Git worktree root",
        )
    return root


def _validated_run_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or _RUN_NAME.fullmatch(value) is None
        or value in {".", ".."}
    ):
        raise M5SyntheticCommandError(
            "argument_error",
            "run_name must be one safe lowercase path component",
        )
    return value


def _assert_running_checkout(root: Path) -> None:
    for module_name, relative_text in _IMPORTED_MODULE_PATHS.items():
        module = importlib.import_module(module_name)
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            raise M5SyntheticCommandError(
                "preflight_failed",
                "an execution module has no filesystem source",
            )
        try:
            actual = Path(raw_path).resolve(strict=True)
            expected = (root / relative_text).resolve(strict=True)
        except OSError as exc:
            raise M5SyntheticCommandError(
                "preflight_failed",
                "an execution module source cannot be resolved",
            ) from exc
        if actual != expected:
            raise M5SyntheticCommandError(
                "preflight_failed",
                "running code does not come from the bound Git checkout",
            )


def _preflight_output(root: Path, run_name: str) -> None:
    status = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored=no",
        "-z",
    )
    if status:
        raise M5SyntheticCommandError(
            "dirty_worktree",
            "the data-free acceptance requires a clean worktree",
        )
    relative = Path("outputs") / "m5" / run_name
    target = root / relative
    if os.path.lexists(target):
        raise M5SyntheticCommandError(
            "output_exists",
            "the requested M5 run already exists",
        )
    ignored = _git_process(
        root,
        "check-ignore",
        "--quiet",
        "--no-index",
        "--",
        relative.as_posix(),
    )
    if ignored.returncode != 0:
        raise M5SyntheticCommandError(
            "output_not_ignored",
            "the requested M5 run is not ignored by Git",
        )
    tracked = _git_bytes(
        root,
        "ls-files",
        "-z",
        "--",
        relative.as_posix(),
    )
    if tracked:
        raise M5SyntheticCommandError(
            "output_visible_to_git",
            "the requested M5 run path contains tracked files",
        )


def _tracked_source_paths(root: Path) -> tuple[str, ...]:
    encoded = _git_bytes(
        root,
        "ls-files",
        "-z",
        "--",
        *_SOURCE_ROOTS,
        *_SOURCE_FILES,
    )
    try:
        paths = tuple(
            item.decode("utf-8", errors="strict")
            for item in encoded.split(b"\0")
            if item
        )
    except UnicodeDecodeError as exc:
        raise M5SyntheticCommandError(
            "preflight_failed",
            "tracked executable paths are not UTF-8",
        ) from exc
    canonical = tuple(sorted(paths))
    if (
        not canonical
        or paths != canonical
        or len(set(canonical)) != len(canonical)
        or not set(_SOURCE_FILES).issubset(canonical)
    ):
        raise M5SyntheticCommandError(
            "preflight_failed",
            "the tracked executable-source set is incomplete or noncanonical",
        )
    for root_name in _SOURCE_ROOTS:
        prefix = f"{root_name}/"
        if not any(path.startswith(prefix) for path in canonical):
            raise M5SyntheticCommandError(
                "preflight_failed",
                "a required executable-source root is empty",
            )
    _reject_untracked_executables(root, set(canonical))
    return canonical


def _reject_untracked_executables(
    root: Path,
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
    for root_name in _SOURCE_ROOTS:
        source_root = root.joinpath(*Path(root_name).parts)
        try:
            if (
                source_root.resolve(strict=True) != source_root
                or not source_root.is_dir()
            ):
                raise OSError("source root is missing or linked")
        except OSError as exc:
            raise M5SyntheticCommandError(
                "preflight_failed",
                "an executable-source root is unsafe",
            ) from exc
        for directory, directory_names, file_names in os.walk(
            source_root,
            topdown=True,
            followlinks=False,
        ):
            directory_path = Path(directory)
            retained: list[str] = []
            for name in directory_names:
                candidate = directory_path / name
                if name in cache_directories:
                    continue
                if candidate.is_symlink():
                    raise M5SyntheticCommandError(
                        "preflight_failed",
                        "executable-source roots cannot contain symlinks",
                    )
                retained.append(name)
            directory_names[:] = retained
            for name in file_names:
                candidate = directory_path / name
                relative = candidate.relative_to(root).as_posix()
                if relative in tracked_paths:
                    continue
                try:
                    mode = candidate.lstat().st_mode
                except OSError as exc:
                    raise M5SyntheticCommandError(
                        "preflight_failed",
                        "an executable-source candidate cannot be inspected",
                    ) from exc
                if stat.S_ISLNK(mode):
                    raise M5SyntheticCommandError(
                        "preflight_failed",
                        "executable-source roots cannot contain symlinks",
                    )
                if (
                    stat.S_ISREG(mode)
                    and (
                        candidate.suffix.lower() in executable_suffixes
                        or mode & 0o111
                    )
                ):
                    raise M5SyntheticCommandError(
                        "preflight_failed",
                        "an untracked executable exists in a bound source root",
                    )


def _package_version(distribution: str) -> str:
    try:
        value = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"
    if not value:
        raise M5SyntheticCommandError(
            "preflight_failed",
            "an installed dependency has no version",
        )
    return value


def _selected_order_fingerprint() -> str:
    from evalsim.evaluation import synthetic_source_evidence

    payload = dict(synthetic_source_evidence())
    expected_fields = {
        "adapter_version",
        "case_count",
        "current_index",
        "dt_seconds",
        "num_steps",
        "order_version",
        "ordered_scenario_ids",
        "seed",
        "source_fingerprint",
        "split",
    }
    ordered_ids = payload.get("ordered_scenario_ids")
    ids_valid = (
        not isinstance(ordered_ids, (str, bytes))
        and isinstance(ordered_ids, Sequence)
        and len(ordered_ids) == M5_SYNTHETIC_SCENARIO_COUNT
        and all(isinstance(item, str) and item for item in ordered_ids)
    )
    if (
        set(payload) != expected_fields
        or payload.get("case_count") != M5_SYNTHETIC_SCENARIO_COUNT
        or payload.get("seed") != M5_SYNTHETIC_SEED
        or payload.get("order_version")
        != _DATA_FREE_SELECTED_ORDER_VERSION
        or not ids_valid
        or len(set(ordered_ids)) != M5_SYNTHETIC_SCENARIO_COUNT
    ):
        raise M5SyntheticCommandError(
            "preflight_failed",
            "the fixed synthetic source evidence is invalid",
        )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(
        b"evalsim-m5-synthetic-selected-order-v1\0"
        + encoded
    ).hexdigest()


def _simulator_specs() -> dict[str, Any]:
    policies = (
        LogReplayPolicy(),
        ConstantVelocityPolicy(),
        IDMPolicy(),
    )
    specs: dict[str, Any] = {}
    for policy in policies:
        metadata = policy.metadata()
        payload = metadata.to_dict()
        specs[metadata.name] = {
            "deterministic": metadata.deterministic,
            "execution_role": "policy",
            "parameters": payload["params"],
            "version": metadata.version,
        }
    specs[WAYMAX_EXACT_LOG_NAME] = {
        "deterministic": True,
        "execution_role": "reference",
        "parameters": {
            "executed": False,
            "reason": "data_free_synthetic_profile",
        },
        "version": WAYMAX_REFERENCE_VERSION,
    }
    return specs


def _build_provenance(root: Path) -> M5RunProvenance:
    paths = _tracked_source_paths(root)
    commit = _git_text(root, "rev-parse", "--verify", "HEAD^{commit}")
    tree = _git_text(root, "rev-parse", "--verify", "HEAD^{tree}")
    if (
        _GIT_OBJECT_ID.fullmatch(commit) is None
        or _GIT_OBJECT_ID.fullmatch(tree) is None
    ):
        raise M5SyntheticCommandError(
            "preflight_failed",
            "Git returned noncanonical commit/tree identities",
        )
    from evalsim.results import executable_source_fingerprint

    return M5RunProvenance(
        m4_manifest_sha256=_NO_M4_MANIFEST_SHA256,
        m4_execution_provenance_sha256=_NO_M4_PROVENANCE_SHA256,
        selected_order_version=_DATA_FREE_SELECTED_ORDER_VERSION,
        selected_order_fingerprint_sha256=_selected_order_fingerprint(),
        executable_source_fingerprint_sha256=(
            executable_source_fingerprint(root, paths)
        ),
        executable_source_paths=paths,
        git_commit=commit,
        git_tree=tree,
        simulator_specs=_simulator_specs(),
        runtime_versions={
            "flax": _package_version("flax"),
            "jax": _package_version("jax"),
            "jaxlib": _package_version("jaxlib"),
            "numpy": _package_version("numpy"),
            "pyarrow": pa.__version__,
            "python": platform.python_version(),
            "tensorflow": _package_version("tensorflow"),
            "waymo_waymax": _package_version("waymo-waymax"),
        },
    )


def _elapsed_ms(start_ns: int, end_ns: int) -> int:
    if (
        isinstance(start_ns, bool)
        or isinstance(end_ns, bool)
        or not isinstance(start_ns, int)
        or not isinstance(end_ns, int)
        or end_ns < start_ns
    ):
        raise M5SyntheticCommandError(
            "result_contract_failed",
            "the monotonic stage clock moved backwards",
        )
    return (end_ns - start_ns) // 1_000_000


def _timed(
    stage: str,
    callback: Callable[[], _T],
    elapsed: dict[str, int],
) -> _T:
    if stage not in M5_SYNTHETIC_STAGE_NAMES or stage in elapsed:
        raise M5SyntheticCommandError(
            "result_contract_failed",
            "a stage label is unknown or repeated",
        )
    start = time.monotonic_ns()
    value = callback()
    elapsed[stage] = _elapsed_ms(start, time.monotonic_ns())
    return value


def _run_evaluation() -> Any:
    from evalsim.evaluation.m5 import run_synthetic_m5_evaluation

    return run_synthetic_m5_evaluation()


def _scorecard_rows(result: Any) -> list[dict[str, Any]]:
    raw_results = getattr(result, "scorecard_results", None)
    if (
        isinstance(raw_results, (str, bytes))
        or not isinstance(raw_results, Sequence)
        or len(raw_results) != M5_SYNTHETIC_EXPECTED_ROWS.scorecards
    ):
        raise M5SyntheticCommandError(
            "result_contract_failed",
            "the synthetic runner returned an invalid scorecard domain",
        )
    try:
        return [scorecard_row_from_result(item) for item in raw_results]
    except (TypeError, ValueError) as exc:
        raise M5SyntheticCommandError(
            "statistics_failed",
            "a runner scorecard result failed its fixed adapter",
        ) from exc


def _validate_runner_result(result: Any) -> None:
    for name, expected in (
        ("metric_rows", M5_SYNTHETIC_EXPECTED_ROWS.metric_results),
        ("slice_rows", M5_SYNTHETIC_EXPECTED_ROWS.slice_membership),
        ("scorecard_results", M5_SYNTHETIC_EXPECTED_ROWS.scorecards),
        ("zero_oracles", 25),
    ):
        value = getattr(result, name, None)
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
            or len(value) != expected
        ):
            raise M5SyntheticCommandError(
                "result_contract_failed",
                f"the synthetic runner returned invalid {name} accounting",
            )


def _write_evaluation(
    store: M5ResultStore,
    result: Any,
    scorecard_rows: Sequence[Mapping[str, Any]],
) -> None:
    metric_rows = list(result.metric_rows)
    indices = {
        row.get("cohort_index")
        for row in metric_rows
        if isinstance(row, Mapping)
    }
    if indices != set(range(M5_SYNTHETIC_SCENARIO_COUNT)):
        raise M5SyntheticCommandError(
            "result_contract_failed",
            "metric rows do not cover the fixed synthetic cohort",
        )
    for part_index in range(M5_SYNTHETIC_SCENARIO_COUNT):
        part = [
            row
            for row in metric_rows
            if isinstance(row, Mapping)
            and row.get("cohort_index") == part_index
        ]
        if len(part) != 39:
            raise M5SyntheticCommandError(
                "result_contract_failed",
                "a synthetic scenario has incomplete metric execution",
            )
        store.write_metric_results_part(part, part_index=part_index)
    store.write_slice_membership(result.slice_rows)
    store.write_scorecards(scorecard_rows)
    store.write_waymax_parity_summary(())
    store.write_human_readable_scorecard()


def _execute_pending(
    store: M5ResultStore,
    root: Path,
    prepared_holder: _PreparedHolder,
) -> _PendingRun:
    elapsed: dict[str, int] = {}
    try:
        provenance = _timed(
            "preflight",
            lambda: _build_provenance(root),
            elapsed,
        )
    except M5SyntheticCommandError:
        raise
    except Exception as exc:
        raise M5SyntheticCommandError(
            "preflight_failed",
            "data-free provenance construction failed",
        ) from exc

    try:
        result = _timed(
            "cohort_evaluation",
            _run_evaluation,
            elapsed,
        )
        _validate_runner_result(result)
    except M5SyntheticCommandError:
        raise
    except Exception as exc:
        raise M5SyntheticCommandError(
            "cohort_evaluation_failed",
            "the fixed synthetic cohort failed evaluation",
        ) from exc

    try:
        scorecards = _timed(
            "statistics",
            lambda: _scorecard_rows(result),
            elapsed,
        )
    except M5SyntheticCommandError:
        raise
    except Exception as exc:
        raise M5SyntheticCommandError(
            "statistics_failed",
            "the fixed synthetic statistics stage failed",
        ) from exc

    try:
        _timed(
            "artifact_publication",
            lambda: _write_evaluation(store, result, scorecards),
            elapsed,
        )
    except M5SyntheticCommandError:
        raise
    except Exception as exc:
        raise M5SyntheticCommandError(
            "artifact_publication_failed",
            "the immutable data-free artifacts could not be written",
        ) from exc

    try:
        start = time.monotonic_ns()
        prepared = store.prepare_finalization(provenance=provenance)
        # Publish the capability immediately.  If timing, serialization, native
        # flushing, or transcript inspection fails after preparation, the caller
        # can still abort this exact run without ever creating SUCCESS.
        prepared_holder.prepared = prepared
        elapsed["finalization"] = _elapsed_ms(
            start,
            time.monotonic_ns(),
        )
    except M5SyntheticCommandError:
        raise
    except Exception as exc:
        raise M5SyntheticCommandError(
            "finalization_failed",
            "the immutable M5 store failed verified preparation",
        ) from exc

    provisional = _RunResult(
        result_relative=store.project_relative_path,
        row_counts=store.row_counts,
        elapsed_ms=elapsed,
    )
    # Serialize and validate the exact public success object before SUCCESS can
    # exist.  The post-commit path performs no timing or result-contract work.
    success_payload = _success_output(provisional)
    return _PendingRun(
        prepared=prepared,
        result=_RunResult(
            result_relative=provisional.result_relative,
            row_counts=provisional.row_counts,
            elapsed_ms=provisional.elapsed_ms,
            success_payload=success_payload,
        ),
    )


def _failure_code(exc: BaseException) -> str:
    if type(exc) is M5SyntheticCommandError and exc.code in _TRUSTED_CODES:
        return exc.code
    if type(exc) is TerminalBoundaryError and exc.code in _TRUSTED_CODES:
        return exc.code
    return "unexpected_failure"


def _diagnostic_bytes(primary: BaseException, transcript: bytes) -> bytes:
    try:
        formatted = "".join(
            traceback.format_exception(primary)
        ).encode("utf-8", errors="backslashreplace")
    except Exception:
        formatted = b"diagnostic traceback unavailable\n"
    payload = (
        b"=== captured stdout/stderr ===\n"
        + transcript[:_MAX_DIAGNOSTIC_BYTES]
        + b"\n=== exception ===\n"
        + formatted[:_MAX_DIAGNOSTIC_BYTES]
    )
    return payload[:_MAX_DIAGNOSTIC_BYTES]


def _write_failure_diagnostics(
    run_path: Path,
    primary: BaseException,
    transcript: bytes,
) -> None:
    path = run_path / "failure-details.log"
    if os.path.lexists(path):
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        payload = _diagnostic_bytes(primary, transcript)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("failure diagnostic accepted no bytes")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_canonical_failure_record(
    store: M5ResultStore,
) -> tuple[Path, str] | None:
    """Return the exact local failure identity only for a trusted record."""

    failure_path = store.run_path / "FAILURE.json"
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(failure_path, flags)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > 4096
        ):
            return None
        chunks = bytearray()
        while len(chunks) <= 4096:
            chunk = os.read(descriptor, 4097 - len(chunks))
            if not chunk:
                break
            chunks.extend(chunk)
        encoded = bytes(chunks)
        if len(encoded) != metadata.st_size or len(encoded) > 4096:
            return None
        payload = json.loads(encoded.decode("ascii", errors="strict"))
        if (
            not isinstance(payload, dict)
            or set(payload)
            != {"complete", "reason_code", "schema_version"}
            or payload.get("complete") is not False
            or payload.get("schema_version")
            != M5_RESULT_STORE_SCHEMA_VERSION
            or _canonical_json_bytes(payload) != encoded
        ):
            return None
        reason = payload.get("reason_code")
        if not isinstance(reason, str) or reason not in _TRUSTED_CODES:
            return None
        relative = store.project_relative_path / "FAILURE.json"
        if _safe_failure_relative(relative) is None:
            return None
        return relative, reason
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        return None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _persist_failure(
    store: M5ResultStore,
    primary: BaseException,
    transcript: bytes,
    *,
    prepared: PreparedM5Finalization | None = None,
) -> tuple[Path | None, str]:
    requested_code = _failure_code(primary)
    failure_path = store.run_path / "FAILURE.json"
    if not os.path.lexists(failure_path):
        try:
            if prepared is None:
                store.fail(requested_code)
            else:
                store.abort_finalization(prepared, requested_code)
        except Exception:
            pass
    persisted = _read_canonical_failure_record(store)
    if persisted is None:
        # Never point at a record whose reason cannot be proven identical to the
        # public status.
        return None, "unexpected_failure"
    try:
        _write_failure_diagnostics(store.run_path, primary, transcript)
    except OSError:
        pass
    return persisted


def run_synthetic(args: argparse.Namespace) -> _RunResult:
    """Execute the fixed synthetic acceptance and finalize one immutable run."""

    root = _validated_root(args.project_root)
    run_name = _validated_run_name(args.run_name)
    _assert_running_checkout(root)
    _preflight_output(root, run_name)
    try:
        store = M5ResultStore.create(
            root,
            run_name,
            expected_rows=M5_SYNTHETIC_EXPECTED_ROWS,
            data_free=True,
        )
    except FileExistsError as exc:
        raise M5SyntheticCommandError(
            "output_exists",
            "the requested M5 run already exists",
        ) from exc
    except Exception as exc:
        raise M5SyntheticCommandError(
            "preflight_failed",
            "the ignored M5 run could not be created",
        ) from exc

    prepared_holder = _PreparedHolder()
    try:
        captured = capture_terminal(
            lambda: _execute_pending(
                store,
                root,
                prepared_holder,
            )
        )
    except TerminalizedFailure as exc:
        relative, reason_code = _persist_failure(
            store,
            exc.primary,
            exc.transcript,
            prepared=prepared_holder.prepared,
        )
        raise _CommandFailure(
            primary=exc.primary,
            failure_relative=relative,
            reason_code=reason_code,
            terminal_status=exc.terminal_status,
        ) from None
    except BaseException as exc:
        relative, reason_code = _persist_failure(
            store,
            exc,
            b"",
            prepared=prepared_holder.prepared,
        )
        raise _CommandFailure(
            primary=exc,
            failure_relative=relative,
            reason_code=reason_code,
        ) from None

    pending = captured.value
    if (
        not isinstance(pending, _PendingRun)
        or not isinstance(
            pending.prepared,
            PreparedM5Finalization,
        )
        or pending.prepared is not prepared_holder.prepared
        or pending.result.success_payload is None
        or type(pending.result.success_payload) is not bytes
    ):
        primary = M5SyntheticCommandError(
            "result_contract_failed",
            "the captured run omitted its prepared success contract",
        )
        relative, reason_code = _persist_failure(
            store,
            primary,
            b"",
            prepared=prepared_holder.prepared,
        )
        raise _CommandFailure(
            primary=primary,
            failure_relative=relative,
            reason_code=reason_code,
            terminal_status=captured.terminal_status,
        ) from None

    result = pending.result
    # Bind the already-open safe status channel before the terminal store bit is
    # committed.  No timing, serialization, validation, or object construction is
    # allowed after commit_finalization succeeds.
    object.__setattr__(
        result,
        "terminal_status",
        captured.terminal_status,
    )
    try:
        store.commit_finalization(pending.prepared)
    except BaseException as exc:
        primary = M5SyntheticCommandError(
            "finalization_failed",
            "the immutable M5 store failed its terminal commit",
        )
        relative, reason_code = _persist_failure(
            store,
            primary,
            b"",
            prepared=pending.prepared,
        )
        raise _CommandFailure(
            primary=primary,
            failure_relative=relative,
            reason_code=reason_code,
            terminal_status=captured.terminal_status,
        ) from exc

    return result


def _emit_failure(failure: _CommandFailure) -> None:
    payload = _failure_output(
        failure.reason_code,
        failure.failure_relative,
    )
    status = failure.terminal_status
    if status is None:
        try:
            sys.stderr.write(payload.decode("ascii"))
            sys.stderr.flush()
        except (OSError, UnicodeError):
            pass
        return
    try:
        write_all(status.stderr_fd, payload)
    except OSError:
        pass
    finally:
        status.close_best_effort()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = run_synthetic(args)
        payload = (
            result.success_payload
            if result.success_payload is not None
            else _success_output(result)
        )
    except _CommandFailure as exc:
        _emit_failure(exc)
        raise SystemExit(1) from None
    except BaseException as exc:
        parser.exit(1, _rejection_output(_failure_code(exc)).decode("ascii"))

    if result.terminal_status is None:
        sys.stdout.write(payload.decode("ascii"))
        sys.stdout.flush()
    else:
        try:
            write_all(result.terminal_status.stdout_fd, payload)
        except OSError:
            raise SystemExit(1) from None
        finally:
            result.terminal_status.close_best_effort()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "M5_SYNTHETIC_EXPECTED_ROWS",
    "M5_SYNTHETIC_PROFILE",
    "M5_SYNTHETIC_SCENARIO_COUNT",
    "M5_SYNTHETIC_STAGE_NAMES",
    "M5_SYNTHETIC_STATUS_SCHEMA_VERSION",
    "M5SyntheticCommandError",
    "main",
    "run_synthetic",
]
