"""Opt-in, locally bound official M5 WOMD acceptance command.

The command is intentionally import-safe without JAX, TensorFlow, or Waymax.  It
performs every repository, source, output, shard-path, and accepted-M4 preflight
before importing the optional execution stack, then visits the accepted cohort
through its bounded reload adapter.  Terminal output is an allowlisted aggregate
status only; detailed failures remain in the ignored run directory.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any, TypeVar

from evalsim.results import (
    M5_M4_INTEGRITY_ASSUMPTION,
    M5_M4_REUSE_SCHEMA_VERSION,
    M5_RESULT_STORE_SCHEMA_VERSION,
    OFFICIAL_M5_ROW_COUNTS,
    OFFICIAL_WAYMAX_REFERENCE_PARAMETERS,
    M5DeterminismReceipt,
    M5RunProvenance,
    M5ResultStore,
    PreparedM5Finalization,
    executable_source_fingerprint,
    official_executable_source_paths,
    scorecard_row_from_result,
    verify_committed_m5_result_store,
    verify_prepared_m5_result_store,
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


M5_OFFICIAL_STATUS_SCHEMA_VERSION = "m5-cli-status-1.0.0"
M5_OFFICIAL_PROFILE = "official_m5"
M5_OFFICIAL_POLICY_SEED = 0
M5_OFFICIAL_SCENARIO_COUNT = 128
M5_OFFICIAL_METRIC_ROWS_PER_CASE = 52
M5_OFFICIAL_SLICE_ROWS_PER_CASE = 8
M5_OFFICIAL_STAGE_NAMES = (
    "preflight",
    "cohort_evaluation",
    "statistics",
    "artifact_publication",
    "finalization",
)

_LOCAL_OPT_IN = "EVALSIM_RUN_WAYMO_LOCAL"
_CANONICAL_REMOTE = "https://github.com/6ickomod3/evalsim.git"
_CANONICAL_REMOTE_REF = "refs/heads/main"
_RUN_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_GIT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_MAX_DIAGNOSTIC_BYTES = 2 * 1024 * 1024
_IMPORTED_MODULE_PATHS = {
    "evalsim": "evalsim/__init__.py",
    "evalsim.cli._terminal": "evalsim/cli/_terminal.py",
    "evalsim.cli.m5_official": "evalsim/cli/m5_official.py",
    "evalsim.evaluation.m5": "evalsim/evaluation/m5.py",
    "evalsim.evaluation.m5_waymax": "evalsim/evaluation/m5_waymax.py",
    "evalsim.metrics.m5": "evalsim/metrics/m5.py",
    "evalsim.report.m5": "evalsim/report/m5.py",
    "evalsim.results.m5": "evalsim/results/m5.py",
    "evalsim.rollout.engine": "evalsim/rollout/engine.py",
    "evalsim.simulators.constant_velocity": (
        "evalsim/simulators/constant_velocity.py"
    ),
    "evalsim.simulators.idm": "evalsim/simulators/idm.py",
    "evalsim.simulators.log_replay": "evalsim/simulators/log_replay.py",
    "evalsim.simulators.waymax_reference": (
        "evalsim/simulators/waymax_reference.py"
    ),
    "evalsim.slices.m5": "evalsim/slices/m5.py",
    "evalsim.sources.m5_m4_reuse": "evalsim/sources/m5_m4_reuse.py",
    "evalsim.sources.waymax_cohort": "evalsim/sources/waymax_cohort.py",
    "evalsim.sources.waymax_loader": "evalsim/sources/waymax_loader.py",
    "evalsim.stats.m5": "evalsim/stats/m5.py",
}
_TRUSTED_CODES = frozenset(
    {
        "argument_error",
        "artifact_exists",
        "artifact_publication_failed",
        "cohort_evaluation_failed",
        "data_directory_invalid",
        "determinism_receipt_before_parity_order",
        "determinism_receipt_write_failed",
        "dirty_worktree",
        "environment_not_enabled",
        "finalization_failed",
        "git_remote_invalid",
        "m4_drift",
        "m4_preflight_failed",
        "metric_results_write_failed",
        "output_drift",
        "output_exists",
        "output_not_ignored",
        "output_visible_to_git",
        "parity_order_receipt_late",
        "parity_order_receipt_missing",
        "parity_order_receipt_write_failed",
        "preflight_failed",
        "project_root_invalid",
        "remote_main_mismatch",
        "result_contract_failed",
        "result_store_failed",
        "scorecard_inputs_incomplete",
        "scorecard_report_exists",
        "scorecard_report_inputs_incomplete",
        "scorecard_report_write_failed",
        "scorecards_write_failed",
        "shard_set_invalid",
        "slice_membership_write_failed",
        "source_binding_failed",
        "source_drift",
        "statistics_failed",
        "terminal_capture_failed",
        "terminal_output_detected",
        "unexpected_failure",
        "unpushed_main",
        "verification_failed",
        "waymax_parity_summary_write_failed",
    }
)
_T = TypeVar("_T")


class M5OfficialCommandError(RuntimeError):
    """A privacy-safe official-command failure with a stable reason code."""

    def __init__(self, code: str, message: str) -> None:
        if code not in _TRUSTED_CODES:
            raise ValueError("unregistered M5 official command error code")
        self.code = code
        super().__init__(f"{code}: {message}")


class _PrivacySafeParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.exit(2, _rejection_output("argument_error").decode("ascii"))


@dataclass(frozen=True, slots=True)
class _GitBinding:
    commit: str
    tree: str
    executable_paths: tuple[str, ...]
    executable_fingerprint: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class _OfficialContext:
    root: Path = field(repr=False)
    data_dir: Path = field(repr=False)
    shard_paths: tuple[Path, ...] = field(repr=False)
    run_name: str
    cohort: Any = field(repr=False)
    parity_selection: Any = field(repr=False)
    provenance: M5RunProvenance = field(repr=False)
    git_binding: _GitBinding = field(repr=False)
    preflight_ms: int


@dataclass(frozen=True, slots=True)
class _CohortState:
    scorecard_accumulator: Any = field(repr=False)
    parity_rows: tuple[Any, ...] = field(repr=False)


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
class _RunHolder:
    store: M5ResultStore | None = field(default=None, repr=False)
    prepared: PreparedM5Finalization | None = field(default=None, repr=False)
    context: _OfficialContext | None = field(default=None, repr=False)


class _CommandFailure(RuntimeError):
    def __init__(
        self,
        *,
        primary: BaseException,
        failure_relative: Path | None,
        reason_code: str,
        status: str,
        terminal_status: TerminalStatus | None = None,
    ) -> None:
        self.primary = primary
        self.failure_relative = failure_relative
        self.reason_code = (
            reason_code
            if reason_code in _TRUSTED_CODES
            else "unexpected_failure"
        )
        self.status = (
            status if status in {"failure", "rejected"} else "rejected"
        )
        self.terminal_status = terminal_status
        super().__init__("M5 official command failure")


def _parser() -> argparse.ArgumentParser:
    parser = _PrivacySafeParser(
        prog="evalsim-m5-official",
        description=(
            "Run the opt-in official M5 acceptance over the exact accepted "
            "local M4 cohort. Generated results remain ignored and local."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Exact clean EvalSim Git worktree root.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Exact local WOMD v1.3.1 validation TFRecord directory.",
    )
    parser.add_argument(
        "--m4-run-dir",
        type=Path,
        required=True,
        help="Exact ignored accepted M4 run directory.",
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
            "schema_version": M5_OFFICIAL_STATUS_SCHEMA_VERSION,
            "status": "rejected",
        }
    )


def _failure_output(
    code: str,
    failure_relative: Path | None,
    *,
    status: str,
) -> bytes:
    if code not in _TRUSTED_CODES:
        code = "unexpected_failure"
    if status == "rejected":
        return _rejection_output(code)
    return _canonical_json_bytes(
        {
            "failure_record": _safe_failure_relative(failure_relative),
            "reason_code": code,
            "schema_version": M5_OFFICIAL_STATUS_SCHEMA_VERSION,
            "status": "failure",
        }
    )


def _success_output(result: _RunResult) -> bytes:
    elapsed = dict(result.elapsed_ms)
    if (
        set(elapsed) != set(M5_OFFICIAL_STAGE_NAMES)
        or any(type(value) is not int or value < 0 for value in elapsed.values())
    ):
        raise M5OfficialCommandError(
            "result_contract_failed",
            "the completed run omitted a fixed stage label",
        )
    rows = dict(result.row_counts)
    if rows != OFFICIAL_M5_ROW_COUNTS.to_dict():
        raise M5OfficialCommandError(
            "result_contract_failed",
            "the completed run has unexpected row accounting",
        )
    return _canonical_json_bytes(
        {
            "elapsed_ms": elapsed,
            "profile": M5_OFFICIAL_PROFILE,
            "result_path": _safe_result_relative(result.result_relative),
            "row_counts": rows,
            "schema_version": M5_OFFICIAL_STATUS_SCHEMA_VERSION,
            "status": "success",
        }
    )


def _safe_result_relative(value: Any) -> str:
    if not isinstance(value, Path):
        raise M5OfficialCommandError(
            "result_contract_failed",
            "the completed run has an invalid result path",
        )
    text = value.as_posix()
    windows = PureWindowsPath(text)
    if (
        value.is_absolute()
        or windows.is_absolute()
        or len(value.parts) != 3
        or value.parts[:2] != ("outputs", "m5")
        or _RUN_NAME.fullmatch(value.parts[2]) is None
        or value.parts[2] in {".", ".."}
        or text != f"outputs/m5/{value.parts[2]}"
    ):
        raise M5OfficialCommandError(
            "result_contract_failed",
            "the completed run path is outside ignored outputs/m5",
        )
    return text


def _safe_failure_relative(value: Path | None) -> str | None:
    if not isinstance(value, Path) or value.name != "FAILURE.json":
        return None
    try:
        parent = _safe_result_relative(value.parent)
    except M5OfficialCommandError:
        return None
    return f"{parent}/FAILURE.json"


def _git_process(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    git_env = _isolated_git_environment()
    try:
        return subprocess.run(
            ("git", "-C", os.fspath(root), *arguments),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=git_env,
        )
    except OSError as exc:
        raise M5OfficialCommandError(
            "project_root_invalid",
            "Git cannot inspect the explicit project root",
        ) from exc


def _isolated_git_environment() -> dict[str, str]:
    """Build a small environment without inherited Git/config redirection."""

    # A copied environment plus a blacklist is fragile: Git adds environment
    # controls over time, and SSL/proxy variables can silently change the live
    # provenance endpoint.  Local read-only commands need only a tool search path
    # and locale; repository selection remains explicit through ``-C``.
    git_env = {
        name: value
        for name in ("LANG", "LC_ALL", "PATH", "SYSTEMROOT")
        if (value := os.environ.get(name)) is not None
    }
    git_env.update(
        {
            "GIT_ALLOW_PROTOCOL": "https",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return git_env


def _git_bytes(root: Path, *arguments: str) -> bytes:
    completed = _git_process(root, *arguments)
    if completed.returncode != 0:
        raise M5OfficialCommandError(
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
        raise M5OfficialCommandError(
            "preflight_failed",
            "Git returned non-UTF-8 provenance",
        ) from exc
    if not value or "\n" in value or "\r" in value:
        raise M5OfficialCommandError(
            "preflight_failed",
            "Git returned invalid provenance",
        )
    return value


def _validated_root(candidate: Path) -> Path:
    try:
        lexical = Path(os.path.abspath(os.fspath(candidate)))
        root = lexical.resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise M5OfficialCommandError(
            "project_root_invalid",
            "the explicit project root does not exist",
        ) from exc
    if (
        lexical != root
        or not root.is_dir()
        or not (root / "pyproject.toml").is_file()
        or not (root / ".gitignore").is_file()
    ):
        raise M5OfficialCommandError(
            "project_root_invalid",
            "the explicit project root is not a canonical EvalSim checkout",
        )
    try:
        git_root = Path(
            _git_text(root, "rev-parse", "--show-toplevel")
        ).resolve(strict=True)
    except OSError as exc:
        raise M5OfficialCommandError(
            "project_root_invalid",
            "the Git worktree root cannot be resolved",
        ) from exc
    if git_root != root:
        raise M5OfficialCommandError(
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
        raise M5OfficialCommandError(
            "argument_error",
            "run_name must be one safe lowercase path component",
        )
    return value


def _require_local_opt_in() -> None:
    if os.environ.get(_LOCAL_OPT_IN) != "1":
        raise M5OfficialCommandError(
            "environment_not_enabled",
            "official local WOMD execution requires the exact opt-in value",
        )


def _assert_running_checkout(root: Path) -> None:
    for module_name, relative_text in _IMPORTED_MODULE_PATHS.items():
        module = importlib.import_module(module_name)
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            raise M5OfficialCommandError(
                "source_binding_failed",
                "an execution module has no filesystem source",
            )
        try:
            actual = Path(raw_path).resolve(strict=True)
            expected = (root / relative_text).resolve(strict=True)
        except OSError as exc:
            raise M5OfficialCommandError(
                "source_binding_failed",
                "an execution module source cannot be resolved",
            ) from exc
        if actual != expected:
            raise M5OfficialCommandError(
                "source_binding_failed",
                "running code does not come from the bound Git checkout",
            )


def _live_main(root: Path) -> str:
    del root
    git_env = _isolated_git_environment()
    git_env["GH_CONFIG_DIR"] = _github_config_dir()
    credential_helper = _github_credential_helper(git_env)
    try:
        completed = subprocess.run(
            (
                "git",
                "-c",
                "credential.interactive=never",
                "-c",
                "credential.helper=",
                "-c",
                (
                    "credential.https://github.com.helper="
                    f"{credential_helper}"
                ),
                "-c",
                "http.followRedirects=false",
                "-c",
                "http.sslVerify=true",
                "-c",
                "protocol.allow=never",
                "-c",
                "protocol.https.allow=always",
                "ls-remote",
                "--exit-code",
                _CANONICAL_REMOTE,
                _CANONICAL_REMOTE_REF,
            ),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            env=git_env,
            cwd=os.sep,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise M5OfficialCommandError(
            "remote_main_mismatch",
            "GitHub main could not be verified live",
        ) from exc
    if completed.returncode != 0:
        raise M5OfficialCommandError(
            "remote_main_mismatch",
            "GitHub main could not be verified live",
        )
    try:
        lines = completed.stdout.decode("ascii", errors="strict").splitlines()
        fields = lines[0].split("\t") if len(lines) == 1 else []
    except UnicodeDecodeError as exc:
        raise M5OfficialCommandError(
            "remote_main_mismatch",
            "GitHub returned invalid main provenance",
        ) from exc
    if (
        len(fields) != 2
        or fields[1] != _CANONICAL_REMOTE_REF
        or _GIT_OBJECT_ID.fullmatch(fields[0]) is None
    ):
        raise M5OfficialCommandError(
            "remote_main_mismatch",
            "GitHub returned an unexpected main ref",
        )
    return fields[0]


def _github_config_dir() -> str:
    """Resolve the one gh credential-store directory exposed to live Git."""

    configured = os.environ.get("GH_CONFIG_DIR")
    if configured:
        candidate = Path(configured)
    elif (xdg_config := os.environ.get("XDG_CONFIG_HOME")):
        candidate = Path(xdg_config) / "gh"
    elif os.name == "nt" and (app_data := os.environ.get("APPDATA")):
        candidate = Path(app_data) / "GitHub CLI"
    else:
        try:
            candidate = Path.home() / ".config" / "gh"
        except RuntimeError as exc:
            raise M5OfficialCommandError(
                "remote_main_mismatch",
                "the fixed GitHub credential store is unavailable",
            ) from exc
    try:
        config_dir = candidate.resolve(strict=True)
        config_text = os.fspath(config_dir)
        if (
            not config_dir.is_dir()
            or any(character in config_text for character in ("\0", "\n", "\r"))
        ):
            raise OSError("unsafe GitHub credential store")
    except OSError as exc:
        raise M5OfficialCommandError(
            "remote_main_mismatch",
            "the fixed GitHub credential store is unavailable",
        ) from exc
    return config_text


def _github_credential_helper(git_env: Mapping[str, str]) -> str:
    """Resolve one fixed GitHub-only helper without importing Git config."""

    search_path = git_env.get("PATH")
    if not search_path:
        raise M5OfficialCommandError(
            "remote_main_mismatch",
            "the fixed GitHub credential helper is unavailable",
        )
    candidate = shutil.which("gh", path=search_path)
    if candidate is None:
        raise M5OfficialCommandError(
            "remote_main_mismatch",
            "the fixed GitHub credential helper is unavailable",
        )
    try:
        executable = Path(candidate).resolve(strict=True)
        executable_text = os.fspath(executable)
        if (
            not executable.is_file()
            or not os.access(executable, os.X_OK)
            or any(character in executable_text for character in ("\0", "\n", "\r"))
        ):
            raise OSError("unsafe GitHub credential helper")
    except OSError as exc:
        raise M5OfficialCommandError(
            "remote_main_mismatch",
            "the fixed GitHub credential helper is unavailable",
        ) from exc
    return f"!{shlex.quote(executable_text)} auth git-credential"


def _git_binding(root: Path) -> _GitBinding:
    remote = _git_text(root, "remote", "get-url", "origin")
    if remote != _CANONICAL_REMOTE:
        raise M5OfficialCommandError(
            "git_remote_invalid",
            "origin is not the canonical EvalSim GitHub remote",
        )
    status = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "-z",
    )
    if status:
        raise M5OfficialCommandError(
            "dirty_worktree",
            "official M5 requires a clean worktree",
        )
    commit = _git_text(root, "rev-parse", "--verify", "HEAD^{commit}")
    tree = _git_text(root, "rev-parse", "--verify", "HEAD^{tree}")
    branch = _git_text(root, "branch", "--show-current")
    upstream = _git_text(
        root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
    )
    upstream_commit = _git_text(
        root,
        "rev-parse",
        "--verify",
        "@{upstream}^{commit}",
    )
    if (
        _GIT_OBJECT_ID.fullmatch(commit) is None
        or _GIT_OBJECT_ID.fullmatch(tree) is None
        or branch != "main"
        or upstream != "origin/main"
        or upstream_commit != commit
    ):
        raise M5OfficialCommandError(
            "unpushed_main",
            "official M5 requires local main at pushed origin/main",
        )
    if _live_main(root) != commit:
        raise M5OfficialCommandError(
            "remote_main_mismatch",
            "live GitHub main differs from local HEAD",
        )
    try:
        paths = official_executable_source_paths(root)
        fingerprint = executable_source_fingerprint(root, paths)
    except Exception as exc:
        raise M5OfficialCommandError(
            "source_binding_failed",
            "official executable-source binding failed",
        ) from exc
    return _GitBinding(
        commit=commit,
        tree=tree,
        executable_paths=paths,
        executable_fingerprint=fingerprint,
    )


def _git_ignored(root: Path, relative: Path) -> bool:
    completed = _git_process(
        root,
        "check-ignore",
        "--quiet",
        "--no-index",
        "--",
        relative.as_posix(),
    )
    return completed.returncode == 0


def _check_output_boundary(
    root: Path,
    run_name: str,
    *,
    expect_exists: bool,
) -> None:
    relative = Path("outputs") / "m5" / run_name
    target = root / relative
    exists = os.path.lexists(target)
    if not expect_exists and exists:
        raise M5OfficialCommandError(
            "output_exists",
            "the requested M5 run already exists",
        )
    if expect_exists:
        try:
            if (
                not exists
                or target.resolve(strict=True) != target
                or not target.is_dir()
            ):
                raise OSError("unsafe result directory")
        except OSError as exc:
            raise M5OfficialCommandError(
                "output_drift",
                "the ignored result directory changed or escaped",
            ) from exc
    if not _git_ignored(root, relative):
        code = "output_drift" if expect_exists else "output_not_ignored"
        raise M5OfficialCommandError(
            code,
            "the requested M5 output is not ignored by Git",
        )
    tracked = _git_bytes(
        root,
        "ls-files",
        "-z",
        "--",
        relative.as_posix(),
    )
    if tracked:
        code = "output_drift" if expect_exists else "output_visible_to_git"
        raise M5OfficialCommandError(
            code,
            "the requested M5 output is visible to Git",
        )
    visible = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored=no",
        "--",
        relative.as_posix(),
    )
    if visible:
        code = "output_drift" if expect_exists else "output_visible_to_git"
        raise M5OfficialCommandError(
            code,
            "the requested M5 output crossed the Git visibility boundary",
        )


def _resolve_shard_paths(root: Path, candidate: Path) -> tuple[Path, ...]:
    from evalsim.sources.waymax_cohort import M4_TFRECORD_SUFFIXES
    from evalsim.sources.waymax_loader import (
        DEFAULT_WOMD_VALIDATION_DIR,
        resolve_m4_validation_shards,
    )

    try:
        lexical = Path(os.path.abspath(os.fspath(candidate)))
        data_dir = lexical.resolve(strict=True)
        expected = (root / DEFAULT_WOMD_VALIDATION_DIR).resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise M5OfficialCommandError(
            "data_directory_invalid",
            "the exact local validation directory does not exist",
        ) from exc
    if lexical != data_dir or data_dir != expected or not data_dir.is_dir():
        raise M5OfficialCommandError(
            "data_directory_invalid",
            "the local validation directory differs from the frozen project path",
        )
    try:
        paths = tuple(resolve_m4_validation_shards(data_dir))
    except Exception as exc:
        raise M5OfficialCommandError(
            "shard_set_invalid",
            "the exact ten local validation shard paths could not be resolved",
        ) from exc
    if (
        len(paths) != 10
        or len(set(paths)) != 10
        or any(
            not path.name.endswith(suffix)
            for path, suffix in zip(paths, M4_TFRECORD_SUFFIXES, strict=True)
        )
    ):
        raise M5OfficialCommandError(
            "shard_set_invalid",
            "the resolved validation paths differ from shards 00000 through 00009",
        )
    for path in paths:
        try:
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or path.resolve(strict=True) != path
                or path.parent != data_dir
            ):
                raise OSError("unsafe shard path")
        except OSError as exc:
            raise M5OfficialCommandError(
                "shard_set_invalid",
                "a required local validation shard path is unsafe",
            ) from exc
    return paths


def _package_version(distribution: str) -> str:
    try:
        value = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise M5OfficialCommandError(
            "preflight_failed",
            "a required official runtime dependency is not installed",
        ) from exc
    if not value:
        raise M5OfficialCommandError(
            "preflight_failed",
            "a required runtime dependency has no version",
        )
    return value


def _simulator_specs() -> dict[str, Any]:
    specs: dict[str, Any] = {}
    for policy in (
        LogReplayPolicy(),
        ConstantVelocityPolicy(),
        IDMPolicy(),
    ):
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
        "parameters": dict(OFFICIAL_WAYMAX_REFERENCE_PARAMETERS),
        "version": WAYMAX_REFERENCE_VERSION,
    }
    return specs


def _runtime_versions() -> dict[str, str]:
    return {
        "flax": _package_version("flax"),
        "jax": _package_version("jax"),
        "jaxlib": _package_version("jaxlib"),
        "numpy": _package_version("numpy"),
        "pyarrow": _package_version("pyarrow"),
        "python": platform.python_version(),
        "tensorflow": _package_version("tensorflow"),
        "waymo_waymax": _package_version("waymo-waymax"),
    }


def _build_provenance(
    binding: _GitBinding,
    cohort: Any,
    parity_selection: Any,
) -> M5RunProvenance:
    evidence = cohort.evidence
    receipt = parity_selection.receipt
    return M5RunProvenance(
        m4_manifest_sha256=evidence.manifest_sha256,
        m4_execution_provenance_sha256=(
            evidence.execution_provenance_sha256
        ),
        selected_order_version=evidence.selected_order_version,
        selected_order_fingerprint_sha256=(
            evidence.selected_order_fingerprint_sha256
        ),
        executable_source_fingerprint_sha256=(
            binding.executable_fingerprint
        ),
        executable_source_paths=binding.executable_paths,
        git_commit=binding.commit,
        git_tree=binding.tree,
        simulator_specs=_simulator_specs(),
        runtime_versions=_runtime_versions(),
        m4_aggregate_summary_sha256=evidence.aggregate_summary_sha256,
        m4_reuse_schema_version=M5_M4_REUSE_SCHEMA_VERSION,
        m4_integrity_assumption=M5_M4_INTEGRITY_ASSUMPTION,
        m4_receipt_verified=cohort.receipt_verified,
        parity_order_version=receipt.order_version,
        parity_order_fingerprint_sha256=(
            receipt.ordered_membership_sha256
        ),
    )


def _elapsed_ms(start_ns: int, end_ns: int) -> int:
    if (
        isinstance(start_ns, bool)
        or isinstance(end_ns, bool)
        or not isinstance(start_ns, int)
        or not isinstance(end_ns, int)
        or end_ns < start_ns
    ):
        raise M5OfficialCommandError(
            "result_contract_failed",
            "the monotonic stage clock moved backwards",
        )
    return (end_ns - start_ns) // 1_000_000


def _timed(
    stage: str,
    callback: Callable[[], _T],
    elapsed: dict[str, int],
) -> _T:
    if stage not in M5_OFFICIAL_STAGE_NAMES or stage in elapsed:
        raise M5OfficialCommandError(
            "result_contract_failed",
            "a stage label is unknown or repeated",
        )
    start = time.monotonic_ns()
    value = callback()
    elapsed[stage] = _elapsed_ms(start, time.monotonic_ns())
    return value


def _preflight(args: argparse.Namespace) -> _OfficialContext:
    start = time.monotonic_ns()
    _require_local_opt_in()
    root = _validated_root(args.project_root)
    run_name = _validated_run_name(args.run_name)
    _assert_running_checkout(root)
    binding = _git_binding(root)
    _check_output_boundary(root, run_name, expect_exists=False)
    shard_paths = _resolve_shard_paths(root, args.data_dir)

    from evalsim.evaluation.m5_waymax import select_m5_parity_members
    from evalsim.sources.m5_m4_reuse import verify_accepted_m4_run

    try:
        cohort = verify_accepted_m4_run(root, args.m4_run_dir)
        parity_selection = select_m5_parity_members(cohort.members)
    except Exception as exc:
        raise M5OfficialCommandError(
            "m4_preflight_failed",
            "the explicit accepted M4 run failed exact reuse validation",
        ) from exc
    provenance = _build_provenance(binding, cohort, parity_selection)
    return _OfficialContext(
        root=root,
        data_dir=shard_paths[0].parent,
        shard_paths=shard_paths,
        run_name=run_name,
        cohort=cohort,
        parity_selection=parity_selection,
        provenance=provenance,
        git_binding=binding,
        preflight_ms=_elapsed_ms(start, time.monotonic_ns()),
    )


def _policy_executions(
    case: Any,
    policies: Sequence[Any],
    executor: Any,
) -> tuple[Any, ...]:
    from evalsim.evaluation.m5 import ExecutionRollout, ExecutionSpec

    executions = []
    for policy in policies:
        metadata = policy.metadata()
        executions.append(
            ExecutionRollout(
                spec=ExecutionSpec(
                    name=metadata.name,
                    version=metadata.version,
                    role="policy",
                    seed=M5_OFFICIAL_POLICY_SEED,
                ),
                rollout=executor.execute(
                    case,
                    policy,
                    M5_OFFICIAL_POLICY_SEED,
                ),
            )
        )
    return tuple(executions)


def _visit_and_evaluate(
    context: _OfficialContext,
    store: M5ResultStore,
) -> _CohortState:
    from evalsim.evaluation.m5 import (
        EvaluationCase,
        M5ScorecardAccumulator,
        NumpyPolicyExecutor,
        canonical_m5_policies,
        evaluate_m5_case,
    )
    from evalsim.evaluation.m5_waymax import (
        M5StreamingParityAccumulator,
        WaymaxExactLogReferenceExecutor,
    )
    from evalsim.sources.m5_m4_reuse import visit_accepted_m4_cohort

    store.write_parity_order_receipt(context.parity_selection.receipt)
    scorecards = M5ScorecardAccumulator()
    parity = M5StreamingParityAccumulator(context.parity_selection)
    parity_indices = (
        context.parity_selection.parity_index_by_cohort_index
    )
    policies = canonical_m5_policies()
    executor = NumpyPolicyExecutor()
    reference_executor = WaymaxExactLogReferenceExecutor()
    visited: set[int] = set()

    def visitor(member: Any) -> None:
        if member.cohort_index in visited:
            raise M5OfficialCommandError(
                "cohort_evaluation_failed",
                "the accepted cohort emitted a duplicate opaque index",
            )
        case = EvaluationCase(
            cohort_index=member.cohort_index,
            scenario=member.scenario,
            reference_payload=member.record,
        )
        policy_runs = _policy_executions(case, policies, executor)
        reference_run = reference_executor.execute(
            case,
            seed=M5_OFFICIAL_POLICY_SEED,
        )
        evaluated = evaluate_m5_case(
            case,
            (*policy_runs, reference_run),
        )
        if (
            len(evaluated.metric_rows)
            != M5_OFFICIAL_METRIC_ROWS_PER_CASE
            or len(evaluated.slice_rows)
            != M5_OFFICIAL_SLICE_ROWS_PER_CASE
        ):
            raise M5OfficialCommandError(
                "result_contract_failed",
                "one cohort member produced an incomplete M5 matrix",
            )
        if member.cohort_index in parity_indices:
            parity.add_case(case, policy_runs)
        store.write_metric_results_part(
            evaluated.metric_rows,
            part_index=member.cohort_index,
        )
        scorecards.add_case(evaluated)
        visited.add(member.cohort_index)

    visit_accepted_m4_cohort(
        context.cohort,
        context.data_dir,
        visitor,
    )
    if visited != set(range(M5_OFFICIAL_SCENARIO_COUNT)):
        raise M5OfficialCommandError(
            "cohort_evaluation_failed",
            "the accepted cohort did not cover opaque indices 0 through 127",
        )
    return _CohortState(
        scorecard_accumulator=scorecards,
        parity_rows=parity.finalize(),
    )


def _finalize_statistics(state: _CohortState) -> Any:
    summary = state.scorecard_accumulator.finalize(
        expected_case_count=M5_OFFICIAL_SCENARIO_COUNT,
    )
    if (
        summary.case_count != M5_OFFICIAL_SCENARIO_COUNT
        or len(summary.slice_rows)
        != OFFICIAL_M5_ROW_COUNTS.slice_membership
        or len(summary.scorecard_results)
        != OFFICIAL_M5_ROW_COUNTS.scorecards
    ):
        raise M5OfficialCommandError(
            "result_contract_failed",
            "the compact streaming statistics domain is incomplete",
        )
    return summary


def _publish_aggregates(
    store: M5ResultStore,
    state: _CohortState,
    summary: Any,
) -> None:
    if len(state.parity_rows) != OFFICIAL_M5_ROW_COUNTS.waymax_parity_summary:
        raise M5OfficialCommandError(
            "result_contract_failed",
            "the native parity summary domain is incomplete",
        )
    scorecard_rows = tuple(
        scorecard_row_from_result(result)
        for result in summary.scorecard_results
    )
    if len(scorecard_rows) != OFFICIAL_M5_ROW_COUNTS.scorecards:
        raise M5OfficialCommandError(
            "result_contract_failed",
            "the scorecard adapter returned an incomplete matrix",
        )
    store.write_slice_membership(summary.slice_rows)
    store.write_scorecards(scorecard_rows)
    store.write_waymax_parity_summary(
        row.to_dict() for row in state.parity_rows
    )
    store.write_determinism_receipt(
        M5DeterminismReceipt(
            metric_pass_1_sha256=summary.metric_pass_1_sha256,
            metric_pass_2_sha256=summary.metric_pass_2_sha256,
            statistics_pass_1_sha256=summary.statistics_pass_1_sha256,
            statistics_pass_2_sha256=summary.statistics_pass_2_sha256,
        )
    )
    store.write_human_readable_scorecard()


def _reverify_context(context: _OfficialContext, store: M5ResultStore) -> None:
    from evalsim.sources.m5_m4_reuse import reverify_accepted_m4_run

    try:
        binding = _git_binding(context.root)
    except M5OfficialCommandError as exc:
        raise M5OfficialCommandError(
            "source_drift",
            "the pushed source binding changed during execution",
        ) from exc
    if binding != context.git_binding:
        raise M5OfficialCommandError(
            "source_drift",
            "the executable-source snapshot changed during execution",
        )
    _assert_running_checkout(context.root)
    _check_output_boundary(
        context.root,
        context.run_name,
        expect_exists=True,
    )
    if store.run_path != context.root / store.project_relative_path:
        raise M5OfficialCommandError(
            "output_drift",
            "the result-store path changed during execution",
        )
    try:
        current_paths = _resolve_shard_paths(context.root, context.data_dir)
        if current_paths != context.shard_paths:
            raise ValueError("shard path order changed")
        reverify_accepted_m4_run(context.cohort)
    except Exception as exc:
        raise M5OfficialCommandError(
            "m4_drift",
            "the accepted M4 trust root or shard-path binding changed",
        ) from exc


def _execute_pending(
    context: _OfficialContext,
    store: M5ResultStore,
    holder: _RunHolder,
) -> _PendingRun:
    elapsed: dict[str, int] = {"preflight": context.preflight_ms}
    try:
        cohort_state = _timed(
            "cohort_evaluation",
            lambda: _visit_and_evaluate(context, store),
            elapsed,
        )
    except M5OfficialCommandError:
        raise
    except Exception as exc:
        raise M5OfficialCommandError(
            "cohort_evaluation_failed",
            "the accepted M4 cohort failed official M5 evaluation",
        ) from exc

    try:
        summary = _timed(
            "statistics",
            lambda: _finalize_statistics(cohort_state),
            elapsed,
        )
    except M5OfficialCommandError:
        raise
    except Exception as exc:
        raise M5OfficialCommandError(
            "statistics_failed",
            "the fixed paired-statistics matrix failed",
        ) from exc

    try:
        _timed(
            "artifact_publication",
            lambda: _publish_aggregates(store, cohort_state, summary),
            elapsed,
        )
    except M5OfficialCommandError:
        raise
    except Exception as exc:
        raise M5OfficialCommandError(
            "artifact_publication_failed",
            "the immutable aggregate artifacts could not be written",
        ) from exc

    start = time.monotonic_ns()
    _reverify_context(context, store)
    try:
        prepared = store.prepare_finalization(
            provenance=context.provenance,
        )
        holder.prepared = prepared
    except M5OfficialCommandError:
        raise
    except Exception as exc:
        raise M5OfficialCommandError(
            "finalization_failed",
            "the official M5 store failed verified preparation",
        ) from exc
    try:
        independently_verified = verify_prepared_m5_result_store(
            context.root,
            context.run_name,
        )
        if independently_verified.run_path != store.run_path:
            raise M5OfficialCommandError(
                "verification_failed",
                "prepared verification resolved a different result path",
            )
    except M5OfficialCommandError:
        raise
    except Exception as exc:
        raise M5OfficialCommandError(
            "verification_failed",
            "the prepared M5 store failed independent verification",
        ) from exc
    # The independent artifact/source scan above can be lengthy.  Recheck the
    # mutable local trust roots once more before creating COMMITTED.
    _reverify_context(context, store)
    elapsed["finalization"] = _elapsed_ms(
        start,
        time.monotonic_ns(),
    )

    provisional = _RunResult(
        result_relative=store.project_relative_path,
        row_counts=store.row_counts,
        elapsed_ms=elapsed,
    )
    payload = _success_output(provisional)
    return _PendingRun(
        prepared=prepared,
        result=_RunResult(
            result_relative=provisional.result_relative,
            row_counts=provisional.row_counts,
            elapsed_ms=provisional.elapsed_ms,
            success_payload=payload,
        ),
    )


def _preflight_create_execute(
    args: argparse.Namespace,
    holder: _RunHolder,
) -> _PendingRun:
    context = _preflight(args)
    holder.context = context
    try:
        store = M5ResultStore.create(context.root, context.run_name)
    except FileExistsError as exc:
        raise M5OfficialCommandError(
            "output_exists",
            "the requested M5 run already exists",
        ) from exc
    except Exception as exc:
        raise M5OfficialCommandError(
            "preflight_failed",
            "the ignored official M5 run could not be created",
        ) from exc
    holder.store = store
    return _execute_pending(context, store, holder)


def _failure_code(exc: BaseException) -> str:
    if type(exc) is M5OfficialCommandError and exc.code in _TRUSTED_CODES:
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
    return (
        b"=== captured stdout/stderr ===\n"
        + transcript[:_MAX_DIAGNOSTIC_BYTES]
        + b"\n=== exception ===\n"
        + formatted[:_MAX_DIAGNOSTIC_BYTES]
    )[:_MAX_DIAGNOSTIC_BYTES]


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
        encoded = os.read(descriptor, 4097)
        payload = json.loads(encoded.decode("ascii", errors="strict"))
        if (
            len(encoded) != metadata.st_size
            or not isinstance(payload, dict)
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
    store: M5ResultStore | None,
    primary: BaseException,
    transcript: bytes,
    *,
    prepared: PreparedM5Finalization | None,
) -> tuple[Path | None, str, str]:
    requested = _failure_code(primary)
    if store is None:
        return None, requested, "rejected"
    if os.path.lexists(store.run_path / "SUCCESS"):
        return None, requested, "failure"
    if not os.path.lexists(store.run_path / "FAILURE.json"):
        try:
            if prepared is None:
                store.fail(requested)
            else:
                store.abort_finalization(prepared, requested)
        except Exception:
            pass
    persisted = _read_canonical_failure_record(store)
    if persisted is None:
        return None, "unexpected_failure", "failure"
    # Establish the canonical terminal bit first.  A crash while writing optional
    # diagnostics must never strand FINALIZING/COMMITTED with only a log file.
    try:
        _write_failure_diagnostics(store.run_path, primary, transcript)
    except OSError:
        pass
    relative, reason = persisted
    return relative, reason, "failure"


def _arm_verified_success(
    store: M5ResultStore,
    pending: _PendingRun,
    context: _OfficialContext,
) -> None:
    """Create and verify the abortable in-process pre-success checkpoint.

    ``COMMITTED`` is deliberately not accepted success.  Every mutable external
    trust root is checked after it exists, and the complete result store is the
    final lengthy operation before the terminal callback is allowed to promote
    the exact ``SUCCESS`` marker.
    """

    try:
        store.mark_committed_for_verification(pending.prepared)
        _reverify_context(context, store)
    except M5OfficialCommandError:
        raise
    except Exception as exc:
        raise M5OfficialCommandError(
            "finalization_failed",
            "the abortable pre-success checkpoint could not be created",
        ) from exc
    try:
        verified = verify_committed_m5_result_store(
            context.root,
            context.run_name,
        )
        if verified.run_path != store.run_path:
            raise M5OfficialCommandError(
                "verification_failed",
                "committed verification resolved a different result path",
            )
    except M5OfficialCommandError:
        raise
    except Exception as exc:
        raise M5OfficialCommandError(
            "verification_failed",
            "the abortable pre-success checkpoint failed verification",
        ) from exc


def _promote_success(
    store: M5ResultStore,
    pending: _PendingRun,
) -> None:
    """Perform only the terminal exact-marker promotion."""

    try:
        store.commit_finalization(pending.prepared)
    except Exception as exc:
        raise M5OfficialCommandError(
            "finalization_failed",
            "the verified M5 store failed SUCCESS promotion",
        ) from exc


def _execute_and_arm_success(
    args: argparse.Namespace,
    holder: _RunHolder,
) -> _PendingRun:
    """Execute, validate, and arm one exact prepared capability while captured."""

    pending = _preflight_create_execute(args, holder)
    if (
        holder.store is None
        or holder.context is None
        or not isinstance(pending, _PendingRun)
        or pending.prepared is not holder.prepared
        or pending.result.success_payload is None
        or type(pending.result.success_payload) is not bytes
    ):
        raise M5OfficialCommandError(
            "result_contract_failed",
            "the captured run omitted its prepared success contract",
        )
    _arm_verified_success(holder.store, pending, holder.context)
    return pending


def run_official(args: argparse.Namespace) -> _RunResult:
    """Run one new official M5 directory under the strict local boundary."""

    holder = _RunHolder()
    try:
        captured = capture_terminal(
            lambda: _execute_and_arm_success(args, holder),
            terminal_commit=lambda pending: _promote_success(
                holder.store,  # type: ignore[arg-type]
                pending,
            ),
            seal_terminal=True,
        )
    except TerminalizedFailure as exc:
        relative, code, status = _persist_failure(
            holder.store,
            exc.primary,
            exc.transcript,
            prepared=holder.prepared,
        )
        raise _CommandFailure(
            primary=exc.primary,
            failure_relative=relative,
            reason_code=code,
            status=status,
            terminal_status=exc.terminal_status,
        ) from None
    except BaseException as exc:
        relative, code, status = _persist_failure(
            holder.store,
            exc,
            b"",
            prepared=holder.prepared,
        )
        raise _CommandFailure(
            primary=exc,
            failure_relative=relative,
            reason_code=code,
            status=status,
        ) from None

    # The callback performed the complete capability validation before arming
    # success.  Repeating it here would introduce a contradictory post-SUCCESS
    # failure path.
    result = captured.value.result
    object.__setattr__(
        result,
        "terminal_status",
        captured.terminal_status,
    )
    return result


def _emit_failure(failure: _CommandFailure) -> None:
    payload = _failure_output(
        failure.reason_code,
        failure.failure_relative,
        status=failure.status,
    )
    terminal = failure.terminal_status
    if terminal is None:
        try:
            sys.stderr.write(payload.decode("ascii"))
            sys.stderr.flush()
        except (OSError, UnicodeError):
            pass
        return
    try:
        write_all(terminal.stderr_fd, payload)
    except OSError:
        pass
    finally:
        terminal.close_best_effort()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = run_official(args)
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
    "M5_OFFICIAL_METRIC_ROWS_PER_CASE",
    "M5_OFFICIAL_POLICY_SEED",
    "M5_OFFICIAL_PROFILE",
    "M5_OFFICIAL_SCENARIO_COUNT",
    "M5_OFFICIAL_SLICE_ROWS_PER_CASE",
    "M5_OFFICIAL_STAGE_NAMES",
    "M5_OFFICIAL_STATUS_SCHEMA_VERSION",
    "M5OfficialCommandError",
    "main",
    "run_official",
]
