"""Stdlib-only cold start for the official M6 command.

Run exactly from the repository root as:

    .venv/bin/python -I -S -B evalsim/cli/m6_bootstrap.py [arguments]

This tracked bootstrap is the explicit trust entrypoint. Its self-checks assume
these executing bytes are honest; defending against arbitrary replacement of this
entrypoint requires an external immutable launcher or hash verifier.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Callable


_DIRECT_COMMAND = (
    ".venv/bin/python",
    "-I",
    "-S",
    "-B",
    "evalsim/cli/m6_bootstrap.py",
)
_STATUS_SCHEMA_VERSION = "m6-cli-status-2.0.0"
_MODES = frozenset({"eligibility_only", "compute_pilot", "official"})
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_RUN_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_M6_REVIEW_COUNT_MAX = 2**31 - 1
_REASON_CODES = frozenset(
    {
        "accepted_m4_snapshot_invalid",
        "approved_commit_mismatch",
        "argument_error",
        "data_directory_invalid",
        "dirty_worktree",
        "environment_not_enabled",
        "execution_failed",
        "execution_not_available",
        "git_remote_invalid",
        "project_root_invalid",
        "remote_main_mismatch",
        "review_rejected",
        "result_contract_failed",
        "result_store_failed",
        "runtime_mismatch",
        "shard_set_invalid",
        "source_binding_failed",
        "terminal_capture_failed",
        "terminal_output_detected",
        "unexpected_failure",
        "unpushed_main",
        "verification_failed",
    }
)
_MAX_STATUS_BYTES = 64 * 1024
_GIT_BINARY = Path("/usr/bin/git")
_LOCAL_OPT_IN = "EVALSIM_RUN_WAYMO_LOCAL"
_CANONICAL_REMOTE = "https://github.com/6ickomod3/evalsim.git"
_CANONICAL_REMOTE_REF = "refs/heads/main"
_APPROVED_IMPLEMENTATION_REF = "refs/tags/m6-approved-v1"
_GIT_OBJECT_ID = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IMPORT_EXECUTABLE_SUFFIXES = (
    ".py",
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".pyd",
    ".dll",
)
_SOURCE_REQUIRED_FILES = (
    ".gitignore",
    ".python-version",
    "AGENTS.md",
    "NOTICE.md",
    "docs/plans/2026-07-29-m6-counterfactual-reactivity.md",
    "docs/plans/2026-07-29-m6-data-free-implementation-checkpoint.md",
    "pyproject.toml",
    "uv.lock",
)
_SOURCE_PATHSPECS = (
    "evalsim",
    ":(glob)tests/test_m6_*.py",
    *_SOURCE_REQUIRED_FILES,
)


class _BootstrapRejected(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = (
            reason_code if reason_code in _REASON_CODES else "unexpected_failure"
        )
        super().__init__(self.reason_code)


def _canonical_json(payload: dict[str, object]) -> bytes:
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


def _rejection(reason_code: str) -> bytes:
    if reason_code not in _REASON_CODES:
        reason_code = "unexpected_failure"
    return _canonical_json(
        {
            "reason_code": reason_code,
            "schema_version": _STATUS_SCHEMA_VERSION,
            "status": "rejected",
        }
    )


def _write_all(descriptor: int, payload: bytes) -> bool:
    remaining = memoryview(payload)
    try:
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                return False
            remaining = remaining[written:]
    except OSError:
        return False
    return True


def _integer_mapping(value: object) -> bool:
    return (
        type(value) is dict
        and bool(value)
        and all(
            isinstance(key, str)
            and _SAFE_COMPONENT.fullmatch(key) is not None
            and type(item) is int
            and item >= 0
            for key, item in value.items()
        )
    )


def _valid_status(payload: object, error: object, return_code: object) -> bool:
    if (
        type(payload) is not bytes
        or not payload
        or len(payload) > _MAX_STATUS_BYTES
        or type(error) is not bool
        or type(return_code) is not int
    ):
        return False
    try:
        text = payload.decode("ascii", errors="strict")
        value = json.loads(text)
    except (UnicodeError, json.JSONDecodeError):
        return False
    if (
        type(value) is not dict
        or _canonical_json(value) != payload
        or value.get("schema_version") != _STATUS_SCHEMA_VERSION
    ):
        return False
    status = value.get("status")
    keys = set(value)
    if status == "success":
        expected_keys = {
            "aggregate_counts",
            "mode",
            "profile",
            "result_path",
            "schema_version",
            "stage_durations_ms",
            "status",
        }
        result_path = value.get("result_path")
        return (
            not error
            and return_code == 0
            and keys == expected_keys
            and value.get("mode") in _MODES
            and value.get("profile") == "official_m6"
            and isinstance(result_path, str)
            and re.fullmatch(r"outputs/m6/[a-z0-9][a-z0-9._-]{0,127}", result_path)
            is not None
            and _integer_mapping(value.get("aggregate_counts"))
            and _integer_mapping(value.get("stage_durations_ms"))
        )
    if status == "awaiting_review":
        expected_keys = {
            "mode",
            "profile",
            "result_path",
            "schema_version",
            "status",
        }
        result_path = value.get("result_path")
        return (
            not error
            and return_code == 0
            and keys == expected_keys
            and value.get("mode") == "official"
            and value.get("profile") == "official_m6"
            and isinstance(result_path, str)
            and re.fullmatch(
                r"outputs/m6/[a-z0-9][a-z0-9._-]{0,127}",
                result_path,
            )
            is not None
        )
    if status not in {"rejected", "failure"}:
        return False
    if not error or return_code == 0 or value.get("reason_code") not in _REASON_CODES:
        return False
    expected_keys = {"reason_code", "schema_version", "status"}
    marker = value.get("failure_marker")
    if status == "failure" and marker is not None:
        expected_keys.add("failure_marker")
        if (
            not isinstance(marker, str)
            or re.fullmatch(
                r"outputs/m6/[a-z0-9][a-z0-9._-]{0,127}/TERMINAL_FAILURE",
                marker,
            )
            is None
        ):
            return False
    return keys == expected_keys


def _parse_argument_surface(root: Path) -> dict[str, str]:
    arguments = tuple(sys.argv[1:])
    review_prefixes = (
        "architecture",
        "methods-statistics",
        "privacy-claim",
    )
    review_options = {
        f"--{prefix}-{suffix}"
        for prefix in review_prefixes
        for suffix in (
            "decision",
            "p1-count",
            "p2-count",
            "p3-count",
        )
    }
    base_options = {
        "--project-root",
        "--data-dir",
        "--m4-run-dir",
        "--run-name",
        "--eligibility-run-name",
        "--pilot-run-name",
        "--mode",
    }
    option_names = base_options | {"--action"} | review_options
    if not arguments or len(arguments) % 2:
        raise _BootstrapRejected("argument_error")
    values: dict[str, str] = {}
    for index in range(0, len(arguments), 2):
        option, value = arguments[index : index + 2]
        if (
            option not in option_names
            or option in values
            or not isinstance(value, str)
            or not value
            or value.startswith("--")
        ):
            raise _BootstrapRejected("argument_error")
        values[option] = value
    required = {
        "--project-root",
        "--data-dir",
        "--m4-run-dir",
        "--run-name",
        "--mode",
    }
    if not required.issubset(values):
        raise _BootstrapRejected("argument_error")
    mode = values["--mode"]
    if mode not in _MODES:
        raise _BootstrapRejected("argument_error")
    predecessor_domain = {
        "eligibility_only": frozenset(),
        "compute_pilot": frozenset({"--eligibility-run-name"}),
        "official": frozenset(
            {"--eligibility-run-name", "--pilot-run-name"}
        ),
    }[mode]
    action = values.get("--action")
    if action is None:
        expected_options = required | set(predecessor_domain)
    elif action == "finalize-review" and mode == "official":
        expected_options = (
            required
            | set(predecessor_domain)
            | {"--action"}
            | review_options
        )
        for prefix in review_prefixes:
            decision = values.get(f"--{prefix}-decision")
            for priority in ("p1", "p2", "p3"):
                raw = values.get(f"--{prefix}-{priority}-count", "")
                if re.fullmatch(r"(?:0|[1-9][0-9]{0,18})", raw) is None:
                    raise _BootstrapRejected("argument_error")
                parsed = int(raw)
                if parsed > _M6_REVIEW_COUNT_MAX:
                    raise _BootstrapRejected("argument_error")
            if decision not in {"accept", "reject"}:
                raise _BootstrapRejected("argument_error")
    else:
        raise _BootstrapRejected("argument_error")
    if set(values) != expected_options:
        raise _BootstrapRejected("argument_error")
    names = [values["--run-name"]]
    names.extend(values[name] for name in sorted(predecessor_domain))
    if (
        any(_RUN_NAME.fullmatch(name) is None or name in {".", ".."} for name in names)
        or len(names) != len(set(names))
    ):
        raise _BootstrapRejected("argument_error")
    try:
        requested_root = Path(
            os.path.abspath(values["--project-root"])
        ).resolve(strict=True)
    except OSError as exc:
        raise _BootstrapRejected("project_root_invalid") from exc
    if requested_root != root:
        raise _BootstrapRejected("project_root_invalid")
    return values


def _trusted_git_binary() -> str:
    try:
        metadata = _GIT_BINARY.lstat()
        resolved = _GIT_BINARY.resolve(strict=True)
    except OSError as exc:
        raise _BootstrapRejected("project_root_invalid") from exc
    if (
        resolved != _GIT_BINARY
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise _BootstrapRejected("project_root_invalid")
    return os.fspath(_GIT_BINARY)


def _git_environment() -> dict[str, str]:
    return {
        "GIT_ALLOW_PROTOCOL": "https",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


def _git_prefix() -> tuple[str, ...]:
    return (
        _trusted_git_binary(),
        "--no-replace-objects",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "core.preloadIndex=false",
    )


def _git_process(
    root: Path,
    *arguments: str,
    timeout: int | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    command = (*_git_prefix(), "-C", os.fspath(root), *arguments)
    try:
        return subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=_git_environment(),
            cwd=os.fspath(root if cwd is None else cwd),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise _BootstrapRejected("project_root_invalid") from exc


def _git_bytes(root: Path, *arguments: str) -> bytes:
    completed = _git_process(root, *arguments)
    if completed.returncode != 0:
        raise _BootstrapRejected("project_root_invalid")
    return completed.stdout


def _git_text(root: Path, *arguments: str) -> str:
    try:
        value = _git_bytes(root, *arguments).decode(
            "utf-8", errors="strict"
        ).strip()
    except UnicodeDecodeError as exc:
        raise _BootstrapRejected("project_root_invalid") from exc
    if not value or "\n" in value or "\r" in value:
        raise _BootstrapRejected("project_root_invalid")
    return value


def _live_remote_ref(root: Path, ref: str, reason_code: str) -> str:
    del root
    if ref not in {_CANONICAL_REMOTE_REF, _APPROVED_IMPLEMENTATION_REF}:
        raise _BootstrapRejected("source_binding_failed")
    try:
        completed = subprocess.run(
            (
                *_git_prefix(),
                "-c",
                "credential.interactive=never",
                "-c",
                "credential.helper=",
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
                ref,
            ),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            env=_git_environment(),
            cwd=os.sep,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise _BootstrapRejected(reason_code) from exc
    try:
        lines = completed.stdout.decode("ascii", errors="strict").splitlines()
        fields = (
            lines[0].split("\t")
            if completed.returncode == 0 and len(lines) == 1
            else ()
        )
    except UnicodeDecodeError as exc:
        raise _BootstrapRejected(reason_code) from exc
    if (
        len(fields) != 2
        or fields[1] != ref
        or _GIT_OBJECT_ID.fullmatch(fields[0]) is None
    ):
        raise _BootstrapRejected(reason_code)
    return fields[0]


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _is_import_executable_name(name: str) -> bool:
    return name.casefold().endswith(_IMPORT_EXECUTABLE_SUFFIXES)


def _reject_unapproved_evalsim_import_artifacts(
    root: Path,
    approved_paths: tuple[str, ...],
) -> None:
    """Reject ignored/untracked code that Python could load from EvalSim."""

    allowed = frozenset(approved_paths)
    package_root = root / "evalsim"

    def visit(directory: Path) -> None:
        before = directory.lstat()
        if (
            not stat.S_ISDIR(before.st_mode)
            or directory.resolve(strict=True) != directory
        ):
            raise OSError("noncanonical package directory")
        with os.scandir(directory) as iterator:
            names = tuple(sorted(entry.name for entry in iterator))
        for name in names:
            path = directory / name
            metadata = path.lstat()
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                raise OSError("linked package node")
            if stat.S_ISDIR(metadata.st_mode):
                # The isolated bootstrap redirects bytecode lookup to its empty
                # pycache prefix, so ordinary source-tree caches cannot shadow.
                if name != "__pycache__":
                    visit(path)
            elif _is_import_executable_name(name) and (
                not stat.S_ISREG(metadata.st_mode)
                or relative not in allowed
            ):
                raise OSError("unapproved import executable")
        after = directory.lstat()
        if _file_identity(before) != _file_identity(after):
            raise OSError("package directory changed during scan")

    try:
        root_before = root.lstat()
        if (
            not stat.S_ISDIR(root_before.st_mode)
            or root.resolve(strict=True) != root
        ):
            raise OSError("noncanonical root")
        with os.scandir(root) as iterator:
            root_names = tuple(sorted(entry.name for entry in iterator))
        if any(
            name != "evalsim"
            and name.casefold().startswith("evalsim.")
            and _is_import_executable_name(name)
            for name in root_names
        ):
            raise OSError("root-level EvalSim import collision")
        visit(package_root)
        root_after = root.lstat()
        if _file_identity(root_before) != _file_identity(root_after):
            raise OSError("root changed during import-artifact scan")
    except (OSError, ValueError) as exc:
        raise _BootstrapRejected("source_binding_failed") from exc


def _guarded_source_sha256(root: Path, relative_text: str) -> str:
    relative = Path(relative_text)
    path = root.joinpath(*relative.parts)
    descriptor = -1
    try:
        before = path.lstat()
        if (
            relative.is_absolute()
            or relative.as_posix() != relative_text
            or ".." in relative.parts
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or path.resolve(strict=True) != path
            or root not in path.parents
        ):
            raise OSError("unsafe source")
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after_read = os.fstat(descriptor)
        after_path = path.lstat()
        identities = {
            _file_identity(before),
            _file_identity(opened),
            _file_identity(after_read),
            _file_identity(after_path),
        }
        if len(identities) != 1:
            raise OSError("source changed")
        return digest.hexdigest()
    except OSError as exc:
        raise _BootstrapRejected("source_binding_failed") from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _source_path_selected(relative_text: str) -> bool:
    return (
        relative_text in _SOURCE_REQUIRED_FILES
        or relative_text.startswith("evalsim/")
        or (
            relative_text.startswith("tests/test_m6_")
            and relative_text.endswith(".py")
        )
    )


def _approved_source_tree(root: Path, commit: str) -> dict[str, str]:
    encoded = _git_bytes(
        root,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        commit,
    )
    entries: dict[str, str] = {}
    try:
        for record in (part for part in encoded.split(b"\0") if part):
            metadata, raw_path = record.split(b"\t", 1)
            mode, kind, raw_object = metadata.split(b" ", 2)
            relative_text = raw_path.decode("utf-8", errors="strict")
            if not _source_path_selected(relative_text):
                continue
            object_id = raw_object.decode("ascii", errors="strict")
            if (
                mode not in {b"100644", b"100755"}
                or kind != b"blob"
                or _GIT_OBJECT_ID.fullmatch(object_id) is None
                or relative_text in entries
            ):
                raise ValueError("invalid source entry")
            entries[relative_text] = object_id
    except (UnicodeDecodeError, ValueError) as exc:
        raise _BootstrapRejected("source_binding_failed") from exc
    paths = tuple(entries)
    if (
        not paths
        or paths != tuple(sorted(set(paths)))
        or not set(_SOURCE_REQUIRED_FILES).issubset(paths)
        or not any(path.startswith("evalsim/") for path in paths)
        or not any(path.startswith("tests/test_m6_") for path in paths)
        or _DIRECT_COMMAND[-1] not in paths
    ):
        raise _BootstrapRejected("source_binding_failed")
    return entries


def _preauthenticate_repository(
    root: Path,
) -> tuple[object, ...]:
    if os.environ.get(_LOCAL_OPT_IN) != "1":
        raise _BootstrapRejected("environment_not_enabled")
    if _git_text(root, "remote", "get-url", "origin") != _CANONICAL_REMOTE:
        raise _BootstrapRejected("git_remote_invalid")
    if _git_bytes(
        root,
        "for-each-ref",
        "--format=%(refname)",
        "refs/replace/",
    ):
        raise _BootstrapRejected("source_binding_failed")
    all_index_records = tuple(
        record
        for record in _git_bytes(root, "ls-files", "-v", "-z").split(b"\0")
        if record
    )
    if not all_index_records or any(
        not record.startswith(b"H ") for record in all_index_records
    ):
        raise _BootstrapRejected("source_binding_failed")
    if _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "-z",
    ):
        raise _BootstrapRejected("dirty_worktree")
    commit = _git_text(root, "rev-parse", "--verify", "HEAD^{commit}")
    tree = _git_text(root, "rev-parse", "--verify", "HEAD^{tree}")
    branch = _git_text(root, "branch", "--show-current")
    local_main = _git_text(
        root, "rev-parse", "--verify", "refs/heads/main^{commit}"
    )
    origin_main = _git_text(
        root,
        "rev-parse",
        "--verify",
        "refs/remotes/origin/main^{commit}",
    )
    if (
        branch != "main"
        or commit != local_main
        or commit != origin_main
        or _GIT_OBJECT_ID.fullmatch(commit) is None
        or _GIT_OBJECT_ID.fullmatch(tree) is None
    ):
        raise _BootstrapRejected("unpushed_main")
    try:
        approval_object = _git_text(
            root, "rev-parse", "--verify", _APPROVED_IMPLEMENTATION_REF
        )
        approval_type = _git_text(
            root, "cat-file", "-t", _APPROVED_IMPLEMENTATION_REF
        )
    except _BootstrapRejected as exc:
        raise _BootstrapRejected("approved_commit_mismatch") from exc
    if approval_type != "commit" or approval_object != commit:
        raise _BootstrapRejected("approved_commit_mismatch")
    if _live_remote_ref(
        root, _CANONICAL_REMOTE_REF, "remote_main_mismatch"
    ) != commit:
        raise _BootstrapRejected("remote_main_mismatch")
    if _live_remote_ref(
        root,
        _APPROVED_IMPLEMENTATION_REF,
        "approved_commit_mismatch",
    ) != commit:
        raise _BootstrapRejected("approved_commit_mismatch")

    approved = _approved_source_tree(root, commit)
    paths = tuple(approved)
    indexed = _git_bytes(
        root,
        "ls-files",
        "-v",
        "-z",
    )
    try:
        all_records = tuple(record for record in indexed.split(b"\0") if record)
        records = tuple(
            record
            for record in all_records
            if _source_path_selected(
                record[2:].decode("utf-8", errors="strict")
            )
        )
        index_paths = tuple(
            record[2:].decode("utf-8", errors="strict")
            for record in records
            if record.startswith(b"H ")
        )
    except UnicodeDecodeError as exc:
        raise _BootstrapRejected("source_binding_failed") from exc
    if len(records) != len(index_paths) or index_paths != paths:
        raise _BootstrapRejected("source_binding_failed")
    _reject_unapproved_evalsim_import_artifacts(root, paths)

    digest = hashlib.sha256(b"evalsim-m6-executable-source-v1\0")
    uv_lock_sha256 = ""
    for relative_text in paths:
        file_sha256 = _guarded_source_sha256(root, relative_text)
        blob = _git_bytes(
            root,
            "cat-file",
            "blob",
            f"{commit}:{relative_text}",
        )
        if hashlib.sha256(blob).hexdigest() != file_sha256:
            raise _BootstrapRejected("source_binding_failed")
        encoded = relative_text.encode("utf-8", errors="strict")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(file_sha256))
        if relative_text == "uv.lock":
            uv_lock_sha256 = file_sha256
    if not uv_lock_sha256:
        raise _BootstrapRejected("source_binding_failed")
    return (
        commit,
        tree,
        paths,
        digest.hexdigest(),
        uv_lock_sha256,
    )


def _initial_paths_are_stdlib(initial: tuple[str, ...]) -> bool:
    try:
        base = Path(os.path.abspath(sys.base_prefix)).resolve(strict=True)
    except OSError:
        return False
    if not initial:
        return False
    for entry in initial:
        if not isinstance(entry, str) or not entry or not Path(entry).is_absolute():
            return False
        try:
            Path(os.path.abspath(entry)).relative_to(base)
        except ValueError:
            return False
    return True


def _validated_invocation() -> tuple[Path, Path, Path, tuple[str, ...]]:
    script_lexical = Path(os.path.abspath(__file__))
    script = script_lexical.resolve(strict=True)
    root = script.parents[2]
    expected_script = root / _DIRECT_COMMAND[-1]
    expected_python = root / _DIRECT_COMMAND[0]
    initial = tuple(sys.path)
    site_packages = (
        root
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    flags = sys.flags
    if (
        script_lexical != script
        or script != expected_script
        or not stat.S_ISREG(script.lstat().st_mode)
        or sys.argv[0] != _DIRECT_COMMAND[-1]
        or Path(os.path.abspath(sys.argv[0])) != script
        or Path(os.path.abspath(os.getcwd())).resolve(strict=True) != root
        or Path(os.path.abspath(sys.executable)) != expected_python
        or not expected_python.exists()
        or site_packages.resolve(strict=True) != site_packages
        or flags.isolated != 1
        or flags.no_site != 1
        or flags.dont_write_bytecode != 1
        or flags.no_user_site != 1
        or not flags.safe_path
        or flags.inspect != 0
        or flags.interactive != 0
        or sys.dont_write_bytecode is not True
        or "site" in sys.modules
        or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
        or not _initial_paths_are_stdlib(initial)
    ):
        raise RuntimeError("invalid bootstrap invocation")
    return root, script, site_packages, initial


def _flush_all() -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    fflush = libc.fflush
    fflush.argtypes = [ctypes.c_void_p]
    fflush.restype = ctypes.c_int
    if fflush(None) != 0:
        raise OSError(ctypes.get_errno(), "native stdio flush failed")


def _run_captured(
    root: Path,
    site_packages: Path,
    initial_sys_path: tuple[str, ...],
    *,
    status_failure_callbacks: list[Callable[[str], bool]] | None = None,
) -> tuple[int, bytes, bool, bool]:
    statuses: list[tuple[bytes, bool]] = []
    duplicate_status = False
    failure_callbacks = (
        [] if status_failure_callbacks is None else status_failure_callbacks
    )
    pycache_text = tempfile.mkdtemp(prefix="evalsim-m6-pycache-")
    pycache_prefix = Path(pycache_text).resolve(strict=True)
    if tuple(pycache_prefix.iterdir()):
        raise RuntimeError("pycache prefix is not empty")
    sys.pycache_prefix = os.fspath(pycache_prefix)
    sys.dont_write_bytecode = True
    # Keep the repository off the generic import path until its exact source
    # catalog has been authenticated.
    sys.path[:] = [*initial_sys_path]

    def status_sink(payload: bytes, error: bool) -> bool:
        nonlocal duplicate_status
        if statuses:
            duplicate_status = True
            return False
        if type(payload) is not bytes or type(error) is not bool:
            duplicate_status = True
            return False
        statuses.append((payload, error))
        return True

    def status_failure_sink(callback: Callable[[str], bool]) -> bool:
        nonlocal duplicate_status
        if failure_callbacks or not callable(callback):
            duplicate_status = True
            return False
        failure_callbacks.append(callback)
        return True

    return_code = 1
    try:
        try:
            _parse_argument_surface(root)
            repository_receipt = _preauthenticate_repository(root)
            if any(
                name == "evalsim" or name.startswith("evalsim.")
                for name in sys.modules
            ):
                raise _BootstrapRejected("source_binding_failed")
        except _BootstrapRejected as exc:
            return 1, _rejection(exc.reason_code), True, False

        root_text = os.fspath(root)
        sys.path_importer_cache.pop(root_text, None)
        sys.path[:] = [*initial_sys_path, root_text]
        try:
            from evalsim.cli import m6_official

            expected_origins = {
                "evalsim": root / "evalsim/__init__.py",
                "evalsim.cli": root / "evalsim/cli/__init__.py",
                "evalsim.cli.m6_official": (
                    root / "evalsim/cli/m6_official.py"
                ),
            }
            for name, expected in expected_origins.items():
                module = sys.modules.get(name)
                raw = getattr(module, "__file__", None)
                if (
                    not isinstance(raw, str)
                    or Path(os.path.abspath(raw)) != expected
                    or Path(raw).resolve(strict=True) != expected
                ):
                    raise _BootstrapRejected("source_binding_failed")
            package_path = getattr(sys.modules["evalsim"], "__path__", ())
            if tuple(package_path) != (os.fspath(root / "evalsim"),):
                raise _BootstrapRejected("source_binding_failed")
        except _BootstrapRejected as exc:
            return 1, _rejection(exc.reason_code), True, False
        finally:
            # Loaded EvalSim retains its authenticated package __path__. Remove
            # the repository permanently so it cannot shadow stdlib or the
            # subsequently enabled site-packages domain.
            sys.path[:] = [*initial_sys_path]
            sys.path_importer_cache.pop(root_text, None)

        context = m6_official._issue_m6_bootstrap_context(
            root,
            site_packages,
            pycache_prefix,
            initial_sys_path,
            repository_receipt,
            status_sink,
            status_failure_sink,
        )
        return_code = m6_official.main(
            sys.argv[1:],
            _bootstrap_context=context,
        )
    finally:
        shutil.rmtree(pycache_prefix)
    if len(statuses) != 1:
        return return_code, b"", True, duplicate_status
    payload, error = statuses[0]
    return return_code, payload, error, duplicate_status


def _bootstrap() -> int:
    try:
        root, _script, site_packages, initial_sys_path = _validated_invocation()
    except BaseException:
        return 1 if _write_all(2, _rejection("runtime_mismatch")) else 1
    saved_stdout = saved_stderr = -1
    stdout_capture = stderr_capture = None
    payload = _rejection("terminal_capture_failed")
    error = True
    return_code = 1
    duplicate_status = False
    capture_failed = False
    status_failure_callbacks: list[Callable[[str], bool]] = []
    try:
        _flush_all()
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        os.set_inheritable(saved_stdout, False)
        os.set_inheritable(saved_stderr, False)
        stdout_capture = tempfile.TemporaryFile(mode="w+b")
        stderr_capture = tempfile.TemporaryFile(mode="w+b")
        os.dup2(stdout_capture.fileno(), 1)
        os.dup2(stderr_capture.fileno(), 2)
        try:
            return_code, payload, error, duplicate_status = _run_captured(
                root,
                site_packages,
                initial_sys_path,
                status_failure_callbacks=status_failure_callbacks,
            )
        except BaseException:
            payload = _rejection("runtime_mismatch")
            error = True
            return_code = 1
        try:
            _flush_all()
        except BaseException:
            capture_failed = True
    except BaseException:
        capture_failed = True
    finally:
        for saved, target in ((saved_stdout, 1), (saved_stderr, 2)):
            if saved >= 0:
                try:
                    os.dup2(saved, target)
                except OSError:
                    capture_failed = True
    extra_output = capture_failed or duplicate_status
    for capture in (stdout_capture, stderr_capture):
        if capture is not None:
            try:
                extra_output = extra_output or os.fstat(capture.fileno()).st_size > 0
            except OSError:
                extra_output = True
            try:
                capture.close()
            except OSError:
                extra_output = True
    if extra_output:
        _invoke_status_failure_callback(
            status_failure_callbacks,
            "terminal_output_detected",
        )
        payload = _rejection("terminal_output_detected")
        error = True
        return_code = 1
    elif not _valid_status(payload, error, return_code):
        _invoke_status_failure_callback(
            status_failure_callbacks,
            "terminal_capture_failed",
        )
        payload = _rejection("terminal_capture_failed")
        error = True
        return_code = 1
    elif return_code != 0:
        _invoke_status_failure_callback(
            status_failure_callbacks,
            "terminal_capture_failed",
        )
    descriptor = saved_stderr if error else saved_stdout
    emitted = descriptor >= 0 and _write_all(descriptor, payload)
    if not emitted:
        _invoke_status_failure_callback(
            status_failure_callbacks,
            "terminal_capture_failed",
        )
    for descriptor_to_close in (saved_stdout, saved_stderr):
        if descriptor_to_close >= 0:
            try:
                os.close(descriptor_to_close)
            except OSError:
                pass
    return return_code if emitted else 1


def _invoke_status_failure_callback(
    callbacks: list[Callable[[str], bool]],
    reason_code: str,
) -> bool:
    if not callbacks:
        return True
    if len(callbacks) != 1:
        return False
    try:
        return callbacks[0](reason_code) is True
    except BaseException:
        return False


if __name__ == "__main__":
    raise SystemExit(_bootstrap())
