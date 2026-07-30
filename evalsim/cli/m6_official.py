"""Import-safe, fail-closed lifecycle for local M6 execution.

The module captures the terminal before parsing, optional imports, or local access;
authorizes one pushed/tagged repository snapshot; creates PENDING before every
M4/runtime/shard operation; writes mode-exact evidence; and gates terminal success on
an independently reopened COMMITTED store plus fresh source/runtime/data checks.
All three production modes use the accepted-M4 visitor and reviewed typed execution
authorities. Tests may replace a mode authority through the explicit preparation seam.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from collections.abc import Callable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import re
import secrets
import stat
import subprocess
import sys
import threading
import time
from types import MappingProxyType
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from evalsim.results.m6 import M6ResultStore, M6VerifiedProvenance
    from evalsim.sources.m5_m4_reuse import AcceptedM4Cohort

from ._terminal import (
    TerminalBoundaryError,
    TerminalStatus,
    TerminalizedFailure,
    capture_terminal,
    write_all,
)


M6_OFFICIAL_STATUS_SCHEMA_VERSION = "m6-cli-status-2.0.0"
M6_COMPUTE_PILOT_EVIDENCE_SCHEMA_VERSION = (
    "m6-compute-pilot-evidence-1.0.0"
)
M6_OFFICIAL_MODES = ("eligibility_only", "compute_pilot", "official")
M6_OFFICIAL_PROFILE = "official_m6"
M6_OFFICIAL_DIRECT_COMMAND = (
    ".venv/bin/python",
    "-I",
    "-S",
    "-B",
    "evalsim/cli/m6_bootstrap.py",
)
_EXPECTED_PYTHON_VERSION = "3.11.5"

_LOCAL_OPT_IN = "EVALSIM_RUN_WAYMO_LOCAL"
_CANONICAL_REMOTE = "https://github.com/6ickomod3/evalsim.git"
_CANONICAL_REMOTE_REF = "refs/heads/main"
_APPROVED_IMPLEMENTATION_REF = "refs/tags/m6-approved-v1"
_GIT_BINARY = Path("/usr/bin/git")
_WAYMAX_REMOTE = "https://github.com/waymo-research/waymax.git"
_RUN_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
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
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_M6_REVIEW_COUNT_MAX = 2**31 - 1
_DEFAULT_DATA_RELATIVE = Path(
    "data/raw/womd/v1.3.1/tf_example/validation"
)
_SHARD_SUFFIXES = tuple(
    f"tfrecord-{index:05d}-of-00150" for index in range(10)
)
_M4_REQUIRED_ARTIFACTS = (
    "aggregate-summary.json",
    "cohort/manifest-pass-1.json",
    "cohort/manifest-pass-2.json",
    "execution-provenance.json",
    "terminal-output.bin",
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
_EXPECTED_RUNTIME_VERSIONS = MappingProxyType(
    {
        "flax": "0.10.4",
        "jax": "0.4.38",
        "jaxlib": "0.4.38",
        "keras": "3.15.0",
        "numpy": "1.26.4",
        "pyarrow": "25.0.0",
        "tensorflow": "2.18.1",
        "waymo-waymax": "0.1.0",
    }
)
_RUNTIME_MODULES = MappingProxyType(
    {
        "flax": "flax",
        "jax": "jax",
        "jaxlib": "jaxlib",
        "keras": "keras",
        "numpy": "numpy",
        "pyarrow": "pyarrow",
        "tensorflow": "tensorflow",
        "waymo-waymax": "waymax",
    }
)
_EXPECTED_RUNTIME_CATALOG_SHA256 = MappingProxyType(
    {
        "flax": "39ac24efa428fbef180f3881ed7a3b6aeb4f91dd10d56c231f125830df4b1a27",
        "jax": "b8198cdb6adff5e7da20131462b13d49d3a92260e2ebda0220316b9db9d4abd7",
        "jaxlib": "c31a48c451ab74d870dcf97e2973cd6485d61433d69a655265d1145a1e6d5473",
        "keras": "67089a2da6226b26adb61a17c49f6173409c0b01aa3d5b021db7432ea16a68d1",
        "numpy": "5a4a48f09d238a62d909be8f2d4f8ce17aa984a3e4721976e11bac7131bc14d3",
        "pyarrow": "a540b8bc6ee15a273f089ccdec51f21d677b98b1bf66476122ed669e6f7e01ac",
        "tensorflow": (
            "5014a33282d15f93b6a0c655446ef6672309a7d32c7a8aabb3e430db45afcd55"
        ),
        "waymo-waymax": (
            "6aa999f7189a21d0b0567afee24d7d95677db8e1bcd491c0e0d09ed8c1342229"
        ),
    }
)
_EXPECTED_ENVIRONMENT_VERSIONS = MappingProxyType(
    {
        "absl-py": "2.5.0",
        "asttokens": "3.0.2",
        "astunparse": "1.6.3",
        "certifi": "2026.7.22",
        "charset-normalizer": "3.4.9",
        "chex": "0.1.90",
        "contourpy": "1.3.3",
        "cycler": "0.12.1",
        "decorator": "5.3.1",
        "dm-env": "1.6",
        "dm-tree": "0.1.8",
        "duckdb": "1.5.5",
        "etils": "1.14.0",
        "executing": "2.2.1",
        "flatbuffers": "25.12.19",
        "flax": "0.10.4",
        "fonttools": "4.63.0",
        "fsspec": "2026.7.0",
        "gast": "0.7.0",
        "google-pasta": "0.2.0",
        "grpcio": "1.83.0",
        "h5py": "3.16.0",
        "humanize": "4.16.0",
        "idna": "3.18",
        "immutabledict": "4.3.1",
        "ipython": "9.15.0",
        "ipython-pygments-lexers": "1.1.1",
        "jax": "0.4.38",
        "jaxlib": "0.4.38",
        "jedi": "0.20.0",
        "joblib": "1.5.3",
        "keras": "3.15.0",
        "kiwisolver": "1.5.0",
        "libclang": "18.1.1",
        "markdown": "3.10.2",
        "markdown-it-py": "4.2.0",
        "markupsafe": "3.0.3",
        "matplotlib": "3.11.1",
        "matplotlib-inline": "0.2.2",
        "mdurl": "0.1.2",
        "mediapy": "1.2.7",
        "ml-dtypes": "0.4.1",
        "msgpack": "1.2.1",
        "namex": "0.1.0",
        "narwhals": "2.24.0",
        "nest-asyncio": "1.6.0",
        "numpy": "1.26.4",
        "opt-einsum": "3.4.0",
        "optax": "0.2.5",
        "optree": "0.19.1",
        "orbax-checkpoint": "0.11.5",
        "packaging": "26.2",
        "pandas": "3.0.5",
        "parso": "0.8.7",
        "pexpect": "4.9.0",
        "pillow": "12.3.0",
        "prompt-toolkit": "3.0.53",
        "protobuf": "5.29.6",
        "psutil": "7.2.2",
        "ptyprocess": "0.7.0",
        "pure-eval": "0.2.3",
        "pyarrow": "25.0.0",
        "pygments": "2.20.0",
        "pyparsing": "3.3.2",
        "python-dateutil": "2.9.0.post0",
        "pyyaml": "6.0.3",
        "requests": "2.34.2",
        "rich": "15.0.0",
        "scikit-learn": "1.9.0",
        "scipy": "1.17.1",
        "setuptools": "83.0.0",
        "simplejson": "4.1.1",
        "six": "1.17.0",
        "stack-data": "0.6.3",
        "tensorboard": "2.18.0",
        "tensorboard-data-server": "0.7.2",
        "tensorflow": "2.18.1",
        "tensorflow-io-gcs-filesystem": "0.37.1",
        "tensorstore": "0.1.74",
        "termcolor": "3.3.0",
        "threadpoolctl": "3.6.0",
        "toolz": "1.1.0",
        "tqdm": "4.70.0",
        "traitlets": "5.15.1",
        "treescope": "0.1.10",
        "typing-extensions": "4.16.0",
        "urllib3": "2.7.0",
        "waymo-waymax": "0.1.0",
        "wcwidth": "0.8.2",
        "werkzeug": "3.1.8",
        "wheel": "0.47.0",
        "wrapt": "2.3.0",
        "zipp": "4.1.0",
    }
)
_EXPECTED_ENVIRONMENT_CATALOG_SHA256 = (
    "d00ef0e3593c572e68fe1a8a0b3c05e14d9a5c24c0095c33af3971eaaf706c22"
)
_RUNTIME_SITE_PACKAGES_RELATIVE = ".venv/lib/python3.11/site-packages"
_M6_RESULT_STORE_SCHEMA_VERSION = "m6-result-store-6.2.0"
_EXPECTED_ENVIRONMENT_INFRASTRUCTURE = (
    f"{_RUNTIME_SITE_PACKAGES_RELATIVE}/_editable_impl_evalsim.pth",
    f"{_RUNTIME_SITE_PACKAGES_RELATIVE}/_virtualenv.pth",
    f"{_RUNTIME_SITE_PACKAGES_RELATIVE}/_virtualenv.py",
)
_EXPECTED_PROJECT_DISTRIBUTION_VERSION = "0.1.0"
_EXPECTED_ENVIRONMENT_SHARED_OWNERS = MappingProxyType(
    {
        ".venv/bin/tensorboard": frozenset({"tensorboard", "tensorflow"}),
    }
)
_RUNTIME_EXECUTABLE_SUFFIXES = tuple(
    sorted(
        {
            ".py",
            ".so",
            ".dylib",
            ".pyd",
            ".dll",
            ".pyc",
            *importlib.machinery.EXTENSION_SUFFIXES,
        }
    )
)
_BOOTSTRAP_SENTINEL = object()
_PREFLIGHT_SENTINEL = object()
_RUNTIME_SENTINEL = object()
_PENDING_RESERVATION_SENTINEL = object()
_ACTIVE_BOOTSTRAP_CONTEXT: _M6BootstrapContext | None = None
_BOOTSTRAP_CONTEXT_LOCK = threading.RLock()
_TRUSTED_CODES = frozenset(
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


class M6OfficialCommandError(RuntimeError):
    """A terminal-safe command rejection carrying one registered code."""

    def __init__(self, code: str, message: str) -> None:
        if code not in _TRUSTED_CODES:
            raise ValueError("unregistered M6 command reason code")
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(slots=True)
class _M6BootstrapContext:
    project_root: Path = field(repr=False)
    site_packages: Path = field(repr=False)
    pycache_prefix: Path = field(repr=False)
    initial_sys_path: tuple[str, ...] = field(repr=False)
    repository_receipt: tuple[object, ...] = field(repr=False)
    status_sink: Callable[[bytes, bool], bool] = field(repr=False)
    status_failure_sink: Callable[
        [Callable[[str], bool]],
        bool,
    ] = field(repr=False)
    _factory_sentinel: object = field(repr=False)
    consumed: bool = field(default=False, repr=False)
    site_packages_enabled: bool = field(default=False, repr=False)


class _SilentParser(argparse.ArgumentParser):
    """Raise without writing; parsing occurs inside descriptor capture."""

    def error(self, message: str) -> None:
        del message
        raise M6OfficialCommandError(
            "argument_error",
            "the command arguments do not match the exact M6 surface",
        )


@dataclass(frozen=True, slots=True)
class M6CommandRequest:
    project_root: Path = field(repr=False)
    data_dir: Path = field(repr=False)
    m4_run_dir: Path = field(repr=False)
    run_name: str
    mode: str
    eligibility_run_name: str | None = None
    pilot_run_name: str | None = None

    def __post_init__(self) -> None:
        for name in ("project_root", "data_dir", "m4_run_dir"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a Path")
        _validated_run_name(self.run_name)
        if self.mode not in M6_OFFICIAL_MODES:
            raise ValueError("mode is not a registered official M6 mode")
        expected = {
            "eligibility_only": (False, False),
            "compute_pilot": (True, False),
            "official": (True, True),
        }[self.mode]
        observed = (
            self.eligibility_run_name is not None,
            self.pilot_run_name is not None,
        )
        if observed != expected:
            raise M6OfficialCommandError(
                "argument_error",
                "predecessor run names do not match the selected mode",
            )
        predecessors = tuple(
            value
            for value in (
                self.eligibility_run_name,
                self.pilot_run_name,
            )
            if value is not None
        )
        for value in predecessors:
            _validated_run_name(value)
        if (
            self.run_name in predecessors
            or len(predecessors) != len(set(predecessors))
        ):
            raise M6OfficialCommandError(
                "argument_error",
                "current and predecessor run names must be distinct",
            )


@dataclass(frozen=True, slots=True)
class M6ReviewInput:
    """One explicit role decision supplied after the precursor is sealed."""

    role: str
    decision: str
    p1_count: int
    p2_count: int
    p3_count: int

    def __post_init__(self) -> None:
        if self.role not in (
            "architecture",
            "methods_statistics",
            "privacy_claim",
        ):
            raise M6OfficialCommandError(
                "argument_error",
                "review role is not registered",
            )
        if self.decision not in {"accept", "reject"}:
            raise M6OfficialCommandError(
                "argument_error",
                "review decision must be accept or reject",
            )
        if any(
            type(value) is not int
            or value < 0
            or value > _M6_REVIEW_COUNT_MAX
            for value in (self.p1_count, self.p2_count, self.p3_count)
        ):
            raise M6OfficialCommandError(
                "argument_error",
                "review counts must fit the persisted int32 domain",
            )


@dataclass(frozen=True, slots=True)
class M6ReviewFinalizationRequest:
    """Tracked noninteractive review decisions for one awaiting official run."""

    command: M6CommandRequest
    reviews: tuple[M6ReviewInput, ...]

    def __post_init__(self) -> None:
        if self.command.mode != "official":
            raise M6OfficialCommandError(
                "argument_error",
                "review finalization is official-mode-only",
            )
        if tuple(item.role for item in self.reviews) != (
            "architecture",
            "methods_statistics",
            "privacy_claim",
        ):
            raise M6OfficialCommandError(
                "argument_error",
                "review finalization requires the exact role domain",
            )


@dataclass(frozen=True, slots=True)
class M6GitSnapshot:
    commit: str
    tree: str
    approval_ref: str

    def __post_init__(self) -> None:
        if _GIT_OBJECT_ID.fullmatch(self.commit) is None or (
            _GIT_OBJECT_ID.fullmatch(self.tree) is None
        ):
            raise ValueError("Git snapshot objects must be 40-hex IDs")
        if self.approval_ref != _APPROVED_IMPLEMENTATION_REF:
            raise ValueError("Git snapshot approval ref is not the fixed M6 tag")


@dataclass(frozen=True, slots=True)
class _GuardedFileSnapshot:
    relative_path: str
    sha256: str
    identity: tuple[int, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.relative_path, str)
            or Path(self.relative_path).is_absolute()
            or Path(self.relative_path).as_posix() != self.relative_path
            or _SHA256.fullmatch(self.sha256) is None
            or not self.identity
            or any(type(value) is not int for value in self.identity)
        ):
            raise ValueError("guarded file snapshot is invalid")


@dataclass(frozen=True, slots=True)
class _NodeIdentity:
    relative_name: str
    identity: tuple[int, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.relative_name, str)
            or Path(self.relative_name).name != self.relative_name
            or not self.identity
            or any(type(value) is not int for value in self.identity)
        ):
            raise ValueError("node identity is invalid")


@dataclass(frozen=True, slots=True)
class _RuntimePackageCatalog:
    distribution_name: str
    module_name: str
    roots: tuple[str, ...]
    files: tuple[_GuardedFileSnapshot, ...] = field(repr=False)
    metadata_files: tuple[_GuardedFileSnapshot, ...] = field(repr=False)
    sha256: str

    def __post_init__(self) -> None:
        paths = tuple(item.relative_path for item in self.files)
        metadata_paths = tuple(item.relative_path for item in self.metadata_files)
        if (
            self.distribution_name not in _RUNTIME_MODULES
            or _RUNTIME_MODULES[self.distribution_name] != self.module_name
            or not self.roots
            or self.roots != tuple(sorted(set(self.roots)))
            or any(
                not isinstance(root, str)
                or Path(root).is_absolute()
                or Path(root).as_posix() != root
                or Path(root).parts[:1] != (".venv",)
                for root in self.roots
            )
            or not paths
            or paths != tuple(sorted(set(paths)))
            or any(Path(path).parts[:1] != (".venv",) for path in paths)
            or metadata_paths != tuple(sorted(set(metadata_paths)))
            or any(
                Path(path).parts[:1] != (".venv",)
                or Path(path).name not in {"RECORD", "direct_url.json"}
                or not Path(path).parent.name.endswith(".dist-info")
                for path in metadata_paths
            )
            or sum(Path(path).name == "RECORD" for path in metadata_paths) != 1
            or sum(
                Path(path).name == "direct_url.json" for path in metadata_paths
            )
            > 1
            or (
                self.distribution_name == "waymo-waymax"
                and sum(
                    Path(path).name == "direct_url.json"
                    for path in metadata_paths
                )
                != 1
            )
            or _SHA256.fullmatch(self.sha256) is None
            or self.sha256 != _runtime_package_fingerprint(
                self.distribution_name,
                self.module_name,
                self.roots,
                self.files,
                self.metadata_files,
            )
        ):
            raise ValueError("runtime package catalog is invalid")


@dataclass(frozen=True, slots=True)
class _EnvironmentDistributionCatalog:
    distribution_name: str
    version: str
    files: tuple[_GuardedFileSnapshot, ...] = field(repr=False)
    sha256: str

    def __post_init__(self) -> None:
        paths = tuple(item.relative_path for item in self.files)
        if (
            not isinstance(self.distribution_name, str)
            or _canonical_distribution_name(self.distribution_name)
            != self.distribution_name
            or not isinstance(self.version, str)
            or not self.version
            or not paths
            or paths != tuple(sorted(set(paths)))
            or any(Path(path).parts[:1] != (".venv",) for path in paths)
            or _SHA256.fullmatch(self.sha256) is None
            or self.sha256 != _environment_distribution_fingerprint(
                self.distribution_name, self.version, self.files
            )
        ):
            raise ValueError("environment distribution catalog is invalid")


@dataclass(frozen=True, slots=True)
class M6RuntimeObservation:
    versions: Mapping[str, str]
    module_origins: Mapping[str, str]
    python_version: str
    jax_backend: str
    jax_device_class: str
    waymax_commit: str
    uv_lock_sha256: str
    runtime_catalog_sha256: str
    environment_catalog_sha256: str
    runtime_config_sha256: str
    _runtime_catalog: tuple[_RuntimePackageCatalog, ...] = field(
        repr=False,
    )
    _environment_catalog: tuple[_EnvironmentDistributionCatalog, ...] = field(
        repr=False,
    )
    _environment_infrastructure: tuple[_GuardedFileSnapshot, ...] = field(
        repr=False,
    )
    _factory_sentinel: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_sentinel is not _RUNTIME_SENTINEL:
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "runtime observations are issued only by the bounded observer",
            )
        versions = dict(self.versions)
        origins = dict(self.module_origins)
        if versions != dict(_EXPECTED_RUNTIME_VERSIONS):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "the observed versions differ from the pinned runtime",
            )
        if tuple(origins) != tuple(_RUNTIME_MODULES) or any(
            not isinstance(value, str) or not value.startswith(".venv/")
            for value in origins.values()
        ):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "runtime module origins differ from the repository-local environment",
            )
        if (
            self.python_version != _EXPECTED_PYTHON_VERSION
            or self.jax_backend != "cpu"
            or self.jax_device_class != "cpu"
            or _GIT_OBJECT_ID.fullmatch(self.waymax_commit) is None
            or _SHA256.fullmatch(self.uv_lock_sha256) is None
            or _SHA256.fullmatch(self.runtime_catalog_sha256) is None
            or _SHA256.fullmatch(self.runtime_config_sha256) is None
            or _SHA256.fullmatch(self.environment_catalog_sha256) is None
            or type(self._runtime_catalog) is not tuple
            or _runtime_catalog_sha256(self._runtime_catalog)
            != self.runtime_catalog_sha256
            or type(self._environment_catalog) is not tuple
            or type(self._environment_infrastructure) is not tuple
            or _environment_catalog_sha256(
                self._environment_catalog,
                self._environment_infrastructure,
            )
            != self.environment_catalog_sha256
        ):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "runtime identity or CPU execution binding is invalid",
            )
        object.__setattr__(self, "versions", MappingProxyType(versions))
        object.__setattr__(self, "module_origins", MappingProxyType(origins))


@dataclass(frozen=True, slots=True)
class M6RepositoryPreflight:
    root: Path = field(repr=False)
    git: M6GitSnapshot
    source_paths: tuple[str, ...]
    source_sha256: str
    source_snapshots: tuple[_GuardedFileSnapshot, ...] = field(repr=False)
    uv_lock_sha256: str
    context_sha256: str
    _factory_sentinel: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_sentinel is not _PREFLIGHT_SENTINEL:
            raise M6OfficialCommandError(
                "source_binding_failed",
                "repository preflight values are verifier-issued only",
            )
        if (
            tuple(item.relative_path for item in self.source_snapshots)
            != self.source_paths
            or _SHA256.fullmatch(self.source_sha256) is None
            or _SHA256.fullmatch(self.uv_lock_sha256) is None
            or _SHA256.fullmatch(self.context_sha256) is None
            or self.context_sha256 != _repository_context_sha256(
                self.git,
                self.source_paths,
                self.source_sha256,
                self.uv_lock_sha256,
            )
        ):
            raise M6OfficialCommandError(
                "source_binding_failed",
                "repository preflight binding is invalid",
            )


@dataclass(frozen=True, slots=True)
class M6LocalInputPreflight:
    run_name: str
    mode: str
    result_relative: str
    repository_context_sha256: str
    data_dir: Path = field(repr=False)
    shard_paths: tuple[Path, ...] = field(repr=False)
    shard_identities: tuple[_NodeIdentity, ...] = field(repr=False)
    m4_run_dir: Path = field(repr=False)
    accepted_m4: object = field(repr=False, compare=False)
    accepted_m4_manifest_sha256: str
    accepted_m4_provenance_sha256: str
    runtime: M6RuntimeObservation
    verified_provenance: Any = field(repr=False, compare=False)
    _factory_sentinel: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_sentinel is not _PREFLIGHT_SENTINEL:
            raise M6OfficialCommandError(
                "source_binding_failed",
                "local preflight values are verifier-issued only",
            )
        _validated_run_name(self.run_name)
        if self.mode not in M6_OFFICIAL_MODES or self.result_relative != (
            f"outputs/m6/{self.run_name}"
        ):
            raise M6OfficialCommandError(
                "result_contract_failed",
                "local preflight result binding is invalid",
            )
        if (
            _SHA256.fullmatch(self.repository_context_sha256) is None
            or _SHA256.fullmatch(self.accepted_m4_manifest_sha256) is None
            or _SHA256.fullmatch(self.accepted_m4_provenance_sha256) is None
            or type(self.runtime) is not M6RuntimeObservation
            or len(self.shard_paths) != len(_SHARD_SUFFIXES)
            or tuple(item.relative_name for item in self.shard_identities)
            != tuple(path.name for path in self.shard_paths)
            or getattr(self.verified_provenance, "mode", None) != self.mode
            or _SHA256.fullmatch(
                getattr(self.verified_provenance, "context_sha256", "")
            )
            is None
        ):
            raise M6OfficialCommandError(
                "source_binding_failed",
                "local input preflight evidence is incomplete",
            )

    @property
    def runtime_versions(self) -> Mapping[str, str]:
        return self.runtime.versions


@dataclass(frozen=True, slots=True)
class _CommandResult:
    mode: str
    result_relative: Path
    aggregate_counts: Mapping[str, int]
    stage_durations_ms: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class _AwaitingReviewResult:
    mode: str
    result_relative: Path
    evidence_catalog_sha256: str
    mechanical_verification_sha256: str


@dataclass(frozen=True, slots=True)
class _M6ComputePilotEvidenceIssuance:
    evidence: Any
    eligibility_rows: object
    eligibility_rows_sha256: str
    selection: object
    pilot_summary: object
    pilot_selection_positions: object
    numpy_observation: object
    waymax_observation: object
    verified_provenance: object
    run_name: str
    result_path: str
    selection_binding_sha256: str
    selected_cohort_indices_sha256: str
    numpy_observation_content_sha256: str
    waymax_observation_content_sha256: str
    pilot_report_binding_sha256: str


_COMPUTE_PILOT_EVIDENCE_ISSUER = object()
_COMPUTE_PILOT_EVIDENCE_LOCK = threading.Lock()
_COMPUTE_PILOT_EVIDENCE_REGISTRY: dict[
    int, _M6ComputePilotEvidenceIssuance
] = {}


def _compute_pilot_eligibility_rows_sha256(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    return hashlib.sha256(
        b"evalsim-m6-compute-pilot-eligibility-v1\x00"
        + _canonical_json_bytes(
            {
                "rows": [dict(row) for row in rows],
                "schema_version": M6_COMPUTE_PILOT_EVIDENCE_SCHEMA_VERSION,
            }
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class M6ModeExecutionEvidence:
    """Outcome-minimal evidence returned by one injected local executor.

    The compute-pilot variant is factory-issued only. Its external issuance
    registry retains the exact selection, provenance, and two independently
    registry-sealed observations so a copied or recomputed aggregate cannot be
    substituted after execution.
    """

    mode: str
    eligibility_rows: tuple[Mapping[str, Any], ...]
    selection: object = field(repr=False, compare=False)
    pilot_summary: Mapping[str, Any] | None = None
    pilot_selection_positions: tuple[int, ...] = ()
    numpy_rows: object | None = field(default=None, repr=False, compare=False)
    waymax_evidence: object | None = field(default=None, repr=False, compare=False)
    stage_durations_ms: Mapping[str, int] | None = None
    fresh_worker_peak_rss_bytes: int | None = None
    promotable: bool = True
    pilot_numpy_observation: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    pilot_waymax_observation: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    pilot_verified_provenance: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    pilot_run_name: str | None = field(default=None, repr=False)
    pilot_result_path: str | None = field(default=None, repr=False)
    pilot_report_binding_sha256: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _pilot_issuance_capability: InitVar[object] = None

    def __post_init__(self, _pilot_issuance_capability: object) -> None:
        if self.mode not in M6_OFFICIAL_MODES:
            raise ValueError("execution evidence mode is not registered")
        rows = tuple(self.eligibility_rows)
        if len(rows) != 128 or any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("execution evidence requires 128 eligibility rows")
        object.__setattr__(self, "eligibility_rows", rows)
        if type(self.promotable) is not bool:
            raise TypeError("execution evidence promotable must be a boolean")
        if self.mode == "eligibility_only":
            if _pilot_issuance_capability is not None:
                raise TypeError("pilot issuance authority is mode-bound")
            if (
                self.pilot_summary is not None
                or self.pilot_selection_positions
                or self.numpy_rows is not None
                or self.waymax_evidence is not None
                or self.stage_durations_ms is not None
                or self.fresh_worker_peak_rss_bytes is not None
                or self._has_pilot_components()
            ):
                raise ValueError("eligibility evidence must contain no outcomes")
            return
        if self.mode == "compute_pilot":
            if _pilot_issuance_capability is not _COMPUTE_PILOT_EVIDENCE_ISSUER:
                raise TypeError("compute-pilot evidence is runner-issued only")
            if (
                self.numpy_rows is not None
                or self.waymax_evidence is not None
                or self.stage_durations_ms is not None
            ):
                raise ValueError("pilot evidence must contain no scene outcomes")
            self._seal_pilot_issuance()
            return
        if _pilot_issuance_capability is not None or self._has_pilot_components():
            raise TypeError("pilot issuance authority is mode-bound")
        if (
            self.pilot_summary is not None
            or self.pilot_selection_positions
            or self.numpy_rows is None
            or self.waymax_evidence is None
            or self.stage_durations_ms is None
            or isinstance(self.fresh_worker_peak_rss_bytes, bool)
            or not isinstance(self.fresh_worker_peak_rss_bytes, int)
            or self.fresh_worker_peak_rss_bytes <= 0
            or self.promotable is not True
        ):
            raise ValueError("official execution evidence is incomplete")
        durations = dict(self.stage_durations_ms)
        if set(durations) != {
            "eligibility",
            "numpy_rollouts",
            "paired_metrics",
            "statistics",
            "waymax",
            "verification",
        } or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in durations.values()
        ):
            raise ValueError("official stage timings must be exact and positive")
        object.__setattr__(self, "stage_durations_ms", MappingProxyType(durations))

    def _has_pilot_components(self) -> bool:
        return any(
            value is not None
            for value in (
                self.pilot_numpy_observation,
                self.pilot_waymax_observation,
                self.pilot_verified_provenance,
                self.pilot_run_name,
                self.pilot_result_path,
                self.pilot_report_binding_sha256,
            )
        )

    def _validated_pilot_components(
        self,
    ) -> tuple[dict[str, Any], str, str, str, str, str]:
        from evalsim.evaluation.m6_pilot import (
            M6NumpyPilotObservation,
            m6_numpy_pilot_selected_cohort_indices_sha256,
        )
        from evalsim.evaluation.m6_waymax_official import (
            M6WaymaxPilotObservation,
            m6_waymax_selection_binding_sha256,
        )
        from evalsim.results.m6 import (
            M6VerifiedProvenance,
            _m6_compute_pilot_rounding_overage_ms,
            m6_compute_pilot_report_binding_sha256,
        )

        if self.pilot_summary is None:
            raise ValueError("compute pilot requires one aggregate summary")
        if (
            type(self.pilot_numpy_observation) is not M6NumpyPilotObservation
            or type(self.pilot_waymax_observation) is not M6WaymaxPilotObservation
            or type(self.pilot_verified_provenance) is not M6VerifiedProvenance
        ):
            raise TypeError(
                "compute pilot requires exact registry-issued observations and provenance"
            )
        numpy_observation = self.pilot_numpy_observation
        waymax_observation = self.pilot_waymax_observation
        provenance = self.pilot_verified_provenance
        numpy_observation.revalidate()
        waymax_observation.revalidate()
        provenance.revalidate()
        if provenance.mode != "compute_pilot":
            raise ValueError("compute-pilot provenance is mode-mismatched")
        if (
            type(self.pilot_run_name) is not str
            or _RUN_NAME.fullmatch(self.pilot_run_name) is None
            or self.pilot_result_path
            != f"outputs/m6/{self.pilot_run_name}"
        ):
            raise ValueError("compute-pilot run/result binding is invalid")
        selection_binding = m6_waymax_selection_binding_sha256(self.selection)
        if self.selection.supported:
            selected_indices = tuple(
                member.cohort_index for member in self.selection.members[:8]
            )
        else:
            selected_indices = tuple(
                row.cohort_index
                for row in sorted(
                    self.selection.qualification_ledger.rows,
                    key=lambda row: (
                        bytes.fromhex(row.rank_sha256),
                        row.cohort_index,
                    ),
                )[:8]
            )
        selected_indices_binding = (
            m6_numpy_pilot_selected_cohort_indices_sha256(selected_indices)
        )
        numpy_binding = numpy_observation.observation_binding_sha256
        waymax_binding = waymax_observation.observation_binding_sha256
        row = dict(self.pilot_summary)
        expected = {
            "pilot_scene_n",
            "total_wall_ms",
            "max_scene_ms",
            "decode_ms",
            "numpy_ms",
            "waymax_ms",
            "verification_ms",
            "fresh_worker_peak_rss_bytes",
            "passed",
        }
        if set(row) != expected:
            raise ValueError("compute pilot summary fields are not exact")
        durations = tuple(
            row[name]
            for name in (
                "total_wall_ms",
                "max_scene_ms",
                "decode_ms",
                "numpy_ms",
                "waymax_ms",
                "verification_ms",
            )
        )
        rss = row["fresh_worker_peak_rss_bytes"]
        expected_max_scene = max(
            numpy_observation.max_scene_ms,
            waymax_observation.max_scene_ms,
        )
        expected_numpy_ms = numpy_observation.total_execution_ms
        expected_waymax_ms = (
            waymax_observation.validation_ms
            + waymax_observation.execution_ms
        )
        rounding_overage_ms = _m6_compute_pilot_rounding_overage_ms(
            numpy_scene_n=numpy_observation.scene_count,
            waymax_scene_n=waymax_observation.scene_count,
        )
        expected_passed = (
            row["total_wall_ms"] <= 30 * 60 * 1000
            and expected_max_scene <= 10 * 60 * 1000
            and rss <= 16 * 1024**3
        )
        expected_waymax_status = (
            "completed" if self.selection.supported else "unsupported"
        )
        expected_waymax_scene_n = 8 if self.selection.supported else 0
        if (
            row["pilot_scene_n"] != 8
            or self.pilot_selection_positions != tuple(range(8))
            or numpy_observation.scene_count != 8
            or numpy_observation.source_selection_binding_sha256
            != selection_binding
            or numpy_observation.selected_cohort_indices_sha256
            != selected_indices_binding
            or waymax_observation.selection_binding_sha256 != selection_binding
            or waymax_observation.selected_cohort_indices_sha256
            != selected_indices_binding
            or waymax_observation.status != expected_waymax_status
            or waymax_observation.scene_count != expected_waymax_scene_n
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in durations
            )
            or row["max_scene_ms"] != expected_max_scene
            or row["numpy_ms"] != expected_numpy_ms
            or row["waymax_ms"] != expected_waymax_ms
            or row["max_scene_ms"] > row["total_wall_ms"]
            or row["total_wall_ms"] + rounding_overage_ms
            < sum(
                row[name]
                for name in (
                    "decode_ms",
                    "numpy_ms",
                    "waymax_ms",
                    "verification_ms",
                )
            )
            or isinstance(rss, bool)
            or not isinstance(rss, int)
            or rss <= 0
            or rss < waymax_observation.peak_process_rss_bytes
            or self.fresh_worker_peak_rss_bytes != rss
            or type(row["passed"]) is not bool
            or row["passed"] is not expected_passed
            or self.promotable is not True
        ):
            raise ValueError("compute pilot evidence is not mechanically consistent")
        report_binding = m6_compute_pilot_report_binding_sha256(
            run_name=self.pilot_run_name,
            result_path=self.pilot_result_path,
            provenance_context_sha256=provenance.context_sha256,
            selection_binding_sha256=selection_binding,
            selected_cohort_indices_sha256=selected_indices_binding,
            numpy_observation_content_sha256=numpy_binding,
            waymax_observation_content_sha256=waymax_binding,
            summary=row,
        )
        return (
            row,
            selection_binding,
            selected_indices_binding,
            numpy_binding,
            waymax_binding,
            report_binding,
        )

    def _seal_pilot_issuance(self) -> None:
        (
            row,
            selection_binding,
            selected_indices_binding,
            numpy_binding,
            waymax_binding,
            report_binding,
        ) = self._validated_pilot_components()
        summary = MappingProxyType(row)
        positions = tuple(self.pilot_selection_positions)
        object.__setattr__(self, "pilot_summary", summary)
        object.__setattr__(self, "pilot_selection_positions", positions)
        object.__setattr__(
            self,
            "pilot_report_binding_sha256",
            report_binding,
        )
        record = _M6ComputePilotEvidenceIssuance(
            evidence=self,
            eligibility_rows=self.eligibility_rows,
            eligibility_rows_sha256=(
                _compute_pilot_eligibility_rows_sha256(self.eligibility_rows)
            ),
            selection=self.selection,
            pilot_summary=summary,
            pilot_selection_positions=positions,
            numpy_observation=self.pilot_numpy_observation,
            waymax_observation=self.pilot_waymax_observation,
            verified_provenance=self.pilot_verified_provenance,
            run_name=self.pilot_run_name,
            result_path=self.pilot_result_path,
            selection_binding_sha256=selection_binding,
            selected_cohort_indices_sha256=selected_indices_binding,
            numpy_observation_content_sha256=numpy_binding,
            waymax_observation_content_sha256=waymax_binding,
            pilot_report_binding_sha256=report_binding,
        )
        with _COMPUTE_PILOT_EVIDENCE_LOCK:
            if id(self) in _COMPUTE_PILOT_EVIDENCE_REGISTRY:
                raise RuntimeError("compute-pilot evidence identity was reused")
            _COMPUTE_PILOT_EVIDENCE_REGISTRY[id(self)] = record

    def revalidate_pilot(
        self,
        *,
        run_name: str | None = None,
        result_path: str | None = None,
        selection: object | None = None,
        verified_provenance: object | None = None,
    ) -> None:
        if self.mode != "compute_pilot":
            raise TypeError("pilot revalidation is compute_pilot-mode-only")
        with _COMPUTE_PILOT_EVIDENCE_LOCK:
            record = _COMPUTE_PILOT_EVIDENCE_REGISTRY.get(id(self))
        (
            _row,
            selection_binding,
            selected_indices_binding,
            numpy_binding,
            waymax_binding,
            report_binding,
        ) = self._validated_pilot_components()
        if (
            record is None
            or record.evidence is not self
            or self.eligibility_rows is not record.eligibility_rows
            or _compute_pilot_eligibility_rows_sha256(self.eligibility_rows)
            != record.eligibility_rows_sha256
            or self.selection is not record.selection
            or self.pilot_summary is not record.pilot_summary
            or self.pilot_selection_positions
            is not record.pilot_selection_positions
            or self.pilot_numpy_observation is not record.numpy_observation
            or self.pilot_waymax_observation is not record.waymax_observation
            or self.pilot_verified_provenance is not record.verified_provenance
            or self.pilot_run_name != record.run_name
            or self.pilot_result_path != record.result_path
            or selection_binding != record.selection_binding_sha256
            or selected_indices_binding
            != record.selected_cohort_indices_sha256
            or numpy_binding != record.numpy_observation_content_sha256
            or waymax_binding != record.waymax_observation_content_sha256
            or report_binding != record.pilot_report_binding_sha256
            or self.pilot_report_binding_sha256 != report_binding
            or (run_name is not None and run_name != record.run_name)
            or (result_path is not None and result_path != record.result_path)
            or (selection is not None and selection is not record.selection)
            or (
                verified_provenance is not None
                and verified_provenance is not record.verified_provenance
            )
        ):
            raise ValueError("compute-pilot evidence integrity binding is invalid")

    @property
    def pilot_selection_binding_sha256(self) -> str:
        self.revalidate_pilot()
        with _COMPUTE_PILOT_EVIDENCE_LOCK:
            return _COMPUTE_PILOT_EVIDENCE_REGISTRY[
                id(self)
            ].selection_binding_sha256

    @property
    def pilot_selected_cohort_indices_sha256(self) -> str:
        self.revalidate_pilot()
        with _COMPUTE_PILOT_EVIDENCE_LOCK:
            return _COMPUTE_PILOT_EVIDENCE_REGISTRY[
                id(self)
            ].selected_cohort_indices_sha256

    @property
    def pilot_numpy_observation_content_sha256(self) -> str:
        self.revalidate_pilot()
        with _COMPUTE_PILOT_EVIDENCE_LOCK:
            return _COMPUTE_PILOT_EVIDENCE_REGISTRY[
                id(self)
            ].numpy_observation_content_sha256

    @property
    def pilot_waymax_observation_content_sha256(self) -> str:
        self.revalidate_pilot()
        with _COMPUTE_PILOT_EVIDENCE_LOCK:
            return _COMPUTE_PILOT_EVIDENCE_REGISTRY[
                id(self)
            ].waymax_observation_content_sha256


def _issue_m6_compute_pilot_evidence(
    *,
    eligibility_rows: Sequence[Mapping[str, Any]],
    selection: object,
    pilot_summary: Mapping[str, Any],
    pilot_selection_positions: Sequence[int],
    numpy_observation: object,
    waymax_observation: object,
    verified_provenance: object,
    run_name: str,
    result_path: str,
    fresh_worker_peak_rss_bytes: int,
) -> M6ModeExecutionEvidence:
    """Seal the aggregate only after both pilot observations are complete."""

    return M6ModeExecutionEvidence(
        mode="compute_pilot",
        eligibility_rows=tuple(eligibility_rows),
        selection=selection,
        pilot_summary=pilot_summary,
        pilot_selection_positions=tuple(pilot_selection_positions),
        fresh_worker_peak_rss_bytes=fresh_worker_peak_rss_bytes,
        pilot_numpy_observation=numpy_observation,
        pilot_waymax_observation=waymax_observation,
        pilot_verified_provenance=verified_provenance,
        pilot_run_name=run_name,
        pilot_result_path=result_path,
        _pilot_issuance_capability=_COMPUTE_PILOT_EVIDENCE_ISSUER,
    )


@dataclass(slots=True)
class _PreparedOfficialRun:
    request: M6CommandRequest
    repository: M6RepositoryPreflight
    local: M6LocalInputPreflight
    store: object = field(repr=False, compare=False)
    results_module: object = field(repr=False, compare=False)
    live_lookup: Callable[[Path], str] = field(repr=False, compare=False)
    live_approval_lookup: Callable[[Path], str] = field(repr=False, compare=False)
    m4_reverifier: Callable[[object], None] | None = field(
        repr=False,
        compare=False,
    )
    runtime_kwargs: Mapping[str, object] = field(repr=False, compare=False)
    shard_reverifier: Callable[[M6LocalInputPreflight], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    eligibility_duration_ms: int = 0
    success_payload: bytes | None = field(default=None, repr=False)


@dataclass(slots=True)
class _RunHolder:
    store: object | None = field(default=None, repr=False)
    outcome_started: bool = field(default=False, repr=False)


@dataclass(frozen=True, slots=True)
class _M6PendingReservation:
    """Stdlib-only one-shot capability for an already durable PENDING store."""

    project_root: Path
    run_name: str
    run_path: Path
    mode: str
    capability_nonce: bytes = field(repr=False)
    pending_payload: bytes = field(repr=False)
    _factory_sentinel: object = field(repr=False)

    def fail(self, reason_code: str) -> Path:
        return _fail_pending_reservation(self, reason_code)


@dataclass(slots=True)
class _M6PendingReservationRecord:
    reservation: _M6PendingReservation
    state: str = "pending"


_PENDING_RESERVATION_LOCK = threading.Lock()
_PENDING_RESERVATION_REGISTRY: dict[int, _M6PendingReservationRecord] = {}


@dataclass(frozen=True, slots=True)
class _FailedCreationView:
    """Read-only handle when store creation already wrote TERMINAL_FAILURE."""

    project_root: Path
    run_name: str
    run_path: Path

    def fail(self, reason_code: str) -> Path:
        del reason_code
        marker = self.run_path / "TERMINAL_FAILURE"
        metadata = marker.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or marker.resolve(strict=True) != marker
        ):
            raise OSError("store creation failure marker is unsafe")
        return marker


def _parser() -> argparse.ArgumentParser:
    parser = _SilentParser(
        prog="evalsim-m6-official",
        add_help=False,
        description=(
            "Run one capture-first M6 local mode through the reviewed, injected "
            "execution authority."
        ),
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--m4-run-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--eligibility-run-name")
    parser.add_argument("--pilot-run-name")
    parser.add_argument("--mode", choices=M6_OFFICIAL_MODES, required=True)
    return parser


def _parse_request(argv: Sequence[str] | None) -> M6CommandRequest:
    args = _parser().parse_args(argv)
    return M6CommandRequest(
        project_root=args.project_root,
        data_dir=args.data_dir,
        m4_run_dir=args.m4_run_dir,
        run_name=args.run_name,
        eligibility_run_name=args.eligibility_run_name,
        pilot_run_name=args.pilot_run_name,
        mode=args.mode,
    )


_REVIEW_ARGUMENT_DOMAIN = (
    ("architecture", "architecture"),
    ("methods_statistics", "methods-statistics"),
    ("privacy_claim", "privacy-claim"),
)


def _nonnegative_review_count(value: str) -> int:
    if re.fullmatch(r"(?:0|[1-9][0-9]{0,18})", value) is None:
        raise argparse.ArgumentTypeError("invalid review count")
    parsed = int(value)
    if parsed > _M6_REVIEW_COUNT_MAX:
        raise argparse.ArgumentTypeError("review count exceeds persisted int32")
    return parsed


def _review_finalization_parser() -> argparse.ArgumentParser:
    parser = _parser()
    parser.prog = "evalsim-m6-finalize-review"
    parser.add_argument(
        "--action",
        choices=("finalize-review",),
        required=True,
    )
    for _role, option_prefix in _REVIEW_ARGUMENT_DOMAIN:
        parser.add_argument(
            f"--{option_prefix}-decision",
            choices=("accept", "reject"),
            required=True,
        )
        for priority in ("p1", "p2", "p3"):
            parser.add_argument(
                f"--{option_prefix}-{priority}-count",
                type=_nonnegative_review_count,
                required=True,
            )
    return parser


def _parse_review_finalization_request(
    argv: Sequence[str] | None,
) -> M6ReviewFinalizationRequest:
    args = _review_finalization_parser().parse_args(argv)
    command = M6CommandRequest(
        project_root=args.project_root,
        data_dir=args.data_dir,
        m4_run_dir=args.m4_run_dir,
        run_name=args.run_name,
        eligibility_run_name=args.eligibility_run_name,
        pilot_run_name=args.pilot_run_name,
        mode=args.mode,
    )
    reviews = tuple(
        M6ReviewInput(
            role=role,
            decision=getattr(args, f"{role}_decision"),
            p1_count=getattr(args, f"{role}_p1_count"),
            p2_count=getattr(args, f"{role}_p2_count"),
            p3_count=getattr(args, f"{role}_p3_count"),
        )
        for role, _option_prefix in _REVIEW_ARGUMENT_DOMAIN
    )
    return M6ReviewFinalizationRequest(command=command, reviews=reviews)


def _parse_invocation(
    argv: Sequence[str] | None,
) -> M6CommandRequest | M6ReviewFinalizationRequest:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if "--action" in arguments:
        return _parse_review_finalization_request(arguments)
    return _parse_request(arguments)


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


def _rejection_output(
    code: str,
    *,
    failed: bool = False,
    failure_marker: str | None = None,
) -> bytes:
    if code not in _TRUSTED_CODES:
        code = "unexpected_failure"
    payload = {
        "reason_code": code,
        "schema_version": M6_OFFICIAL_STATUS_SCHEMA_VERSION,
        "status": "failure" if failed else "rejected",
    }
    if failure_marker is not None:
        if not failed:
            raise ValueError("only failures may expose a safe marker path")
        marker = PureWindowsPath(failure_marker)
        parts = Path(failure_marker).parts
        if (
            marker.is_absolute()
            or len(parts) != 4
            or parts[:2] != ("outputs", "m6")
            or _RUN_NAME.fullmatch(parts[2]) is None
            or parts[3] != "TERMINAL_FAILURE"
        ):
            raise ValueError("failure marker path is not allowlisted")
        payload["failure_marker"] = failure_marker
    return _canonical_json_bytes(payload)


def _safe_result_relative(value: Path) -> str:
    if not isinstance(value, Path):
        raise M6OfficialCommandError(
            "result_contract_failed",
            "result path must be a Path",
        )
    text = value.as_posix()
    windows = PureWindowsPath(text)
    if (
        value.is_absolute()
        or windows.is_absolute()
        or len(value.parts) != 3
        or value.parts[:2] != ("outputs", "m6")
        or _RUN_NAME.fullmatch(value.parts[2]) is None
        or text != f"outputs/m6/{value.parts[2]}"
    ):
        raise M6OfficialCommandError(
            "result_contract_failed",
            "result path lies outside outputs/m6",
        )
    return text


def _success_output(result: _CommandResult) -> bytes:
    if result.mode not in M6_OFFICIAL_MODES:
        raise M6OfficialCommandError(
            "result_contract_failed",
            "result mode is invalid",
        )
    counts = _safe_integer_status_mapping(
        result.aggregate_counts,
        name="aggregate_counts",
    )
    durations = _safe_integer_status_mapping(
        result.stage_durations_ms,
        name="stage_durations_ms",
    )
    return _canonical_json_bytes(
        {
            "aggregate_counts": counts,
            "mode": result.mode,
            "profile": M6_OFFICIAL_PROFILE,
            "result_path": _safe_result_relative(result.result_relative),
            "schema_version": M6_OFFICIAL_STATUS_SCHEMA_VERSION,
            "stage_durations_ms": durations,
            "status": "success",
        }
    )


def _awaiting_review_output(result: _AwaitingReviewResult) -> bytes:
    if (
        result.mode != "official"
        or _SHA256.fullmatch(result.evidence_catalog_sha256) is None
        or _SHA256.fullmatch(result.mechanical_verification_sha256) is None
    ):
        raise M6OfficialCommandError(
            "result_contract_failed",
            "awaiting-review result binding is invalid",
        )
    return _canonical_json_bytes(
        {
            "mode": result.mode,
            "profile": M6_OFFICIAL_PROFILE,
            "result_path": _safe_result_relative(result.result_relative),
            "schema_version": M6_OFFICIAL_STATUS_SCHEMA_VERSION,
            "status": "awaiting_review",
        }
    )


def _safe_integer_status_mapping(
    value: Mapping[str, int],
    *,
    name: str,
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise M6OfficialCommandError(
            "result_contract_failed",
            f"{name} must be an aggregate mapping",
        )
    normalized = dict(value)
    if (
        not normalized
        or tuple(normalized) != tuple(sorted(normalized))
        or any(
            _SAFE_COMPONENT.fullmatch(key) is None
            or isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            for key, item in normalized.items()
        )
    ):
        raise M6OfficialCommandError(
            "result_contract_failed",
            f"{name} is not a safe nonnegative integer mapping",
        )
    return normalized


def _failure_code(exc: BaseException) -> str:
    if isinstance(exc, M6OfficialCommandError) and exc.code in _TRUSTED_CODES:
        return exc.code
    if isinstance(exc, TerminalBoundaryError) and exc.code in _TRUSTED_CODES:
        return exc.code
    return "unexpected_failure"


def _dispatch(
    request: M6CommandRequest | M6ReviewFinalizationRequest,
    holder: _RunHolder,
) -> _CommandResult | _PreparedOfficialRun | _AwaitingReviewResult:
    if type(request) is M6ReviewFinalizationRequest:
        return finalize_m6_review(request, holder)
    return prepare_m6_official_run(
        request,
        holder,
        eligibility_executor=(
            run_m6_eligibility_only_execution
            if request.mode == "eligibility_only"
            else None
        ),
        pilot_executor=(
            run_m6_compute_pilot_execution
            if request.mode == "compute_pilot"
            else None
        ),
        official_executor=(
            run_m6_official_execution
            if request.mode == "official"
            else None
        ),
    )


def _valid_bootstrap_repository_receipt(value: object) -> bool:
    if type(value) is not tuple or len(value) != 5:
        return False
    commit, tree, paths, source_sha256, uv_lock_sha256 = value
    return (
        isinstance(commit, str)
        and _GIT_OBJECT_ID.fullmatch(commit) is not None
        and isinstance(tree, str)
        and _GIT_OBJECT_ID.fullmatch(tree) is not None
        and type(paths) is tuple
        and bool(paths)
        and paths == tuple(sorted(set(paths)))
        and all(isinstance(path, str) and path for path in paths)
        and isinstance(source_sha256, str)
        and _SHA256.fullmatch(source_sha256) is not None
        and isinstance(uv_lock_sha256, str)
        and _SHA256.fullmatch(uv_lock_sha256) is not None
    )


def _validate_bootstrap_runtime(
    context: _M6BootstrapContext,
    *,
    issuing: bool,
) -> None:
    def reject(message: str) -> None:
        raise M6OfficialCommandError("runtime_mismatch", message)

    if (
        type(context) is not _M6BootstrapContext
        or context._factory_sentinel is not _BOOTSTRAP_SENTINEL
        or not callable(context.status_sink)
        or not callable(context.status_failure_sink)
        or type(context.initial_sys_path) is not tuple
        or not _valid_bootstrap_repository_receipt(context.repository_receipt)
    ):
        reject("official M6 requires a verifier-issued bootstrap context")
    try:
        root_lexical = Path(os.path.abspath(os.fspath(context.project_root)))
        root = root_lexical.resolve(strict=True)
        site_lexical = Path(os.path.abspath(os.fspath(context.site_packages)))
        site_packages = site_lexical.resolve(strict=True)
        pycache_lexical = Path(
            os.path.abspath(os.fspath(context.pycache_prefix))
        )
        pycache_prefix = pycache_lexical.resolve(strict=True)
        script = root / M6_OFFICIAL_DIRECT_COMMAND[-1]
        script_metadata = script.lstat()
        expected_python = root / M6_OFFICIAL_DIRECT_COMMAND[0]
        initial = tuple(context.initial_sys_path)
        base_prefix = Path(os.path.abspath(sys.base_prefix)).resolve(strict=True)
        pycache_entries = tuple(pycache_prefix.iterdir())
    except (OSError, TypeError, ValueError) as exc:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "official M6 bootstrap paths are unavailable",
        ) from exc
    expected_site = (
        root
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if (
        root_lexical != root
        or site_lexical != site_packages
        or pycache_lexical != pycache_prefix
        or not root.is_dir()
        or not site_packages.is_dir()
        or not pycache_prefix.is_dir()
        or site_packages != expected_site
        or pycache_entries
        or not stat.S_ISREG(script_metadata.st_mode)
        or script.resolve(strict=True) != script
        or Path(os.path.abspath(sys.argv[0])) != script
        or sys.argv[0] != M6_OFFICIAL_DIRECT_COMMAND[-1]
        or Path(os.path.abspath(os.getcwd())).resolve(strict=True) != root
        or Path(os.path.abspath(sys.executable)) != expected_python
        or not expected_python.exists()
        or sys.pycache_prefix != os.fspath(pycache_prefix)
    ):
        reject("official M6 was not started by the exact repository bootstrap")
    flags = sys.flags
    if (
        flags.isolated != 1
        or flags.no_site != 1
        or flags.dont_write_bytecode != 1
        or flags.no_user_site != 1
        or not flags.safe_path
        or flags.inspect != 0
        or flags.interactive != 0
        or sys.dont_write_bytecode is not True
        or "sitecustomize" in sys.modules
        or "usercustomize" in sys.modules
        or (issuing and "site" in sys.modules)
    ):
        reject("official M6 isolation/no-site/no-bytecode flags are not active")
    if (
        not initial
        or any(
            not isinstance(entry, str)
            or not entry
            or not Path(entry).is_absolute()
            for entry in initial
        )
        or type(context.site_packages_enabled) is not bool
        or tuple(sys.path)
        != (
            (*initial, os.fspath(site_packages))
            if context.site_packages_enabled
            else initial
        )
    ):
        reject("official M6 sys.path differs from the bootstrap allowlist")
    for entry in initial:
        try:
            Path(os.path.abspath(entry)).relative_to(base_prefix)
        except ValueError as exc:
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "official M6 inherited a non-stdlib import path",
            ) from exc


def _issue_m6_bootstrap_context(
    project_root: Path,
    site_packages: Path,
    pycache_prefix: Path,
    initial_sys_path: tuple[str, ...],
    repository_receipt: tuple[object, ...],
    status_sink: Callable[[bytes, bool], bool],
    status_failure_sink: Callable[[Callable[[str], bool]], bool],
) -> _M6BootstrapContext:
    context = _M6BootstrapContext(
        project_root=project_root,
        site_packages=site_packages,
        pycache_prefix=pycache_prefix,
        initial_sys_path=initial_sys_path,
        repository_receipt=repository_receipt,
        status_sink=status_sink,
        status_failure_sink=status_failure_sink,
        _factory_sentinel=_BOOTSTRAP_SENTINEL,
    )
    _validate_bootstrap_runtime(context, issuing=True)
    return context


def _activate_bootstrap_context(
    context: _M6BootstrapContext | None,
) -> _M6BootstrapContext:
    if (
        type(context) is not _M6BootstrapContext
        or context._factory_sentinel is not _BOOTSTRAP_SENTINEL
        or context.consumed
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "official M6 requires one fresh bootstrap context",
        )
    context.consumed = True
    return context


def _require_active_bootstrap_context() -> _M6BootstrapContext:
    context = _ACTIVE_BOOTSTRAP_CONTEXT
    if (
        type(context) is not _M6BootstrapContext
        or context._factory_sentinel is not _BOOTSTRAP_SENTINEL
        or context.consumed is not True
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "official M6 preflight requires the active bootstrap context",
        )
    _validate_bootstrap_runtime(context, issuing=False)
    return context


def _enable_active_bootstrap_site_packages() -> _M6BootstrapContext:
    context = _require_active_bootstrap_context()
    if context.site_packages_enabled:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "official M6 site-packages were already enabled",
        )
    sys.path[:] = [
        *context.initial_sys_path,
        os.fspath(context.site_packages),
    ]
    context.site_packages_enabled = True
    _validate_bootstrap_runtime(context, issuing=False)
    return context


def _emit_status(
    terminal: TerminalStatus | None,
    payload: bytes,
    *,
    error: bool,
) -> bool:
    context = _ACTIVE_BOOTSTRAP_CONTEXT
    if context is not None:
        if terminal is not None:
            terminal.close_best_effort()
        try:
            return context.status_sink(payload, error) is True
        except BaseException:
            return False
    if terminal is None:
        stream = sys.stderr if error else sys.stdout
        try:
            stream.write(payload.decode("ascii"))
            stream.flush()
            return True
        except (OSError, UnicodeError):
            return False
    descriptor = terminal.stderr_fd if error else terminal.stdout_fd
    try:
        write_all(descriptor, payload)
        return True
    except OSError:
        return False
    finally:
        terminal.close_best_effort()


def _main_with_active_bootstrap(argv: Sequence[str] | None = None) -> int:
    """Capture native/Python output before constructing or invoking argparse."""

    holder = _RunHolder()
    context = _ACTIVE_BOOTSTRAP_CONTEXT
    if (
        type(context) is not _M6BootstrapContext
        or context._factory_sentinel is not _BOOTSTRAP_SENTINEL
        or context.consumed is not True
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "terminal invalidation requires the active bootstrap context",
        )

    def invalidate_status_failure(reason_code: str) -> bool:
        return _invalidate_holder_store_after_status_failure(
            holder.store,
            reason_code,
        )

    if context.status_failure_sink(invalidate_status_failure) is not True:
        raise M6OfficialCommandError(
            "terminal_capture_failed",
            "the bootstrap rejected the terminal-failure invalidator",
        )
    try:
        captured = capture_terminal(
            lambda: _dispatch(_parse_invocation(argv), holder),
            terminal_commit=_finalize_and_terminalize,
            seal_terminal=True,
        )
    except TerminalizedFailure as exc:
        code = _failure_code(exc.primary)
        failure_marker = _fail_store(holder.store, code)
        if code != "review_rejected":
            _persist_failure_diagnostic(
                holder.store,
                code,
                exc.primary,
                exc.transcript,
                redact_transcript=holder.outcome_started,
            )
        _emit_status(
            exc.terminal_status,
            _rejection_output(
                code,
                failed=holder.store is not None,
                failure_marker=failure_marker,
            ),
            error=True,
        )
        return 1
    except BaseException as exc:
        code = _failure_code(exc)
        failure_marker = _fail_store(holder.store, code)
        if code != "review_rejected":
            _persist_failure_diagnostic(
                holder.store,
                code,
                exc,
                b"",
                redact_transcript=holder.outcome_started,
            )
        _emit_status(
            None,
            _rejection_output(
                code,
                failed=holder.store is not None,
                failure_marker=failure_marker,
            ),
            error=True,
        )
        return 1
    emitted = _emit_status(
        captured.terminal_status,
        (
            _prepared_success_payload(captured.value)
            if isinstance(captured.value, _PreparedOfficialRun)
            else (
                _awaiting_review_output(captured.value)
                if isinstance(captured.value, _AwaitingReviewResult)
                else _success_output(captured.value)
            )
        ),
        error=False,
    )
    if not emitted:
        _invalidate_holder_store_after_status_failure(
            holder.store,
            "terminal_capture_failed",
        )
        return 1
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    _bootstrap_context: _M6BootstrapContext | None = None,
) -> int:
    """Run only inside the exact one-shot stdlib bootstrap boundary."""

    global _ACTIVE_BOOTSTRAP_CONTEXT
    try:
        context = _activate_bootstrap_context(_bootstrap_context)
    except BaseException:
        return 1
    with _BOOTSTRAP_CONTEXT_LOCK:
        if _ACTIVE_BOOTSTRAP_CONTEXT is not None:
            return 1
        _ACTIVE_BOOTSTRAP_CONTEXT = context
        try:
            return _main_with_active_bootstrap(argv)
        finally:
            _ACTIVE_BOOTSTRAP_CONTEXT = None


def _trusted_git_binary() -> str:
    try:
        metadata = _GIT_BINARY.lstat()
        resolved = _GIT_BINARY.resolve(strict=True)
    except OSError as exc:
        raise M6OfficialCommandError(
            "project_root_invalid",
            "the fixed system Git binary is unavailable",
        ) from exc
    if (
        resolved != _GIT_BINARY
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise M6OfficialCommandError(
            "project_root_invalid",
            "the fixed system Git binary is not a trusted root-owned executable",
        )
    return os.fspath(_GIT_BINARY)


def _isolated_git_environment() -> dict[str, str]:
    """Return a fixed Git environment with no caller-controlled executable path."""

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


def _git_process(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            (*_git_prefix(), "-C", os.fspath(root), *arguments),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_isolated_git_environment(),
        )
    except OSError as exc:
        raise M6OfficialCommandError(
            "project_root_invalid",
            "Git cannot inspect the explicit project root",
        ) from exc


def _git_bytes(root: Path, *arguments: str) -> bytes:
    completed = _git_process(root, *arguments)
    if completed.returncode != 0:
        raise M6OfficialCommandError(
            "project_root_invalid",
            "a required read-only Git inspection failed",
        )
    return completed.stdout


def _git_text(root: Path, *arguments: str) -> str:
    try:
        value = _git_bytes(root, *arguments).decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise M6OfficialCommandError(
            "project_root_invalid",
            "Git returned invalid provenance",
        ) from exc
    if not value or "\n" in value or "\r" in value:
        raise M6OfficialCommandError(
            "project_root_invalid",
            "Git returned invalid provenance",
        )
    return value


def _validated_root(candidate: Path) -> Path:
    try:
        lexical = Path(os.path.abspath(os.fspath(candidate)))
        root = lexical.resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise M6OfficialCommandError(
            "project_root_invalid",
            "the explicit project root does not exist",
        ) from exc
    if (
        lexical != root
        or not root.is_dir()
        or not (root / "pyproject.toml").is_file()
        or not (root / ".gitignore").is_file()
    ):
        raise M6OfficialCommandError(
            "project_root_invalid",
            "the explicit project root is not a canonical EvalSim checkout",
        )
    try:
        git_root = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve(
            strict=True
        )
    except OSError as exc:
        raise M6OfficialCommandError(
            "project_root_invalid",
            "the Git worktree root cannot be resolved",
        ) from exc
    if git_root != root:
        raise M6OfficialCommandError(
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
        raise M6OfficialCommandError(
            "argument_error",
            "run name must be one safe lowercase path component",
        )
    return value


def _require_local_opt_in() -> None:
    if os.environ.get(_LOCAL_OPT_IN) != "1":
        raise M6OfficialCommandError(
            "environment_not_enabled",
            "local WOMD use requires the exact opt-in value",
        )


def _live_remote_ref(root: Path, ref: str, code: str) -> str:
    del root
    if ref not in {_CANONICAL_REMOTE_REF, _APPROVED_IMPLEMENTATION_REF}:
        raise ValueError("live remote lookup ref is not allowlisted")
    if code not in {"remote_main_mismatch", "approved_commit_mismatch"}:
        raise ValueError("live remote lookup code is not allowlisted")
    environment = _isolated_git_environment()
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
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            env=environment,
            cwd=os.sep,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise M6OfficialCommandError(
            code,
            "canonical remote ref could not be verified live",
        ) from exc
    try:
        lines = completed.stdout.decode("ascii", errors="strict").splitlines()
        fields = (
            lines[0].split("\t")
            if completed.returncode == 0 and len(lines) == 1
            else []
        )
    except UnicodeDecodeError as exc:
        raise M6OfficialCommandError(
            code,
            "canonical remote returned invalid provenance",
        ) from exc
    if (
        len(fields) != 2
        or fields[1] != ref
        or _GIT_OBJECT_ID.fullmatch(fields[0]) is None
    ):
        raise M6OfficialCommandError(
            code,
            "canonical remote returned an unexpected ref",
        )
    return fields[0]


def _live_main(root: Path) -> str:
    return _live_remote_ref(root, _CANONICAL_REMOTE_REF, "remote_main_mismatch")


def _live_approved_commit(root: Path) -> str:
    return _live_remote_ref(
        root,
        _APPROVED_IMPLEMENTATION_REF,
        "approved_commit_mismatch",
    )


def _assert_entire_index_is_plain(root: Path) -> None:
    encoded = _git_bytes(root, "ls-files", "-v", "-z")
    records = tuple(part for part in encoded.split(b"\0") if part)
    if not records or any(not record.startswith(b"H ") for record in records):
        raise M6OfficialCommandError(
            "source_binding_failed",
            "the tracked index contains assume-unchanged, skip-worktree, or "
            "noncanonical flags",
        )


def _reject_git_replacement_refs(root: Path) -> None:
    if _git_bytes(
        root,
        "for-each-ref",
        "--format=%(refname)",
        "refs/replace/",
    ):
        raise M6OfficialCommandError(
            "source_binding_failed",
            "Git replacement refs are forbidden for official source verification",
        )


def _git_snapshot(
    root: Path,
    *,
    live_lookup: Callable[[Path], str] = _live_main,
    live_approval_lookup: Callable[[Path], str] = _live_approved_commit,
) -> M6GitSnapshot:
    _reject_git_replacement_refs(root)
    _assert_entire_index_is_plain(root)
    if _git_text(root, "remote", "get-url", "origin") != _CANONICAL_REMOTE:
        raise M6OfficialCommandError(
            "git_remote_invalid",
            "origin is not the credential-free canonical remote",
        )
    if _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "-z",
    ):
        raise M6OfficialCommandError(
            "dirty_worktree",
            "official M6 preflight requires a clean worktree and index",
        )
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
        raise M6OfficialCommandError(
            "unpushed_main",
            "HEAD, local main, and origin/main must be one pushed snapshot",
        )
    try:
        approval_object = _git_text(
            root,
            "rev-parse",
            "--verify",
            _APPROVED_IMPLEMENTATION_REF,
        )
        approval_type = _git_text(
            root,
            "cat-file",
            "-t",
            _APPROVED_IMPLEMENTATION_REF,
        )
    except M6OfficialCommandError as exc:
        raise M6OfficialCommandError(
            "approved_commit_mismatch",
            "the fixed local implementation-approval tag is unavailable",
        ) from exc
    if approval_type != "commit" or approval_object != commit:
        raise M6OfficialCommandError(
            "approved_commit_mismatch",
            "the fixed approval ref must be a lightweight tag at HEAD",
        )
    if live_lookup(root) != commit:
        raise M6OfficialCommandError(
            "remote_main_mismatch",
            "live canonical main differs from the local snapshot",
        )
    if live_approval_lookup(root) != commit:
        raise M6OfficialCommandError(
            "approved_commit_mismatch",
            "live canonical approval tag differs from HEAD",
        )
    return M6GitSnapshot(
        commit=commit,
        tree=tree,
        approval_ref=_APPROVED_IMPLEMENTATION_REF,
    )


def _validate_source_path_domain(
    root: Path,
    paths: Sequence[str],
) -> tuple[str, ...]:
    canonical = tuple(paths)
    if (
        not canonical
        or canonical != tuple(sorted(set(canonical)))
        or len(canonical) != len(set(canonical))
    ):
        raise M6OfficialCommandError(
            "source_binding_failed",
            "approved source allowlist is not unique and canonical",
        )
    if not set(_SOURCE_REQUIRED_FILES).issubset(canonical):
        raise M6OfficialCommandError(
            "source_binding_failed",
            "a required repository/source contract file is absent from approved HEAD",
        )
    if not any(path.startswith("evalsim/") for path in canonical) or not any(
        path.startswith("tests/test_m6_") for path in canonical
    ):
        raise M6OfficialCommandError(
            "source_binding_failed",
            "M6 source or tests are absent from approved HEAD",
        )
    for relative_text in canonical:
        relative = Path(relative_text)
        candidate = root.joinpath(*relative.parts)
        try:
            metadata = candidate.lstat()
            if (
                relative.is_absolute()
                or relative.as_posix() != relative_text
                or ".." in relative.parts
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or candidate.resolve(strict=True) != candidate
                or root not in candidate.parents
            ):
                raise OSError("unsafe source node")
        except OSError as exc:
            raise M6OfficialCommandError(
                "source_binding_failed",
                "an approved executable source is linked, missing, or unsafe",
            ) from exc
    return canonical


def _source_path_selected(relative_text: str) -> bool:
    return (
        relative_text in _SOURCE_REQUIRED_FILES
        or relative_text.startswith("evalsim/")
        or (
            relative_text.startswith("tests/test_m6_")
            and relative_text.endswith(".py")
        )
    )


def _approved_source_tree(
    root: Path,
    commit: str,
) -> Mapping[str, str]:
    if _GIT_OBJECT_ID.fullmatch(commit) is None:
        raise ValueError("approved source commit must be a 40-hex object ID")
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
        records = tuple(part for part in encoded.split(b"\0") if part)
        for record in records:
            metadata, raw_path = record.split(b"	", 1)
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
                raise ValueError("noncanonical approved source tree entry")
            entries[relative_text] = object_id
    except (UnicodeDecodeError, ValueError) as exc:
        raise M6OfficialCommandError(
            "source_binding_failed",
            "approved HEAD has an invalid executable source tree",
        ) from exc
    paths = _validate_source_path_domain(root, tuple(sorted(entries)))
    if tuple(entries) != paths:
        raise M6OfficialCommandError(
            "source_binding_failed",
            "approved HEAD source tree ordering is not canonical",
        )
    return MappingProxyType(entries)


def _is_import_executable_name(name: str) -> bool:
    return name.casefold().endswith(_IMPORT_EXECUTABLE_SUFFIXES)


def _reject_unapproved_evalsim_import_artifacts(
    root: Path,
    approved_paths: Sequence[str],
) -> None:
    """Reject ignored/untracked code that could alter later EvalSim imports."""

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
        if _stat_identity(before) != _stat_identity(after):
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
        if _stat_identity(root_before) != _stat_identity(root_after):
            raise OSError("root changed during import-artifact scan")
    except (OSError, ValueError) as exc:
        raise M6OfficialCommandError(
            "source_binding_failed",
            "an ignored or untracked executable can alter EvalSim imports",
        ) from exc


def _assert_source_index_is_plain(
    root: Path,
    approved_paths: Sequence[str],
) -> None:
    encoded = _git_bytes(
        root,
        "ls-files",
        "-v",
        "-z",
    )
    indexed: list[str] = []
    try:
        for record in (part for part in encoded.split(b"\0") if part):
            if not record.startswith(b"H "):
                raise ValueError("source index flags are not plain")
            relative_text = record[2:].decode("utf-8", errors="strict")
            if _source_path_selected(relative_text):
                indexed.append(relative_text)
    except (UnicodeDecodeError, ValueError) as exc:
        raise M6OfficialCommandError(
            "source_binding_failed",
            "assume-unchanged, skip-worktree, or noncanonical source index flags exist",
        ) from exc
    if tuple(indexed) != tuple(approved_paths):
        raise M6OfficialCommandError(
            "source_binding_failed",
            "the source index domain differs from approved HEAD",
        )


def _tracked_source_allowlist(root: Path) -> tuple[str, ...]:
    commit = _git_text(root, "rev-parse", "--verify", "HEAD^{commit}")
    return tuple(_approved_source_tree(root, commit))


def _approved_source_catalog(
    root: Path,
    commit: str,
) -> tuple[tuple[str, ...], tuple[_GuardedFileSnapshot, ...]]:
    tree = _approved_source_tree(root, commit)
    paths = tuple(tree)
    _assert_source_index_is_plain(root, paths)
    _reject_unapproved_evalsim_import_artifacts(root, paths)
    catalog = _source_catalog(root, paths)
    for item in catalog:
        blob = _git_bytes(
            root,
            "cat-file",
            "blob",
            f"{commit}:{item.relative_path}",
        )
        if hashlib.sha256(blob).hexdigest() != item.sha256:
            raise M6OfficialCommandError(
                "source_binding_failed",
                "a guarded source file differs from its approved HEAD blob",
            )
    return paths, catalog


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
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


def _guarded_file_snapshot(
    root: Path,
    relative_text: str,
    *,
    require_single_link: bool = True,
) -> _GuardedFileSnapshot:
    if type(require_single_link) is not bool:
        raise TypeError("require_single_link must be an exact bool")
    relative = Path(relative_text)
    if (
        not isinstance(relative_text, str)
        or relative.is_absolute()
        or relative.as_posix() != relative_text
        or ".." in relative.parts
    ):
        raise M6OfficialCommandError(
            "source_binding_failed",
            "a source path is not canonical and relative",
        )
    path = root.joinpath(*relative.parts)
    descriptor = -1
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or (require_single_link and before.st_nlink != 1)
            or root not in path.parents
            or path.resolve(strict=True) != path
        ):
            raise OSError("unsafe source node")
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            after_read = os.fstat(handle.fileno())
        after_path = path.lstat()
        identities = (
            _stat_identity(before),
            _stat_identity(opened),
            _stat_identity(after_read),
            _stat_identity(after_path),
        )
        if len(set(identities)) != 1:
            raise OSError("source identity changed during guarded read")
    except (OSError, UnicodeError) as exc:
        raise M6OfficialCommandError(
            "source_binding_failed",
            "a source file changed or failed guarded no-follow hashing",
        ) from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    return _GuardedFileSnapshot(
        relative_path=relative_text,
        sha256=digest.hexdigest(),
        identity=identities[0],
    )


def _source_catalog(
    root: Path,
    paths: Sequence[str],
) -> tuple[_GuardedFileSnapshot, ...]:
    normalized = tuple(paths)
    if normalized != tuple(sorted(set(normalized))):
        raise ValueError("source paths must be unique and sorted")
    return tuple(_guarded_file_snapshot(root, path) for path in normalized)


def _source_fingerprint_from_catalog(
    catalog: Sequence[_GuardedFileSnapshot],
) -> str:
    digest = hashlib.sha256(b"evalsim-m6-executable-source-v1\0")
    for item in catalog:
        encoded = item.relative_path.encode("utf-8", errors="strict")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(item.sha256))
    return digest.hexdigest()


def _source_fingerprint(root: Path, paths: Sequence[str]) -> str:
    return _source_fingerprint_from_catalog(_source_catalog(root, paths))


def _repository_context_sha256(
    git: M6GitSnapshot,
    source_paths: Sequence[str],
    source_sha256: str,
    uv_lock_sha256: str,
) -> str:
    return hashlib.sha256(
        b"evalsim-m6-repository-preflight-v1\0"
        + _canonical_json_bytes(
            {
                "approval_ref": git.approval_ref,
                "commit": git.commit,
                "source_paths": list(source_paths),
                "source_sha256": source_sha256,
                "tree": git.tree,
                "uv_lock_sha256": uv_lock_sha256,
            }
        )
    ).hexdigest()


def _expected_evalsim_source_paths(module_name: str) -> frozenset[str]:
    parts = tuple(module_name.split("."))
    if (
        not parts
        or parts[0] != "evalsim"
        or any(not part or not part.isidentifier() for part in parts)
    ):
        raise M6OfficialCommandError(
            "source_binding_failed",
            "a loaded EvalSim module has a noncanonical import name",
        )
    stem = Path(*parts)
    paths = {
        stem.with_suffix(".py").as_posix(),
        (stem / "__init__.py").as_posix(),
    }
    paths.update(
        Path(os.fspath(stem) + suffix).as_posix()
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )
    return frozenset(paths)


def _validate_loaded_evalsim_modules(
    root: Path,
    allowed_paths: Sequence[str],
    *,
    loaded_modules: Mapping[str, object] | None = None,
) -> None:
    allowed = set(allowed_paths)
    if any(not isinstance(path, str) for path in allowed):
        raise ValueError("allowed source paths must be strings")
    modules = sys.modules if loaded_modules is None else loaded_modules
    for name, module in tuple(modules.items()):
        if name != "evalsim" and not name.startswith("evalsim."):
            continue
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str):
            raise M6OfficialCommandError(
                "source_binding_failed",
                "a loaded EvalSim module has invalid source provenance",
            )
        try:
            declared = Path(raw)
            lexical = Path(os.path.abspath(raw))
            actual = lexical.resolve(strict=True)
            relative = actual.relative_to(root).as_posix()
        except (OSError, ValueError) as exc:
            raise M6OfficialCommandError(
                "source_binding_failed",
                "a loaded EvalSim module lies outside the canonical checkout",
            ) from exc
        if (
            not declared.is_absolute()
            or declared != lexical
            or lexical != actual
        ):
            raise M6OfficialCommandError(
                "source_binding_failed",
                "a loaded EvalSim module uses a linked or noncanonical origin",
            )
        if (
            relative not in allowed
            or relative not in _expected_evalsim_source_paths(name)
        ):
            raise M6OfficialCommandError(
                "source_binding_failed",
                "a loaded EvalSim module does not match its tracked source path",
            )
        try:
            _guarded_file_snapshot(root, relative)
        except M6OfficialCommandError as exc:
            raise M6OfficialCommandError(
                "source_binding_failed",
                "a loaded EvalSim module failed guarded source verification",
            ) from exc


def _resolve_shard_inventory(root: Path, candidate: Path) -> tuple[Path, ...]:
    try:
        lexical = Path(os.path.abspath(os.fspath(candidate)))
        data_dir = lexical.resolve(strict=True)
        expected = (root / _DEFAULT_DATA_RELATIVE).resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise M6OfficialCommandError(
            "data_directory_invalid",
            "the exact local validation directory does not exist",
        ) from exc
    if lexical != data_dir or data_dir != expected or not data_dir.is_dir():
        raise M6OfficialCommandError(
            "data_directory_invalid",
            "the local validation directory differs from the frozen project path",
        )
    inventory: list[Path] = []
    children = tuple(data_dir.iterdir())
    for suffix in _SHARD_SUFFIXES:
        matches = tuple(path for path in children if path.name.endswith(suffix))
        if len(matches) != 1:
            raise M6OfficialCommandError(
                "shard_set_invalid",
                "each frozen shard suffix must resolve exactly once",
            )
        path = matches[0]
        try:
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or path.resolve(strict=True) != path
                or path.parent != data_dir
            ):
                raise OSError("unsafe shard node")
        except OSError as exc:
            raise M6OfficialCommandError(
                "shard_set_invalid",
                "a frozen shard path is linked, missing, or unsafe",
            ) from exc
        inventory.append(path)
    return tuple(inventory)


def _shard_identities(paths: Sequence[Path]) -> tuple[_NodeIdentity, ...]:
    identities: list[_NodeIdentity] = []
    for path in paths:
        try:
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or path.resolve(strict=True) != path
            ):
                raise OSError("unsafe shard node")
        except OSError as exc:
            raise M6OfficialCommandError(
                "shard_set_invalid",
                "a frozen shard identity could not be retained",
            ) from exc
        identities.append(
            _NodeIdentity(
                relative_name=path.name,
                identity=_stat_identity(metadata),
            )
        )
    return tuple(identities)


def _git_ignored(root: Path, relative: Path) -> bool:
    return (
        _git_process(
            root,
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            relative.as_posix(),
        ).returncode
        == 0
    )


def _accepted_m4_snapshot_presence(root: Path, candidate: Path) -> Path:
    try:
        raw = candidate if candidate.is_absolute() else root / candidate
        lexical = Path(os.path.abspath(os.fspath(raw)))
        run_dir = lexical.resolve(strict=True)
        allowed = (root / "outputs" / "m4").resolve(strict=True)
        relative = run_dir.relative_to(allowed)
    except (OSError, TypeError, ValueError) as exc:
        raise M6OfficialCommandError(
            "accepted_m4_snapshot_invalid",
            "the explicit accepted M4 snapshot path is invalid",
        ) from exc
    if (
        lexical != run_dir
        or run_dir == allowed
        or not run_dir.is_dir()
        or not relative.parts
        or any(_SAFE_COMPONENT.fullmatch(part) is None for part in relative.parts)
    ):
        raise M6OfficialCommandError(
            "accepted_m4_snapshot_invalid",
            "the accepted M4 snapshot is linked or noncanonical",
        )
    project_relative = run_dir.relative_to(root)
    if not _git_ignored(root, project_relative):
        raise M6OfficialCommandError(
            "accepted_m4_snapshot_invalid",
            "the accepted M4 snapshot is not ignored by Git",
        )
    if _git_bytes(root, "ls-files", "-z", "--", project_relative.as_posix()):
        raise M6OfficialCommandError(
            "accepted_m4_snapshot_invalid",
            "the accepted M4 snapshot contains tracked files",
        )
    if _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored=no",
        "--",
        project_relative.as_posix(),
    ):
        raise M6OfficialCommandError(
            "accepted_m4_snapshot_invalid",
            "the accepted M4 snapshot is visible to Git",
        )
    for relative_text in _M4_REQUIRED_ARTIFACTS:
        path = run_dir.joinpath(*Path(relative_text).parts)
        try:
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or path.resolve(strict=True) != path
                or run_dir not in path.parents
                or (
                    relative_text == "terminal-output.bin"
                    and metadata.st_size != 0
                )
            ):
                raise OSError("unsafe M4 artifact")
        except OSError as exc:
            raise M6OfficialCommandError(
                "accepted_m4_snapshot_invalid",
                "the accepted M4 snapshot is incomplete or unsafe",
            ) from exc
    return run_dir


def _runtime_allowlist(
    *,
    version_resolver: Callable[[str], str] | None = None,
    python_version_resolver: Callable[[], str] | None = None,
) -> Mapping[str, str]:
    resolver = (
        importlib.metadata.version
        if version_resolver is None
        else version_resolver
    )
    python_resolver = (
        platform.python_version
        if python_version_resolver is None
        else python_version_resolver
    )
    observed = {"python": python_resolver()}
    try:
        observed.update(
            {name: resolver(name) for name in _EXPECTED_RUNTIME_VERSIONS}
        )
    except (importlib.metadata.PackageNotFoundError, KeyError) as exc:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a pinned local runtime distribution is unavailable",
        ) from exc
    expected = {
        "python": _EXPECTED_PYTHON_VERSION,
        **dict(_EXPECTED_RUNTIME_VERSIONS),
    }
    if observed != expected:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "the installed runtime differs from the pinned M6 allowlist",
        )
    return MappingProxyType(observed)


def _runtime_module_snapshot(
    root: Path,
    module: object,
) -> _GuardedFileSnapshot:
    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a pinned runtime module has no concrete origin",
        )
    try:
        declared = Path(raw)
        lexical = Path(os.path.abspath(raw))
        actual = lexical.resolve(strict=True)
        relative = actual.relative_to(root).as_posix()
    except (OSError, ValueError) as exc:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a pinned runtime module lies outside the local environment",
        ) from exc
    if (
        not declared.is_absolute()
        or declared != lexical
        or lexical != actual
        or Path(relative).parts[:1] != (".venv",)
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a pinned runtime module is not from the canonical local .venv",
        )
    try:
        return _guarded_file_snapshot(
            root,
            relative,
            require_single_link=False,
        )
    except M6OfficialCommandError as exc:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a pinned runtime module failed guarded origin verification",
        ) from exc


def _runtime_executable_file(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(_RUNTIME_EXECUTABLE_SUFFIXES) or ".so." in lowered


def _runtime_package_roots(
    root: Path,
    module: object,
    origin: _GuardedFileSnapshot,
) -> tuple[str, ...]:
    origin_path = root.joinpath(*Path(origin.relative_path).parts)
    raw_paths = getattr(module, "__path__", None)
    if raw_paths is None:
        candidates = (
            origin_path.parent
            if origin_path.name.startswith("__init__.")
            else origin_path,
        )
    else:
        try:
            values = tuple(raw_paths)
        except TypeError as exc:
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "a pinned runtime package has an invalid search path",
            ) from exc
        if not values or any(not isinstance(value, str) for value in values):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "a pinned runtime package has an invalid search path",
            )
        candidates = tuple(Path(os.path.abspath(value)) for value in values)
        if any(
            not Path(value).is_absolute()
            or Path(value) != candidate
            for value, candidate in zip(values, candidates, strict=True)
        ):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "a pinned runtime package search path is noncanonical",
            )

    normalized: list[str] = []
    for lexical in candidates:
        try:
            metadata = lexical.lstat()
        except FileNotFoundError:
            # Some importers declare optional package paths that are absent in a
            # concrete wheel. The complete RECORD inventory below remains the
            # authority for installed executable ownership.
            continue
        except OSError as exc:
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "a pinned runtime package root could not be inspected",
            ) from exc
        try:
            actual = lexical.resolve(strict=True)
            relative = actual.relative_to(root).as_posix()
        except (OSError, ValueError) as exc:
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "a pinned runtime package root lies outside the local environment",
            ) from exc
        if (
            lexical != actual
            or Path(relative).parts[:1] != (".venv",)
            or not (
                stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISREG(metadata.st_mode)
            )
        ):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "a pinned runtime package root is linked or noncanonical",
            )
        normalized.append(relative)
    roots = tuple(sorted(set(normalized)))
    if len(roots) != len(normalized) or not any(
        origin_path == root.joinpath(*Path(relative).parts)
        or root.joinpath(*Path(relative).parts) in origin_path.parents
        for relative in roots
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a pinned runtime origin is outside its imported package roots",
        )
    return roots


def _runtime_tree_paths(root: Path, relative_root: str) -> tuple[str, ...]:
    start = root.joinpath(*Path(relative_root).parts)

    def visit(path: Path) -> tuple[str, ...]:
        try:
            before = path.lstat()
            actual = path.resolve(strict=True)
            relative = actual.relative_to(root).as_posix()
        except (OSError, ValueError) as exc:
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "a runtime package node lies outside the local environment",
            ) from exc
        if actual != path or Path(relative).parts[:1] != (".venv",):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "a runtime package node is linked or noncanonical",
            )
        if stat.S_ISREG(before.st_mode):
            return (relative,) if _runtime_executable_file(path.name) else ()
        if not stat.S_ISDIR(before.st_mode):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "a runtime package contains an unsupported filesystem node",
            )
        try:
            with os.scandir(path) as iterator:
                children = tuple(sorted((entry.name for entry in iterator)))
            if len(children) != len(set(children)):
                raise OSError("duplicate runtime package entry")
            paths = tuple(
                relative_path
                for name in children
                for relative_path in visit(path / name)
            )
            after = path.lstat()
        except OSError as exc:
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "a runtime package changed during guarded traversal",
            ) from exc
        if _stat_identity(before) != _stat_identity(after):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "a runtime package changed during guarded traversal",
            )
        return paths

    return visit(start)


def _runtime_package_fingerprint(
    distribution_name: str,
    module_name: str,
    roots: Sequence[str],
    files: Sequence[_GuardedFileSnapshot],
    metadata_files: Sequence[_GuardedFileSnapshot],
) -> str:
    return hashlib.sha256(
        b"evalsim-m6-runtime-package-v1\0"
        + _canonical_json_bytes(
            {
                "distribution_name": distribution_name,
                "files": [
                    {"path": item.relative_path, "sha256": item.sha256}
                    for item in files
                ],
                "metadata_files": [
                    {"path": item.relative_path, "sha256": item.sha256}
                    for item in metadata_files
                ],
                "module_name": module_name,
                "roots": list(roots),
            }
        )
    ).hexdigest()


def _runtime_record_hash_and_size(entry: object) -> tuple[str, int]:
    file_hash = getattr(entry, "hash", None)
    size = getattr(entry, "size", None)
    mode = getattr(file_hash, "mode", None)
    value = getattr(file_hash, "value", None)
    if (
        mode != "sha256"
        or not isinstance(value, str)
        or not value
        or type(size) is not int
        or size < 0
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a runtime RECORD entry lacks a SHA-256 hash or exact size",
        )
    try:
        decoded = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a runtime RECORD entry has an invalid SHA-256 hash",
        ) from exc
    if len(decoded) != hashlib.sha256().digest_size:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a runtime RECORD entry has an invalid SHA-256 hash",
        )
    return decoded.hex(), size


def _runtime_distribution_inventory(
    root: Path,
    distribution_name: str,
    distribution: object,
) -> tuple[
    tuple[_GuardedFileSnapshot, ...],
    tuple[_GuardedFileSnapshot, ...],
]:
    try:
        entries = tuple(getattr(distribution, "files"))
    except (AttributeError, TypeError) as exc:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a pinned runtime distribution has no installed RECORD inventory",
        ) from exc
    if not entries:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a pinned runtime distribution has an empty RECORD inventory",
        )

    recorded: dict[str, object] = {}
    metadata_entries: dict[str, tuple[str, object]] = {}
    for entry in entries:
        try:
            record_text = os.fspath(entry)
        except TypeError as exc:
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "a pinned runtime RECORD path is invalid",
            ) from exc
        if not isinstance(record_text, str) or not record_text:
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "a pinned runtime RECORD path is invalid",
            )
        record_path = Path(record_text)
        metadata_kind = (
            record_path.name
            if record_path.name in {"RECORD", "direct_url.json"}
            and record_path.parent.name.endswith(".dist-info")
            else None
        )
        executable = _runtime_executable_file(record_path.name)
        if not executable and metadata_kind is None:
            continue
        try:
            located = Path(os.fspath(distribution.locate_file(entry)))
            lexical = Path(os.path.abspath(os.fspath(located)))
            actual = lexical.resolve(strict=True)
            relative = actual.relative_to(root).as_posix()
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "a runtime RECORD entry lies outside the local environment",
            ) from exc
        if lexical != actual or Path(relative).parts[:1] != (".venv",):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "a runtime RECORD entry is linked or outside the local .venv",
            )
        if executable:
            if relative in recorded:
                raise M6OfficialCommandError(
                    "runtime_mismatch",
                    "a runtime executable has duplicate RECORD ownership",
                )
            recorded[relative] = entry
        if metadata_kind is not None:
            if metadata_kind in metadata_entries:
                raise M6OfficialCommandError(
                    "runtime_mismatch",
                    "a runtime distribution has duplicate provenance metadata",
                )
            metadata_entries[metadata_kind] = (relative, entry)

    if "RECORD" not in metadata_entries or (
        distribution_name == "waymo-waymax"
        and "direct_url.json" not in metadata_entries
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a runtime distribution lacks required RECORD or direct-url metadata",
        )
    install_root = Path(metadata_entries["RECORD"][0]).parent.parent
    owned_roots: set[str] = set()
    for recorded_path in recorded:
        try:
            installed_relative = Path(recorded_path).relative_to(install_root)
        except ValueError:
            continue
        if installed_relative.parts:
            owned_roots.add(
                (install_root / installed_relative.parts[0]).as_posix()
            )
    traversed = {
        path
        for relative_root in owned_roots
        for path in _runtime_tree_paths(root, relative_root)
    }
    if not traversed.issubset(recorded):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a runtime package contains an unrecorded executable",
        )
    paths = tuple(sorted(recorded))
    try:
        files = tuple(
            _guarded_file_snapshot(root, path, require_single_link=False)
            for path in paths
        )
        metadata_files = tuple(
            _guarded_file_snapshot(
                root,
                metadata_entries[kind][0],
                require_single_link=False,
            )
            for kind in sorted(metadata_entries)
        )
    except M6OfficialCommandError as exc:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a runtime distribution failed guarded no-follow hashing",
        ) from exc
    for snapshot in files:
        expected_sha256, expected_size = _runtime_record_hash_and_size(
            recorded[snapshot.relative_path]
        )
        if (
            snapshot.sha256 != expected_sha256
            or snapshot.identity[5] != expected_size
        ):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "a runtime executable differs from its RECORD hash or size",
            )
    for snapshot in metadata_files:
        entry = metadata_entries[Path(snapshot.relative_path).name][1]
        if (
            getattr(entry, "hash", None) is None
            and getattr(entry, "size", None) is None
        ):
            continue
        expected_sha256, expected_size = _runtime_record_hash_and_size(entry)
        if (
            snapshot.sha256 != expected_sha256
            or snapshot.identity[5] != expected_size
        ):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "runtime provenance metadata differs from its RECORD entry",
            )
    return files, metadata_files


def _runtime_package_catalog(
    root: Path,
    distribution_name: str,
    module_name: str,
    module: object,
    origin: _GuardedFileSnapshot,
    distribution: object,
) -> _RuntimePackageCatalog:
    roots = _runtime_package_roots(root, module, origin)
    files, metadata_files = _runtime_distribution_inventory(
        root,
        distribution_name,
        distribution,
    )
    catalog_origins = tuple(
        item for item in files if item.relative_path == origin.relative_path
    )
    if len(catalog_origins) != 1:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a pinned runtime origin is absent from its executable catalog",
        )
    if catalog_origins[0] != origin:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a pinned runtime module origin changed during cataloging",
        )
    return _RuntimePackageCatalog(
        distribution_name=distribution_name,
        module_name=module_name,
        roots=roots,
        files=files,
        metadata_files=metadata_files,
        sha256=_runtime_package_fingerprint(
            distribution_name,
            module_name,
            roots,
            files,
            metadata_files,
        ),
    )


def _validate_runtime_catalog_anchor(
    catalog: Sequence[_RuntimePackageCatalog],
) -> None:
    normalized = tuple(catalog)
    observed = {
        item.distribution_name: item.sha256
        for item in normalized
    }
    expected = dict(_EXPECTED_RUNTIME_CATALOG_SHA256)
    if (
        tuple(observed) != tuple(sorted(_RUNTIME_MODULES))
        or set(expected) != set(_RUNTIME_MODULES)
        or any(
            not isinstance(value, str)
            or _SHA256.fullmatch(value) is None
            for value in expected.values()
        )
        or observed != expected
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a runtime distribution differs from its immutable catalog anchor",
        )


def _runtime_artifact_catalog(
    root: Path,
    modules: Mapping[str, object],
    origins: Mapping[str, _GuardedFileSnapshot],
    distributions: Mapping[str, object],
) -> tuple[_RuntimePackageCatalog, ...]:
    if any(
        set(values) != set(_RUNTIME_MODULES)
        for values in (modules, origins, distributions)
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "the pinned runtime package domain is incomplete",
        )
    catalog = tuple(
        _runtime_package_catalog(
            root,
            distribution_name,
            _RUNTIME_MODULES[distribution_name],
            modules[distribution_name],
            origins[distribution_name],
            distributions[distribution_name],
        )
        for distribution_name in sorted(_RUNTIME_MODULES)
    )
    owners: dict[str, str] = {}
    for package in catalog:
        for file in package.files:
            prior = owners.setdefault(
                file.relative_path,
                package.distribution_name,
            )
            if prior != package.distribution_name:
                raise M6OfficialCommandError(
                    "runtime_mismatch",
                    "a runtime executable has ambiguous distribution ownership",
                )
    keras_owned = {
        file.relative_path
        for package in catalog
        if package.distribution_name == "keras"
        for file in package.files
    }
    for package in catalog:
        declared: set[str] = set()
        for relative_root in package.roots:
            root_files = set(_runtime_tree_paths(root, relative_root))
            parts = Path(relative_root).parts
            is_keras_root = any(
                parts[index : index + 2] == ("site-packages", "keras")
                for index in range(len(parts) - 1)
            )
            if (
                package.distribution_name == "tensorflow"
                and is_keras_root
                and not root_files.issubset(keras_owned)
            ):
                raise M6OfficialCommandError(
                    "runtime_mismatch",
                    "a TensorFlow Keras root is not owned by the pinned Keras RECORD",
                )
            declared.update(root_files)
        if not declared.issubset(owners):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "a declared runtime package root contains an executable without "
                "pinned RECORD ownership",
            )
    _validate_runtime_catalog_anchor(catalog)
    return catalog


def _runtime_catalog_sha256(
    catalog: Sequence[_RuntimePackageCatalog],
) -> str:
    normalized = tuple(catalog)
    if tuple(item.distribution_name for item in normalized) != tuple(
        sorted(_RUNTIME_MODULES)
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "the runtime executable catalog is not canonical",
        )
    return hashlib.sha256(
        b"evalsim-m6-runtime-catalog-v1\0"
        + _canonical_json_bytes(
            {
                item.distribution_name: {
                    "files": [
                        {"path": file.relative_path, "sha256": file.sha256}
                        for file in item.files
                    ],
                    "metadata_files": [
                        {"path": file.relative_path, "sha256": file.sha256}
                        for file in item.metadata_files
                    ],
                    "module_name": item.module_name,
                    "roots": list(item.roots),
                    "sha256": item.sha256,
                }
                for item in normalized
            }
        )
    ).hexdigest()



def _canonical_distribution_name(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "an installed distribution has no canonical name",
        )
    normalized = re.sub(r"[-_.]+", "-", value).lower()
    if (
        _SAFE_COMPONENT.fullmatch(normalized) is None
        or normalized.startswith("-")
        or normalized.endswith("-")
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "an installed distribution has no canonical name",
        )
    return normalized


def _environment_distribution_identity(
    distribution: object,
) -> tuple[str, str]:
    try:
        raw_name = distribution.metadata["Name"]
        version = distribution.version
    except (AttributeError, KeyError, TypeError) as exc:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "installed distribution metadata is incomplete",
        ) from exc
    if not isinstance(version, str) or not version:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "installed distribution metadata is incomplete",
        )
    return _canonical_distribution_name(raw_name), version


def _environment_distribution_fingerprint(
    distribution_name: str,
    version: str,
    files: Sequence[_GuardedFileSnapshot],
) -> str:
    return hashlib.sha256(
        b"evalsim-m6-environment-distribution-v1\0"
        + _canonical_json_bytes(
            {
                "distribution_name": distribution_name,
                "files": [
                    {"path": item.relative_path, "sha256": item.sha256}
                    for item in files
                ],
                "version": version,
            }
        )
    ).hexdigest()


def _environment_tree_paths(root: Path, relative_root: str) -> tuple[str, ...]:
    start = root.joinpath(*Path(relative_root).parts)

    def visit(path: Path) -> tuple[str, ...]:
        try:
            before = path.lstat()
            actual = path.resolve(strict=True)
            relative = actual.relative_to(root).as_posix()
        except (OSError, ValueError) as exc:
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "an environment node lies outside the local runtime",
            ) from exc
        if actual != path or Path(relative).parts[:1] != (".venv",):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "an environment node is linked or noncanonical",
            )
        if stat.S_ISREG(before.st_mode):
            return (relative,)
        if not stat.S_ISDIR(before.st_mode):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "the environment contains an unsupported filesystem node",
            )
        try:
            with os.scandir(path) as iterator:
                children = tuple(sorted(entry.name for entry in iterator))
            paths = tuple(
                item
                for name in children
                for item in visit(path / name)
            )
            after = path.lstat()
        except OSError as exc:
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "the environment changed during guarded traversal",
            ) from exc
        if _stat_identity(before) != _stat_identity(after):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "the environment changed during guarded traversal",
            )
        return paths

    return visit(start)


def _environment_distribution_catalog(
    root: Path,
    distribution: object,
) -> _EnvironmentDistributionCatalog:
    distribution_name, version = _environment_distribution_identity(distribution)
    try:
        entries = tuple(distribution.files)
    except (AttributeError, TypeError) as exc:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "an installed distribution has no RECORD inventory",
        ) from exc
    if not entries:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "an installed distribution has an empty RECORD inventory",
        )
    recorded: dict[str, object] = {}
    for entry in entries:
        try:
            raw = os.fspath(distribution.locate_file(entry))
            lexical = Path(os.path.abspath(raw))
            actual = lexical.resolve(strict=True)
            relative = actual.relative_to(root).as_posix()
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "an installed RECORD file lies outside the local runtime",
            ) from exc
        if lexical != actual or Path(relative).parts[:1] != (".venv",):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "an installed RECORD file is linked or outside the local runtime",
            )
        if relative in recorded:
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "an installed distribution RECORD contains duplicate files",
            )
        recorded[relative] = entry
    try:
        files = tuple(
            _guarded_file_snapshot(root, path, require_single_link=False)
            for path in sorted(recorded)
        )
    except M6OfficialCommandError as exc:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "an installed distribution file failed guarded hashing",
        ) from exc
    for snapshot in files:
        entry = recorded[snapshot.relative_path]
        file_hash = getattr(entry, "hash", None)
        size = getattr(entry, "size", None)
        if file_hash is None and size is None:
            path = Path(os.fspath(entry))
            if path.name != "RECORD" or not path.parent.name.endswith(".dist-info"):
                raise M6OfficialCommandError(
                    "runtime_mismatch",
                    "an installed file lacks a RECORD hash or size",
                )
            continue
        expected_sha256, expected_size = _runtime_record_hash_and_size(entry)
        if (
            snapshot.sha256 != expected_sha256
            or snapshot.identity[5] != expected_size
        ):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "an installed file differs from its RECORD hash or size",
            )
    return _EnvironmentDistributionCatalog(
        distribution_name=distribution_name,
        version=version,
        files=files,
        sha256=_environment_distribution_fingerprint(
            distribution_name,
            version,
            files,
        ),
    )


def _environment_catalog_sha256(
    catalog: Sequence[_EnvironmentDistributionCatalog],
    infrastructure: Sequence[_GuardedFileSnapshot],
) -> str:
    normalized = tuple(catalog)
    normalized_infrastructure = tuple(infrastructure)
    names = tuple(item.distribution_name for item in normalized)
    if names != tuple(sorted(set(names))):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "the complete environment catalog is not canonical",
        )
    infrastructure_paths = tuple(
        item.relative_path for item in normalized_infrastructure
    )
    if infrastructure_paths != _EXPECTED_ENVIRONMENT_INFRASTRUCTURE:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "the environment infrastructure catalog is not canonical",
        )
    return hashlib.sha256(
        b"evalsim-m6-complete-environment-v1\0"
        + _canonical_json_bytes(
            {
                "distributions": {
                    item.distribution_name: {
                        "files": [
                            {"path": file.relative_path, "sha256": file.sha256}
                            for file in item.files
                        ],
                        "sha256": item.sha256,
                        "version": item.version,
                    }
                    for item in normalized
                },
                "uv_runtime_infrastructure": [
                    {"path": item.relative_path, "sha256": item.sha256}
                    for item in normalized_infrastructure
                ],
            }
        )
    ).hexdigest()


def _complete_environment_catalog(
    root: Path,
    distributions_resolver: Callable[[], Sequence[object]],
) -> tuple[
    tuple[_EnvironmentDistributionCatalog, ...],
    tuple[_GuardedFileSnapshot, ...],
]:
    try:
        distributions = tuple(distributions_resolver())
    except Exception as exc:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "the installed distribution domain could not be enumerated",
        ) from exc
    catalogs: list[_EnvironmentDistributionCatalog] = []
    names: set[str] = set()
    for distribution in distributions:
        item = _environment_distribution_catalog(root, distribution)
        if item.distribution_name in names:
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "the installed distribution domain contains duplicate names",
            )
        names.add(item.distribution_name)
        catalogs.append(item)
    catalog = tuple(sorted(catalogs, key=lambda item: item.distribution_name))
    versions = {item.distribution_name: item.version for item in catalog}
    expected_versions = {
        **dict(_EXPECTED_ENVIRONMENT_VERSIONS),
        "evalsim": _EXPECTED_PROJECT_DISTRIBUTION_VERSION,
    }
    if versions != expected_versions:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "the installed package/version domain differs from the frozen runtime",
        )
    owners: dict[str, set[str]] = {}
    for item in catalog:
        for file in item.files:
            owners.setdefault(file.relative_path, set()).add(
                item.distribution_name
            )
    shared_owners = {
        path: frozenset(values)
        for path, values in owners.items()
        if len(values) > 1
    }
    if any(
        _EXPECTED_ENVIRONMENT_SHARED_OWNERS.get(path) != values
        for path, values in shared_owners.items()
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "an installed file has ambiguous distribution ownership",
        )
    try:
        infrastructure = tuple(
            _guarded_file_snapshot(root, path, require_single_link=False)
            for path in _EXPECTED_ENVIRONMENT_INFRASTRUCTURE
        )
    except M6OfficialCommandError as exc:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "the exact uv runtime infrastructure is unavailable",
        ) from exc
    site_files = set(
        _environment_tree_paths(root, _RUNTIME_SITE_PACKAGES_RELATIVE)
    )
    site_prefix = f"{_RUNTIME_SITE_PACKAGES_RELATIVE}/"
    owned_site_files = {
        path for path in owners if path.startswith(site_prefix)
    }
    infrastructure_files = {
        item.relative_path for item in infrastructure
    }
    if site_files != owned_site_files | infrastructure_files:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "site-packages contains an unowned executable or data file",
        )
    if (
        _environment_catalog_sha256(catalog, infrastructure)
        != _EXPECTED_ENVIRONMENT_CATALOG_SHA256
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "the complete environment differs from its immutable catalog anchor",
        )
    return catalog, infrastructure

def _waymax_direct_url_commit(
    distribution: object,
    direct_url: _GuardedFileSnapshot,
    expected_commit: str,
) -> str:
    try:
        encoded = distribution.read_text("direct_url.json")
        if not isinstance(encoded, str) or (
            hashlib.sha256(encoded.encode("utf-8", errors="strict")).hexdigest()
            != direct_url.sha256
        ):
            raise ValueError("direct-url content differs from guarded metadata")
        payload = json.loads(encoded)
        vcs = payload["vcs_info"]
    except (
        AttributeError,
        importlib.metadata.PackageNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "Waymax direct VCS provenance is unavailable",
        ) from exc
    if (
        payload.get("url") != _WAYMAX_REMOTE
        or type(vcs) is not dict
        or vcs.get("vcs") != "git"
        or vcs.get("commit_id") != expected_commit
        or vcs.get("requested_revision") != expected_commit
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "Waymax is not installed from the pinned immutable revision",
        )
    return expected_commit


def _runtime_config_sha256(
    results_module: object,
    versions: Mapping[str, str],
    module_origins: Mapping[str, str],
    module_origin_sha256: Mapping[str, str],
    runtime_catalog: Sequence[_RuntimePackageCatalog],
    environment_catalog: Sequence[_EnvironmentDistributionCatalog],
    environment_infrastructure: Sequence[_GuardedFileSnapshot],
    *,
    backend: str,
    device_class: str,
    waymax_commit: str,
) -> str:
    if sys.dont_write_bytecode is not True:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "the runtime no-bytecode guard is not active",
        )
    fields = {
        "config_version": getattr(results_module, "M6_CONFIG_VERSION"),
        "plan_version": getattr(results_module, "M6_PLAN_VERSION"),
        "result_store_schema_version": getattr(
            results_module,
            "M6_RESULT_STORE_SCHEMA_VERSION",
        ),
        "statistics_schema_version": getattr(
            results_module,
            "M6_STATISTICS_SCHEMA_VERSION",
        ),
        "typed_provenance_schema_version": getattr(
            results_module,
            "M6_TYPED_PROVENANCE_SCHEMA_VERSION",
        ),
    }
    if any(not isinstance(value, str) or not value for value in fields.values()):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "M6 runtime/config schema versions are unavailable",
        )
    return hashlib.sha256(
        b"evalsim-m6-runtime-config-v1\0"
        + _canonical_json_bytes(
            {
                **fields,
                "environment_catalog_sha256": _environment_catalog_sha256(
                    environment_catalog, environment_infrastructure
                ),
                "environment_versions": {
                    **dict(_EXPECTED_ENVIRONMENT_VERSIONS),
                    "evalsim": _EXPECTED_PROJECT_DISTRIBUTION_VERSION,
                },
                "jax_backend": backend,
                "jax_device_class": device_class,
                "python_dont_write_bytecode": True,
                "module_origin_sha256": dict(module_origin_sha256),
                "module_origins": dict(module_origins),
                "runtime_catalog": {
                    item.distribution_name: {
                        "files": [
                            {"path": file.relative_path, "sha256": file.sha256}
                            for file in item.files
                        ],
                        "metadata_files": [
                            {"path": file.relative_path, "sha256": file.sha256}
                            for file in item.metadata_files
                        ],
                        "module_name": item.module_name,
                        "roots": list(item.roots),
                        "sha256": item.sha256,
                    }
                    for item in runtime_catalog
                },
                "runtime_catalog_sha256": _runtime_catalog_sha256(runtime_catalog),
                "versions": dict(versions),
                "waymax_commit": waymax_commit,
            }
        )
    ).hexdigest()


def _observe_runtime(
    root: Path,
    repository: M6RepositoryPreflight,
    results_module: object,
    *,
    version_resolver: Callable[[str], str] | None = None,
    module_importer: Callable[[str], object] | None = None,
    distribution_resolver: Callable[[str], object] | None = None,
    python_version_resolver: Callable[[], str] | None = None,
    environment_distributions_resolver: (
        Callable[[], Sequence[object]] | None
    ) = None,
) -> M6RuntimeObservation:
    if (
        type(repository) is not M6RepositoryPreflight
        or repository._factory_sentinel is not _PREFLIGHT_SENTINEL
        or repository.root != root
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "runtime observation requires the exact repository preflight",
        )
    sys.dont_write_bytecode = True
    observed = dict(
        _runtime_allowlist(
            version_resolver=version_resolver,
            python_version_resolver=python_version_resolver,
        )
    )
    python_version = observed.pop("python")
    importer = importlib.import_module if module_importer is None else module_importer
    distribution = (
        importlib.metadata.distribution
        if distribution_resolver is None
        else distribution_resolver
    )
    environment_distributions = (
        (lambda: tuple(importlib.metadata.distributions()))
        if environment_distributions_resolver is None
        else environment_distributions_resolver
    )
    environment_catalog, environment_infrastructure = (
        _complete_environment_catalog(root, environment_distributions)
    )
    environment_catalog_sha256 = _environment_catalog_sha256(
        environment_catalog, environment_infrastructure
    )
    try:
        distributions = {
            name: distribution(name) for name in _RUNTIME_MODULES
        }
    except Exception as exc:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a pinned runtime distribution could not be resolved",
        ) from exc
    modules: dict[str, object] = {}
    snapshots: dict[str, _GuardedFileSnapshot] = {}
    try:
        for distribution_name, module_name in _RUNTIME_MODULES.items():
            module = importer(module_name)
            modules[distribution_name] = module
            snapshots[distribution_name] = _runtime_module_snapshot(root, module)
    except M6OfficialCommandError:
        raise
    except Exception as exc:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "a pinned optional runtime module could not be imported",
        ) from exc
    jax = modules["jax"]
    try:
        backend = jax.default_backend()
        devices = tuple(jax.devices())
        device_class = (
            "cpu"
            if devices
            and all(getattr(device, "platform", None) == "cpu" for device in devices)
            else "non_cpu"
        )
    except Exception as exc:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "JAX backend/device identity could not be observed",
        ) from exc
    if backend != "cpu" or device_class != "cpu":
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "official M6 requires the pinned CPU JAX backend and devices",
        )
    expected_waymax = getattr(results_module, "WAYMAX_COMMIT", None)
    if (
        not isinstance(expected_waymax, str)
        or _GIT_OBJECT_ID.fullmatch(expected_waymax) is None
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "the pinned EvalSim Waymax commit is unavailable",
        )
    runtime_catalog = _runtime_artifact_catalog(
        root,
        modules,
        snapshots,
        distributions,
    )
    waymax_catalog = next(
        item
        for item in runtime_catalog
        if item.distribution_name == "waymo-waymax"
    )
    direct_url = next(
        item
        for item in waymax_catalog.metadata_files
        if Path(item.relative_path).name == "direct_url.json"
    )
    waymax_commit = _waymax_direct_url_commit(
        distributions["waymo-waymax"], direct_url, expected_waymax
    )
    runtime_catalog_sha256 = _runtime_catalog_sha256(runtime_catalog)
    origins = {
        name: snapshots[name].relative_path for name in _RUNTIME_MODULES
    }
    origin_hashes = {
        name: snapshots[name].sha256 for name in _RUNTIME_MODULES
    }
    config_sha256 = _runtime_config_sha256(
        results_module,
        observed,
        origins,
        origin_hashes,
        runtime_catalog,
        environment_catalog,
        environment_infrastructure,
        backend=backend,
        device_class=device_class,
        waymax_commit=waymax_commit,
    )
    if (
        _runtime_artifact_catalog(
            root,
            modules,
            snapshots,
            distributions,
        )
        != runtime_catalog
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "the runtime executable catalog changed during observation",
        )
    if (
        _complete_environment_catalog(root, environment_distributions)
        != (environment_catalog, environment_infrastructure)
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "the complete environment changed during observation",
        )
    return M6RuntimeObservation(
        versions=observed,
        module_origins=origins,
        python_version=python_version,
        jax_backend=backend,
        jax_device_class=device_class,
        waymax_commit=waymax_commit,
        uv_lock_sha256=repository.uv_lock_sha256,
        runtime_catalog_sha256=runtime_catalog_sha256,
        environment_catalog_sha256=environment_catalog_sha256,
        runtime_config_sha256=config_sha256,
        _runtime_catalog=runtime_catalog,
        _environment_catalog=environment_catalog,
        _environment_infrastructure=environment_infrastructure,
        _factory_sentinel=_RUNTIME_SENTINEL,
    )


def _assert_repository_binding(
    request: M6CommandRequest,
    repository: M6RepositoryPreflight,
) -> None:
    if not isinstance(request, M6CommandRequest):
        raise TypeError("request must be M6CommandRequest")
    if (
        type(repository) is not M6RepositoryPreflight
        or repository._factory_sentinel is not _PREFLIGHT_SENTINEL
    ):
        raise M6OfficialCommandError(
            "source_binding_failed",
            "local preflight requires verifier-issued repository evidence",
        )
    try:
        requested_root = Path(
            os.path.abspath(os.fspath(request.project_root))
        ).resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise M6OfficialCommandError(
            "project_root_invalid",
            "the request root changed after repository preflight",
        ) from exc
    if requested_root != repository.root:
        raise M6OfficialCommandError(
            "project_root_invalid",
            "the request root differs from repository preflight",
        )


def _assert_store_binding(
    request: M6CommandRequest,
    repository: M6RepositoryPreflight,
    store: object,
    *,
    require_pending: bool,
) -> object:
    _assert_repository_binding(request, repository)
    results_module = sys.modules.get("evalsim.results.m6")
    store_type = (
        None
        if results_module is None
        else getattr(results_module, "M6ResultStore", None)
    )
    if store_type is None or type(store) is not store_type:
        raise M6OfficialCommandError(
            "result_contract_failed",
            "local preflight requires one exact M6 result-store writer",
        )
    if (
        store.project_root != repository.root
        or store.run_name != request.run_name
        or store.profile.mode != request.mode
        or store.project_relative_path.as_posix()
        != f"outputs/m6/{request.run_name}"
    ):
        raise M6OfficialCommandError(
            "result_contract_failed",
            "the pending store differs from the exact command request",
        )
    try:
        if require_pending:
            store._assert_pending_capability()
        elif store._phase == "pending":
            store._assert_pending_capability()
        elif store._phase == "awaiting_review":
            store._assert_awaiting_review_capability()
        elif store._phase == "committed":
            store._assert_committed_capability()
        else:
            raise RuntimeError("store phase is not re-verifiable")
    except Exception as exc:
        raise M6OfficialCommandError(
            "result_contract_failed",
            "the exact result store is not in a re-verifiable phase",
        ) from exc
    return results_module


def _verify_accepted_m4(
    root: Path,
    run_dir: Path,
    verifier: Callable[[Path, Path], object] | None,
) -> object:
    if verifier is None:
        from evalsim.sources.m5_m4_reuse import verify_accepted_m4_run

        verifier = verify_accepted_m4_run
    try:
        cohort = verifier(root, run_dir)
        cohort_root = Path(cohort.project_root).resolve(strict=True)
        cohort_run = Path(cohort.run_dir).resolve(strict=True)
        evidence = cohort.evidence
        manifest_sha256 = evidence.manifest_sha256
        provenance_sha256 = evidence.execution_provenance_sha256
    except Exception as exc:
        raise M6OfficialCommandError(
            "accepted_m4_snapshot_invalid",
            "the accepted M4 snapshot failed exact reuse verification",
        ) from exc
    if (
        cohort_root != root
        or cohort_run != run_dir
        or _SHA256.fullmatch(manifest_sha256) is None
        or _SHA256.fullmatch(provenance_sha256) is None
    ):
        raise M6OfficialCommandError(
            "accepted_m4_snapshot_invalid",
            "the accepted M4 verifier returned mismatched evidence",
        )
    return cohort


def _issue_verified_provenance(
    request: M6CommandRequest,
    repository: M6RepositoryPreflight,
    accepted_m4: object,
    runtime: M6RuntimeObservation,
    results_module: object,
) -> object:
    evidence = accepted_m4.evidence
    row = {
        "plan_version": results_module.M6_PLAN_VERSION,
        "config_version": results_module.M6_CONFIG_VERSION,
        "statistics_schema_version": (
            results_module.M6_STATISTICS_SCHEMA_VERSION
        ),
        "population_label": "accepted_m4_complete_case_ten_shard_cohort",
        "source_shard_start": "00000",
        "source_shard_end": "00009",
        "approved_git_commit": repository.git.commit,
        "git_tree": repository.git.tree,
        "executable_source_sha256": repository.source_sha256,
        "uv_lock_sha256": runtime.uv_lock_sha256,
        "runtime_config_sha256": runtime.runtime_config_sha256,
        "accepted_m4_manifest_sha256": evidence.manifest_sha256,
        "accepted_m4_provenance_sha256": (
            evidence.execution_provenance_sha256
        ),
        "python_version": runtime.python_version,
        "numpy_version": runtime.versions["numpy"],
        "pyarrow_version": runtime.versions["pyarrow"],
        "jax_version": runtime.versions["jax"],
        "jaxlib_version": runtime.versions["jaxlib"],
        "tensorflow_version": runtime.versions["tensorflow"],
        "waymax_commit": runtime.waymax_commit,
        "jax_backend": runtime.jax_backend,
        "jax_device_class": runtime.jax_device_class,
        "primary_intervention_fingerprint": (
            results_module.M6_PRIMARY_INTERVENTION_FINGERPRINT
        ),
        "secondary_intervention_fingerprint": (
            results_module.M6_SECONDARY_INTERVENTION_FINGERPRINT
        ),
    }
    issuer = getattr(results_module, "_issue_m6_verified_provenance", None)
    if not callable(issuer):
        raise M6OfficialCommandError(
            "source_binding_failed",
            "the M6 verified-provenance issuer is unavailable",
        )
    try:
        return issuer(
            mode=request.mode,
            row=row,
            source_paths=repository.source_paths,
        )
    except Exception as exc:
        raise M6OfficialCommandError(
            "source_binding_failed",
            "mechanical M6 provenance issuance failed",
        ) from exc


def preflight_repository(
    request: M6CommandRequest,
    *,
    live_lookup: Callable[[Path], str] = _live_main,
    live_approval_lookup: Callable[[Path], str] = _live_approved_commit,
) -> M6RepositoryPreflight:
    """Run only repository checks safe before result-store creation."""

    bootstrap = _require_active_bootstrap_context()
    if not isinstance(request, M6CommandRequest):
        raise TypeError("request must be M6CommandRequest")
    _require_local_opt_in()
    root = _validated_root(request.project_root)
    if root != bootstrap.project_root:
        raise M6OfficialCommandError(
            "project_root_invalid",
            "the requested root differs from the active M6 bootstrap",
        )
    git = _git_snapshot(
        root,
        live_lookup=live_lookup,
        live_approval_lookup=live_approval_lookup,
    )
    paths, catalog = _approved_source_catalog(root, git.commit)
    _validate_loaded_evalsim_modules(root, paths)
    source_sha256 = _source_fingerprint_from_catalog(catalog)
    try:
        uv_lock_sha256 = next(
            item.sha256 for item in catalog if item.relative_path == "uv.lock"
        )
    except StopIteration as exc:
        raise M6OfficialCommandError(
            "source_binding_failed",
            "the guarded source catalog lacks uv.lock",
        ) from exc
    context = _repository_context_sha256(
        git,
        paths,
        source_sha256,
        uv_lock_sha256,
    )
    preflight = M6RepositoryPreflight(
        root=root,
        git=git,
        source_paths=paths,
        source_sha256=source_sha256,
        source_snapshots=catalog,
        uv_lock_sha256=uv_lock_sha256,
        context_sha256=context,
        _factory_sentinel=_PREFLIGHT_SENTINEL,
    )
    observed_receipt = (
        git.commit,
        git.tree,
        paths,
        source_sha256,
        uv_lock_sha256,
    )
    if (
        type(bootstrap) is _M6BootstrapContext
        and observed_receipt != bootstrap.repository_receipt
    ):
        raise M6OfficialCommandError(
            "source_binding_failed",
            "repository facts differ from the cold-bootstrap approval receipt",
        )
    return preflight


def reverify_repository_preflight(
    request: M6CommandRequest,
    repository: M6RepositoryPreflight,
    *,
    live_lookup: Callable[[Path], str] = _live_main,
    live_approval_lookup: Callable[[Path], str] = _live_approved_commit,
) -> M6RepositoryPreflight:
    _assert_repository_binding(request, repository)
    fresh = preflight_repository(
        request,
        live_lookup=live_lookup,
        live_approval_lookup=live_approval_lookup,
    )
    if fresh != repository:
        raise M6OfficialCommandError(
            "source_binding_failed",
            "repository/source facts changed after preflight",
        )
    return fresh


def preflight_local_inputs(
    request: M6CommandRequest,
    repository: M6RepositoryPreflight,
    store: "M6ResultStore",
    *,
    m4_verifier: Callable[[Path, Path], object] | None = None,
    version_resolver: Callable[[str], str] | None = None,
    module_importer: Callable[[str], object] | None = None,
    distribution_resolver: Callable[[str], object] | None = None,
    python_version_resolver: Callable[[], str] | None = None,
    environment_distributions_resolver: (
        Callable[[], Sequence[object]] | None
    ) = None,
    _allow_awaiting_review: bool = False,
) -> M6LocalInputPreflight:
    """Verify local inputs/runtime only after the exact store is PENDING."""

    results_module = _assert_store_binding(
        request,
        repository,
        store,
        require_pending=not _allow_awaiting_review,
    )
    shards = _resolve_shard_inventory(repository.root, request.data_dir)
    shard_identities = _shard_identities(shards)
    m4_run = _accepted_m4_snapshot_presence(
        repository.root,
        request.m4_run_dir,
    )
    accepted_m4 = _verify_accepted_m4(
        repository.root,
        m4_run,
        m4_verifier,
    )
    runtime = _observe_runtime(
        repository.root,
        repository,
        results_module,
        version_resolver=version_resolver,
        module_importer=module_importer,
        distribution_resolver=distribution_resolver,
        python_version_resolver=python_version_resolver,
        environment_distributions_resolver=environment_distributions_resolver,
    )
    _validate_loaded_evalsim_modules(
        repository.root,
        repository.source_paths,
    )
    provenance = _issue_verified_provenance(
        request,
        repository,
        accepted_m4,
        runtime,
        results_module,
    )
    return M6LocalInputPreflight(
        run_name=request.run_name,
        mode=request.mode,
        result_relative=f"outputs/m6/{request.run_name}",
        repository_context_sha256=repository.context_sha256,
        data_dir=shards[0].parent,
        shard_paths=shards,
        shard_identities=shard_identities,
        m4_run_dir=m4_run,
        accepted_m4=accepted_m4,
        accepted_m4_manifest_sha256=(
            accepted_m4.evidence.manifest_sha256
        ),
        accepted_m4_provenance_sha256=(
            accepted_m4.evidence.execution_provenance_sha256
        ),
        runtime=runtime,
        verified_provenance=provenance,
        _factory_sentinel=_PREFLIGHT_SENTINEL,
    )


def reverify_verified_provenance(
    request: M6CommandRequest,
    repository: M6RepositoryPreflight,
    local: M6LocalInputPreflight,
    results_module: object,
) -> object:
    if (
        type(local) is not M6LocalInputPreflight
        or local._factory_sentinel is not _PREFLIGHT_SENTINEL
        or local.repository_context_sha256 != repository.context_sha256
    ):
        raise M6OfficialCommandError(
            "source_binding_failed",
            "provenance recheck requires exact preflight evidence",
        )
    fresh = _issue_verified_provenance(
        request,
        repository,
        local.accepted_m4,
        local.runtime,
        results_module,
    )
    try:
        local.verified_provenance.revalidate()
        fresh.revalidate()
    except Exception as exc:
        raise M6OfficialCommandError(
            "source_binding_failed",
            "verified provenance failed revalidation",
        ) from exc
    if (
        fresh.context_sha256
        != local.verified_provenance.context_sha256
        or fresh.to_store_row()
        != local.verified_provenance.to_store_row()
    ):
        raise M6OfficialCommandError(
            "source_binding_failed",
            "verified provenance changed after preflight",
        )
    return fresh


def reverify_local_inputs(
    request: M6CommandRequest,
    repository: M6RepositoryPreflight,
    store: "M6ResultStore",
    local: M6LocalInputPreflight,
    *,
    m4_reverifier: Callable[[object], None] | None = None,
    version_resolver: Callable[[str], str] | None = None,
    module_importer: Callable[[str], object] | None = None,
    distribution_resolver: Callable[[str], object] | None = None,
    python_version_resolver: Callable[[], str] | None = None,
    environment_distributions_resolver: (
        Callable[[], Sequence[object]] | None
    ) = None,
) -> M6LocalInputPreflight:
    """Repeat local/M4/runtime/provenance checks for pending or committed stores."""

    results_module = _assert_store_binding(
        request,
        repository,
        store,
        require_pending=False,
    )
    if (
        type(local) is not M6LocalInputPreflight
        or local._factory_sentinel is not _PREFLIGHT_SENTINEL
        or local.run_name != request.run_name
        or local.mode != request.mode
        or local.repository_context_sha256 != repository.context_sha256
    ):
        raise M6OfficialCommandError(
            "source_binding_failed",
            "local recheck evidence differs from the command context",
        )
    shards = _resolve_shard_inventory(repository.root, request.data_dir)
    shard_identities = _shard_identities(shards)
    if (
        shards != local.shard_paths
        or shard_identities != local.shard_identities
        or shards[0].parent != local.data_dir
    ):
        raise M6OfficialCommandError(
            "shard_set_invalid",
            "the frozen shard inventory changed after preflight",
        )
    m4_run = _accepted_m4_snapshot_presence(
        repository.root,
        request.m4_run_dir,
    )
    if m4_run != local.m4_run_dir:
        raise M6OfficialCommandError(
            "accepted_m4_snapshot_invalid",
            "the accepted M4 path changed after preflight",
        )
    if m4_reverifier is None:
        from evalsim.sources.m5_m4_reuse import reverify_accepted_m4_run

        m4_reverifier = reverify_accepted_m4_run
    try:
        m4_reverifier(local.accepted_m4)
    except Exception as exc:
        raise M6OfficialCommandError(
            "accepted_m4_snapshot_invalid",
            "the accepted M4 evidence changed after preflight",
        ) from exc
    runtime = _observe_runtime(
        repository.root,
        repository,
        results_module,
        version_resolver=version_resolver,
        module_importer=module_importer,
        distribution_resolver=distribution_resolver,
        python_version_resolver=python_version_resolver,
        environment_distributions_resolver=environment_distributions_resolver,
    )
    if runtime != local.runtime:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "runtime facts changed after preflight",
        )
    _validate_loaded_evalsim_modules(
        repository.root,
        repository.source_paths,
    )
    fresh_provenance = _issue_verified_provenance(
        request,
        repository,
        local.accepted_m4,
        runtime,
        results_module,
    )
    if (
        fresh_provenance.context_sha256
        != local.verified_provenance.context_sha256
        or fresh_provenance.to_store_row()
        != local.verified_provenance.to_store_row()
    ):
        raise M6OfficialCommandError(
            "source_binding_failed",
            "verified provenance changed with local inputs",
        )
    return M6LocalInputPreflight(
        run_name=local.run_name,
        mode=local.mode,
        result_relative=local.result_relative,
        repository_context_sha256=local.repository_context_sha256,
        data_dir=local.data_dir,
        shard_paths=shards,
        shard_identities=shard_identities,
        m4_run_dir=m4_run,
        accepted_m4=local.accepted_m4,
        accepted_m4_manifest_sha256=local.accepted_m4_manifest_sha256,
        accepted_m4_provenance_sha256=local.accepted_m4_provenance_sha256,
        runtime=runtime,
        verified_provenance=fresh_provenance,
        _factory_sentinel=_PREFLIGHT_SENTINEL,
    )


M6Preregistrar = Callable[
    [Sequence[Mapping[str, Any]], object],
    None,
]
M6OutcomeBoundary = Callable[[], None]
M6EligibilityExecutor = Callable[
    [M6CommandRequest, M6LocalInputPreflight, M6Preregistrar, M6OutcomeBoundary],
    M6ModeExecutionEvidence,
]
M6ComputePilotExecutor = Callable[
    [M6CommandRequest, M6LocalInputPreflight, M6Preregistrar, M6OutcomeBoundary],
    M6ModeExecutionEvidence,
]
M6OfficialExecutor = Callable[
    [M6CommandRequest, M6LocalInputPreflight, M6Preregistrar, M6OutcomeBoundary],
    M6ModeExecutionEvidence,
]
M6PredecessorGate = Callable[
    [Sequence[Mapping[str, Any]], object],
    None,
]
M6PredecessorGateFactory = Callable[
    [M6CommandRequest, M6LocalInputPreflight, object],
    M6PredecessorGate,
]

_FAILURE_DIAGNOSTIC_NAME = "failure-diagnostic.bin"
_MAX_FAILURE_DIAGNOSTIC_BYTES = 2 * 1024 * 1024


def run_m6_eligibility_only_execution(
    request: M6CommandRequest,
    local: M6LocalInputPreflight,
    preregister: M6Preregistrar,
    begin_outcomes: M6OutcomeBoundary,
) -> M6ModeExecutionEvidence:
    """Visit accepted M4 once and issue source-only eligibility evidence."""

    if (
        not isinstance(request, M6CommandRequest)
        or request.mode != "eligibility_only"
        or type(local) is not M6LocalInputPreflight
        or local._factory_sentinel is not _PREFLIGHT_SENTINEL
        or local.mode != request.mode
        or local.run_name != request.run_name
        or not callable(preregister)
        or not callable(begin_outcomes)
    ):
        raise M6OfficialCommandError(
            "result_contract_failed",
            "eligibility execution requires exact local preflight evidence",
        )
    from evalsim.evaluation.m6 import evaluate_m6_source_eligibility
    from evalsim.evaluation.m6_official import (
        M6OfficialCaseCollector,
        m6_eligibility_rows,
    )
    from evalsim.evaluation.m6_waymax_official import (
        M6WaymaxOfficialCollector,
        build_m6_waymax_verified_source_authority,
    )
    from evalsim.sources.m5_m4_reuse import visit_accepted_m4_cohort

    source_authority = build_m6_waymax_verified_source_authority(
        local.accepted_m4
    )
    case_collector = M6OfficialCaseCollector()
    waymax_collector = M6WaymaxOfficialCollector(source_authority)

    def combined_visitor(member: object) -> None:
        case_collector(member)
        waymax_collector(member)

    visit_accepted_m4_cohort(
        local.accepted_m4,
        local.data_dir,
        combined_visitor,
    )
    cases = case_collector.cases
    ledger = evaluate_m6_source_eligibility(cases)
    source = waymax_collector.finalize(ledger)
    source.revalidate()
    if source.promotable is not True:
        raise M6OfficialCommandError(
            "result_contract_failed",
            "eligibility source was not accepted-M4-authorized",
        )
    rows = m6_eligibility_rows(
        ledger,
        mode="eligibility_only",
    )
    preregister(rows, source.selection)
    return M6ModeExecutionEvidence(
        mode="eligibility_only",
        eligibility_rows=rows,
        selection=source.selection,
    )


def _validate_production_execution_context(
    request: M6CommandRequest,
    local: M6LocalInputPreflight,
    *,
    mode: str,
) -> None:
    if (
        type(request) is not M6CommandRequest
        or request.mode != mode
        or type(local) is not M6LocalInputPreflight
        or local._factory_sentinel is not _PREFLIGHT_SENTINEL
        or local.mode != mode
        or local.run_name != request.run_name
        or local.result_relative != f"outputs/m6/{request.run_name}"
    ):
        raise M6OfficialCommandError(
            "result_contract_failed",
            "production execution requires exact local preflight evidence",
        )


def _positive_elapsed_ms(start_ns: int) -> int:
    stop_ns = time.monotonic_ns()
    if (
        isinstance(start_ns, bool)
        or not isinstance(start_ns, int)
        or isinstance(stop_ns, bool)
        or not isinstance(stop_ns, int)
        or stop_ns <= start_ns
    ):
        raise M6OfficialCommandError(
            "execution_failed",
            "the monotonic execution clock is invalid",
        )
    return (stop_ns - start_ns + 999_999) // 1_000_000


def _peak_process_rss_bytes() -> int:
    """Return the standalone CLI worker's conservative process peak.

    Production invokes one fresh isolated ``m6_bootstrap.py`` process per run,
    so ``RUSAGE_SELF`` includes preflight, decode, and all policy work in that
    worker.
    """
    import resource

    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if raw <= 0:
        raise M6OfficialCommandError(
            "execution_failed",
            "fresh-worker peak process RSS is unavailable",
        )
    observed = raw if sys.platform == "darwin" else raw * 1024
    if observed <= 0:
        raise M6OfficialCommandError(
            "execution_failed",
            "fresh-worker peak process RSS is invalid",
        )
    return observed


def _collect_m6_execution_inputs(
    local: M6LocalInputPreflight,
) -> tuple[tuple[object, ...], object, object]:
    """Visit accepted M4 once for detached NumPy and bounded native evidence."""

    from evalsim.evaluation.m6 import evaluate_m6_source_eligibility
    from evalsim.evaluation.m6_official import M6OfficialCaseCollector
    from evalsim.evaluation.m6_waymax_official import (
        M6WaymaxOfficialCollector,
        build_m6_waymax_verified_source_authority,
    )
    from evalsim.sources.m5_m4_reuse import visit_accepted_m4_cohort

    source_authority = build_m6_waymax_verified_source_authority(
        local.accepted_m4
    )
    case_collector = M6OfficialCaseCollector()
    waymax_collector = M6WaymaxOfficialCollector(source_authority)

    def combined_visitor(member: object) -> None:
        case_collector(member)
        waymax_collector(member)

    visit_accepted_m4_cohort(
        local.accepted_m4,
        local.data_dir,
        combined_visitor,
    )
    cases = case_collector.cases
    ledger = evaluate_m6_source_eligibility(cases)
    source = waymax_collector.finalize(ledger)
    source.revalidate()
    if source.promotable is not True:
        raise M6OfficialCommandError(
            "result_contract_failed",
            "execution source was not accepted-M4-authorized",
        )
    return cases, ledger, source



def _build_m6_predecessor_gate(
    request: M6CommandRequest,
    local: M6LocalInputPreflight,
    results_module: object,
) -> M6PredecessorGate:
    """Reopen terminal predecessors and bind them to this exact local context."""

    if request.mode == "eligibility_only":
        return lambda rows, selection: None

    def fail(message: str, cause: BaseException | None = None) -> None:
        error = M6OfficialCommandError("verification_failed", message)
        if cause is None:
            raise error
        raise error from cause

    def reopen(run_name: str | None, mode: str) -> object:
        if run_name is None:
            fail("a required predecessor run name is absent")
        try:
            return results_module.verify_m6_result_store(
                request.project_root,
                run_name,
                expected_mode=mode,
            )
        except Exception as exc:
            fail("a named predecessor is not terminal-success verified", exc)
        raise AssertionError("unreachable")

    eligibility = reopen(
        request.eligibility_run_name,
        "eligibility_only",
    )
    predecessors = [eligibility]
    pilot = None
    if request.mode == "official":
        pilot = reopen(request.pilot_run_name, "compute_pilot")
        predecessors.append(pilot)
        try:
            pilot_rows = pilot.read_dataset(
                results_module.COMPUTE_PILOT_SUMMARY
            ).to_pylist()
        except Exception as exc:
            fail("the compute-pilot predecessor cannot be reopened", exc)
        if len(pilot_rows) != 1 or pilot_rows[0].get("passed") is not True:
            fail("the compute-pilot predecessor did not pass its frozen gates")
        pilot_row = dict(pilot_rows[0])
        binding_names = (
            "selection_binding_sha256",
            "selected_cohort_indices_sha256",
            "numpy_observation_content_sha256",
            "waymax_observation_content_sha256",
            "pilot_report_binding_sha256",
        )
        try:
            pilot_provenance_rows = pilot.read_dataset(
                results_module.TYPED_PROVENANCE
            ).to_pylist()
            if len(pilot_provenance_rows) != 1:
                fail("the compute-pilot predecessor provenance is not singular")
            pilot_provenance_context = pilot_provenance_rows[0][
                "verification_context_sha256"
            ]
            pilot_selection_binding = (
                pilot.waymax_selection_receipt.selection_binding_sha256
            )
            pilot_qualification_rows = pilot.read_dataset(
                results_module.WAYMAX_QUALIFICATION
            ).to_pylist()
            expected_selected_indices_binding = (
                results_module._m6_compute_pilot_selected_indices_sha256(
                    pilot_qualification_rows,
                    pilot.waymax_selection_receipt,
                )
            )
            pilot_summary = {
                name: value
                for name, value in pilot_row.items()
                if name not in binding_names
            }
            expected_pilot_report = (
                results_module.m6_compute_pilot_report_binding_sha256(
                    run_name=pilot.run_path.name,
                    result_path=(
                        Path("outputs") / "m6" / pilot.run_path.name
                    ).as_posix(),
                    provenance_context_sha256=pilot_provenance_context,
                    selection_binding_sha256=pilot_selection_binding,
                    selected_cohort_indices_sha256=pilot_row[
                        "selected_cohort_indices_sha256"
                    ],
                    numpy_observation_content_sha256=pilot_row[
                        "numpy_observation_content_sha256"
                    ],
                    waymax_observation_content_sha256=pilot_row[
                        "waymax_observation_content_sha256"
                    ],
                    summary=pilot_summary,
                )
            )
        except M6OfficialCommandError:
            raise
        except Exception as exc:
            fail("the compute-pilot predecessor bindings cannot be reopened", exc)
        if (
            pilot_row.get("selection_binding_sha256")
            != pilot_selection_binding
            or pilot_row.get("selected_cohort_indices_sha256")
            != expected_selected_indices_binding
            or pilot_row.get("pilot_report_binding_sha256")
            != expected_pilot_report
            or any(
                _SHA256.fullmatch(str(pilot_row.get(name, ""))) is None
                for name in binding_names
            )
        ):
            fail("the compute-pilot predecessor report binding is inconsistent")

    try:
        current_provenance = local.verified_provenance.to_store_row()
    except Exception as exc:
        fail("current typed provenance cannot be revalidated", exc)
    current_provenance.pop("verification_context_sha256", None)
    current_provenance.pop("mode", None)
    for predecessor in predecessors:
        try:
            rows = predecessor.read_dataset(
                results_module.TYPED_PROVENANCE
            ).to_pylist()
        except Exception as exc:
            fail("predecessor typed provenance cannot be reopened", exc)
        if len(rows) != 1:
            fail("predecessor typed provenance is not singular")
        predecessor_provenance = dict(rows[0])
        predecessor_provenance.pop("verification_context_sha256", None)
        predecessor_provenance.pop("mode", None)
        if predecessor_provenance != current_provenance:
            fail("predecessor source/runtime/M4 provenance differs")

    def primary_receipt_projection(verified: object) -> tuple[object, ...]:
        receipt = verified.receipt
        return (
            receipt.population_size,
            tuple(receipt.eligible_cohort_indices),
            tuple(sorted(dict(receipt.rejection_reason_counts).items())),
            receipt.primary_intervention_fingerprint,
        )

    def selection_receipt_projection(verified: object) -> dict[str, Any]:
        value = verified.waymax_selection_receipt.to_dict()
        value.pop("mode", None)
        return value

    if pilot is not None and (
        primary_receipt_projection(pilot)
        != primary_receipt_projection(eligibility)
        or selection_receipt_projection(pilot)
        != selection_receipt_projection(eligibility)
    ):
        fail("pilot and eligibility predecessors are not source-identical")

    eligibility_receipt = eligibility.receipt
    selection_receipt = eligibility.waymax_selection_receipt

    def gate(
        rows: Sequence[Mapping[str, Any]],
        selection: object,
    ) -> None:
        try:
            normalized = tuple(dict(row) for row in rows)
            eligible = tuple(
                row["cohort_index"]
                for row in normalized
                if row["primary_eligible"] is True
            )
            reasons = {
                reason: sum(
                    row["rejection_reason"] == reason
                    for row in normalized
                )
                for reason in results_module.M6_PRIMARY_REJECTION_REASONS
            }
            selection.revalidate()
            from evalsim.evaluation.m6_waymax_official import (
                m6_waymax_selection_binding_sha256,
            )

            selection_binding = m6_waymax_selection_binding_sha256(selection)
        except Exception as exc:
            fail("current preregistration cannot be predecessor-bound", exc)
        if (
            len(normalized) != 128
            or eligible
            != tuple(eligibility_receipt.eligible_cohort_indices)
            or reasons != dict(eligibility_receipt.rejection_reason_counts)
            or selection.primary_domain_sha256
            != selection_receipt.primary_domain_sha256
            or selection.primary_domain_member_count
            != selection_receipt.primary_domain_member_count
            or selection.qualification_ledger_sha256
            != selection_receipt.qualification_ledger_sha256
            or selection.selection_sha256
            != selection_receipt.selector_selection_sha256
            or selection_binding
            != selection_receipt.selection_binding_sha256
            or selection.supported
            is not selection_receipt.selection_supported
            or selection.eligible_count != selection_receipt.eligible_count
            or len(selection.members)
            != selection_receipt.selection_member_count
        ):
            fail("current eligibility/selection differs from its predecessor")
        if pilot is not None:
            current_secondary = tuple(
                row["cohort_index"]
                for row in normalized
                if row["secondary_b4_feasible"] is True
            )
            if current_secondary != tuple(
                pilot.receipt.secondary_b4_cohort_indices
            ):
                fail("current secondary feasibility differs from the pilot")

    return gate
def run_m6_compute_pilot_execution(
    request: M6CommandRequest,
    local: M6LocalInputPreflight,
    preregister: M6Preregistrar,
    begin_outcomes: M6OutcomeBoundary,
) -> M6ModeExecutionEvidence:
    """Run the outcome-suppressed first-eight source-ranked compute pilot."""

    _validate_production_execution_context(
        request,
        local,
        mode="compute_pilot",
    )
    if not callable(preregister) or not callable(begin_outcomes):
        raise M6OfficialCommandError(
            "result_contract_failed",
            "compute pilot requires the one-shot preregistration gate",
        )
    pilot_wall_start = time.monotonic_ns()
    from evalsim.evaluation.m6_official import m6_eligibility_rows
    from evalsim.evaluation.m6_pilot import run_m6_numpy_pilot
    from evalsim.evaluation.m6_waymax_official import (
        build_pinned_m6_waymax_execution_authority,
        m6_waymax_selection_binding_sha256,
        run_m6_waymax_outcome_suppressed_pilot,
    )

    decode_start = time.monotonic_ns()
    cases, ledger, source = _collect_m6_execution_inputs(local)
    if (
        isinstance(getattr(ledger, "eligible_n", None), bool)
        or not isinstance(getattr(ledger, "eligible_n", None), int)
        or ledger.eligible_n < 10
    ):
        raise M6OfficialCommandError(
            "execution_failed",
            "the preregistered primary eligibility floor is not met",
        )
    rows = m6_eligibility_rows(ledger, mode="compute_pilot")
    selection = source.selection
    selection.revalidate(primary_domain=source.primary_domain)
    ranked_primary = tuple(
        sorted(
            source.qualification_ledger.rows,
            key=lambda row: (
                bytes.fromhex(row.rank_sha256),
                row.cohort_index,
            ),
        )
    )
    selected_members = (
        selection.members[:8]
        if selection.supported
        else ranked_primary[:8]
    )
    if len(selected_members) != 8:
        raise M6OfficialCommandError(
            "execution_failed",
            "the source-ranked pilot does not contain exactly eight scenes",
        )
    selected_indices = tuple(
        member.cohort_index for member in selected_members
    )
    selection_binding = m6_waymax_selection_binding_sha256(selection)
    preregister(rows, selection)
    begin_outcomes()
    decode_ms = _positive_elapsed_ms(decode_start)

    numpy_start = time.monotonic_ns()
    numpy_observation = run_m6_numpy_pilot(
        cases,
        ledger,
        selected_indices,
        selection_binding_sha256=selection_binding,
    )
    numpy_outer_ms = _positive_elapsed_ms(numpy_start)

    waymax_start = time.monotonic_ns()
    execution_authority = build_pinned_m6_waymax_execution_authority()
    waymax_observation = run_m6_waymax_outcome_suppressed_pilot(
        source,
        execution_authority,
    )
    waymax_outer_ms = _positive_elapsed_ms(waymax_start)

    verification_start = time.monotonic_ns()
    numpy_observation.revalidate()
    waymax_observation.revalidate()
    source.revalidate()
    if (
        numpy_observation.total_execution_ms
        > numpy_outer_ms + numpy_observation.scene_count - 1
        or (
            waymax_observation.validation_ms
            + waymax_observation.execution_ms
            > waymax_outer_ms + waymax_observation.scene_count
        )
        or numpy_observation.scene_count != len(selected_indices)
        or numpy_observation.source_selection_binding_sha256
        != selection_binding
        or (
            selection.supported
            and (
                waymax_observation.status != "completed"
                or waymax_observation.scene_count != len(selected_indices)
            )
        )
        or (
            not selection.supported
            and waymax_observation.status != "unsupported"
        )
    ):
        raise M6OfficialCommandError(
            "result_contract_failed",
            "pilot observations differ from the canonical source selection",
        )
    fresh_worker_peak_rss_bytes = _peak_process_rss_bytes()
    if (
        waymax_observation.peak_process_rss_bytes
        and fresh_worker_peak_rss_bytes
        < waymax_observation.peak_process_rss_bytes
    ):
        raise M6OfficialCommandError(
            "result_contract_failed",
            "pilot peak RSS regressed during final observation",
        )
    verification_ms = _positive_elapsed_ms(verification_start)
    total_wall_ms = _positive_elapsed_ms(pilot_wall_start)
    max_scene_ms = max(
        numpy_observation.max_scene_ms,
        waymax_observation.max_scene_ms,
    )
    passed = (
        total_wall_ms <= 30 * 60 * 1000
        and max_scene_ms <= 10 * 60 * 1000
        and fresh_worker_peak_rss_bytes <= 16 * 1024**3
    )
    summary = {
        "pilot_scene_n": len(selected_indices),
        "total_wall_ms": total_wall_ms,
        "max_scene_ms": max_scene_ms,
        "decode_ms": decode_ms,
        "numpy_ms": numpy_observation.total_execution_ms,
        "waymax_ms": (
            waymax_observation.validation_ms
            + waymax_observation.execution_ms
        ),
        "verification_ms": verification_ms,
        "fresh_worker_peak_rss_bytes": fresh_worker_peak_rss_bytes,
        "passed": passed,
    }
    return _issue_m6_compute_pilot_evidence(
        eligibility_rows=rows,
        selection=selection,
        pilot_summary=summary,
        pilot_selection_positions=tuple(range(len(selected_indices))),
        numpy_observation=numpy_observation,
        waymax_observation=waymax_observation,
        verified_provenance=local.verified_provenance,
        run_name=request.run_name,
        result_path=local.result_relative,
        fresh_worker_peak_rss_bytes=fresh_worker_peak_rss_bytes,
    )


def run_m6_official_execution(
    request: M6CommandRequest,
    local: M6LocalInputPreflight,
    preregister: M6Preregistrar,
    begin_outcomes: M6OutcomeBoundary,
) -> M6ModeExecutionEvidence:
    """Execute the fixed official NumPy and Waymax analyses once."""

    _validate_production_execution_context(
        request,
        local,
        mode="official",
    )
    if not callable(preregister) or not callable(begin_outcomes):
        raise M6OfficialCommandError(
            "result_contract_failed",
            "official execution requires the one-shot preregistration gate",
        )
    from evalsim.evaluation.m6_official import (
        m6_eligibility_rows,
        run_m6_official_numpy,
    )
    from evalsim.evaluation.m6_waymax_official import (
        build_pinned_m6_waymax_execution_authority,
        run_m6_waymax_official,
    )

    eligibility_start = time.monotonic_ns()
    cases, source_ledger, source = _collect_m6_execution_inputs(local)
    source_rows = m6_eligibility_rows(source_ledger, mode="official")
    preregister(source_rows, source.selection)
    begin_outcomes()
    eligibility_ms = _positive_elapsed_ms(eligibility_start)

    numpy_rows = run_m6_official_numpy(cases)
    if tuple(numpy_rows.eligibility_rows) != tuple(source_rows):
        raise M6OfficialCommandError(
            "result_contract_failed",
            "official NumPy eligibility differs from accepted-M4 source evidence",
        )

    waymax_start = time.monotonic_ns()
    execution_authority = build_pinned_m6_waymax_execution_authority()
    waymax_evidence = run_m6_waymax_official(
        source,
        execution_authority,
        numpy_rows,
    )
    waymax_ms = _positive_elapsed_ms(waymax_start)

    verification_start = time.monotonic_ns()
    source.revalidate()
    numpy_rows.revalidate()
    waymax_evidence.revalidate()
    if (
        waymax_evidence.selection is not source.selection
        or waymax_evidence.promotable is not bool(source.selection.supported)
    ):
        raise M6OfficialCommandError(
            "result_contract_failed",
            "official Waymax evidence differs from its source/runtime authority",
        )
    fresh_worker_peak_rss_bytes = _peak_process_rss_bytes()
    verification_ms = (
        numpy_rows.phase_durations_ms["verification"]
        + _positive_elapsed_ms(verification_start)
    )
    return M6ModeExecutionEvidence(
        mode="official",
        eligibility_rows=tuple(numpy_rows.eligibility_rows),
        selection=source.selection,
        numpy_rows=numpy_rows,
        waymax_evidence=waymax_evidence,
        stage_durations_ms={
            "eligibility": eligibility_ms,
            "numpy_rollouts": numpy_rows.phase_durations_ms["numpy_rollouts"],
            "paired_metrics": numpy_rows.phase_durations_ms["paired_metrics"],
            "statistics": numpy_rows.phase_durations_ms["statistics"],
            "waymax": waymax_ms,
            "verification": verification_ms,
        },
        fresh_worker_peak_rss_bytes=fresh_worker_peak_rss_bytes,
    )




def _selected_executor(
    mode: str,
    *,
    eligibility_executor: M6EligibilityExecutor | None,
    pilot_executor: M6ComputePilotExecutor | None,
    official_executor: M6OfficialExecutor | None,
) -> Callable[
    [M6CommandRequest, M6LocalInputPreflight, M6Preregistrar, M6OutcomeBoundary],
    M6ModeExecutionEvidence,
]:
    candidates = {
        "eligibility_only": eligibility_executor,
        "compute_pilot": pilot_executor,
        "official": official_executor,
    }
    selected = candidates[mode]
    if not callable(selected):
        raise M6OfficialCommandError(
            "execution_not_available",
            "the selected mode has no reviewed runtime execution authority",
        )
    return selected


def _reservation_fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reservation_guard_directory(
    path: Path,
    *,
    parent: Path,
    require_mode: bool,
) -> None:
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or (require_mode and stat.S_IMODE(metadata.st_mode) != 0o700)
        or path.parent != parent
        or path.resolve(strict=True) != path
    ):
        raise M6OfficialCommandError(
            "result_store_failed",
            "the stdlib PENDING directory boundary is unsafe",
        )


def _reservation_ensure_directory(
    parent: Path,
    name: str,
    *,
    require_mode: bool,
) -> Path:
    path = parent / name
    try:
        if not os.path.lexists(path):
            os.mkdir(path, 0o700)
            _reservation_fsync_directory(parent)
        _reservation_guard_directory(
            path,
            parent=parent,
            require_mode=require_mode,
        )
    except OSError as exc:
        raise M6OfficialCommandError(
            "result_store_failed",
            "the stdlib PENDING directory boundary could not be created",
        ) from exc
    return path


def _reservation_guard_file(path: Path, run_path: Path) -> bytes:
    if path.parent != run_path or path.name in {"", ".", ".."}:
        raise M6OfficialCommandError(
            "result_store_failed",
            "the stdlib PENDING marker is not contained",
        )
    descriptor = -1
    try:
        before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        payload = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            payload.extend(chunk)
        after = os.fstat(descriptor)
        final = path.lstat()
        identities = {
            _stat_identity(before),
            _stat_identity(opened),
            _stat_identity(after),
            _stat_identity(final),
        }
        if (
            len(identities) != 1
            or not stat.S_ISREG(after.st_mode)
            or stat.S_IMODE(after.st_mode) != 0o600
            or after.st_uid != os.geteuid()
            or after.st_nlink != 1
            or path.resolve(strict=True) != path
        ):
            raise OSError("unsafe reservation marker")
        return bytes(payload)
    except OSError as exc:
        raise M6OfficialCommandError(
            "result_store_failed",
            "the stdlib PENDING marker failed guarded verification",
        ) from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _reservation_write_exclusive(
    path: Path,
    payload: bytes,
    run_path: Path,
) -> None:
    if type(payload) is not bytes or path.parent != run_path:
        raise TypeError("reservation payload/path is invalid")
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("reservation marker accepted no bytes")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _reservation_fsync_directory(run_path)
        if _reservation_guard_file(path, run_path) != payload:
            raise OSError("reservation marker bytes changed")
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _reservation_validate_tree(reservation: _M6PendingReservation) -> None:
    _reservation_guard_directory(
        reservation.run_path.parent,
        parent=reservation.project_root / "outputs",
        require_mode=True,
    )
    _reservation_guard_directory(
        reservation.run_path,
        parent=reservation.run_path.parent,
        require_mode=True,
    )
    with os.scandir(reservation.run_path) as iterator:
        names = tuple(sorted(entry.name for entry in iterator))
    allowed = {"PENDING"}
    if set(names) - allowed:
        raise M6OfficialCommandError(
            "result_store_failed",
            "the stdlib PENDING reservation contains an unexpected member",
        )
    if "PENDING" not in names or (
        _reservation_guard_file(
            reservation.run_path / "PENDING",
            reservation.run_path,
        )
        != reservation.pending_payload
    ):
        raise M6OfficialCommandError(
            "result_store_failed",
            "the stdlib PENDING reservation changed before adoption",
        )


def _pending_reservation_payload(
    run_name: str,
    mode: str,
    capability_sha256: str,
) -> bytes:
    return _canonical_json_bytes(
        {
            "capability_sha256": capability_sha256,
            "mode": mode,
            "result_path": f"outputs/m6/{run_name}",
            "schema_version": _M6_RESULT_STORE_SCHEMA_VERSION,
            "state": "PENDING",
        }
    )


def _require_result_path_git_invisible(root: Path, relative: Path) -> None:
    ignored = _git_process(
        root,
        "check-ignore",
        "--quiet",
        "--no-index",
        "--",
        relative.as_posix(),
    )
    tracked = _git_bytes(
        root,
        "ls-files",
        "-z",
        "--",
        relative.as_posix(),
    )
    if ignored.returncode != 0 or tracked:
        raise M6OfficialCommandError(
            "result_store_failed",
            "the M6 result path is visible to Git",
        )


def _create_m6_pending_reservation(
    request: M6CommandRequest,
    repository: M6RepositoryPreflight,
) -> _M6PendingReservation:
    _assert_repository_binding(request, repository)
    relative = Path("outputs") / "m6" / request.run_name
    _require_result_path_git_invisible(repository.root, relative)
    outputs = _reservation_ensure_directory(
        repository.root,
        "outputs",
        require_mode=False,
    )
    m6_root = _reservation_ensure_directory(
        outputs,
        "m6",
        require_mode=True,
    )
    run_path = m6_root / request.run_name
    if os.path.lexists(run_path):
        raise FileExistsError(
            f"M6 run {request.run_name!r} already exists and cannot be resumed"
        )
    nonce = secrets.token_bytes(32)
    payload = _pending_reservation_payload(
        request.run_name,
        request.mode,
        hashlib.sha256(nonce).hexdigest(),
    )
    created = False
    try:
        os.mkdir(run_path, 0o700)
        created = True
        _reservation_fsync_directory(m6_root)
        _reservation_guard_directory(
            run_path,
            parent=m6_root,
            require_mode=True,
        )
        _reservation_write_exclusive(
            run_path / "PENDING",
            payload,
            run_path,
        )
        reservation = _M6PendingReservation(
            project_root=repository.root,
            run_name=request.run_name,
            run_path=run_path,
            mode=request.mode,
            capability_nonce=nonce,
            pending_payload=payload,
            _factory_sentinel=_PENDING_RESERVATION_SENTINEL,
        )
        _reservation_validate_tree(reservation)
        record = _M6PendingReservationRecord(reservation=reservation)
        with _PENDING_RESERVATION_LOCK:
            if id(reservation) in _PENDING_RESERVATION_REGISTRY:
                raise RuntimeError("PENDING reservation identity was reused")
            _PENDING_RESERVATION_REGISTRY[id(reservation)] = record
        return reservation
    except BaseException:
        if created:
            failure = _canonical_json_bytes(
                {
                    "mode": request.mode,
                    "reason_code": "creation_failed",
                    "schema_version": _M6_RESULT_STORE_SCHEMA_VERSION,
                    "state": "TERMINAL_FAILURE",
                }
            )
            try:
                if not os.path.lexists(run_path / "TERMINAL_FAILURE"):
                    _reservation_write_exclusive(
                        run_path / "TERMINAL_FAILURE",
                        failure,
                        run_path,
                    )
            except BaseException:
                pass
        raise


def _consume_m6_pending_reservation(
    reservation: object,
) -> tuple[Path, str, Path, str, bytes]:
    if (
        type(reservation) is not _M6PendingReservation
        or reservation._factory_sentinel is not _PENDING_RESERVATION_SENTINEL
    ):
        raise M6OfficialCommandError(
            "result_store_failed",
            "M6ResultStore received no verifier-issued PENDING reservation",
        )
    with _PENDING_RESERVATION_LOCK:
        record = _PENDING_RESERVATION_REGISTRY.get(id(reservation))
        if (
            record is None
            or record.reservation is not reservation
            or record.state != "pending"
        ):
            raise M6OfficialCommandError(
                "result_store_failed",
                "the PENDING reservation was already consumed or failed",
            )
        _reservation_validate_tree(reservation)
        record.state = "consumed"
    return (
        reservation.project_root,
        reservation.run_name,
        reservation.run_path,
        reservation.mode,
        reservation.capability_nonce,
    )


def _fail_pending_reservation(
    reservation: _M6PendingReservation,
    reason_code: str,
) -> Path:
    if (
        type(reservation) is not _M6PendingReservation
        or reservation._factory_sentinel is not _PENDING_RESERVATION_SENTINEL
    ):
        raise M6OfficialCommandError(
            "result_store_failed",
            "failure requires an issued PENDING reservation",
        )
    normalized = (
        reason_code if reason_code in _TRUSTED_CODES else "unexpected_failure"
    )
    failure_path = reservation.run_path / "TERMINAL_FAILURE"
    payload = _canonical_json_bytes(
        {
            "mode": reservation.mode,
            "reason_code": normalized,
            "schema_version": _M6_RESULT_STORE_SCHEMA_VERSION,
            "state": "TERMINAL_FAILURE",
        }
    )
    with _PENDING_RESERVATION_LOCK:
        record = _PENDING_RESERVATION_REGISTRY.get(id(reservation))
        if record is None or record.reservation is not reservation:
            raise M6OfficialCommandError(
                "result_store_failed",
                "the PENDING reservation issuance is unavailable",
            )
        if record.state == "failed":
            if _reservation_guard_file(
                failure_path,
                reservation.run_path,
            ) != payload:
                raise M6OfficialCommandError(
                    "result_store_failed",
                    "the existing reservation failure marker is contradictory",
                )
            return failure_path
        if record.state not in {"pending", "consumed"}:
            raise M6OfficialCommandError(
                "result_store_failed",
                "the PENDING reservation cannot transition to failure",
            )
        if os.path.lexists(failure_path):
            if _reservation_guard_file(
                failure_path,
                reservation.run_path,
            ) != payload:
                raise M6OfficialCommandError(
                    "result_store_failed",
                    "the existing reservation failure marker is contradictory",
                )
        else:
            _reservation_write_exclusive(
                failure_path,
                payload,
                reservation.run_path,
            )
        record.state = "failed"
    return failure_path


def _load_m6_results_module() -> object:
    from evalsim.results import m6 as results_module

    return results_module


def _load_authenticated_review_results(
    repository: M6RepositoryPreflight,
    *,
    environment_distributions_resolver: (
        Callable[[], Sequence[object]] | None
    ) = None,
) -> tuple[object, tuple[object, object] | None]:
    """Authenticate the installed environment before review-time imports."""

    if _ACTIVE_BOOTSTRAP_CONTEXT is None:
        results_module = _load_m6_results_module()
        return results_module, None
    bootstrap = _require_active_bootstrap_context()
    if any(
        module_name in sys.modules
        for module_name in _RUNTIME_MODULES.values()
    ):
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "an optional runtime module loaded before review authentication",
        )
    resolver = (
        (
            lambda: tuple(
                importlib.metadata.distributions(
                    path=[os.fspath(bootstrap.site_packages)]
                )
            )
        )
        if environment_distributions_resolver is None
        else environment_distributions_resolver
    )
    preimport_environment = _complete_environment_catalog(
        repository.root,
        resolver,
    )
    _enable_active_bootstrap_site_packages()
    results_module = _load_m6_results_module()
    if results_module.M6_RESULT_STORE_SCHEMA_VERSION != (
        _M6_RESULT_STORE_SCHEMA_VERSION
    ):
        raise M6OfficialCommandError(
            "result_store_failed",
            "stdlib bootstrap and result-store schemas differ",
        )
    return results_module, preimport_environment


def _reverify_awaiting_review_precursor(
    request: M6CommandRequest,
    local: M6LocalInputPreflight,
    store: object,
    results_module: object,
) -> object:
    """Rebind the sealed precursor to fresh provenance and predecessors."""

    def fail(message: str, cause: BaseException | None = None) -> None:
        error = M6OfficialCommandError("verification_failed", message)
        if cause is None:
            raise error
        raise error from cause

    try:
        store._assert_awaiting_review_capability()
        receipt = store.eligibility_receipt
        if receipt is None or receipt.mode != "official":
            fail("awaiting-review eligibility receipt is unavailable")
        stored_provenance = results_module._normalize_typed_provenance(
            store._read_dataset_rows(results_module.TYPED_PROVENANCE),
            receipt,
        )[0]
        expected_provenance = results_module._normalize_typed_provenance(
            (local.verified_provenance.to_store_row(),),
            receipt,
        )[0]
        if stored_provenance != expected_provenance:
            fail("awaiting-review provenance differs from fresh verified facts")

        predecessor_gate = _build_m6_predecessor_gate(
            request,
            local,
            results_module,
        )
        if not callable(predecessor_gate):
            fail("predecessor re-verification omitted its exact gate")
        eligibility = results_module.verify_m6_result_store(
            request.project_root,
            request.eligibility_run_name,
            expected_mode="eligibility_only",
        )
        pilot = results_module.verify_m6_result_store(
            request.project_root,
            request.pilot_run_name,
            expected_mode="compute_pilot",
        )
        current_rows = tuple(
            dict(row)
            for row in store._read_dataset_rows(
                results_module.ELIGIBILITY_LEDGER
            )
        )
        for predecessor in (eligibility, pilot):
            predecessor_rows = tuple(
                dict(row)
                for row in predecessor.read_dataset(
                    results_module.ELIGIBILITY_LEDGER
                ).to_pylist()
            )
            if predecessor_rows != current_rows:
                fail("awaiting-review eligibility differs from a predecessor")

        def primary_projection(value: object) -> tuple[object, ...]:
            item = value
            return (
                item.population_size,
                tuple(item.eligible_cohort_indices),
                tuple(sorted(dict(item.rejection_reason_counts).items())),
                item.primary_intervention_fingerprint,
            )

        current_selection = store._require_waymax_selection_receipt().to_dict()
        current_selection.pop("mode", None)
        predecessor_selection = (
            eligibility.waymax_selection_receipt.to_dict()
        )
        predecessor_selection.pop("mode", None)
        if (
            primary_projection(receipt) != primary_projection(eligibility.receipt)
            or current_selection != predecessor_selection
            or tuple(receipt.secondary_b4_cohort_indices)
            != tuple(pilot.receipt.secondary_b4_cohort_indices)
        ):
            fail("awaiting-review cohort/selection differs from predecessors")

        verification = results_module.M6MechanicalVerificationReceipt.from_dict(
            results_module._decode_canonical_mapping(
                results_module._read_guarded_bytes(
                    store.run_path / results_module.REVIEW_REQUEST_PATH,
                    store.run_path,
                ),
                "stored mechanical verification",
            )
        )
        verification.revalidate()
        return verification
    except M6OfficialCommandError:
        raise
    except Exception as exc:
        fail("awaiting-review precursor re-verification failed", exc)
    raise AssertionError("unreachable")


def finalize_m6_review(
    finalization: M6ReviewFinalizationRequest,
    holder: _RunHolder,
    *,
    live_lookup: Callable[[Path], str] = _live_main,
    live_approval_lookup: Callable[[Path], str] = _live_approved_commit,
    m4_verifier: Callable[[Path, Path], object] | None = None,
    m4_reverifier: Callable[[object], None] | None = None,
    shard_reverifier: Callable[[M6LocalInputPreflight], None] | None = None,
    version_resolver: Callable[[str], str] | None = None,
    module_importer: Callable[[str], object] | None = None,
    distribution_resolver: Callable[[str], object] | None = None,
    python_version_resolver: Callable[[], str] | None = None,
    environment_distributions_resolver: (
        Callable[[], Sequence[object]] | None
    ) = None,
) -> _PreparedOfficialRun:
    """Finalize explicit reviews without re-entering the outcome executor."""

    if type(finalization) is not M6ReviewFinalizationRequest:
        raise TypeError("finalization must be M6ReviewFinalizationRequest")
    if not isinstance(holder, _RunHolder):
        raise TypeError("holder must be the command's exact run holder")
    request = finalization.command
    repository = preflight_repository(
        request,
        live_lookup=live_lookup,
        live_approval_lookup=live_approval_lookup,
    )
    results_module, preimport_environment = _load_authenticated_review_results(
        repository,
        environment_distributions_resolver=(
            environment_distributions_resolver
        ),
    )
    try:
        store = results_module.M6ResultStore.adopt_awaiting_review(
            repository.root,
            request.run_name,
        )
    except Exception as exc:
        raise M6OfficialCommandError(
            "result_store_failed",
            "the sealed AWAITING_REVIEW store could not be adopted",
        ) from exc
    holder.store = store
    runtime_kwargs: dict[str, object] = {
        "version_resolver": version_resolver,
        "module_importer": module_importer,
        "distribution_resolver": distribution_resolver,
        "python_version_resolver": python_version_resolver,
        "environment_distributions_resolver": (
            environment_distributions_resolver
        ),
    }
    local = preflight_local_inputs(
        request,
        repository,
        store,
        m4_verifier=m4_verifier,
        _allow_awaiting_review=True,
        **runtime_kwargs,
    )
    if preimport_environment is not None and (
        local.runtime._environment_catalog,
        local.runtime._environment_infrastructure,
    ) != preimport_environment:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "the complete environment changed across the review import gate",
        )
    verification = _reverify_awaiting_review_precursor(
        request,
        local,
        store,
        results_module,
    )
    try:
        decisions = tuple(
            results_module.issue_m6_review_decision(
                verification,
                role=item.role,
                decision=item.decision,
                p1_count=item.p1_count,
                p2_count=item.p2_count,
                p3_count=item.p3_count,
            )
            for item in finalization.reviews
        )
        store.write_review_decisions(verification, decisions)
        store.write_execution_summary(
            fresh_worker_peak_rss_bytes=(
                store.awaiting_review_fresh_worker_peak_rss_bytes
            )
        )
        summary_rows = store._read_dataset_rows(
            results_module.EXECUTION_SUMMARY
        )
    except M6OfficialCommandError:
        raise
    except Exception as exc:
        raise M6OfficialCommandError(
            "result_store_failed",
            "the explicit review evidence could not be sealed",
        ) from exc
    all_accepted = all(
        item.decision == "accept"
        and item.p1_count == 0
        and item.p2_count == 0
        for item in finalization.reviews
    )
    if len(summary_rows) != 1:
        raise M6OfficialCommandError(
            "verification_failed",
            "review execution summary is not singular",
        )
    if not all_accepted:
        if summary_rows[0]["release_gate_status"] != "rejected":
            raise M6OfficialCommandError(
                "verification_failed",
                "a rejected review did not close the release gate",
            )
        raise M6OfficialCommandError(
            "review_rejected",
            "one or more independent reviewer roles rejected the result",
        )
    if summary_rows[0]["release_gate_status"] != "accepted":
        raise M6OfficialCommandError(
            "verification_failed",
            "accepted reviews did not satisfy the release gate",
        )
    try:
        store.commit()
    except Exception as exc:
        raise M6OfficialCommandError(
            "result_store_failed",
            "the accepted reviewed result could not be committed",
        ) from exc
    return _PreparedOfficialRun(
        request=request,
        repository=repository,
        local=local,
        store=store,
        results_module=results_module,
        live_lookup=live_lookup,
        live_approval_lookup=live_approval_lookup,
        m4_reverifier=m4_reverifier,
        runtime_kwargs=MappingProxyType(runtime_kwargs),
        shard_reverifier=shard_reverifier,
    )


def prepare_m6_official_run(
    request: M6CommandRequest,
    holder: _RunHolder,
    *,
    eligibility_executor: M6EligibilityExecutor | None = None,
    pilot_executor: M6ComputePilotExecutor | None = None,
    official_executor: M6OfficialExecutor | None = None,
    live_lookup: Callable[[Path], str] = _live_main,
    live_approval_lookup: Callable[[Path], str] = _live_approved_commit,
    m4_verifier: Callable[[Path, Path], object] | None = None,
    m4_reverifier: Callable[[object], None] | None = None,
    shard_reverifier: Callable[[M6LocalInputPreflight], None] | None = None,
    version_resolver: Callable[[str], str] | None = None,
    module_importer: Callable[[str], object] | None = None,
    distribution_resolver: Callable[[str], object] | None = None,
    python_version_resolver: Callable[[], str] | None = None,
    predecessor_gate_factory: M6PredecessorGateFactory | None = None,
    environment_distributions_resolver: (
        Callable[[], Sequence[object]] | None
    ) = None,
) -> _PreparedOfficialRun | _AwaitingReviewResult:
    """Prepare one mode; official results pause sealed awaiting review.

    This is the dependency-injection seam used by tests and production authorities.
    Public dispatch supplies the reviewed source-only, pilot, or official authority.
    Repository authorization and PENDING creation still precede runtime, M4, WOMD,
    policy, and outcome access.
    """

    if not isinstance(request, M6CommandRequest):
        raise TypeError("request must be M6CommandRequest")
    if not isinstance(holder, _RunHolder):
        raise TypeError("holder must be the command's exact run holder")
    executor = _selected_executor(
        request.mode,
        eligibility_executor=eligibility_executor,
        pilot_executor=pilot_executor,
        official_executor=official_executor,
    )
    repository = preflight_repository(
        request,
        live_lookup=live_lookup,
        live_approval_lookup=live_approval_lookup,
    )

    sys.dont_write_bytecode = True
    try:
        reservation = _create_m6_pending_reservation(request, repository)
    except FileExistsError as exc:
        raise M6OfficialCommandError(
            "result_store_failed",
            "the requested M6 result path already exists",
        ) from exc
    except BaseException as exc:
        failed = repository.root / "outputs" / "m6" / request.run_name
        marker = failed / "TERMINAL_FAILURE"
        try:
            metadata = marker.lstat()
            if (
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_nlink == 1
                and marker.resolve(strict=True) == marker
            ):
                holder.store = _FailedCreationView(
                    project_root=repository.root,
                    run_name=request.run_name,
                    run_path=failed,
                )
        except OSError:
            pass
        raise M6OfficialCommandError(
            "result_store_failed",
            "the exclusive ignored M6 PENDING store could not be created",
        ) from exc

    # PENDING is durable and failure-capable before any third-party import. Keep
    # site-packages off sys.path while stdlib importlib.metadata authenticates every
    # installed file, including the editable EvalSim distribution.
    holder.store = reservation
    preimport_environment = None
    if _ACTIVE_BOOTSTRAP_CONTEXT is not None:
        bootstrap = _require_active_bootstrap_context()
        if any(
            module_name in sys.modules
            for module_name in _RUNTIME_MODULES.values()
        ):
            raise M6OfficialCommandError(
                "runtime_mismatch",
                "an optional runtime module loaded before the PENDING/catalog gate",
            )
        preimport_environment_resolver = (
            (
                lambda: tuple(
                    importlib.metadata.distributions(
                        path=[os.fspath(bootstrap.site_packages)]
                    )
                )
            )
            if environment_distributions_resolver is None
            else environment_distributions_resolver
        )
        preimport_environment = _complete_environment_catalog(
            repository.root,
            preimport_environment_resolver,
        )
        _enable_active_bootstrap_site_packages()
    try:
        results_module = _load_m6_results_module()

        if (
            results_module.M6_RESULT_STORE_SCHEMA_VERSION
            != _M6_RESULT_STORE_SCHEMA_VERSION
        ):
            raise M6OfficialCommandError(
                "result_store_failed",
                "stdlib reservation and result-store schemas differ",
            )
        store = results_module.M6ResultStore.adopt_pending(reservation)
    except M6OfficialCommandError:
        raise
    except BaseException as exc:
        raise M6OfficialCommandError(
            "result_store_failed",
            "the stdlib PENDING reservation could not be adopted",
        ) from exc
    holder.store = store

    runtime_kwargs: dict[str, object] = {
        "version_resolver": version_resolver,
        "module_importer": module_importer,
        "distribution_resolver": distribution_resolver,
        "python_version_resolver": python_version_resolver,
        "environment_distributions_resolver": environment_distributions_resolver,
    }
    local = preflight_local_inputs(
        request,
        repository,
        store,
        m4_verifier=m4_verifier,
        version_resolver=version_resolver,
        module_importer=module_importer,
        distribution_resolver=distribution_resolver,
        python_version_resolver=python_version_resolver,
        environment_distributions_resolver=environment_distributions_resolver,
    )
    if preimport_environment is not None and (
        local.runtime._environment_catalog,
        local.runtime._environment_infrastructure,
    ) != preimport_environment:
        raise M6OfficialCommandError(
            "runtime_mismatch",
            "the complete environment changed across the first optional import",
        )
    gate_factory = (
        _build_m6_predecessor_gate
        if predecessor_gate_factory is None
        else predecessor_gate_factory
    )
    try:
        predecessor_gate = gate_factory(request, local, results_module)
    except M6OfficialCommandError:
        raise
    except Exception as exc:
        raise M6OfficialCommandError(
            "verification_failed",
            "predecessor verification failed",
        ) from exc
    if not callable(predecessor_gate):
        raise M6OfficialCommandError(
            "verification_failed",
            "predecessor verification omitted its preregistration gate",
        )
    preregistered_rows: tuple[Mapping[str, Any], ...] | None = None
    preregistered_selection: object | None = None

    def preregister(
        rows: Sequence[Mapping[str, Any]],
        selection: object,
    ) -> None:
        nonlocal preregistered_rows, preregistered_selection
        if preregistered_rows is not None:
            raise M6OfficialCommandError(
                "result_contract_failed",
                "the eligibility/selection gate was invoked more than once",
            )
        try:
            frozen_rows = tuple(
                MappingProxyType(dict(row)) for row in rows
            )
        except (TypeError, ValueError) as exc:
            raise M6OfficialCommandError(
                "result_contract_failed",
                "preregistered eligibility evidence is malformed",
            ) from exc
        if len(frozen_rows) != 128:
            raise M6OfficialCommandError(
                "result_contract_failed",
                "preregistered eligibility evidence is incomplete",
            )
        predecessor_gate(frozen_rows, selection)
        try:
            store.write_eligibility_ledger(frozen_rows)
            store.write_waymax_qualification(selection)
        except M6OfficialCommandError:
            raise
        except Exception as exc:
            raise M6OfficialCommandError(
                "result_store_failed",
                "eligibility/selection receipts could not be sealed",
            ) from exc
        preregistered_rows = frozen_rows
        preregistered_selection = selection

    def begin_outcomes() -> None:
        if preregistered_rows is None or preregistered_selection is None:
            raise M6OfficialCommandError(
                "result_contract_failed",
                "outcome execution began before eligibility/selection sealing",
            )
        if holder.outcome_started:
            raise M6OfficialCommandError(
                "result_contract_failed",
                "the outcome boundary was crossed more than once",
            )
        holder.outcome_started = True

    started = time.monotonic_ns()
    try:
        evidence = executor(request, local, preregister, begin_outcomes)
    except M6OfficialCommandError:
        raise
    except Exception as exc:
        raise M6OfficialCommandError(
            "execution_failed",
            "the injected mode execution authority failed",
        ) from exc
    elapsed_ms = _positive_elapsed_ms(started)
    if type(evidence) is not M6ModeExecutionEvidence or evidence.mode != request.mode:
        raise M6OfficialCommandError(
            "result_contract_failed",
            "the execution authority returned mismatched mode evidence",
        )
    if evidence.mode == "compute_pilot":
        try:
            evidence.revalidate_pilot(
                run_name=request.run_name,
                result_path=local.result_relative,
                selection=preregistered_selection,
                verified_provenance=local.verified_provenance,
            )
        except (TypeError, ValueError) as exc:
            raise M6OfficialCommandError(
                "result_contract_failed",
                "compute-pilot evidence failed its runner issuance binding",
            ) from exc
    if (
        preregistered_rows is None
        or evidence.selection is not preregistered_selection
        or tuple(dict(row) for row in evidence.eligibility_rows)
        != tuple(dict(row) for row in preregistered_rows)
    ):
        raise M6OfficialCommandError(
            "result_contract_failed",
            "execution evidence differs from its sealed preregistration",
        )
    try:
        review_request = _write_mode_artifacts(
            store,
            evidence,
            local,
            results_module=results_module,
        )
        if evidence.mode == "official":
            assert evidence.fresh_worker_peak_rss_bytes is not None
            verification = store.seal_awaiting_review(
                fresh_worker_peak_rss_bytes=(
                    evidence.fresh_worker_peak_rss_bytes
                )
            )
            if review_request.to_dict() != verification.to_dict():
                raise M6OfficialCommandError(
                    "result_contract_failed",
                    "sealed review request identity drifted",
                )
            assert verification.verification_sha256 is not None
            return _AwaitingReviewResult(
                mode="official",
                result_relative=store.project_relative_path,
                evidence_catalog_sha256=(
                    verification.evidence_catalog_sha256
                ),
                mechanical_verification_sha256=(
                    verification.verification_sha256
                ),
            )
        store.commit()
    except M6OfficialCommandError:
        raise
    except Exception as exc:
        raise M6OfficialCommandError(
            "result_store_failed",
            "the exact mode evidence could not be committed",
        ) from exc
    return _PreparedOfficialRun(
        request=request,
        repository=repository,
        local=local,
        store=store,
        results_module=results_module,
        live_lookup=live_lookup,
        live_approval_lookup=live_approval_lookup,
        m4_reverifier=m4_reverifier,
        runtime_kwargs=MappingProxyType(runtime_kwargs),
        shard_reverifier=shard_reverifier,
        eligibility_duration_ms=elapsed_ms,
    )


def _write_mode_artifacts(
    store: object,
    evidence: M6ModeExecutionEvidence,
    local: M6LocalInputPreflight,
    *,
    results_module: object,
) -> object | None:
    if evidence.mode == "compute_pilot":
        evidence.revalidate_pilot(
            run_name=store.run_name,
            result_path=store.project_relative_path.as_posix(),
            selection=evidence.selection,
            verified_provenance=local.verified_provenance,
        )
        store.write_compute_pilot_summary(evidence)
    elif evidence.mode == "official":
        from evalsim.evaluation.m6_official import M6OfficialNumpyRows
        from evalsim.evaluation.m6_waymax_official import M6WaymaxOfficialEvidence

        numpy_rows = evidence.numpy_rows
        waymax = evidence.waymax_evidence
        if (
            type(numpy_rows) is not M6OfficialNumpyRows
            or not callable(getattr(numpy_rows, "revalidate", None))
            or type(waymax) is not M6WaymaxOfficialEvidence
            or getattr(waymax, "promotable", None)
            is not bool(waymax.selection.supported)
            or waymax.selection is not evidence.selection
            or tuple(numpy_rows.eligibility_rows) != evidence.eligibility_rows
        ):
            raise M6OfficialCommandError(
                "result_contract_failed",
                "official evidence is not runner-issued and promotable",
            )
        numpy_rows.revalidate()
        waymax.revalidate()
        store.write_primary_scene_scalars(numpy_rows.primary_scene_scalar_rows)
        store.write_primary_matrix()
        store.write_primary_repeat_scene_scalars(
            numpy_rows.primary_repeat_scene_scalar_rows
        )
        store.write_primary_repeat_matrix()
        store.write_secondary_scene_scalars(
            numpy_rows.secondary_scene_scalar_rows
        )
        store.write_secondary_matrix()
        store.write_negative_timing_observations(
            numpy_rows.negative_timing_observation_rows
        )
        store.write_waymax_scene_scalars(waymax)
        store.write_waymax_numpy_comparisons(waymax)
        store.write_waymax_field_comparisons(waymax)
        store.write_waymax_determinism(waymax)
        store.write_waymax_accounting()

    store.write_typed_provenance(local.verified_provenance)
    if evidence.mode == "official":
        assert evidence.stage_durations_ms is not None
        store.write_stage_timings(
            tuple(
                {
                    "stage_name": stage,
                    "duration_ms": evidence.stage_durations_ms[stage],
                }
                for stage in results_module.M6_STAGE_DOMAIN
            )
        )
        store.write_determinism_receipt()
        store.write_claim_limitations()
        return store.write_mechanical_verification_receipt()
    return None


def _verify_committed_semantics(
    prepared: _PreparedOfficialRun,
) -> object:
    module = prepared.results_module
    try:
        verified = module.verify_committed_m6_result_store(
            prepared.store.project_root,
            prepared.store.run_name,
            expected_mode=prepared.request.mode,
        )
    except Exception as exc:
        raise M6OfficialCommandError(
            "verification_failed",
            "the COMMITTED store failed independent reopening",
        ) from exc
    expected_tables = {
        "eligibility_only": {
            module.ELIGIBILITY_LEDGER,
            module.WAYMAX_QUALIFICATION,
            module.TYPED_PROVENANCE,
        },
        "compute_pilot": {
            module.ELIGIBILITY_LEDGER,
            module.WAYMAX_QUALIFICATION,
            module.COMPUTE_PILOT_SUMMARY,
            module.TYPED_PROVENANCE,
        },
    }
    if prepared.request.mode in expected_tables:
        if set(verified.tables) != expected_tables[prepared.request.mode]:
            raise M6OfficialCommandError(
                "verification_failed",
                "non-outcome mode reopened with an unexpected table",
            )
    else:
        execution = verified.read_dataset(module.EXECUTION_SUMMARY).to_pylist()
        if (
            len(execution) != 1
            or execution[0]["deterministic_repeat_status"] != "passed"
            or execution[0]["waymax_gate_status"]
            not in {"accepted", "unsupported"}
            or execution[0]["release_gate_status"] != "accepted"
        ):
            raise M6OfficialCommandError(
                "verification_failed",
                "official committed semantic gates are not accepted",
            )
    return verified


def _success_result_from_verified(
    prepared: _PreparedOfficialRun,
    verified: object,
) -> _CommandResult:
    module = prepared.results_module
    receipt = verified.receipt
    counts = {
        "population_n": receipt.population_size,
        "primary_eligible_n": receipt.eligible_count,
        **{
            f"primary_rejection_{reason}_n": count
            for reason, count in receipt.rejection_reason_counts.items()
        },
    }
    if prepared.request.mode == "compute_pilot":
        pilot = verified.read_dataset(
            module.COMPUTE_PILOT_SUMMARY
        ).to_pylist()[0]
        counts["pilot_scene_n"] = pilot["pilot_scene_n"]
        durations = {
            name: pilot[f"{name}_ms"]
            for name in ("decode", "numpy", "verification", "waymax")
        }
        durations["total_wall"] = pilot["total_wall_ms"]
    elif prepared.request.mode == "official":
        qualification = verified.read_dataset(
            module.WAYMAX_QUALIFICATION
        ).to_pylist()
        counts.update(
            {
                "secondary_b4_feasible_n": receipt.secondary_b4_count,
                "waymax_qualified_n": sum(
                    row["assessment_status"] == "qualified"
                    for row in qualification
                ),
                "waymax_selected_n": sum(
                    row["selected"] is True for row in qualification
                ),
            }
        )
        durations = {
            row["stage_name"]: row["duration_ms"]
            for row in verified.read_dataset(module.STAGE_TIMINGS).to_pylist()
        }
    else:
        durations = {"eligibility": prepared.eligibility_duration_ms}
    return _CommandResult(
        mode=prepared.request.mode,
        result_relative=prepared.store.project_relative_path,
        aggregate_counts=MappingProxyType(dict(sorted(counts.items()))),
        stage_durations_ms=MappingProxyType(dict(sorted(durations.items()))),
    )


def _finalize_and_terminalize(
    value: _CommandResult | _PreparedOfficialRun | _AwaitingReviewResult,
) -> None:
    if isinstance(value, (_CommandResult, _AwaitingReviewResult)):
        return
    if not isinstance(value, _PreparedOfficialRun):
        raise M6OfficialCommandError(
            "result_contract_failed",
            "captured execution omitted its exact prepared store",
        )
    verified = _verify_committed_semantics(value)
    status_result = _success_result_from_verified(value, verified)
    value.success_payload = _success_output(status_result)

    fresh_repository = reverify_repository_preflight(
        value.request,
        value.repository,
        live_lookup=value.live_lookup,
        live_approval_lookup=value.live_approval_lookup,
    )
    fresh_local = reverify_local_inputs(
        value.request,
        fresh_repository,
        value.store,
        value.local,
        m4_reverifier=value.m4_reverifier,
        **dict(value.runtime_kwargs),
    )
    try:
        (
            _reverify_accepted_shards
            if value.shard_reverifier is None
            else value.shard_reverifier
        )(fresh_local)
    except M6OfficialCommandError:
        raise
    except Exception as exc:
        raise M6OfficialCommandError(
            "shard_set_invalid",
            "the accepted shard contents changed before terminalization",
        ) from exc
    fresh_provenance = reverify_verified_provenance(
        value.request,
        fresh_repository,
        fresh_local,
        value.results_module,
    )
    (
        rebound,
        manifest_sha256,
        committed_sha256,
        evidence_catalog_sha256,
        provenance_context_sha256,
    ) = value.results_module._verified_committed_terminal_binding(value.store)
    if rebound.run_path != verified.run_path:
        raise M6OfficialCommandError(
            "verification_failed",
            "final committed verification resolved a different result store",
        )
    observed = value.results_module._expected_m6_observed_preflight(
        mode=value.request.mode,
        result_path=value.store.project_relative_path.as_posix(),
        manifest_sha256=manifest_sha256,
        committed_sha256=committed_sha256,
        evidence_catalog_sha256=evidence_catalog_sha256,
        provenance_context_sha256=provenance_context_sha256,
    )
    capability = value.results_module._mint_m6_terminal_capability(
        value.store,
        observed,
        fresh_provenance,
    )
    # No code may follow this call: creating TERMINAL_SUCCESS is the last fallible
    # filesystem action inside the silent terminal-commit interval.
    return value.store.mark_terminal_success(capability=capability)


def _prepared_success_payload(value: _PreparedOfficialRun) -> bytes:
    if type(value.success_payload) is not bytes:
        raise M6OfficialCommandError(
            "verification_failed",
            "terminal commit omitted the mechanically derived status",
        )
    return value.success_payload


def _reverify_accepted_shards(
    local: M6LocalInputPreflight,
    *,
    digest_resolver: Callable[[Path], str] | None = None,
    cache_clearer: Callable[[], None] | None = None,
) -> None:
    """Rehash the exact ten accepted shards against the frozen M4 manifest."""

    if (
        type(local) is not M6LocalInputPreflight
        or local._factory_sentinel is not _PREFLIGHT_SENTINEL
    ):
        raise M6OfficialCommandError(
            "shard_set_invalid",
            "shard recheck requires exact local preflight evidence",
        )
    if digest_resolver is None or cache_clearer is None:
        from evalsim.sources.waymax_loader import (
            clear_shard_digest_cache,
            m4_shard_sha256,
        )

        digest_resolver = (
            m4_shard_sha256 if digest_resolver is None else digest_resolver
        )
        cache_clearer = (
            clear_shard_digest_cache if cache_clearer is None else cache_clearer
        )
    expected: dict[str, str] = {}
    try:
        for event in local.accepted_m4.manifest.events:
            prior = expected.setdefault(event.shard_suffix, event.shard_sha256)
            if prior != event.shard_sha256:
                raise ValueError("manifest shard digest is contradictory")
        if set(expected) != {f"{index:05d}" for index in range(10)}:
            raise ValueError("manifest shard domain is incomplete")
        cache_clearer()
        for path, node in zip(
            local.shard_paths,
            local.shard_identities,
            strict=True,
        ):
            suffixes = tuple(
                suffix
                for suffix in expected
                if path.name.endswith(f"tfrecord-{suffix}-of-00150")
            )
            if (
                len(suffixes) != 1
                or node.relative_name != path.name
                or digest_resolver(path) != expected[suffixes[0]]
            ):
                raise ValueError("accepted shard digest changed")
        if _shard_identities(local.shard_paths) != local.shard_identities:
            raise ValueError("accepted shard identity changed during final hash")
    except Exception as exc:
        raise M6OfficialCommandError(
            "shard_set_invalid",
            "the ten accepted shard hashes failed final verification",
        ) from exc


def _fail_store(store: object | None, code: str) -> str | None:
    if store is None:
        return None
    expected = f"outputs/m6/{store.run_name}/TERMINAL_FAILURE"
    try:
        marker = store.fail(
            code if code in _TRUSTED_CODES else "unexpected_failure"
        )
    except BaseException:
        marker = store.run_path / "TERMINAL_FAILURE"
    try:
        metadata = marker.lstat()
        relative = marker.relative_to(store.project_root).as_posix()
        if (
            relative != expected
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or marker.resolve(strict=True) != marker
        ):
            return None
    except (OSError, ValueError):
        return None
    return expected


def _invalidate_holder_store_after_status_failure(
    store: object | None,
    code: str,
) -> bool:
    """Best-effort terminal invalidation for an undeliverable status."""

    if store is None:
        return True
    trusted = code if code in _TRUSTED_CODES else "terminal_capture_failed"
    if _fail_store(store, trusted) is not None:
        return True
    invalidator = getattr(store, "_invalidate_terminal_status_failure", None)
    if not callable(invalidator):
        return False
    try:
        marker = invalidator(trusted)
        metadata = marker.lstat()
        relative = marker.relative_to(store.project_root).as_posix()
        return (
            relative == f"outputs/m6/{store.run_name}/TERMINAL_FAILURE"
            and stat.S_ISREG(metadata.st_mode)
            and metadata.st_nlink == 1
            and marker.resolve(strict=True) == marker
        )
    except BaseException:
        return False


def _persist_failure_diagnostic(
    store: object | None,
    code: str,
    exc: BaseException,
    transcript: bytes,
    *,
    redact_transcript: bool = False,
) -> None:
    if store is None or not hasattr(store, "run_path"):
        return
    error_type = type(exc).__name__.encode("ascii", errors="replace")[:256]
    frames: list[bytes] = []
    traceback_cursor = exc.__traceback__
    while traceback_cursor is not None and len(frames) < 256:
        frame_code = traceback_cursor.tb_frame.f_code
        function = re.sub(
            rb"[^0-9A-Za-z_.<>-]",
            b"_",
            frame_code.co_name.encode("ascii", errors="replace"),
        )[:128]
        frames.append(
            b"frame="
            + str(len(frames)).encode("ascii")
            + b",function="
            + function
            + b",line="
            + str(traceback_cursor.tb_lineno).encode("ascii")
            + b"\n"
        )
        traceback_cursor = traceback_cursor.tb_next
    traceback_payload = b"".join(frames)
    header = (
        b"evalsim-m6-local-failure-v1\n"
        + code.encode("ascii", errors="replace")[:128]
        + b"\n"
        + error_type
        + b"\n--- sanitized traceback ---\n"
        + traceback_payload
        + b"--- bounded terminal transcript ---\n"
    )
    if redact_transcript:
        transcript = (
            b"[post-outcome transcript redacted; captured_byte_count="
            + str(len(transcript)).encode("ascii")
            + b"]\n"
        )
    available = max(0, _MAX_FAILURE_DIAGNOSTIC_BYTES - len(header))
    body = transcript[:available]
    if len(transcript) > available:
        suffix = b"\n...[diagnostic truncated]...\n"
        body = body[: max(0, available - len(suffix))] + suffix
    payload = header + body
    path = store.run_path / _FAILURE_DIAGNOSTIC_NAME
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("diagnostic accepted no bytes")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except BaseException:
        pass
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "M6ComputePilotExecutor",
    "M6CommandRequest",
    "M6EligibilityExecutor",
    "M6GitSnapshot",
    "M6LocalInputPreflight",
    "M6OfficialCommandError",
    "M6OfficialExecutor",
    "M6PredecessorGate",
    "M6PredecessorGateFactory",
    "M6Preregistrar",
    "M6OutcomeBoundary",
    "M6RepositoryPreflight",
    "M6ReviewFinalizationRequest",
    "M6ReviewInput",
    "M6RuntimeObservation",
    "M6ModeExecutionEvidence",
    "M6_COMPUTE_PILOT_EVIDENCE_SCHEMA_VERSION",
    "M6_OFFICIAL_DIRECT_COMMAND",
    "M6_OFFICIAL_MODES",
    "M6_OFFICIAL_PROFILE",
    "M6_OFFICIAL_STATUS_SCHEMA_VERSION",
    "finalize_m6_review",
    "main",
    "prepare_m6_official_run",
    "run_m6_compute_pilot_execution",
    "run_m6_eligibility_only_execution",
    "run_m6_official_execution",
    "preflight_local_inputs",
    "preflight_repository",
    "reverify_local_inputs",
    "reverify_repository_preflight",
    "reverify_verified_provenance",
]
