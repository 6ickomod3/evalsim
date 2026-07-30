"""Capture-first, data-free M6 end-to-end acceptance command.

The command executes exactly two fresh synthetic acceptance passes, projects them
through the official M6 row adapters, and seals one data-free M6 result store. It has
no data-directory surface and never imports JAX, TensorFlow, Waymax, or Flax.

Implementation authorization is external: the verifier requires the fixed lightweight
local and live approval tag at the exact clean pushed HEAD. The resulting immutable
authorization context, including its guarded source catalog, is fixed before either
outcome pass. The sealed evidence precursor is mechanically verified, but data-free
evidence records no independent result-review decisions and remains nonpromotable.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import re
import stat
import sys
import time
import traceback
from typing import Any, TypeAlias, TypeVar

import numpy as np
import pyarrow as pa

from evalsim.evaluation.m6_official import (
    m6_eligibility_rows,
    m6_negative_timing_observation_rows,
    m6_primary_scene_scalar_rows,
    m6_secondary_scene_scalar_rows,
)
from evalsim.evaluation.m6_synthetic import (
    M6SyntheticAcceptanceResult,
    run_m6_synthetic_acceptance,
    synthetic_m6_source_evidence,
)
from evalsim.results.m6 import (
    COMMITTED_MARKER,
    DATA_FREE_MODE,
    EXECUTION_SUMMARY,
    M6_CONFIG_VERSION,
    M6_PLAN_VERSION,
    M6_PRIMARY_INTERVENTION_FINGERPRINT,
    M6_RESULT_STORE_SCHEMA_VERSION,
    M6_REVIEW_ROLE_DOMAIN,
    M6_SECONDARY_INTERVENTION_FINGERPRINT,
    M6_STAGE_DOMAIN,
    REVIEW_DECISIONS,
    TERMINAL_FAILURE_MARKER,
    TYPED_PROVENANCE,
    M6ResultStore,
    M6VerifiedProvenance,
    VerifiedM6ResultStore,
    _issue_m6_verified_provenance,
    verify_committed_m6_result_store,
)
from evalsim.stats.m6 import M6_STATISTICS_SCHEMA_VERSION

from . import m6_official as _official
from ._terminal import (
    TerminalBoundaryError,
    TerminalStatus,
    TerminalizedFailure,
    capture_terminal,
    write_all,
)


M6_SYNTHETIC_STATUS_SCHEMA_VERSION = "m6-synthetic-cli-status-1.0.0"
M6_SYNTHETIC_PROFILE = "data_free_m6"

_RUN_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_SAFE_REASON = re.compile(r"[a-z][a-z0-9_]{0,127}")
_APPROVAL_SENTINEL = object()
_MAX_DIAGNOSTIC_BYTES = 2 * 1024 * 1024
_TRUSTED_CODES = frozenset(
    {
        "approved_commit_mismatch",
        "argument_error",
        "determinism_failed",
        "dirty_worktree",
        "evaluation_failed",
        "evidence_adapter_failed",
        "finalization_failed",
        "git_remote_invalid",
        "output_exists",
        "project_root_invalid",
        "provenance_failed",
        "remote_main_mismatch",
        "result_contract_failed",
        "result_store_failed",
        "source_binding_failed",
        "terminal_capture_failed",
        "terminal_output_detected",
        "unexpected_failure",
        "unpushed_main",
    }
)
_T = TypeVar("_T")


class M6SyntheticCommandError(RuntimeError):
    """A data-free command failure carrying one terminal-safe reason code."""

    def __init__(self, code: str, message: str) -> None:
        if code not in _TRUSTED_CODES:
            raise ValueError("unregistered M6 synthetic command reason code")
        self.code = code
        super().__init__(f"{code}: {message}")


class _SilentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise M6SyntheticCommandError(
            "argument_error",
            "the command arguments do not match the exact synthetic surface",
        )


@dataclass(frozen=True, slots=True)
class M6SyntheticRequest:
    project_root: Path = field(repr=False)
    run_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.project_root, Path):
            raise TypeError("project_root must be a Path")
        _validated_run_name(self.run_name)


@dataclass(frozen=True, slots=True)
class M6ImplementationApproval:
    """Verifier-issued implementation authorization and guarded source context."""

    repository: _official.M6RepositoryPreflight = field(repr=False)
    _factory_sentinel: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_sentinel is not _APPROVAL_SENTINEL:
            raise M6SyntheticCommandError(
                "approved_commit_mismatch",
                "implementation approval is verifier-issued only",
            )
        if (
            type(self.repository) is not _official.M6RepositoryPreflight
            or self.repository.git.approval_ref
            != _official._APPROVED_IMPLEMENTATION_REF
        ):
            raise M6SyntheticCommandError(
                "approved_commit_mismatch",
                "implementation approval context is incomplete",
            )

    @property
    def root(self) -> Path:
        return self.repository.root


ImplementationApprovalProvider: TypeAlias = Callable[
    [Path], M6ImplementationApproval
]
ProvenanceObserver: TypeAlias = Callable[
    [M6ImplementationApproval], M6VerifiedProvenance
]


@dataclass(frozen=True, slots=True)
class _EligibilityRows:
    first: tuple[Mapping[str, Any], ...]
    second: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _OutcomeRows:
    first_primary: tuple[Mapping[str, Any], ...]
    second_primary: tuple[Mapping[str, Any], ...]
    first_secondary: tuple[Mapping[str, Any], ...]
    second_secondary: tuple[Mapping[str, Any], ...]
    first_negative: tuple[Mapping[str, Any], ...]
    second_negative: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    store: M6ResultStore = field(repr=False, compare=False)
    approval: M6ImplementationApproval = field(repr=False)
    provenance: M6VerifiedProvenance = field(repr=False, compare=False)
    approval_provider: ImplementationApprovalProvider = field(
        repr=False,
        compare=False,
    )
    provenance_observer: ProvenanceObserver = field(
        repr=False,
        compare=False,
    )
    success_payload: bytes = field(repr=False)


@dataclass(slots=True)
class _RunHolder:
    store: M6ResultStore | None = field(default=None, repr=False)


def _parser() -> argparse.ArgumentParser:
    parser = _SilentParser(
        prog="evalsim-m6-synthetic",
        add_help=False,
        description=(
            "Run the fixed data-free M6 acceptance twice and seal one local, "
            "ignored result store."
        ),
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    return parser


def _parse_request(argv: Sequence[str] | None) -> M6SyntheticRequest:
    args = _parser().parse_args(argv)
    return M6SyntheticRequest(
        project_root=args.project_root,
        run_name=args.run_name,
    )


def _validated_run_name(value: Any) -> str:
    if (
        type(value) is not str
        or _RUN_NAME.fullmatch(value) is None
        or value in {".", ".."}
    ):
        raise M6SyntheticCommandError(
            "argument_error",
            "run_name must be one safe lowercase path component",
        )
    return value


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _rejection_output(code: str) -> bytes:
    if code not in _TRUSTED_CODES:
        code = "unexpected_failure"
    return _canonical_json_bytes(
        {
            "reason_code": code,
            "schema_version": M6_SYNTHETIC_STATUS_SCHEMA_VERSION,
            "status": "rejected",
        }
    )


def _failure_output(code: str, failure_relative: Path | None) -> bytes:
    if code not in _TRUSTED_CODES:
        code = "unexpected_failure"
    return _canonical_json_bytes(
        {
            "failure_record": _safe_failure_relative(failure_relative),
            "reason_code": code,
            "schema_version": M6_SYNTHETIC_STATUS_SCHEMA_VERSION,
            "status": "failure",
        }
    )


def _safe_result_relative(value: Path) -> str:
    if not isinstance(value, Path):
        raise M6SyntheticCommandError(
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
        raise M6SyntheticCommandError(
            "result_contract_failed",
            "result path lies outside outputs/m6",
        )
    return text


def _safe_failure_relative(value: Path | None) -> str | None:
    if not isinstance(value, Path) or value.name != TERMINAL_FAILURE_MARKER:
        return None
    try:
        parent = _safe_result_relative(value.parent)
    except M6SyntheticCommandError:
        return None
    text = f"{parent}/{TERMINAL_FAILURE_MARKER}"
    return text if value.as_posix() == text else None


def _success_output(result_relative: Path) -> bytes:
    return _canonical_json_bytes(
        {
            "profile": M6_SYNTHETIC_PROFILE,
            "result_path": _safe_result_relative(result_relative),
            "schema_version": M6_SYNTHETIC_STATUS_SCHEMA_VERSION,
            "status": "success",
        }
    )


def _validated_root(candidate: Path) -> Path:
    try:
        root = _official._validated_root(candidate)
        for relative in (".gitignore", "NOTICE.md"):
            path = root / relative
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or path.resolve(strict=True) != path
            ):
                raise OSError("unsafe required repository file")
    except _official.M6OfficialCommandError as exc:
        raise M6SyntheticCommandError(
            "project_root_invalid",
            "the explicit project root is not one canonical checkout",
        ) from exc
    except (OSError, TypeError) as exc:
        raise M6SyntheticCommandError(
            "project_root_invalid",
            "NOTICE.md and .gitignore must be canonical regular files",
        ) from exc
    return root


def _synthetic_error_from_official(
    exc: _official.M6OfficialCommandError,
) -> M6SyntheticCommandError:
    code = exc.code if exc.code in _TRUSTED_CODES else "source_binding_failed"
    return M6SyntheticCommandError(
        code,
        "the external approval/source verifier rejected the checkout",
    )



def _validate_loaded_evalsim_origins(
    root: Path,
    allowed_paths: Sequence[str],
) -> None:
    allowed = set(allowed_paths)
    for name, module in tuple(sys.modules.items()):
        if name != "evalsim" and not name.startswith("evalsim."):
            continue
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str):
            raise M6SyntheticCommandError(
                "source_binding_failed",
                "a loaded EvalSim module has no exact source origin",
            )
        try:
            actual = Path(raw).resolve(strict=True)
            relative = actual.relative_to(root).as_posix()
            metadata = actual.lstat()
        except (OSError, ValueError) as exc:
            raise M6SyntheticCommandError(
                "source_binding_failed",
                "a loaded EvalSim module lies outside the approved checkout",
            ) from exc
        if (
            relative not in allowed
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or actual != root.joinpath(*Path(relative).parts)
        ):
            raise M6SyntheticCommandError(
                "source_binding_failed",
                "a loaded EvalSim module is absent from the guarded catalog",
            )


def _verify_approved_implementation(
    root: Path,
    *,
    live_lookup: Callable[[Path], str] | None = None,
    live_approval_lookup: Callable[[Path], str] | None = None,
    source_paths_resolver: Callable[[Path], tuple[str, ...]] | None = None,
    module_validator: Callable[[Path, Sequence[str]], None] | None = None,
    catalog_builder: Callable[
        [Path, Sequence[str]], tuple[_official._GuardedFileSnapshot, ...]
    ]
    | None = None,
) -> M6ImplementationApproval:
    """Verify external approval and issue one immutable guarded context."""

    live_main = _official._live_main if live_lookup is None else live_lookup
    live_approval = (
        _official._live_approved_commit
        if live_approval_lookup is None
        else live_approval_lookup
    )
    resolve_paths = (
        _official._tracked_source_allowlist
        if source_paths_resolver is None
        else source_paths_resolver
    )
    validate_modules = (
        _validate_loaded_evalsim_origins
        if module_validator is None
        else module_validator
    )
    build_catalog = (
        _official._source_catalog
        if catalog_builder is None
        else catalog_builder
    )
    try:
        git = _official._git_snapshot(
            root,
            live_lookup=live_main,
            live_approval_lookup=live_approval,
        )
        paths = tuple(resolve_paths(root))
        if (
            paths != tuple(sorted(set(paths)))
            or not {".gitignore", "NOTICE.md", "uv.lock"}.issubset(paths)
        ):
            raise _official.M6OfficialCommandError(
                "source_binding_failed",
                "the approved source path catalog is incomplete",
            )
        validate_modules(root, paths)
        catalog = tuple(build_catalog(root, paths))
        source_sha256 = _official._source_fingerprint_from_catalog(catalog)
        uv_lock_sha256 = next(
            item.sha256 for item in catalog if item.relative_path == "uv.lock"
        )
        context_sha256 = _official._repository_context_sha256(
            git,
            paths,
            source_sha256,
            uv_lock_sha256,
        )
        repository = _official.M6RepositoryPreflight(
            root=root,
            git=git,
            source_paths=paths,
            source_sha256=source_sha256,
            source_snapshots=catalog,
            uv_lock_sha256=uv_lock_sha256,
            context_sha256=context_sha256,
            _factory_sentinel=_official._PREFLIGHT_SENTINEL,
        )
    except _official.M6OfficialCommandError as exc:
        raise _synthetic_error_from_official(exc) from exc
    except (OSError, StopIteration, TypeError, ValueError) as exc:
        raise M6SyntheticCommandError(
            "source_binding_failed",
            "guarded approved source verification failed",
        ) from exc
    return M6ImplementationApproval(
        repository=repository,
        _factory_sentinel=_APPROVAL_SENTINEL,
    )


def _build_typed_provenance(
    approval: M6ImplementationApproval,
) -> M6VerifiedProvenance:
    """Issue typed provenance from one externally approved guarded context."""

    if (
        type(approval) is not M6ImplementationApproval
        or approval._factory_sentinel is not _APPROVAL_SENTINEL
    ):
        raise M6SyntheticCommandError(
            "approved_commit_mismatch",
            "typed provenance requires verifier-issued approval",
        )
    repository = approval.repository
    runtime_config = {
        "config_version": M6_CONFIG_VERSION,
        "implementation_authorization": {
            "approval_ref": repository.git.approval_ref,
            "approved_git_commit": repository.git.commit,
            "repository_context_sha256": repository.context_sha256,
        },
        "mode": DATA_FREE_MODE,
        "plan_version": M6_PLAN_VERSION,
        "repository_context_sha256": repository.context_sha256,
        "source_evidence": dict(synthetic_m6_source_evidence()),
        "statistics_schema_version": M6_STATISTICS_SCHEMA_VERSION,
    }
    row = {
        "plan_version": M6_PLAN_VERSION,
        "config_version": M6_CONFIG_VERSION,
        "statistics_schema_version": M6_STATISTICS_SCHEMA_VERSION,
        "population_label": "synthetic_data_free_n10",
        "source_shard_start": None,
        "source_shard_end": None,
        "approved_git_commit": repository.git.commit,
        "git_tree": repository.git.tree,
        "executable_source_sha256": repository.source_sha256,
        "uv_lock_sha256": repository.uv_lock_sha256,
        "runtime_config_sha256": hashlib.sha256(
            b"evalsim-m6-synthetic-runtime-config-v3\x00"
            + _canonical_json_bytes(runtime_config)
        ).hexdigest(),
        "accepted_m4_manifest_sha256": None,
        "accepted_m4_provenance_sha256": None,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pyarrow_version": pa.__version__,
        "jax_version": None,
        "jaxlib_version": None,
        "tensorflow_version": None,
        "waymax_commit": None,
        "jax_backend": None,
        "jax_device_class": None,
        "primary_intervention_fingerprint": (
            M6_PRIMARY_INTERVENTION_FINGERPRINT
        ),
        "secondary_intervention_fingerprint": (
            M6_SECONDARY_INTERVENTION_FINGERPRINT
        ),
    }
    try:
        return _issue_m6_verified_provenance(
            mode=DATA_FREE_MODE,
            row=row,
            source_paths=repository.source_paths,
        )
    except Exception as exc:
        raise M6SyntheticCommandError(
            "provenance_failed",
            "mechanical data-free provenance issuance failed",
        ) from exc


def _elapsed_ms(start_ns: int, end_ns: int) -> int:
    if (
        isinstance(start_ns, bool)
        or isinstance(end_ns, bool)
        or not isinstance(start_ns, int)
        or not isinstance(end_ns, int)
        or end_ns < start_ns
    ):
        raise M6SyntheticCommandError(
            "result_contract_failed",
            "the monotonic stage clock moved backwards",
        )
    return (end_ns - start_ns) // 1_000_000


def _timed(
    stage: str,
    callback: Callable[[], _T],
    timings: dict[str, int],
    clock: Callable[[], int],
) -> _T:
    if stage not in M6_STAGE_DOMAIN or stage in timings:
        raise M6SyntheticCommandError(
            "result_contract_failed",
            "a stage timing label is unknown or repeated",
        )
    started = clock()
    value = callback()
    timings[stage] = _elapsed_ms(started, clock())
    return value


def _eligibility_from_passes(
    first: M6SyntheticAcceptanceResult,
    second: M6SyntheticAcceptanceResult,
) -> _EligibilityRows:
    def rows(
        result: M6SyntheticAcceptanceResult,
    ) -> tuple[Mapping[str, Any], ...]:
        return m6_eligibility_rows(
            result.evaluation.eligibility_ledger,
            mode=DATA_FREE_MODE,
            secondary_plan_ledger=result.evaluation.secondary_plan_ledger,
        )

    return _EligibilityRows(first=rows(first), second=rows(second))


def _outcomes_from_passes(
    first: M6SyntheticAcceptanceResult,
    second: M6SyntheticAcceptanceResult,
) -> _OutcomeRows:
    return _OutcomeRows(
        first_primary=m6_primary_scene_scalar_rows(first.evaluation),
        second_primary=m6_primary_scene_scalar_rows(second.evaluation),
        first_secondary=m6_secondary_scene_scalar_rows(first.evaluation),
        second_secondary=m6_secondary_scene_scalar_rows(second.evaluation),
        first_negative=m6_negative_timing_observation_rows(first.evaluation),
        second_negative=m6_negative_timing_observation_rows(second.evaluation),
    )


def _verify_repeat_evidence(
    first: M6SyntheticAcceptanceResult,
    second: M6SyntheticAcceptanceResult,
    eligibility: _EligibilityRows,
    outcomes: _OutcomeRows,
) -> None:
    if first is second or first.evaluation is second.evaluation:
        raise M6SyntheticCommandError(
            "determinism_failed",
            "the two synthetic passes reused one result object",
        )
    first.revalidate()
    second.revalidate()
    if (
        first.to_local_dict() != second.to_local_dict()
        or eligibility.first != eligibility.second
        or outcomes.first_primary != outcomes.second_primary
        or outcomes.first_secondary != outcomes.second_secondary
        or outcomes.first_negative != outcomes.second_negative
    ):
        raise M6SyntheticCommandError(
            "determinism_failed",
            "the two independent synthetic passes produced different evidence",
        )


def _write_numpy_artifacts(
    store: M6ResultStore,
    eligibility: _EligibilityRows,
    outcomes: _OutcomeRows,
) -> None:
    store.write_eligibility_ledger(eligibility.first)
    store.write_primary_scene_scalars(outcomes.first_primary)
    store.write_primary_matrix()
    store.write_primary_repeat_scene_scalars(outcomes.second_primary)
    store.write_primary_repeat_matrix()
    store.write_secondary_scene_scalars(outcomes.first_secondary)
    store.write_secondary_matrix()
    store.write_negative_timing_observations(outcomes.first_negative)


def _write_waymax_placeholders(store: M6ResultStore) -> None:
    store.write_waymax_qualification()
    store.write_waymax_scene_scalars()
    store.write_waymax_field_comparisons()
    store.write_waymax_numpy_comparisons()
    store.write_waymax_determinism()
    store.write_waymax_accounting()


def _stage_rows(timings: Mapping[str, int]) -> tuple[dict[str, object], ...]:
    if set(timings) != set(M6_STAGE_DOMAIN):
        raise M6SyntheticCommandError(
            "result_contract_failed",
            "the stage timing domain is incomplete",
        )
    return tuple(
        {"stage_name": stage, "duration_ms": timings[stage]}
        for stage in M6_STAGE_DOMAIN
    )


def _prepare_run(
    request: M6SyntheticRequest,
    holder: _RunHolder,
    *,
    approval_provider: ImplementationApprovalProvider | None = None,
    provenance_observer: ProvenanceObserver | None = None,
    synthetic_runner: Callable[[], M6SyntheticAcceptanceResult] | None = None,
    clock: Callable[[], int] | None = None,
) -> _PreparedRun:
    root = _validated_root(request.project_root)
    approve = (
        _verify_approved_implementation
        if approval_provider is None
        else approval_provider
    )
    observe = (
        _build_typed_provenance
        if provenance_observer is None
        else provenance_observer
    )
    try:
        approval = approve(root)
    except M6SyntheticCommandError:
        raise
    except Exception as exc:
        raise M6SyntheticCommandError(
            "approved_commit_mismatch",
            "implementation approval verification failed",
        ) from exc
    if (
        type(approval) is not M6ImplementationApproval
        or approval._factory_sentinel is not _APPROVAL_SENTINEL
        or approval.root != root
    ):
        raise M6SyntheticCommandError(
            "approved_commit_mismatch",
            "approval provider returned unverified or mismatched authority",
        )
    try:
        provenance = observe(approval)
    except M6SyntheticCommandError:
        raise
    except Exception as exc:
        raise M6SyntheticCommandError(
            "provenance_failed",
            "typed data-free provenance construction failed",
        ) from exc
    if type(provenance) is not M6VerifiedProvenance:
        raise M6SyntheticCommandError(
            "provenance_failed",
            "provenance observer returned unverified evidence",
        )
    try:
        store = M6ResultStore.create(root, request.run_name, mode=DATA_FREE_MODE)
    except FileExistsError as exc:
        raise M6SyntheticCommandError(
            "output_exists",
            "the requested M6 run already exists",
        ) from exc
    except Exception as exc:
        raise M6SyntheticCommandError(
            "result_store_failed",
            "the exclusive ignored M6 store could not be created",
        ) from exc
    holder.store = store

    runner = (
        run_m6_synthetic_acceptance
        if synthetic_runner is None
        else synthetic_runner
    )
    monotonic = time.monotonic_ns if clock is None else clock
    timings: dict[str, int] = {}
    try:
        first, second = _timed(
            "numpy_rollouts",
            lambda: (runner(), runner()),
            timings,
            monotonic,
        )
    except Exception as exc:
        raise M6SyntheticCommandError(
            "evaluation_failed",
            "the two fixed synthetic acceptance passes failed",
        ) from exc
    if not isinstance(first, M6SyntheticAcceptanceResult) or not isinstance(
        second,
        M6SyntheticAcceptanceResult,
    ):
        raise M6SyntheticCommandError(
            "evaluation_failed",
            "the synthetic runner returned an invalid typed result",
        )
    try:
        eligibility = _timed(
            "eligibility",
            lambda: _eligibility_from_passes(first, second),
            timings,
            monotonic,
        )
        outcomes = _timed(
            "paired_metrics",
            lambda: _outcomes_from_passes(first, second),
            timings,
            monotonic,
        )
    except Exception as exc:
        raise M6SyntheticCommandError(
            "evidence_adapter_failed",
            "official row adaptation of synthetic evidence failed",
        ) from exc
    _timed(
        "verification",
        lambda: _verify_repeat_evidence(first, second, eligibility, outcomes),
        timings,
        monotonic,
    )
    try:
        _timed(
            "statistics",
            lambda: _write_numpy_artifacts(store, eligibility, outcomes),
            timings,
            monotonic,
        )
        _timed(
            "waymax",
            lambda: _write_waymax_placeholders(store),
            timings,
            monotonic,
        )
        store.write_typed_provenance(provenance)
        store.write_stage_timings(_stage_rows(timings))
        store.write_determinism_receipt()
        store.write_claim_limitations()
        store.write_data_free_review_absence()
        store.write_execution_summary(fresh_worker_peak_rss_bytes=0)
    except M6SyntheticCommandError:
        raise
    except Exception as exc:
        raise M6SyntheticCommandError(
            "result_store_failed",
            "the complete data-free evidence catalog could not be sealed",
        ) from exc
    return _PreparedRun(
        store=store,
        approval=approval,
        provenance=provenance,
        approval_provider=approve,
        provenance_observer=observe,
        success_payload=_success_output(store.project_relative_path),
    )


def _dispatch(argv: Sequence[str] | None, holder: _RunHolder) -> _PreparedRun:
    return _prepare_run(_parse_request(argv), holder)


def _same_verified_provenance(
    left: M6VerifiedProvenance,
    right: M6VerifiedProvenance,
) -> bool:
    left.revalidate()
    right.revalidate()
    return (
        left.mode == right.mode
        and left.source_paths == right.source_paths
        and left.context_sha256 == right.context_sha256
        and left.to_store_row() == right.to_store_row()
    )


def _verify_committed_semantics(
    prepared: _PreparedRun,
    verified: VerifiedM6ResultStore,
) -> None:
    store = prepared.store
    reviews = verified.read_dataset(REVIEW_DECISIONS).to_pylist()
    execution = verified.read_dataset(EXECUTION_SUMMARY).to_pylist()
    provenance_rows = verified.read_dataset(TYPED_PROVENANCE).to_pylist()
    observed_reviews = tuple(reviews)
    if (
        verified.run_path != store.run_path
        or not verified.profile.data_free
        or verified.receipt.eligible_cohort_indices != tuple(range(10))
        or verified.receipt.secondary_b4_cohort_indices != tuple(range(10))
        or observed_reviews != ()
        or len(provenance_rows) != 1
        or provenance_rows[0] != prepared.provenance.to_store_row()
        or len(execution) != 1
        or execution[0]["deterministic_repeat_status"] != "passed"
        or execution[0]["waymax_gate_status"] != "unsupported"
        or execution[0]["real_reactivity_claim_status"] != "blocked"
        or execution[0]["release_gate_status"] != "nonpromotable"
    ):
        raise M6SyntheticCommandError(
            "finalization_failed",
            "the independently reopened COMMITTED store differs from "
            "the prepared semantic contract",
        )



def _reverify_execution_context(prepared: _PreparedRun) -> None:
    fresh_approval = prepared.approval_provider(prepared.store.project_root)
    if fresh_approval != prepared.approval:
        raise M6SyntheticCommandError(
            "source_binding_failed",
            "approval or guarded source identities changed during execution",
        )
    fresh_provenance = prepared.provenance_observer(fresh_approval)
    if not _same_verified_provenance(
        fresh_provenance,
        prepared.provenance,
    ):
        raise M6SyntheticCommandError(
            "source_binding_failed",
            "source/runtime provenance changed during execution",
        )


def _commit_verify_and_mark_success(prepared: _PreparedRun) -> None:
    """Make TERMINAL_SUCCESS only after all independent checks have passed."""

    if not isinstance(prepared, _PreparedRun):
        raise M6SyntheticCommandError(
            "result_contract_failed",
            "the captured command omitted its prepared run",
        )
    store = prepared.store
    try:
        _reverify_execution_context(prepared)
        store.commit()
        verified = verify_committed_m6_result_store(
            store.project_root,
            store.run_name,
            allow_data_free=True,
            expected_mode=DATA_FREE_MODE,
        )
        _verify_committed_semantics(prepared, verified)
        _reverify_execution_context(prepared)
    except M6SyntheticCommandError:
        raise
    except Exception as exc:
        raise M6SyntheticCommandError(
            "finalization_failed",
            "the COMMITTED store failed independent pre-terminal verification",
        ) from exc

    # This is the last filesystem mutation and last fallible semantic operation.
    # mark_terminal_success independently reopens COMMITTED once more internally.
    try:
        store.mark_terminal_success()
    except Exception as exc:
        raise M6SyntheticCommandError(
            "finalization_failed",
            "the verified COMMITTED store could not become terminal success",
        ) from exc


def _failure_code(exc: BaseException) -> str:
    if isinstance(exc, M6SyntheticCommandError) and exc.code in _TRUSTED_CODES:
        return exc.code
    if isinstance(exc, TerminalBoundaryError) and exc.code in _TRUSTED_CODES:
        return exc.code
    return "unexpected_failure"


def _diagnostic_bytes(primary: BaseException, transcript: bytes) -> bytes:
    try:
        formatted = "".join(
            traceback.format_exception(primary)
        ).encode("utf-8", errors="backslashreplace")
    except Exception:
        formatted = b"diagnostic traceback unavailable\n"
    half = _MAX_DIAGNOSTIC_BYTES // 2
    payload = (
        b"=== captured stdout/stderr ===\n"
        + transcript[:half]
        + b"\n=== exception ===\n"
        + formatted[:half]
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
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
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
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
            or metadata.st_size > _MAX_DIAGNOSTIC_BYTES
        ):
            raise OSError("failure diagnostic metadata is unsafe")
    finally:
        os.close(descriptor)


def _read_failure_marker(
    store: M6ResultStore,
) -> tuple[Path, str] | None:
    path = store.run_path / TERMINAL_FAILURE_MARKER
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
            or metadata.st_size <= 0
            or metadata.st_size > 4096
        ):
            return None
        encoded = os.read(descriptor, 4097)
        if len(encoded) != metadata.st_size:
            return None
        payload = json.loads(encoded.decode("ascii", errors="strict"))
        if (
            not isinstance(payload, dict)
            or set(payload)
            != {"mode", "reason_code", "schema_version", "state"}
            or payload["mode"] != DATA_FREE_MODE
            or payload["schema_version"] != M6_RESULT_STORE_SCHEMA_VERSION
            or payload["state"] != "TERMINAL_FAILURE"
            or _canonical_json_bytes(payload) != encoded
            or not isinstance(payload["reason_code"], str)
            or _SAFE_REASON.fullmatch(payload["reason_code"]) is None
        ):
            return None
        relative = store.project_relative_path / TERMINAL_FAILURE_MARKER
        if _safe_failure_relative(relative) is None:
            return None
        return relative, payload["reason_code"]
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        return None
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _persist_failure(
    store: M6ResultStore,
    primary: BaseException,
    transcript: bytes,
) -> tuple[Path | None, str]:
    requested = _failure_code(primary)
    try:
        store.fail(requested)
    except Exception:
        pass
    persisted = _read_failure_marker(store)
    try:
        _write_failure_diagnostics(store.run_path, primary, transcript)
    except OSError:
        pass
    if persisted is None:
        return None, "unexpected_failure"
    relative, stored_code = persisted
    public_code = (
        stored_code if stored_code in _TRUSTED_CODES else "unexpected_failure"
    )
    return relative, public_code


def _emit_status(
    terminal: TerminalStatus | None,
    payload: bytes,
    *,
    error: bool,
) -> bool:
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


def main(argv: Sequence[str] | None = None) -> int:
    """Capture before argparse, commit only after verification, and emit once."""

    holder = _RunHolder()
    try:
        captured = capture_terminal(
            lambda: _dispatch(argv, holder),
            terminal_commit=_commit_verify_and_mark_success,
            seal_terminal=True,
        )
    except TerminalizedFailure as exc:
        code = _failure_code(exc.primary)
        if holder.store is None:
            payload = _rejection_output(code)
        else:
            relative, code = _persist_failure(
                holder.store,
                exc.primary,
                exc.transcript,
            )
            payload = _failure_output(code, relative)
        _emit_status(exc.terminal_status, payload, error=True)
        return 1
    except BaseException as exc:
        code = _failure_code(exc)
        if holder.store is None:
            payload = _rejection_output(code)
        else:
            relative, code = _persist_failure(holder.store, exc, b"")
            payload = _failure_output(code, relative)
        _emit_status(None, payload, error=True)
        return 1
    emitted = _emit_status(
        captured.terminal_status,
        captured.value.success_payload,
        error=False,
    )
    return 0 if emitted else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ImplementationApprovalProvider",
    "M6_SYNTHETIC_PROFILE",
    "M6_SYNTHETIC_STATUS_SCHEMA_VERSION",
    "M6ImplementationApproval",
    "M6SyntheticCommandError",
    "M6SyntheticRequest",
    "main",
]
