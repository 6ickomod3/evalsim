"""Data-free tests for exact reuse of the explicitly trusted local M4 run."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback
from types import MappingProxyType

import numpy as np
import pytest

from evalsim import Agent, AgentType, Scenario
import evalsim.sources.m5_m4_reuse as reuse
from evalsim.sources.m5_m4_reuse import (
    M4AcceptanceReceipt,
    M4ReuseError,
    reverify_accepted_m4_run,
    verify_accepted_m4_run,
    visit_accepted_m4_cohort,
)
from evalsim.sources.waymax_cohort import (
    SOURCE_REJECTION_CODES,
    ScanEvent,
    ShardScanCounts,
    WaymaxCohortManifest,
)
from evalsim.sources.waymax_loader import (
    M4StreamRecord,
    WaymaxRecord,
)


def _manifest(
    *,
    identity_prefix: int = 0,
    first_shard_raw_delta: int = 0,
    first_shard_eligible_delta: int = 0,
) -> WaymaxCohortManifest:
    events: list[ScanEvent] = []
    counts: list[ShardScanCounts] = []
    for shard_index, (
        suffix,
        raw,
        eligible,
        rejected,
        _,
    ) in enumerate(reuse._M4_ACCEPTED_COUNTS):
        if shard_index == 0:
            raw += first_shard_raw_delta
            eligible += first_shard_eligible_delta
        assert raw == eligible + rejected
        shard_sha256 = hashlib.sha256(
            f"invented-shard-{suffix}".encode("ascii")
        ).hexdigest()
        for ordinal in range(raw):
            native_id = (
                f"{identity_prefix:02x}{shard_index:02x}{ordinal:08x}"
            )
            if ordinal < eligible:
                event = ScanEvent.eligible_event(
                    shard_suffix=suffix,
                    record_ordinal=ordinal,
                    native_scenario_id=native_id,
                    shard_sha256=shard_sha256,
                    dataset_config_fingerprint=(
                        reuse.M4_ACCEPTED_DATASET_CONFIG_FINGERPRINT
                    ),
                )
            else:
                event = ScanEvent.rejected_event(
                    shard_suffix=suffix,
                    record_ordinal=ordinal,
                    native_scenario_id=native_id,
                    shard_sha256=shard_sha256,
                    dataset_config_fingerprint=(
                        reuse.M4_ACCEPTED_DATASET_CONFIG_FINGERPRINT
                    ),
                    rejection_code="source_no_supported_map",
                )
            events.append(event)
        counts.append(
            ShardScanCounts(
                shard_suffix=suffix,
                raw_seen=raw,
                decode_attempted=raw,
                event_emitted=raw,
                eligible=eligible,
                rejected=rejected,
                clean_eof=True,
            )
        )
    return WaymaxCohortManifest.build(events=events, shard_counts=counts)


@pytest.fixture(scope="session")
def accepted_manifest() -> WaymaxCohortManifest:
    return _manifest()


@pytest.fixture(scope="session")
def alternate_manifest() -> WaymaxCohortManifest:
    return _manifest(identity_prefix=1)


@pytest.fixture(scope="session")
def count_drift_manifest() -> WaymaxCohortManifest:
    return _manifest(
        first_shard_raw_delta=-1,
        first_shard_eligible_delta=-1,
    )


def _benchmark() -> dict[str, object]:
    durations = [0.01 + index * 0.001 for index in range(20)]
    ordered = sorted(durations)
    median = float((ordered[9] + ordered[10]) / 2.0)
    return {
        "batch_size": 2,
        "compile_seconds": 0.25,
        "device_transfer_before_timing": True,
        "eager_sequential_parity": True,
        "fresh_worker_process": True,
        "horizon_transitions": 80,
        "jit_vmap": True,
        "memory_measurement": (
            "process_high_water_rss_not_jax_device_memory"
        ),
        "median_seconds": median,
        "nearest_rank_p95_seconds": float(ordered[18]),
        "peak_rss_bytes": 123_456,
        "permutation_invariance": True,
        "runs": 20,
        "scenarios_per_second_at_median": 2 / median,
        "warm_durations_seconds": durations,
    }


def _aggregate(manifest: WaymaxCohortManifest) -> dict[str, object]:
    return {
        "accepted": True,
        "benchmark": _benchmark(),
        "checks": {name: True for name in reuse._AGGREGATE_CHECKS},
        "cohort": reuse._cohort_summary(manifest),
        "idm": dict(reuse._M4_ACCEPTED_IDM),
        "privacy": {name: True for name in reuse._AGGREGATE_PRIVACY},
        "purpose": "personal_non_commercial_interview_preparation",
        "runtime": {
            **dict(reuse._M4_ACCEPTED_RUNTIME),
            "jax_backend": "cpu",
            "jax_devices": ["cpu"],
            "platform": "invented-apple-cpu",
        },
        "schema_version": "1",
        "shared_decode_limitation": (
            "EvalSim and Waymax reference paths share the pinned Waymax WOMD decode"
        ),
    }


def _provenance() -> dict[str, object]:
    return {
        "files": dict(reuse._M4_ACCEPTED_EXECUTABLE_SHA256),
        "git_commit": reuse.M4_ACCEPTED_GIT_COMMIT,
        "git_tree": reuse.M4_ACCEPTED_GIT_TREE,
        "reference_config_fingerprint": (
            reuse.M4_ACCEPTED_REFERENCE_CONFIG_FINGERPRINT
        ),
        "schema_version": "1",
        "selector_config_fingerprint": (
            reuse.M4_SELECTOR_CONFIG_FINGERPRINT
        ),
    }


def _pretty(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_run(
    project: Path,
    manifest: WaymaxCohortManifest,
    *,
    second_manifest: WaymaxCohortManifest | None = None,
) -> Path:
    run = project / "outputs" / "m4" / "accepted"
    cohort = run / "cohort"
    cohort.mkdir(parents=True)
    (run / "aggregate-summary.json").write_bytes(
        _pretty(_aggregate(manifest))
    )
    (run / "execution-provenance.json").write_bytes(
        _pretty(_provenance())
    )
    (run / "terminal-output.bin").write_bytes(b"")
    (cohort / "manifest-pass-1.json").write_bytes(
        manifest.canonical_bytes()
    )
    (cohort / "manifest-pass-2.json").write_bytes(
        (second_manifest or manifest).canonical_bytes()
    )
    return run


@pytest.fixture
def accepted_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_manifest: WaymaxCohortManifest,
) -> tuple[Path, Path, WaymaxCohortManifest]:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        "[project]\nname='invented'\nversion='0.0.0'\n",
        encoding="utf-8",
    )
    (project / ".gitignore").write_text("outputs/\n", encoding="utf-8")
    subprocess.run(
        ("git", "init", "-b", "main"),
        cwd=project,
        check=True,
        capture_output=True,
    )
    run = _write_run(project, accepted_manifest)
    monkeypatch.setattr(
        reuse,
        "_verify_historical_snapshot",
        lambda _: None,
    )
    return project, run, accepted_manifest


def _receipt(cohort) -> M4AcceptanceReceipt:
    return M4AcceptanceReceipt(
        aggregate_summary_sha256=(
            cohort.evidence.aggregate_summary_sha256
        ),
        execution_provenance_sha256=(
            cohort.evidence.execution_provenance_sha256
        ),
        manifest_sha256=cohort.evidence.manifest_sha256,
    )


def _rewrite_json(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_bytes(_pretty(payload))


def _scenario(scenario_id: str) -> Scenario:
    steps = 91
    zeros = np.zeros(steps, dtype=float)
    return Scenario(
        scenario_id=scenario_id,
        timestamps=np.arange(steps, dtype=float) * 0.1,
        agents=[
            Agent(
                id=1,
                type=AgentType.VEHICLE,
                valid=np.ones(steps, dtype=bool),
                x=zeros,
                y=zeros,
                heading=zeros,
                vx=zeros,
                vy=zeros,
                length=4.5,
                width=2.0,
            )
        ],
        ego_index=0,
        metadata={
            "current_index": 10,
            "source": "invented",
            "source_fingerprint": "invented",
        },
    )


def test_historical_snapshot_descriptor_matches_repository() -> None:
    project = Path(__file__).resolve().parents[1]
    reuse._verify_historical_snapshot(project)


def test_valid_preflight_receipt_and_private_repr(accepted_run) -> None:
    project, run, manifest = accepted_run
    cohort = verify_accepted_m4_run(project, run)

    assert len(cohort.members) == 128
    assert cohort.manifest == manifest
    assert cohort.receipt_verified is False
    assert cohort.evidence.integrity_assumption == (
        "explicit_ignored_m4_run_is_local_trust_root"
    )
    assert cohort.evidence.runtime_versions == dict(
        reuse._M4_ACCEPTED_RUNTIME
    )
    assert tuple(item.cohort_index for item in cohort.members) == tuple(
        range(128)
    )

    sensitive = cohort.members[0].event
    representation = repr(cohort) + repr(cohort.members[0])
    assert sensitive.native_scenario_id not in representation
    assert sensitive.shard_sha256 not in representation
    assert cohort.evidence.manifest_sha256 not in representation
    assert (
        cohort.evidence.selected_order_fingerprint_sha256
        not in representation
    )
    assert os.fspath(run) not in representation

    pinned = verify_accepted_m4_run(
        project,
        run,
        expected_receipt=_receipt(cohort),
    )
    assert pinned.receipt_verified is True
    assert pinned.evidence == cohort.evidence


def test_receipt_rejects_any_different_artifact_hash(accepted_run) -> None:
    project, run, _ = accepted_run
    cohort = verify_accepted_m4_run(project, run)
    receipt = _receipt(cohort)
    forged = M4AcceptanceReceipt(
        aggregate_summary_sha256="0" * 64,
        execution_provenance_sha256=(
            receipt.execution_provenance_sha256
        ),
        manifest_sha256=receipt.manifest_sha256,
    )
    with pytest.raises(M4ReuseError) as caught:
        verify_accepted_m4_run(
            project,
            run,
            expected_receipt=forged,
        )
    assert caught.value.code == "m4_reuse_receipt_mismatch"


def test_layout_and_symlink_paths_fail_closed(
    accepted_run,
    tmp_path: Path,
) -> None:
    project, run, _ = accepted_run
    extra = run / "invented-extra.json"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(M4ReuseError) as caught:
        verify_accepted_m4_run(project, run)
    assert caught.value.code == "m4_reuse_artifact_layout"
    extra.unlink()

    transcript = run / "terminal-output.bin"
    transcript.write_bytes(b"invented diagnostic")
    with pytest.raises(M4ReuseError) as caught:
        verify_accepted_m4_run(project, run)
    assert caught.value.code == "m4_reuse_artifact_unsafe"
    transcript.write_bytes(b"")

    alias = project / "outputs" / "m4" / "alias"
    alias.symlink_to(run, target_is_directory=True)
    with pytest.raises(M4ReuseError) as caught:
        verify_accepted_m4_run(project, alias)
    assert caught.value.code == "m4_reuse_path_invalid"

    with pytest.raises(M4ReuseError) as caught:
        verify_accepted_m4_run(project, tmp_path)
    assert caught.value.code == "m4_reuse_path_invalid"


def test_committed_force_added_run_is_rejected(accepted_run) -> None:
    project, run, _ = accepted_run
    relative_run = run.relative_to(project).as_posix()
    subprocess.run(
        ("git", "add", "-f", "--", relative_run),
        cwd=project,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=EvalSim Test",
            "-c",
            "user.email=evalsim-test@example.invalid",
            "commit",
            "-m",
            "force-add ignored synthetic run",
        ),
        cwd=project,
        check=True,
        capture_output=True,
    )
    assert subprocess.run(
        (
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            relative_run,
        ),
        cwd=project,
        check=True,
        capture_output=True,
    ).stdout == b""
    assert subprocess.run(
        (
            "git",
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            relative_run,
        ),
        cwd=project,
        check=False,
        capture_output=True,
    ).returncode == 0

    with pytest.raises(M4ReuseError) as caught:
        verify_accepted_m4_run(project, run)
    assert caught.value.code == "m4_reuse_path_invalid"
    assert "tracked files" in str(caught.value)


def test_pretty_json_and_provenance_drift_are_rejected(accepted_run) -> None:
    project, run, _ = accepted_run
    aggregate_path = run / "aggregate-summary.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate_path.write_text(
        json.dumps(aggregate, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(M4ReuseError) as caught:
        verify_accepted_m4_run(project, run)
    assert caught.value.code == "m4_reuse_json_noncanonical"


def test_execution_snapshot_drift_is_rejected(accepted_run) -> None:
    project, run, _ = accepted_run
    _rewrite_json(
        run / "execution-provenance.json",
        lambda payload: payload.update({"git_commit": "0" * 40}),
    )
    with pytest.raises(M4ReuseError) as caught:
        verify_accepted_m4_run(project, run)
    assert caught.value.code == "m4_reuse_snapshot_mismatch"


def test_independently_valid_manifest_pass_drift_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted_manifest: WaymaxCohortManifest,
    alternate_manifest: WaymaxCohortManifest,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project / ".gitignore").write_text("outputs/\n", encoding="utf-8")
    subprocess.run(
        ("git", "init", "-b", "main"),
        cwd=project,
        check=True,
        capture_output=True,
    )
    run = _write_run(
        project,
        accepted_manifest,
        second_manifest=alternate_manifest,
    )
    monkeypatch.setattr(
        reuse,
        "_verify_historical_snapshot",
        lambda _: None,
    )
    with pytest.raises(M4ReuseError) as caught:
        verify_accepted_m4_run(project, run)
    assert caught.value.code == "m4_reuse_manifest_repeat_mismatch"


def test_public_accepted_cohort_accounting_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count_drift_manifest: WaymaxCohortManifest,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (project / ".gitignore").write_text("outputs/\n", encoding="utf-8")
    subprocess.run(
        ("git", "init", "-b", "main"),
        cwd=project,
        check=True,
        capture_output=True,
    )
    run = _write_run(project, count_drift_manifest)
    monkeypatch.setattr(
        reuse,
        "_verify_historical_snapshot",
        lambda _: None,
    )
    with pytest.raises(M4ReuseError) as caught:
        verify_accepted_m4_run(project, run)
    assert caught.value.code == "m4_reuse_cohort_mismatch"


@pytest.mark.parametrize(
    ("mutate", "code"),
    (
        (
            lambda payload: payload.update({"accepted": False}),
            "m4_reuse_not_accepted",
        ),
        (
            lambda payload: payload["checks"].update(
                {"evalsim_cv_full_80": False}
            ),
            "m4_reuse_not_accepted",
        ),
        (
            lambda payload: payload["runtime"].update({"python": "3.12.0"}),
            "m4_reuse_runtime_mismatch",
        ),
        (
            lambda payload: payload["cohort"].update({"selected": 127}),
            "m4_reuse_aggregate_invalid",
        ),
        (
            lambda payload: payload["cohort"].update({"selected": 128.0}),
            "m4_reuse_aggregate_invalid",
        ),
        (
            lambda payload: payload["idm"].update(
                {"requested_controlled_transitions": 8_468}
            ),
            "m4_reuse_aggregate_invalid",
        ),
        (
            lambda payload: payload["idm"].update(
                {"horizon_transitions": 20.0}
            ),
            "m4_reuse_aggregate_invalid",
        ),
        (
            lambda payload: payload["benchmark"].update(
                {"batch_size": 2.0}
            ),
            "m4_reuse_aggregate_invalid",
        ),
    ),
)
def test_aggregate_acceptance_drift_fails_closed(
    accepted_run,
    mutate,
    code: str,
) -> None:
    project, run, _ = accepted_run
    _rewrite_json(run / "aggregate-summary.json", mutate)
    with pytest.raises(M4ReuseError) as caught:
        verify_accepted_m4_run(project, run)
    assert caught.value.code == code


def test_selected_order_fingerprint_binds_order_and_every_event(
    accepted_manifest: WaymaxCohortManifest,
) -> None:
    selected = accepted_manifest.selected_events
    baseline = reuse._selected_order_fingerprint(selected)
    assert baseline == reuse._selected_order_fingerprint(selected)
    permuted = (selected[1], selected[0], *selected[2:])
    assert reuse._selected_order_fingerprint(permuted) != baseline
    changed = (
        replace(
            selected[0],
            shard_sha256="f" * 64,
        ),
        *selected[1:],
    )
    assert reuse._selected_order_fingerprint(changed) != baseline


def test_bounded_reload_visits_exact_canonical_members(
    accepted_run,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, run, _ = accepted_run
    cohort = verify_accepted_m4_run(project, run)
    by_locator = {
        member.event.locator: member for member in cohort.members
    }
    calls: list[tuple[str, int]] = []

    def fake_reload(data_dir, expectations):
        del data_dir
        calls.append(
            (
                expectations[0].locator.shard_suffix,
                len(expectations),
            )
        )
        records = []
        for expectation in expectations:
            locator = expectation.locator
            records.append(
                M4StreamRecord(
                    locator=locator,
                    record=WaymaxRecord(
                        scenario_id=expectation.expected_scenario_id,
                        state=object(),
                        audit=MappingProxyType({}),
                        shard_suffix=locator.shard_suffix,
                        record_ordinal=locator.record_ordinal,
                        shard_sha256=(
                            expectation.expected_shard_sha256
                        ),
                        dataset_config_fingerprint=(
                            expectation
                            .expected_dataset_config_fingerprint
                        ),
                    ),
                )
            )
        return tuple(records)

    def fake_verify(stream_record):
        member = by_locator[
            (
                stream_record.locator.shard_suffix,
                stream_record.locator.record_ordinal,
            )
        ]
        return replace(member.event, selected=False), _scenario(
            stream_record.record.scenario_id
        )

    monkeypatch.setattr(reuse, "_validate_current_runtime", lambda: None)
    monkeypatch.setattr(reuse, "reload_m4_waymax_records", fake_reload)
    monkeypatch.setattr(reuse, "verify_m4_stream_record", fake_verify)
    monkeypatch.setattr(reuse, "_verify_shards_unchanged", lambda *args: None)
    monkeypatch.setattr(
        reuse,
        "reverify_accepted_m4_run",
        lambda _: None,
    )
    visited: list[int] = []
    visit_accepted_m4_cohort(
        cohort,
        Path("invented-data"),
        lambda member: visited.append(member.cohort_index),
    )

    assert visited == list(range(128))
    assert calls == [
        (f"{index:05d}", 13 if index < 8 else 12)
        for index in range(10)
    ]


@pytest.mark.parametrize("failure_type", (RuntimeError, Exception))
def test_current_runtime_failure_is_privacy_safe(
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[Exception],
) -> None:
    sentinel = "private-runtime-sentinel-must-not-escape"

    def fail_runtime_summary() -> dict[str, object]:
        raise failure_type(sentinel)

    monkeypatch.setattr(reuse, "runtime_summary", fail_runtime_summary)
    with pytest.raises(M4ReuseError) as caught:
        reuse._validate_current_runtime()
    assert caught.value.code == "m4_reuse_runtime_mismatch"
    assert sentinel not in str(caught.value)
    assert sentinel not in "".join(
        traceback.format_exception(caught.value)
    )


@pytest.mark.parametrize("failure_type", (KeyboardInterrupt, SystemExit))
def test_current_runtime_does_not_swallow_terminal_base_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[BaseException],
) -> None:
    def stop_runtime_summary() -> dict[str, object]:
        raise failure_type()

    monkeypatch.setattr(reuse, "runtime_summary", stop_runtime_summary)
    with pytest.raises(failure_type):
        reuse._validate_current_runtime()


def test_bounded_reload_rejects_reordering_before_visit(
    accepted_run,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project, run, _ = accepted_run
    cohort = verify_accepted_m4_run(project, run)

    def reversed_reload(data_dir, expectations):
        del data_dir
        records = [
            M4StreamRecord(
                locator=expectation.locator,
                record=WaymaxRecord(
                    scenario_id=expectation.expected_scenario_id,
                    state=object(),
                    audit=MappingProxyType({}),
                    shard_suffix=expectation.locator.shard_suffix,
                    record_ordinal=expectation.locator.record_ordinal,
                    shard_sha256=expectation.expected_shard_sha256,
                    dataset_config_fingerprint=(
                        expectation.expected_dataset_config_fingerprint
                    ),
                ),
            )
            for expectation in expectations
        ]
        return tuple(reversed(records))

    monkeypatch.setattr(reuse, "_validate_current_runtime", lambda: None)
    monkeypatch.setattr(
        reuse,
        "reload_m4_waymax_records",
        reversed_reload,
    )
    with pytest.raises(M4ReuseError) as caught:
        visit_accepted_m4_cohort(
            cohort,
            Path("invented-data"),
            lambda _: pytest.fail("reordered member must not be visited"),
        )
    assert caught.value.code == "m4_reuse_reload_mismatch"


def test_reverification_detects_identity_change_without_byte_change(
    accepted_run,
) -> None:
    project, run, _ = accepted_run
    cohort = verify_accepted_m4_run(project, run)
    artifact = run / "aggregate-summary.json"
    stat_before = artifact.stat()
    os.utime(
        artifact,
        ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns + 1),
    )
    with pytest.raises(M4ReuseError) as caught:
        reverify_accepted_m4_run(cohort)
    assert caught.value.code == "m4_reuse_artifact_changed"


def test_error_text_does_not_echo_private_values(accepted_run) -> None:
    project, run, manifest = accepted_run
    _rewrite_json(
        run / "aggregate-summary.json",
        lambda payload: payload.update({"accepted": False}),
    )
    with pytest.raises(M4ReuseError) as caught:
        verify_accepted_m4_run(project, run)
    message = str(caught.value)
    private_event = manifest.selected_events[0]
    assert os.fspath(run) not in message
    assert private_event.native_scenario_id not in message
    assert private_event.shard_sha256 not in message
    assert str(private_event.record_ordinal) not in message
    assert Counter(
        code in message for code in SOURCE_REJECTION_CODES
    )[True] == 0
