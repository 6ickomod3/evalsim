"""Data-free acceptance tests for the immutable M5 local result store."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pyarrow as pa
import pytest

import evalsim.results.m5 as result_store_module
from evalsim.results import (
    ExpectedRowCounts,
    M5_M4_INTEGRITY_ASSUMPTION,
    M5_M4_REUSE_SCHEMA_VERSION,
    M5_PARITY_ORDER_VERSION,
    M5_PARITY_RANK_DOMAIN,
    M5_RESULT_SCHEMAS,
    M5DeterminismReceipt,
    M5ParityOrderReceipt,
    M5RunProvenance,
    M5ResultStore,
    M5ResultStoreError,
    M5ResultStoreIntegrityError,
    M5ResultStoreStateError,
    METRIC_RESULTS,
    OFFICIAL_WAYMAX_REFERENCE_PARAMETERS,
    SCORECARDS,
    SLICE_MEMBERSHIP,
    PreparedM5Finalization,
    WAYMAX_PARITY_SUMMARY,
    executable_source_fingerprint,
    official_executable_source_paths,
    scorecard_row_from_result,
    verify_committed_m5_result_store,
    verify_m5_result_store,
    verify_prepared_m5_result_store,
)
from evalsim.stats.m5 import (
    PairedCellSpec,
    PolicyContrast,
    ScenarioScalar,
    analyze_paired_cell,
    make_resampling_key,
)


_TEST_COUNTS = ExpectedRowCounts(
    metric_results=4,
    slice_membership=2,
    scorecards=1,
    waymax_parity_summary=1,
)


def _project(tmp_path: Path, name: str = "project") -> Path:
    path = tmp_path / name
    path.mkdir()
    (path / "AGENTS.md").write_text("test instructions\n", encoding="utf-8")
    (path / "evalsim" / "results").mkdir(parents=True)
    (path / "evalsim" / "results" / "m5.py").write_text(
        "TEST_SOURCE = True\n",
        encoding="utf-8",
    )
    (path / "tests").mkdir()
    (path / "tests" / "test_m5_result_store.py").write_text(
        "def test_sentinel(): pass\n",
        encoding="utf-8",
    )
    return path


def _provenance(project: Path | None = None) -> M5RunProvenance:
    source_paths = (
        "AGENTS.md",
        "evalsim/results/m5.py",
        "tests/test_m5_result_store.py",
    )
    return M5RunProvenance(
        m4_manifest_sha256="a" * 64,
        m4_execution_provenance_sha256="b" * 64,
        selected_order_version="m4-selected-order-1",
        selected_order_fingerprint_sha256="e" * 64,
        executable_source_fingerprint_sha256=(
            executable_source_fingerprint(project, source_paths)
            if project is not None
            else "f" * 64
        ),
        executable_source_paths=source_paths,
        git_commit="c" * 40,
        git_tree="d" * 40,
        simulator_specs={
            "constant_velocity": {
                "deterministic": True,
                "execution_role": "policy",
                "parameters": {},
                "version": "0.1.0",
            },
            "idm": {
                "deterministic": True,
                "execution_role": "policy",
                "parameters": {},
                "version": "0.1.0",
            },
            "log_replay": {
                "deterministic": True,
                "execution_role": "policy",
                "parameters": {},
                "version": "0.1.0",
            },
            "waymax_exact_log_state_dynamics": {
                "deterministic": True,
                "execution_role": "reference",
                "parameters": {},
                "version": "0.1.0",
            },
        },
        runtime_versions={
            "flax": "test",
            "jax": "test",
            "jaxlib": "test",
            "numpy": "test",
            "pyarrow": pa.__version__,
            "python": "3.11",
            "tensorflow": "test",
            "waymo_waymax": "a64dfec9",
        },
    )


def _parity_receipt() -> M5ParityOrderReceipt:
    return M5ParityOrderReceipt(
        rank_domain=M5_PARITY_RANK_DOMAIN,
        order_version=M5_PARITY_ORDER_VERSION,
        ordered_membership_sha256="9" * 64,
    )


def _determinism_receipt(
    *,
    metric_sha256: str = "7" * 64,
    statistics_sha256: str = "8" * 64,
) -> M5DeterminismReceipt:
    return M5DeterminismReceipt(
        metric_pass_1_sha256=metric_sha256,
        metric_pass_2_sha256=metric_sha256,
        statistics_pass_1_sha256=statistics_sha256,
        statistics_pass_2_sha256=statistics_sha256,
    )


def _extended_provenance(project: Path | None = None) -> M5RunProvenance:
    payload = _provenance(project).to_dict()
    payload.update(
        {
            "m4_aggregate_summary_sha256": "1" * 64,
            "m4_integrity_assumption": M5_M4_INTEGRITY_ASSUMPTION,
            "m4_receipt_verified": False,
            "m4_reuse_schema_version": M5_M4_REUSE_SCHEMA_VERSION,
            "parity_order_fingerprint_sha256": "9" * 64,
            "parity_order_version": M5_PARITY_ORDER_VERSION,
        }
    )
    payload["simulator_specs"]["waymax_exact_log_state_dynamics"].update(
        {
            "parameters": dict(OFFICIAL_WAYMAX_REFERENCE_PARAMETERS),
            "version": result_store_module.WAYMAX_REFERENCE_VERSION,
        }
    )
    return M5RunProvenance(**payload)


def _official_source_project(tmp_path: Path) -> tuple[Path, tuple[str, ...]]:
    project = tmp_path / "official-source-project"
    project.mkdir()
    expected: list[str] = []
    for root_name in result_store_module._OFFICIAL_EXECUTABLE_ROOTS:
        sentinel = project / root_name / "sentinel.py"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("SENTINEL = True\n", encoding="utf-8")
        expected.append(sentinel.relative_to(project).as_posix())
    for relative_name in result_store_module._OFFICIAL_EXECUTABLE_FILES:
        source_file = project / relative_name
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text("official source sentinel\n", encoding="utf-8")
        expected.append(relative_name)
    (project / ".gitignore").write_text(
        "evalsim/results/ignored.py\n",
        encoding="utf-8",
    )
    subprocess.run(
        ("git", "init", "-b", "main"),
        cwd=project,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "config", "user.email", "evalsim-test@example.invalid"),
        cwd=project,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "EvalSim Test"),
        cwd=project,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "add", "."),
        cwd=project,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "commit", "-m", "test source snapshot"),
        cwd=project,
        check=True,
        capture_output=True,
    )
    return project, tuple(sorted(expected))


def _metric_row(
    cohort_index: int,
    *,
    execution_name: str = "log_replay",
    seed: int = 7,
) -> dict[str, object]:
    return {
        "cohort_index": cohort_index,
        "details_json": '{"oracle":"data_free"}',
        "distribution": [float(cohort_index)],
        "eligible_components": 1,
        "execution_name": execution_name,
        "execution_role": "policy",
        "invalid_reason": None,
        "metric_name": "position_error_m",
        "metric_version": "1.0.0",
        "seed": seed,
        "total_components": 1,
        "valid": True,
        "value": float(cohort_index),
    }


def _slice_row(cohort_index: int) -> dict[str, object]:
    return {
        "cohort_index": cohort_index,
        "eligible": True,
        "member": True,
        "reason": None,
        "slice_name": "all",
        "slice_version": "m5-womd-slices-1.0.0",
    }


def _scorecard_row() -> dict[str, object]:
    spec = PairedCellSpec(
        metric_name="position_error_m",
        contrast=PolicyContrast("constant_velocity", "log_replay"),
        slice_name="all",
    )
    key = make_resampling_key(spec, paired_n=2)
    return {
        "adjusted_level": None,
        "adjusted_lower": None,
        "adjusted_upper": None,
        "asymmetric_component_n": 0,
        "asymmetric_missing_n": 0,
        "asymmetric_reason_n": 0,
        "base_seed": 20260728,
        "both_missing_n": 0,
        "cohort_n": 2,
        "direction": "lower",
        "directional_language_allowed": False,
        "eligible_components_a": 2,
        "eligible_components_b": 2,
        "excluded_n": 0,
        "favorable_proportion": None,
        "index_dtype": "int64",
        "metric_name": "position_error_m",
        "value_unit": "m",
        "metric_version": "1.0.0",
        "missing_reasons_a_json": "{}",
        "missing_reasons_b_json": "{}",
        "nonzero_effect_n": None,
        "oriented_mean_advantage": None,
        "paired_n": 2,
        "pointwise_level": None,
        "pointwise_lower": None,
        "pointwise_upper": None,
        "policy_a": "constant_velocity",
        "policy_a_mean": None,
        "policy_a_median": None,
        "policy_b": "log_replay",
        "policy_b_mean": None,
        "policy_b_median": None,
        "quantile_method": "linear",
        "raw_mean_difference": None,
        "raw_median_difference": None,
        "resamples": key.resamples,
        "resampling_digest_words": list(key.digest_words),
        "resampling_key_json": key.canonical_json,
        "resampling_sha256": key.sha256,
        "rng": "PCG64",
        "slice_name": "all",
        "slice_version": "m5-womd-slices-1.0.0",
        "source_pairing_complete": True,
        "standardized_signal_to_heterogeneity": None,
        "status": "insufficient_n",
        "total_components_a": 2,
        "total_components_b": 2,
        "valid_a_n": 2,
        "valid_b_n": 2,
    }


def _parity_row() -> dict[str, object]:
    return {
        "compared_components": 20,
        "exact_match": True,
        "max_abs_error": 0.0,
        "max_tolerance_excess": -1e-6,
        "metric_name": "log_divergence",
        "metric_version": "1.0.0",
        "mismatch_count": 0,
        "parity_index": 0,
        "policy_name": "log_replay",
        "status": "accepted",
    }


def _paired_metric_rows() -> list[dict[str, object]]:
    return [
        _metric_row(1, execution_name="log_replay"),
        _metric_row(0, execution_name="constant_velocity"),
        _metric_row(1, execution_name="constant_velocity"),
        _metric_row(0, execution_name="log_replay"),
    ]


def _write_all(store: M5ResultStore) -> None:
    # Deliberately reverse the input rows; persisted order must be canonical.
    store.write_metric_results_part(_paired_metric_rows())
    store.write_slice_membership([_slice_row(1), _slice_row(0)])
    store.write_scorecards([_scorecard_row()])
    store.write_waymax_parity_summary([_parity_row()])
    store.write_human_readable_scorecard()


def test_roundtrip_fixed_schemas_hashes_and_final_manifest(tmp_path: Path) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "roundtrip",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    _write_all(store)

    metric_table = store.read_dataset(METRIC_RESULTS)
    assert metric_table.schema.equals(
        M5_RESULT_SCHEMAS[METRIC_RESULTS],
        check_metadata=True,
    )
    assert metric_table.column("cohort_index").to_pylist() == [0, 0, 1, 1]
    assert store.read_dataset(SLICE_MEMBERSHIP).num_rows == 2
    assert store.read_dataset(SCORECARDS).num_rows == 1
    assert store.read_dataset(WAYMAX_PARITY_SUMMARY).num_rows == 1
    report_record = store.scorecard_report
    assert report_record is not None
    report_path = store.run_path / report_record.path
    pending_report_path = store.run_path / "pending" / report_record.path
    assert report_path.stat().st_ino == pending_report_path.stat().st_ino
    assert report_path.stat().st_dev == pending_report_path.stat().st_dev
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == (
        report_record.sha256
    )
    assert report_path.stat().st_size == report_record.size_bytes
    report_bytes = report_path.read_bytes()
    assert b"roundtrip" not in report_bytes
    assert b"scenario_id" not in report_bytes
    assert b"resampling_sha256" not in report_bytes

    for record in store.artifacts:
        canonical = store.run_path / record.path
        pending = store.run_path / "pending" / record.path
        assert canonical.stat().st_ino == pending.stat().st_ino
        assert canonical.stat().st_dev == pending.stat().st_dev
        assert hashlib.sha256(canonical.read_bytes()).hexdigest() == record.sha256
        assert canonical.stat().st_size == record.size_bytes

    store.finalize(provenance=_provenance(project))
    with pytest.raises(M5ResultStoreIntegrityError, match="production"):
        verify_m5_result_store(project, "roundtrip")
    verified = verify_m5_result_store(
        project,
        "roundtrip",
        allow_data_free=True,
    )
    assert verified.run_path == project / "outputs" / "m5" / "roundtrip"
    assert verified.manifest["complete"] is True
    assert verified.manifest["result_path"] == "outputs/m5/roundtrip"
    assert verified.manifest["actual_rows"] == _TEST_COUNTS.to_dict()
    assert verified.manifest["hash_policy"] == {
        "algorithm": "sha256",
        "manifest_self_hash": False,
    }
    assert verified.scorecard_report == report_record
    assert verified.manifest["scorecard_report"] == report_record.to_dict()
    artifact_paths = {record.path for record in verified.artifacts}
    assert "evaluation-manifest.json" not in artifact_paths
    assert "SUCCESS" not in artifact_paths
    assert (verified.run_path / "SUCCESS").read_bytes() == b"SUCCESS\n"

    restored_manifest = json.loads(
        (verified.run_path / "evaluation-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert restored_manifest == dict(verified.manifest)
    with pytest.raises(M5ResultStoreStateError):
        store.write_metric_results_part([_metric_row(2)], part_index=1)


def test_duplicate_keys_poison_run_and_preserve_prior_part(tmp_path: Path) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "duplicate-key",
        expected_rows=ExpectedRowCounts(2, 0, 0, 0),
        data_free=True,
    )
    first = store.write_metric_results_part([_metric_row(0)])
    first_path = store.run_path / first.path
    original = first_path.read_bytes()

    with pytest.raises(M5ResultStoreIntegrityError, match="duplicate"):
        store.write_metric_results_part([_metric_row(0)], part_index=1)

    assert first_path.read_bytes() == original
    assert (store.run_path / "FAILURE.json").is_file()
    assert not (store.run_path / "SUCCESS").exists()
    with pytest.raises(M5ResultStoreStateError):
        store.finalize(provenance=_provenance(project))


def test_hash_tamper_is_detected_after_success(tmp_path: Path) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "tampered",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    _write_all(store)
    store.finalize(provenance=_provenance(project))

    metric_path = store.run_path / "metric-results" / "part-00000.parquet"
    with metric_path.open("r+b") as handle:
        original = handle.read(1)
        handle.seek(0)
        handle.write(bytes([original[0] ^ 0xFF]))
        handle.flush()
        os.fsync(handle.fileno())

    with pytest.raises(M5ResultStoreIntegrityError, match="SHA-256"):
        verify_m5_result_store(project, "tampered", allow_data_free=True)


def test_scorecard_report_is_required_and_tamper_evident(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    missing = M5ResultStore.create(
        project,
        "missing-report",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    missing.write_metric_results_part(_paired_metric_rows())
    missing.write_slice_membership([_slice_row(0), _slice_row(1)])
    missing.write_scorecards([_scorecard_row()])
    missing.write_waymax_parity_summary([_parity_row()])
    with pytest.raises(M5ResultStoreIntegrityError, match="report is missing"):
        missing.finalize(provenance=_provenance(project))

    tampered = M5ResultStore.create(
        project,
        "tampered-report",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    _write_all(tampered)
    tampered.finalize(provenance=_provenance(project))
    report_path = tampered.run_path / "scorecard.md"
    with report_path.open("r+b") as handle:
        original = handle.read(1)
        handle.seek(0)
        handle.write(bytes([original[0] ^ 0x01]))
        handle.flush()
        os.fsync(handle.fileno())
    with pytest.raises(M5ResultStoreIntegrityError, match="SHA-256"):
        verify_m5_result_store(
            project,
            "tampered-report",
            allow_data_free=True,
        )


def test_scorecard_report_cannot_be_written_before_its_source_table(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "early-report",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    with pytest.raises(M5ResultStoreIntegrityError, match="scorecard table"):
        store.write_human_readable_scorecard()
    assert (store.run_path / "FAILURE.json").is_file()


def test_scorecard_report_cannot_be_forged_with_a_rehashed_manifest(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "rehashed-report",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    _write_all(store)
    store.finalize(provenance=_provenance(project))

    report_path = store.run_path / "scorecard.md"
    forged = report_path.read_bytes() + b"\nForged conclusion.\n"
    report_path.write_bytes(forged)
    manifest_path = store.run_path / "evaluation-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scorecard_report"]["sha256"] = hashlib.sha256(forged).hexdigest()
    manifest["scorecard_report"]["size_bytes"] = len(forged)
    manifest_path.write_bytes(
        (
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )

    with pytest.raises(
        M5ResultStoreIntegrityError,
        match="deterministic rendering",
    ):
        verify_m5_result_store(
            project,
            "rehashed-report",
            allow_data_free=True,
        )


def test_data_free_verification_detects_bound_source_changes(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "source-changed",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    _write_all(store)
    store.finalize(provenance=_provenance(project))

    (project / "AGENTS.md").write_text(
        "changed after finalization\n",
        encoding="utf-8",
    )
    with pytest.raises(
        M5ResultStoreIntegrityError,
        match="source files changed",
    ):
        verify_m5_result_store(
            project,
            "source-changed",
            allow_data_free=True,
        )


def test_interrupted_finalization_cannot_be_resumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "interrupted",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    _write_all(store)
    real_write = result_store_module._write_bytes_exclusive

    def interrupt_success(path: Path, payload: bytes) -> None:
        if path.name == "SUCCESS":
            raise OSError("injected interruption")
        real_write(path, payload)

    monkeypatch.setattr(
        result_store_module,
        "_write_bytes_exclusive",
        interrupt_success,
    )
    with pytest.raises(OSError, match="injected interruption"):
        store.finalize(provenance=_provenance(project))

    assert (store.run_path / "FINALIZING").is_file()
    assert (store.run_path / "evaluation-manifest.json").is_file()
    assert (store.run_path / "FAILURE.json").is_file()
    assert not (store.run_path / "SUCCESS").exists()
    with pytest.raises(M5ResultStoreStateError):
        store.finalize(provenance=_provenance(project))
    with pytest.raises(M5ResultStoreIntegrityError):
        verify_m5_result_store(
            project,
            "interrupted",
            allow_data_free=True,
        )


def test_post_create_success_error_reconciles_as_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "post-create-success-error",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    _write_all(store)
    real_write = result_store_module._write_bytes_exclusive

    def write_then_raise(path: Path, payload: bytes) -> None:
        real_write(path, payload)
        if path.name == "SUCCESS":
            raise OSError("injected post-create directory fsync error")

    monkeypatch.setattr(
        result_store_module,
        "_write_bytes_exclusive",
        write_then_raise,
    )
    assert store.finalize(provenance=_provenance(project)) == store.run_path

    assert (store.run_path / "SUCCESS").read_bytes() == b"SUCCESS\n"
    assert not (store.run_path / "FAILURE.json").exists()
    verified = verify_m5_result_store(
        project,
        "post-create-success-error",
        allow_data_free=True,
    )
    assert verified.manifest["complete"] is True


def test_prepared_finalization_requires_exact_capability(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "prepared-capability",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    _write_all(store)

    prepared = store.prepare_finalization(
        provenance=_provenance(project)
    )
    assert isinstance(prepared, PreparedM5Finalization)
    assert (store.run_path / "FINALIZING").read_bytes() == b"FINALIZING\n"
    assert (store.run_path / "evaluation-manifest.json").is_file()
    assert not (store.run_path / "SUCCESS").exists()
    prepared_verified = verify_prepared_m5_result_store(
        project,
        "prepared-capability",
        allow_data_free=True,
    )
    assert prepared_verified.manifest["complete"] is True

    forged = PreparedM5Finalization(
        run_path=prepared.run_path,
        _nonce=object(),
    )
    with pytest.raises(M5ResultStoreStateError, match="active prepared"):
        store.commit_finalization(forged)
    assert not (store.run_path / "SUCCESS").exists()

    assert store.commit_finalization(prepared) == store.run_path
    with pytest.raises(
        M5ResultStoreIntegrityError,
        match="cannot already be committed",
    ):
        verify_prepared_m5_result_store(
            project,
            "prepared-capability",
            allow_data_free=True,
        )
    verified = verify_m5_result_store(
        project,
        "prepared-capability",
        allow_data_free=True,
    )
    assert verified.manifest["complete"] is True


def test_prepared_finalization_can_abort_but_never_commit(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "prepared-abort",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    _write_all(store)
    prepared = store.prepare_finalization(
        provenance=_provenance(project)
    )

    failure_path = store.abort_finalization(
        prepared,
        "terminal_output_detected",
    )
    record = json.loads(failure_path.read_text(encoding="utf-8"))
    assert record["reason_code"] == "terminal_output_detected"
    assert not (store.run_path / "SUCCESS").exists()
    with pytest.raises(M5ResultStoreStateError, match="active prepared"):
        store.commit_finalization(prepared)


def test_official_committed_checkpoint_is_abortable_not_success(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "abortable-committed",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    _write_all(store)
    prepared = store.prepare_finalization(
        provenance=_provenance(project)
    )

    # Exercise the state machine without constructing WOMD-sized official
    # artifacts.  The manifest remains data-free, so the independent official
    # committed verifier must reject it even though the checkpoint itself is
    # exact and abortable within this writer process.
    store.row_accounting_profile = result_store_module.OFFICIAL_M5_PROFILE
    assert (
        store.mark_committed_for_verification(prepared)
        == store.run_path
    )
    assert (store.run_path / "COMMITTED").read_bytes() == b"COMMITTED\n"
    assert not (store.run_path / "SUCCESS").exists()
    with pytest.raises(
        M5ResultStoreIntegrityError,
        match="only official M5 runs",
    ):
        verify_committed_m5_result_store(
            project,
            "abortable-committed",
            allow_data_free=True,
        )

    failure = store.abort_finalization(
        prepared,
        "verification_failed",
    )
    assert json.loads(failure.read_text(encoding="utf-8"))[
        "reason_code"
    ] == "verification_failed"
    assert (store.run_path / "COMMITTED").read_bytes() == b"COMMITTED\n"
    assert not (store.run_path / "SUCCESS").exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "finalizing",
        "manifest",
        "artifact",
        "pending_link",
        "scorecard_report",
        "unexpected_member",
    ),
)
def test_terminal_byte_seal_rejects_post_prepare_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        f"post-prepare-{mutation}",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    _write_all(store)
    prepared = store.prepare_finalization(
        provenance=_provenance(project)
    )

    if mutation == "finalizing":
        (store.run_path / "FINALIZING").write_bytes(b"changed\n")
    elif mutation == "manifest":
        manifest = store.run_path / "evaluation-manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b" ")
    elif mutation == "artifact":
        artifact = store.run_path / store.artifacts[0].path
        artifact.write_bytes(artifact.read_bytes() + b"x")
    elif mutation == "pending_link":
        record = store.artifacts[0]
        canonical = store.run_path / record.path
        pending = store.run_path / "pending" / record.path
        payload = pending.read_bytes()
        pending.unlink()
        pending.write_bytes(payload)
        assert canonical.stat().st_ino != pending.stat().st_ino
    elif mutation == "scorecard_report":
        report = store.run_path / "scorecard.md"
        report.write_bytes(report.read_bytes() + b"x")
    else:
        (store.run_path / "unregistered.txt").write_text(
            "unexpected\n",
            encoding="utf-8",
        )

    with pytest.raises(M5ResultStoreIntegrityError):
        store.commit_finalization(prepared)
    assert (store.run_path / "FAILURE.json").is_file()
    assert not (store.run_path / "SUCCESS").exists()


def test_run_and_artifact_creation_are_exclusive(tmp_path: Path) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "exclusive",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    with pytest.raises(FileExistsError):
        M5ResultStore.create(
            project,
            "exclusive",
            expected_rows=_TEST_COUNTS,
            data_free=True,
        )

    store.write_metric_results_part([_metric_row(0)])
    with pytest.raises(FileExistsError):
        store.write_metric_results_part([_metric_row(1)])
    assert (store.run_path / "FAILURE.json").is_file()


@pytest.mark.parametrize("failure_point", ("run_fsync", "child_fsync"))
def test_fresh_store_creation_failure_leaves_no_ownerless_partial_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    project = _project(tmp_path)
    run_name = f"creation-{failure_point}"
    target = project / "outputs/m5" / run_name
    real_fsync = result_store_module._fsync_directory
    injected = False

    def failing_fsync(path: Path) -> None:
        nonlocal injected
        should_fail = (
            failure_point == "run_fsync"
            and path == project / "outputs/m5"
            and target.exists()
        ) or (
            failure_point == "child_fsync"
            and path == target
            and (target / "pending").exists()
        )
        if should_fail and not injected:
            injected = True
            raise OSError("injected creation fsync failure")
        real_fsync(path)

    monkeypatch.setattr(
        result_store_module,
        "_fsync_directory",
        failing_fsync,
    )
    with pytest.raises(M5ResultStoreError):
        M5ResultStore.create(
            project,
            run_name,
            expected_rows=_TEST_COUNTS,
            data_free=True,
        )
    assert injected
    assert not target.exists()

    monkeypatch.setattr(
        result_store_module,
        "_fsync_directory",
        real_fsync,
    )
    retry = M5ResultStore.create(
        project,
        run_name,
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    assert retry.run_path == target


def test_fresh_store_rollback_marks_unexpected_retained_member_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path)
    run_name = "creation-retained"
    target = project / "outputs/m5" / run_name
    real_child = result_store_module._create_child_directory

    def fail_after_unexpected_member(parent: Path, name: str) -> Path:
        if parent == target and name == "pending":
            (target / "unexpected.txt").write_text(
                "concurrent member\n",
                encoding="utf-8",
            )
            raise M5ResultStoreError("injected child creation failure")
        return real_child(parent, name)

    monkeypatch.setattr(
        result_store_module,
        "_create_child_directory",
        fail_after_unexpected_member,
    )
    with pytest.raises(M5ResultStoreError):
        M5ResultStore.create(
            project,
            run_name,
            expected_rows=_TEST_COUNTS,
            data_free=True,
        )
    assert target.is_dir()
    failure = json.loads(
        (target / "FAILURE.json").read_text(encoding="utf-8")
    )
    assert failure == {
        "complete": False,
        "reason_code": "result_store_failed",
        "schema_version": result_store_module.M5_RESULT_STORE_SCHEMA_VERSION,
    }


def test_scorecards_wait_for_exact_metric_and_slice_accounting(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "early-scorecard",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    store.write_metric_results_part([_metric_row(0)])
    store.write_slice_membership([_slice_row(0), _slice_row(1)])

    with pytest.raises(M5ResultStoreIntegrityError, match="exact metric"):
        store.write_scorecards([_scorecard_row()])
    assert (store.run_path / "FAILURE.json").is_file()


def test_finalization_requires_exact_row_accounting(tmp_path: Path) -> None:
    project = _project(tmp_path)
    expected = ExpectedRowCounts(4, 2, 1, 2)
    store = M5ResultStore.create(
        project,
        "row-mismatch",
        expected_rows=expected,
        data_free=True,
    )
    _write_all(store)

    with pytest.raises(M5ResultStoreIntegrityError, match="row accounting"):
        store.finalize(provenance=_provenance(project))
    assert (store.run_path / "FAILURE.json").is_file()
    assert not (store.run_path / "evaluation-manifest.json").exists()
    assert not (store.run_path / "SUCCESS").exists()


@pytest.mark.parametrize(
    "run_name",
    (
        "",
        ".",
        "..",
        "../escape",
        "nested/run",
        "/absolute",
        "Uppercase",
        "has space",
    ),
)
def test_run_name_rejects_traversal_and_noncanonical_paths(
    tmp_path: Path,
    run_name: str,
) -> None:
    with pytest.raises(ValueError):
        M5ResultStore.create(_project(tmp_path), run_name)


def test_containment_rejects_symlinked_output_roots(tmp_path: Path) -> None:
    project = _project(tmp_path)
    outside = _project(tmp_path, "outside")
    (project / "outputs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(M5ResultStoreError):
        M5ResultStore.create(project, "escape")
    assert not (outside / "m5" / "escape").exists()


def test_containment_rejects_symlinked_project_root(tmp_path: Path) -> None:
    project = _project(tmp_path)
    alias = tmp_path / "project-alias"
    alias.symlink_to(project, target_is_directory=True)

    with pytest.raises(M5ResultStoreError):
        M5ResultStore.create(alias, "escape")


def test_unregistered_payload_blocks_success_without_deletion(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "unexpected-payload",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    _write_all(store)
    unexpected = store.run_path / "copied-payload.tfrecord"
    unexpected.write_bytes(b"not dataset data, only a sentinel")

    with pytest.raises(
        M5ResultStoreIntegrityError,
        match="unregistered",
    ):
        store.finalize(provenance=_provenance(project))
    assert unexpected.is_file()
    assert (store.run_path / "FAILURE.json").is_file()


def test_fixed_schema_and_row_implications_fail_closed(tmp_path: Path) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "bad-row",
        expected_rows=ExpectedRowCounts(1, 0, 0, 0),
        data_free=True,
    )
    bad = _metric_row(0)
    bad["unexpected"] = "field"

    with pytest.raises(ValueError, match="fixed schema"):
        store.write_metric_results_part([bad])
    assert (store.run_path / "FAILURE.json").is_file()


def test_custom_row_counts_require_explicit_data_free_profile(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    with pytest.raises(ValueError, match="data-free-test-only"):
        M5ResultStore.create(
            project,
            "unsafe-override",
            expected_rows=_TEST_COUNTS,
        )
    assert not (project / "outputs").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("value_unit", "feet", "unit/direction"),
        ("direction", "higher", "unit/direction"),
        (
            "directional_language_allowed",
            True,
            "directional_language_allowed",
        ),
        ("policy_b", "idm", "canonical M5"),
    ),
)
def test_scorecard_semantic_drift_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        f"bad-scorecard-{field.replace('_', '-')}",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    store.write_metric_results_part(_paired_metric_rows())
    store.write_slice_membership([_slice_row(0), _slice_row(1)])
    row = _scorecard_row()
    row[field] = value

    with pytest.raises(ValueError, match=message):
        store.write_scorecards([row])
    assert (store.run_path / "FAILURE.json").is_file()


def test_scorecard_rejects_self_consistent_but_noncanonical_resampling_key(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "bad-resampling-key",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    store.write_metric_results_part(_paired_metric_rows())
    store.write_slice_membership([_slice_row(0), _slice_row(1)])
    row = _scorecard_row()
    fake_key = '{"paired_n":2,"resamples":1000,"schema":"invented"}'
    fake_digest = hashlib.sha256(fake_key.encode("utf-8")).digest()
    row.update(
        {
            "resamples": 1000,
            "resampling_digest_words": [
                int.from_bytes(fake_digest[offset : offset + 4], "big")
                for offset in range(0, 32, 4)
            ],
            "resampling_key_json": fake_key,
            "resampling_sha256": fake_digest.hex(),
        }
    )

    with pytest.raises(ValueError, match="frozen M5 substream"):
        store.write_scorecards([row])


def test_metric_slice_and_parity_catalog_drift_fail_closed(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)

    metric_store = M5ResultStore.create(
        project,
        "bad-metric-catalog",
        expected_rows=ExpectedRowCounts(1, 0, 0, 0),
        data_free=True,
    )
    metric = _metric_row(0)
    metric["metric_version"] = "2.0.0"
    with pytest.raises(ValueError, match="canonical M5 catalog"):
        metric_store.write_metric_results_part([metric])

    slice_store = M5ResultStore.create(
        project,
        "bad-slice-catalog",
        expected_rows=ExpectedRowCounts(0, 1, 0, 0),
        data_free=True,
    )
    slice_row = _slice_row(0)
    slice_row["slice_name"] = "invented_slice"
    with pytest.raises(ValueError, match="canonical M5 catalog"):
        slice_store.write_slice_membership([slice_row])

    parity_store = M5ResultStore.create(
        project,
        "bad-parity-acceptance",
        expected_rows=ExpectedRowCounts(0, 0, 0, 1),
        data_free=True,
    )
    parity = _parity_row()
    parity.update(
        {
            "exact_match": False,
            "max_abs_error": 1e-4,
            "max_tolerance_excess": 1e-4,
            "mismatch_count": 1,
            "status": "accepted",
        }
    )
    with pytest.raises(ValueError, match="accepted parity"):
        parity_store.write_waymax_parity_summary([parity])


def test_metric_values_must_match_the_registered_reducer() -> None:
    forged_mean = _metric_row(0)
    forged_mean.update(
        {
            "distribution": [1.0, 1.0],
            "eligible_components": 2,
            "total_components": 2,
            "value": 999.0,
        }
    )
    with pytest.raises(ValueError, match="registered distribution reducer"):
        result_store_module._metric_row(forged_mean)

    forged_minimum = _metric_row(0)
    forged_minimum.update(
        {
            "distribution": [3.0, 1.0],
            "eligible_components": 2,
            "metric_name": "minimum_center_distance_m",
            "total_components": 2,
            "value": 3.0,
        }
    )
    with pytest.raises(ValueError, match="registered distribution reducer"):
        result_store_module._metric_row(forged_minimum)


def test_exact_discrete_parity_cannot_carry_nonzero_error() -> None:
    row = _parity_row()
    row.update(
        {
            "max_abs_error": 0.25,
            "max_tolerance_excess": 0.25,
            "metric_name": "overlap",
        }
    )
    with pytest.raises(ValueError, match="discrete parity"):
        result_store_module._parity_row(row)


def test_log_divergence_parity_is_bound_to_tolerance_excess() -> None:
    out_of_range = _parity_row()
    out_of_range.update(
        {
            "max_abs_error": 1e100,
            "max_tolerance_excess": -1.0,
        }
    )
    with pytest.raises(ValueError, match="finite float32 range"):
        result_store_module._parity_row(out_of_range)

    exceeded = _parity_row()
    exceeded.update(
        {
            "max_abs_error": 2e-6,
            "max_tolerance_excess": 1e-6,
        }
    )
    with pytest.raises(ValueError, match="tolerance excess"):
        result_store_module._parity_row(exceeded)

    missing_floor = _parity_row()
    missing_floor["max_tolerance_excess"] = 0.0
    with pytest.raises(ValueError, match="absolute floor"):
        result_store_module._parity_row(missing_floor)

    impossible_excess = _parity_row()
    impossible_excess["max_tolerance_excess"] = -1e300
    with pytest.raises(ValueError, match="exceeds the float32 range"):
        result_store_module._parity_row(impossible_excess)


def test_waymax_reference_equality_and_zero_oracle_are_exact() -> None:
    log_row = _metric_row(0, execution_name="log_replay", seed=0)
    log_row.update({"distribution": [0.0], "value": 0.0})
    reference_row = dict(log_row)
    reference_row.update(
        {
            "execution_name": "waymax_exact_log_state_dynamics",
            "execution_role": "reference",
        }
    )
    normalized_log = result_store_module._metric_row(log_row)
    normalized_reference = result_store_module._metric_row(reference_row)
    assert result_store_module._reference_equivalence_digest(
        normalized_reference
    ) == result_store_module._reference_equivalence_digest(normalized_log)
    assert (
        result_store_module._validate_exact_log_zero_oracle(
            normalized_reference
        )
        == 1
    )

    forged_reference = dict(normalized_reference)
    forged_reference.update({"distribution": [1.0], "value": 1.0})
    assert result_store_module._reference_equivalence_digest(
        forged_reference
    ) != result_store_module._reference_equivalence_digest(normalized_log)
    with pytest.raises(M5ResultStoreIntegrityError, match="zero-error"):
        result_store_module._validate_exact_log_zero_oracle(
            forged_reference
        )


def test_finalize_requires_typed_complete_provenance(tmp_path: Path) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "typed-provenance",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    _write_all(store)

    with pytest.raises(TypeError, match="M5RunProvenance"):
        store.finalize(provenance={})  # type: ignore[arg-type]
    assert not (store.run_path / "FINALIZING").exists()
    store.finalize(provenance=_provenance(project))
    assert (
        verify_m5_result_store(
            project,
            "typed-provenance",
            allow_data_free=True,
        ).manifest["provenance"]["m4_manifest_sha256"]
        == "a" * 64
    )


def test_provenance_is_recursively_frozen_and_to_dict_is_detached() -> None:
    provenance = _provenance()
    with pytest.raises(TypeError):
        provenance.simulator_specs["idm"]["version"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        provenance.runtime_versions["numpy"] = "changed"  # type: ignore[index]

    detached = provenance.to_dict()
    detached["simulator_specs"]["idm"]["version"] = "changed"
    assert provenance.simulator_specs["idm"]["version"] == "0.1.0"


def test_provenance_rejects_nondeterministic_or_incomplete_runtime() -> None:
    payload = _provenance().to_dict()
    payload["simulator_specs"]["idm"]["deterministic"] = False
    with pytest.raises(ValueError, match="deterministic=true"):
        M5RunProvenance(**payload)

    payload = _provenance().to_dict()
    del payload["runtime_versions"]["jax"]
    with pytest.raises(ValueError, match="dependency version"):
        M5RunProvenance(**payload)


def test_extended_provenance_binds_m4_reuse_and_parity_order() -> None:
    legacy = _provenance()
    assert legacy.has_official_extensions is False
    assert "m4_aggregate_summary_sha256" not in legacy.to_dict()
    assert M5RunProvenance.from_dict(legacy.to_dict()) == legacy

    extended = _extended_provenance()
    assert extended.has_official_extensions is True
    assert extended.m4_receipt_verified is False
    assert (
        extended.parity_order_fingerprint_sha256
        == _parity_receipt().ordered_membership_sha256
    )
    assert M5RunProvenance.from_dict(extended.to_dict()) == extended
    result_store_module._validate_official_provenance_contract(extended)

    partial = legacy.to_dict()
    partial["m4_aggregate_summary_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="all present or absent"):
        M5RunProvenance(**partial)

    wrong_order = extended.to_dict()
    wrong_order["selected_order_version"] = "invented-order"
    with pytest.raises(
        M5ResultStoreIntegrityError,
        match="wrong accepted-M4 order",
    ):
        result_store_module._validate_official_provenance_contract(
            M5RunProvenance(**wrong_order)
        )


def test_official_waymax_spec_must_be_exact_and_executed() -> None:
    payload = _extended_provenance().to_dict()
    payload["simulator_specs"][
        "waymax_exact_log_state_dynamics"
    ]["parameters"]["executed"] = False
    provenance = M5RunProvenance(**payload)
    with pytest.raises(
        M5ResultStoreIntegrityError,
        match="not frozen or executed",
    ):
        result_store_module._validate_official_provenance_contract(
            provenance
        )


def test_typed_receipts_reject_digest_drift_and_expose_no_digest_repr() -> None:
    parity = _parity_receipt()
    assert set(parity.to_dict()) == {
        "member_count",
        "order_version",
        "ordered_membership_sha256",
        "rank_domain",
        "transition_count",
    }
    assert parity.ordered_membership_sha256 not in repr(parity)
    assert M5ParityOrderReceipt.from_dict(parity.to_dict()) == parity

    with pytest.raises(ValueError, match="metric pass digests"):
        M5DeterminismReceipt(
            metric_pass_1_sha256="1" * 64,
            metric_pass_2_sha256="2" * 64,
            statistics_pass_1_sha256="3" * 64,
            statistics_pass_2_sha256="3" * 64,
        )
    receipt = _determinism_receipt()
    assert receipt.metric_pass_1_sha256 not in repr(receipt)
    assert M5DeterminismReceipt.from_dict(receipt.to_dict()) == receipt


def test_official_parity_receipt_is_immutable_and_pre_metric(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(project, "official-receipt")
    record = store.write_parity_order_receipt(_parity_receipt())
    canonical = store.run_path / record.path
    pending = store.run_path / "pending" / record.path
    assert canonical.stat().st_ino == pending.stat().st_ino
    assert canonical.stat().st_dev == pending.stat().st_dev
    assert canonical.stat().st_mode & 0o777 == 0o600
    assert store.parity_order_receipt == _parity_receipt()

    store.write_metric_results_part([])
    with pytest.raises(M5ResultStoreStateError, match="precede metric"):
        store.write_parity_order_receipt(_parity_receipt())
    assert (store.run_path / "FAILURE.json").is_file()

    missing = M5ResultStore.create(project, "official-receipt-missing")
    with pytest.raises(M5ResultStoreStateError, match="pre-metric"):
        missing.write_metric_results_part([])
    assert (missing.run_path / "FAILURE.json").is_file()


def test_official_receipt_accepts_waymax_adapter_type(tmp_path: Path) -> None:
    from evalsim.evaluation.m5_waymax import (
        M5ParityOrderReceipt as WaymaxParityOrderReceipt,
    )

    adapter_receipt = WaymaxParityOrderReceipt(
        rank_domain=M5_PARITY_RANK_DOMAIN,
        order_version=M5_PARITY_ORDER_VERSION,
        ordered_membership_sha256="9" * 64,
        member_count=16,
        transition_count=20,
    )
    project = _project(tmp_path)
    store = M5ResultStore.create(project, "adapter-receipt")
    store.write_parity_order_receipt(adapter_receipt)  # type: ignore[arg-type]
    assert store.parity_order_receipt == _parity_receipt()


def test_official_determinism_receipt_is_typed_and_immutable(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(project, "determinism-receipt")
    store.write_parity_order_receipt(_parity_receipt())
    record = store.write_determinism_receipt(_determinism_receipt())
    canonical = store.run_path / record.path
    pending = store.run_path / "pending" / record.path
    assert canonical.stat().st_ino == pending.stat().st_ino
    assert canonical.stat().st_dev == pending.stat().st_dev
    assert store.determinism_receipt == _determinism_receipt()
    with pytest.raises(FileExistsError):
        store.write_determinism_receipt(_determinism_receipt())
    assert (store.run_path / "FAILURE.json").is_file()


def test_determinism_receipt_requires_exact_domains_and_equality_oracles() -> None:
    payload = _determinism_receipt().to_dict()
    assert payload["metric_row_count"] == 6_656
    assert payload["statistics_row_count"] == 312
    assert payload["metric_passes_equal"] is True
    assert payload["statistics_passes_equal"] is True
    assert payload["reference_matches_log_replay"] is True
    assert payload["zero_error_oracles_passed"] is True

    for key in (
        "metric_passes_equal",
        "statistics_passes_equal",
        "reference_matches_log_replay",
        "zero_error_oracles_passed",
    ):
        with pytest.raises(ValueError, match="exactly true"):
            M5DeterminismReceipt(**{**payload, key: False})
    with pytest.raises(ValueError, match="official domains"):
        M5DeterminismReceipt(
            **{
                **payload,
                "metric_row_count": payload["metric_row_count"] - 1,
            }
        )


def test_result_store_digest_protocol_matches_streaming_evaluator() -> None:
    import evalsim.evaluation.m5 as evaluation_module

    payload = (
        {"cohort_index": 0, "rows": ({"value": 1.0},)},
        {"cohort_index": 1, "rows": ({"value": 2.0},)},
    )
    assert result_store_module._canonical_evaluation_digest(
        "evalsim-m5-case-metric-pass-v1",
        payload,
    ) == evaluation_module._canonical_digest(
        "evalsim-m5-case-metric-pass-v1",
        payload,
    )
    case_digests = {0: "1" * 64, 1: "2" * 64}
    cohort_payload = tuple(
        {
            "cohort_index": cohort_index,
            "sha256": case_digests[cohort_index],
        }
        for cohort_index in sorted(case_digests)
    )
    assert result_store_module._canonical_evaluation_digest(
        "evalsim-m5-cohort-metric-pass-v1",
        cohort_payload,
    ) == evaluation_module._cohort_metric_digest(case_digests)


def test_streaming_artifact_verifier_avoids_full_table_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "streaming-artifacts",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    _write_all(store)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("full-table materialization is forbidden")

    monkeypatch.setattr(
        result_store_module,
        "_read_and_validate_table",
        forbidden,
    )
    monkeypatch.setattr(
        result_store_module,
        "_read_all_artifact_rows",
        forbidden,
    )
    assert result_store_module._verify_artifact_records(
        store.run_path,
        store.artifacts,
        streaming=True,
    ) == _TEST_COUNTS.to_dict()


def test_official_preflight_never_calls_read_all_artifact_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "streaming-preflight",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    _write_all(store)
    scorecard_rows = tuple(
        store.read_dataset(SCORECARDS).to_pylist()
    )
    store.row_accounting_profile = result_store_module.OFFICIAL_M5_PROFILE
    store._parity_order_receipt = _parity_receipt()
    store._determinism_receipt = _determinism_receipt()
    summary = result_store_module._OfficialScanSummary(
        metric_pass_sha256="7" * 64,
        statistics_pass_sha256="8" * 64,
        scorecard_rows=scorecard_rows,
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("official preflight called _read_all_artifact_rows")

    monkeypatch.setattr(
        result_store_module,
        "_read_all_artifact_rows",
        forbidden,
    )
    monkeypatch.setattr(
        M5ResultStore,
        "_verified_supplemental_artifacts",
        lambda self: (),
    )
    monkeypatch.setattr(
        M5ResultStore,
        "_verified_scorecard_report",
        lambda self, records, *, scorecard_rows=None: self.scorecard_report,
    )
    monkeypatch.setattr(
        result_store_module,
        "_scan_official_artifacts",
        lambda run_path, records, receipt: summary,
    )
    monkeypatch.setattr(
        result_store_module,
        "_validate_official_source_binding",
        lambda project_root, provenance: None,
    )
    monkeypatch.setattr(
        result_store_module,
        "_validate_run_members",
        lambda *args, **kwargs: None,
    )
    assert store._preflight_finalization(
        _extended_provenance(project)
    ) == summary


def test_official_source_enumeration_is_exhaustive_and_rejects_ignored_code(
    tmp_path: Path,
) -> None:
    project, expected = _official_source_project(tmp_path)
    assert official_executable_source_paths(project) == expected

    ignored_code = project / "evalsim" / "results" / "ignored.py"
    ignored_code.write_text("IGNORED_EXECUTABLE = True\n", encoding="utf-8")
    with pytest.raises(
        M5ResultStoreIntegrityError,
        match="untracked executable",
    ):
        official_executable_source_paths(project)


def test_historical_official_source_binding_uses_the_recorded_git_tree(
    tmp_path: Path,
) -> None:
    project, source_paths = _official_source_project(tmp_path)
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ("git", "rev-parse", "HEAD^{tree}"),
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = _provenance().to_dict()
    payload.update(
        {
            "executable_source_fingerprint_sha256": (
                executable_source_fingerprint(project, source_paths)
            ),
            "executable_source_paths": list(source_paths),
            "git_commit": commit,
            "git_tree": tree,
        }
    )
    provenance = M5RunProvenance(**payload)
    result_store_module._validate_recorded_official_source_binding(
        project,
        provenance,
    )

    changed = project / "evalsim" / "results" / "sentinel.py"
    changed.write_text("SENTINEL = 'later commit'\n", encoding="utf-8")
    subprocess.run(
        ("git", "add", changed.relative_to(project).as_posix()),
        cwd=project,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "commit", "-m", "later source snapshot"),
        cwd=project,
        check=True,
        capture_output=True,
    )
    result_store_module._validate_recorded_official_source_binding(
        project,
        provenance,
    )


def test_scorecard_adapter_roundtrips_canonical_statistical_states(
    tmp_path: Path,
) -> None:
    suppressed_spec = PairedCellSpec(
        "position_error_m",
        PolicyContrast("constant_velocity", "log_replay"),
    )
    suppressed = analyze_paired_cell(
        suppressed_spec,
        [ScenarioScalar(index, float(index)) for index in range(2)],
        [ScenarioScalar(index, float(index)) for index in range(2)],
    )

    directional_spec = PairedCellSpec(
        "position_error_m",
        PolicyContrast("idm", "log_replay"),
    )
    directional = analyze_paired_cell(
        directional_spec,
        [ScenarioScalar(index, float(index + 1)) for index in range(30)],
        [ScenarioScalar(index, 0.0) for index in range(30)],
    )

    neutral_spec = PairedCellSpec(
        "minimum_center_distance_m",
        PolicyContrast("idm", "constant_velocity"),
    )
    neutral = analyze_paired_cell(
        neutral_spec,
        [ScenarioScalar(index, 2.0) for index in range(10)],
        [ScenarioScalar(index, 1.0) for index in range(10)],
    )

    incomplete_spec = PairedCellSpec(
        "speed_error_mps",
        PolicyContrast("constant_velocity", "log_replay"),
    )
    incomplete = analyze_paired_cell(
        incomplete_spec,
        [ScenarioScalar(index, 0.0) for index in range(2)],
        [
            ScenarioScalar(0, 0.0),
            ScenarioScalar.missing(
                1,
                "no_eligible_target_frame",
                total_components=1,
            ),
        ],
    )
    rows = [
        scorecard_row_from_result(result)
        for result in (suppressed, directional, neutral, incomplete)
    ]
    assert rows[0]["status"] == "insufficient_n"
    assert rows[0]["pointwise_level"] is None
    assert rows[1]["adjusted_level"] is not None
    assert rows[1]["directional_language_allowed"] is True
    assert rows[2]["oriented_mean_advantage"] is None
    assert rows[2]["adjusted_level"] is None
    assert rows[3]["status"] == "pairing_incomplete"

    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "scorecard-adapter",
        expected_rows=ExpectedRowCounts(0, 0, 4, 0),
        data_free=True,
    )
    store.write_metric_results_part([])
    store.write_slice_membership([])
    store.write_scorecards(rows)
    store.write_waymax_parity_summary([])
    store.write_human_readable_scorecard()
    assert store.read_dataset(SCORECARDS).num_rows == 4
    with pytest.raises(
        M5ResultStoreIntegrityError,
        match="exact stored metric/slice derivation",
    ):
        store.finalize(provenance=_provenance(project))


def test_scorecard_adapter_defense_rejects_effect_and_reason_forgery() -> None:
    spec = PairedCellSpec(
        "position_error_m",
        PolicyContrast("idm", "log_replay"),
    )
    result = analyze_paired_cell(
        spec,
        [ScenarioScalar(index, float(index + 1)) for index in range(30)],
        [ScenarioScalar(index, 0.0) for index in range(30)],
    )
    row = scorecard_row_from_result(result)

    wrong_oriented = dict(row)
    wrong_oriented["oriented_mean_advantage"] += 1.0
    with pytest.raises(ValueError, match="oriented mean"):
        result_store_module._scorecard_row(wrong_oriented)

    wrong_raw = dict(row)
    wrong_raw["raw_mean_difference"] += 1.0
    with pytest.raises(ValueError, match="raw mean"):
        result_store_module._scorecard_row(wrong_raw)

    tiny = analyze_paired_cell(
        spec,
        [ScenarioScalar(index, 1e-15) for index in range(10)],
        [ScenarioScalar(index, 0.0) for index in range(10)],
    )
    wrong_tiny_sign = scorecard_row_from_result(tiny)
    wrong_tiny_sign["oriented_mean_advantage"] = abs(
        wrong_tiny_sign["oriented_mean_advantage"]
    )
    with pytest.raises(ValueError, match="oriented mean"):
        result_store_module._scorecard_row(wrong_tiny_sign)

    incomplete_spec = PairedCellSpec(
        "speed_error_mps",
        PolicyContrast("constant_velocity", "log_replay"),
    )
    incomplete = analyze_paired_cell(
        incomplete_spec,
        [ScenarioScalar(index, 0.0) for index in range(2)],
        [
            ScenarioScalar(0, 0.0),
            ScenarioScalar.missing(
                1,
                "no_eligible_target_frame",
                total_components=1,
            ),
        ],
    )
    zero_reason = scorecard_row_from_result(incomplete)
    zero_reason["missing_reasons_b_json"] = (
        '{"no_eligible_target_frame":0}'
    )
    with pytest.raises(ValueError, match="at least 1"):
        result_store_module._scorecard_row(zero_reason)


def test_scorecard_adapter_accepts_cancellation_heavy_canonical_means() -> None:
    spec = PairedCellSpec(
        "minimum_center_distance_m",
        PolicyContrast("idm", "constant_velocity"),
    )
    base = 1000.0
    effect = 1e-12
    result = analyze_paired_cell(
        spec,
        [
            ScenarioScalar(index, base + (index + 1) * effect)
            for index in range(10)
        ],
        [
            ScenarioScalar(index, base + index * effect)
            for index in range(10)
        ],
    )
    assert (
        result.raw_mean_difference
        != result.policy_a_mean - result.policy_b_mean
    )
    assert scorecard_row_from_result(result)["status"] == "small_or_sparse"


def test_scorecard_semantic_invariants_reject_impossible_aggregates() -> None:
    small_spec = PairedCellSpec(
        "position_error_m",
        PolicyContrast("idm", "log_replay"),
    )
    small = analyze_paired_cell(
        small_spec,
        [ScenarioScalar(index, float(index + 1)) for index in range(10)],
        [ScenarioScalar(index, 0.0) for index in range(10)],
    )
    small_row = scorecard_row_from_result(small)

    forged_standardized = dict(small_row)
    forged_standardized["standardized_signal_to_heterogeneity"] = 123.0
    with pytest.raises(ValueError, match="sample thresholds"):
        result_store_module._scorecard_row(forged_standardized)

    forged_favorable = dict(small_row)
    forged_favorable["favorable_proportion"] = 0.55
    with pytest.raises(ValueError, match="wins/ties lattice"):
        result_store_module._scorecard_row(forged_favorable)

    component_shortfall = _scorecard_row()
    component_shortfall["eligible_components_a"] = 0
    with pytest.raises(ValueError, match="valid scenario scalar"):
        result_store_module._scorecard_row(component_shortfall)

    zero_effect = analyze_paired_cell(
        small_spec,
        [ScenarioScalar(index, 1.0) for index in range(10)],
        [ScenarioScalar(index, 1.0) for index in range(10)],
    )
    forged_zero_effect = scorecard_row_from_result(zero_effect)
    forged_zero_effect["raw_median_difference"] = 0.25
    with pytest.raises(ValueError, match="zero effects and bands"):
        result_store_module._scorecard_row(forged_zero_effect)

    large = analyze_paired_cell(
        small_spec,
        [ScenarioScalar(index, float(index + 1)) for index in range(30)],
        [ScenarioScalar(index, 0.0) for index in range(30)],
    )
    forged_sign = scorecard_row_from_result(large)
    forged_sign["standardized_signal_to_heterogeneity"] = abs(
        forged_sign["standardized_signal_to_heterogeneity"]
    )
    with pytest.raises(ValueError, match="oriented mean sign"):
        result_store_module._scorecard_row(forged_sign)

    forged_adjusted = scorecard_row_from_result(large)
    pointwise_midpoint = (
        forged_adjusted["pointwise_lower"]
        + forged_adjusted["pointwise_upper"]
    ) / 2.0
    forged_adjusted["adjusted_lower"] = pointwise_midpoint
    forged_adjusted["adjusted_upper"] = pointwise_midpoint
    with pytest.raises(ValueError, match="contain its pointwise band"):
        result_store_module._scorecard_row(forged_adjusted)


def test_scorecard_derivation_is_bound_to_metric_and_slice_rows() -> None:
    spec = PairedCellSpec(
        "position_error_m",
        PolicyContrast("constant_velocity", "log_replay"),
    )
    values_a = [
        ScenarioScalar(index, float(index + 1))
        for index in range(10)
    ]
    values_b = [
        ScenarioScalar(index, 0.0)
        for index in range(10)
    ]
    metric_rows: list[dict[str, object]] = []
    for execution_name, values in (
        ("constant_velocity", values_a),
        ("log_replay", values_b),
    ):
        for scalar in values:
            row = _metric_row(
                scalar.cohort_index,
                execution_name=execution_name,
                seed=0,
            )
            row["distribution"] = [scalar.value]
            row["value"] = scalar.value
            metric_rows.append(row)
    slice_rows = []
    for index in range(10):
        row = _slice_row(index)
        row["member"] = True
        slice_rows.append(row)

    expected = scorecard_row_from_result(
        analyze_paired_cell(spec, values_a, values_b)
    )
    rows = {
        METRIC_RESULTS: metric_rows,
        SLICE_MEMBERSHIP: slice_rows,
        SCORECARDS: [expected],
    }
    result_store_module._validate_scorecards_derived_from_rows(rows)

    coherent_but_forged = scorecard_row_from_result(
        analyze_paired_cell(
            spec,
            [
                ScenarioScalar(index, float(index + 2))
                for index in range(10)
            ],
            values_b,
        )
    )
    rows[SCORECARDS] = [coherent_but_forged]
    with pytest.raises(
        M5ResultStoreIntegrityError,
        match="exact stored metric/slice derivation",
    ):
        result_store_module._validate_scorecards_derived_from_rows(rows)


def test_data_free_verification_rederives_scorecards_after_coherent_rehash(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    store = M5ResultStore.create(
        project,
        "rehash-slice-membership",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    _write_all(store)
    store.finalize(provenance=_provenance(project))

    slice_path = store.run_path / "slice-membership.parquet"
    rows = store.read_dataset(SLICE_MEMBERSHIP).to_pylist()
    rows[1]["member"] = False
    table = pa.Table.from_pylist(
        rows,
        schema=M5_RESULT_SCHEMAS[SLICE_MEMBERSHIP],
    )
    with slice_path.open("wb") as handle:
        result_store_module.pq.write_table(
            table,
            handle,
            compression="NONE",
            use_dictionary=False,
            write_statistics=True,
            version="2.6",
            data_page_version="2.0",
        )
        handle.flush()
        os.fsync(handle.fileno())

    manifest_path = store.run_path / "evaluation-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        if artifact["path"] == "slice-membership.parquet":
            encoded = slice_path.read_bytes()
            artifact["sha256"] = hashlib.sha256(encoded).hexdigest()
            artifact["size_bytes"] = len(encoded)
            break
    else:  # pragma: no cover - protected by the valid finalized fixture
        raise AssertionError("slice-membership artifact is absent")
    manifest_path.write_bytes(
        (
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )

    with pytest.raises(
        M5ResultStoreIntegrityError,
        match="exact stored metric/slice derivation",
    ):
        verify_m5_result_store(
            project,
            "rehash-slice-membership",
            allow_data_free=True,
        )


def test_official_cartesian_domain_gate_rejects_equal_count_drift() -> None:
    metric_names = tuple(result_store_module.M5_METRIC_SPECS)
    execution_names = tuple(sorted(result_store_module._M5_EXECUTION_ROLES))
    slice_names = tuple(result_store_module._M5_SLICE_SPECS)
    policy_pairs = result_store_module._M5_POLICY_CONTRASTS
    policy_names = result_store_module._M5_POLICY_NAMES
    parity_metrics = result_store_module._M5_PARITY_METRIC_NAMES
    rows = {
        METRIC_RESULTS: [
            {
                "cohort_index": cohort_index,
                "execution_name": execution_name,
                "metric_name": metric_name,
                "seed": 0,
                "valid": True,
            }
            for cohort_index in range(128)
            for execution_name in execution_names
            for metric_name in metric_names
        ],
        SLICE_MEMBERSHIP: [
            {
                "cohort_index": cohort_index,
                "eligible": True,
                "member": (
                    slice_name == "all" or cohort_index % 2 == 0
                ),
                "slice_name": slice_name,
            }
            for cohort_index in range(128)
            for slice_name in slice_names
        ],
        SCORECARDS: [
            {
                "cohort_n": 128 if slice_name == "all" else 64,
                "metric_name": metric_name,
                "paired_n": 128 if slice_name == "all" else 64,
                "policy_a": policy_a,
                "policy_b": policy_b,
                "slice_name": slice_name,
                "source_pairing_complete": True,
                "valid_a_n": 128 if slice_name == "all" else 64,
                "valid_b_n": 128 if slice_name == "all" else 64,
            }
            for metric_name in metric_names
            for slice_name in slice_names
            for policy_a, policy_b in policy_pairs
        ],
        WAYMAX_PARITY_SUMMARY: [
            {
                "compared_components": 1,
                "exact_match": True,
                "metric_name": metric_name,
                "mismatch_count": 0,
                "parity_index": parity_index,
                "policy_name": policy_name,
                "status": "accepted",
            }
            for parity_index in range(16)
            for policy_name in policy_names
            for metric_name in parity_metrics
        ],
    }
    result_store_module._validate_official_key_domains(rows)

    rows[METRIC_RESULTS][0]["seed"] = 7
    with pytest.raises(M5ResultStoreIntegrityError, match="seed zero"):
        result_store_module._validate_official_key_domains(rows)
    rows[METRIC_RESULTS][0]["seed"] = 0

    primary_reference = next(
        row
        for row in rows[METRIC_RESULTS]
        if (
            row["execution_name"]
            == "waymax_exact_log_state_dynamics"
            and row["metric_name"]
            in result_store_module.M5_PRIMARY_METRIC_NAMES
        )
    )
    primary_reference["valid"] = False
    with pytest.raises(
        M5ResultStoreIntegrityError,
        match="all four executions",
    ):
        result_store_module._validate_official_key_domains(rows)
    primary_reference["valid"] = True

    rows[WAYMAX_PARITY_SUMMARY][0]["status"] = "rejected"
    with pytest.raises(M5ResultStoreIntegrityError, match="16×3×3"):
        result_store_module._validate_official_key_domains(rows)
    rows[WAYMAX_PARITY_SUMMARY][0]["status"] = "accepted"

    sliced_scorecard = next(
        row
        for row in rows[SCORECARDS]
        if row["slice_name"] != "all"
    )
    sliced_scorecard["cohort_n"] += 1
    with pytest.raises(
        M5ResultStoreIntegrityError,
        match="source-only slice membership",
    ):
        result_store_module._validate_official_key_domains(rows)
    sliced_scorecard["cohort_n"] -= 1

    primary_all = next(
        row
        for row in rows[SCORECARDS]
        if (
            row["metric_name"]
            in result_store_module.M5_PRIMARY_METRIC_NAMES
            and row["slice_name"] == "all"
        )
    )
    primary_all["paired_n"] = 127
    with pytest.raises(
        M5ResultStoreIntegrityError,
        match="128 valid paired",
    ):
        result_store_module._validate_official_key_domains(rows)
    primary_all["paired_n"] = 128

    rows[METRIC_RESULTS][0]["cohort_index"] = 128
    with pytest.raises(
        M5ResultStoreIntegrityError,
        match="128×4×13",
    ):
        result_store_module._validate_official_key_domains(rows)


def test_canonical_key_definitions_do_not_repeat_fields() -> None:
    for fields in result_store_module._KEY_FIELDS.values():
        assert len(fields) == len(set(fields))


def test_every_write_path_fsyncs_files_and_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path)
    calls: list[int] = []
    real_fsync = result_store_module._fsync_descriptor

    def recording_fsync(descriptor: int) -> None:
        calls.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(
        result_store_module,
        "_fsync_descriptor",
        recording_fsync,
    )
    store = M5ResultStore.create(
        project,
        "fsync",
        expected_rows=_TEST_COUNTS,
        data_free=True,
    )
    _write_all(store)
    store.finalize(provenance=_provenance(project))

    # Creation, four Parquet parts, the report, their canonical hard links,
    # three control files, and parent directories cross the durability seam.
    assert len(calls) >= 17
    assert (
        verify_m5_result_store(
            project,
            "fsync",
            allow_data_free=True,
        ).manifest["complete"]
        is True
    )


def test_schema_constants_are_fixed_and_metadata_bound() -> None:
    assert set(M5_RESULT_SCHEMAS) == {
        METRIC_RESULTS,
        SLICE_MEMBERSHIP,
        SCORECARDS,
        WAYMAX_PARITY_SUMMARY,
    }
    for dataset, schema in M5_RESULT_SCHEMAS.items():
        assert isinstance(schema, pa.Schema)
        assert schema.metadata[b"evalsim.dataset"].decode("ascii") == dataset
        assert (
            schema.metadata[b"evalsim.schema_version"].decode("ascii")
            == "m5-result-store-1.0.0"
        )
