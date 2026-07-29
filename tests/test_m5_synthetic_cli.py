"""Data-free tests for the isolated M5 synthetic command boundary."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

import evalsim.results.m5 as result_store_module
from evalsim.cli import m5_synthetic as cli
from evalsim.cli._terminal import (
    TerminalBoundaryError,
    TerminalizedFailure,
    capture_terminal,
)
from evalsim.results import (
    M5RunProvenance,
    M5ResultStore,
    executable_source_fingerprint,
    verify_m5_result_store,
)


def _elapsed(value: int = 1) -> dict[str, int]:
    return {
        name: value
        for name in cli.M5_SYNTHETIC_STAGE_NAMES
    }


def _argv(root: Path, run_name: str = "synthetic-test") -> list[str]:
    return [
        "--project-root",
        os.fspath(root),
        "--run-name",
        run_name,
    ]


def _git_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        "[project]\nname='synthetic-test'\nversion='0.0.0'\n",
        encoding="utf-8",
    )
    (project / ".gitignore").write_text("outputs/\n", encoding="utf-8")
    subprocess.run(
        ("git", "init", "-b", "main"),
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
        (
            "git",
            "-c",
            "user.name=EvalSim Test",
            "-c",
            "user.email=evalsim-test@example.invalid",
            "commit",
            "-m",
            "synthetic CLI fixture",
        ),
        cwd=project,
        check=True,
        capture_output=True,
    )
    return project


def _provenance(project: Path) -> M5RunProvenance:
    source = project / "source.py"
    source.write_text("SYNTHETIC_CLI_TEST = True\n", encoding="utf-8")
    paths = ("source.py",)
    return M5RunProvenance(
        m4_manifest_sha256="a" * 64,
        m4_execution_provenance_sha256="b" * 64,
        selected_order_version="m5-synthetic-order-1",
        selected_order_fingerprint_sha256="c" * 64,
        executable_source_fingerprint_sha256=(
            executable_source_fingerprint(project, paths)
        ),
        executable_source_paths=paths,
        git_commit="d" * 40,
        git_tree="e" * 40,
        simulator_specs=cli._simulator_specs(),
        runtime_versions={
            "flax": "not-loaded",
            "jax": "not-loaded",
            "jaxlib": "not-loaded",
            "numpy": "test",
            "pyarrow": "test",
            "python": "test",
            "tensorflow": "not-loaded",
            "waymo_waymax": "not-loaded",
        },
    )


def test_main_success_stdout_is_one_exact_canonical_object(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = cli._RunResult(
        result_relative=Path("outputs/m5/synthetic-test"),
        row_counts=cli.M5_SYNTHETIC_EXPECTED_ROWS.to_dict(),
        elapsed_ms=_elapsed(7),
    )
    monkeypatch.setattr(cli, "run_synthetic", lambda _: result)

    assert cli.main(_argv(tmp_path)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        '{"elapsed_ms":{"artifact_publication":7,"cohort_evaluation":7,'
        '"finalization":7,"preflight":7,"statistics":7},'
        '"profile":"data_free_test",'
        '"result_path":"outputs/m5/synthetic-test",'
        '"row_counts":{"metric-results":195,"scorecards":312,'
        '"slice-membership":40,"waymax-parity-summary":0},'
        '"schema_version":"m5-cli-status-1.0.0","status":"success"}\n'
    )


@pytest.mark.parametrize(
    "extra",
    (
        ["--data-dir", "/private/data/sentinel"],
        ["--m4-run-dir", "/private/outputs/m4/sentinel"],
        ["--mode", "official"],
    ),
)
def test_parser_rejects_every_data_or_official_argument_without_echo(
    extra: list[str],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main([*_argv(tmp_path), *extra])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        '{"reason_code":"argument_error",'
        '"schema_version":"m5-cli-status-1.0.0",'
        '"status":"rejected"}\n'
    )
    assert "/private/" not in captured.err
    assert "official" not in captured.err


def test_untrusted_main_failure_never_echoes_exception_or_absolute_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    sentinel = "/private/local/native-scenario-sentinel"

    def fail(_: argparse.Namespace) -> cli._RunResult:
        raise RuntimeError(sentinel)

    monkeypatch.setattr(cli, "run_synthetic", fail)
    with pytest.raises(SystemExit) as caught:
        cli.main(_argv(tmp_path))
    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        '{"reason_code":"unexpected_failure",'
        '"schema_version":"m5-cli-status-1.0.0",'
        '"status":"rejected"}\n'
    )
    assert sentinel not in captured.err


def test_terminal_capture_requires_silent_success_and_retains_details_locally(
    capfd: pytest.CaptureFixture[str],
) -> None:
    sentinel = "synthetic-private-terminal-sentinel"

    def noisy() -> None:
        print(sentinel, flush=True)
        os.write(2, b"native-stderr-private-sentinel\n")

    with capfd.disabled():
        with pytest.raises(TerminalizedFailure) as caught:
            capture_terminal(noisy)
    failure = caught.value
    try:
        assert type(failure.primary) is TerminalBoundaryError
        assert failure.primary.code == "terminal_output_detected"
        assert sentinel.encode("ascii") in failure.transcript
        assert b"native-stderr-private-sentinel" in failure.transcript
        assert sentinel not in repr(failure)
    finally:
        failure.terminal_status.close_best_effort()
    captured = capfd.readouterr()
    assert sentinel not in captured.out
    assert sentinel not in captured.err


def test_capture_success_preserves_noninheritable_status_descriptors() -> None:
    captured = capture_terminal(lambda: 17)
    try:
        assert captured.value == 17
        assert os.get_inheritable(captured.terminal_status.stdout_fd) is False
        assert os.get_inheritable(captured.terminal_status.stderr_fd) is False
    finally:
        captured.terminal_status.close_best_effort()
    for descriptor in (
        captured.terminal_status.stdout_fd,
        captured.terminal_status.stderr_fd,
    ):
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_captured_failure_creates_owner_only_ignored_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    sentinel = "synthetic-private-diagnostic-sentinel"
    monkeypatch.setattr(cli, "_validated_root", lambda _: tmp_path)
    monkeypatch.setattr(cli, "_assert_running_checkout", lambda _: None)
    monkeypatch.setattr(cli, "_preflight_output", lambda *args: None)

    def noisy_failure(*args: object) -> cli._PendingRun:
        del args
        print(sentinel, flush=True)
        raise RuntimeError("/private/native-id-in-local-trace")

    monkeypatch.setattr(cli, "_execute_pending", noisy_failure)
    args = argparse.Namespace(
        project_root=tmp_path,
        run_name="injected-failure",
    )
    with capfd.disabled():
        with pytest.raises(cli._CommandFailure) as caught:
            cli.run_synthetic(args)
    failure = caught.value
    try:
        assert failure.failure_relative == Path(
            "outputs/m5/injected-failure/FAILURE.json"
        )
        run = tmp_path / "outputs" / "m5" / "injected-failure"
        diagnostic = run / "failure-details.log"
        assert sentinel.encode("ascii") in diagnostic.read_bytes()
        assert b"/private/native-id-in-local-trace" in diagnostic.read_bytes()
        assert stat.S_IMODE(diagnostic.stat().st_mode) == 0o600
        assert json.loads(
            (run / "FAILURE.json").read_text(encoding="utf-8")
        )["complete"] is False
        assert not (run / "SUCCESS").exists()
        assert not (run / "evaluation-manifest.json").exists()
    finally:
        if failure.terminal_status is not None:
            failure.terminal_status.close_best_effort()
    captured = capfd.readouterr()
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert "/private/" not in captured.err


def test_git_preflight_requires_clean_ignored_new_output(
    tmp_path: Path,
) -> None:
    project = _git_project(tmp_path)
    root = cli._validated_root(project)
    cli._preflight_output(root, "accepted")

    (project / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(cli.M5SyntheticCommandError) as caught:
        cli._preflight_output(root, "rejected")
    assert caught.value.code == "dirty_worktree"


def test_success_contract_rejects_missing_stage_or_wrong_rows() -> None:
    missing_stage = _elapsed()
    missing_stage.pop("statistics")
    with pytest.raises(cli.M5SyntheticCommandError) as caught:
        cli._success_output(
            cli._RunResult(
                result_relative=Path("outputs/m5/invented"),
                row_counts=cli.M5_SYNTHETIC_EXPECTED_ROWS.to_dict(),
                elapsed_ms=missing_stage,
            )
        )
    assert caught.value.code == "result_contract_failed"

    wrong_rows = cli.M5_SYNTHETIC_EXPECTED_ROWS.to_dict()
    wrong_rows["metric-results"] -= 1
    with pytest.raises(cli.M5SyntheticCommandError) as caught:
        cli._success_output(
            cli._RunResult(
                result_relative=Path("outputs/m5/invented"),
                row_counts=wrong_rows,
                elapsed_ms=_elapsed(),
            )
        )
    assert caught.value.code == "result_contract_failed"


def test_full_synthetic_runner_store_and_report_path_is_data_free(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provenance = _provenance(tmp_path)
    monkeypatch.setattr(cli, "_build_provenance", lambda _: provenance)
    store = M5ResultStore.create(
        tmp_path,
        "full-data-free",
        expected_rows=cli.M5_SYNTHETIC_EXPECTED_ROWS,
        data_free=True,
    )

    holder = cli._PreparedHolder()
    pending = cli._execute_pending(store, tmp_path, holder)
    assert set(pending.result.elapsed_ms) == set(
        cli.M5_SYNTHETIC_STAGE_NAMES
    )
    assert pending.prepared is holder.prepared
    assert pending.result.success_payload == cli._success_output(
        pending.result
    )
    assert (store.run_path / "FINALIZING").is_file()
    assert (store.run_path / "evaluation-manifest.json").is_file()
    assert not (store.run_path / "SUCCESS").exists()
    store.commit_finalization(pending.prepared)
    verified = verify_m5_result_store(
        tmp_path,
        "full-data-free",
        allow_data_free=True,
    )

    assert verified.manifest["actual_rows"] == (
        cli.M5_SYNTHETIC_EXPECTED_ROWS.to_dict()
    )
    assert verified.manifest["row_accounting_profile"] == "data_free_test"
    assert (
        store.read_dataset("waymax-parity-summary").num_rows
        == 0
    )
    assert (store.run_path / "scorecard.md").is_file()


def test_finalization_warning_and_native_write_are_captured_before_success(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    sentinel = "m5-finalization-private-warning-sentinel"
    provenance = _provenance(tmp_path)
    monkeypatch.setattr(cli, "_validated_root", lambda _: tmp_path)
    monkeypatch.setattr(cli, "_assert_running_checkout", lambda _: None)
    monkeypatch.setattr(cli, "_preflight_output", lambda *args: None)
    monkeypatch.setattr(cli, "_build_provenance", lambda _: provenance)
    real_prepare = M5ResultStore.prepare_finalization

    def noisy_prepare(
        self: M5ResultStore,
        *,
        provenance: M5RunProvenance,
    ) -> object:
        prepared = real_prepare(self, provenance=provenance)
        sys.stderr.write(f"RuntimeWarning: {sentinel}\n")
        sys.stderr.flush()
        os.write(1, f"{sentinel}-native\n".encode("ascii"))
        return prepared

    monkeypatch.setattr(
        M5ResultStore,
        "prepare_finalization",
        noisy_prepare,
    )
    args = argparse.Namespace(
        project_root=tmp_path,
        run_name="finalization-output",
    )
    with capfd.disabled():
        with pytest.raises(cli._CommandFailure) as caught:
            cli.run_synthetic(args)
    failure = caught.value
    try:
        assert failure.reason_code == "terminal_output_detected"
        run = tmp_path / "outputs" / "m5" / "finalization-output"
        record = json.loads(
            (run / "FAILURE.json").read_text(encoding="utf-8")
        )
        assert record["reason_code"] == failure.reason_code
        assert sentinel.encode("ascii") in (
            run / "failure-details.log"
        ).read_bytes()
        assert (run / "evaluation-manifest.json").is_file()
        assert not (run / "SUCCESS").exists()
    finally:
        if failure.terminal_status is not None:
            failure.terminal_status.close_best_effort()
    captured = capfd.readouterr()
    assert sentinel not in captured.out
    assert sentinel not in captured.err


def test_post_prepare_timing_failure_aborts_before_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provenance = _provenance(tmp_path)
    monkeypatch.setattr(cli, "_validated_root", lambda _: tmp_path)
    monkeypatch.setattr(cli, "_assert_running_checkout", lambda _: None)
    monkeypatch.setattr(cli, "_preflight_output", lambda *args: None)
    monkeypatch.setattr(cli, "_build_provenance", lambda _: provenance)
    real_elapsed = cli._elapsed_ms
    calls = 0

    def injected_elapsed(start_ns: int, end_ns: int) -> int:
        nonlocal calls
        calls += 1
        if calls == len(cli.M5_SYNTHETIC_STAGE_NAMES):
            raise cli.M5SyntheticCommandError(
                "result_contract_failed",
                "injected post-prepare timing failure",
            )
        return real_elapsed(start_ns, end_ns)

    monkeypatch.setattr(cli, "_elapsed_ms", injected_elapsed)
    args = argparse.Namespace(
        project_root=tmp_path,
        run_name="timing-failure",
    )
    with pytest.raises(cli._CommandFailure) as caught:
        cli.run_synthetic(args)
    failure = caught.value
    try:
        assert calls == len(cli.M5_SYNTHETIC_STAGE_NAMES)
        assert failure.reason_code == "result_contract_failed"
        run = tmp_path / "outputs" / "m5" / "timing-failure"
        record = json.loads(
            (run / "FAILURE.json").read_text(encoding="utf-8")
        )
        assert record["reason_code"] == failure.reason_code
        assert (run / "evaluation-manifest.json").is_file()
        assert not (run / "SUCCESS").exists()
    finally:
        if failure.terminal_status is not None:
            failure.terminal_status.close_best_effort()


def test_success_serialization_failure_cannot_follow_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provenance = _provenance(tmp_path)
    monkeypatch.setattr(cli, "_validated_root", lambda _: tmp_path)
    monkeypatch.setattr(cli, "_assert_running_checkout", lambda _: None)
    monkeypatch.setattr(cli, "_preflight_output", lambda *args: None)
    monkeypatch.setattr(cli, "_build_provenance", lambda _: provenance)

    def reject_success(_: cli._RunResult) -> bytes:
        raise cli.M5SyntheticCommandError(
            "result_contract_failed",
            "injected success serialization failure",
        )

    monkeypatch.setattr(cli, "_success_output", reject_success)
    args = argparse.Namespace(
        project_root=tmp_path,
        run_name="success-serialization-failure",
    )
    with pytest.raises(cli._CommandFailure) as caught:
        cli.run_synthetic(args)
    failure = caught.value
    try:
        assert failure.reason_code == "result_contract_failed"
        run = (
            tmp_path
            / "outputs"
            / "m5"
            / "success-serialization-failure"
        )
        assert (run / "evaluation-manifest.json").is_file()
        assert json.loads(
            (run / "FAILURE.json").read_text(encoding="utf-8")
        )["reason_code"] == failure.reason_code
        assert not (run / "SUCCESS").exists()
    finally:
        if failure.terminal_status is not None:
            failure.terminal_status.close_best_effort()


def test_post_create_commit_error_cannot_report_failure_after_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provenance = _provenance(tmp_path)
    monkeypatch.setattr(cli, "_validated_root", lambda _: tmp_path)
    monkeypatch.setattr(cli, "_assert_running_checkout", lambda _: None)
    monkeypatch.setattr(cli, "_preflight_output", lambda *args: None)
    monkeypatch.setattr(cli, "_build_provenance", lambda _: provenance)
    real_write = result_store_module._write_bytes_exclusive

    def write_then_raise(path: Path, payload: bytes) -> None:
        real_write(path, payload)
        if path.name == "SUCCESS":
            raise OSError("injected post-create commit error")

    monkeypatch.setattr(
        result_store_module,
        "_write_bytes_exclusive",
        write_then_raise,
    )
    result = cli.run_synthetic(
        argparse.Namespace(
            project_root=tmp_path,
            run_name="post-create-commit-error",
        )
    )
    try:
        run = (
            tmp_path
            / "outputs"
            / "m5"
            / "post-create-commit-error"
        )
        assert (run / "SUCCESS").read_bytes() == b"SUCCESS\n"
        assert not (run / "FAILURE.json").exists()
        assert result.success_payload == cli._success_output(result)
    finally:
        if result.terminal_status is not None:
            result.terminal_status.close_best_effort()


def test_artifact_failure_terminal_code_matches_failure_record(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    provenance = _provenance(tmp_path)
    monkeypatch.setattr(cli, "_validated_root", lambda _: tmp_path)
    monkeypatch.setattr(cli, "_assert_running_checkout", lambda _: None)
    monkeypatch.setattr(cli, "_preflight_output", lambda *args: None)
    monkeypatch.setattr(cli, "_build_provenance", lambda _: provenance)

    def injected_write_failure(
        self: M5ResultStore,
        *args: object,
        **kwargs: object,
    ) -> object:
        del args, kwargs
        self._poison("metric_results_write_failed")
        raise OSError("/private/artifact-write-sentinel")

    monkeypatch.setattr(
        M5ResultStore,
        "write_metric_results_part",
        injected_write_failure,
    )
    with pytest.raises(SystemExit) as caught:
        cli.main(_argv(tmp_path, "artifact-write-failure"))
    assert caught.value.code == 1
    captured = capfd.readouterr()
    assert captured.out == ""
    terminal = json.loads(captured.err)
    record = json.loads(
        (
            tmp_path
            / "outputs"
            / "m5"
            / "artifact-write-failure"
            / "FAILURE.json"
        ).read_text(encoding="utf-8")
    )
    assert terminal == {
        "failure_record": (
            "outputs/m5/artifact-write-failure/FAILURE.json"
        ),
        "reason_code": "metric_results_write_failed",
        "schema_version": cli.M5_SYNTHETIC_STATUS_SCHEMA_VERSION,
        "status": "failure",
    }
    assert terminal["reason_code"] == record["reason_code"]
    assert "/private/" not in captured.err


def test_core_only_cli_import_does_not_load_optional_waymo_stack() -> None:
    script = r"""
import importlib.abc
import sys

blocked = ("jax", "tensorflow", "waymax")

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == blocked or fullname.startswith(tuple(name + "." for name in blocked)):
            raise AssertionError("optional import attempted: " + fullname)
        return None

sys.meta_path.insert(0, BlockOptional())
import evalsim.cli.m5_synthetic
assert not any(
    name == root or name.startswith(root + ".")
    for name in sys.modules
    for root in blocked
)
"""
    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
