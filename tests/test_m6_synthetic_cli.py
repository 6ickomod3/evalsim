"""Focused hostile tests for the capture-first M6 synthetic command."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest

from evalsim.cli import m6_synthetic as cli
from evalsim.results import m6 as results


_SOURCE_PATHS = (
    ".gitignore",
    "NOTICE.md",
    "pyproject.toml",
    "uv.lock",
)


def _run_git(project: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _initialize_approved_git(project: Path, message: str) -> None:
    _run_git(project, "init", "-b", "main")
    _run_git(project, "add", ".")
    _run_git(
        project,
        "-c",
        "user.name=EvalSim Test",
        "-c",
        "user.email=evalsim-test@example.invalid",
        "commit",
        "-m",
        message,
    )
    commit = _run_git(project, "rev-parse", "HEAD")
    _run_git(project, "remote", "add", "origin", cli._official._CANONICAL_REMOTE)
    _run_git(project, "update-ref", "refs/remotes/origin/main", commit)
    tag = cli._official._APPROVED_IMPLEMENTATION_REF.removeprefix("refs/tags/")
    _run_git(project, "tag", tag, commit)


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.parent.mkdir(parents=True, exist_ok=True)
    project.mkdir()
    (project / ".gitignore").write_text("outputs/\n", encoding="utf-8")
    (project / "NOTICE.md").write_text("test notice\n", encoding="utf-8")
    (project / "pyproject.toml").write_text(
        "[project]\nname='m6-synthetic-test'\nversion='0.0.0'\n",
        encoding="utf-8",
    )
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _initialize_approved_git(project, "approved fixture")
    return project


def _production_project(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[1]
    project = tmp_path / "production-project"
    project.mkdir()
    for relative in (
        ".gitignore",
        ".python-version",
        "AGENTS.md",
        "NOTICE.md",
        "pyproject.toml",
        "uv.lock",
    ):
        shutil.copy2(source / relative, project / relative)
    shutil.copytree(
        source / "evalsim",
        project / "evalsim",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    tests_dir = project / "tests"
    tests_dir.mkdir()
    for path in sorted((source / "tests").glob("test_m6_*.py")):
        shutil.copy2(path, tests_dir / path.name)
    plans_dir = project / "docs/plans"
    plans_dir.mkdir(parents=True)
    for name in (
        "2026-07-29-m6-counterfactual-reactivity.md",
        "2026-07-29-m6-data-free-implementation-checkpoint.md",
    ):
        shutil.copy2(source / "docs/plans" / name, plans_dir / name)
    _initialize_approved_git(project, "full approved fixture")
    return project


def _approval(
    project: Path,
    *,
    events: list[str] | None = None,
    verifier=cli._verify_approved_implementation,
) -> cli.M6ImplementationApproval:
    if events is not None:
        events.append("approval")
    commit = _run_git(project, "rev-parse", "HEAD")
    return verifier(
        project,
        live_lookup=lambda _: commit,
        live_approval_lookup=lambda _: commit,
        source_paths_resolver=lambda _: _SOURCE_PATHS,
        module_validator=lambda _root, _paths: None,
    )


def _complete_prepared_store(
    project: Path,
    run_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    cli._PreparedRun,
    cli.M6ImplementationApproval,
    results.M6VerifiedProvenance,
]:
    from tests import test_m6_results as fixtures

    approval = _approval(project)
    provenance = cli._build_typed_provenance(approval)
    monkeypatch.setattr(
        fixtures,
        "_verified_provenance",
        lambda _mode: provenance,
    )
    store = results.M6ResultStore.create(
        project,
        run_name,
        mode=results.DATA_FREE_MODE,
    )
    fixtures._write_complete(store)
    prepared = cli._PreparedRun(
        store=store,
        approval=approval,
        provenance=provenance,
        approval_provider=lambda _: approval,
        provenance_observer=lambda _: provenance,
        success_payload=b"unused",
    )
    return prepared, approval, provenance


def test_import_attempts_no_optional_runtime() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = r'''
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {
            "flax", "jax", "tensorflow", "waymax"
        }:
            raise AssertionError(f"optional import attempted: {fullname}")
        return None

sys.meta_path.insert(0, BlockOptional())
from evalsim.cli.m6_synthetic import (
    M6_SYNTHETIC_PROFILE,
    M6ImplementationApproval,
)
assert M6_SYNTHETIC_PROFILE == "data_free_m6"
assert M6ImplementationApproval.__name__ == "M6ImplementationApproval"
'''
    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_parser_surface_is_exact(tmp_path: Path) -> None:
    request = cli._parse_request(
        [
            "--project-root",
            os.fspath(tmp_path),
            "--run-name",
            "m6-data-free",
        ]
    )
    assert request.project_root == tmp_path
    assert request.run_name == "m6-data-free"
    with pytest.raises(cli.M6SyntheticCommandError, match="argument_error"):
        cli._parse_request(["--project-root", os.fspath(tmp_path)])
    with pytest.raises(cli.M6SyntheticCommandError, match="argument_error"):
        cli._parse_request(
            [
                "--project-root",
                os.fspath(tmp_path),
                "--run-name",
                "Bad/Path",
            ]
        )


@pytest.mark.parametrize("required", [".gitignore", "NOTICE.md"])
def test_project_root_requires_notice_and_gitignore(
    tmp_path: Path,
    required: str,
) -> None:
    project = _project(tmp_path)
    (project / required).unlink()
    with pytest.raises(
        cli.M6SyntheticCommandError,
        match="project_root_invalid",
    ):
        cli._validated_root(project)


def test_approval_is_verifier_issued_and_bound_to_exact_head(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    approval = _approval(project)
    head = _run_git(project, "rev-parse", "HEAD")
    assert approval.repository.git.commit == head
    assert (
        approval.repository.git.approval_ref
        == cli._official._APPROVED_IMPLEMENTATION_REF
    )
    assert not hasattr(approval, "decisions")

    (project / "change.txt").write_text("new commit\n", encoding="utf-8")
    _run_git(project, "add", "change.txt")
    _run_git(
        project,
        "-c",
        "user.name=EvalSim Test",
        "-c",
        "user.email=evalsim-test@example.invalid",
        "commit",
        "-m",
        "unapproved change",
    )
    changed = _run_git(project, "rev-parse", "HEAD")
    _run_git(project, "update-ref", "refs/remotes/origin/main", changed)
    with pytest.raises(
        cli.M6SyntheticCommandError,
        match="approved_commit_mismatch",
    ):
        cli._verify_approved_implementation(
            project,
            live_lookup=lambda _: changed,
            live_approval_lookup=lambda _: head,
            source_paths_resolver=lambda _: _SOURCE_PATHS,
            module_validator=lambda _root, _paths: None,
        )


def test_guarded_catalog_rejects_linked_uv_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    approval = _approval(project)
    external = tmp_path / "external-lock"
    external.write_text("replacement\n", encoding="utf-8")
    (project / "uv.lock").unlink()
    (project / "uv.lock").symlink_to(external)
    monkeypatch.setattr(
        cli._official,
        "_git_snapshot",
        lambda *_args, **_kwargs: approval.repository.git,
    )
    with pytest.raises(cli.M6SyntheticCommandError, match="source_binding_failed"):
        cli._verify_approved_implementation(
            project,
            live_lookup=lambda _: approval.repository.git.commit,
            live_approval_lookup=lambda _: approval.repository.git.commit,
            source_paths_resolver=lambda _: _SOURCE_PATHS,
            module_validator=lambda _root, _paths: None,
        )


def test_loaded_evalsim_module_outside_root_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    approval = _approval(project)
    monkeypatch.setattr(
        cli._official,
        "_git_snapshot",
        lambda *_args, **_kwargs: approval.repository.git,
    )
    with pytest.raises(cli.M6SyntheticCommandError, match="source_binding_failed"):
        cli._verify_approved_implementation(
            project,
            live_lookup=lambda _: approval.repository.git.commit,
            live_approval_lookup=lambda _: approval.repository.git.commit,
            source_paths_resolver=lambda _: _SOURCE_PATHS,
        )


def test_capture_begins_before_argument_parsing(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    approval = _approval(project)
    provenance = cli._build_typed_provenance(approval)
    request = cli.M6SyntheticRequest(project, "capture-test")

    def noisy_parse(_argv):
        os.write(1, b"private-parse-payload")
        return request

    prepared = cli._PreparedRun(
        store=object(),  # type: ignore[arg-type]
        approval=approval,
        provenance=provenance,
        approval_provider=lambda _: approval,
        provenance_observer=lambda _: provenance,
        success_payload=b"unused",
    )
    monkeypatch.setattr(cli, "_parse_request", noisy_parse)
    monkeypatch.setattr(cli, "_prepare_run", lambda *_args, **_kwargs: prepared)

    assert cli.main([]) == 1
    captured = capfd.readouterr()
    assert captured.out == ""
    assert "private-parse-payload" not in captured.err
    assert json.loads(captured.err) == {
        "reason_code": "terminal_output_detected",
        "schema_version": cli.M6_SYNTHETIC_STATUS_SCHEMA_VERSION,
        "status": "rejected",
    }


def test_forged_approval_stops_before_provenance_store_or_outcome(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    events: list[str] = []

    def forged_approval(_root):
        events.append("approval")
        return object()

    def forbidden_provenance(_approval):
        events.append("provenance")
        raise AssertionError("provenance must not run")

    def forbidden_runner():
        events.append("outcome")
        raise AssertionError("outcome must not run")

    with pytest.raises(
        cli.M6SyntheticCommandError,
        match="approved_commit_mismatch",
    ):
        cli._prepare_run(
            cli.M6SyntheticRequest(project, "invalid-approval"),
            cli._RunHolder(),
            approval_provider=forged_approval,  # type: ignore[arg-type]
            provenance_observer=forbidden_provenance,  # type: ignore[arg-type]
            synthetic_runner=forbidden_runner,  # type: ignore[arg-type]
        )
    assert events == ["approval"]
    assert not (project / "outputs").exists()


def test_real_command_uses_full_production_source_preflight(
    tmp_path: Path,
) -> None:
    project = _production_project(tmp_path)
    script = f"""
import subprocess
from pathlib import Path
from evalsim.cli import m6_synthetic as cli

root = Path({os.fspath(project)!r})
head = subprocess.run(
    ("git", "rev-parse", "HEAD"),
    cwd=root,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
cli._official._live_main = lambda _root: head
cli._official._live_approved_commit = lambda _root: head
raise SystemExit(
    cli.main(
        [
            "--project-root",
            str(root),
            "--run-name",
            "accepted-synthetic",
        ]
    )
)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.fspath(project)
    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == {
        "profile": cli.M6_SYNTHETIC_PROFILE,
        "result_path": "outputs/m6/accepted-synthetic",
        "schema_version": cli.M6_SYNTHETIC_STATUS_SCHEMA_VERSION,
        "status": "success",
    }

    verified = results.verify_m6_result_store(
        project,
        "accepted-synthetic",
        allow_data_free=True,
        expected_mode=results.DATA_FREE_MODE,
    )
    assert verified.receipt.eligible_cohort_indices == tuple(range(10))
    assert verified.receipt.secondary_b4_cohort_indices == tuple(range(10))
    assert verified.read_dataset(results.PRIMARY_SCENE_SCALARS).to_pylist() == (
        verified.read_dataset(results.PRIMARY_REPEAT_SCENE_SCALARS).to_pylist()
    )
    provenance = verified.read_dataset(results.TYPED_PROVENANCE).to_pylist()[0]
    assert "evalsim/cli/m6_synthetic.py" in provenance[
        "executable_source_paths"
    ]
    assert "tests/test_m6_synthetic_cli.py" in provenance[
        "executable_source_paths"
    ]
    assert provenance["approved_git_commit"] == _run_git(
        project,
        "rev-parse",
        "HEAD",
    )
    assert (verified.run_path / results.TERMINAL_SUCCESS_MARKER).is_file()
    assert not (project / "data").exists()


def test_data_free_store_records_no_independent_result_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    prepared, approval, _provenance = _complete_prepared_store(
        project,
        "no-independent-result-review",
        monkeypatch,
    )
    assert not hasattr(approval, "decisions")
    prepared.store.commit()
    verified = results.verify_committed_m6_result_store(
        project,
        prepared.store.run_name,
        allow_data_free=True,
        expected_mode=results.DATA_FREE_MODE,
    )
    reviews = verified.read_dataset(results.REVIEW_DECISIONS).to_pylist()
    execution = verified.read_dataset(results.EXECUTION_SUMMARY).to_pylist()

    assert reviews == []
    assert execution[0]["review_decision_rows"] == 0
    assert execution[0]["release_gate_status"] == "nonpromotable"
    assert not (verified.run_path / results.REVIEW_REQUEST_PATH).exists()
    cli._verify_committed_semantics(prepared, verified)


def test_source_identity_drift_blocks_commit(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    approval = _approval(project)
    provenance = cli._build_typed_provenance(approval)

    class Store:
        project_root = project
        committed = False

        def commit(self):
            self.committed = True
            raise AssertionError("source drift must block commit")

    changed_project = _project(tmp_path / "changed")
    changed_approval = _approval(changed_project)
    prepared = cli._PreparedRun(
        store=Store(),  # type: ignore[arg-type]
        approval=approval,
        provenance=provenance,
        approval_provider=lambda _: changed_approval,
        provenance_observer=cli._build_typed_provenance,
        success_payload=b"unused",
    )
    with pytest.raises(cli.M6SyntheticCommandError, match="source_binding_failed"):
        cli._commit_verify_and_mark_success(prepared)
    assert prepared.store.committed is False


def test_real_store_post_verification_failure_never_creates_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    prepared, _approval_value, _provenance = _complete_prepared_store(
        project,
        "post-verify-failure",
        monkeypatch,
    )

    def fail_semantics(_prepared, _verified):
        raise cli.M6SyntheticCommandError(
            "finalization_failed",
            "injected post-COMMITTED semantic failure",
        )

    monkeypatch.setattr(cli, "_verify_committed_semantics", fail_semantics)
    with pytest.raises(cli.M6SyntheticCommandError) as raised:
        cli._commit_verify_and_mark_success(prepared)
    relative, code = cli._persist_failure(
        prepared.store,
        raised.value,
        b"private post-verification transcript",
    )

    assert code == "finalization_failed"
    assert relative == Path(
        "outputs/m6/post-verify-failure/TERMINAL_FAILURE"
    )
    assert (prepared.store.run_path / results.COMMITTED_MARKER).is_file()
    assert (
        prepared.store.run_path / results.TERMINAL_FAILURE_MARKER
    ).is_file()
    assert not (
        prepared.store.run_path / results.TERMINAL_SUCCESS_MARKER
    ).exists()
    diagnostic = prepared.store.run_path / "failure-details.log"
    metadata = diagnostic.stat()
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_size <= cli._MAX_DIAGNOSTIC_BYTES


def test_second_post_semantic_recheck_drift_never_creates_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    prepared, approval, provenance = _complete_prepared_store(
        project,
        "second-recheck-drift",
        monkeypatch,
    )
    changed_project = _project(tmp_path / "changed")
    changed_approval = _approval(changed_project)
    events: list[str] = []
    provider_calls = 0

    def provider(_root):
        nonlocal provider_calls
        provider_calls += 1
        events.append(f"recheck-{provider_calls}")
        return approval if provider_calls == 1 else changed_approval

    original_semantics = cli._verify_committed_semantics

    def semantics(actual_prepared, verified):
        events.append("semantics")
        return original_semantics(actual_prepared, verified)

    prepared = cli._PreparedRun(
        store=prepared.store,
        approval=approval,
        provenance=provenance,
        approval_provider=provider,
        provenance_observer=lambda _: provenance,
        success_payload=b"unused",
    )
    monkeypatch.setattr(cli, "_verify_committed_semantics", semantics)
    with pytest.raises(
        cli.M6SyntheticCommandError,
        match="source_binding_failed",
    ) as raised:
        cli._commit_verify_and_mark_success(prepared)
    relative, code = cli._persist_failure(
        prepared.store,
        raised.value,
        b"second recheck drift",
    )

    assert events == ["recheck-1", "semantics", "recheck-2"]
    assert code == "source_binding_failed"
    assert relative == Path(
        "outputs/m6/second-recheck-drift/TERMINAL_FAILURE"
    )
    assert (prepared.store.run_path / results.COMMITTED_MARKER).is_file()
    assert (
        prepared.store.run_path / results.TERMINAL_FAILURE_MARKER
    ).is_file()
    assert not (
        prepared.store.run_path / results.TERMINAL_SUCCESS_MARKER
    ).exists()


def test_runner_failure_is_terminal_failure_without_public_disclosure(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)

    def failed_runner():
        os.write(2, b"private-evaluation-diagnostic")
        raise RuntimeError("private runner failure")

    monkeypatch.setattr(
        cli,
        "_verify_approved_implementation",
        lambda root: _approval(root),
    )
    monkeypatch.setattr(cli, "run_m6_synthetic_acceptance", failed_runner)
    assert (
        cli.main(
            [
                "--project-root",
                os.fspath(project),
                "--run-name",
                "failed-synthetic",
            ]
        )
        == 1
    )
    captured = capfd.readouterr()
    assert captured.out == ""
    assert "private" not in captured.err
    status_payload = json.loads(captured.err)
    assert status_payload == {
        "failure_record": "outputs/m6/failed-synthetic/TERMINAL_FAILURE",
        "reason_code": "evaluation_failed",
        "schema_version": cli.M6_SYNTHETIC_STATUS_SCHEMA_VERSION,
        "status": "failure",
    }
    run_path = project / "outputs/m6/failed-synthetic"
    assert (run_path / results.TERMINAL_FAILURE_MARKER).is_file()
    assert not (run_path / results.TERMINAL_SUCCESS_MARKER).exists()
    diagnostic = run_path / "failure-details.log"
    assert stat.S_IMODE(diagnostic.stat().st_mode) == 0o600
    assert b"private-evaluation-diagnostic" in diagnostic.read_bytes()
    assert not (project / "data").exists()


def test_preflight_rejection_has_no_failure_record_or_private_path(
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    private = tmp_path / "private" / "interview" / "candidate"
    assert (
        cli.main(
            [
                "--project-root",
                os.fspath(private),
                "--run-name",
                "never-created",
            ]
        )
        == 1
    )
    captured = capfd.readouterr()
    assert captured.out == ""
    assert os.fspath(private) not in captured.err
    payload = json.loads(captured.err)
    assert payload["status"] == "rejected"
    assert "failure_record" not in payload
    assert not (tmp_path / "outputs").exists()


def test_emit_status_reports_descriptor_write_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = cli.TerminalStatus(os.dup(1), os.dup(2))

    def failed_write(_descriptor: int, _payload: bytes) -> None:
        raise OSError("status write failed")

    monkeypatch.setattr(cli, "write_all", failed_write)
    assert cli._emit_status(terminal, b"status\n", error=False) is False


def test_main_cannot_return_success_after_status_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    approval = _approval(project)
    provenance = cli._build_typed_provenance(approval)
    prepared = cli._PreparedRun(
        store=object(),  # type: ignore[arg-type]
        approval=approval,
        provenance=provenance,
        approval_provider=lambda _: approval,
        provenance_observer=lambda _: provenance,
        success_payload=b"status\n",
    )

    class Captured:
        value = prepared
        terminal_status = object()

    monkeypatch.setattr(
        cli,
        "capture_terminal",
        lambda *_args, **_kwargs: Captured(),
    )
    monkeypatch.setattr(cli, "_emit_status", lambda *_args, **_kwargs: False)
    assert cli.main([]) == 1
