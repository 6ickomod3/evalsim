"""Data-free tests for the isolated official M5 command boundary."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

import evalsim.evaluation.m5 as evaluation_module
import evalsim.evaluation.m5_waymax as waymax_evaluation_module
import evalsim.sources.m5_m4_reuse as reuse_module
from evalsim.cli import m5_official as cli
from evalsim.results import OFFICIAL_M5_ROW_COUNTS, PreparedM5Finalization


def _argv(root: Path, run_name: str = "official-test") -> list[str]:
    return [
        "--project-root",
        os.fspath(root),
        "--data-dir",
        os.fspath(
            root / "data/raw/womd/v1.3.1/tf_example/validation"
        ),
        "--m4-run-dir",
        os.fspath(root / "outputs/m4/accepted"),
        "--run-name",
        run_name,
    ]


def _args(root: Path, run_name: str = "official-test") -> argparse.Namespace:
    return argparse.Namespace(
        project_root=root,
        data_dir=root / "data/raw/womd/v1.3.1/tf_example/validation",
        m4_run_dir=root / "outputs/m4/accepted",
        run_name=run_name,
    )


def _elapsed(value: int = 1) -> dict[str, int]:
    return {name: value for name in cli.M5_OFFICIAL_STAGE_NAMES}


def _git_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        "[project]\nname='official-test'\nversion='0.0.0'\n",
        encoding="utf-8",
    )
    (project / ".gitignore").write_text(
        "data/\noutputs/\n",
        encoding="utf-8",
    )
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
            "official CLI fixture",
        ),
        cwd=project,
        check=True,
        capture_output=True,
    )
    return project


def test_main_success_is_one_exact_sanitized_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = cli._RunResult(
        result_relative=Path("outputs/m5/official-test"),
        row_counts=OFFICIAL_M5_ROW_COUNTS.to_dict(),
        elapsed_ms=_elapsed(7),
    )
    monkeypatch.setattr(cli, "run_official", lambda _: result)

    assert cli.main(_argv(tmp_path)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "elapsed_ms": _elapsed(7),
        "profile": "official_m5",
        "result_path": "outputs/m5/official-test",
        "row_counts": {
            "metric-results": 6656,
            "scorecards": 312,
            "slice-membership": 1024,
            "waymax-parity-summary": 144,
        },
        "schema_version": "m5-cli-status-1.0.0",
        "status": "success",
    }
    assert captured.out == cli._canonical_json_bytes(
        json.loads(captured.out)
    ).decode("ascii")


@pytest.mark.parametrize(
    "argv",
    (
        [],
        ["--project-root", "/private/sentinel"],
        [
            "--project-root",
            "/private/sentinel",
            "--data-dir",
            "/private/data",
            "--m4-run-dir",
            "/private/m4",
            "--run-name",
            "ok",
            "--unexpected",
            "private-value",
        ],
    ),
)
def test_parser_requires_exact_four_arguments_without_echo(
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(argv)
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "reason_code": "argument_error",
        "schema_version": "m5-cli-status-1.0.0",
        "status": "rejected",
    }
    assert "/private/" not in captured.err


def test_help_imports_no_optional_native_runtime() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"jax", "tensorflow", "waymax"}:
            raise AssertionError(f"optional import attempted: {fullname}")
        return None

sys.meta_path.insert(0, BlockOptional())
from evalsim.cli.m5_official import main
raise SystemExit(main(["--help"]))
"""
    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "evalsim-m5-official" in completed.stdout
    assert completed.stderr == ""


def test_opt_in_rejection_occurs_before_git_m4_or_shard_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("EVALSIM_RUN_WAYMO_LOCAL", raising=False)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("preflight crossed the opt-in boundary")

    monkeypatch.setattr(cli, "_validated_root", forbidden)
    monkeypatch.setattr(cli, "_resolve_shard_paths", forbidden)
    with pytest.raises(
        cli.M5OfficialCommandError,
        match="environment_not_enabled",
    ):
        cli._preflight(_args(tmp_path))


def test_git_or_output_rejection_occurs_before_m4_or_shard_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("EVALSIM_RUN_WAYMO_LOCAL", "1")
    monkeypatch.setattr(cli, "_validated_root", lambda _: tmp_path)
    monkeypatch.setattr(cli, "_assert_running_checkout", lambda _: None)
    monkeypatch.setattr(
        cli,
        "_git_binding",
        lambda _: (_ for _ in ()).throw(
            cli.M5OfficialCommandError(
                "dirty_worktree",
                "test rejection",
            )
        ),
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("data or M4 preflight ran after Git rejection")

    monkeypatch.setattr(cli, "_resolve_shard_paths", forbidden)
    monkeypatch.setattr(reuse_module, "verify_accepted_m4_run", forbidden)
    with pytest.raises(cli.M5OfficialCommandError, match="dirty_worktree"):
        cli._preflight(_args(tmp_path))


def test_output_boundary_is_ignored_new_and_never_overwritten(
    tmp_path: Path,
) -> None:
    project = _git_project(tmp_path)
    cli._check_output_boundary(
        project,
        "official-test",
        expect_exists=False,
    )
    target = project / "outputs/m5/official-test"
    target.mkdir(parents=True)
    with pytest.raises(cli.M5OfficialCommandError, match="output_exists"):
        cli._check_output_boundary(
            project,
            "official-test",
            expect_exists=False,
        )
    cli._check_output_boundary(
        project,
        "official-test",
        expect_exists=True,
    )


def _mock_git_commands(
    monkeypatch: pytest.MonkeyPatch,
    *,
    remote: str = cli._CANONICAL_REMOTE,
    dirty: bool = False,
    branch: str = "main",
    upstream: str = "origin/main",
    upstream_commit: str = "a" * 40,
    live_commit: str = "a" * 40,
) -> None:
    def git_text(root: Path, *arguments: str) -> str:
        del root
        key = tuple(arguments)
        values = {
            ("remote", "get-url", "origin"): remote,
            ("rev-parse", "--verify", "HEAD^{commit}"): "a" * 40,
            ("rev-parse", "--verify", "HEAD^{tree}"): "b" * 40,
            ("branch", "--show-current"): branch,
            (
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{upstream}",
            ): upstream,
            (
                "rev-parse",
                "--verify",
                "@{upstream}^{commit}",
            ): upstream_commit,
        }
        return values[key]

    monkeypatch.setattr(cli, "_git_text", git_text)
    monkeypatch.setattr(
        cli,
        "_git_bytes",
        lambda root, *args: (
            b"dirty\0"
            if dirty and args[:2] == ("status", "--porcelain=v1")
            else b""
        ),
    )
    monkeypatch.setattr(cli, "_live_main", lambda _: live_commit)
    monkeypatch.setattr(
        cli,
        "official_executable_source_paths",
        lambda _: ("source.py",),
    )
    monkeypatch.setattr(
        cli,
        "executable_source_fingerprint",
        lambda root, paths: "c" * 64,
    )


@pytest.mark.parametrize(
    ("changes", "code"),
    (
        ({"remote": "git@example.invalid:wrong/repo.git"}, "git_remote_invalid"),
        ({"dirty": True}, "dirty_worktree"),
        ({"branch": "feature"}, "unpushed_main"),
        ({"upstream": "fork/main"}, "unpushed_main"),
        ({"upstream_commit": "d" * 40}, "unpushed_main"),
        ({"live_commit": "e" * 40}, "remote_main_mismatch"),
    ),
)
def test_git_binding_rejects_remote_cleanliness_and_push_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    changes: dict[str, Any],
    code: str,
) -> None:
    _mock_git_commands(monkeypatch, **changes)
    with pytest.raises(cli.M5OfficialCommandError, match=code):
        cli._git_binding(tmp_path)


def test_git_binding_records_exact_live_source_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _mock_git_commands(monkeypatch)
    binding = cli._git_binding(tmp_path)
    assert binding == cli._GitBinding(
        commit="a" * 40,
        tree="b" * 40,
        executable_paths=("source.py",),
        executable_fingerprint="c" * 64,
    )


def test_live_main_uses_bounded_https_only_nonredirected_git(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, Any] = {}
    monkeypatch.setenv("GIT_DIR", "/private/redirected-git-dir")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "url.https://evil.invalid/.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "https://github.com/")

    def run(command: Any, **kwargs: Any) -> Any:
        observed["command"] = tuple(command)
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                ("a" * 40 + "\trefs/heads/main\n").encode("ascii")
            ),
            stderr=b"",
        )

    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli._live_main(tmp_path) == "a" * 40
    assert observed["timeout"] == 30
    assert observed["cwd"] == os.sep
    assert observed["command"][-2:] == (
        cli._CANONICAL_REMOTE,
        "refs/heads/main",
    )
    assert "http.followRedirects=false" in observed["command"]
    assert "http.sslVerify=true" in observed["command"]
    environment = observed["env"]
    assert environment["GIT_ALLOW_PROTOCOL"] == "https"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert "GIT_DIR" not in environment
    assert "evil.invalid" not in repr(environment)


@dataclass
class _FakeMetricEvaluation:
    metric_rows: tuple[dict[str, Any], ...]
    slice_rows: tuple[dict[str, Any], ...]


@dataclass
class _FakeParityRow:
    index: int

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index}


class _FakeScorecardAccumulator:
    instance: "_FakeScorecardAccumulator | None" = None

    def __init__(self) -> None:
        self.indices: list[int] = []
        type(self).instance = self

    def add_case(self, evaluated: Any) -> None:
        self.indices.append(evaluated.cohort_index)

    def finalize(self, *, expected_case_count: int) -> Any:
        assert expected_case_count == 128
        assert sorted(self.indices) == list(range(128))
        return SimpleNamespace(
            case_count=128,
            slice_rows=tuple({} for _ in range(1024)),
            scorecard_results=tuple(object() for _ in range(312)),
            metric_pass_1_sha256="a" * 64,
            metric_pass_2_sha256="a" * 64,
            statistics_pass_1_sha256="b" * 64,
            statistics_pass_2_sha256="b" * 64,
        )


class _FakeParityAccumulator:
    instance: "_FakeParityAccumulator | None" = None

    def __init__(self, selection: Any) -> None:
        self.selection = selection
        self.indices: list[int] = []
        type(self).instance = self

    def add_case(self, case: Any, executions: Any) -> None:
        assert len(executions) == 3
        self.indices.append(case.cohort_index)

    def finalize(self) -> tuple[_FakeParityRow, ...]:
        assert sorted(self.indices) == list(range(16))
        return tuple(_FakeParityRow(index) for index in range(144))


class _FakeStore:
    def __init__(
        self,
        root: Path,
        run_name: str,
        *,
        events: list[str] | None = None,
    ) -> None:
        self.project_root = root
        self.run_name = run_name
        self.run_path = root / "outputs/m5" / run_name
        self.project_relative_path = Path("outputs/m5") / run_name
        self.run_path.mkdir(parents=True)
        self.events = events
        self.parts: dict[int, int] = {}
        self.receipts: list[Any] = []
        self._rows = {
            "metric-results": 0,
            "slice-membership": 0,
            "scorecards": 0,
            "waymax-parity-summary": 0,
        }
        self.prepared = PreparedM5Finalization(
            run_path=self.run_path,
            _nonce=object(),
        )

    @property
    def row_counts(self) -> dict[str, int]:
        return dict(self._rows)

    def write_parity_order_receipt(self, receipt: Any) -> None:
        if self.events is not None:
            self.events.append("parity_receipt")
        self.receipts.append(receipt)

    def write_metric_results_part(
        self,
        rows: Any,
        *,
        part_index: int,
    ) -> None:
        values = tuple(rows)
        assert part_index not in self.parts
        self.parts[part_index] = len(values)
        self._rows["metric-results"] += len(values)

    def write_slice_membership(self, rows: Any) -> None:
        values = tuple(rows)
        self._rows["slice-membership"] = len(values)
        if self.events is not None:
            self.events.append("slice_membership")

    def write_scorecards(self, rows: Any) -> None:
        values = tuple(rows)
        self._rows["scorecards"] = len(values)
        if self.events is not None:
            self.events.append("scorecards")

    def write_waymax_parity_summary(self, rows: Any) -> None:
        values = tuple(rows)
        self._rows["waymax-parity-summary"] = len(values)
        if self.events is not None:
            self.events.append("parity_summary")

    def write_determinism_receipt(self, receipt: Any) -> None:
        if self.events is not None:
            self.events.append("determinism_receipt")
        self.receipts.append(receipt)

    def write_human_readable_scorecard(self) -> None:
        if self.events is not None:
            self.events.append("scorecard_report")
        return None

    def prepare_finalization(self, *, provenance: Any) -> Any:
        assert provenance == "provenance"
        (self.run_path / "FINALIZING").write_bytes(b"FINALIZING\n")
        (self.run_path / "evaluation-manifest.json").write_bytes(b"{}\n")
        if self.events is not None:
            self.events.append("prepare")
        return self.prepared

    def mark_committed_for_verification(self, prepared: Any) -> None:
        assert prepared is self.prepared
        (self.run_path / "COMMITTED").write_bytes(b"COMMITTED\n")
        if self.events is not None:
            self.events.append("committed")

    def commit_finalization(self, prepared: Any) -> Path:
        assert prepared is self.prepared
        assert (self.run_path / "COMMITTED").read_bytes() == b"COMMITTED\n"
        (self.run_path / "SUCCESS").write_bytes(b"SUCCESS\n")
        if self.events is not None:
            self.events.append("success")
        return self.run_path

    def _write_failure(self, reason: str) -> Path:
        path = self.run_path / "FAILURE.json"
        path.write_bytes(
            cli._canonical_json_bytes(
                {
                    "complete": False,
                    "reason_code": reason,
                    "schema_version": cli.M5_RESULT_STORE_SCHEMA_VERSION,
                }
            )
        )
        if self.events is not None:
            self.events.append("failure")
        return path

    def abort_finalization(self, prepared: Any, reason: str) -> Path:
        assert prepared is self.prepared
        return self._write_failure(reason)

    def fail(self, reason: str) -> Path:
        return self._write_failure(reason)


def _install_full_official_mocks(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    fail_at: str | None = None,
) -> SimpleNamespace:
    """Install only data/network/native seams for a public ``main`` run."""

    events: list[str] = []
    visited: list[int] = []
    binding = cli._GitBinding(
        commit="a" * 40,
        tree="b" * 40,
        executable_paths=("source.py",),
        executable_fingerprint="c" * 64,
    )
    shard_paths = tuple(
        root
        / "data/raw/womd/v1.3.1/tf_example/validation"
        / f"validation-{index:05d}"
        for index in range(10)
    )
    selection = SimpleNamespace(
        receipt=SimpleNamespace(),
        parity_index_by_cohort_index={
            index: index for index in range(16)
        },
    )
    cohort = SimpleNamespace(members=tuple(range(128)))
    state = SimpleNamespace(store=None)

    def event(name: str) -> None:
        events.append(name)
        if fail_at == name:
            code = {
                "root": "project_root_invalid",
                "checkout": "source_binding_failed",
                "git": "dirty_worktree",
                "live": "remote_main_mismatch",
                "output_new": "output_not_ignored",
                "shards": "shard_set_invalid",
            }.get(name)
            if code is not None:
                raise cli.M5OfficialCommandError(
                    code,
                    f"injected {name} boundary failure",
                )
            raise RuntimeError(f"private-{name}-sentinel")

    monkeypatch.setenv("EVALSIM_RUN_WAYMO_LOCAL", "1")

    def validated_root(candidate: Path) -> Path:
        assert candidate == root
        event("root")
        return root

    def checkout(actual_root: Path) -> None:
        assert actual_root == root
        event("checkout")

    git_values = {
        ("remote", "get-url", "origin"): cli._CANONICAL_REMOTE,
        ("rev-parse", "--verify", "HEAD^{commit}"): binding.commit,
        ("rev-parse", "--verify", "HEAD^{tree}"): binding.tree,
        ("branch", "--show-current"): "main",
        (
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ): "origin/main",
        (
            "rev-parse",
            "--verify",
            "@{upstream}^{commit}",
        ): binding.commit,
    }

    def git_text(actual_root: Path, *arguments: str) -> str:
        assert actual_root == root
        if arguments == ("remote", "get-url", "origin"):
            event("git")
        return git_values[tuple(arguments)]

    def git_bytes(actual_root: Path, *arguments: str) -> bytes:
        assert actual_root == root
        assert arguments[:2] == ("status", "--porcelain=v1")
        return b""

    def live_main(actual_root: Path) -> str:
        assert actual_root == root
        event("live")
        return binding.commit

    def output_boundary(
        actual_root: Path,
        run_name: str,
        *,
        expect_exists: bool,
    ) -> None:
        assert actual_root == root
        assert run_name == "official-test"
        target = root / "outputs/m5/official-test"
        assert target.exists() is expect_exists
        if not expect_exists:
            event("output_new")
            return
        existing_count = events.count("output_existing") + 1
        events.append("output_existing")
        if fail_at == "final_context" and existing_count == 3:
            events.append("final_context")
            raise cli.M5OfficialCommandError(
                "output_drift",
                "injected final output-boundary drift",
            )

    def resolve_shards(
        actual_root: Path,
        data_dir: Path,
    ) -> tuple[Path, ...]:
        assert actual_root == root
        assert data_dir == shard_paths[0].parent
        event("shards")
        return shard_paths

    def verify_m4(actual_root: Path, m4_run_dir: Path) -> Any:
        assert actual_root == root
        assert m4_run_dir == root / "outputs/m4/accepted"
        event("m4")
        return cohort

    def reverify_m4(actual_cohort: Any) -> None:
        assert actual_cohort is cohort
        event("m4_reverify")

    def select(members: Any) -> Any:
        assert members is cohort.members
        event("parity_select")
        return selection

    def provenance(
        actual_binding: Any,
        actual_cohort: Any,
        actual_selection: Any,
    ) -> str:
        assert actual_binding == binding
        assert actual_cohort is cohort
        assert actual_selection is selection
        event("provenance")
        return "provenance"

    class Store(_FakeStore):
        def write_parity_order_receipt(self, receipt: Any) -> None:
            event("parity_receipt")
            super().write_parity_order_receipt(receipt)

        def write_metric_results_part(
            self,
            rows: Any,
            *,
            part_index: int,
        ) -> None:
            if part_index == 127:
                event("metric_part")
            super().write_metric_results_part(
                rows,
                part_index=part_index,
            )

        def write_slice_membership(self, rows: Any) -> None:
            event("aggregates")
            super().write_slice_membership(rows)

        def prepare_finalization(self, *, provenance: Any) -> Any:
            event("prepare")
            return super().prepare_finalization(
                provenance=provenance,
            )

        def mark_committed_for_verification(
            self,
            prepared: Any,
        ) -> None:
            event("committed")
            super().mark_committed_for_verification(prepared)

        def commit_finalization(self, prepared: Any) -> Path:
            event("success")
            return super().commit_finalization(prepared)

    def create(actual_root: Path, run_name: str) -> Store:
        assert actual_root == root
        assert run_name == "official-test"
        event("create")
        store = Store(root, run_name)
        state.store = store
        return store

    class ScorecardAccumulator(_FakeScorecardAccumulator):
        def finalize(self, *, expected_case_count: int) -> Any:
            event("statistics")
            return super().finalize(
                expected_case_count=expected_case_count
            )

    class ParityAccumulator(_FakeParityAccumulator):
        pass

    class FakeCase:
        def __init__(
            self,
            *,
            cohort_index: int,
            scenario: Any,
            reference_payload: Any,
        ) -> None:
            del scenario, reference_payload
            self.cohort_index = cohort_index

    def visit(accepted: Any, data_dir: Path, visitor: Any) -> None:
        assert accepted is cohort
        assert data_dir == shard_paths[0].parent
        for index in reversed(range(128)):
            visited.append(index)
            visitor(
                SimpleNamespace(
                    cohort_index=index,
                    scenario=object(),
                    record=object(),
                )
            )

    def evaluate(case: Any, executions: Any) -> Any:
        assert len(executions) == 4
        result = _FakeMetricEvaluation(
            metric_rows=tuple({} for _ in range(52)),
            slice_rows=tuple({} for _ in range(8)),
        )
        result.cohort_index = case.cohort_index  # type: ignore[attr-defined]
        return result

    def prepared_verify(actual_root: Path, run_name: str) -> Any:
        assert actual_root == root
        assert run_name == "official-test"
        event("prepared_verify")
        store = state.store
        assert isinstance(store, Store)
        assert (store.run_path / "FINALIZING").read_bytes() == b"FINALIZING\n"
        assert not (store.run_path / "COMMITTED").exists()
        assert not (store.run_path / "SUCCESS").exists()
        return SimpleNamespace(run_path=store.run_path)

    def committed_verify(actual_root: Path, run_name: str) -> Any:
        assert actual_root == root
        assert run_name == "official-test"
        event("committed_verify")
        store = state.store
        assert isinstance(store, Store)
        assert store.row_counts == OFFICIAL_M5_ROW_COUNTS.to_dict()
        assert (store.run_path / "COMMITTED").read_bytes() == b"COMMITTED\n"
        assert not (store.run_path / "SUCCESS").exists()
        return SimpleNamespace(run_path=store.run_path)

    monkeypatch.setattr(cli, "_validated_root", validated_root)
    monkeypatch.setattr(cli, "_assert_running_checkout", checkout)
    monkeypatch.setattr(cli, "_git_text", git_text)
    monkeypatch.setattr(cli, "_git_bytes", git_bytes)
    monkeypatch.setattr(cli, "_live_main", live_main)
    monkeypatch.setattr(
        cli,
        "official_executable_source_paths",
        lambda actual_root: binding.executable_paths,
    )
    monkeypatch.setattr(
        cli,
        "executable_source_fingerprint",
        lambda actual_root, paths: binding.executable_fingerprint,
    )
    monkeypatch.setattr(cli, "_check_output_boundary", output_boundary)
    monkeypatch.setattr(cli, "_resolve_shard_paths", resolve_shards)
    monkeypatch.setattr(reuse_module, "verify_accepted_m4_run", verify_m4)
    monkeypatch.setattr(reuse_module, "reverify_accepted_m4_run", reverify_m4)
    monkeypatch.setattr(
        waymax_evaluation_module,
        "select_m5_parity_members",
        select,
    )
    monkeypatch.setattr(cli, "_build_provenance", provenance)
    monkeypatch.setattr(cli.M5ResultStore, "create", create)
    monkeypatch.setattr(reuse_module, "visit_accepted_m4_cohort", visit)
    monkeypatch.setattr(evaluation_module, "EvaluationCase", FakeCase)
    monkeypatch.setattr(
        evaluation_module,
        "M5ScorecardAccumulator",
        ScorecardAccumulator,
    )
    monkeypatch.setattr(
        evaluation_module,
        "NumpyPolicyExecutor",
        lambda: object(),
    )
    monkeypatch.setattr(
        evaluation_module,
        "canonical_m5_policies",
        lambda: (object(), object(), object()),
    )
    monkeypatch.setattr(evaluation_module, "evaluate_m5_case", evaluate)
    monkeypatch.setattr(
        waymax_evaluation_module,
        "M5StreamingParityAccumulator",
        ParityAccumulator,
    )
    monkeypatch.setattr(
        waymax_evaluation_module,
        "WaymaxExactLogReferenceExecutor",
        lambda: SimpleNamespace(
            execute=lambda case, seed: object(),
        ),
    )
    monkeypatch.setattr(
        cli,
        "_policy_executions",
        lambda case, policies, executor: (object(), object(), object()),
    )
    monkeypatch.setattr(cli, "scorecard_row_from_result", lambda _: {})
    monkeypatch.setattr(
        cli,
        "verify_prepared_m5_result_store",
        prepared_verify,
    )
    monkeypatch.setattr(
        cli,
        "verify_committed_m5_result_store",
        committed_verify,
    )
    return SimpleNamespace(
        events=events,
        visited=visited,
        state=state,
    )


def test_fully_mocked_128_member_stream_writes_exact_official_domains(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selection = SimpleNamespace(
        receipt=SimpleNamespace(),
        parity_index_by_cohort_index={index: index for index in range(16)},
    )
    cohort = object()
    context = cli._OfficialContext(
        root=tmp_path,
        data_dir=tmp_path / "data",
        shard_paths=tuple(tmp_path / f"shard-{index}" for index in range(10)),
        run_name="official-test",
        cohort=cohort,
        parity_selection=selection,
        provenance="provenance",  # type: ignore[arg-type]
        git_binding=cli._GitBinding(
            commit="a" * 40,
            tree="b" * 40,
            executable_paths=("source.py",),
            executable_fingerprint="c" * 64,
        ),
        preflight_ms=3,
    )
    store = _FakeStore(tmp_path, "official-test")

    class FakeCase:
        def __init__(
            self,
            *,
            cohort_index: int,
            scenario: Any,
            reference_payload: Any,
        ) -> None:
            del scenario, reference_payload
            self.cohort_index = cohort_index

    def visit(accepted: Any, data_dir: Path, visitor: Any) -> None:
        assert accepted is cohort
        assert data_dir == context.data_dir
        # The real accepted-M4 visitor is shard-grouped rather than cohort-ranked.
        # Exercise the runner's explicit part indices and order-independent compact
        # accumulators instead of accidentally relying on 0..127 visitation.
        for index in reversed(range(128)):
            visitor(
                SimpleNamespace(
                    cohort_index=index,
                    scenario=object(),
                    record=object(),
                )
            )

    def evaluate(case: Any, executions: Any) -> Any:
        assert len(executions) == 4
        result = _FakeMetricEvaluation(
            metric_rows=tuple({} for _ in range(52)),
            slice_rows=tuple({} for _ in range(8)),
        )
        result.cohort_index = case.cohort_index  # type: ignore[attr-defined]
        return result

    monkeypatch.setattr(reuse_module, "visit_accepted_m4_cohort", visit)
    monkeypatch.setattr(evaluation_module, "EvaluationCase", FakeCase)
    monkeypatch.setattr(
        evaluation_module,
        "M5ScorecardAccumulator",
        _FakeScorecardAccumulator,
    )
    monkeypatch.setattr(
        evaluation_module,
        "NumpyPolicyExecutor",
        lambda: object(),
    )
    monkeypatch.setattr(
        evaluation_module,
        "canonical_m5_policies",
        lambda: (object(), object(), object()),
    )
    monkeypatch.setattr(evaluation_module, "evaluate_m5_case", evaluate)
    monkeypatch.setattr(
        waymax_evaluation_module,
        "M5StreamingParityAccumulator",
        _FakeParityAccumulator,
    )
    monkeypatch.setattr(
        waymax_evaluation_module,
        "WaymaxExactLogReferenceExecutor",
        lambda: SimpleNamespace(
            execute=lambda case, seed: object(),
        ),
    )
    monkeypatch.setattr(
        cli,
        "_policy_executions",
        lambda case, policies, executor: (object(), object(), object()),
    )
    monkeypatch.setattr(cli, "scorecard_row_from_result", lambda _: {})
    monkeypatch.setattr(cli, "_reverify_context", lambda context, store: None)
    monkeypatch.setattr(
        cli,
        "verify_prepared_m5_result_store",
        lambda root, run_name: SimpleNamespace(run_path=store.run_path),
    )

    holder = cli._RunHolder(store=store)  # type: ignore[arg-type]
    pending = cli._execute_pending(
        context,
        store,  # type: ignore[arg-type]
        holder,
    )
    assert pending.prepared is store.prepared
    assert set(store.parts) == set(range(128))
    assert set(store.parts.values()) == {52}
    assert sum(store.parts.values()) == 6656
    assert store.row_counts == OFFICIAL_M5_ROW_COUNTS.to_dict()
    assert len(store.receipts) == 2
    assert _FakeScorecardAccumulator.instance is not None
    assert len(_FakeScorecardAccumulator.instance.indices) == 128
    assert _FakeParityAccumulator.instance is not None
    assert sorted(_FakeParityAccumulator.instance.indices) == list(range(16))
    assert pending.result.success_payload is not None
    assert json.loads(pending.result.success_payload)["row_counts"] == (
        OFFICIAL_M5_ROW_COUNTS.to_dict()
    )


def test_main_runs_complete_mocked_128_member_official_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    harness = _install_full_official_mocks(monkeypatch, tmp_path)

    assert cli.main(_argv(tmp_path)) == 0
    captured = capfd.readouterr()
    assert captured.err == ""
    status = json.loads(captured.out)
    assert status["status"] == "success"
    assert status["result_path"] == "outputs/m5/official-test"
    assert status["row_counts"] == OFFICIAL_M5_ROW_COUNTS.to_dict()

    store = harness.state.store
    assert isinstance(store, _FakeStore)
    assert harness.visited == list(reversed(range(128)))
    assert set(store.parts) == set(range(128))
    assert set(store.parts.values()) == {52}
    assert store.row_counts == OFFICIAL_M5_ROW_COUNTS.to_dict()
    assert (store.run_path / "SUCCESS").read_bytes() == b"SUCCESS\n"
    assert not (store.run_path / "FAILURE.json").exists()

    events = harness.events
    first = {
        name: events.index(name)
        for name in (
            "root",
            "checkout",
            "git",
            "live",
            "output_new",
            "shards",
            "m4",
            "parity_select",
            "provenance",
            "create",
            "parity_receipt",
            "metric_part",
            "statistics",
            "aggregates",
            "prepare",
            "prepared_verify",
            "committed",
            "committed_verify",
            "success",
        )
    }
    assert list(first.values()) == sorted(first.values())
    assert events.count("output_existing") == 3
    assert events.count("m4_reverify") == 3
    last_output_check = len(events) - 1 - events[::-1].index(
        "output_existing"
    )
    assert events.index("committed") < last_output_check
    assert last_output_check < events.index("committed_verify")
    assert events.index("committed") < events.index("committed_verify")
    assert events.index("committed_verify") < events.index("success")


@pytest.mark.parametrize(
    (
        "fail_at",
        "reason_code",
        "status",
        "expects_store",
        "expects_committed",
    ),
    (
        ("root", "project_root_invalid", "rejected", False, False),
        ("checkout", "source_binding_failed", "rejected", False, False),
        ("git", "dirty_worktree", "rejected", False, False),
        ("live", "remote_main_mismatch", "rejected", False, False),
        ("output_new", "output_not_ignored", "rejected", False, False),
        ("shards", "shard_set_invalid", "rejected", False, False),
        ("m4", "m4_preflight_failed", "rejected", False, False),
        ("create", "preflight_failed", "rejected", False, False),
        (
            "parity_receipt",
            "cohort_evaluation_failed",
            "failure",
            True,
            False,
        ),
        (
            "metric_part",
            "cohort_evaluation_failed",
            "failure",
            True,
            False,
        ),
        ("statistics", "statistics_failed", "failure", True, False),
        (
            "aggregates",
            "artifact_publication_failed",
            "failure",
            True,
            False,
        ),
        ("prepare", "finalization_failed", "failure", True, False),
        (
            "prepared_verify",
            "verification_failed",
            "failure",
            True,
            False,
        ),
        ("committed", "finalization_failed", "failure", True, False),
        ("final_context", "output_drift", "failure", True, True),
        (
            "committed_verify",
            "verification_failed",
            "failure",
            True,
            True,
        ),
        ("success", "finalization_failed", "failure", True, True),
    ),
)
def test_public_main_failure_matrix_stops_and_persists_one_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
    fail_at: str,
    reason_code: str,
    status: str,
    expects_store: bool,
    expects_committed: bool,
) -> None:
    harness = _install_full_official_mocks(
        monkeypatch,
        tmp_path,
        fail_at=fail_at,
    )

    with pytest.raises(SystemExit) as caught:
        cli.main(_argv(tmp_path))
    assert caught.value.code == 1
    captured = capfd.readouterr()
    assert captured.out == ""
    public = json.loads(captured.err)
    assert public["status"] == status
    assert public["reason_code"] == reason_code
    assert "private-" not in captured.err
    assert harness.events[-1] == fail_at

    store = harness.state.store
    if not expects_store:
        assert store is None
        assert public.get("failure_record") is None
        assert not (tmp_path / "outputs/m5/official-test").exists()
        return

    assert isinstance(store, _FakeStore)
    assert public["failure_record"] == (
        "outputs/m5/official-test/FAILURE.json"
    )
    assert (store.run_path / "FAILURE.json").is_file()
    assert not (store.run_path / "SUCCESS").exists()
    assert (store.run_path / "COMMITTED").exists() is expects_committed


@pytest.mark.parametrize(
    ("target", "code"),
    (
        ("_visit_and_evaluate", "cohort_evaluation_failed"),
        ("_finalize_statistics", "statistics_failed"),
        ("_publish_aggregates", "artifact_publication_failed"),
        ("_reverify_context", "source_drift"),
    ),
)
def test_each_post_create_stage_fails_with_one_stable_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target: str,
    code: str,
) -> None:
    context = SimpleNamespace(preflight_ms=0, provenance="provenance")
    store = SimpleNamespace()
    holder = cli._RunHolder()
    state = SimpleNamespace()
    summary = SimpleNamespace()
    monkeypatch.setattr(cli, "_visit_and_evaluate", lambda context, store: state)
    monkeypatch.setattr(cli, "_finalize_statistics", lambda state: summary)
    monkeypatch.setattr(
        cli,
        "_publish_aggregates",
        lambda store, state, summary: None,
    )
    monkeypatch.setattr(cli, "_reverify_context", lambda context, store: None)

    if target == "_reverify_context":
        failure: BaseException = cli.M5OfficialCommandError(
            "source_drift",
            "injected source drift",
        )
    else:
        failure = RuntimeError("private-stage-sentinel")

    def injected(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise failure

    monkeypatch.setattr(cli, target, injected)
    with pytest.raises(cli.M5OfficialCommandError, match=code):
        cli._execute_pending(context, store, holder)  # type: ignore[arg-type]


def test_terminal_flood_and_private_sentinel_never_cross_status_boundary(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    sentinel = b"native-id-private-sentinel"

    def noisy(args: argparse.Namespace, holder: Any) -> Any:
        del args, holder
        for _ in range(40):
            os.write(1, sentinel + b"x" * (64 * 1024))
        raise RuntimeError("/private/local/absolute-path-sentinel")

    monkeypatch.setattr(cli, "_preflight_create_execute", noisy)
    with capfd.disabled():
        with pytest.raises(cli._CommandFailure) as caught:
            cli.run_official(_args(tmp_path))
    failure = caught.value
    try:
        assert failure.status == "rejected"
        assert failure.reason_code == "unexpected_failure"
        assert failure.terminal_status is not None
        payload = cli._failure_output(
            failure.reason_code,
            failure.failure_relative,
            status=failure.status,
        )
        assert sentinel not in payload
        assert b"/private/" not in payload
    finally:
        if failure.terminal_status is not None:
            failure.terminal_status.close_best_effort()
    captured = capfd.readouterr()
    assert "native-id-private-sentinel" not in captured.out
    assert "native-id-private-sentinel" not in captured.err


def test_run_lifecycle_is_prepare_capture_then_commit_and_verify(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    events: list[str] = []
    prepared = PreparedM5Finalization(
        run_path=tmp_path / "outputs/m5/official-test",
        _nonce=object(),
    )
    store = SimpleNamespace(
        run_path=prepared.run_path,
        project_relative_path=Path("outputs/m5/official-test"),
        commit_finalization=lambda actual: (
            events.append("commit")
            if actual is prepared
            else (_ for _ in ()).throw(AssertionError("wrong capability"))
        ),
    )
    pending = cli._PendingRun(
        prepared=prepared,
        result=cli._RunResult(
            result_relative=Path("outputs/m5/official-test"),
            row_counts=OFFICIAL_M5_ROW_COUNTS.to_dict(),
            elapsed_ms=_elapsed(),
            success_payload=b"success\n",
        ),
    )

    def prepare(args: Any, holder: Any) -> Any:
        del args
        events.append("prepare")
        holder.store = store
        holder.prepared = prepared
        holder.context = SimpleNamespace()
        return pending

    def arm(actual_store: Any, actual_pending: Any, context: Any) -> None:
        assert actual_store is store
        assert actual_pending is pending
        assert context is not None
        events.append("arm")

    monkeypatch.setattr(cli, "_preflight_create_execute", prepare)
    monkeypatch.setattr(cli, "_arm_verified_success", arm)
    with capfd.disabled():
        result = cli.run_official(_args(tmp_path))
    try:
        assert events == ["prepare", "arm", "commit"]
        assert result.success_payload == b"success\n"
        assert result.terminal_status is not None
    finally:
        if result.terminal_status is not None:
            result.terminal_status.close_best_effort()


@pytest.mark.parametrize(
    ("mode", "expected_code"),
    (
        ("verification_failure", "verification_failed"),
        ("terminal_noise", "terminal_output_detected"),
    ),
)
def test_committed_checkpoint_failure_aborts_without_success(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
    mode: str,
    expected_code: str,
) -> None:
    run_path = tmp_path / "outputs/m5/official-test"
    run_path.mkdir(parents=True)
    prepared = PreparedM5Finalization(run_path=run_path, _nonce=object())
    class Store:
        project_relative_path = Path("outputs/m5/official-test")

        def __init__(self) -> None:
            self.run_path = run_path

        def mark_committed_for_verification(self, actual: Any) -> None:
            assert actual is prepared
            (run_path / "COMMITTED").write_bytes(b"COMMITTED\n")

        def commit_finalization(self, actual: Any) -> None:
            assert actual is prepared
            (run_path / "SUCCESS").write_bytes(b"SUCCESS\n")

        def abort_finalization(self, actual: Any, reason: str) -> Path:
            assert actual is prepared
            payload = {
                "complete": False,
                "reason_code": reason,
                "schema_version": cli.M5_RESULT_STORE_SCHEMA_VERSION,
            }
            path = run_path / "FAILURE.json"
            path.write_bytes(cli._canonical_json_bytes(payload))
            return path

        def fail(self, reason: str) -> Path:
            return self.abort_finalization(prepared, reason)

    store = Store()
    pending = cli._PendingRun(
        prepared=prepared,
        result=cli._RunResult(
            result_relative=Path("outputs/m5/official-test"),
            row_counts=OFFICIAL_M5_ROW_COUNTS.to_dict(),
            elapsed_ms=_elapsed(),
            success_payload=b"success\n",
        ),
    )

    def prepare(args: Any, holder: Any) -> Any:
        del args
        holder.store = store
        holder.prepared = prepared
        holder.context = SimpleNamespace(
            root=tmp_path,
            run_name="official-test",
        )
        return pending

    sentinel = b"checkpoint-private-sentinel"

    def verify(root: Path, run_name: str) -> Any:
        assert root == tmp_path
        assert run_name == "official-test"
        if mode == "terminal_noise":
            os.write(2, sentinel)
            return SimpleNamespace(run_path=run_path)
        raise RuntimeError("injected committed verifier failure")

    monkeypatch.setattr(cli, "_preflight_create_execute", prepare)
    monkeypatch.setattr(cli, "_reverify_context", lambda *args: None)
    monkeypatch.setattr(cli, "verify_committed_m5_result_store", verify)
    with capfd.disabled():
        with pytest.raises(cli._CommandFailure) as caught:
            cli.run_official(_args(tmp_path))
    failure = caught.value
    try:
        assert failure.reason_code == expected_code
        assert failure.status == "failure"
        assert failure.failure_relative == Path(
            "outputs/m5/official-test/FAILURE.json"
        )
        assert (run_path / "COMMITTED").read_bytes() == b"COMMITTED\n"
        assert (run_path / "FAILURE.json").is_file()
        assert not (run_path / "SUCCESS").exists()
        assert sentinel not in cli._failure_output(
            failure.reason_code,
            failure.failure_relative,
            status=failure.status,
        )
    finally:
        if failure.terminal_status is not None:
            failure.terminal_status.close_best_effort()
    captured = capfd.readouterr()
    assert "checkpoint-private-sentinel" not in captured.out
    assert "checkpoint-private-sentinel" not in captured.err


def test_failure_marker_precedes_optional_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_path = tmp_path / "outputs/m5/official-test"
    run_path.mkdir(parents=True)
    prepared = PreparedM5Finalization(run_path=run_path, _nonce=object())
    events: list[str] = []

    class Store:
        project_relative_path = Path("outputs/m5/official-test")

        def __init__(self) -> None:
            self.run_path = run_path

        def abort_finalization(self, actual: Any, reason: str) -> Path:
            assert actual is prepared
            events.append("failure")
            path = run_path / "FAILURE.json"
            path.write_bytes(
                cli._canonical_json_bytes(
                    {
                        "complete": False,
                        "reason_code": reason,
                        "schema_version": cli.M5_RESULT_STORE_SCHEMA_VERSION,
                    }
                )
            )
            return path

    def diagnostics(*args: Any) -> None:
        del args
        assert (run_path / "FAILURE.json").is_file()
        events.append("diagnostics")
        raise OSError("injected optional diagnostic failure")

    monkeypatch.setattr(cli, "_write_failure_diagnostics", diagnostics)
    relative, reason, status = cli._persist_failure(
        Store(),  # type: ignore[arg-type]
        cli.M5OfficialCommandError(
            "verification_failed",
            "injected failure",
        ),
        b"private-transcript",
        prepared=prepared,
    )
    assert events == ["failure", "diagnostics"]
    assert relative == Path(
        "outputs/m5/official-test/FAILURE.json"
    )
    assert reason == "verification_failed"
    assert status == "failure"


def test_finalization_prepare_failure_has_stable_stage_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(preflight_ms=0, provenance="provenance")
    state = SimpleNamespace()
    summary = SimpleNamespace()
    store = SimpleNamespace(
        prepare_finalization=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("private-finalization-sentinel")
        )
    )
    monkeypatch.setattr(cli, "_visit_and_evaluate", lambda context, store: state)
    monkeypatch.setattr(cli, "_finalize_statistics", lambda state: summary)
    monkeypatch.setattr(
        cli,
        "_publish_aggregates",
        lambda store, state, summary: None,
    )
    monkeypatch.setattr(cli, "_reverify_context", lambda context, store: None)
    with pytest.raises(cli.M5OfficialCommandError, match="finalization_failed"):
        cli._execute_pending(
            context,
            store,  # type: ignore[arg-type]
            cli._RunHolder(),
        )


def test_reverification_rejects_source_and_output_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    binding = cli._GitBinding(
        commit="a" * 40,
        tree="b" * 40,
        executable_paths=("source.py",),
        executable_fingerprint="c" * 64,
    )
    context = SimpleNamespace(
        root=tmp_path,
        run_name="official-test",
        git_binding=binding,
        data_dir=tmp_path / "data",
        shard_paths=(),
        cohort=object(),
    )
    store = SimpleNamespace(
        run_path=tmp_path / "outputs/m5/official-test",
        project_relative_path=Path("outputs/m5/official-test"),
    )
    monkeypatch.setattr(
        cli,
        "_git_binding",
        lambda _: cli._GitBinding(
            commit="d" * 40,
            tree="b" * 40,
            executable_paths=("source.py",),
            executable_fingerprint="c" * 64,
        ),
    )
    with pytest.raises(cli.M5OfficialCommandError, match="source_drift"):
        cli._reverify_context(context, store)

    monkeypatch.setattr(cli, "_git_binding", lambda _: binding)
    monkeypatch.setattr(cli, "_assert_running_checkout", lambda _: None)
    monkeypatch.setattr(
        cli,
        "_check_output_boundary",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            cli.M5OfficialCommandError(
                "output_drift",
                "injected output drift",
            )
        ),
    )
    with pytest.raises(cli.M5OfficialCommandError, match="output_drift"):
        cli._reverify_context(context, store)


def test_exact_official_domains_are_frozen() -> None:
    assert cli.M5_OFFICIAL_SCENARIO_COUNT == 128
    assert cli.M5_OFFICIAL_METRIC_ROWS_PER_CASE == 52
    assert cli.M5_OFFICIAL_SLICE_ROWS_PER_CASE == 8
    assert OFFICIAL_M5_ROW_COUNTS.to_dict() == {
        "metric-results": 6656,
        "slice-membership": 1024,
        "scorecards": 312,
        "waymax-parity-summary": 144,
    }
    assert cli.M5_OFFICIAL_STAGE_NAMES == (
        "preflight",
        "cohort_evaluation",
        "statistics",
        "artifact_publication",
        "finalization",
    )
