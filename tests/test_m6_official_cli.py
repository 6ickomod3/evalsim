"""Data-free tests for the capture-first future M6 command boundary."""
from __future__ import annotations

import base64
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import py_compile
import shutil
import stat
import subprocess
import sys
import time
import tomllib
from types import ModuleType, SimpleNamespace

import pytest

from evalsim.cli import m6_official as cli


def _run_git(project: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_project(tmp_path: Path) -> tuple[Path, str]:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        "[project]\nname='m6-cli-test'\nversion='0.0.0'\n",
        encoding="utf-8",
    )
    (project / ".python-version").write_text(
        "3.11.5\n", encoding="ascii"
    )
    (project / ".gitignore").write_text(
        "data/\noutputs/\n.venv/\n",
        encoding="utf-8",
    )
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (project / "AGENTS.md").write_text("# fixture\n", encoding="utf-8")
    (project / "NOTICE.md").write_text("fixture\n", encoding="utf-8")
    for relative in (
        "docs/plans/2026-07-29-m6-counterfactual-reactivity.md",
        "docs/plans/2026-07-29-m6-data-free-implementation-checkpoint.md",
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    source = project / "evalsim/base.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    test = project / "tests/test_m6_base.py"
    test.parent.mkdir()
    test.write_text("def test_base(): pass\n", encoding="utf-8")
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
        "fixture",
    )
    commit = _run_git(project, "rev-parse", "HEAD")
    tag_name = cli._APPROVED_IMPLEMENTATION_REF.removeprefix("refs/tags/")
    _run_git(project, "tag", tag_name, commit)
    _run_git(project, "remote", "add", "origin", cli._CANONICAL_REMOTE)
    _run_git(project, "update-ref", "refs/remotes/origin/main", commit)
    return project, commit


def _request(project: Path, *, mode: str = "eligibility_only") -> cli.M6CommandRequest:
    eligibility_run_name = (
        "m6-eligibility" if mode in {"compute_pilot", "official"} else None
    )
    pilot_run_name = "m6-pilot" if mode == "official" else None
    return cli.M6CommandRequest(
        project_root=project,
        data_dir=project / cli._DEFAULT_DATA_RELATIVE,
        m4_run_dir=project / "outputs/m4/accepted",
        run_name="m6-test",
        mode=mode,
        eligibility_run_name=eligibility_run_name,
        pilot_run_name=pilot_run_name,
    )


def _command_result(mode: str) -> cli._CommandResult:
    return cli._CommandResult(
        mode=mode,
        result_relative=Path("outputs/m6/m6-test"),
        aggregate_counts={
            "population_n": 128,
            "primary_eligible_n": 10,
        },
        stage_durations_ms={"eligibility": 1},
    )


def _test_bootstrap_context(project: Path) -> cli._M6BootstrapContext:
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    os.set_inheritable(stdout_fd, False)
    os.set_inheritable(stderr_fd, False)

    def sink(payload: bytes, error: bool) -> bool:
        descriptor = stderr_fd if error else stdout_fd
        try:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    return False
                remaining = remaining[written:]
            return True
        finally:
            for saved in (stdout_fd, stderr_fd):
                try:
                    os.close(saved)
                except OSError:
                    pass

    failure_callbacks: list[object] = []

    def status_failure_sink(callback) -> bool:
        if failure_callbacks or not callable(callback):
            return False
        failure_callbacks.append(callback)
        return True

    return cli._M6BootstrapContext(
        project_root=project,
        site_packages=project / ".venv/lib/python3.11/site-packages",
        pycache_prefix=project / ".test-pycache-prefix",
        initial_sys_path=tuple(sys.path),
        repository_receipt=(
            "a" * 40,
            "b" * 40,
            ("uv.lock",),
            "c" * 64,
            "d" * 64,
        ),
        status_sink=sink,
        status_failure_sink=status_failure_sink,
        _factory_sentinel=cli._BOOTSTRAP_SENTINEL,
    )


def test_main_rejects_direct_entry_without_bootstrap_context(
    capfd: pytest.CaptureFixture[str],
) -> None:
    assert cli.main([]) == 1
    assert capfd.readouterr() == ("", "")


def test_module_import_attempts_no_optional_native_runtime() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = r'''
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {
            "flax",
            "jax",
            "numpy",
            "pyarrow",
            "tensorflow",
            "waymax",
        }:
            raise AssertionError(f"optional import attempted: {fullname}")
        return None

sys.meta_path.insert(0, BlockOptional())
from evalsim.cli.m6_official import M6_OFFICIAL_MODES, _parser
assert M6_OFFICIAL_MODES == ("eligibility_only", "compute_pilot", "official")
assert _parser().prog == "evalsim-m6-official"
assert "numpy" not in sys.modules
assert "pyarrow" not in sys.modules
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



def _fake_official_source(*, import_output: bytes | None = None) -> str:
    prefix = ""
    if import_output is not None:
        prefix = (
            "import os\n"
            f"os.write(1, {import_output!r})\n"
        )
    return prefix + '''import json

def _issue_m6_bootstrap_context(
    project_root,
    site_packages,
    pycache_prefix,
    initial_sys_path,
    repository_receipt,
    status_sink,
    status_failure_sink,
):
    del (
        project_root,
        site_packages,
        pycache_prefix,
        initial_sys_path,
        repository_receipt,
        status_failure_sink,
    )
    return status_sink

def main(argv, *, _bootstrap_context=None):
    del argv
    payload = (json.dumps(
        {
            "aggregate_counts": {"population_n": 1},
            "mode": "official",
            "profile": "official_m6",
            "result_path": "outputs/m6/bootstrap-fixture",
            "schema_version": "m6-cli-status-2.0.0",
            "stage_durations_ms": {"bootstrap": 1},
            "status": "success",
        },
        sort_keys=True,
        separators=(",", ":"),
    ) + "\\n").encode("ascii")
    return 0 if _bootstrap_context(payload, False) else 1
'''


def _fake_bootstrap_project(
    tmp_path: Path,
    official_source: str,
) -> tuple[Path, Path]:
    project = tmp_path / "bootstrap-project"
    cli_dir = project / "evalsim/cli"
    cli_dir.mkdir(parents=True)
    (project / "evalsim/__init__.py").write_text("", encoding="utf-8")
    (cli_dir / "__init__.py").write_text("", encoding="utf-8")
    source_bootstrap = (
        Path(__file__).resolve().parents[1]
        / "evalsim/cli/m6_bootstrap.py"
    )
    shutil.copyfile(source_bootstrap, cli_dir / "m6_bootstrap.py")
    (cli_dir / "m6_official.py").write_text(
        official_source, encoding="utf-8"
    )
    python = project / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(sys.executable).resolve())
    site_packages = (
        project
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site_packages.mkdir(parents=True)
    return project, site_packages


def _run_fake_bootstrap(
    project: Path,
    *,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    isolated_environment = (
        dict(os.environ) if environment is None else dict(environment)
    )
    isolated_environment.pop(cli._LOCAL_OPT_IN, None)
    command = (
        *cli.M6_OFFICIAL_DIRECT_COMMAND,
        "--project-root",
        os.fspath(project),
        "--data-dir",
        os.fspath(project / "data"),
        "--m4-run-dir",
        os.fspath(project / "outputs/m4/accepted"),
        "--run-name",
        "bootstrap-fixture",
        "--mode",
        "eligibility_only",
    )
    return subprocess.run(
        command,
        cwd=project,
        env=isolated_environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_legacy_sourceless_module(
    root: Path,
    module_name: str,
    source: str,
) -> Path:
    source_path = root / f"{module_name}.py"
    pyc_path = root / f"{module_name}.pyc"
    source_path.write_text(source, encoding="utf-8")
    py_compile.compile(
        os.fspath(source_path),
        cfile=os.fspath(pyc_path),
        doraise=True,
    )
    source_path.unlink()
    return pyc_path


def test_bootstrap_import_path_cannot_load_root_argparse_pyc(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "argparse-shadowed"
    project, site_packages = _fake_bootstrap_project(
        tmp_path,
        "import argparse\n" + _fake_official_source(),
    )
    _write_legacy_sourceless_module(
        project,
        "argparse",
        f"open({os.fspath(marker)!r}, 'w').write('shadowed')\n",
    )
    bootstrap_path = project / "evalsim/cli/m6_bootstrap.py"
    script = f"""
import importlib.util
from pathlib import Path
import sys

root = Path({os.fspath(project)!r})
site_packages = Path({os.fspath(site_packages)!r})
spec = importlib.util.spec_from_file_location(
    "_isolated_m6_bootstrap_test",
    {os.fspath(bootstrap_path)!r},
)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
bootstrap._parse_argument_surface = lambda root: {{}}
bootstrap._preauthenticate_repository = lambda root: (
    "a" * 40,
    "b" * 40,
    ("uv.lock",),
    "c" * 64,
    "d" * 64,
)
sys.modules.pop("argparse", None)
initial = tuple(sys.path)
return_code, payload, error, duplicate = bootstrap._run_captured(
    root,
    site_packages,
    initial,
)
assert return_code == 0
assert error is False
assert duplicate is False
assert b'"status":"success"' in payload
assert str(root) not in sys.path
"""
    completed = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", script),
        cwd=project,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert not marker.exists()


def test_site_activation_cannot_load_root_numpy_pyc(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    marker = tmp_path / "numpy-shadowed"
    _write_legacy_sourceless_module(
        project,
        "numpy",
        f"open({os.fspath(marker)!r}, 'w').write('shadowed')\n",
    )
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    (site_packages / "numpy.py").write_text(
        "SAFE_SITE_NUMPY = True\n",
        encoding="utf-8",
    )
    repository_root = Path(__file__).resolve().parents[1]
    script = f"""
from pathlib import Path
from types import SimpleNamespace
import sys

repository_root = {os.fspath(repository_root)!r}
sys.path.append(repository_root)
from evalsim.cli import m6_official as cli
sys.path.remove(repository_root)
initial = tuple(sys.path)
context = SimpleNamespace(
    project_root=Path({os.fspath(project)!r}),
    initial_sys_path=initial,
    site_packages=Path({os.fspath(site_packages)!r}),
    site_packages_enabled=False,
)
cli._require_active_bootstrap_context = lambda: context
cli._validate_bootstrap_runtime = lambda context, issuing: None
cli._enable_active_bootstrap_site_packages()
sys.modules.pop("numpy", None)
import numpy
assert numpy.SAFE_SITE_NUMPY is True
assert tuple(sys.path) == (*initial, {os.fspath(site_packages)!r})
assert {os.fspath(project)!r} not in sys.path
"""
    completed = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", script),
        cwd=project,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert not marker.exists()


def test_repository_python_pin_matches_official_runtime() -> None:
    from evalsim.cli import m6_bootstrap as bootstrap

    project = Path(__file__).resolve().parents[1]
    pin = (project / ".python-version").read_text(encoding="ascii")
    pyproject = tomllib.loads(
        (project / "pyproject.toml").read_text(encoding="utf-8")
    )
    lock = tomllib.loads(
        (project / "uv.lock").read_text(encoding="utf-8")
    )
    ignored = {
        line.strip()
        for line in (project / ".gitignore").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert pin == f"{cli._EXPECTED_PYTHON_VERSION}\n"
    assert cli._EXPECTED_PYTHON_VERSION == "3.11.5"
    assert cli._RUNTIME_SITE_PACKAGES_RELATIVE == (
        ".venv/lib/python3.11/site-packages"
    )
    assert pyproject["project"]["requires-python"] == ">=3.11"
    assert lock["requires-python"] == pyproject["project"]["requires-python"]
    assert ".python-version" not in ignored
    assert ".python-version" in cli._SOURCE_REQUIRED_FILES
    assert ".python-version" in bootstrap._SOURCE_REQUIRED_FILES


def test_exact_direct_bootstrap_command_and_console_entry_disabled() -> None:
    project = Path(__file__).resolve().parents[1]
    assert cli.M6_OFFICIAL_DIRECT_COMMAND == (
        ".venv/bin/python",
        "-I",
        "-S",
        "-B",
        "evalsim/cli/m6_bootstrap.py",
    )
    pyproject = (project / "pyproject.toml").read_text(encoding="utf-8")
    assert "evalsim-m6-official =" not in pyproject

    completed = subprocess.run(
        cli.M6_OFFICIAL_DIRECT_COMMAND,
        cwd=project,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert json.loads(completed.stderr) == {
        "reason_code": "argument_error",
        "schema_version": cli.M6_OFFICIAL_STATUS_SCHEMA_VERSION,
        "status": "rejected",
    }


def test_bootstrap_ignores_pythonpath_sitecustomize_and_pth(
    tmp_path: Path,
) -> None:
    project, site_packages = _fake_bootstrap_project(
        tmp_path, _fake_official_source()
    )
    sentinel = "site-bootstrap-sentinel"
    (site_packages / "sitecustomize.py").write_text(
        f"print({sentinel!r})\n", encoding="utf-8"
    )
    (site_packages / "pth_attack.py").write_text(
        f"print({sentinel!r})\n", encoding="utf-8"
    )
    (site_packages / "attack.pth").write_text(
        "import pth_attack\n", encoding="utf-8"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.fspath(site_packages)

    completed = _run_fake_bootstrap(project, environment=environment)

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert sentinel not in completed.stderr
    assert json.loads(completed.stderr) == {
        "reason_code": "environment_not_enabled",
        "schema_version": cli.M6_OFFICIAL_STATUS_SCHEMA_VERSION,
        "status": "rejected",
    }


def test_bootstrap_ignores_matching_timestamp_adjacent_pyc(
    tmp_path: Path,
) -> None:
    sentinel = b"timestamp-pyc-sentinel"
    safe = _fake_official_source()
    malicious = _fake_official_source(import_output=sentinel)
    width = max(len(safe), len(malicious)) + 128

    def pad(source: str) -> str:
        return source + "#" + ("x" * (width - len(source) - 1))

    malicious = pad(malicious)
    safe = pad(safe)
    assert len(malicious.encode("utf-8")) == len(safe.encode("utf-8"))
    project, _site_packages = _fake_bootstrap_project(tmp_path, malicious)
    official = project / "evalsim/cli/m6_official.py"
    timestamp = int(time.time()) - 60
    os.utime(official, (timestamp, timestamp))
    pyc = Path(
        py_compile.compile(
            os.fspath(official),
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
        )
    )
    malicious_stat = official.stat()
    official.write_text(safe, encoding="utf-8")
    os.utime(
        official,
        ns=(malicious_stat.st_atime_ns, malicious_stat.st_mtime_ns),
    )
    assert official.stat().st_size == malicious_stat.st_size

    completed = _run_fake_bootstrap(project)

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert sentinel.decode("ascii") not in completed.stderr
    assert json.loads(completed.stderr)["reason_code"] == "environment_not_enabled"
    assert pyc.is_file()
    assert tuple(pyc.parent.glob("*.pyc")) == (pyc,)


def test_bootstrap_converts_import_output_to_one_safe_rejection(
    tmp_path: Path,
) -> None:
    sentinel = b"private-import-output-sentinel"
    project, _site_packages = _fake_bootstrap_project(
        tmp_path, _fake_official_source(import_output=sentinel)
    )

    completed = _run_fake_bootstrap(project)

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert sentinel.decode("ascii") not in completed.stderr
    assert json.loads(completed.stderr) == {
        "reason_code": "environment_not_enabled",
        "schema_version": cli.M6_OFFICIAL_STATUS_SCHEMA_VERSION,
        "status": "rejected",
    }


def test_parser_exposes_exact_mode_and_path_surface(tmp_path: Path) -> None:
    project = tmp_path / "project"
    argv = [
        "--project-root",
        os.fspath(project),
        "--data-dir",
        os.fspath(project / "private-data"),
        "--m4-run-dir",
        os.fspath(project / "private-m4"),
        "--run-name",
        "run-01",
        "--mode",
        "compute_pilot",
        "--eligibility-run-name",
        "eligibility-ok",
    ]
    request = cli._parse_request(argv)
    assert request.project_root == project
    assert request.data_dir == project / "private-data"
    assert request.m4_run_dir == project / "private-m4"
    assert request.run_name == "run-01"
    assert request.mode == "compute_pilot"
    assert request.eligibility_run_name == "eligibility-ok"
    assert request.pilot_run_name is None

    bad_mode = list(argv)
    bad_mode[bad_mode.index("compute_pilot")] = "unsupported"
    with pytest.raises(cli.M6OfficialCommandError, match="argument_error"):
        cli._parse_request(bad_mode)
    with pytest.raises(cli.M6OfficialCommandError, match="argument_error"):
        cli._parse_request(argv[:-2])

    with pytest.raises(cli.M6OfficialCommandError, match="argument_error"):
        cli.M6CommandRequest(
            project_root=project,
            data_dir=project / "data",
            m4_run_dir=project / "m4",
            run_name="official-run",
            mode="official",
            eligibility_run_name="eligibility-ok",
        )


def test_finalize_review_parser_requires_exact_explicit_role_decisions(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    argv = [
        "--project-root",
        os.fspath(project),
        "--data-dir",
        os.fspath(project / "private-data"),
        "--m4-run-dir",
        os.fspath(project / "private-m4"),
        "--run-name",
        "official-run",
        "--eligibility-run-name",
        "eligibility-ok",
        "--pilot-run-name",
        "pilot-ok",
        "--mode",
        "official",
        "--action",
        "finalize-review",
    ]
    for prefix in ("architecture", "methods-statistics", "privacy-claim"):
        argv.extend(
            [
                f"--{prefix}-decision",
                "accept",
                f"--{prefix}-p1-count",
                "0",
                f"--{prefix}-p2-count",
                "0",
                f"--{prefix}-p3-count",
                "2",
            ]
        )
    parsed = cli._parse_invocation(argv)
    assert type(parsed) is cli.M6ReviewFinalizationRequest
    assert tuple(item.role for item in parsed.reviews) == (
        "architecture",
        "methods_statistics",
        "privacy_claim",
    )
    assert all(item.decision == "accept" for item in parsed.reviews)
    assert all(item.p3_count == 2 for item in parsed.reviews)

    reject = list(argv)
    reject[reject.index("accept")] = "reject"
    parsed_reject = cli._parse_invocation(reject)
    assert parsed_reject.reviews[0].decision == "reject"
    assert parsed_reject.reviews[0].p1_count == 0

    inconsistent = list(argv)
    p1_option = inconsistent.index("--architecture-p1-count")
    inconsistent[p1_option + 1] = "1"
    parsed_inconsistent = cli._parse_invocation(inconsistent)
    assert parsed_inconsistent.reviews[0].decision == "accept"
    assert parsed_inconsistent.reviews[0].p1_count == 1

    maximum = list(argv)
    p3_option = maximum.index("--architecture-p3-count")
    maximum[p3_option + 1] = str(2**31 - 1)
    assert cli._parse_invocation(maximum).reviews[0].p3_count == 2**31 - 1
    too_large = list(maximum)
    too_large[p3_option + 1] = str(2**31)
    with pytest.raises(cli.M6OfficialCommandError, match="argument_error"):
        cli._parse_invocation(too_large)
    with pytest.raises(cli.M6OfficialCommandError, match="argument_error"):
        cli._parse_invocation(argv[:-2])


def test_bootstrap_accepts_only_exact_finalize_review_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.cli import m6_bootstrap as bootstrap

    project = tmp_path / "project"
    project.mkdir()
    arguments = [
        "--project-root",
        os.fspath(project),
        "--data-dir",
        os.fspath(project / "data"),
        "--m4-run-dir",
        os.fspath(project / "m4"),
        "--run-name",
        "official-run",
        "--eligibility-run-name",
        "eligibility-run",
        "--pilot-run-name",
        "pilot-run",
        "--mode",
        "official",
        "--action",
        "finalize-review",
    ]
    for prefix in ("architecture", "methods-statistics", "privacy-claim"):
        arguments.extend(
            [
                f"--{prefix}-decision",
                "accept",
                f"--{prefix}-p1-count",
                "0",
                f"--{prefix}-p2-count",
                "0",
                f"--{prefix}-p3-count",
                "0",
            ]
        )
    monkeypatch.setattr(sys, "argv", ["m6_bootstrap.py", *arguments])
    parsed = bootstrap._parse_argument_surface(project)
    assert parsed["--action"] == "finalize-review"

    monkeypatch.setattr(sys, "argv", ["m6_bootstrap.py", *arguments[:-2]])
    with pytest.raises(bootstrap._BootstrapRejected) as missing:
        bootstrap._parse_argument_surface(project)
    assert missing.value.reason_code == "argument_error"

    inconsistent = list(arguments)
    p1 = inconsistent.index("--architecture-p1-count")
    inconsistent[p1 + 1] = "1"
    monkeypatch.setattr(sys, "argv", ["m6_bootstrap.py", *inconsistent])
    parsed_inconsistent = bootstrap._parse_argument_surface(project)
    assert parsed_inconsistent["--architecture-p1-count"] == "1"

    maximum = list(arguments)
    p3 = maximum.index("--architecture-p3-count")
    maximum[p3 + 1] = str(2**31 - 1)
    monkeypatch.setattr(sys, "argv", ["m6_bootstrap.py", *maximum])
    assert bootstrap._parse_argument_surface(project)[
        "--architecture-p3-count"
    ] == str(2**31 - 1)
    too_large = list(maximum)
    too_large[p3 + 1] = str(2**31)
    monkeypatch.setattr(sys, "argv", ["m6_bootstrap.py", *too_large])
    with pytest.raises(bootstrap._BootstrapRejected) as overflow:
        bootstrap._parse_argument_surface(project)
    assert overflow.value.reason_code == "argument_error"


def test_awaiting_review_status_is_minimal_and_bootstrap_validated() -> None:
    from evalsim.cli import m6_bootstrap as bootstrap

    result = cli._AwaitingReviewResult(
        mode="official",
        result_relative=Path("outputs/m6/official-run"),
        evidence_catalog_sha256="a" * 64,
        mechanical_verification_sha256="b" * 64,
    )
    payload = cli._awaiting_review_output(result)
    assert json.loads(payload) == {
        "mode": "official",
        "profile": "official_m6",
        "result_path": "outputs/m6/official-run",
        "schema_version": "m6-cli-status-2.0.0",
        "status": "awaiting_review",
    }
    assert bootstrap._valid_status(payload, False, 0)

    failure = bootstrap._canonical_json(
        {
            "failure_marker": (
                "outputs/m6/official-run/TERMINAL_FAILURE"
            ),
            "reason_code": "review_rejected",
            "schema_version": "m6-cli-status-2.0.0",
            "status": "failure",
        }
    )
    assert bootstrap._valid_status(failure, True, 1)
    assert not bootstrap._valid_status(failure, False, 0)


@pytest.mark.parametrize(
    ("review_case", "expected_status"),
    (
        ("accepted_max_p3", "accepted"),
        ("explicit_reject", "rejected"),
        ("accept_with_p1", "rejected"),
    ),
)
def test_finalize_review_seals_explicit_decisions_without_outcome_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    review_case: str,
    expected_status: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    command = _request(project, mode="official")
    reviews = tuple(
        cli.M6ReviewInput(
            role=role,
            decision=(
                "reject"
                if review_case == "explicit_reject" and role == "architecture"
                else "accept"
            ),
            p1_count=(
                1
                if review_case == "accept_with_p1"
                and role == "architecture"
                else 0
            ),
            p2_count=0,
            p3_count=(
                2**31 - 1 if review_case == "accepted_max_p3" else 1
            ),
        )
        for role in (
            "architecture",
            "methods_statistics",
            "privacy_claim",
        )
    )
    finalization = cli.M6ReviewFinalizationRequest(
        command=command,
        reviews=reviews,
    )
    events: list[object] = []

    class Store:
        awaiting_review_fresh_worker_peak_rss_bytes = 8192
        project_relative_path = Path("outputs/m6/m6-test")

        def write_review_decisions(self, verification, decisions):
            events.append(("reviews", verification, tuple(decisions)))

        def write_execution_summary(self, *, fresh_worker_peak_rss_bytes):
            assert fresh_worker_peak_rss_bytes == 8192
            events.append("summary")

        def _read_dataset_rows(self, name):
            assert name == "execution"
            return ({"release_gate_status": expected_status},)

        def commit(self):
            events.append("commit")

    store = Store()
    repository = _minimal_repository(project)
    local = SimpleNamespace()
    verification = object()
    issued: list[object] = []

    def issue(actual_verification, **values):
        assert actual_verification is verification
        receipt = SimpleNamespace(**values)
        issued.append(receipt)
        return receipt

    module = SimpleNamespace(
        M6ResultStore=SimpleNamespace(
            adopt_awaiting_review=lambda root, run_name: (
                store
                if root == project and run_name == "m6-test"
                else (_ for _ in ()).throw(AssertionError("wrong store"))
            )
        ),
        EXECUTION_SUMMARY="execution",
        issue_m6_review_decision=issue,
    )
    monkeypatch.setattr(
        cli,
        "preflight_repository",
        lambda request, **kwargs: repository,
    )
    monkeypatch.setattr(
        cli,
        "_load_authenticated_review_results",
        lambda *args, **kwargs: (module, None),
    )
    monkeypatch.setattr(
        cli,
        "preflight_local_inputs",
        lambda *args, **kwargs: local,
    )
    monkeypatch.setattr(
        cli,
        "_reverify_awaiting_review_precursor",
        lambda *args, **kwargs: verification,
    )
    holder = cli._RunHolder()
    if expected_status == "rejected":
        with pytest.raises(cli.M6OfficialCommandError) as rejected:
            cli.finalize_m6_review(finalization, holder)
        assert rejected.value.code == "review_rejected"
        assert "commit" not in events
    else:
        prepared = cli.finalize_m6_review(finalization, holder)
        assert prepared.store is store
        assert events[-1] == "commit"
    assert holder.store is store
    assert tuple(item.role for item in issued) == (
        "architecture",
        "methods_statistics",
        "privacy_claim",
    )
    if review_case == "accept_with_p1":
        assert issued[0].decision == "accept"
        assert issued[0].p1_count == 1
    if review_case == "accepted_max_p3":
        assert all(item.p3_count == 2**31 - 1 for item in issued)
    assert not any(item == "outcome" for item in events)


def test_main_captures_before_parse_and_never_discloses_native_output(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)

    def noisy_parse(argv):
        del argv
        os.write(1, b"private-native-parse-output")
        return request

    monkeypatch.setattr(cli, "_parse_request", noisy_parse)
    monkeypatch.setattr(
        cli,
        "_dispatch",
        lambda actual, holder: _command_result(actual.mode),
    )
    assert cli.main(
        [], _bootstrap_context=_test_bootstrap_context(tmp_path)
    ) == 1
    captured = capfd.readouterr()
    assert captured.out == ""
    assert "private-native" not in captured.err
    assert json.loads(captured.err) == {
        "reason_code": "terminal_output_detected",
        "schema_version": cli.M6_OFFICIAL_STATUS_SCHEMA_VERSION,
        "status": "rejected",
    }


def test_main_success_is_one_allowlisted_status(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, mode="official")
    monkeypatch.setattr(cli, "_parse_request", lambda _: request)
    monkeypatch.setattr(
        cli,
        "_dispatch",
        lambda actual, holder: _command_result(actual.mode),
    )
    assert cli.main(
        [], _bootstrap_context=_test_bootstrap_context(tmp_path)
    ) == 0
    captured = capfd.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "aggregate_counts": {
            "population_n": 128,
            "primary_eligible_n": 10,
        },
        "mode": "official",
        "profile": "official_m6",
        "result_path": "outputs/m6/m6-test",
        "schema_version": cli.M6_OFFICIAL_STATUS_SCHEMA_VERSION,
        "stage_durations_ms": {"eligibility": 1},
        "status": "success",
    }


def test_invalid_arguments_do_not_echo_private_paths(
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    assert cli.main(
        ["--project-root", "/private/path/sentinel"],
        _bootstrap_context=_test_bootstrap_context(tmp_path),
    ) == 1
    captured = capfd.readouterr()
    assert captured.out == ""
    assert "/private/path" not in captured.err
    assert json.loads(captured.err)["reason_code"] == "argument_error"


def test_default_dispatch_requires_opt_in_before_touching_inputs(
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, mode="official")
    argv = [
        "--project-root",
        os.fspath(request.project_root),
        "--data-dir",
        os.fspath(request.data_dir),
        "--m4-run-dir",
        os.fspath(request.m4_run_dir),
        "--run-name",
        request.run_name,
        "--mode",
        request.mode,
        "--eligibility-run-name",
        request.eligibility_run_name,
        "--pilot-run-name",
        request.pilot_run_name,
    ]
    monkeypatch.setattr(
        cli,
        "_require_active_bootstrap_context",
        lambda: SimpleNamespace(project_root=tmp_path),
    )
    assert cli.main(
        argv, _bootstrap_context=_test_bootstrap_context(tmp_path)
    ) == 1
    captured = capfd.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["reason_code"] == "environment_not_enabled"
    assert not request.data_dir.exists()
    assert not request.m4_run_dir.exists()


@pytest.mark.parametrize(
    ("mode", "expected_keyword", "expected_executor"),
    (
        (
            "eligibility_only",
            "eligibility_executor",
            cli.run_m6_eligibility_only_execution,
        ),
        (
            "compute_pilot",
            "pilot_executor",
            cli.run_m6_compute_pilot_execution,
        ),
        (
            "official",
            "official_executor",
            cli.run_m6_official_execution,
        ),
    ),
)
def test_default_dispatch_selects_exact_production_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    expected_keyword: str,
    expected_executor,
) -> None:
    request = _request(tmp_path, mode=mode)
    holder = cli._RunHolder()
    sentinel = object()

    def prepare(actual_request, actual_holder, **executors):
        assert actual_request is request
        assert actual_holder is holder
        assert executors[expected_keyword] is expected_executor
        assert sum(value is not None for value in executors.values()) == 1
        return sentinel

    monkeypatch.setattr(cli, "prepare_m6_official_run", prepare)
    assert cli._dispatch(request, holder) is sentinel


def test_prepare_enables_no_bytecode_before_results_store_use(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6

    request = _request(tmp_path)
    holder = cli._RunHolder()
    repository = SimpleNamespace(root=tmp_path)
    monkeypatch.setattr(
        cli,
        "preflight_repository",
        lambda *_, **__: repository,
    )

    def reject_reservation(actual_request, actual_repository):
        assert actual_request is request
        assert actual_repository is repository
        assert sys.dont_write_bytecode is True
        raise FileExistsError

    monkeypatch.setattr(
        cli,
        "_create_m6_pending_reservation",
        reject_reservation,
    )
    monkeypatch.setattr(sys, "dont_write_bytecode", False)

    with pytest.raises(cli.M6OfficialCommandError, match="result_store_failed"):
        cli.prepare_m6_official_run(
            request,
            holder,
            eligibility_executor=lambda *_: None,
        )
    assert sys.dont_write_bytecode is True


def test_concrete_eligibility_executor_visits_once_with_both_collectors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import evalsim.evaluation.m6 as evaluation_m6
    import evalsim.evaluation.m6_official as numpy_official
    import evalsim.evaluation.m6_waymax_official as waymax_official
    import evalsim.sources.m5_m4_reuse as m4_reuse

    request = _request(tmp_path, mode="eligibility_only")
    cohort = object()
    local = object.__new__(cli.M6LocalInputPreflight)
    object.__setattr__(local, "_factory_sentinel", cli._PREFLIGHT_SENTINEL)
    object.__setattr__(local, "mode", request.mode)
    object.__setattr__(local, "run_name", request.run_name)
    object.__setattr__(local, "accepted_m4", cohort)
    object.__setattr__(local, "data_dir", request.data_dir)
    events: list[str] = []
    authority = object()
    ledger = object()
    selection = object()

    class CaseCollector:
        def __call__(self, member):
            events.append(f"case:{member.index}")

        @property
        def cases(self):
            return ("detached-cases",)

    class WaymaxCollector:
        def __init__(self, actual_authority):
            assert actual_authority is authority

        def __call__(self, member):
            events.append(f"waymax:{member.index}")

        def finalize(self, actual_ledger):
            assert actual_ledger is ledger
            events.append("finalize")
            return SimpleNamespace(
                promotable=True,
                selection=selection,
                revalidate=lambda: events.append("source_revalidate"),
            )

    def visit(actual_cohort, data_dir, visitor):
        assert actual_cohort is cohort
        assert data_dir == request.data_dir
        events.append("visit")
        visitor(SimpleNamespace(index=0))
        visitor(SimpleNamespace(index=1))

    monkeypatch.setattr(
        waymax_official,
        "build_m6_waymax_verified_source_authority",
        lambda actual: authority if actual is cohort else None,
    )
    monkeypatch.setattr(
        waymax_official,
        "M6WaymaxOfficialCollector",
        WaymaxCollector,
    )
    monkeypatch.setattr(
        numpy_official,
        "M6OfficialCaseCollector",
        CaseCollector,
    )
    monkeypatch.setattr(
        evaluation_m6,
        "evaluate_m6_source_eligibility",
        lambda cases: ledger if cases == ("detached-cases",) else None,
    )
    monkeypatch.setattr(
        evaluation_m6,
        "run_m6_numpy_evaluation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("eligibility mode ran policies")
        ),
    )
    monkeypatch.setattr(
        numpy_official,
        "m6_eligibility_rows",
        lambda actual, *, mode: (
            _minimal_eligibility_rows()
            if actual is ledger and mode == "eligibility_only"
            else ()
        ),
    )
    monkeypatch.setattr(m4_reuse, "visit_accepted_m4_cohort", visit)

    def preregister(rows, actual_selection):
        assert tuple(rows) == _minimal_eligibility_rows()
        assert actual_selection is selection
        events.append("preregister")

    evidence = cli.run_m6_eligibility_only_execution(
        request,
        local,
        preregister,
        lambda: (_ for _ in ()).throw(AssertionError("eligibility ran outcomes")),
    )
    assert evidence.mode == "eligibility_only"
    assert evidence.selection is selection
    assert evidence.numpy_rows is None
    assert evidence.waymax_evidence is None
    assert events == [
        "visit",
        "case:0",
        "waymax:0",
        "case:1",
        "waymax:1",
        "finalize",
        "source_revalidate",
        "preregister",
    ]


def _pilot_verified_provenance():
    from evalsim.results import m6 as result_m6

    return result_m6._issue_m6_verified_provenance(
        mode="compute_pilot",
        source_paths=("evalsim/base.py", "uv.lock"),
        row={
            "plan_version": result_m6.M6_PLAN_VERSION,
            "config_version": result_m6.M6_CONFIG_VERSION,
            "statistics_schema_version": result_m6.M6_STATISTICS_SCHEMA_VERSION,
            "population_label": "accepted_m4_complete_case_ten_shard_cohort",
            "source_shard_start": "00000",
            "source_shard_end": "00009",
            "approved_git_commit": "a" * 40,
            "git_tree": "b" * 40,
            "executable_source_sha256": "c" * 64,
            "uv_lock_sha256": "d" * 64,
            "runtime_config_sha256": "e" * 64,
            "accepted_m4_manifest_sha256": "f" * 64,
            "accepted_m4_provenance_sha256": "0" * 64,
            "python_version": "3.11.5",
            "numpy_version": "1.26.4",
            "pyarrow_version": "25.0.0",
            "jax_version": "0.4.38",
            "jaxlib_version": "0.4.38",
            "tensorflow_version": "2.18.1",
            "waymax_commit": result_m6.WAYMAX_COMMIT,
            "jax_backend": "cpu",
            "jax_device_class": "cpu",
            "primary_intervention_fingerprint": (
                result_m6.M6_PRIMARY_INTERVENTION_FINGERPRINT
            ),
            "secondary_intervention_fingerprint": (
                result_m6.M6_SECONDARY_INTERVENTION_FINGERPRINT
            ),
        },
    )


def _issued_pilot_observations(
    selection_binding: str,
    *,
    supported: bool,
    numpy_max_scene_ms: int,
    waymax_max_scene_ms: int,
    waymax_peak_rss_bytes: int,
    selected_cohort_indices: tuple[int, ...],
):
    import evalsim.evaluation.m6_pilot as numpy_pilot
    import evalsim.evaluation.m6_waymax_official as waymax_official

    selected_indices_sha256 = (
        numpy_pilot.m6_numpy_pilot_selected_cohort_indices_sha256(
            selected_cohort_indices
        )
    )
    numpy_durations = (numpy_max_scene_ms,) + (1,) * 7
    numpy_observation = numpy_pilot.M6NumpyPilotObservation(
        scene_count=8,
        scene_durations_ms=numpy_durations,
        total_execution_ms=sum(numpy_durations),
        max_scene_ms=numpy_max_scene_ms,
        selected_cohort_indices_sha256=selected_indices_sha256,
        source_selection_binding_sha256=selection_binding,
        execution_binding_sha256="9" * 64,
        _issuance_capability=numpy_pilot._ISSUER,
    )
    source_binding = "1" * 64
    authority_binding = "2" * 64
    runner_binding = waymax_official._pilot_runner_binding_sha256(
        source_binding_sha256=source_binding,
        selection_binding_sha256=selection_binding,
        selected_cohort_indices_sha256=selected_indices_sha256,
        execution_authority_sha256=authority_binding,
    )
    if supported:
        waymax_durations = (waymax_max_scene_ms,) + (1,) * 7
        waymax_observation = waymax_official.M6WaymaxPilotObservation(
            status="completed",
            scene_count=8,
            validation_ms=1,
            scene_durations_ms=waymax_durations,
            execution_ms=sum(waymax_durations),
            total_wall_ms=1 + sum(waymax_durations),
            max_scene_ms=waymax_max_scene_ms,
            peak_process_rss_bytes=waymax_peak_rss_bytes,
            source_binding_sha256=source_binding,
            selection_binding_sha256=selection_binding,
            selected_cohort_indices_sha256=selected_indices_sha256,
            execution_authority_sha256=authority_binding,
            runner_binding_sha256=runner_binding,
            _issuance_capability=waymax_official._PILOT_ISSUER,
        )
    else:
        waymax_observation = waymax_official.M6WaymaxPilotObservation(
            status="unsupported",
            scene_count=0,
            validation_ms=1,
            scene_durations_ms=(),
            execution_ms=0,
            total_wall_ms=0,
            max_scene_ms=0,
            peak_process_rss_bytes=0,
            source_binding_sha256=source_binding,
            selection_binding_sha256=selection_binding,
            selected_cohort_indices_sha256=selected_indices_sha256,
            execution_authority_sha256=authority_binding,
            runner_binding_sha256=runner_binding,
            _issuance_capability=waymax_official._PILOT_ISSUER,
        )
    return numpy_observation, waymax_observation


def _execution_local(request: cli.M6CommandRequest) -> cli.M6LocalInputPreflight:
    local = object.__new__(cli.M6LocalInputPreflight)
    object.__setattr__(local, "_factory_sentinel", cli._PREFLIGHT_SENTINEL)
    object.__setattr__(local, "mode", request.mode)
    object.__setattr__(local, "run_name", request.run_name)
    object.__setattr__(
        local,
        "result_relative",
        f"outputs/m6/{request.run_name}",
    )
    object.__setattr__(local, "verified_provenance", _pilot_verified_provenance())
    return local


@pytest.mark.parametrize(
    ("supported", "member_count", "waymax_status"),
    (
        (True, 8, "completed"),
        (False, 3, "unsupported"),
    ),
)
def test_compute_pilot_factory_retains_only_aggregate_timing_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    supported: bool,
    member_count: int,
    waymax_status: str,
) -> None:
    import evalsim.evaluation.m6_official as numpy_official
    import evalsim.evaluation.m6_pilot as numpy_pilot
    import evalsim.evaluation.m6_waymax_official as waymax_official

    request = _request(tmp_path, mode="compute_pilot")
    local = _execution_local(request)
    primary_members = tuple(
        SimpleNamespace(
            cohort_index=index + 10,
            rank_sha256=f"{index:064x}",
            eligible=index < member_count,
        )
        for index in range(10)
    )
    qualified_members = tuple(
        member for member in primary_members if member.eligible
    )
    pilot_members = (
        qualified_members[:8] if supported else primary_members[:8]
    )
    selection = SimpleNamespace(
        supported=supported,
        members=qualified_members if supported else (),
        qualification_ledger=SimpleNamespace(rows=primary_members),
        revalidate=lambda **_kwargs: None,
    )
    source = SimpleNamespace(
        selection=selection,
        primary_domain=object(),
        qualification_ledger=SimpleNamespace(rows=primary_members),
        revalidate=lambda: None,
    )
    cases = object()
    ledger = SimpleNamespace(eligible_n=10)
    binding = "a" * 64
    rows = _minimal_eligibility_rows()
    calls: list[object] = []

    monkeypatch.setattr(
        cli,
        "_collect_m6_execution_inputs",
        lambda actual: (
            (cases, ledger, source)
            if actual is local
            else (_ for _ in ()).throw(AssertionError("wrong local"))
        ),
    )
    monkeypatch.setattr(
        numpy_official,
        "m6_eligibility_rows",
        lambda actual, *, mode: (
            rows
            if actual is ledger and mode == "compute_pilot"
            else (_ for _ in ()).throw(AssertionError("wrong ledger"))
        ),
    )
    monkeypatch.setattr(
        waymax_official,
        "m6_waymax_selection_binding_sha256",
        lambda actual: binding if actual is selection else "b" * 64,
    )

    def run_numpy(
        actual_cases,
        actual_ledger,
        selected_indices,
        *,
        selection_binding_sha256,
    ):
        assert actual_cases is cases
        assert actual_ledger is ledger
        assert tuple(selected_indices) == tuple(
            member.cohort_index for member in pilot_members
        )
        assert selection_binding_sha256 == binding
        calls.append("numpy")
        numpy_observation, _ = _issued_pilot_observations(
            binding,
            supported=supported,
            numpy_max_scene_ms=4,
            waymax_max_scene_ms=5 if supported else 0,
            waymax_peak_rss_bytes=1000 if supported else 0,
            selected_cohort_indices=tuple(
                member.cohort_index for member in pilot_members
            ),
        )
        return numpy_observation

    authority = object()

    def run_waymax(actual_source, actual_authority):
        assert actual_source is source
        assert actual_authority is authority
        calls.append("waymax")
        _, waymax_observation = _issued_pilot_observations(
            binding,
            supported=supported,
            numpy_max_scene_ms=4,
            waymax_max_scene_ms=5 if supported else 0,
            waymax_peak_rss_bytes=1000 if supported else 0,
            selected_cohort_indices=tuple(
                member.cohort_index for member in pilot_members
            ),
        )
        assert waymax_observation.status == waymax_status
        return waymax_observation

    monkeypatch.setattr(numpy_pilot, "run_m6_numpy_pilot", run_numpy)
    monkeypatch.setattr(
        waymax_official,
        "build_pinned_m6_waymax_execution_authority",
        lambda: authority,
    )
    monkeypatch.setattr(
        waymax_official,
        "run_m6_waymax_outcome_suppressed_pilot",
        run_waymax,
    )
    monkeypatch.setattr(
        cli,
        "_peak_process_rss_bytes",
        lambda: calls.append("rss") or 1024,
    )
    ticks = iter(
        value * 1_000_000
        for value in (0, 1, 2, 3, 7, 8, 13, 14, 15, 16)
    )
    monkeypatch.setattr(cli.time, "monotonic_ns", lambda: next(ticks))

    def preregister(actual_rows, actual_selection):
        assert tuple(actual_rows) == rows
        assert actual_selection is selection
        calls.append("preregister")

    def begin_outcomes():
        calls.append("begin_outcomes")


    evidence = cli.run_m6_compute_pilot_execution(
        request,
        local,
        preregister,
        begin_outcomes,
    )
    assert evidence.mode == "compute_pilot"
    assert evidence.selection is selection
    assert evidence.pilot_selection_positions == tuple(range(8))
    assert dict(evidence.pilot_summary or {}) == {
        "pilot_scene_n": 8,
        "total_wall_ms": 16,
        "max_scene_ms": 5 if supported else 4,
        "decode_ms": 1,
        "numpy_ms": 11,
        "waymax_ms": 13 if supported else 1,
        "verification_ms": 1,
        "fresh_worker_peak_rss_bytes": 1024,
        "passed": True,
    }
    assert evidence.numpy_rows is None
    assert evidence.waymax_evidence is None
    assert calls == [
        "preregister",
        "begin_outcomes",
        "numpy",
        "waymax",
        "rss",
    ]


def test_compute_pilot_blocks_outcomes_below_primary_floor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import evalsim.evaluation.m6_pilot as numpy_pilot

    request = _request(tmp_path, mode="compute_pilot")
    local = _execution_local(request)
    calls: list[str] = []
    monkeypatch.setattr(
        cli,
        "_collect_m6_execution_inputs",
        lambda actual: (
            ((), SimpleNamespace(eligible_n=9), object())
            if actual is local
            else (_ for _ in ()).throw(AssertionError("wrong local"))
        ),
    )
    monkeypatch.setattr(
        numpy_pilot,
        "run_m6_numpy_pilot",
        lambda *args, **kwargs: calls.append("numpy"),
    )
    with pytest.raises(cli.M6OfficialCommandError, match="eligibility floor"):
        cli.run_m6_compute_pilot_execution(
            request,
            local,
            lambda *_: calls.append("preregister"),
            lambda: calls.append("outcomes"),
        )
    assert calls == []


def test_elapsed_timer_rejects_nonadvancing_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.time, "monotonic_ns", lambda: 100)
    with pytest.raises(cli.M6OfficialCommandError, match="execution clock"):
        cli._positive_elapsed_ms(100)


@pytest.mark.parametrize("supported", (True, False))
def test_official_factory_passes_full_typed_numpy_evidence_to_waymax(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    supported: bool,
) -> None:
    import evalsim.evaluation.m6_official as numpy_official
    import evalsim.evaluation.m6_waymax_official as waymax_official

    request = _request(tmp_path, mode="official")
    local = _execution_local(request)
    selection = SimpleNamespace(
        supported=supported,
        revalidate=lambda **_kwargs: None,
    )
    source = SimpleNamespace(
        selection=selection,
        revalidate=lambda: None,
    )
    cases = object()
    source_ledger = object()
    rows = _minimal_eligibility_rows()
    scene_events: list[str] = []

    class Scene:
        def revalidate(self):
            scene_events.append("scene")

    typed_result = SimpleNamespace(
        primary_scene_results=(Scene(),),
        secondary_scene_results=(Scene(),),
        revalidate=lambda: scene_events.append("typed"),
    )
    numpy_rows = SimpleNamespace(
        revalidate=lambda: scene_events.append("numpy_rows"),
        typed_result=typed_result,
        eligibility_rows=rows,
        phase_durations_ms={
            "numpy_rollouts": 2,
            "paired_metrics": 3,
            "statistics": 4,
            "verification": 5,
        },
    )
    waymax_evidence = SimpleNamespace(
        selection=selection,
        promotable=supported,
        revalidate=lambda: scene_events.append("waymax"),
    )
    authority = object()

    monkeypatch.setattr(
        cli,
        "_collect_m6_execution_inputs",
        lambda actual: (
            (cases, source_ledger, source)
            if actual is local
            else (_ for _ in ()).throw(AssertionError("wrong local"))
        ),
    )
    monkeypatch.setattr(
        numpy_official,
        "m6_eligibility_rows",
        lambda actual, *, mode: (
            rows
            if actual is source_ledger and mode == "official"
            else (_ for _ in ()).throw(AssertionError("wrong ledger"))
        ),
    )
    monkeypatch.setattr(
        numpy_official,
        "run_m6_official_numpy",
        lambda actual: (
            numpy_rows
            if actual is cases
            else (_ for _ in ()).throw(AssertionError("wrong cases"))
        ),
    )
    monkeypatch.setattr(
        waymax_official,
        "build_pinned_m6_waymax_execution_authority",
        lambda: authority,
    )

    def run_waymax(actual_source, actual_authority, actual_numpy):
        assert actual_source is source
        assert actual_authority is authority
        assert actual_numpy is numpy_rows
        return waymax_evidence

    monkeypatch.setattr(
        waymax_official,
        "run_m6_waymax_official",
        run_waymax,
    )
    monkeypatch.setattr(
        cli,
        "_peak_process_rss_bytes",
        lambda: scene_events.append("rss") or 1024,
    )
    ticks = iter(range(0, 12_000_000, 1_000_000))
    monkeypatch.setattr(cli.time, "monotonic_ns", lambda: next(ticks))

    def preregister(actual_rows, actual_selection):
        assert tuple(actual_rows) == rows
        assert actual_selection is selection
        scene_events.append("preregister")

    def begin_outcomes():
        scene_events.append("begin_outcomes")


    evidence = cli.run_m6_official_execution(
        request,
        local,
        preregister,
        begin_outcomes,
    )
    assert evidence.mode == "official"
    assert evidence.selection is selection
    assert evidence.numpy_rows is numpy_rows
    assert evidence.waymax_evidence is waymax_evidence
    assert dict(evidence.stage_durations_ms or {}) == {
        "eligibility": 1,
        "numpy_rollouts": 2,
        "paired_metrics": 3,
        "statistics": 4,
        "waymax": 1,
        "verification": 6,
    }
    assert evidence.fresh_worker_peak_rss_bytes == 1024
    assert scene_events == [
        "preregister",
        "begin_outcomes",
        "numpy_rows",
        "waymax",
        "rss",
    ]


def test_project_root_and_git_snapshot_are_canonical_and_clean(
    tmp_path: Path,
) -> None:
    project, commit = _git_project(tmp_path)
    root = cli._validated_root(project)
    snapshot = cli._git_snapshot(
        root,
        live_lookup=lambda _: commit,
        live_approval_lookup=lambda _: commit,
    )
    assert snapshot.commit == commit
    assert snapshot.tree == _run_git(project, "rev-parse", "HEAD^{tree}")
    assert snapshot.approval_ref == cli._APPROVED_IMPLEMENTATION_REF

    (project / "untracked.py").write_text("private = True\n", encoding="utf-8")
    with pytest.raises(cli.M6OfficialCommandError, match="dirty_worktree"):
        cli._git_snapshot(
            root,
            live_lookup=lambda _: commit,
            live_approval_lookup=lambda _: commit,
        )


def test_git_commands_ignore_caller_path_shims(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.cli import m6_bootstrap as bootstrap

    project, commit = _git_project(tmp_path)
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    marker = tmp_path / "shim-invoked"
    shim = shim_dir / "git"
    shim.write_text(
        "#!/bin/sh\nprintf invoked > "
        + repr(os.fspath(marker))
        + "\nexit 99\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    monkeypatch.setenv(
        "PATH",
        os.fspath(shim_dir) + os.pathsep + os.environ.get("PATH", ""),
    )

    assert cli._git_text(project, "rev-parse", "HEAD") == commit
    assert bootstrap._git_text(project, "rev-parse", "HEAD") == commit
    assert not marker.exists()


@pytest.mark.parametrize(
    "index_flag",
    ("--assume-unchanged", "--skip-worktree"),
)
def test_git_preflight_rejects_hidden_index_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    index_flag: str,
) -> None:
    from evalsim.cli import m6_bootstrap as bootstrap

    project, commit = _git_project(tmp_path)
    _run_git(project, "update-index", index_flag, "evalsim/base.py")
    assert _run_git(project, "status", "--porcelain") == ""

    with pytest.raises(cli.M6OfficialCommandError, match="source_binding_failed"):
        cli._git_snapshot(
            project,
            live_lookup=lambda _: commit,
            live_approval_lookup=lambda _: commit,
        )

    monkeypatch.setenv(bootstrap._LOCAL_OPT_IN, "1")
    with pytest.raises(bootstrap._BootstrapRejected) as rejected:
        bootstrap._preauthenticate_repository(project)
    assert rejected.value.reason_code == "source_binding_failed"


def test_git_preflight_rejects_replacement_refs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.cli import m6_bootstrap as bootstrap

    project, commit = _git_project(tmp_path)
    _run_git(project, "update-ref", f"refs/replace/{commit}", commit)

    with pytest.raises(cli.M6OfficialCommandError, match="replacement refs"):
        cli._git_snapshot(
            project,
            live_lookup=lambda _: commit,
            live_approval_lookup=lambda _: commit,
        )

    monkeypatch.setenv(bootstrap._LOCAL_OPT_IN, "1")
    with pytest.raises(bootstrap._BootstrapRejected) as rejected:
        bootstrap._preauthenticate_repository(project)
    assert rejected.value.reason_code == "source_binding_failed"


def test_approved_source_catalog_compares_disk_bytes_to_head_blob(
    tmp_path: Path,
) -> None:
    project, commit = _git_project(tmp_path)
    (project / "evalsim/base.py").write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(
        cli.M6OfficialCommandError,
        match="differs from its approved HEAD blob",
    ):
        cli._approved_source_catalog(project, commit)


def test_repository_preflight_rejects_bootstrap_receipt_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project, commit = _git_project(tmp_path)
    context = cli._M6BootstrapContext(
        project_root=project,
        site_packages=project / ".venv/lib/python3.11/site-packages",
        pycache_prefix=project / ".pycache-prefix",
        initial_sys_path=("/usr/lib/python3.11",),
        repository_receipt=(
            "0" * 40,
            "1" * 40,
            ("uv.lock",),
            "2" * 64,
            "3" * 64,
        ),
        status_sink=lambda payload, error: True,
        status_failure_sink=lambda callback: True,
        _factory_sentinel=cli._BOOTSTRAP_SENTINEL,
    )
    monkeypatch.setenv(cli._LOCAL_OPT_IN, "1")
    monkeypatch.setattr(
        cli,
        "_require_active_bootstrap_context",
        lambda: context,
    )
    monkeypatch.setattr(
        cli,
        "_validate_loaded_evalsim_modules",
        lambda *_: None,
    )

    with pytest.raises(
        cli.M6OfficialCommandError,
        match="cold-bootstrap approval receipt",
    ):
        cli.preflight_repository(
            _request(project),
            live_lookup=lambda _: commit,
            live_approval_lookup=lambda _: commit,
        )


@pytest.mark.parametrize(
    "relative",
    (
        "evalsim.pyc",
        "evalsim/extra.pyc",
        "evalsim/native.so",
        "evalsim/extra/__init__.pyc",
    ),
)
def test_source_auth_rejects_unapproved_evalsim_import_artifacts(
    tmp_path: Path,
    relative: str,
) -> None:
    from evalsim.cli import m6_bootstrap as bootstrap

    root = tmp_path / "project"
    approved = (
        "evalsim/__init__.py",
        "evalsim/cli/__init__.py",
        "evalsim/cli/m6_official.py",
    )
    for approved_path in approved:
        path = root / approved_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# approved\n", encoding="utf-8")
    ordinary_cache = (
        root / "evalsim/__pycache__/__init__.cpython-311.pyc"
    )
    ordinary_cache.parent.mkdir()
    ordinary_cache.write_bytes(b"redirected-cache")
    bootstrap._reject_unapproved_evalsim_import_artifacts(root, approved)
    cli._reject_unapproved_evalsim_import_artifacts(root, approved)

    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"unapproved-executable")
    with pytest.raises(bootstrap._BootstrapRejected) as rejected:
        bootstrap._reject_unapproved_evalsim_import_artifacts(root, approved)
    assert rejected.value.reason_code == "source_binding_failed"
    with pytest.raises(
        cli.M6OfficialCommandError,
        match="ignored or untracked executable",
    ):
        cli._reject_unapproved_evalsim_import_artifacts(root, approved)


def test_official_predecessor_gate_binds_terminal_same_source_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6
    import evalsim.evaluation.m6_waymax_official as waymax_official

    request = _request(tmp_path, mode="official")
    reason = result_m6.M6_PRIMARY_REJECTION_REASONS[0]
    reason_counts = {
        item: (118 if item == reason else 0)
        for item in result_m6.M6_PRIMARY_REJECTION_REASONS
    }
    selection_fields = {
        "primary_domain_sha256": "b" * 64,
        "primary_domain_member_count": 10,
        "qualification_ledger_sha256": "c" * 64,
        "selector_selection_sha256": "d" * 64,
        "selection_binding_sha256": "e" * 64,
        "selection_supported": True,
        "eligible_count": 8,
        "selection_member_count": 8,
    }

    def table(rows):
        return SimpleNamespace(to_pylist=lambda: [dict(row) for row in rows])

    def predecessor(mode):
        selection_payload = {"mode": mode, **selection_fields}
        selection_receipt = SimpleNamespace(
            **selection_fields,
            to_dict=lambda: dict(selection_payload),
        )
        receipt = SimpleNamespace(
            population_size=128,
            eligible_cohort_indices=tuple(range(10)),
            rejection_reason_counts=reason_counts,
            primary_intervention_fingerprint="a" * 64,
            secondary_b4_cohort_indices=tuple(range(10)),
        )
        context = hashlib.sha256(mode.encode("ascii")).hexdigest()
        run_name = (
            request.pilot_run_name
            if mode == "compute_pilot"
            else request.eligibility_run_name
        )
        assert run_name is not None
        pilot_summary = _pilot_summary()
        qualification_rows = tuple(
            {
                "cohort_index": index,
                "rank_sha256": f"{index:064x}",
                "selected": index < 8,
                "selection_position": index if index < 8 else None,
            }
            for index in range(10)
        )
        selected_indices_sha256 = (
            result_m6._m6_compute_pilot_selected_indices_sha256(
                qualification_rows,
                selection_receipt,
            )
        )
        pilot_row = {
            **pilot_summary,
            "selection_binding_sha256": selection_fields[
                "selection_binding_sha256"
            ],
            "selected_cohort_indices_sha256": selected_indices_sha256,
            "numpy_observation_content_sha256": "1" * 64,
            "waymax_observation_content_sha256": "2" * 64,
        }
        pilot_row["pilot_report_binding_sha256"] = (
            result_m6.m6_compute_pilot_report_binding_sha256(
                run_name=run_name,
                result_path=f"outputs/m6/{run_name}",
                provenance_context_sha256=context,
                selection_binding_sha256=selection_fields[
                    "selection_binding_sha256"
                ],
                selected_cohort_indices_sha256=selected_indices_sha256,
                numpy_observation_content_sha256="1" * 64,
                waymax_observation_content_sha256="2" * 64,
                summary=pilot_summary,
            )
        )
        tables = {
            "typed": table(
                ({
                    "mode": mode,
                    "source": "same",
                    "verification_context_sha256": context,
                },)
            ),
            "pilot": table((pilot_row,)),
            "qualification": table(qualification_rows),
        }
        return SimpleNamespace(
            run_path=tmp_path / "outputs" / "m6" / run_name,
            receipt=receipt,
            waymax_selection_receipt=selection_receipt,
            read_dataset=lambda name: tables[name],
        )

    eligibility = predecessor("eligibility_only")
    pilot = predecessor("compute_pilot")
    calls = []

    def reopen(root, run_name, *, expected_mode):
        assert root == tmp_path
        calls.append((run_name, expected_mode))
        return eligibility if expected_mode == "eligibility_only" else pilot

    module = SimpleNamespace(
        verify_m6_result_store=reopen,
        COMPUTE_PILOT_SUMMARY="pilot",
        TYPED_PROVENANCE="typed",
        WAYMAX_QUALIFICATION="qualification",
        M6_PRIMARY_REJECTION_REASONS=(
            result_m6.M6_PRIMARY_REJECTION_REASONS
        ),
        m6_compute_pilot_report_binding_sha256=(
            result_m6.m6_compute_pilot_report_binding_sha256
        ),
        _m6_compute_pilot_selected_indices_sha256=(
            result_m6._m6_compute_pilot_selected_indices_sha256
        ),
    )
    local = SimpleNamespace(
        verified_provenance=SimpleNamespace(
            to_store_row=lambda: {
                "mode": "official",
                "source": "same",
                "verification_context_sha256": "official",
            }
        )
    )
    selection = SimpleNamespace(
        primary_domain_sha256="b" * 64,
        primary_domain_member_count=10,
        qualification_ledger_sha256="c" * 64,
        selection_sha256="d" * 64,
        supported=True,
        eligible_count=8,
        members=tuple(range(8)),
        revalidate=lambda: None,
    )
    monkeypatch.setattr(
        waymax_official,
        "m6_waymax_selection_binding_sha256",
        lambda actual: "e" * 64 if actual is selection else "f" * 64,
    )
    rows = tuple(
        {
            "cohort_index": index,
            "primary_eligible": index < 10,
            "rejection_reason": None if index < 10 else reason,
            "secondary_b4_feasible": True if index < 10 else None,
        }
        for index in range(128)
    )
    gate = cli._build_m6_predecessor_gate(request, local, module)
    gate(rows, selection)
    assert calls == [
        (request.eligibility_run_name, "eligibility_only"),
        (request.pilot_run_name, "compute_pilot"),
    ]
    drifted = list(rows)
    drifted[0] = {**drifted[0], "primary_eligible": False}
    with pytest.raises(cli.M6OfficialCommandError, match="differs"):
        gate(drifted, selection)

def test_git_snapshot_rejects_noncanonical_origin(tmp_path: Path) -> None:
    project, commit = _git_project(tmp_path)
    _run_git(project, "remote", "set-url", "origin", "https://evil.invalid/repo.git")
    with pytest.raises(cli.M6OfficialCommandError, match="git_remote_invalid"):
        cli._git_snapshot(project, live_lookup=lambda _: commit)


def test_live_main_uses_isolated_credential_free_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed = {}
    monkeypatch.setenv("HOME", "/private/home")
    monkeypatch.setenv("GH_TOKEN", "private-token")
    monkeypatch.setenv("HTTPS_PROXY", "https://evil.invalid")

    def run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=("a" * 40 + "\trefs/heads/main\n").encode("ascii"),
            stderr=None,
        )

    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli._live_main(tmp_path) == "a" * 40
    assert observed["timeout"] == 30
    assert observed["cwd"] == os.sep
    assert observed["stderr"] is subprocess.DEVNULL
    command = observed["command"]
    assert "credential.helper=" in command
    assert "http.followRedirects=false" in command
    environment = observed["env"]
    assert "HOME" not in environment
    assert "GH_TOKEN" not in environment
    assert "HTTPS_PROXY" not in environment
    assert environment["GIT_ALLOW_PROTOCOL"] == "https"


def test_exact_shard_inventory_is_canonical_without_opening_payloads(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    data_dir = root / cli._DEFAULT_DATA_RELATIVE
    data_dir.mkdir(parents=True)
    expected = []
    for suffix in cli._SHARD_SUFFIXES:
        path = data_dir / f"validation-{suffix}"
        path.write_bytes(b"")
        expected.append(path)
    assert cli._resolve_shard_inventory(root, data_dir) == tuple(expected)

    duplicate = data_dir / f"duplicate-{cli._SHARD_SUFFIXES[0]}"
    duplicate.write_bytes(b"")
    with pytest.raises(cli.M6OfficialCommandError, match="shard_set_invalid"):
        cli._resolve_shard_inventory(root, data_dir)


def test_shard_inventory_rejects_linked_input(tmp_path: Path) -> None:
    root = tmp_path / "project"
    data_dir = root / cli._DEFAULT_DATA_RELATIVE
    data_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"")
    for index, suffix in enumerate(cli._SHARD_SUFFIXES):
        path = data_dir / f"validation-{suffix}"
        if index == 0:
            path.symlink_to(outside)
        else:
            path.write_bytes(b"")
    with pytest.raises(cli.M6OfficialCommandError, match="shard_set_invalid"):
        cli._resolve_shard_inventory(root, data_dir)


def _write_m4_presence_fixture(project: Path) -> Path:
    run_dir = project / "outputs/m4/accepted"
    for relative in cli._M4_REQUIRED_ARTIFACTS:
        path = run_dir.joinpath(*Path(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return run_dir


def test_accepted_m4_presence_requires_complete_ignored_regular_files(
    tmp_path: Path,
) -> None:
    project, _ = _git_project(tmp_path)
    run_dir = _write_m4_presence_fixture(project)
    assert cli._accepted_m4_snapshot_presence(project, run_dir) == run_dir

    (run_dir / "execution-provenance.json").unlink()
    with pytest.raises(
        cli.M6OfficialCommandError,
        match="accepted_m4_snapshot_invalid",
    ):
        cli._accepted_m4_snapshot_presence(project, run_dir)


def test_source_allowlist_is_tracked_ordered_and_content_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project, _ = _git_project(tmp_path)
    required = (".gitignore", "pyproject.toml")
    monkeypatch.setattr(cli, "_SOURCE_REQUIRED_FILES", required)
    monkeypatch.setattr(
        cli,
        "_SOURCE_PATHSPECS",
        ("evalsim", ":(glob)tests/test_m6_*.py", *required),
    )
    source = project / "evalsim/source.py"
    test = project / "tests/test_m6_source.py"
    source.parent.mkdir(exist_ok=True)
    test.parent.mkdir(exist_ok=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    test.write_text("def test_value(): pass\n", encoding="utf-8")
    _run_git(project, "add", ".")
    _run_git(
        project,
        "-c",
        "user.name=EvalSim Test",
        "-c",
        "user.email=evalsim-test@example.invalid",
        "commit",
        "-m",
        "sources",
    )
    paths = cli._tracked_source_allowlist(project)
    assert paths == tuple(sorted(paths))
    assert "evalsim/source.py" in paths
    assert "tests/test_m6_source.py" in paths
    first = cli._source_fingerprint(project, paths)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert cli._source_fingerprint(project, paths) != first


def test_loaded_module_allowlist_rejects_source_outside_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    loaded_project_paths = set()
    for module in tuple(sys.modules.values()):
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str):
            continue
        try:
            relative = Path(raw).resolve(strict=True).relative_to(root)
        except (OSError, ValueError):
            continue
        if relative.parts[:1] != (".venv",):
            loaded_project_paths.add(relative.as_posix())
    allowed = tuple(
        sorted(
            loaded_project_paths
            | {
                path.relative_to(root).as_posix()
                for path in (root / "evalsim").rglob("*.py")
            }
        )
    )
    cli._validate_loaded_evalsim_modules(root, allowed)

    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")
    fake = ModuleType("evalsim.untrusted")
    fake.__file__ = os.fspath(outside)
    monkeypatch.setitem(sys.modules, "evalsim.untrusted", fake)
    with pytest.raises(cli.M6OfficialCommandError, match="source_binding_failed"):
        cli._validate_loaded_evalsim_modules(root, allowed)


def test_loaded_evalsim_modules_require_exact_tracked_checkout_origins(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    package = root / "evalsim/__init__.py"
    source = root / "evalsim/worker.py"
    package.parent.mkdir(parents=True)
    package.write_text("# package\n", encoding="utf-8")
    source.write_text("VALUE = 1\n", encoding="utf-8")
    allowed = ("evalsim/__init__.py", "evalsim/worker.py")

    loaded = {
        "evalsim": SimpleNamespace(__file__=os.fspath(package)),
        "evalsim.worker": SimpleNamespace(__file__=os.fspath(source)),
    }
    cli._validate_loaded_evalsim_modules(
        root,
        allowed,
        loaded_modules=loaded,
    )

    stale = (
        root
        / ".venv/lib/python3.11/site-packages/evalsim/worker.py"
    )
    stale.parent.mkdir(parents=True)
    stale.write_text("VALUE = 0\n", encoding="utf-8")
    loaded["evalsim.worker"] = SimpleNamespace(__file__=os.fspath(stale))
    with pytest.raises(cli.M6OfficialCommandError, match="source_binding_failed"):
        cli._validate_loaded_evalsim_modules(
            root,
            allowed,
            loaded_modules=loaded,
        )

    loaded["evalsim.worker"] = SimpleNamespace(__file__=os.fspath(package))
    with pytest.raises(cli.M6OfficialCommandError, match="source_binding_failed"):
        cli._validate_loaded_evalsim_modules(
            root,
            allowed,
            loaded_modules=loaded,
        )

    loaded["evalsim.worker"] = SimpleNamespace(__file__=None)
    with pytest.raises(cli.M6OfficialCommandError, match="source_binding_failed"):
        cli._validate_loaded_evalsim_modules(
            root,
            allowed,
            loaded_modules=loaded,
        )


def test_loaded_evalsim_module_origin_is_no_follow(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    source = root / "evalsim/worker.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    linked = root / "evalsim/linked.py"
    linked.symlink_to(source)

    with pytest.raises(cli.M6OfficialCommandError, match="source_binding_failed"):
        cli._validate_loaded_evalsim_modules(
            root,
            ("evalsim/linked.py",),
            loaded_modules={
                "evalsim.linked": SimpleNamespace(__file__=os.fspath(linked))
            },
        )


def test_runtime_allowlist_uses_metadata_without_importing_optional_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.platform, "python_version", lambda: "3.11.5")
    expected = dict(cli._EXPECTED_RUNTIME_VERSIONS)
    observed = cli._runtime_allowlist(version_resolver=expected.__getitem__)
    assert dict(observed) == {"python": "3.11.5", **expected}

    wrong = dict(expected)
    wrong["jax"] = "9.9.9"
    with pytest.raises(cli.M6OfficialCommandError, match="runtime_mismatch"):
        cli._runtime_allowlist(version_resolver=wrong.__getitem__)


def test_repository_preflight_requires_active_bootstrap_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(cli._LOCAL_OPT_IN, "1")
    monkeypatch.setattr(cli, "_ACTIVE_BOOTSTRAP_CONTEXT", None)
    with pytest.raises(cli.M6OfficialCommandError, match="active bootstrap context"):
        cli.preflight_repository(_request(tmp_path))


def test_repository_preflight_stops_at_opt_in_before_git(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(cli._LOCAL_OPT_IN, raising=False)
    monkeypatch.setattr(
        cli,
        "_require_active_bootstrap_context",
        lambda: SimpleNamespace(project_root=tmp_path),
    )
    monkeypatch.setattr(
        cli,
        "_validated_root",
        lambda _: (_ for _ in ()).throw(
            AssertionError("Git/root boundary crossed before opt-in")
        ),
    )
    with pytest.raises(cli.M6OfficialCommandError, match="environment_not_enabled"):
        cli.preflight_repository(_request(tmp_path))


class _FakeRecordPath:
    def __init__(self, relative: str, payload: bytes | None) -> None:
        self.relative = relative
        if payload is None:
            self.hash = None
            self.size = None
        else:
            digest = hashlib.sha256(payload).digest()
            value = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
            self.hash = SimpleNamespace(mode="sha256", value=value)
            self.size = len(payload)

    def __fspath__(self) -> str:
        return self.relative


class _FakeDistribution:
    def __init__(
        self,
        site_packages: Path,
        distribution_name: str,
        module_name: str,
        commit: str | None,
        version: str,
    ) -> None:
        self.site_packages = site_packages
        self.commit = commit
        self.metadata = {"Name": distribution_name}
        self.version = version
        normalized = distribution_name.replace("-", "_")
        self.dist_info = site_packages / f"{normalized}.dist-info"
        self.dist_info.mkdir()
        payloads = [
            path
            for owned_root in (
                site_packages / module_name,
                site_packages / f"{module_name}.libs",
            )
            if owned_root.exists()
            for path in owned_root.rglob("*")
            if path.is_file()
        ]
        if distribution_name == "waymo-waymax" and commit is not None:
            direct_url = self.dist_info / "direct_url.json"
            direct_url.write_text(
                json.dumps(
                    {
                        "url": cli._WAYMAX_REMOTE,
                        "vcs_info": {
                            "commit_id": commit,
                            "requested_revision": commit,
                            "vcs": "git",
                        },
                    }
                ),
                encoding="utf-8",
            )
            payloads.append(direct_url)
        payloads.sort()
        record = self.dist_info / "RECORD"
        rows = []
        entries = []
        for path in payloads:
            relative = path.relative_to(site_packages).as_posix()
            payload = path.read_bytes()
            entry = _FakeRecordPath(relative, payload)
            entries.append(entry)
            rows.append(f"{relative},sha256={entry.hash.value},{entry.size}")
        record_relative = record.relative_to(site_packages).as_posix()
        rows.append(f"{record_relative},,")
        record.write_text("\n".join(rows) + "\n", encoding="utf-8")
        entries.append(_FakeRecordPath(record_relative, None))
        self.files = tuple(entries)

    def read_text(self, name: str) -> str:
        return (self.dist_info / name).read_text(encoding="utf-8")

    def locate_file(self, entry: _FakeRecordPath) -> Path:
        return self.site_packages / os.fspath(entry)


def _repository_preflight(
    monkeypatch: pytest.MonkeyPatch,
    project: Path,
    commit: str,
) -> cli.M6RepositoryPreflight:
    monkeypatch.setenv(cli._LOCAL_OPT_IN, "1")
    monkeypatch.setattr(
        cli,
        "_require_active_bootstrap_context",
        lambda: SimpleNamespace(project_root=project),
    )
    monkeypatch.setattr(
        cli,
        "_validate_loaded_evalsim_modules",
        lambda *_: None,
    )
    return cli.preflight_repository(
        _request(project),
        live_lookup=lambda _: commit,
        live_approval_lookup=lambda _: commit,
    )


def _write_shard_fixture(project: Path) -> None:
    data_dir = project / cli._DEFAULT_DATA_RELATIVE
    data_dir.mkdir(parents=True)
    for suffix in cli._SHARD_SUFFIXES:
        (data_dir / f"validation-{suffix}").write_bytes(b"fixture")


_FIXTURE_RUNTIME_CATALOG_SHA256 = {
    "flax": "2b4a6709db2a7980eb41940aa536c7bcebe840b5eb1e9da9c3b6e404d051f222",
    "jax": "7eb3bf80e63b1a98d3658312ac989fbdeb4784ba5cfe99ecd77a36ff23ba9e38",
    "jaxlib": "8f159d7df04432ef19050ccbac67d074b244dca159b31dcbb3a8769d0e2f6c84",
    "keras": "b0ca48f911d959a390ae090d2f111e7ed8e8842b13a1e91e5579f08fbff58efc",
    "numpy": "c838473cb5082fd4c051b3a40c58f2df753ea4dfa0b1823f19e23dcedd460fa4",
    "pyarrow": "8ae782d8c0952f69e546c6b454e65098f42392fe6fc1f298cdfa399206ca5802",
    "tensorflow": "a7163ced61b17fe00a0f660a5dafa929ac16ad79f652595f074ed465d968c538",
    "waymo-waymax": "82655d21ceede48df4035a4a052d63c944a917b0c614c0bc7d31289233c6408e",
}


def _runtime_kwargs(
    project: Path,
    waymax_commit: str,
    *,
    monkeypatch: pytest.MonkeyPatch,
    backend: str = "cpu",
) -> dict[str, object]:
    modules = {}
    site_packages = project / ".venv/lib/python3.11/site-packages"
    for module_name in cli._RUNTIME_MODULES.values():
        origin = (
            site_packages
            / module_name
            / "__init__.py"
        )
        origin.parent.mkdir(parents=True, exist_ok=True)
        origin.write_text(f"# {module_name}\n", encoding="utf-8")
        modules[module_name] = SimpleNamespace(__file__=os.fspath(origin))
    fixture_files = {
        site_packages / "jax/_src/core.py": b"jax-core",
        site_packages / "jaxlib/xla_extension.so": b"jaxlib-native",
        site_packages / "waymax/_native.dylib": b"waymax-native",
        site_packages
        / "jax/__pycache__/core.cpython-311.pyc": b"recorded-bytecode",
        site_packages / "numpy.libs/libopenblas.so.0": b"numpy-native",
        site_packages / "jax/_src/shared.py": b"hardlinked-source",
        site_packages / "keras/_tf_keras/__init__.py": b"keras-tf-api",
        site_packages
        / "tensorflow/_api/v2/__init__.py": b"tensorflow-v2-api",
    }
    for path, payload in fixture_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    os.link(
        site_packages / "jax/_src/shared.py",
        site_packages / "jax/_src/shared_alias.py",
    )
    modules["tensorflow"].__path__ = (
        os.fspath(site_packages / "tensorflow"),
        os.fspath(site_packages / "tensorflow/_api/v2"),
        os.fspath(site_packages / "keras/_tf_keras"),
        os.fspath(site_packages / "keras/api"),
    )
    modules["jax"].default_backend = lambda: backend
    modules["jax"].devices = lambda: [SimpleNamespace(platform=backend)]
    versions = dict(cli._EXPECTED_RUNTIME_VERSIONS)
    distributions = {
        distribution_name: _FakeDistribution(
            site_packages,
            distribution_name,
            module_name,
            waymax_commit,
            versions[distribution_name],
        )
        for distribution_name, module_name in cli._RUNTIME_MODULES.items()
    }
    transitive_versions = {
        "chex": "0.1.90",
        "optax": "0.2.5",
        "scipy": "1.17.1",
    }
    for distribution_name, version in transitive_versions.items():
        package = site_packages / distribution_name
        package.mkdir()
        (package / "__init__.py").write_text(
            f"# {distribution_name}\n", encoding="utf-8"
        )
        (package / "fixture-data.bin").write_bytes(
            f"{distribution_name}-data".encode("ascii")
        )
        distributions[distribution_name] = _FakeDistribution(
            site_packages,
            distribution_name,
            distribution_name,
            None,
            version,
        )
    evalsim_fixture = site_packages / "evalsim_fixture/__init__.py"
    evalsim_fixture.parent.mkdir()
    evalsim_fixture.write_text("# editable project metadata\n", encoding="utf-8")
    distributions["evalsim"] = _FakeDistribution(
        site_packages,
        "evalsim",
        "evalsim_fixture",
        None,
        cli._EXPECTED_PROJECT_DISTRIBUTION_VERSION,
    )
    for relative, payload in {
        "_editable_impl_evalsim.pth": os.fspath(project).encode("utf-8"),
        "_virtualenv.pth": b"import _virtualenv",
        "_virtualenv.py": b"# uv virtualenv bootstrap\n",
    }.items():
        (site_packages / relative).write_bytes(payload)
    environment_versions = {
        **versions,
        **transitive_versions,
    }
    monkeypatch.setattr(
        cli,
        "_EXPECTED_RUNTIME_CATALOG_SHA256",
        _FIXTURE_RUNTIME_CATALOG_SHA256,
    )
    monkeypatch.setattr(
        cli,
        "_EXPECTED_ENVIRONMENT_VERSIONS",
        environment_versions,
    )
    environment_catalog = tuple(
        sorted(
            (
                cli._environment_distribution_catalog(project, distribution)
                for distribution in distributions.values()
            ),
            key=lambda item: item.distribution_name,
        )
    )
    infrastructure = tuple(
        cli._guarded_file_snapshot(project, relative, require_single_link=False)
        for relative in cli._EXPECTED_ENVIRONMENT_INFRASTRUCTURE
    )
    monkeypatch.setattr(
        cli,
        "_EXPECTED_ENVIRONMENT_CATALOG_SHA256",
        cli._environment_catalog_sha256(environment_catalog, infrastructure),
    )
    return {
        "version_resolver": versions.__getitem__,
        "module_importer": modules.__getitem__,
        "distribution_resolver": distributions.__getitem__,
        "python_version_resolver": lambda: "3.11.5",
        "environment_distributions_resolver": lambda: tuple(
            distributions.values()
        ),
    }


def _fake_m4_cohort(project: Path, run_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        project_root=project,
        run_dir=run_dir,
        evidence=SimpleNamespace(
            manifest_sha256="a" * 64,
            execution_provenance_sha256="b" * 64,
        ),
    )


def test_git_snapshot_requires_fixed_lightweight_approval_tag(
    tmp_path: Path,
) -> None:
    project, approved = _git_project(tmp_path)
    (project / "uv.lock").write_text("version = 2\n", encoding="utf-8")
    _run_git(project, "add", "uv.lock")
    _run_git(
        project,
        "-c",
        "user.name=EvalSim Test",
        "-c",
        "user.email=evalsim-test@example.invalid",
        "commit",
        "-m",
        "unapproved",
    )
    head = _run_git(project, "rev-parse", "HEAD")
    _run_git(project, "update-ref", "refs/remotes/origin/main", head)

    with pytest.raises(
        cli.M6OfficialCommandError,
        match="approved_commit_mismatch",
    ):
        cli._git_snapshot(
            project,
            live_lookup=lambda _: head,
            live_approval_lookup=lambda _: approved,
        )


def test_stdlib_pending_reservation_is_exact_and_one_shot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6

    project, commit = _git_project(tmp_path)
    request = _request(project)
    repository = _repository_preflight(monkeypatch, project, commit)

    reservation = cli._create_m6_pending_reservation(request, repository)
    pending = reservation.run_path / result_m6.PENDING_MARKER
    payload = json.loads(pending.read_text(encoding="ascii"))
    assert payload == {
        "capability_sha256": hashlib.sha256(
            reservation.capability_nonce
        ).hexdigest(),
        "mode": request.mode,
        "result_path": f"outputs/m6/{request.run_name}",
        "schema_version": result_m6.M6_RESULT_STORE_SCHEMA_VERSION,
        "state": "PENDING",
    }

    store = result_m6.M6ResultStore.adopt_pending(reservation)
    assert store.run_path == reservation.run_path
    assert store._phase == "pending"
    with pytest.raises(
        cli.M6OfficialCommandError,
        match="already consumed",
    ):
        result_m6.M6ResultStore.adopt_pending(reservation)


def test_local_preflight_requires_exact_pending_store_before_m4_or_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project, commit = _git_project(tmp_path)
    repository = _repository_preflight(monkeypatch, project, commit)
    crossed = {"m4": False}

    def verifier(*_):
        crossed["m4"] = True
        raise AssertionError("M4 verifier crossed before exact PENDING")

    with pytest.raises(cli.M6OfficialCommandError, match="result_contract_failed"):
        cli.preflight_local_inputs(
            _request(project),
            repository,
            object(),
            m4_verifier=verifier,
        )
    assert crossed == {"m4": False}


def test_zero_byte_m4_presence_is_not_verified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6

    project, commit = _git_project(tmp_path)
    repository = _repository_preflight(monkeypatch, project, commit)
    _write_shard_fixture(project)
    _write_m4_presence_fixture(project)
    store = result_m6.M6ResultStore.create(
        project,
        "m6-test",
        mode=result_m6.ELIGIBILITY_ONLY_MODE,
    )
    crossed = {"m4": False, "runtime": False}

    def reject_m4(*_):
        crossed["m4"] = True
        raise ValueError("empty fixture is not accepted M4 evidence")

    def reject_runtime(_):
        crossed["runtime"] = True
        raise AssertionError("runtime crossed after failed M4 verification")

    with pytest.raises(
        cli.M6OfficialCommandError,
        match="accepted_m4_snapshot_invalid",
    ):
        cli.preflight_local_inputs(
            _request(project),
            repository,
            store,
            m4_verifier=reject_m4,
            module_importer=reject_runtime,
        )
    assert crossed == {"m4": True, "runtime": False}


def test_pending_preflight_issues_and_reverifies_mechanical_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6

    project, commit = _git_project(tmp_path)
    request = _request(project)
    repository = _repository_preflight(monkeypatch, project, commit)
    _write_shard_fixture(project)
    m4_run = _write_m4_presence_fixture(project)
    store = result_m6.M6ResultStore.create(
        project,
        request.run_name,
        mode=request.mode,
    )
    cohort = _fake_m4_cohort(project, m4_run)
    crossed = {"verified_pending": False, "reverified": False}

    def verify_m4(root: Path, run_dir: Path) -> SimpleNamespace:
        assert root == project
        assert run_dir == m4_run
        assert store._phase == "pending"
        assert (store.run_path / result_m6.PENDING_MARKER).is_file()
        crossed["verified_pending"] = True
        return cohort

    runtime_kwargs = _runtime_kwargs(
        project,
        result_m6.WAYMAX_COMMIT,
        monkeypatch=monkeypatch,
    )
    local = cli.preflight_local_inputs(
        request,
        repository,
        store,
        m4_verifier=verify_m4,
        **runtime_kwargs,
    )
    assert crossed["verified_pending"] is True
    assert type(local.verified_provenance) is result_m6.M6VerifiedProvenance
    row = local.verified_provenance.to_store_row()
    assert row["approved_git_commit"] == commit
    assert row["git_tree"] == repository.git.tree
    assert row["executable_source_paths"] == list(repository.source_paths)
    assert row["executable_source_sha256"] == repository.source_sha256
    assert row["uv_lock_sha256"] == repository.uv_lock_sha256
    assert row["runtime_config_sha256"] == local.runtime.runtime_config_sha256
    assert row["accepted_m4_manifest_sha256"] == "a" * 64
    assert row["accepted_m4_provenance_sha256"] == "b" * 64
    assert row["jax_backend"] == row["jax_device_class"] == "cpu"
    assert row["waymax_commit"] == result_m6.WAYMAX_COMMIT

    def reverify_m4(actual: object) -> None:
        assert actual is cohort
        crossed["reverified"] = True

    fresh = cli.reverify_local_inputs(
        request,
        repository,
        store,
        local,
        m4_reverifier=reverify_m4,
        **runtime_kwargs,
    )
    assert crossed["reverified"] is True
    assert (
        fresh.verified_provenance.context_sha256
        == local.verified_provenance.context_sha256
    )
    assert (
        cli.reverify_verified_provenance(
            request,
            repository,
            fresh,
            result_m6,
        ).context_sha256
        == local.verified_provenance.context_sha256
    )


def test_runtime_observer_rejects_non_cpu_backend_after_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6

    project, commit = _git_project(tmp_path)
    repository = _repository_preflight(monkeypatch, project, commit)
    kwargs = _runtime_kwargs(
        project,
        result_m6.WAYMAX_COMMIT,
        monkeypatch=monkeypatch,
        backend="gpu",
    )
    with pytest.raises(cli.M6OfficialCommandError, match="runtime_mismatch"):
        cli._observe_runtime(
            project,
            repository,
            result_m6,
            **kwargs,
        )


def test_runtime_catalog_ignores_only_missing_declared_package_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6

    project, commit = _git_project(tmp_path)
    repository = _repository_preflight(monkeypatch, project, commit)
    kwargs = _runtime_kwargs(
        project,
        result_m6.WAYMAX_COMMIT,
        monkeypatch=monkeypatch,
    )
    importer = kwargs["module_importer"]
    assert callable(importer)
    tensorflow = importer("tensorflow")
    declared_paths = tuple(tensorflow.__path__)
    missing = project / ".venv/lib/python3.11/site-packages/keras/api"
    assert not missing.exists()
    assert os.fspath(missing) in declared_paths
    assert any("keras/_tf_keras" in path for path in declared_paths)

    observed = cli._observe_runtime(
        project,
        repository,
        result_m6,
        **kwargs,
    )
    assert cli._SHA256.fullmatch(observed.runtime_catalog_sha256)

    unowned = project / ".venv/lib/python3.11/site-packages/unowned-root"
    unowned.mkdir()
    (unowned / "bad.py").write_text("VALUE = 1\n", encoding="utf-8")
    tensorflow.__path__ = (*declared_paths, os.fspath(unowned))
    with pytest.raises(
        cli.M6OfficialCommandError,
        match="unowned executable or data file",
    ):
        cli._observe_runtime(
            project,
            repository,
            result_m6,
            **kwargs,
        )

    outside = tmp_path / "existing-outside-root"
    outside.mkdir()
    tensorflow.__path__ = (*declared_paths, os.fspath(outside))
    with pytest.raises(cli.M6OfficialCommandError, match="runtime_mismatch"):
        cli._observe_runtime(
            project,
            repository,
            result_m6,
            **kwargs,
        )

    linked = project / ".venv/lib/python3.11/site-packages/linked-root"
    linked.symlink_to(outside, target_is_directory=True)
    tensorflow.__path__ = (*declared_paths, os.fspath(linked))
    with pytest.raises(cli.M6OfficialCommandError, match="runtime_mismatch"):
        cli._observe_runtime(
            project,
            repository,
            result_m6,
            **kwargs,
        )


def test_runtime_catalog_verifies_recorded_python_native_and_bytecode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6

    project, commit = _git_project(tmp_path)
    repository = _repository_preflight(monkeypatch, project, commit)
    site_packages = project / ".venv/lib/python3.11/site-packages"
    targets = (
        site_packages / "jax/_src/core.py",
        site_packages / "jaxlib/xla_extension.so",
        site_packages / "waymax/_native.dylib",
        site_packages / "jax/__pycache__/core.cpython-311.pyc",
        site_packages / "numpy.libs/libopenblas.so.0",
    )
    kwargs = _runtime_kwargs(
        project, result_m6.WAYMAX_COMMIT, monkeypatch=monkeypatch
    )
    monkeypatch.setattr(sys, "dont_write_bytecode", False)

    baseline = cli._observe_runtime(
        project,
        repository,
        result_m6,
        **kwargs,
    )

    catalog_paths = {
        file.relative_path
        for package in baseline._runtime_catalog
        for file in package.files
    }
    assert {
        target.relative_to(project).as_posix() for target in targets
    }.issubset(catalog_paths)
    assert cli._SHA256.fullmatch(baseline.runtime_catalog_sha256)
    assert cli._SHA256.fullmatch(baseline.runtime_config_sha256)
    assert sys.dont_write_bytecode is True

    for target in targets:
        original = target.read_bytes()
        target.write_bytes(original + b"-changed")
        with pytest.raises(
            cli.M6OfficialCommandError,
            match="RECORD hash or size",
        ):
            cli._observe_runtime(
                project,
                repository,
                result_m6,
                **kwargs,
            )
        target.write_bytes(original)

    for added in (
        site_packages / "waymax/__pycache__/new_module.cpython-311.pyc",
        site_packages / "numpy.libs/unrecorded.so",
    ):
        added.parent.mkdir(parents=True, exist_ok=True)
        added.write_bytes(b"unrecorded-executable")
        with pytest.raises(
            cli.M6OfficialCommandError,
            match="unowned executable or data file",
        ):
            cli._observe_runtime(
                project,
                repository,
                result_m6,
                **kwargs,
            )
        added.unlink()



def test_complete_environment_catalog_rejects_transitive_tamper_and_extra(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6

    project, commit = _git_project(tmp_path)
    repository = _repository_preflight(monkeypatch, project, commit)
    kwargs = _runtime_kwargs(
        project, result_m6.WAYMAX_COMMIT, monkeypatch=monkeypatch
    )
    baseline = cli._observe_runtime(
        project,
        repository,
        result_m6,
        **kwargs,
    )
    environment_names = {
        item.distribution_name for item in baseline._environment_catalog
    }
    assert {"chex", "optax", "scipy"}.issubset(environment_names)
    assert cli._SHA256.fullmatch(baseline.environment_catalog_sha256)

    target = (
        project
        / ".venv/lib/python3.11/site-packages/chex/fixture-data.bin"
    )
    original = target.read_bytes()
    target.write_bytes(original + b"-tampered")
    with pytest.raises(
        cli.M6OfficialCommandError,
        match="installed file differs from its RECORD hash or size",
    ):
        cli._observe_runtime(
            project,
            repository,
            result_m6,
            **kwargs,
        )
    target.write_bytes(original)

    resolver = kwargs["environment_distributions_resolver"]
    assert callable(resolver)
    installed = tuple(resolver())
    site_packages = project / ".venv/lib/python3.11/site-packages"
    extra_package = site_packages / "unexpected_dependency"
    extra_package.mkdir()
    (extra_package / "__init__.py").write_text(
        "# unexpected dependency\n", encoding="utf-8"
    )
    extra = _FakeDistribution(
        site_packages,
        "unexpected-dependency",
        "unexpected_dependency",
        None,
        "1.0.0",
    )
    kwargs["environment_distributions_resolver"] = lambda: installed + (extra,)
    with pytest.raises(
        cli.M6OfficialCommandError,
        match="package/version domain differs",
    ):
        cli._observe_runtime(
            project,
            repository,
            result_m6,
            **kwargs,
        )


def test_complete_environment_catalog_binds_editable_evalsim_record_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6

    project, commit = _git_project(tmp_path)
    _repository_preflight(monkeypatch, project, commit)
    kwargs = _runtime_kwargs(
        project, result_m6.WAYMAX_COMMIT, monkeypatch=monkeypatch
    )
    resolver = kwargs["environment_distributions_resolver"]
    assert callable(resolver)
    installed = tuple(resolver())
    distribution = next(
        item
        for item in installed
        if item.metadata["Name"] == "evalsim"
    )
    target = (
        project
        / ".venv/lib/python3.11/site-packages"
        / "evalsim_fixture/extra.py"
    )
    payload = b"# coherently recorded but not anchored\n"
    target.write_bytes(payload)
    relative = target.relative_to(distribution.site_packages).as_posix()
    entry = _FakeRecordPath(relative, payload)
    record = distribution.dist_info / "RECORD"
    terminal_row = distribution.files[-1].relative + ",,\n"
    assert record.read_text(encoding="utf-8").endswith(terminal_row)
    record.write_text(
        record.read_text(encoding="utf-8")[: -len(terminal_row)]
        + f"{relative},sha256={entry.hash.value},{entry.size}\n"
        + terminal_row,
        encoding="utf-8",
    )
    distribution.files = (
        *distribution.files[:-1],
        entry,
        distribution.files[-1],
    )

    with pytest.raises(
        cli.M6OfficialCommandError,
        match="immutable catalog anchor",
    ):
        cli._complete_environment_catalog(
            project,
            lambda: installed,
        )


@pytest.mark.parametrize("relative", ("sitecustomize.py", "malicious.pth"))
def test_complete_environment_catalog_rejects_unowned_site_bootstrap_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative: str,
) -> None:
    from evalsim.results import m6 as result_m6

    project, commit = _git_project(tmp_path)
    repository = _repository_preflight(monkeypatch, project, commit)
    kwargs = _runtime_kwargs(
        project, result_m6.WAYMAX_COMMIT, monkeypatch=monkeypatch
    )
    target = project / ".venv/lib/python3.11/site-packages" / relative
    target.write_text("raise RuntimeError('must never execute')\n", encoding="utf-8")

    with pytest.raises(
        cli.M6OfficialCommandError,
        match="unowned executable or data file",
    ):
        cli._observe_runtime(
            project,
            repository,
            result_m6,
            **kwargs,
        )


def test_runtime_catalog_rejects_mutation_during_observation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6

    project, commit = _git_project(tmp_path)
    repository = _repository_preflight(monkeypatch, project, commit)
    kwargs = _runtime_kwargs(
        project, result_m6.WAYMAX_COMMIT, monkeypatch=monkeypatch
    )
    target = project / ".venv/lib/python3.11/site-packages/jax/_src/core.py"
    original = cli._runtime_config_sha256

    def mutate_after_catalog(*args, **inner_kwargs):
        value = original(*args, **inner_kwargs)
        target.write_text("VALUE = 2\n", encoding="utf-8")
        return value

    monkeypatch.setattr(cli, "_runtime_config_sha256", mutate_after_catalog)
    with pytest.raises(
        cli.M6OfficialCommandError,
        match="RECORD hash or size",
    ):
        cli._observe_runtime(
            project,
            repository,
            result_m6,
            **kwargs,
        )


def test_runtime_catalog_rejects_symlinked_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6

    project, commit = _git_project(tmp_path)
    repository = _repository_preflight(monkeypatch, project, commit)
    kwargs = _runtime_kwargs(
        project, result_m6.WAYMAX_COMMIT, monkeypatch=monkeypatch
    )
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")
    linked = (
        project
        / ".venv/lib/python3.11/site-packages/waymax/linked.py"
    )
    linked.symlink_to(outside)

    with pytest.raises(cli.M6OfficialCommandError, match="runtime_mismatch"):
        cli._observe_runtime(
            project,
            repository,
            result_m6,
            **kwargs,
        )


def test_runtime_catalog_allows_canonical_hardlinked_executables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6

    project, commit = _git_project(tmp_path)
    repository = _repository_preflight(monkeypatch, project, commit)
    kwargs = _runtime_kwargs(
        project, result_m6.WAYMAX_COMMIT, monkeypatch=monkeypatch
    )
    package = project / ".venv/lib/python3.11/site-packages/jax/_src"
    target = package / "shared.py"
    alias = package / "shared_alias.py"

    observed = cli._observe_runtime(
        project,
        repository,
        result_m6,
        **kwargs,
    )

    assert target.stat().st_ino == alias.stat().st_ino
    assert cli._SHA256.fullmatch(observed.runtime_catalog_sha256)


def test_runtime_catalog_rejects_preexisting_recorded_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6

    project, commit = _git_project(tmp_path)
    repository = _repository_preflight(monkeypatch, project, commit)
    kwargs = _runtime_kwargs(
        project, result_m6.WAYMAX_COMMIT, monkeypatch=monkeypatch
    )
    target = project / ".venv/lib/python3.11/site-packages/jax/__init__.py"
    target.write_bytes(target.read_bytes() + b"# tampered before observation\n")

    with pytest.raises(
        cli.M6OfficialCommandError,
        match="RECORD hash or size",
    ):
        cli._observe_runtime(
            project,
            repository,
            result_m6,
            **kwargs,
        )


def test_runtime_catalog_anchor_rejects_coherent_record_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6

    project, commit = _git_project(tmp_path)
    repository = _repository_preflight(monkeypatch, project, commit)
    kwargs = _runtime_kwargs(
        project, result_m6.WAYMAX_COMMIT, monkeypatch=monkeypatch
    )
    resolver = kwargs["distribution_resolver"]
    assert callable(resolver)
    distribution = resolver("jax")
    relative = "jax/_src/core.py"
    target = project / ".venv/lib/python3.11/site-packages" / relative
    target.write_bytes(target.read_bytes() + b"-coherent-tamper")
    replacement = _FakeRecordPath(relative, target.read_bytes())
    entry = next(
        item
        for item in distribution.files
        if os.fspath(item) == relative
    )
    entry.hash = replacement.hash
    entry.size = replacement.size
    record = distribution.dist_info / "RECORD"
    rows = record.read_text(encoding="utf-8").splitlines()
    record.write_text(
        "\n".join(
            f"{relative},sha256={entry.hash.value},{entry.size}"
            if row.startswith(f"{relative},")
            else row
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        cli.M6OfficialCommandError,
        match="immutable catalog anchor",
    ):
        cli._observe_runtime(
            project,
            repository,
            result_m6,
            **kwargs,
        )


def test_runtime_catalog_binds_record_and_direct_url_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6

    project, commit = _git_project(tmp_path)
    repository = _repository_preflight(monkeypatch, project, commit)
    kwargs = _runtime_kwargs(
        project, result_m6.WAYMAX_COMMIT, monkeypatch=monkeypatch
    )
    cli._observe_runtime(
        project,
        repository,
        result_m6,
        **kwargs,
    )
    site_packages = project / ".venv/lib/python3.11/site-packages"
    record = site_packages / "jax.dist-info/RECORD"
    original_record = record.read_bytes()
    record.write_bytes(original_record + b"\n")
    with pytest.raises(
        cli.M6OfficialCommandError,
        match="immutable catalog anchor",
    ):
        cli._observe_runtime(
            project,
            repository,
            result_m6,
            **kwargs,
        )
    record.write_bytes(original_record)

    direct_url = site_packages / "waymo_waymax.dist-info/direct_url.json"
    direct_url.write_bytes(direct_url.read_bytes() + b" ")
    with pytest.raises(
        cli.M6OfficialCommandError,
        match="installed file differs from its RECORD hash or size",
    ):
        cli._observe_runtime(
            project,
            repository,
            result_m6,
            **kwargs,
        )


def test_runtime_observation_retains_same_content_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6

    project, commit = _git_project(tmp_path)
    repository = _repository_preflight(monkeypatch, project, commit)
    kwargs = _runtime_kwargs(
        project, result_m6.WAYMAX_COMMIT, monkeypatch=monkeypatch
    )
    baseline = cli._observe_runtime(
        project,
        repository,
        result_m6,
        **kwargs,
    )
    target = project / ".venv/lib/python3.11/site-packages/jax/__init__.py"
    replacement = tmp_path / "replacement.py"
    replacement.write_bytes(target.read_bytes())
    os.replace(replacement, target)
    changed = cli._observe_runtime(
        project,
        repository,
        result_m6,
        **kwargs,
    )

    assert changed.runtime_catalog_sha256 == baseline.runtime_catalog_sha256
    assert changed.runtime_config_sha256 == baseline.runtime_config_sha256
    assert changed != baseline


def test_runtime_catalog_rejects_module_outside_local_venv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6

    project, commit = _git_project(tmp_path)
    repository = _repository_preflight(monkeypatch, project, commit)
    kwargs = _runtime_kwargs(
        project, result_m6.WAYMAX_COMMIT, monkeypatch=monkeypatch
    )
    importer = kwargs["module_importer"]
    assert callable(importer)
    outside = tmp_path / "flax/__init__.py"
    outside.parent.mkdir(parents=True)
    outside.write_text("# stale flax\n", encoding="utf-8")

    def import_with_outside_flax(name: str):
        if name == "flax":
            return SimpleNamespace(__file__=os.fspath(outside))
        return importer(name)

    kwargs["module_importer"] = import_with_outside_flax
    with pytest.raises(cli.M6OfficialCommandError, match="runtime_mismatch"):
        cli._observe_runtime(
            project,
            repository,
            result_m6,
            **kwargs,
        )


def _minimal_repository(project: Path) -> cli.M6RepositoryPreflight:
    git = cli.M6GitSnapshot(
        commit="a" * 40,
        tree="b" * 40,
        approval_ref=cli._APPROVED_IMPLEMENTATION_REF,
    )
    snapshot = cli._GuardedFileSnapshot(
        relative_path="uv.lock",
        sha256="c" * 64,
        identity=(1,),
    )
    paths = ("uv.lock",)
    context = cli._repository_context_sha256(
        git,
        paths,
        "d" * 64,
        "c" * 64,
    )
    return cli.M6RepositoryPreflight(
        root=project,
        git=git,
        source_paths=paths,
        source_sha256="d" * 64,
        source_snapshots=(snapshot,),
        uv_lock_sha256="c" * 64,
        context_sha256=context,
        _factory_sentinel=cli._PREFLIGHT_SENTINEL,
    )


def _minimal_eligibility_rows() -> tuple[dict[str, object], ...]:
    return tuple({"cohort_index": index} for index in range(128))


def _fake_pilot_selection(*, supported: bool = False):
    rows = tuple(
        SimpleNamespace(
            cohort_index=index,
            rank_sha256=f"{index:064x}",
        )
        for index in range(10)
    )
    return SimpleNamespace(
        supported=supported,
        members=rows[:8] if supported else (),
        qualification_ledger=SimpleNamespace(rows=rows),
    )


def _pilot_summary() -> dict[str, object]:
    return {
        "pilot_scene_n": 8,
        "total_wall_ms": 100,
        "max_scene_ms": 20,
        "decode_ms": 10,
        "numpy_ms": 27,
        "waymax_ms": 1,
        "verification_ms": 10,
        "fresh_worker_peak_rss_bytes": 1024,
        "passed": True,
    }


def _sealed_test_pilot_evidence(
    *,
    selection,
    selection_binding: str,
    provenance,
    summary: dict[str, object] | None = None,
    run_name: str = "m6-test",
):
    pilot_summary = _pilot_summary() if summary is None else dict(summary)
    selected_cohort_indices = tuple(
        member.cohort_index
        for member in (
            selection.members[:8]
            if selection.supported
            else sorted(
                selection.qualification_ledger.rows,
                key=lambda row: (
                    bytes.fromhex(row.rank_sha256),
                    row.cohort_index,
                ),
            )[:8]
        )
    )
    numpy_observation, waymax_observation = _issued_pilot_observations(
        selection_binding,
        supported=selection.supported,
        numpy_max_scene_ms=int(pilot_summary["max_scene_ms"]),
        waymax_max_scene_ms=(
            int(pilot_summary["max_scene_ms"]) if selection.supported else 0
        ),
        waymax_peak_rss_bytes=(
            int(pilot_summary["fresh_worker_peak_rss_bytes"])
            if selection.supported
            else 0
        ),
        selected_cohort_indices=selected_cohort_indices,
    )
    return cli._issue_m6_compute_pilot_evidence(
        eligibility_rows=_minimal_eligibility_rows(),
        selection=selection,
        pilot_summary=pilot_summary,
        pilot_selection_positions=tuple(range(8)),
        numpy_observation=numpy_observation,
        waymax_observation=waymax_observation,
        verified_provenance=provenance,
        run_name=run_name,
        result_path=f"outputs/m6/{run_name}",
        fresh_worker_peak_rss_bytes=int(
            pilot_summary["fresh_worker_peak_rss_bytes"]
        ),
    )


def test_mode_evidence_excludes_eligibility_outcomes_and_bounds_pilot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _minimal_eligibility_rows()
    eligibility = cli.M6ModeExecutionEvidence(
        mode="eligibility_only",
        eligibility_rows=rows,
        selection=object(),
    )
    assert eligibility.numpy_rows is None
    assert eligibility.waymax_evidence is None
    assert eligibility.pilot_summary is None
    with pytest.raises(ValueError, match="no outcomes"):
        cli.M6ModeExecutionEvidence(
            mode="eligibility_only",
            eligibility_rows=rows,
            selection=object(),
            numpy_rows=object(),
        )

    selection = _fake_pilot_selection()
    selection_binding = "7" * 64
    import evalsim.evaluation.m6_waymax_official as waymax_official

    monkeypatch.setattr(
        waymax_official,
        "m6_waymax_selection_binding_sha256",
        lambda actual: (
            selection_binding
            if actual is selection
            else (_ for _ in ()).throw(AssertionError("selection transplant"))
        ),
    )
    provenance = _pilot_verified_provenance()
    with pytest.raises(TypeError, match="runner-issued"):
        cli.M6ModeExecutionEvidence(
            mode="compute_pilot",
            eligibility_rows=rows,
            selection=selection,
            pilot_summary=_pilot_summary(),
            pilot_selection_positions=tuple(range(8)),
            fresh_worker_peak_rss_bytes=1024,
        )
    pilot = _sealed_test_pilot_evidence(
        selection=selection,
        selection_binding=selection_binding,
        provenance=provenance,
    )
    pilot.revalidate_pilot(
        run_name="m6-test",
        result_path="outputs/m6/m6-test",
        selection=selection,
        verified_provenance=provenance,
    )
    assert set(pilot.pilot_summary or ()) == set(_pilot_summary())
    assert len(pilot.pilot_report_binding_sha256 or "") == 64
    assert not hasattr(pilot, "scene_scalars")
    assert not hasattr(pilot, "rollouts")
    with pytest.raises(ValueError, match="mechanically consistent"):
        _sealed_test_pilot_evidence(
            selection=selection,
            selection_binding=selection_binding,
            provenance=provenance,
            summary={**_pilot_summary(), "pilot_scene_n": 9},
        )


class _LifecycleStore:
    def __init__(self, project: Path, run_name: str, events: list[str]) -> None:
        self.project_root = project
        self.run_name = run_name
        self.run_path = project / "outputs/m6" / run_name
        self.project_relative_path = Path("outputs/m6") / run_name
        self.phase = "pending"
        self.events = events

    def write_eligibility_ledger(self, rows) -> None:
        assert len(tuple(rows)) == 128
        self.events.append("eligibility")

    def write_waymax_qualification(self, selection) -> None:
        assert selection is not None
        self.events.append("qualification")

    def write_compute_pilot_summary(self, evidence) -> None:
        assert type(evidence) is cli.M6ModeExecutionEvidence
        assert set(evidence.pilot_summary or ()) == set(_pilot_summary())
        evidence.revalidate_pilot(
            run_name=self.run_name,
            result_path=self.project_relative_path.as_posix(),
            selection=evidence.selection,
            verified_provenance=evidence.pilot_verified_provenance,
        )
        self.events.append("pilot_summary")

    def write_typed_provenance(self, provenance) -> None:
        assert provenance is not None
        self.events.append("provenance")

    def commit(self) -> None:
        self.events.append("commit")
        self.phase = "committed"


@pytest.mark.parametrize(
    ("mode", "expected_artifacts"),
    (
        (
            "eligibility_only",
            ["eligibility", "qualification", "provenance", "commit"],
        ),
        (
            "compute_pilot",
            [
                "eligibility",
                "qualification",
                "pilot_summary",
                "provenance",
                "commit",
            ],
        ),
    ),
)
def test_prepare_orders_pending_before_local_inputs_and_writes_exact_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    expected_artifacts: list[str],
) -> None:
    from evalsim.results import m6 as result_m6

    project = tmp_path / "project"
    project.mkdir()
    request = _request(project, mode=mode)
    repository = _minimal_repository(project)
    events: list[str] = []
    store = _LifecycleStore(project, request.run_name, events)
    local = SimpleNamespace(
        verified_provenance=_pilot_verified_provenance(),
        result_relative=f"outputs/m6/{request.run_name}",
    )
    selection = _fake_pilot_selection()
    selection_binding = "7" * 64
    import evalsim.evaluation.m6_waymax_official as waymax_official

    monkeypatch.setattr(
        waymax_official,
        "m6_waymax_selection_binding_sha256",
        lambda actual: (
            selection_binding
            if actual is selection
            else (_ for _ in ()).throw(AssertionError("selection transplant"))
        ),
    )

    def repository_preflight(*args, **kwargs):
        events.append("repository")
        return repository

    reservation = object()
    holder = cli._RunHolder()

    def create_reservation(actual_request, actual_repository):
        assert actual_request is request
        assert actual_repository is repository
        events.append("pending")
        return reservation

    def adopt_reservation(cls, actual_reservation):
        assert cls is result_m6.M6ResultStore
        assert actual_reservation is reservation
        events.append("adopt")
        return store

    def load_results_module():
        assert holder.store is reservation
        events.append("results_import")
        return result_m6

    def local_preflight(*args, **kwargs):
        assert store.phase == "pending"
        events.append("local")
        return local

    def execute(actual_request, actual_local, preregister, begin_outcomes):
        assert actual_request is request
        assert actual_local is local
        assert store.phase == "pending"
        events.append("execute")
        preregister(_minimal_eligibility_rows(), selection)
        if mode == "compute_pilot":
            begin_outcomes()
        if mode == "compute_pilot":
            return _sealed_test_pilot_evidence(
                selection=selection,
                selection_binding=selection_binding,
                provenance=local.verified_provenance,
            )
        return cli.M6ModeExecutionEvidence(
            mode=mode,
            eligibility_rows=_minimal_eligibility_rows(),
            selection=selection,
        )

    monkeypatch.setattr(cli, "preflight_repository", repository_preflight)
    monkeypatch.setattr(cli, "preflight_local_inputs", local_preflight)
    monkeypatch.setattr(
        cli,
        "_create_m6_pending_reservation",
        create_reservation,
    )
    monkeypatch.setattr(
        result_m6.M6ResultStore,
        "adopt_pending",
        classmethod(adopt_reservation),
    )
    monkeypatch.setattr(
        cli,
        "_load_m6_results_module",
        load_results_module,
    )
    prepared = cli.prepare_m6_official_run(
        request,
        holder,
        eligibility_executor=execute if mode == "eligibility_only" else None,
        pilot_executor=execute if mode == "compute_pilot" else None,
        predecessor_gate_factory=(
            lambda request, local, module: lambda rows, selection: None
        ),
    )
    assert prepared.store is store
    assert holder.store is store
    assert events[:6] == [
        "repository",
        "pending",
        "results_import",
        "adopt",
        "local",
        "execute",
    ]
    assert events[6:] == expected_artifacts


def test_creation_baseexception_after_failure_marker_is_post_store_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from evalsim.results import m6 as result_m6

    project = tmp_path / "project"
    project.mkdir()
    request = _request(project)
    repository = _minimal_repository(project)

    def create_then_interrupt(actual_request, actual_repository):
        assert actual_request is request
        assert actual_repository is repository
        run_path = project / "outputs/m6/m6-test"
        run_path.mkdir(parents=True)
        (run_path / "TERMINAL_FAILURE").write_text(
            "creation_failed",
            encoding="ascii",
        )
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        cli,
        "preflight_repository",
        lambda *args, **kwargs: repository,
    )
    monkeypatch.setattr(
        cli,
        "_create_m6_pending_reservation",
        create_then_interrupt,
    )
    holder = cli._RunHolder()
    with pytest.raises(cli.M6OfficialCommandError, match="result_store_failed"):
        cli.prepare_m6_official_run(
            request,
            holder,
            eligibility_executor=lambda *_: None,
        )
    assert isinstance(holder.store, cli._FailedCreationView)
    assert cli._fail_store(holder.store, "result_store_failed") == (
        "outputs/m6/m6-test/TERMINAL_FAILURE"
    )


def _prepared_terminal_fixture(
    tmp_path: Path,
    events: list[str],
) -> tuple[cli._PreparedOfficialRun, object]:
    project = tmp_path / "project"
    project.mkdir()
    request = _request(project)
    repository = _minimal_repository(project)
    run_path = project / "outputs/m6/m6-test"
    store = SimpleNamespace(
        project_root=project,
        run_name="m6-test",
        run_path=run_path,
        project_relative_path=Path("outputs/m6/m6-test"),
        mark_terminal_success=lambda *, capability: events.append("success"),
    )
    module = SimpleNamespace()
    prepared = cli._PreparedOfficialRun(
        request=request,
        repository=repository,
        local=SimpleNamespace(),
        store=store,
        results_module=module,
        live_lookup=lambda _: "a" * 40,
        live_approval_lookup=lambda _: "a" * 40,
        m4_reverifier=lambda _: None,
        runtime_kwargs={},
        shard_reverifier=lambda _: events.append("shards"),
    )
    verified = SimpleNamespace(run_path=run_path)
    return prepared, verified


def test_terminal_commit_rechecks_everything_and_success_is_last(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    prepared, verified = _prepared_terminal_fixture(tmp_path, events)
    module = prepared.results_module
    monkeypatch.setattr(
        cli,
        "_verify_committed_semantics",
        lambda _: events.append("committed") or verified,
    )
    monkeypatch.setattr(
        cli,
        "_success_result_from_verified",
        lambda *_: events.append("status") or _command_result("eligibility_only"),
    )
    monkeypatch.setattr(
        cli,
        "reverify_repository_preflight",
        lambda *args, **kwargs: events.append("repository") or prepared.repository,
    )
    fresh_local = SimpleNamespace()
    monkeypatch.setattr(
        cli,
        "reverify_local_inputs",
        lambda *args, **kwargs: events.append("local") or fresh_local,
    )
    fresh_provenance = object()
    monkeypatch.setattr(
        cli,
        "reverify_verified_provenance",
        lambda *args, **kwargs: events.append("provenance") or fresh_provenance,
    )
    module._verified_committed_terminal_binding = (
        lambda store: events.append("binding")
        or (verified, "a" * 64, "b" * 64, "c" * 64, "d" * 64)
    )
    module._expected_m6_observed_preflight = (
        lambda **kwargs: events.append("observed") or object()
    )
    capability = object()
    module._mint_m6_terminal_capability = (
        lambda store, observed, provenance: events.append("mint") or capability
    )

    cli._finalize_and_terminalize(prepared)
    assert events == [
        "committed",
        "status",
        "repository",
        "local",
        "shards",
        "provenance",
        "binding",
        "observed",
        "mint",
        "success",
    ]
    assert prepared.success_payload == cli._success_output(
        _command_result("eligibility_only")
    )


def test_stale_final_repository_recheck_cannot_create_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    prepared, verified = _prepared_terminal_fixture(tmp_path, events)
    monkeypatch.setattr(cli, "_verify_committed_semantics", lambda _: verified)
    monkeypatch.setattr(
        cli,
        "_success_result_from_verified",
        lambda *_: _command_result("eligibility_only"),
    )
    monkeypatch.setattr(
        cli,
        "reverify_repository_preflight",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            cli.M6OfficialCommandError(
                "source_binding_failed",
                "stale source",
            )
        ),
    )
    with pytest.raises(cli.M6OfficialCommandError, match="source_binding_failed"):
        cli._finalize_and_terminalize(prepared)
    assert "success" not in events


def test_official_committed_semantics_allow_derived_waymax_unsupported(
    tmp_path: Path,
) -> None:
    run_path = tmp_path / "outputs/m6/m6-test"
    execution = [
        {
            "deterministic_repeat_status": "passed",
            "waymax_gate_status": "unsupported",
            "release_gate_status": "accepted",
        }
    ]
    verified = SimpleNamespace(
        tables={},
        read_dataset=lambda name: SimpleNamespace(
            to_pylist=lambda: execution
        ),
    )
    module = SimpleNamespace(
        ELIGIBILITY_LEDGER="eligibility",
        WAYMAX_QUALIFICATION="qualification",
        TYPED_PROVENANCE="provenance",
        COMPUTE_PILOT_SUMMARY="pilot",
        EXECUTION_SUMMARY="execution",
        verify_committed_m6_result_store=lambda *args, **kwargs: verified,
    )
    prepared = SimpleNamespace(
        results_module=module,
        store=SimpleNamespace(
            project_root=tmp_path,
            run_name="m6-test",
        ),
        request=SimpleNamespace(mode="official"),
    )
    assert cli._verify_committed_semantics(prepared) is verified


@pytest.mark.parametrize(
    ("mode", "extra_count_keys"),
    (
        ("eligibility_only", set()),
        ("compute_pilot", {"pilot_scene_n"}),
        (
            "official",
            {
                "secondary_b4_feasible_n",
                "waymax_qualified_n",
                "waymax_selected_n",
            },
        ),
    ),
)
def test_success_counts_are_exactly_mode_scoped(
    tmp_path: Path,
    mode: str,
    extra_count_keys: set[str],
) -> None:
    from evalsim.results import m6 as result_m6

    rejection_counts = {
        reason: index
        for index, reason in enumerate(result_m6.M6_PRIMARY_REJECTION_REASONS)
    }
    receipt = SimpleNamespace(
        population_size=128,
        eligible_count=10,
        secondary_b4_count=4,
        rejection_reason_counts=rejection_counts,
    )
    pilot = {
        **_pilot_summary(),
        "pilot_scene_n": 4,
    }
    qualification = [
        {"assessment_status": "qualified", "selected": True},
        {"assessment_status": "rejected", "selected": False},
    ]
    stages = [
        {"stage_name": stage, "duration_ms": index + 1}
        for index, stage in enumerate(result_m6.M6_STAGE_DOMAIN)
    ]

    def read_dataset(name: str):
        rows = {
            result_m6.COMPUTE_PILOT_SUMMARY: [pilot],
            result_m6.WAYMAX_QUALIFICATION: qualification,
            result_m6.STAGE_TIMINGS: stages,
        }[name]
        return SimpleNamespace(to_pylist=lambda: rows)

    verified = SimpleNamespace(receipt=receipt, read_dataset=read_dataset)
    prepared = SimpleNamespace(
        results_module=result_m6,
        request=SimpleNamespace(mode=mode),
        store=SimpleNamespace(
            project_relative_path=Path("outputs/m6/m6-test")
        ),
        eligibility_duration_ms=1,
    )
    result = cli._success_result_from_verified(prepared, verified)
    base = {
        "population_n",
        "primary_eligible_n",
        *{
            f"primary_rejection_{reason}_n"
            for reason in result_m6.M6_PRIMARY_REJECTION_REASONS
        },
    }
    assert set(result.aggregate_counts) == base | extra_count_keys
    if mode != "official":
        assert not any(
            key.startswith(("secondary_", "waymax_"))
            for key in result.aggregate_counts
        )


def test_final_shard_recheck_detects_content_drift_with_same_identity(
    tmp_path: Path,
) -> None:
    paths = []
    for index in range(10):
        path = tmp_path / f"validation-tfrecord-{index:05d}-of-00150"
        path.write_bytes(b"same-identity-fixture")
        paths.append(path)
    identities = cli._shard_identities(paths)
    events = tuple(
        SimpleNamespace(
            shard_suffix=f"{index:05d}",
            shard_sha256="a" * 64,
        )
        for index in range(10)
    )
    local = object.__new__(cli.M6LocalInputPreflight)
    object.__setattr__(local, "_factory_sentinel", cli._PREFLIGHT_SENTINEL)
    object.__setattr__(
        local,
        "accepted_m4",
        SimpleNamespace(manifest=SimpleNamespace(events=events)),
    )
    object.__setattr__(local, "shard_paths", tuple(paths))
    object.__setattr__(local, "shard_identities", identities)
    cleared: list[bool] = []

    def changed_digest(path: Path) -> str:
        assert path in paths
        return "b" * 64 if path == paths[3] else "a" * 64

    with pytest.raises(cli.M6OfficialCommandError, match="shard_set_invalid"):
        cli._reverify_accepted_shards(
            local,
            digest_resolver=changed_digest,
            cache_clearer=lambda: cleared.append(True),
        )
    assert cleared == [True]
    assert cli._shard_identities(paths) == identities


class _FailureStore:
    def __init__(self, project: Path) -> None:
        self.project_root = project
        self.run_name = "m6-test"
        self.run_path = project / "outputs/m6/m6-test"
        self.run_path.mkdir(parents=True)

    def fail(self, code: str) -> Path:
        marker = self.run_path / "TERMINAL_FAILURE"
        marker.write_text(code, encoding="ascii")
        return marker


def test_post_pending_baseexception_writes_failure_before_0600_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    store = _FailureStore(tmp_path)
    original_diagnostic = cli._persist_failure_diagnostic

    def dispatch(actual, holder):
        assert actual is request
        holder.store = store
        raise KeyboardInterrupt()

    def diagnostic(*args, **kwargs):
        assert (store.run_path / "TERMINAL_FAILURE").is_file()
        return original_diagnostic(*args, **kwargs)

    monkeypatch.setattr(cli, "_parse_request", lambda _: request)
    monkeypatch.setattr(cli, "_dispatch", dispatch)
    monkeypatch.setattr(cli, "_persist_failure_diagnostic", diagnostic)
    assert cli.main(
        [], _bootstrap_context=_test_bootstrap_context(tmp_path)
    ) == 1
    captured = capfd.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["status"] == "failure"
    assert payload["reason_code"] == "unexpected_failure"
    assert payload["failure_marker"] == (
        "outputs/m6/m6-test/TERMINAL_FAILURE"
    )
    diagnostic_path = store.run_path / cli._FAILURE_DIAGNOSTIC_NAME
    assert diagnostic_path.is_file()
    assert stat.S_IMODE(diagnostic_path.stat().st_mode) == 0o600
    assert diagnostic_path.stat().st_size <= cli._MAX_FAILURE_DIAGNOSTIC_BYTES
    assert b"sanitized traceback" in diagnostic_path.read_bytes()


def test_post_pending_native_output_is_local_only_failure(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    store = _FailureStore(tmp_path)

    def dispatch(actual, holder):
        holder.store = store
        os.write(1, b"private-native-sentinel")
        return _command_result(actual.mode)

    monkeypatch.setattr(cli, "_parse_request", lambda _: request)
    monkeypatch.setattr(cli, "_dispatch", dispatch)
    assert cli.main(
        [], _bootstrap_context=_test_bootstrap_context(tmp_path)
    ) == 1
    captured = capfd.readouterr()
    assert captured.out == ""
    assert "private-native-sentinel" not in captured.err
    payload = json.loads(captured.err)
    assert payload["status"] == "failure"
    assert payload["reason_code"] == "terminal_output_detected"
    diagnostic = (store.run_path / cli._FAILURE_DIAGNOSTIC_NAME).read_bytes()
    assert b"private-native-sentinel" in diagnostic


def test_post_outcome_native_output_is_redacted_from_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, mode="compute_pilot")
    store = _FailureStore(tmp_path)
    secret = b"private-policy-value-sentinel"

    def dispatch(actual, holder):
        assert actual is request
        holder.store = store
        holder.outcome_started = True
        os.write(1, secret)
        return _command_result(actual.mode)

    monkeypatch.setattr(cli, "_parse_request", lambda _: request)
    monkeypatch.setattr(cli, "_dispatch", dispatch)
    assert cli.main(
        [], _bootstrap_context=_test_bootstrap_context(tmp_path)
    ) == 1
    captured = capfd.readouterr()
    assert captured.out == ""
    assert secret.decode("ascii") not in captured.err
    payload = json.loads(captured.err)
    assert payload["reason_code"] == "terminal_output_detected"
    diagnostic = (store.run_path / cli._FAILURE_DIAGNOSTIC_NAME).read_bytes()
    assert secret not in diagnostic
    assert b"post-outcome transcript redacted" in diagnostic
    assert b"captured_byte_count=29" in diagnostic


def test_status_write_failure_never_returns_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    store = _FailureStore(tmp_path)

    def dispatch(actual, holder):
        assert actual is request
        holder.store = store
        return cli._AwaitingReviewResult(
            mode="official",
            result_relative=Path("outputs/m6/m6-test"),
            evidence_catalog_sha256="a" * 64,
            mechanical_verification_sha256="b" * 64,
        )

    monkeypatch.setattr(cli, "_parse_request", lambda _: request)
    monkeypatch.setattr(cli, "_dispatch", dispatch)
    monkeypatch.setattr(cli, "_emit_status", lambda *args, **kwargs: False)
    assert cli.main(
        [], _bootstrap_context=_test_bootstrap_context(tmp_path)
    ) == 1
    assert (store.run_path / "TERMINAL_FAILURE").read_text(
        encoding="ascii"
    ) == "terminal_capture_failed"


@pytest.mark.parametrize(
    ("failure_mode", "expected_reason"),
    (
        ("descriptor", "terminal_capture_failed"),
        ("outer_flush", "terminal_output_detected"),
    ),
)
def test_outer_bootstrap_failure_invalidates_registered_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_mode: str,
    expected_reason: str,
) -> None:
    from evalsim.cli import m6_bootstrap as bootstrap

    store = _FailureStore(tmp_path)
    script = tmp_path / "evalsim/cli/m6_bootstrap.py"
    site_packages = tmp_path / ".venv/lib/python3.12/site-packages"
    payload = bootstrap._canonical_json(
        {
            "mode": "official",
            "profile": "official_m6",
            "result_path": "outputs/m6/m6-test",
            "schema_version": "m6-cli-status-2.0.0",
            "status": "awaiting_review",
        }
    )

    def run_captured(
        root,
        site,
        initial,
        *,
        status_failure_callbacks,
    ):
        del root, site, initial

        def invalidate(reason_code: str) -> bool:
            store.fail(reason_code)
            return True

        status_failure_callbacks.append(invalidate)
        return 0, payload, False, False

    flush_calls = 0

    def flush_all() -> None:
        nonlocal flush_calls
        flush_calls += 1
        if failure_mode == "outer_flush" and flush_calls == 2:
            raise OSError("outer flush failed")

    monkeypatch.setattr(
        bootstrap,
        "_validated_invocation",
        lambda: (tmp_path, script, site_packages, ("/stdlib",)),
    )
    monkeypatch.setattr(bootstrap, "_run_captured", run_captured)
    monkeypatch.setattr(bootstrap, "_flush_all", flush_all)
    monkeypatch.setattr(
        bootstrap,
        "_write_all",
        lambda descriptor, value: failure_mode != "descriptor",
    )

    assert bootstrap._bootstrap() == 1
    assert (store.run_path / "TERMINAL_FAILURE").read_text(
        encoding="ascii"
    ) == expected_reason


def test_failure_diagnostic_is_bounded_and_explicitly_truncated(
    tmp_path: Path,
) -> None:
    store = _FailureStore(tmp_path)
    store.fail("unexpected_failure")
    try:
        raise RuntimeError("private message is deliberately omitted")
    except RuntimeError as exc:
        cli._persist_failure_diagnostic(
            store,
            "unexpected_failure",
            exc,
            b"x" * (cli._MAX_FAILURE_DIAGNOSTIC_BYTES + 1024),
        )
    path = store.run_path / cli._FAILURE_DIAGNOSTIC_NAME
    payload = path.read_bytes()
    assert len(payload) <= cli._MAX_FAILURE_DIAGNOSTIC_BYTES
    assert payload.endswith(b"...[diagnostic truncated]...\n")
    assert b"private message" not in payload
    assert b"sanitized traceback" in payload
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
