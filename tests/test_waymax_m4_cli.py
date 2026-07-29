"""Data-free safety and orchestration tests for the local M4 command."""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace
import warnings

import numpy as np
import pytest

from evalsim.contracts import Agent, AgentType, Rollout, Scenario
from evalsim.sources import waymax_m4_cli as cli
from evalsim.sources.waymax_cohort import ScanEvent
from evalsim.sources.waymax_loader import (
    M4ShardLocator,
    M4StreamRecord,
    WaymaxRecord,
)


def _spawn_terminal_writer() -> None:
    warnings.warn(
        "spawn-python-warning-" + "a" * 64,
        RuntimeWarning,
        stacklevel=1,
    )
    os.write(1, b"spawn-stdout-absolute-/private/invented\n")
    os.write(2, b"spawn-stderr-identity-InventedPrivate123\n")


def _stream_record(identity: str = "invented-identity") -> M4StreamRecord:
    record = WaymaxRecord(
        scenario_id=identity,
        state=object(),
        audit=MappingProxyType({"invented": np.asarray([1])}),
        shard_suffix="00000",
        record_ordinal=0,
        shard_sha256="a" * 64,
        dataset_config_fingerprint="b" * 64,
    )
    return M4StreamRecord(
        locator=M4ShardLocator("00000", 0),
        record=record,
    )


def test_help_requires_no_optional_runtime_or_checkout(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli,
        "run_acceptance",
        lambda args: (_ for _ in ()).throw(
            AssertionError("help must not execute")
        ),
    )
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
    output = capsys.readouterr().out
    assert "--project-root" in output
    assert "--data-dir" in output
    assert "--output-dir" in output
    assert "00000 through 00009" in output


def test_argument_errors_do_not_echo_paths_or_values(capsys) -> None:
    secret = "/private/local/path/NativeScenario-Private-123"
    with pytest.raises(SystemExit) as caught:
        cli.main(["--unknown", secret])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert secret not in captured.err
    assert "argument_error" in captured.err


def test_local_opt_in_fails_before_project_or_data_inspection(
    monkeypatch,
) -> None:
    monkeypatch.delenv("EVALSIM_RUN_WAYMO_LOCAL", raising=False)
    monkeypatch.setattr(
        cli,
        "_project_root",
        lambda _: (_ for _ in ()).throw(
            AssertionError("project inspection must not run")
        ),
    )
    args = argparse.Namespace(
        project_root=Path("/not/inspected"),
        data_dir=Path("/not/inspected"),
        output_dir=Path("/not/inspected"),
    )
    with pytest.raises(cli.M4CommandError, match="local_opt_in_required"):
        cli.run_acceptance(args)


def test_dirty_tree_fails_before_output_creation_or_shard_resolution(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("EVALSIM_RUN_WAYMO_LOCAL", "1")
    monkeypatch.setattr(cli, "_project_root", lambda _: tmp_path)
    monkeypatch.setattr(cli, "_assert_running_checkout", lambda _: None)
    monkeypatch.setattr(
        cli,
        "_assert_clean_worktree",
        lambda _: (_ for _ in ()).throw(
            cli.M4CommandError("dirty_worktree", "invented")
        ),
    )
    monkeypatch.setattr(
        cli,
        "_prepare_output_directory",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("output creation must not run")
        ),
    )
    monkeypatch.setattr(
        cli,
        "resolve_m4_validation_shards",
        lambda _: (_ for _ in ()).throw(
            AssertionError("data resolution must not run")
        ),
    )
    args = argparse.Namespace(
        project_root=tmp_path,
        data_dir=tmp_path,
        output_dir=Path("outputs/m4/invented"),
    )
    with pytest.raises(cli.M4CommandError, match="dirty_worktree"):
        cli.run_acceptance(args)


def test_output_scope_and_existing_directory_fail_closed(
    monkeypatch,
    tmp_path,
) -> None:
    root = tmp_path.resolve()
    data = root / "data" / "validation"
    data.mkdir(parents=True)
    monkeypatch.setattr(cli, "_is_git_ignored", lambda *_: True)

    with pytest.raises(cli.M4CommandError, match="output_scope_invalid"):
        cli._prepare_output_directory(
            Path("outputs/m3/not-m4"),
            root=root,
            data_dir=data,
        )
    with pytest.raises(cli.M4CommandError, match="output_scope_invalid"):
        cli._prepare_output_directory(
            Path("outputs/m4"),
            root=root,
            data_dir=data,
        )

    with pytest.raises(cli.M4CommandError, match="output_name_invalid"):
        cli._prepare_output_directory(
            Path("outputs/m4/not\nsafe"),
            root=root,
            data_dir=data,
        )

    existing = root / "outputs" / "m4" / "existing"
    existing.mkdir(parents=True)
    with pytest.raises(cli.M4CommandError, match="output_exists"):
        cli._prepare_output_directory(
            existing,
            root=root,
            data_dir=data,
        )


def test_output_must_be_ignored_before_it_is_created(
    monkeypatch,
    tmp_path,
) -> None:
    root = tmp_path.resolve()
    data = root / "data" / "validation"
    data.mkdir(parents=True)
    target = root / "outputs" / "m4" / "invented"
    monkeypatch.setattr(cli, "_is_git_ignored", lambda *_: False)

    with pytest.raises(cli.M4CommandError, match="output_not_ignored"):
        cli._prepare_output_directory(
            target,
            root=root,
            data_dir=data,
        )
    assert not target.exists()


def test_json_artifacts_publish_atomically_and_never_overwrite(tmp_path) -> None:
    path = tmp_path / "evidence.json"
    cli._write_json_exclusive(path, {"accepted": False, "value": 1})
    assert path.read_text(encoding="utf-8") == (
        '{\n  "accepted": false,\n  "value": 1\n}\n'
    )
    with pytest.raises(cli.M4CommandError, match="artifact_exists"):
        cli._write_json_exclusive(path, {"accepted": True})
    assert '"accepted": false' in path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.pending"))


def test_output_symlink_cannot_escape_checkout(tmp_path) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    data = root / "data" / "validation"
    data.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "outputs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(cli.M4CommandError, match="output_root_escape"):
        cli._prepare_output_directory(
            Path("outputs/m4/invented"),
            root=root.resolve(),
            data_dir=data.resolve(),
        )


def test_running_modules_must_come_from_bound_checkout(
    monkeypatch,
    tmp_path,
) -> None:
    root = Path.cwd().resolve()
    cli._assert_running_checkout(root)
    monkeypatch.setattr(
        cli,
        "_IMPORTED_MODULE_PATHS",
        {
            "evalsim.sources.waymax_m4_cli": (
                "evalsim/sources/waymax_m4_cli.py"
            )
        },
    )
    other = tmp_path / "different-waymax-m4-cli.py"
    other.write_text("# invented alternate checkout\n", encoding="utf-8")
    monkeypatch.setattr(cli, "__file__", str(other))
    with pytest.raises(cli.M4CommandError, match="import_checkout_mismatch"):
        cli._assert_running_checkout(root)


def test_eligible_scan_event_converts_then_runs_independent_parity(
    monkeypatch,
) -> None:
    item = _stream_record()
    scenario = object()
    calls: list[str] = []

    def source_check(audit):
        assert audit is item.record.audit
        calls.append("source")
        return None

    def convert(state, *, scenario_id, provenance):
        assert state is item.record.state
        assert scenario_id == item.record.scenario_id
        assert provenance == {
            "shard_suffix": "00000",
            "record_ordinal": 0,
            "shard_sha256": "a" * 64,
            "dataset_config_fingerprint": "b" * 64,
        }
        calls.append("convert")
        return scenario

    def parity(record, converted):
        assert record is item.record
        assert converted is scenario
        calls.append("parity")
        return MappingProxyType({"invented": True})

    monkeypatch.setattr(cli, "source_rejection_code", source_check)
    monkeypatch.setattr(cli, "scenario_from_waymax_state", convert)
    monkeypatch.setattr(cli, "validate_record_parity", parity)

    event, converted = cli._classify_stream_record(item)

    assert calls == ["source", "convert", "parity"]
    assert converted is scenario
    assert event == ScanEvent.eligible_event(
        shard_suffix="00000",
        record_ordinal=0,
        native_scenario_id="invented-identity",
        shard_sha256="a" * 64,
        dataset_config_fingerprint="b" * 64,
    )


def test_source_rejection_never_reaches_adapter_or_parity(
    monkeypatch,
) -> None:
    item = _stream_record()
    monkeypatch.setattr(
        cli,
        "source_rejection_code",
        lambda _: "source_no_supported_map",
    )
    monkeypatch.setattr(
        cli,
        "scenario_from_waymax_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("rejected records must not be converted")
        ),
    )
    monkeypatch.setattr(
        cli,
        "validate_record_parity",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("rejected records must not reach parity")
        ),
    )

    event, scenario = cli._classify_stream_record(item)

    assert scenario is None
    assert event.outcome == "rejected"
    assert event.rejection_code == "source_no_supported_map"


def test_adapter_or_parity_failures_are_fatal_not_rejections(
    monkeypatch,
) -> None:
    item = _stream_record()
    monkeypatch.setattr(cli, "source_rejection_code", lambda _: None)
    monkeypatch.setattr(
        cli,
        "scenario_from_waymax_state",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        cli,
        "validate_record_parity",
        lambda *args, **kwargs: MappingProxyType({"invented": False}),
    )

    with pytest.raises(cli.M4CommandError, match="adapter_parity_false"):
        cli._classify_stream_record(item)

    monkeypatch.setattr(
        cli,
        "scenario_from_waymax_state",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("invented adapter failure")
        ),
    )
    with pytest.raises(RuntimeError, match="invented adapter failure"):
        cli._classify_stream_record(item)


def test_population_scan_consumes_only_loader_verified_events(
    monkeypatch,
    tmp_path,
) -> None:
    class FakeCounters:
        def __init__(self, suffix):
            self.shard_suffix = suffix
            self.raw_seen = 0
            self.decode_attempted = 0
            self.event_emitted = 0
            self.clean_eof = False

    calls: list[str] = []

    def verified_iterator(path, *, counters):
        # No event_factory parameter is accepted: passing one would make this test
        # fail and would reopen the caller-trust boundary.
        suffix = path.name
        calls.append(suffix)
        for ordinal in range(4):
            yield ScanEvent.eligible_event(
                shard_suffix=suffix,
                record_ordinal=ordinal,
                native_scenario_id=f"invented-{suffix}-{ordinal}",
                shard_sha256="a" * 64,
                dataset_config_fingerprint="b" * 64,
            )
        counters.raw_seen = 4
        counters.decode_attempted = 4
        counters.event_emitted = 4
        counters.clean_eof = True

    monkeypatch.setattr(cli, "M4StreamCounters", FakeCounters)
    monkeypatch.setattr(cli, "iter_m4_waymax_records", verified_iterator)
    paths = tuple(tmp_path / suffix for suffix in cli.M4_SHARD_SUFFIXES)

    manifest = cli._scan_population(paths)

    assert calls == list(cli.M4_SHARD_SUFFIXES)
    assert manifest.selection.fallback_used is True
    assert len(manifest.selected_events) == 40


def test_grouped_selected_reload_reconverts_and_detects_drift(
    monkeypatch,
    tmp_path,
) -> None:
    source = ScanEvent.eligible_event(
        shard_suffix="00000",
        record_ordinal=0,
        native_scenario_id="aa",
        shard_sha256="a" * 64,
        dataset_config_fingerprint="b" * 64,
    )
    selected = dataclasses.replace(source, selected=True)
    manifest = SimpleNamespace(selected_events=(selected,))
    stream = _stream_record("aa")
    observed_expectations = []

    def grouped_reload(data_dir, expectations):
        assert data_dir == tmp_path
        observed_expectations.extend(expectations)
        return (stream,)

    monkeypatch.setattr(cli, "reload_m4_waymax_records", grouped_reload)
    monkeypatch.setattr(
        cli,
        "_classify_stream_record",
        lambda _: (source, object()),
    )
    visited = []
    cli._visit_selected_records(
        data_dir=tmp_path,
        manifest=manifest,
        selected_events=(selected,),
        visitor=lambda event, record, scenario: visited.append(
            (event, record, scenario)
        ),
    )
    assert len(observed_expectations) == 1
    assert observed_expectations[0].locator == M4ShardLocator("00000", 0)
    assert visited[0][0] == selected
    assert visited[0][1] is stream.record

    contradicted = ScanEvent.eligible_event(
        shard_suffix="00000",
        record_ordinal=0,
        native_scenario_id="bb",
        shard_sha256="a" * 64,
        dataset_config_fingerprint="b" * 64,
    )
    monkeypatch.setattr(
        cli,
        "_classify_stream_record",
        lambda _: (contradicted, object()),
    )
    with pytest.raises(cli.M4CommandError, match="selected_reload_mismatch"):
        cli._visit_selected_records(
            data_dir=tmp_path,
            manifest=manifest,
            selected_events=(selected,),
            visitor=lambda *args: None,
        )


def _invented_idm_inputs():
    objects = 128
    frames = 91
    valid = np.zeros((objects, frames), dtype=bool)
    valid[0] = True
    valid[1] = True
    ids = np.full(objects, -1, dtype=np.int64)
    ids[:3] = (101, 102, 103)
    object_types = np.zeros(objects, dtype=np.int32)
    object_types[:3] = 1
    is_sdc = np.zeros(objects, dtype=bool)
    is_sdc[0] = True
    zeros = np.zeros((objects, frames), dtype=np.float32)
    timestamps = np.broadcast_to(
        np.arange(frames, dtype=np.int64) * 100_000,
        (objects, frames),
    ).copy()
    state = SimpleNamespace(
        object_metadata=SimpleNamespace(
            ids=ids,
            object_types=object_types,
            is_sdc=is_sdc,
        ),
        log_trajectory=SimpleNamespace(
            valid=valid,
            x=zeros,
            y=zeros,
            yaw=zeros,
            vel_x=zeros,
            vel_y=zeros,
            timestamp_micros=timestamps,
        ),
    )
    requested = np.zeros((20, objects), dtype=bool)
    requested[:, 1] = True
    effective = requested.copy()
    lifecycle = np.zeros_like(requested)
    lifecycle[:, 2] = True
    overlap = np.zeros_like(requested)
    compact_valid = valid[:, 11:31].T
    compact = cli.CompactWaymaxIDMRollout(
        x=np.zeros((20, objects), dtype=np.float32),
        y=np.zeros((20, objects), dtype=np.float32),
        yaw=np.zeros((20, objects), dtype=np.float32),
        vx=np.zeros((20, objects), dtype=np.float32),
        vy=np.zeros((20, objects), dtype=np.float32),
        valid=compact_valid,
        timestamp_micros=timestamps[:, 11:31].T,
        timestep=np.arange(11, 31, dtype=np.int32),
        requested_control=requested,
        effective_control=effective,
        lifecycle_fallback=lifecycle,
        initialized_overlap_excluded=overlap,
    )
    qualifying = np.zeros(objects, dtype=bool)
    qualifying[1] = True
    qualification = cli._IDMQualification(
        qualifying_vehicle_mask=qualifying,
        initialized_overlap_mask=np.zeros(objects, dtype=bool),
        initialized_overlap_vehicle_exclusions=0,
    )
    return state, compact, qualification


def test_idm_accounting_requires_the_qualifying_actor_for_all_20_steps() -> None:
    state, compact, qualification = _invented_idm_inputs()
    accounting = cli._independent_idm_accounting(
        compact,
        state=state,
        qualification=qualification,
    )
    assert accounting["qualifying_vehicle_effective_transitions"] == 20

    values = list(compact)
    contradicted_effective = np.array(compact.effective_control, copy=True)
    contradicted_effective[:, 1] = False
    contradicted_effective[:, 3] = True  # Preserve the misleading total count.
    values[9] = contradicted_effective
    contradicted = cli.CompactWaymaxIDMRollout(*values)
    with pytest.raises(cli.M4CommandError, match="idm_control_mask_drift"):
        cli._independent_idm_accounting(
            contradicted,
            state=state,
            qualification=qualification,
        )

    lifecycle_values = list(compact)
    contradicted_valid = np.array(compact.valid, copy=True)
    contradicted_valid[0, 1] = False
    lifecycle_values[5] = contradicted_valid
    lifecycle_contradiction = cli.CompactWaymaxIDMRollout(
        *lifecycle_values
    )
    with pytest.raises(cli.M4CommandError, match="idm_lifecycle_drift"):
        cli._independent_idm_accounting(
            lifecycle_contradiction,
            state=state,
            qualification=qualification,
        )


def _invented_conversion_inputs():
    objects = 128
    frames = 91
    valid = np.zeros((objects, frames), dtype=bool)
    valid[:2] = True
    timestamps_micros = np.broadcast_to(
        np.arange(frames, dtype=np.int64) * 100_000,
        (objects, frames),
    ).copy()
    zeros = np.zeros((objects, frames), dtype=np.float32)
    ids = np.full(objects, -1, dtype=np.int64)
    ids[:2] = (101, 102)
    state = SimpleNamespace(
        object_metadata=SimpleNamespace(ids=ids),
        log_trajectory=SimpleNamespace(
            valid=valid,
            timestamp_micros=timestamps_micros,
        ),
    )
    agents = [
        Agent(
            id=101 + index,
            type=AgentType.VEHICLE,
            valid=np.ones(frames, dtype=bool),
            x=np.zeros(frames),
            y=np.zeros(frames),
            heading=np.zeros(frames),
            vx=np.zeros(frames),
            vy=np.zeros(frames),
            length=4.0,
            width=2.0,
        )
        for index in range(2)
    ]
    scenario = Scenario(
        scenario_id="invented",
        timestamps=np.arange(frames, dtype=float) * 0.1,
        agents=agents,
        ego_index=0,
        metadata={
            "current_index": 10,
            "source": "invented",
            "source_fingerprint": "invented-fingerprint",
        },
    )
    compact = cli.CompactWaymaxRollout(
        x=zeros[:, 11:31].T,
        y=zeros[:, 11:31].T,
        yaw=zeros[:, 11:31].T,
        vx=zeros[:, 11:31].T,
        vy=zeros[:, 11:31].T,
        valid=valid[:, 11:31].T,
        timestamp_micros=timestamps_micros[:, 11:31].T,
        timestep=np.arange(11, 31, dtype=np.int32),
    )
    metadata = {
        "backend": "waymax",
        "backend_commit": cli.WAYMAX_COMMIT,
        "compact_reference_version": cli.WAYMAX_REFERENCE_VERSION,
        "control_accounting": {},
        "horizon_transitions": 20,
        "init_steps": 11,
        "invalid_fill": "finite_zero_where_invalid",
        "reference_config_fingerprint": cli.reference_config_fingerprint(),
        "rollout_start_index": 10,
        "scenario_source": "invented",
        "scenario_source_fingerprint": "invented-fingerprint",
        "time_source": "direct_waymax_emission_checked_against_log",
    }

    def make_rollout():
        return Rollout(
            scenario_id="invented",
            sim_name=cli.WAYMAX_EXACT_LOG_NAME,
            sim_version=cli.WAYMAX_REFERENCE_VERSION,
            seed=cli.M4_RANDOM_SEED,
            timestamps=np.arange(31, dtype=float) * 0.1,
            agents=[
                dataclasses.replace(
                    agent,
                    valid=np.ones(31, dtype=bool),
                    x=np.zeros(31),
                    y=np.zeros(31),
                    heading=np.zeros(31),
                    vx=np.zeros(31),
                    vy=np.zeros(31),
                )
                for agent in agents
            ],
            perturbation=None,
            metadata=dict(metadata),
        )

    return state, scenario, compact, make_rollout


def test_waymax_conversion_audit_rejects_identity_history_time_and_metadata() -> None:
    state, scenario, compact, make_rollout = _invented_conversion_inputs()

    def check(rollout, compact_value=compact):
        cli._assert_waymax_conversion_mapping(
            rollout,
            compact=compact_value,
            state=state,
            scenario=scenario,
            expected_sim_name=cli.WAYMAX_EXACT_LOG_NAME,
            expected_control_accounting={},
        )

    check(make_rollout())

    contradictions = []
    wrong_id = make_rollout()
    wrong_id.agents[0].id = 999
    contradictions.append(wrong_id)
    changed_history = make_rollout()
    changed_history.agents[0].x[0] = 1.0
    contradictions.append(changed_history)
    changed_time = make_rollout()
    changed_time.timestamps[12] += 1.0
    contradictions.append(changed_time)
    changed_valid = make_rollout()
    changed_valid.agents[0].valid[12] = False
    contradictions.append(changed_valid)
    changed_future = make_rollout()
    changed_future.agents[0].x[12] = 1.0
    contradictions.append(changed_future)
    changed_metadata = make_rollout()
    changed_metadata.metadata = {}
    contradictions.append(changed_metadata)

    for contradicted in contradictions:
        with pytest.raises(cli.M4CommandError):
            check(contradicted)

    # Matching a Rollout to a contradicted compact mask is insufficient: compact
    # lifecycle itself must independently agree with the source and Scenario.
    compact_values = list(compact)
    contradicted_compact_valid = np.array(compact.valid, copy=True)
    contradicted_compact_valid[0, 0] = False
    compact_values[5] = contradicted_compact_valid
    contradicted_compact = cli.CompactWaymaxRollout(*compact_values)
    derived_rollout = make_rollout()
    derived_rollout.agents[0].valid[11] = False
    with pytest.raises(
        cli.M4CommandError,
        match="waymax_conversion_lifecycle",
    ):
        check(derived_rollout, contradicted_compact)


def test_untrusted_fresh_worker_code_is_collapsed(
    monkeypatch,
) -> None:
    message = {"ok": False, "code": "NativeScenario-Private-123"}

    class Receive:
        def poll(self, timeout):
            assert timeout == cli.M4_BENCHMARK_TIMEOUT_SECONDS
            return True

        def recv(self):
            return message

        def close(self):
            return None

    class Send:
        def close(self):
            return None

    class Process:
        exitcode = 0

        def start(self):
            return None

        def join(self, timeout):
            return None

        def is_alive(self):
            return False

    context = SimpleNamespace(
        Pipe=lambda duplex: (Receive(), Send()),
        Process=lambda **kwargs: Process(),
    )
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(device_get=lambda value: value),
    )
    monkeypatch.setattr(
        cli.multiprocessing,
        "get_context",
        lambda mode: context,
    )
    with pytest.raises(cli.M4CommandError, match="benchmark_worker_failure"):
        cli._fresh_worker_benchmark((np.asarray([1]), np.asarray([2])))


def test_fresh_worker_protocol_uses_spawn_and_returns_only_report(
    monkeypatch,
) -> None:
    report = {
        "fresh_worker_process": True,
        "jit_vmap": True,
        "runs": 20,
    }
    observed = {}

    class Receive:
        def poll(self, timeout):
            return True

        def recv(self):
            return {"ok": True, "report": report}

        def close(self):
            return None

    class Send:
        def close(self):
            return None

    class Process:
        exitcode = 0

        def start(self):
            observed["started"] = True

        def join(self, timeout):
            return None

        def is_alive(self):
            return False

    class Context:
        def Pipe(self, *, duplex):
            assert duplex is False
            return Receive(), Send()

        def Process(self, **kwargs):
            observed.update(kwargs)
            return Process()

    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(device_get=lambda value: value),
    )

    def context(mode):
        observed["mode"] = mode
        return Context()

    monkeypatch.setattr(cli.multiprocessing, "get_context", context)
    actual = cli._fresh_worker_benchmark(
        (np.asarray([1]), np.asarray([2]))
    )
    assert actual == report
    assert observed["mode"] == "spawn"
    assert observed["target"] is cli._benchmark_worker
    assert observed["name"] == "evalsim-m4-vmap-benchmark"
    assert observed["started"] is True


@pytest.mark.parametrize(
    ("mode", "expected_code"),
    [
        ("normal", None),
        ("timeout", "benchmark_timeout"),
        ("eof", "benchmark_worker_eof"),
        ("exception", None),
    ],
)
def test_fresh_worker_all_post_start_paths_close_and_reap(
    monkeypatch,
    mode,
    expected_code,
) -> None:
    report = {"fresh_worker_process": True}
    arbitrary_failure = RuntimeError("synthetic receive failure")
    observed: dict[str, object] = {
        "receive_closed": False,
        "send_closed": False,
        "join_timeouts": [],
        "terminate_calls": 0,
        "kill_calls": 0,
    }

    class Receive:
        def poll(self, timeout):
            assert timeout == cli.M4_BENCHMARK_TIMEOUT_SECONDS
            if mode == "exception":
                raise arbitrary_failure
            return mode != "timeout"

        def recv(self):
            if mode == "eof":
                raise EOFError("synthetic closed pipe")
            return {"ok": True, "report": report}

        def close(self):
            observed["receive_closed"] = True

    class Send:
        def close(self):
            observed["send_closed"] = True

    class Process:
        exitcode = 0

        def __init__(self):
            self.alive = mode in {"timeout", "exception"}

        def start(self):
            return None

        def join(self, timeout=None):
            observed["join_timeouts"].append(timeout)

        def is_alive(self):
            return self.alive

        def terminate(self):
            observed["terminate_calls"] += 1
            self.alive = False

        def kill(self):
            observed["kill_calls"] += 1
            self.alive = False

    process = Process()
    context = SimpleNamespace(
        Pipe=lambda duplex: (Receive(), Send()),
        Process=lambda **kwargs: process,
    )
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(device_get=lambda value: value),
    )
    monkeypatch.setattr(
        cli.multiprocessing,
        "get_context",
        lambda requested: context,
    )

    if mode == "normal":
        assert cli._fresh_worker_benchmark(
            (np.asarray([1]), np.asarray([2]))
        ) == report
    elif mode == "exception":
        with pytest.raises(RuntimeError) as caught:
            cli._fresh_worker_benchmark(
                (np.asarray([1]), np.asarray([2]))
            )
        assert caught.value is arbitrary_failure
    else:
        with pytest.raises(cli.M4CommandError) as caught:
            cli._fresh_worker_benchmark(
                (np.asarray([1]), np.asarray([2]))
            )
        assert caught.value.code == expected_code

    assert observed["receive_closed"] is True
    assert observed["send_closed"] is True
    assert observed["join_timeouts"]
    assert process.is_alive() is False
    assert observed["kill_calls"] == 0
    if mode in {"timeout", "exception"}:
        assert observed["terminate_calls"] == 1


def test_fresh_worker_preserves_timeout_after_stubborn_child_is_killed(
    monkeypatch,
) -> None:
    observed: dict[str, object] = {
        "receive_closed": False,
        "send_closed": False,
        "join_timeouts": [],
        "terminate_calls": 0,
        "kill_calls": 0,
    }

    class Receive:
        def poll(self, timeout):
            assert timeout == cli.M4_BENCHMARK_TIMEOUT_SECONDS
            return False

        def close(self):
            observed["receive_closed"] = True

    class Send:
        def close(self):
            observed["send_closed"] = True

    class Process:
        exitcode = -9

        def __init__(self):
            self.alive = True

        def start(self):
            return None

        def join(self, timeout=None):
            observed["join_timeouts"].append(timeout)

        def is_alive(self):
            return self.alive

        def terminate(self):
            observed["terminate_calls"] += 1

        def kill(self):
            observed["kill_calls"] += 1
            self.alive = False

    process = Process()
    context = SimpleNamespace(
        Pipe=lambda duplex: (Receive(), Send()),
        Process=lambda **kwargs: process,
    )
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(device_get=lambda value: value),
    )
    monkeypatch.setattr(
        cli.multiprocessing,
        "get_context",
        lambda requested: context,
    )

    with pytest.raises(cli.M4CommandError) as caught:
        cli._fresh_worker_benchmark(
            (np.asarray([1]), np.asarray([2]))
        )

    assert caught.value.code == "benchmark_timeout"
    assert observed["receive_closed"] is True
    assert observed["send_closed"] is True
    assert observed["terminate_calls"] == 2
    assert observed["kill_calls"] == 1
    assert observed["join_timeouts"] == [10.0, 10.0, None]
    assert process.is_alive() is False


@pytest.mark.parametrize("failing_gate", ["clean", "provenance", "ignore"])
def test_final_gate_failure_leaves_no_accepted_report(
    monkeypatch,
    tmp_path,
    failing_gate,
) -> None:
    output = tmp_path / "outputs" / "m4" / "invented"
    output.mkdir(parents=True)
    provenance = {"invented": True}
    monkeypatch.setattr(cli, "_assert_sanitized_aggregate", lambda _: None)
    monkeypatch.setattr(cli, "_is_git_ignored", lambda *args: True)

    if failing_gate == "clean":
        monkeypatch.setattr(
            cli,
            "_assert_clean_worktree",
            lambda _: (_ for _ in ()).throw(
                cli.M4CommandError("dirty_worktree", "invented")
            ),
        )
        monkeypatch.setattr(cli, "_execution_provenance", lambda _: provenance)
        monkeypatch.setattr(cli, "_assert_output_ignored", lambda *args: None)
    elif failing_gate == "provenance":
        monkeypatch.setattr(cli, "_assert_clean_worktree", lambda _: None)
        monkeypatch.setattr(
            cli,
            "_execution_provenance",
            lambda _: {"changed": True},
        )
        monkeypatch.setattr(cli, "_assert_output_ignored", lambda *args: None)
    else:
        monkeypatch.setattr(cli, "_assert_clean_worktree", lambda _: None)
        monkeypatch.setattr(cli, "_execution_provenance", lambda _: provenance)
        monkeypatch.setattr(
            cli,
            "_assert_output_ignored",
            lambda *args: (_ for _ in ()).throw(
                cli.M4CommandError("output_not_ignored", "invented")
            ),
        )

    with pytest.raises(cli.M4CommandError):
        cli._publish_accepted_aggregate(
            root=tmp_path,
            output=output,
            provenance=provenance,
            aggregate={"accepted": True},
        )
    assert not (output / "aggregate-summary.json").exists()


def test_aggregate_privacy_gate_rejects_private_fields_and_paths() -> None:
    cli._assert_sanitized_aggregate(
        {
            "accepted": True,
            "per_shard": [{"shard_suffix": "00000", "raw": 5}],
            "privacy": {
                "absolute_local_values_absent": True,
                "motion_samples_absent": True,
                "private_manifests_remain_local": True,
                "source_hashes_absent": True,
                "source_identifiers_absent": True,
            },
        }
    )
    for payload in (
        {"native_scenario_id": "invented"},
        {"shard_digest": "not-even-a-real-digest"},
        {"output": "/absolute/local/path"},
        {"value": "a" * 64},
    ):
        with pytest.raises(cli.M4CommandError, match="aggregate_privacy"):
            cli._assert_sanitized_aggregate(payload)


@pytest.mark.parametrize(
    "secret",
    (
        "/private/local/path/native-id-deadbeef",
        "NativeScenario-Private-123",
    ),
)
def test_main_failure_output_never_echoes_exception_details(
    monkeypatch,
    capsys,
    secret,
) -> None:
    class UnsafeFailure(RuntimeError):
        code = secret

    monkeypatch.setattr(
        cli,
        "run_acceptance",
        lambda _: (_ for _ in ()).throw(UnsafeFailure(secret)),
    )
    argv = [
        "--project-root",
        "/private/project",
        "--data-dir",
        "/private/data",
        "--output-dir",
        "/private/project/outputs/m4/run",
    ]
    with pytest.raises(SystemExit) as caught:
        cli.main(argv)
    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert secret not in captured.err
    assert "unexpected_failure" in captured.err


def test_success_output_contains_only_relative_ignored_report(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli,
        "run_acceptance",
        lambda _: cli._RunResult(
            report_relative=Path("outputs/m4/invented/aggregate-summary.json")
        ),
    )
    argv = [
        "--project-root",
        "/private/project",
        "--data-dir",
        "/private/data",
        "--output-dir",
        "/private/project/outputs/m4/run",
    ]
    assert cli.main(argv) == 0
    captured = capsys.readouterr()
    assert "/private/" not in captured.out
    assert "outputs/m4/invented/aggregate-summary.json" in captured.out
    assert "PASS" in captured.out


def _terminal_test_output(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    observed_ignored: list[Path] = []

    def ignored(root, path):
        assert root == tmp_path
        observed_ignored.append(path)
        return True

    monkeypatch.setattr(cli, "_is_git_ignored", ignored)
    return output, observed_ignored


def _pending_acceptance():
    return cli._PendingAcceptance(aggregate={"accepted": True})


def _main_argv(tmp_path):
    return [
        "--project-root",
        str(tmp_path),
        "--data-dir",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "run"),
    ]


def test_terminal_capture_clean_success_is_exclusive_ignored_and_owner_only(
    monkeypatch,
    tmp_path,
) -> None:
    output, ignored = _terminal_test_output(tmp_path, monkeypatch)
    raw_descriptors: list[int] = []
    saved_descriptors: list[int] = []
    real_open = cli._terminal_open
    real_dup = cli._terminal_dup

    def observed_open(path, flags, mode):
        descriptor = real_open(path, flags, mode)
        raw_descriptors.append(descriptor)
        return descriptor

    def observed_dup(descriptor):
        duplicate = real_dup(descriptor)
        saved_descriptors.append(duplicate)
        return duplicate

    def clean_callback():
        assert len(raw_descriptors) == 1
        assert len(saved_descriptors) == 4
        assert os.get_inheritable(raw_descriptors[0]) is False
        assert all(
            os.get_inheritable(descriptor) is False
            for descriptor in saved_descriptors
        )
        assert os.get_inheritable(1) is True
        assert os.get_inheritable(2) is True
        return _pending_acceptance()

    monkeypatch.setattr(cli, "_terminal_open", observed_open)
    monkeypatch.setattr(cli, "_terminal_dup", observed_dup)
    pending, terminal_status = cli._run_captured_phase(
        root=tmp_path,
        output=output,
        callback=clean_callback,
    )
    transcript = output / "terminal-output.bin"
    try:
        assert pending.aggregate == {"accepted": True}
        assert transcript.read_bytes() == b""
        transcript_stat = transcript.stat(follow_symlinks=False)
        assert stat.S_ISREG(transcript_stat.st_mode)
        assert stat.S_IMODE(transcript_stat.st_mode) == 0o600
        assert ignored == [transcript, transcript]
        assert os.get_inheritable(terminal_status.stdout_fd) is False
        assert os.get_inheritable(terminal_status.stderr_fd) is False
    finally:
        terminal_status.close_best_effort()

    callback_called = False

    def forbidden_callback():
        nonlocal callback_called
        callback_called = True
        return _pending_acceptance()

    with pytest.raises(
        cli.M4CommandError,
        match="terminal_capture_failed",
    ):
        cli._run_captured_phase(
            root=tmp_path,
            output=output,
            callback=forbidden_callback,
        )
    assert callback_called is False
    assert transcript.read_bytes() == b""


@pytest.mark.parametrize("existing_kind", ["file", "symlink"])
def test_terminal_capture_refuses_existing_file_or_symlink_without_overwrite(
    monkeypatch,
    tmp_path,
    existing_kind,
) -> None:
    output, _ = _terminal_test_output(tmp_path, monkeypatch)
    transcript = output / "terminal-output.bin"
    outside = tmp_path / "outside-private"
    outside.write_bytes(b"must-remain")
    if existing_kind == "file":
        transcript.write_bytes(b"existing-private")
    else:
        transcript.symlink_to(outside)

    called = False

    def forbidden():
        nonlocal called
        called = True
        return _pending_acceptance()

    with pytest.raises(
        cli.M4CommandError,
        match="terminal_capture_failed",
    ):
        cli._run_captured_phase(
            root=tmp_path,
            output=output,
            callback=forbidden,
        )
    assert called is False
    assert outside.read_bytes() == b"must-remain"
    if existing_kind == "file":
        assert transcript.read_bytes() == b"existing-private"
    else:
        assert transcript.is_symlink()


def test_terminal_capture_catches_python_native_and_spawn_output(
    monkeypatch,
    tmp_path,
    capfd,
) -> None:
    output, _ = _terminal_test_output(tmp_path, monkeypatch)
    logger = logging.getLogger("evalsim.synthetic.terminal")
    logger.propagate = False
    logger.setLevel(logging.WARNING)

    def emit_every_boundary():
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)
        try:
            print("python-print-/private/invented", flush=False)
            logger.warning("python-log-InventedPrivate123")
            os.write(1, b"native-stdout-/private/invented\n")
            os.write(2, b"native-stderr-InventedPrivate123\n")
            process = cli.multiprocessing.get_context("spawn").Process(
                target=_spawn_terminal_writer,
            )
            process.start()
            process.join(timeout=30.0)
            assert process.exitcode == 0
            return _pending_acceptance()
        finally:
            logger.removeHandler(handler)

    with capfd.disabled():
        with pytest.raises(cli._TerminalizedFailure) as caught:
            cli._run_captured_phase(
                root=tmp_path,
                output=output,
                callback=emit_every_boundary,
            )

    failure = caught.value
    assert type(failure.primary) is cli.M4CommandError
    assert failure.primary.code == "terminal_output_detected"
    transcript = (output / "terminal-output.bin").read_bytes()
    for sentinel in (
        b"python-print-/private/invented",
        b"python-log-InventedPrivate123",
        b"spawn-python-warning-" + b"a" * 64,
        b"native-stdout-/private/invented",
        b"native-stderr-InventedPrivate123",
        b"spawn-stdout-absolute-/private/invented",
        b"spawn-stderr-identity-InventedPrivate123",
    ):
        assert sentinel in transcript

    failure.terminal_status.close_best_effort()
    status = cli._TerminalStatus(
        stdout_fd=os.dup(1),
        stderr_fd=os.dup(2),
    )
    os.set_inheritable(status.stdout_fd, False)
    os.set_inheritable(status.stderr_fd, False)
    failure = cli._TerminalizedFailure(failure.primary, status)
    monkeypatch.setattr(
        cli,
        "run_acceptance",
        lambda _: (_ for _ in ()).throw(failure),
    )
    with pytest.raises(SystemExit) as exited:
        cli.main(_main_argv(tmp_path))
    assert exited.value.code == 1
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "M4 local acceptance: FAIL (terminal_output_detected)\n"
    )
    assert "/private/invented" not in captured.err
    assert "InventedPrivate123" not in captured.err
    assert "a" * 64 not in captured.err


@pytest.mark.parametrize(
    ("primary", "expected_code"),
    [
        (
            cli.M4CommandError("source_changed", "private detail"),
            "source_changed",
        ),
        (RuntimeError("private untrusted detail"), "unexpected_failure"),
    ],
)
def test_terminal_primary_failure_precedes_diagnostics_and_capture_fault(
    monkeypatch,
    tmp_path,
    capfd,
    primary,
    expected_code,
) -> None:
    output, _ = _terminal_test_output(tmp_path, monkeypatch)
    real_dup2 = cli._terminal_dup2
    calls = 0

    def restore_then_report_failure(source, target, *, inheritable):
        nonlocal calls
        calls += 1
        result = real_dup2(source, target, inheritable=inheritable)
        if calls >= 3:
            raise OSError("invented restoration detail")
        return result

    monkeypatch.setattr(
        cli,
        "_terminal_dup2",
        restore_then_report_failure,
    )

    def fail():
        os.write(2, b"/private/diagnostic-before-primary\n")
        raise primary

    with pytest.raises(cli._TerminalizedFailure) as caught:
        cli._run_captured_phase(
            root=tmp_path,
            output=output,
            callback=fail,
        )
    failure = caught.value
    assert failure.primary is primary
    assert calls == 4

    monkeypatch.setattr(
        cli,
        "run_acceptance",
        lambda _: (_ for _ in ()).throw(failure),
    )
    with pytest.raises(SystemExit) as exited:
        cli.main(_main_argv(tmp_path))
    assert exited.value.code == 1
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == (
        f"M4 local acceptance: FAIL ({expected_code})\n"
    )
    assert "private" not in captured.err


@pytest.mark.parametrize("failing_target", [1, 2])
def test_genuine_pre_dup2_restore_failure_uses_only_saved_status_descriptor(
    tmp_path,
    failing_target,
) -> None:
    output = tmp_path / f"restore-target-{failing_target}"
    output.mkdir()
    project_root = Path(cli.__file__).resolve().parents[2]
    script = r"""
import os
from pathlib import Path
import sys

from evalsim.sources import waymax_m4_cli as cli

output = Path(sys.argv[1])
failing_target = int(sys.argv[2])
cli._is_git_ignored = lambda root, path: True
real_dup2 = cli._terminal_dup2
calls = 0

def fail_before_real_restore(source, target, *, inheritable):
    global calls
    calls += 1
    if calls >= 3 and target == failing_target:
        raise OSError("private genuine restore failure")
    return real_dup2(source, target, inheritable=inheritable)

cli._terminal_dup2 = fail_before_real_restore

def callback():
    os.write(1, b"PRIVATE_STDOUT_SENTINEL\n")
    os.write(2, b"PRIVATE_STDERR_SENTINEL\n")
    return cli._PendingAcceptance(aggregate={"accepted": True})

try:
    cli._run_captured_phase(
        root=output.parent,
        output=output,
        callback=callback,
    )
except cli._TerminalizedFailure as failure:
    if calls != 4:
        raise SystemExit(30)
    transcript_stat = os.stat(
        output / "terminal-output.bin",
        follow_symlinks=False,
    )
    target_stat = os.fstat(failing_target)
    if (
        target_stat.st_dev,
        target_stat.st_ino,
    ) != (
        transcript_stat.st_dev,
        transcript_stat.st_ino,
    ):
        raise SystemExit(31)
    cli.run_acceptance = lambda args: (_ for _ in ()).throw(failure)
    cli.main(
        [
            "--project-root",
            str(output.parent),
            "--data-dir",
            str(output.parent),
            "--output-dir",
            str(output),
        ]
    )
raise SystemExit(32)
"""

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(output),
            str(failing_target),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert completed.stdout == b""
    assert completed.stderr == (
        b"M4 local acceptance: FAIL (terminal_capture_failed)\n"
    )
    transcript = (output / "terminal-output.bin").read_bytes()
    assert b"PRIVATE_STDOUT_SENTINEL\n" in transcript
    assert b"PRIVATE_STDERR_SENTINEL\n" in transcript
    assert b"M4 local acceptance:" not in transcript
    assert not (output / "aggregate-summary.json").exists()
    assert b"PRIVATE_" not in completed.stdout
    assert b"PRIVATE_" not in completed.stderr


@pytest.mark.parametrize(
    "fault",
    [
        "open",
        "duplicate",
        "partial_redirect",
        "initial_flush",
        "final_flush",
        "native_flush",
        "fsync",
        "fstat_identity",
        "lstat_identity",
        "acceptance_close",
        "restoration_stdout",
        "restoration_stderr",
    ],
)
def test_terminal_capture_faults_fail_closed_without_publication(
    monkeypatch,
    tmp_path,
    capfd,
    fault,
) -> None:
    output, _ = _terminal_test_output(tmp_path, monkeypatch)
    callback_calls = 0
    publication_calls = 0
    native_flush_calls = 0
    restoration_targets: list[int] = []

    def callback():
        nonlocal callback_calls
        callback_calls += 1
        return _pending_acceptance()

    def publication(*args, **kwargs):
        nonlocal publication_calls
        publication_calls += 1
        raise AssertionError("publication must not run")

    monkeypatch.setattr(cli, "_publish_accepted_aggregate", publication)
    if fault == "open":
        monkeypatch.setattr(
            cli,
            "_terminal_open",
            lambda *args: (_ for _ in ()).throw(
                OSError("invented open detail")
            ),
        )
    elif fault == "duplicate":
        monkeypatch.setattr(
            cli,
            "_terminal_dup",
            lambda *args: (_ for _ in ()).throw(
                OSError("invented duplicate detail")
            ),
        )
    elif fault == "initial_flush":
        monkeypatch.setattr(
            cli,
            "_flush_python_streams",
            lambda: (_ for _ in ()).throw(
                OSError("invented initial flush detail")
            ),
        )
    elif fault == "partial_redirect":
        real_dup2 = cli._terminal_dup2
        calls = 0

        def partial(source, target, *, inheritable):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("invented partial redirect detail")
            return real_dup2(source, target, inheritable=inheritable)

        monkeypatch.setattr(cli, "_terminal_dup2", partial)
        real_native_flush = cli._flush_native_stdio

        def observed_native_flush():
            nonlocal native_flush_calls
            native_flush_calls += 1
            real_native_flush()

        monkeypatch.setattr(
            cli,
            "_flush_native_stdio",
            observed_native_flush,
        )
    elif fault == "final_flush":
        real_flush = cli._flush_python_streams
        calls = 0

        def final_flush():
            nonlocal calls
            calls += 1
            real_flush()
            if calls == 2:
                raise OSError("invented final flush detail")

        monkeypatch.setattr(cli, "_flush_python_streams", final_flush)
    elif fault == "native_flush":
        real_native_flush = cli._flush_native_stdio

        def native_flush_then_fail():
            nonlocal native_flush_calls
            native_flush_calls += 1
            real_native_flush()
            raise OSError("invented native flush detail")

        monkeypatch.setattr(
            cli,
            "_flush_native_stdio",
            native_flush_then_fail,
        )
    elif fault == "fsync":
        real_fsync = cli._terminal_fsync

        def fsync_then_fail(descriptor):
            real_fsync(descriptor)
            raise OSError("invented fsync detail")

        monkeypatch.setattr(cli, "_terminal_fsync", fsync_then_fail)
    elif fault == "fstat_identity":
        real_fstat = cli._terminal_fstat
        calls = 0

        def changed_fstat(descriptor):
            nonlocal calls
            calls += 1
            observed = real_fstat(descriptor)
            if calls == 2:
                return SimpleNamespace(
                    st_dev=observed.st_dev,
                    st_ino=observed.st_ino + 1,
                    st_mode=observed.st_mode,
                    st_nlink=observed.st_nlink,
                    st_size=observed.st_size,
                )
            return observed

        monkeypatch.setattr(cli, "_terminal_fstat", changed_fstat)
    elif fault == "lstat_identity":
        real_lstat = cli._terminal_lstat
        calls = 0

        def changed_lstat(path):
            nonlocal calls
            calls += 1
            observed = real_lstat(path)
            if calls == 2:
                return SimpleNamespace(
                    st_dev=observed.st_dev,
                    st_ino=observed.st_ino + 1,
                    st_mode=observed.st_mode,
                    st_nlink=observed.st_nlink,
                    st_size=observed.st_size,
                )
            return observed

        monkeypatch.setattr(cli, "_terminal_lstat", changed_lstat)
    elif fault == "acceptance_close":
        real_close = cli._terminal_close
        calls = 0

        def close_then_fail(descriptor):
            nonlocal calls
            calls += 1
            real_close(descriptor)
            if calls == 1:
                raise OSError("invented close detail")

        monkeypatch.setattr(cli, "_terminal_close", close_then_fail)
    elif fault in {"restoration_stdout", "restoration_stderr"}:
        real_dup2 = cli._terminal_dup2
        calls = 0
        failing_target = 1 if fault == "restoration_stdout" else 2

        def restore_then_fail(source, target, *, inheritable):
            nonlocal calls
            calls += 1
            result = real_dup2(source, target, inheritable=inheritable)
            if calls >= 3:
                restoration_targets.append(target)
            if calls >= 3 and target == failing_target:
                raise OSError("invented restoration detail")
            return result

        monkeypatch.setattr(cli, "_terminal_dup2", restore_then_fail)

    try:
        cli._run_captured_phase(
            root=tmp_path,
            output=output,
            callback=callback,
        )
    except cli._TerminalizedFailure as failure:
        monkeypatch.setattr(
            cli,
            "run_acceptance",
            lambda _: (_ for _ in ()).throw(failure),
        )
        with pytest.raises(SystemExit) as exited:
            cli.main(_main_argv(tmp_path))
        assert exited.value.code == 1
    except cli.M4CommandError as failure:
        assert failure.code == "terminal_capture_failed"
        monkeypatch.setattr(
            cli,
            "run_acceptance",
            lambda _: (_ for _ in ()).throw(failure),
        )
        with pytest.raises(SystemExit) as exited:
            cli.main(_main_argv(tmp_path))
        assert exited.value.code == 1
    else:
        pytest.fail("an injected terminal capture fault was accepted")

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "M4 local acceptance: FAIL (terminal_capture_failed)\n"
    )
    assert "invented" not in captured.err
    assert publication_calls == 0
    assert not (output / "aggregate-summary.json").exists()
    if fault in {"open", "duplicate", "initial_flush", "partial_redirect"}:
        assert callback_calls == 0
    else:
        assert callback_calls == 1
    if fault == "partial_redirect":
        assert native_flush_calls == 0
    if fault == "native_flush":
        assert native_flush_calls == 1
    if fault in {"restoration_stdout", "restoration_stderr"}:
        assert restoration_targets == [1, 2]


def _mock_run_preflight(monkeypatch, tmp_path, output):
    data_dir = tmp_path / "validation"
    data_dir.mkdir()
    monkeypatch.setenv("EVALSIM_RUN_WAYMO_LOCAL", "1")
    monkeypatch.setattr(cli, "_project_root", lambda _: tmp_path)
    monkeypatch.setattr(cli, "_assert_running_checkout", lambda _: None)
    monkeypatch.setattr(cli, "_assert_clean_worktree", lambda _: None)
    monkeypatch.setattr(
        cli,
        "_prepare_output_directory",
        lambda *args, **kwargs: output,
    )
    monkeypatch.setattr(cli, "_execution_provenance", lambda _: {"p": 1})
    monkeypatch.setattr(cli, "_write_json_exclusive", lambda *args: None)
    return argparse.Namespace(
        project_root=tmp_path,
        data_dir=data_dir,
        output_dir=output,
    )


def test_run_acceptance_publishes_only_after_zero_byte_finalization(
    monkeypatch,
    tmp_path,
) -> None:
    output, _ = _terminal_test_output(tmp_path, monkeypatch)
    args = _mock_run_preflight(monkeypatch, tmp_path, output)
    phase: list[str] = []

    def optional_runtime(**kwargs):
        phase.append("optional")
        return cli._PendingAcceptance(
            aggregate={
                "accepted": True,
                "privacy": {"terminal_path_absent": True},
            }
        )

    def publish(*, root, output, provenance, aggregate):
        phase.append("publish")
        transcript = output / "terminal-output.bin"
        assert transcript.read_bytes() == b""
        assert aggregate == {
            "accepted": True,
            "privacy": {"terminal_path_absent": True},
        }
        assert "terminal-output.bin" not in repr(aggregate)
        return output / "aggregate-summary.json"

    monkeypatch.setattr(
        cli,
        "_execute_captured_acceptance",
        optional_runtime,
    )
    monkeypatch.setattr(cli, "_publish_accepted_aggregate", publish)

    result = cli.run_acceptance(args)
    try:
        assert result.report_relative == Path(
            "run/aggregate-summary.json"
        )
        assert phase == ["optional", "publish"]
    finally:
        assert result.terminal_status is not None
        result.terminal_status.close_best_effort()


def test_run_acceptance_diagnostic_blocks_aggregate_publication(
    monkeypatch,
    tmp_path,
) -> None:
    output, _ = _terminal_test_output(tmp_path, monkeypatch)
    args = _mock_run_preflight(monkeypatch, tmp_path, output)
    publication_calls = 0

    def optional_runtime(**kwargs):
        os.write(2, b"/private/never-publish-this\n")
        return _pending_acceptance()

    def publish(**kwargs):
        nonlocal publication_calls
        publication_calls += 1
        return output / "aggregate-summary.json"

    monkeypatch.setattr(
        cli,
        "_execute_captured_acceptance",
        optional_runtime,
    )
    monkeypatch.setattr(cli, "_publish_accepted_aggregate", publish)

    with pytest.raises(cli._TerminalizedFailure) as caught:
        cli.run_acceptance(args)
    try:
        assert caught.value.primary.code == "terminal_output_detected"
        assert publication_calls == 0
        assert not (output / "aggregate-summary.json").exists()
    finally:
        caught.value.terminal_status.close_best_effort()


def test_main_clean_success_output_is_exact(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    result = cli._RunResult(
        report_relative=Path(
            "outputs/m4/invented/aggregate-summary.json"
        )
    )
    monkeypatch.setattr(cli, "run_acceptance", lambda _: result)

    assert cli.main(_main_argv(tmp_path)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        "M4 local acceptance: PASS\n"
        "Native WOMD identities, locators, digests, and coordinates were not printed.\n"
        "Ignored aggregate report: "
        "outputs/m4/invented/aggregate-summary.json\n"
    )


def test_main_fd_status_success_output_is_exact_and_status_fds_close(
    monkeypatch,
    tmp_path,
    capfd,
) -> None:
    stdout_status = os.dup(1)
    stderr_status = os.dup(2)
    os.set_inheritable(stdout_status, False)
    os.set_inheritable(stderr_status, False)
    result = cli._RunResult(
        report_relative=Path(
            "outputs/m4/invented/aggregate-summary.json"
        ),
        terminal_status=cli._TerminalStatus(
            stdout_fd=stdout_status,
            stderr_fd=stderr_status,
        ),
    )
    monkeypatch.setattr(cli, "run_acceptance", lambda _: result)

    assert cli.main(_main_argv(tmp_path)) == 0
    captured = capfd.readouterr()
    assert captured.err == ""
    assert captured.out == (
        "M4 local acceptance: PASS\n"
        "Native WOMD identities, locators, digests, and coordinates were not printed.\n"
        "Ignored aggregate report: "
        "outputs/m4/invented/aggregate-summary.json\n"
    )
    for descriptor in (stdout_status, stderr_status):
        with pytest.raises(OSError):
            os.fstat(descriptor)
