"""Data-free safety and orchestration tests for the local M4 command."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys
from types import MappingProxyType, SimpleNamespace

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
