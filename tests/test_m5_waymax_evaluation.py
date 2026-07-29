"""Data-free gates for the official M5 Waymax reference/parity adapter."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, NamedTuple

import numpy as np
import pytest

from evalsim import Agent, AgentType, Rollout, Scenario
from evalsim.evaluation.m5 import (
    EvaluationCase,
    ExecutionRollout,
    ExecutionSpec,
    canonical_m5_policies,
    evaluate_m5_case,
)
from evalsim.evaluation.m5_waymax import (
    M5_PARITY_METRIC_NAMES,
    M5_PARITY_POLICY_NAMES,
    M5_PARITY_RANK_DOMAIN,
    M5_PARITY_ROW_COUNT,
    M5_PARITY_SCENE_COUNT,
    M5ParityCaseInput,
    M5StreamingParityAccumulator,
    M5ParityOrderReceipt,
    M5WaymaxEvaluationError,
    M5WaymaxParityRow,
    WaymaxExactLogReferenceExecutor,
    WaymaxM5MetricParityAdapter,
    build_continuous_parity_row,
    build_discrete_parity_row,
    build_waymax_parity_rows,
    select_m5_parity_members,
)
from evalsim.rollout import RolloutEngine
from evalsim.simulators.waymax_reference import (
    M4_EXACT_LOG_TRANSITIONS,
    WAYMAX_EXACT_LOG_NAME,
    WAYMAX_REFERENCE_VERSION,
)
from evalsim.sources.m5_m4_reuse import AcceptedM4MemberRef
from evalsim.sources.waymax_cohort import ScanEvent, rank_record
from evalsim.sources.waymax_loader import (
    M4ReloadExpectation,
    M4ShardLocator,
    WaymaxRecord,
)


def _accepted_members() -> tuple[AcceptedM4MemberRef, ...]:
    result = []
    for cohort_index in range(128):
        suffix = f"{cohort_index % 10:05d}"
        native_id = f"{cohort_index + 1:032x}"
        event = replace(
            ScanEvent.eligible_event(
                shard_suffix=suffix,
                record_ordinal=cohort_index * 3 + 1,
                native_scenario_id=native_id,
                shard_sha256="a" * 64,
                dataset_config_fingerprint="b" * 64,
            ),
            selected=True,
        )
        expectation = M4ReloadExpectation(
            locator=M4ShardLocator(
                shard_suffix=suffix,
                record_ordinal=cohort_index * 3 + 1,
            ),
            expected_scenario_id=native_id,
            expected_shard_sha256="a" * 64,
            expected_dataset_config_fingerprint="b" * 64,
        )
        result.append(
            AcceptedM4MemberRef(
                cohort_index=cohort_index,
                event=event,
                expectation=expectation,
            )
        )
    return tuple(result)


def test_parity_selector_uses_frozen_rank_order_and_opaque_receipt() -> None:
    accepted = _accepted_members()
    selection = select_m5_parity_members(accepted[::-1])
    expected = sorted(
        accepted,
        key=lambda member: (
            rank_record(
                M5_PARITY_RANK_DOMAIN,
                member.event.shard_suffix,
                member.event.record_ordinal,
                member.event.native_scenario_id,
            ),
            member.event.shard_suffix,
            member.event.record_ordinal,
        ),
    )[:M5_PARITY_SCENE_COUNT]
    assert tuple(item.cohort_index for item in selection.members) == tuple(
        item.cohort_index for item in expected
    )
    assert tuple(item.parity_index for item in selection.members) == tuple(
        range(M5_PARITY_SCENE_COUNT)
    )
    repeat = select_m5_parity_members(accepted)
    assert repeat.receipt == selection.receipt
    payload = {
        "members": [
            {
                "cohort_index": member.cohort_index,
                "parity_index": member.parity_index,
                "rank_sha256": member.rank_sha256,
                "source_event": member.source_ref.event.to_dict(),
            }
            for member in selection.members
        ],
        "rank_domain": M5_PARITY_RANK_DOMAIN,
        "version": "m5-metric-parity-order-1",
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"evalsim-m5-metric-parity-order-v1\0")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    assert (
        selection.receipt.ordered_membership_sha256
        == digest.hexdigest()
    )
    assert M5ParityOrderReceipt.from_dict(
        selection.receipt.to_dict()
    ) == selection.receipt
    representation = repr(selection) + repr(selection.members[0])
    assert selection.receipt.ordered_membership_sha256 not in representation
    assert selection.members[0].source_ref.event.native_scenario_id not in representation
    assert selection.members[0].rank_sha256 not in representation


def test_parity_selector_rejects_incomplete_or_inconsistent_source_binding() -> None:
    accepted = _accepted_members()
    with pytest.raises(M5WaymaxEvaluationError, match="parity_selection_invalid"):
        select_m5_parity_members(accepted[:-1])
    first = accepted[0]
    contradicted = AcceptedM4MemberRef(
        cohort_index=first.cohort_index,
        event=first.event,
        expectation=replace(
            first.expectation,
            expected_scenario_id="f" * 32,
        ),
    )
    with pytest.raises(M5WaymaxEvaluationError, match="parity_selection_invalid"):
        select_m5_parity_members((contradicted,) + accepted[1:])


def test_parity_receipt_changes_when_private_membership_changes() -> None:
    accepted = list(_accepted_members())
    original = select_m5_parity_members(accepted)
    target_index = original.members[0].cohort_index
    target = accepted[target_index]
    new_native_id = "f" * 31 + "e"
    event = replace(
        ScanEvent.eligible_event(
            shard_suffix=target.event.shard_suffix,
            record_ordinal=target.event.record_ordinal,
            native_scenario_id=new_native_id,
            shard_sha256=target.event.shard_sha256,
            dataset_config_fingerprint=target.event.dataset_config_fingerprint,
        ),
        selected=True,
    )
    accepted[target_index] = AcceptedM4MemberRef(
        cohort_index=target.cohort_index,
        event=event,
        expectation=replace(
            target.expectation,
            expected_scenario_id=new_native_id,
        ),
    )
    changed = select_m5_parity_members(accepted)
    assert (
        changed.receipt.ordered_membership_sha256
        != original.receipt.ordered_membership_sha256
    )


def _advance_float32(value: np.float32, count: int) -> np.float32:
    result = np.float32(value)
    for _ in range(count):
        result = np.nextafter(
            result,
            np.float32(np.inf),
            dtype=np.float32,
        )
    return result


def test_continuous_row_applies_frozen_float32_ulp_tolerance() -> None:
    mask = np.array([[True]], dtype=bool)
    reference = np.array([[1.0]], dtype=np.float32)
    accepted = build_continuous_parity_row(
        parity_index=0,
        policy_name="constant_velocity",
        metric_name="log_divergence",
        custom=np.array([[_advance_float32(np.float32(1.0), 8)]], dtype=np.float32),
        reference=reference,
        custom_mask=mask,
        reference_mask=mask.copy(),
    )
    assert accepted.status == "accepted"
    assert accepted.mismatch_count == 0
    exact = build_continuous_parity_row(
        parity_index=0,
        policy_name="constant_velocity",
        metric_name="log_divergence",
        custom=reference.copy(),
        reference=reference,
        custom_mask=mask,
        reference_mask=mask.copy(),
    )
    assert exact.max_tolerance_excess < 0.0
    rejected = build_continuous_parity_row(
        parity_index=0,
        policy_name="constant_velocity",
        metric_name="log_divergence",
        custom=np.array([[_advance_float32(np.float32(1.0), 9)]], dtype=np.float32),
        reference=reference,
        custom_mask=mask,
        reference_mask=mask.copy(),
    )
    assert rejected.status == "rejected"
    assert rejected.mismatch_count == 1


def test_component_rows_fail_on_mask_empty_nonfinite_and_branch_drift() -> None:
    values = np.zeros((1, 2), dtype=np.float32)
    mask = np.array([[True, False]])
    with pytest.raises(M5WaymaxEvaluationError, match="component_mask_mismatch"):
        build_continuous_parity_row(
            parity_index=0,
            policy_name="idm",
            metric_name="log_divergence",
            custom=values,
            reference=values,
            custom_mask=mask,
            reference_mask=~mask,
        )
    with pytest.raises(M5WaymaxEvaluationError, match="component_empty"):
        build_discrete_parity_row(
            parity_index=0,
            policy_name="idm",
            metric_name="overlap",
            custom=np.zeros((1, 2), dtype=bool),
            reference=np.zeros((1, 2), dtype=bool),
            custom_mask=np.zeros((1, 2), dtype=bool),
            reference_mask=np.zeros((1, 2), dtype=bool),
        )
    nonfinite = values.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(M5WaymaxEvaluationError, match="component_nonfinite"):
        build_continuous_parity_row(
            parity_index=0,
            policy_name="idm",
            metric_name="log_divergence",
            custom=nonfinite,
            reference=values,
            custom_mask=mask,
            reference_mask=mask.copy(),
        )
    invalid_raw_ignored = build_discrete_parity_row(
        parity_index=0,
        policy_name="log_replay",
        metric_name="overlap",
        custom=np.array([[False, False]]),
        reference=np.array([[0.0, np.nan]]),
        custom_mask=mask,
        reference_mask=mask.copy(),
    )
    assert invalid_raw_ignored.status == "accepted"
    branch_drift = build_discrete_parity_row(
        parity_index=0,
        policy_name="log_replay",
        metric_name="kinematic_infeasibility",
        custom=np.array([[False, False]]),
        reference=np.array([[False, False]]),
        custom_mask=mask,
        reference_mask=mask.copy(),
        additional_mismatches=(np.array([[True, False]]),),
    )
    assert branch_drift.status == "rejected"
    assert branch_drift.mismatch_count == 1


def _agent(
    identifier: int,
    *,
    frames: int = 91,
    x_offset: float = 0.0,
    speed: float = 1.0,
) -> Agent:
    timestamps = np.arange(frames, dtype=np.float64) * 0.1
    return Agent(
        id=identifier,
        type=AgentType.VEHICLE,
        valid=np.ones(frames, dtype=bool),
        x=x_offset + speed * timestamps,
        y=np.zeros(frames),
        heading=np.zeros(frames),
        vx=np.full(frames, speed),
        vy=np.zeros(frames),
        length=4.0,
        width=2.0,
    )


def _scenario(identifier: str = "m5-waymax-mock") -> Scenario:
    return Scenario(
        scenario_id=identifier,
        timestamps=np.arange(91, dtype=np.float64) * 0.1,
        agents=[
            _agent(101, x_offset=0.0, speed=2.0),
            _agent(102, x_offset=20.0, speed=1.0),
        ],
        ego_index=0,
        metadata={"current_index": 10, "source": "synthetic"},
    )


def _rollout(
    scenario: Scenario,
    *,
    name: str = WAYMAX_EXACT_LOG_NAME,
    version: str = WAYMAX_REFERENCE_VERSION,
    seed: int = 0,
) -> Rollout:
    return Rollout(
        scenario_id=scenario.scenario_id,
        sim_name=name,
        sim_version=version,
        seed=seed,
        timestamps=np.array(scenario.timestamps, copy=True),
        agents=[
            Agent(
                id=agent.id,
                type=agent.type,
                valid=np.array(agent.valid, copy=True),
                x=np.array(agent.x, copy=True),
                y=np.array(agent.y, copy=True),
                heading=np.array(agent.heading, copy=True),
                vx=np.array(agent.vx, copy=True),
                vy=np.array(agent.vy, copy=True),
                length=agent.length,
                width=agent.width,
            )
            for agent in scenario.agents
        ],
    )


@dataclass
class _FakeState:
    values: np.ndarray


class _FakeCompact(NamedTuple):
    values: np.ndarray


def _mock_case() -> EvaluationCase:
    scenario = _scenario()
    record = WaymaxRecord(
        scenario_id=scenario.scenario_id,
        state=_FakeState(values=np.arange(3, dtype=np.int64)),
        audit={},
        shard_suffix="00000",
        record_ordinal=0,
        shard_sha256="a" * 64,
        dataset_config_fingerprint="b" * 64,
    )
    return EvaluationCase(
        cohort_index=0,
        scenario=scenario,
        reference_payload=record,
    )


def test_exact_log_executor_is_seed_locked_and_checks_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evalsim.evaluation import m5_waymax

    case = _mock_case()
    calls: list[tuple[str, Any]] = []

    def compact(state: Any, *, num_steps: int) -> _FakeCompact:
        calls.append(("compact", num_steps))
        return _FakeCompact(values=np.arange(num_steps, dtype=np.int32))

    def validate(state: Any, result: Any) -> dict[str, bool]:
        calls.append(("validate", result.values.shape))
        return {
            "fields": True,
            "timestamps": True,
            "timesteps": True,
            "validity": True,
        }

    def convert(
        result: Any,
        *,
        state: Any,
        scenario: Scenario,
        sim_name: str,
        seed: int,
    ) -> Rollout:
        calls.append(("convert", (sim_name, seed)))
        return _rollout(scenario, name=sim_name, seed=seed)

    monkeypatch.setattr(m5_waymax, "compact_exact_log_rollout", compact)
    monkeypatch.setattr(m5_waymax, "validate_exact_log_compact", validate)
    monkeypatch.setattr(m5_waymax, "compact_waymax_to_rollout", convert)
    execution = WaymaxExactLogReferenceExecutor().execute(case)
    assert execution.spec.role == "reference"
    assert execution.spec.name == WAYMAX_EXACT_LOG_NAME
    assert execution.spec.version == WAYMAX_REFERENCE_VERSION
    assert execution.spec.seed == 0
    assert calls == [
        ("compact", M4_EXACT_LOG_TRANSITIONS),
        ("validate", (M4_EXACT_LOG_TRANSITIONS,)),
        ("convert", (WAYMAX_EXACT_LOG_NAME, 0)),
    ]
    with pytest.raises(M5WaymaxEvaluationError, match="execution_invalid"):
        WaymaxExactLogReferenceExecutor().execute(case, seed=1)


def test_exact_log_executor_detects_output_drift_and_source_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evalsim.evaluation import m5_waymax

    case = _mock_case()
    monkeypatch.setattr(
        m5_waymax,
        "compact_exact_log_rollout",
        lambda state, *, num_steps: _FakeCompact(
            values=np.arange(num_steps, dtype=np.int32)
        ),
    )
    monkeypatch.setattr(
        m5_waymax,
        "validate_exact_log_compact",
        lambda state, compact: {
            "fields": True,
            "timestamps": True,
            "timesteps": True,
            "validity": True,
        },
    )
    monkeypatch.setattr(
        m5_waymax,
        "compact_waymax_to_rollout",
        lambda compact, *, state, scenario, sim_name, seed: _rollout(
            scenario,
            name="wrong_reference",
        ),
    )
    with pytest.raises(M5WaymaxEvaluationError, match="exact_log_mapping_mismatch"):
        WaymaxExactLogReferenceExecutor().execute(case)

    def mutate(
        compact: Any,
        *,
        state: _FakeState,
        scenario: Scenario,
        sim_name: str,
        seed: int,
    ) -> Rollout:
        state.values[0] += 1
        return _rollout(scenario, name=sim_name)

    monkeypatch.setattr(m5_waymax, "compact_waymax_to_rollout", mutate)
    with pytest.raises(M5WaymaxEvaluationError, match="source_mutated"):
        WaymaxExactLogReferenceExecutor().execute(case)


def _policy_executions(scenario: Scenario) -> tuple[ExecutionRollout, ...]:
    result = []
    for policy in canonical_m5_policies():
        metadata = policy.metadata()
        result.append(
            ExecutionRollout(
                spec=ExecutionSpec(
                    name=metadata.name,
                    version=metadata.version,
                    role="policy",
                    seed=0,
                ),
                rollout=RolloutEngine().run(scenario, policy, seed=0),
            )
        )
    return tuple(result)


class _AcceptedAdapter:
    def evaluate_case(
        self,
        case: EvaluationCase,
        *,
        parity_index: int,
        executions: Sequence[ExecutionRollout],
    ) -> tuple[M5WaymaxParityRow, ...]:
        del case, executions
        return tuple(
            M5WaymaxParityRow(
                parity_index=parity_index,
                policy_name=policy_name,
                metric_name=metric_name,
                metric_version="1.0.0",
                compared_components=20,
                mismatch_count=0,
                max_abs_error=0.0,
                max_tolerance_excess=0.0,
                exact_match=True,
                status="accepted",
            )
            for policy_name in M5_PARITY_POLICY_NAMES
            for metric_name in M5_PARITY_METRIC_NAMES
        )


def _matrix_inputs() -> tuple[Any, tuple[M5ParityCaseInput, ...]]:
    selection = select_m5_parity_members(_accepted_members())
    result = []
    for member in selection.members:
        index = member.parity_index
        event = member.source_ref.event
        scenario = _scenario(event.native_scenario_id)
        record = WaymaxRecord(
            scenario_id=event.native_scenario_id,
            state=_FakeState(values=np.asarray([index], dtype=np.int64)),
            audit={},
            shard_suffix=event.shard_suffix,
            record_ordinal=event.record_ordinal,
            shard_sha256=event.shard_sha256,
            dataset_config_fingerprint=event.dataset_config_fingerprint,
        )
        result.append(
            M5ParityCaseInput(
                parity_index=index,
                case=EvaluationCase(
                    cohort_index=member.cohort_index,
                    scenario=scenario,
                    reference_payload=record,
                ),
                executions=_policy_executions(scenario),
            )
        )
    return selection, tuple(result)


def test_official_parity_builder_requires_exact_accepted_144_row_matrix() -> None:
    selection, inputs = _matrix_inputs()
    rows = build_waymax_parity_rows(
        inputs[::-1],
        selection=selection,
        adapter=_AcceptedAdapter(),
    )
    assert len(rows) == M5_PARITY_ROW_COUNT
    assert rows[0].parity_index == 0
    assert rows[-1].parity_index == M5_PARITY_SCENE_COUNT - 1
    with pytest.raises(M5WaymaxEvaluationError, match="parity_matrix_invalid"):
        build_waymax_parity_rows(
            inputs[:-1],
            selection=selection,
            adapter=_AcceptedAdapter(),
        )
    first = inputs[0]
    wrong_record = replace(
        first.case.reference_payload,
        record_ordinal=first.case.reference_payload.record_ordinal + 1,
    )
    wrong_case = EvaluationCase(
        cohort_index=first.case.cohort_index,
        scenario=first.case.scenario,
        reference_payload=wrong_record,
    )
    contradicted_inputs = (replace(first, case=wrong_case),) + inputs[1:]
    with pytest.raises(M5WaymaxEvaluationError, match="source_identity_mismatch"):
        build_waymax_parity_rows(
            contradicted_inputs,
            selection=selection,
            adapter=_AcceptedAdapter(),
        )

    class RejectedAdapter(_AcceptedAdapter):
        def evaluate_case(self, *args: Any, **kwargs: Any):
            rows = list(super().evaluate_case(*args, **kwargs))
            if kwargs["parity_index"] == 0:
                rows[0] = replace(
                    rows[0],
                    mismatch_count=1,
                    max_abs_error=1.0,
                    max_tolerance_excess=1.0,
                    exact_match=False,
                    status="rejected",
                )
            return tuple(rows)

    with pytest.raises(M5WaymaxEvaluationError, match="parity_mismatch"):
        build_waymax_parity_rows(
            inputs,
            selection=selection,
            adapter=RejectedAdapter(),
        )


def test_streaming_parity_consumes_arbitrary_order_without_retaining_cases() -> None:
    selection, inputs = _matrix_inputs()
    accumulator = M5StreamingParityAccumulator(
        selection,
        adapter=_AcceptedAdapter(),
    )
    for item in reversed(inputs):
        rows = accumulator.add_case(item.case, item.executions)
        assert len(rows) == 9
    assert accumulator.case_count == M5_PARITY_SCENE_COUNT
    rows = accumulator.finalize()
    assert len(rows) == M5_PARITY_ROW_COUNT
    assert tuple(row.parity_index for row in rows[::9]) == tuple(
        range(M5_PARITY_SCENE_COUNT)
    )
    with pytest.raises(M5WaymaxEvaluationError, match="parity_matrix_invalid"):
        accumulator.finalize()


def test_module_import_does_not_load_optional_native_stack() -> None:
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
import evalsim.evaluation.m5_waymax
assert not ({"jax", "tensorflow", "waymax"} & set(sys.modules))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _synthetic_native_fixture() -> tuple[Any, Scenario]:
    jnp = pytest.importorskip("jax.numpy")
    datatypes = pytest.importorskip("waymax.datatypes")

    objects = 128
    steps = 91
    active = 5
    shape = (objects, steps)
    frame = np.arange(steps, dtype=np.float32)
    numeric = {
        name: np.full(shape, -1000.0, dtype=np.float32)
        for name in (
            "x",
            "y",
            "z",
            "vel_x",
            "vel_y",
            "yaw",
            "length",
            "width",
            "height",
        )
    }
    valid = np.zeros(shape, dtype=bool)
    valid[:active] = True
    numeric["x"][0] = 10.0 + frame * 0.5
    numeric["y"][0] = 100.0
    numeric["vel_x"][0] = 5.0
    numeric["x"][1] = 100.0 + frame
    numeric["y"][1] = 0.0
    numeric["vel_x"][1] = 10.0
    numeric["x"][2] = 123.0
    numeric["y"][2] = 0.0
    numeric["vel_x"][2] = 0.0
    numeric["x"][3] = 200.0 + frame
    numeric["y"][3] = 40.0
    numeric["vel_x"][3] = 10.0
    numeric["x"][4] = 300.0 + frame * 0.25
    numeric["y"][4] = -40.0
    numeric["vel_x"][4] = 2.5
    numeric["z"][:active] = 0.0
    numeric["vel_y"][:active] = 0.0
    numeric["yaw"][:active] = 0.0
    for name, value in (("length", 4.0), ("width", 2.0), ("height", 1.5)):
        numeric[name][:active] = value
    timestamps = np.broadcast_to(
        np.arange(steps, dtype=np.int32) * 100_000,
        shape,
    ).copy()
    trajectory = datatypes.Trajectory(
        x=jnp.asarray(numeric["x"]),
        y=jnp.asarray(numeric["y"]),
        z=jnp.asarray(numeric["z"]),
        vel_x=jnp.asarray(numeric["vel_x"]),
        vel_y=jnp.asarray(numeric["vel_y"]),
        yaw=jnp.asarray(numeric["yaw"]),
        valid=jnp.asarray(valid),
        timestamp_micros=jnp.asarray(timestamps),
        length=jnp.asarray(numeric["length"]),
        width=jnp.asarray(numeric["width"]),
        height=jnp.asarray(numeric["height"]),
    )
    identifiers = np.full(objects, -1, dtype=np.int32)
    identifiers[:active] = (101, 102, 103, 104, 105)
    object_types = np.zeros(objects, dtype=np.int32)
    object_types[:4] = 1
    object_types[4] = 2
    is_sdc = np.zeros(objects, dtype=bool)
    is_sdc[0] = True
    is_valid = np.zeros(objects, dtype=bool)
    is_valid[:active] = True
    metadata = datatypes.ObjectMetadata(
        ids=jnp.asarray(identifiers),
        object_types=jnp.asarray(object_types),
        is_sdc=jnp.asarray(is_sdc),
        is_modeled=jnp.asarray(is_sdc),
        is_valid=jnp.asarray(is_valid),
        objects_of_interest=jnp.zeros(objects, dtype=bool),
        is_controlled=jnp.asarray(is_sdc),
    )
    traffic_lights = datatypes.TrafficLights(
        x=jnp.zeros((1, steps), dtype=jnp.float32),
        y=jnp.zeros((1, steps), dtype=jnp.float32),
        z=jnp.zeros((1, steps), dtype=jnp.float32),
        state=jnp.zeros((1, steps), dtype=jnp.int32),
        lane_ids=jnp.zeros((1, steps), dtype=jnp.int32),
        valid=jnp.zeros((1, steps), dtype=bool),
    )
    state = datatypes.SimulatorState(
        sim_trajectory=trajectory,
        log_trajectory=trajectory,
        log_traffic_light=traffic_lights,
        object_metadata=metadata,
        timestep=jnp.asarray(0, dtype=jnp.int32),
    )
    state.validate()
    type_map = {
        1: AgentType.VEHICLE,
        2: AgentType.PEDESTRIAN,
        3: AgentType.CYCLIST,
    }
    agents = []
    for index in range(active):
        agents.append(
            Agent(
                id=int(identifiers[index]),
                type=type_map[int(object_types[index])],
                valid=np.array(valid[index], copy=True),
                x=np.asarray(numeric["x"][index], dtype=np.float64),
                y=np.asarray(numeric["y"][index], dtype=np.float64),
                heading=np.asarray(numeric["yaw"][index], dtype=np.float64),
                vx=np.asarray(numeric["vel_x"][index], dtype=np.float64),
                vy=np.asarray(numeric["vel_y"][index], dtype=np.float64),
                length=4.0,
                width=2.0,
            )
        )
    scenario = Scenario(
        scenario_id="synthetic-m5-native",
        timestamps=(
            np.arange(steps, dtype=np.int64) * 100_000
        ).astype(np.float64)
        * 1e-6,
        agents=agents,
        ego_index=0,
        metadata={"current_index": 10, "source": "synthetic"},
    )
    return state, scenario


def test_pinned_native_exact_log_and_metric_parity_on_synthetic_state() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("waymax")
    state, scenario = _synthetic_native_fixture()
    record = WaymaxRecord(
        scenario_id=scenario.scenario_id,
        state=state,
        audit={},
        shard_suffix="00000",
        record_ordinal=0,
        shard_sha256="a" * 64,
        dataset_config_fingerprint="b" * 64,
    )
    case = EvaluationCase(
        cohort_index=0,
        scenario=scenario,
        reference_payload=record,
    )
    source_x_before = np.asarray(state.sim_trajectory.x).copy()
    source_timestep_before = np.asarray(state.timestep).copy()
    reference = WaymaxExactLogReferenceExecutor().execute(case)
    assert reference.spec.role == "reference"
    executions = _policy_executions(scenario)
    evaluated = evaluate_m5_case(case, executions + (reference,))
    assert len(evaluated.metric_rows) == 52
    assert {row["execution_role"] for row in evaluated.metric_rows} == {
        "policy",
        "reference",
    }
    rows = WaymaxM5MetricParityAdapter().evaluate_case(
        case,
        parity_index=0,
        executions=executions,
    )
    assert len(rows) == 9
    assert all(row.status == "accepted" for row in rows)
    assert all(row.compared_components > 0 for row in rows)
    np.testing.assert_array_equal(np.asarray(state.sim_trajectory.x), source_x_before)
    np.testing.assert_array_equal(np.asarray(state.timestep), source_timestep_before)


def test_native_parity_rejects_a_scored_source_cadence_drift() -> None:
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("waymax")
    state, scenario = _synthetic_native_fixture()
    timestamps = np.asarray(state.log_trajectory.timestamp_micros).copy()
    timestamps[1, 11] += 1
    contradicted_log = replace(
        state.log_trajectory,
        timestamp_micros=jnp.asarray(timestamps),
    )
    contradicted_state = replace(state, log_trajectory=contradicted_log)
    record = WaymaxRecord(
        scenario_id=scenario.scenario_id,
        state=contradicted_state,
        audit={},
        shard_suffix="00000",
        record_ordinal=0,
        shard_sha256="a" * 64,
        dataset_config_fingerprint="b" * 64,
    )
    case = EvaluationCase(
        cohort_index=0,
        scenario=scenario,
        reference_payload=record,
    )
    with pytest.raises(M5WaymaxEvaluationError, match="source_cadence_drift"):
        WaymaxM5MetricParityAdapter().evaluate_case(
            case,
            parity_index=0,
            executions=_policy_executions(scenario),
        )
