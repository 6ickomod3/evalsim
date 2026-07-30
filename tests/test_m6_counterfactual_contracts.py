"""M6 source-neutral counterfactual contract acceptance tests."""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from evalsim import Agent, AgentType, MapPolyline, MapType, Rollout, Scenario
from evalsim.contracts.counterfactual import (
    CONFIGURATION_DOMAIN,
    M6_PLAN_FRAME_COUNT,
    PLAN_DOMAIN,
    CounterfactualPair,
    EgoInterventionSpec,
    EgoTrajectoryPlan,
    FeasibilityAudit,
    InterventionEligibility,
    PairedMetric,
    PairedMetricResult,
    ScenarioSnapshot,
    canonical_configuration_bytes,
    canonical_configuration_json,
    evaluate_paired_metric,
)
from evalsim.contracts.metric import MetricSpec
from evalsim.perturb.m6 import (
    compile_identity_plan,
    compile_longitudinal_brake_pulse_plan,
)


def _spec(
    *,
    family: str = "longitudinal_brake_pulse",
    dose: float = 2.0,
) -> EgoInterventionSpec:
    return EgoInterventionSpec(
        family=family,
        version="v1",
        dose=dose,
        duration_s=0.0 if family == "identity" else 1.0,
        parameters={"speed_floor_mps": 0.0, "nested": {"enabled": True}},
        access_class="logged_future_privileged",
    )


def _audit() -> FeasibilityAudit:
    return FeasibilityAudit.accepted(
        {
            "finite": True,
            "speed": True,
            "acceleration": True,
            "yaw_rate": True,
            "continuity": True,
        },
        details={"interval_count": 40},
    )


def _plan(
    *,
    spec: EgoInterventionSpec | None = None,
    timestamps: np.ndarray | None = None,
    x: np.ndarray | None = None,
) -> EgoTrajectoryPlan:
    timestamps = (
        np.arange(M6_PLAN_FRAME_COUNT, dtype=np.float64) * 0.1
        if timestamps is None
        else timestamps
    )
    x = (
        np.arange(M6_PLAN_FRAME_COUNT, dtype=np.float64) * 0.5
        if x is None
        else x
    )
    return EgoTrajectoryPlan(
        spec=_spec() if spec is None else spec,
        timestamps=timestamps,
        valid=np.ones(M6_PLAN_FRAME_COUNT, dtype=bool),
        applied=np.r_[
            np.array([False]),
            np.ones(M6_PLAN_FRAME_COUNT - 1, dtype=bool),
        ],
        x=x,
        y=np.zeros(M6_PLAN_FRAME_COUNT),
        heading=np.zeros(M6_PLAN_FRAME_COUNT),
        vx=np.full(M6_PLAN_FRAME_COUNT, 5.0),
        vy=np.zeros(M6_PLAN_FRAME_COUNT),
        realization_type="logged_future_privileged",
        feasibility=_audit(),
    )


def test_intervention_spec_is_detached_frozen_and_canonical() -> None:
    parameters = {
        "z": [1, {"value": 2.0}],
        "ascii": "é",
    }
    spec = EgoInterventionSpec(
        family="longitudinal_brake_pulse",
        version="v1",
        dose=2,
        duration_s=1,
        parameters=parameters,
        access_class="logged_future_privileged",
    )
    parameters["z"][1]["value"] = 99.0

    expected_json = json.dumps(
        {
            "access_class": "logged_future_privileged",
            "dose": 2.0,
            "duration_s": 1.0,
            "family": "longitudinal_brake_pulse",
            "parameters": {
                "ascii": "é",
                "z": [1, {"value": 2.0}],
            },
            "schema_version": "1.0.0",
            "version": "v1",
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    expected_bytes = CONFIGURATION_DOMAIN + b"\x00" + expected_json.encode()

    assert spec.canonical_json == expected_json
    assert spec.canonical_bytes == expected_bytes
    assert spec.configuration_fingerprint == hashlib.sha256(
        expected_bytes
    ).hexdigest()
    assert spec.intervention_id == "longitudinal_brake_pulse/v1"
    assert spec.parameters["z"][1]["value"] == 2.0
    with pytest.raises(TypeError):
        spec.parameters["new"] = 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        spec.dose = 4.0  # type: ignore[misc]


def test_canonical_json_rejects_ambiguous_or_nonfinite_values() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    for payload in (
        {"bad": np.inf},
        {1: "non-string-key"},
        cyclic,
        {"bad": object()},
    ):
        with pytest.raises(ValueError):
            canonical_configuration_json(payload)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        canonical_configuration_bytes({"ok": 1}, domain="not-áscii")
    with pytest.raises(ValueError):
        canonical_configuration_bytes({"ok": 1}, domain=b"bad\x00domain")


def test_intervention_spec_roundtrip_rejects_tamper_and_schema_drift() -> None:
    spec = _spec()
    assert EgoInterventionSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()

    tampered = spec.to_dict()
    tampered["dose"] = 4.0
    with pytest.raises(ValueError, match="fingerprint"):
        EgoInterventionSpec.from_dict(tampered)

    extra = spec.to_dict()
    extra["unregistered"] = True
    with pytest.raises(ValueError, match="fields"):
        EgoInterventionSpec.from_dict(extra)

    wrong_schema = spec.to_dict()
    wrong_schema["schema_version"] = "2.0.0"
    with pytest.raises(ValueError, match="schema"):
        EgoInterventionSpec.from_dict(wrong_schema)


def test_feasibility_audit_implications_and_detachment() -> None:
    checks = {"finite": True, "speed": True}
    details = {"limits": [-8.0, 4.0]}
    audit = FeasibilityAudit.accepted(checks, details=details)
    checks["finite"] = False
    details["limits"][0] = -99.0

    assert audit.passed
    assert audit.checks == {"finite": True, "speed": True}
    assert audit.details["limits"] == (-8.0, 4.0)
    assert FeasibilityAudit.from_dict(audit.to_dict()).to_dict() == audit.to_dict()

    failed = FeasibilityAudit.rejected(
        "primary_ego_plan_infeasible",
        {"finite": True, "speed": False},
    )
    assert not failed.passed
    with pytest.raises(ValueError):
        FeasibilityAudit(True, {"finite": False})
    with pytest.raises(ValueError):
        FeasibilityAudit(False, {"finite": True}, "infeasible")
    with pytest.raises(ValueError):
        FeasibilityAudit(False, {"finite": False})


def test_eligibility_freezes_exact_40_transition_window_and_target() -> None:
    accepted = InterventionEligibility.accepted((10, 50), 4)
    assert accepted.current_index == 10
    assert accepted.stop_index == 50
    assert accepted.target_index == 4
    assert (
        InterventionEligibility.from_dict(accepted.to_dict()).to_dict()
        == accepted.to_dict()
    )

    rejected = InterventionEligibility.rejected(
        "no_stable_aligned_follower",
        (10, 50),
    )
    assert rejected.reason == "no_stable_aligned_follower"
    assert rejected.target_index is None
    with pytest.raises(ValueError, match="40"):
        InterventionEligibility.accepted((10, 49), 4)
    with pytest.raises(ValueError):
        InterventionEligibility(True, "bad", (10, 50), 4)
    with pytest.raises(ValueError):
        InterventionEligibility(True, None, (10, 50), None)
    with pytest.raises(ValueError):
        InterventionEligibility(False, None, (10, 50), None)
    with pytest.raises(ValueError, match="registered"):
        InterventionEligibility.rejected("posthoc_reason", (10, 50))


def test_plan_canonical_bytes_follow_exact_registered_layout() -> None:
    plan = _plan()
    config = plan.configuration_fingerprint.encode("ascii")
    realization = b"logged_future_privileged"
    expected = b"".join(
        (
            PLAN_DOMAIN,
            b"\x00",
            struct.pack(">Q", len(config)),
            config,
            struct.pack(">Q", len(realization)),
            realization,
            struct.pack(">Q", M6_PLAN_FRAME_COUNT),
            plan.valid.astype(np.uint8).tobytes(),
            plan.applied.astype(np.uint8).tobytes(),
            *(
                np.asarray(getattr(plan, name), dtype="<f8").tobytes(order="C")
                for name in ("timestamps", "x", "y", "heading", "vx", "vy")
            ),
        )
    )

    assert plan.canonical_bytes == expected
    assert plan.plan_fingerprint == hashlib.sha256(expected).hexdigest()
    assert plan.perturbation_identity == (
        f"longitudinal_brake_pulse/v1@sha256:{plan.plan_fingerprint}"
    )
    # The frame count and both masks use fixed widths before any float payload.
    mask_offset = (
        len(PLAN_DOMAIN)
        + 1
        + 8
        + len(config)
        + 8
        + len(realization)
        + 8
    )
    assert expected[mask_offset : mask_offset + 41] == b"\x01" * 41
    assert expected[mask_offset + 41] == 0
    assert expected[mask_offset + 42 : mask_offset + 82] == b"\x01" * 40


def test_plan_is_detached_bytes_backed_and_roundtrips_exact_bits() -> None:
    source_x = np.arange(M6_PLAN_FRAME_COUNT, dtype=np.float64)
    source_x[3] = -0.0
    plan = _plan(x=source_x)
    source_x[:] = 999.0

    assert np.signbit(plan.x[3])
    assert plan.x[4] == 4.0
    with pytest.raises(ValueError):
        plan.x.setflags(write=True)
    with pytest.raises(ValueError):
        plan.x[0] = 1.0

    restored = EgoTrajectoryPlan.deserialize(plan.serialize())
    assert restored.plan_fingerprint == plan.plan_fingerprint
    assert restored.perturbation_identity == plan.perturbation_identity
    for name in (
        "valid",
        "applied",
        "timestamps",
        "x",
        "y",
        "heading",
        "vx",
        "vy",
    ):
        np.testing.assert_array_equal(getattr(restored, name), getattr(plan, name))
        assert getattr(restored, name).tobytes() == getattr(plan, name).tobytes()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("frame_count", 40),
        ("plan_fingerprint", "0" * 64),
        (
            "perturbation_identity",
            "longitudinal_brake_pulse/v1@sha256:" + "0" * 64,
        ),
    ],
)
def test_plan_deserialization_rejects_tampered_identity_fields(
    field: str,
    replacement: object,
) -> None:
    payload = _plan().to_dict()
    payload[field] = replacement
    with pytest.raises(ValueError):
        EgoTrajectoryPlan.from_dict(payload)


def test_plan_deserialization_rejects_array_and_envelope_tamper() -> None:
    plan = _plan()
    payload = plan.to_dict()
    encoded = payload["x_f64le_hex"]
    assert isinstance(encoded, str)
    payload["x_f64le_hex"] = (
        ("1" if encoded[0] != "1" else "2") + encoded[1:]
    )
    with pytest.raises(ValueError, match="fingerprint"):
        EgoTrajectoryPlan.from_dict(payload)

    uppercase = plan.to_dict()
    uppercase["x_f64le_hex"] = str(uppercase["x_f64le_hex"]).upper()
    with pytest.raises(ValueError, match="lowercase"):
        EgoTrajectoryPlan.from_dict(uppercase)

    nonbinary_mask = plan.to_dict()
    valid_hex = str(nonbinary_mask["valid_u8_hex"])
    nonbinary_mask["valid_u8_hex"] = "02" + valid_hex[2:]
    with pytest.raises(ValueError, match="0 or 1"):
        EgoTrajectoryPlan.from_dict(nonbinary_mask)

    with pytest.raises(ValueError, match="domain"):
        EgoTrajectoryPlan.deserialize(b"wrong\x00" + plan.to_json().encode())
    pretty = json.dumps(plan.to_dict(), indent=2, sort_keys=True)
    with pytest.raises(ValueError, match="canonical"):
        EgoTrajectoryPlan.from_json(pretty)


def test_plan_deserialization_rejects_feasibility_tamper() -> None:
    payload = _plan().to_dict()
    payload["feasibility"]["details"]["interval_count"] = 999
    with pytest.raises(ValueError, match="audit_fingerprint"):
        EgoTrajectoryPlan.from_dict(payload)


def test_plan_rejects_bad_shape_masks_finiteness_and_failed_audit() -> None:
    base = {
        "spec": _spec(),
        "timestamps": np.arange(41) * 0.1,
        "valid": np.ones(41, dtype=bool),
        "applied": np.r_[False, np.ones(40, dtype=bool)],
        "x": np.arange(41, dtype=float),
        "y": np.zeros(41),
        "heading": np.zeros(41),
        "vx": np.ones(41),
        "vy": np.zeros(41),
        "realization_type": "logged_future_privileged",
        "feasibility": _audit(),
    }
    for field, value in (
        ("x", np.zeros(40)),
        ("valid", np.r_[False, np.ones(40, dtype=bool)]),
        ("applied", np.ones(41, dtype=bool)),
        ("timestamps", np.r_[0.0, 0.0, np.arange(2, 41) * 0.1]),
        ("heading", np.r_[4.0, np.zeros(40)]),
        ("vx", np.r_[np.nan, np.ones(40)]),
        (
            "feasibility",
            FeasibilityAudit.rejected(
                "primary_ego_plan_infeasible",
                {"finite": False},
            ),
        ),
    ):
        kwargs = dict(base)
        kwargs[field] = value
        with pytest.raises(ValueError):
            EgoTrajectoryPlan(**kwargs)


def _agent(
    agent_id: int,
    steps: int,
    *,
    x_offset: float,
) -> Agent:
    timestamps = np.arange(steps, dtype=float) * 0.1
    return Agent(
        id=agent_id,
        type=AgentType.VEHICLE,
        valid=np.ones(steps, dtype=bool),
        x=x_offset + timestamps * 5.0,
        y=np.zeros(steps),
        heading=np.zeros(steps),
        vx=np.full(steps, 5.0),
        vy=np.zeros(steps),
        length=4.5,
        width=2.0,
    )


def _rollout_metadata(
    *,
    current_index: int,
    clamp_acceleration: int = 0,
) -> dict[str, object]:
    return {
        "engine": {"name": "numpy_rollout_engine", "version": "0.1.0"},
        "dynamics": {
            "name": "point_mass",
            "version": "1.0.0",
            "integration": "midpoint_heading_trapezoidal_speed",
            "limits": {"max_speed": 60.0},
            "clamp_counts": {
                "acceleration": clamp_acceleration,
                "speed": 0,
            },
        },
        "policy": {
            "name": "idm",
            "version": "0.1.0",
            "deterministic": True,
            "required_features": [],
            "supported_agent_types": ["vehicle"],
            "params": {},
            "known_limitations": [],
            "fallback_policy": None,
        },
        "ego_control": "typed_ego_plan",
        "rollout_start_index": current_index,
        "controlled_agent_ids": [202],
        "agent_control_modes": {
            "101": "typed_ego_plan",
            "202": "idm",
        },
        "scenario_source": "unit",
        "scenario_source_fingerprint": "f" * 64,
    }


def _pair_inputs(
    *,
    source_steps: int = 50,
    current_index: int = 5,
) -> tuple[
    Scenario,
    Rollout,
    Rollout,
    InterventionEligibility,
    EgoTrajectoryPlan,
    EgoTrajectoryPlan,
]:
    stop_index = current_index + 40
    timestamps = np.arange(source_steps, dtype=float) * 0.1
    source = Scenario(
        scenario_id="scene",
        timestamps=timestamps,
        agents=[
            _agent(101, source_steps, x_offset=10.0),
            _agent(202, source_steps, x_offset=0.0),
        ],
        map=[
            MapPolyline(
                MapType.LANE,
                np.array([[0.0, 0.0], [100.0, 0.0]]),
            )
        ],
        ego_index=0,
        metadata={
            "source": "unit",
            "source_fingerprint": "f" * 64,
            "current_index": current_index,
        },
    )
    identity_plan = compile_identity_plan(source)
    brake_plan = compile_longitudinal_brake_pulse_plan(source, 2.0)
    baseline_agents = [
        Agent(
            id=agent.id,
            type=agent.type,
            valid=agent.valid[: stop_index + 1].copy(),
            x=agent.x[: stop_index + 1].copy(),
            y=agent.y[: stop_index + 1].copy(),
            heading=agent.heading[: stop_index + 1].copy(),
            vx=agent.vx[: stop_index + 1].copy(),
            vy=agent.vy[: stop_index + 1].copy(),
            length=agent.length,
            width=agent.width,
        )
        for agent in source.agents
    ]
    treatment_agents = [
        Agent(
            id=agent.id,
            type=agent.type,
            valid=agent.valid.copy(),
            x=agent.x.copy(),
            y=agent.y.copy(),
            heading=agent.heading.copy(),
            vx=agent.vx.copy(),
            vy=agent.vy.copy(),
            length=agent.length,
            width=agent.width,
        )
        for agent in baseline_agents
    ]
    plan_slice = slice(current_index, stop_index + 1)
    for field_name in ("valid", "x", "y", "heading", "vx", "vy"):
        getattr(treatment_agents[0], field_name)[plan_slice] = getattr(
            brake_plan,
            field_name,
        )
    treatment_agents[1].x[current_index + 2 :] -= 0.25
    baseline = Rollout(
        scenario_id=source.scenario_id,
        sim_name="idm",
        sim_version="0.1.0",
        seed=0,
        timestamps=timestamps[: stop_index + 1],
        agents=baseline_agents,
        perturbation=identity_plan.perturbation_identity,
        metadata=_rollout_metadata(current_index=current_index),
    )
    intervention = Rollout(
        scenario_id=source.scenario_id,
        sim_name="idm",
        sim_version="0.1.0",
        seed=0,
        timestamps=timestamps[: stop_index + 1],
        agents=treatment_agents,
        perturbation=brake_plan.perturbation_identity,
        metadata=_rollout_metadata(
            current_index=current_index,
            clamp_acceleration=1,
        ),
    )
    eligibility = InterventionEligibility.accepted(
        (current_index, stop_index),
        1,
    )
    return (
        source,
        baseline,
        intervention,
        eligibility,
        identity_plan,
        brake_plan,
    )


def test_pair_snapshots_full_source_and_typed_prefix_without_aliasing() -> None:
    (
        source,
        baseline,
        treatment,
        eligibility,
        identity_plan,
        brake_plan,
    ) = _pair_inputs()
    original_source_value = float(source.agents[0].x[0])
    original_treatment_value = float(treatment.agents[1].x[-1])
    pair = CounterfactualPair(
        scenario=source,
        baseline=baseline,
        intervention=treatment,
        baseline_plan=identity_plan,
        intervention_plan=brake_plan,
        eligibility=eligibility,
        intervention_identity=brake_plan.perturbation_identity,
    )

    assert isinstance(pair.scenario, ScenarioSnapshot)
    assert pair.scenario.num_steps == 50
    assert pair.baseline.num_steps == pair.intervention.num_steps == 46
    assert pair.scenario.agents[0].valid.shape == (50,)
    assert pair.baseline.agents[0].valid.shape == (46,)
    assert (
        pair.intervention.metadata["dynamics"]["clamp_counts"]["acceleration"]
        == 1
    )
    # Outcome clamp counts may differ without weakening configuration pairing.
    pair.revalidate()

    source.agents[0].x[0] = 999.0
    treatment.agents[1].x[-1] = 999.0
    treatment.metadata["policy"]["name"] = "mutated"
    assert pair.scenario.agents[0].x[0] == original_source_value
    assert pair.intervention.agents[1].x[-1] == original_treatment_value
    assert pair.intervention.metadata["policy"]["name"] == "idm"
    with pytest.raises(ValueError):
        pair.baseline.timestamps.setflags(write=True)


@pytest.mark.parametrize(
    ("location", "value", "match"),
    [
        ("seed", 1, "seed"),
        ("sim_version", "9.9.9", "configuration"),
        ("policy", {"name": "cv"}, "configuration"),
        ("engine", {"name": "different"}, "configuration"),
        ("dynamics_limits", {"max_speed": 30.0}, "configuration"),
        ("ego_control", "logged", "configuration"),
        ("rollout_start_index", 4, "configuration"),
        ("controlled_agent_ids", [101], "configuration"),
        ("agent_control_modes", {"202": "cv"}, "configuration"),
        ("scenario_source", "other", "configuration"),
        ("source_fingerprint", "e" * 64, "configuration"),
    ],
)
def test_pair_rejects_asymmetric_configuration(
    location: str,
    value: object,
    match: str,
) -> None:
    (
        source,
        baseline,
        treatment,
        eligibility,
        identity_plan,
        brake_plan,
    ) = _pair_inputs()
    if location == "seed":
        treatment.seed = value  # type: ignore[assignment]
    elif location == "sim_version":
        treatment.sim_version = value  # type: ignore[assignment]
    elif location == "dynamics_limits":
        treatment.metadata["dynamics"]["limits"] = value
    elif location == "source_fingerprint":
        treatment.metadata["scenario_source_fingerprint"] = value
    else:
        treatment.metadata[location] = value
    with pytest.raises(ValueError, match=match):
        CounterfactualPair(
            source,
            baseline,
            treatment,
            identity_plan,
            brake_plan,
            eligibility,
            brake_plan.perturbation_identity,
        )


def test_pair_rejects_prefix_mask_history_target_and_identity_drift() -> None:
    (
        source,
        baseline,
        treatment,
        eligibility,
        identity_plan,
        brake_plan,
    ) = _pair_inputs()

    treatment.agents[1].valid[-1] = False
    with pytest.raises(ValueError, match="lifecycle"):
        CounterfactualPair(
            source,
            baseline,
            treatment,
            identity_plan,
            brake_plan,
            eligibility,
            brake_plan.perturbation_identity,
        )
    treatment.agents[1].valid[-1] = True

    treatment.agents[1].x[eligibility.current_index] += 1.0
    with pytest.raises(ValueError, match="history"):
        CounterfactualPair(
            source,
            baseline,
            treatment,
            identity_plan,
            brake_plan,
            eligibility,
            brake_plan.perturbation_identity,
        )
    treatment.agents[1].x[eligibility.current_index] -= 1.0

    with pytest.raises(ValueError, match="world agent"):
        CounterfactualPair(
            source,
            baseline,
            treatment,
            identity_plan,
            brake_plan,
            InterventionEligibility.accepted(
                eligibility.analysis_window,
                source.ego_index,
            ),
            brake_plan.perturbation_identity,
        )
    with pytest.raises(ValueError, match="typed ego plan"):
        CounterfactualPair(
            source,
            baseline,
            treatment,
            identity_plan,
            brake_plan,
            eligibility,
            "free text",
        )


def test_pair_rejects_label_only_brake_and_impossible_t_plus_one_response() -> None:
    (
        source,
        baseline,
        treatment,
        eligibility,
        identity_plan,
        brake_plan,
    ) = _pair_inputs()
    ego_index = source.ego_index
    for field_name in ("valid", "x", "y", "heading", "vx", "vy"):
        setattr(
            treatment.agents[ego_index],
            field_name,
            np.array(getattr(baseline.agents[ego_index], field_name), copy=True),
        )
    # Retaining the brake identity while replacing its tensors with the sham must
    # never create an apparently valid treatment pair.
    with pytest.raises(ValueError, match="ego tensors"):
        CounterfactualPair(
            source,
            baseline,
            treatment,
            identity_plan,
            brake_plan,
            eligibility,
            brake_plan.perturbation_identity,
        )

    (
        source,
        baseline,
        treatment,
        eligibility,
        identity_plan,
        brake_plan,
    ) = _pair_inputs()
    treatment.agents[1].x[eligibility.current_index + 1] -= 1.0
    with pytest.raises(ValueError, match=r"t\+2"):
        CounterfactualPair(
            source,
            baseline,
            treatment,
            identity_plan,
            brake_plan,
            eligibility,
            brake_plan.perturbation_identity,
        )


def test_pair_rejects_incomplete_static_control_mode_mask() -> None:
    (
        source,
        baseline,
        treatment,
        eligibility,
        identity_plan,
        brake_plan,
    ) = _pair_inputs()
    incomplete = {"202": "idm"}
    baseline.metadata["agent_control_modes"] = incomplete
    treatment.metadata["agent_control_modes"] = incomplete
    with pytest.raises(ValueError, match="agent_control_modes"):
        CounterfactualPair(
            source,
            baseline,
            treatment,
            identity_plan,
            brake_plan,
            eligibility,
            brake_plan.perturbation_identity,
        )


def test_pair_revalidation_detects_postconstruction_integrity_bypass() -> None:
    (
        source,
        baseline,
        treatment,
        eligibility,
        identity_plan,
        brake_plan,
    ) = _pair_inputs()
    pair = CounterfactualPair(
        source,
        baseline,
        treatment,
        identity_plan,
        brake_plan,
        eligibility,
        brake_plan.perturbation_identity,
    )
    object.__setattr__(
        pair.intervention.agents[1],
        "x",
        np.zeros(pair.intervention.num_steps),
    )
    with pytest.raises(ValueError, match="snapshot was mutated"):
        pair.revalidate()


class _ProgressMetric:
    spec = MetricSpec(
        name="target_progress_loss_m",
        version="1.0.0",
        value_unit="metres",
        unit_of_analysis="scenario",
        direction="neutral",
        aggregation="mean",
        agent_scope="world",
    )

    def compute(self, pair: CounterfactualPair) -> PairedMetricResult:
        target = pair.eligibility.target_index
        assert target is not None
        value = float(
            pair.baseline.agents[target].x[-1]
            - pair.intervention.agents[target].x[-1]
        )
        return PairedMetricResult(
            metric_name=self.spec.name,
            metric_version=self.spec.version,
            scenario_id=pair.scenario.scenario_id,
            intervention_identity=pair.intervention_identity,
            value=value,
            details={"target_index": target},
        )


def test_paired_metric_protocol_revalidates_and_binds_result() -> None:
    (
        source,
        baseline,
        treatment,
        eligibility,
        identity_plan,
        brake_plan,
    ) = _pair_inputs()
    pair = CounterfactualPair(
        source,
        baseline,
        treatment,
        identity_plan,
        brake_plan,
        eligibility,
        brake_plan.perturbation_identity,
    )
    metric = _ProgressMetric()
    assert isinstance(metric, PairedMetric)
    result = evaluate_paired_metric(metric, pair)
    assert result.metric_id == "target_progress_loss_m@1.0.0"
    assert result.value == pytest.approx(0.25)
    assert result.details["target_index"] == 1
    assert PairedMetricResult.from_dict(result.to_dict()).to_dict() == (
        result.to_dict()
    )


def test_paired_metric_wrapper_rejects_mutation_and_misbinding() -> None:
    (
        source,
        baseline,
        treatment,
        eligibility,
        identity_plan,
        brake_plan,
    ) = _pair_inputs()
    pair = CounterfactualPair(
        source,
        baseline,
        treatment,
        identity_plan,
        brake_plan,
        eligibility,
        brake_plan.perturbation_identity,
    )

    class _MutatingMetric(_ProgressMetric):
        def compute(self, pair: CounterfactualPair) -> PairedMetricResult:
            object.__setattr__(
                pair.baseline.agents[1],
                "x",
                np.zeros(pair.baseline.num_steps),
            )
            return super().compute(pair)

    with pytest.raises(ValueError, match="snapshot was mutated"):
        evaluate_paired_metric(_MutatingMetric(), pair)

    (
        source,
        baseline,
        treatment,
        eligibility,
        identity_plan,
        brake_plan,
    ) = _pair_inputs()
    clean_pair = CounterfactualPair(
        source,
        baseline,
        treatment,
        identity_plan,
        brake_plan,
        eligibility,
        brake_plan.perturbation_identity,
    )

    class _WrongResult(_ProgressMetric):
        def compute(self, pair: CounterfactualPair) -> PairedMetricResult:
            result = super().compute(pair)
            return PairedMetricResult(
                metric_name="other_metric",
                metric_version=result.metric_version,
                scenario_id=result.scenario_id,
                intervention_identity=result.intervention_identity,
                value=result.value,
            )

    with pytest.raises(ValueError, match="metric spec"):
        evaluate_paired_metric(_WrongResult(), clean_pair)
