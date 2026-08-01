"""M7 invariance probes: metrics must not change under semantics-preserving transforms.

A metric whose value moves when world agents are reordered or the whole scene is rigidly
translated is buggy. The harness measures per-metric deltas; the non-tautology test proves
it flags a genuinely semantics-breaking transform (translating only the rollout).
"""
from __future__ import annotations

import numpy as np

from evalsim import Agent, AgentType, MapPolyline, MapType, Rollout, Scenario
from evalsim.metrics.m5 import (
    OrientedBoxOverlapRateMetric,
    PositionErrorMetric,
    WaymaxKinematicInfeasibilityRateMetric,
)
from evalsim.stress.invariance import (
    AgentPermutationProbe,
    InvarianceResult,
    RolloutOnlyTranslationProbe,
    RolloutOnlyVelocityImpulseProbe,
    TranslationProbe,
    check_invariance,
    invariance_matrix,
)


def _agent(i, t, **s) -> Agent:
    n = len(t)
    d = {"x": 0.0, "y": 0.0, "heading": 0.0, "vx": 0.0, "vy": 0.0}
    d.update(s)
    arr = lambda v, dt=float: (  # noqa: E731
        np.full(n, v, dtype=dt) if np.ndim(v) == 0 else np.array(v, dtype=dt)
    )
    return Agent(
        i, AgentType.VEHICLE, arr(s.get("valid", True), bool),
        arr(d["x"]), arr(d["y"]), arr(d["heading"]), arr(d["vx"]), arr(d["vy"]),
        2.0, 2.0,
    )


def _case(
    case_index: int = 0,
    n_world: int = 4,
    steps: int = 6,
    current_index: int = 1,
):
    """One clean, separated constant-velocity construct-audit case."""
    t = np.arange(steps, dtype=np.float64) * 0.1
    src = [_agent(0, t, x=-100.0 - case_index)]
    for k in range(n_world):
        src.append(
            _agent(
                10 + k,
                t,
                x=(2.0 + k) * t + case_index,
                y=50.0 * k,
                vx=2.0 + k,
            )
        )
    cand = [_agent(agent.id, t, valid=agent.valid, x=agent.x, y=agent.y,
                   heading=agent.heading, vx=agent.vx, vy=agent.vy)
            for agent in src]
    scenario = Scenario(
        scenario_id=f"m7-construct-{case_index}",
        timestamps=np.array(t, copy=True),
        agents=src,
        ego_index=0,
        metadata={
            "source": "m7_construct",
            "current_index": current_index,
            "case_index": case_index,
        },
    )
    rollout = Rollout(
        scenario_id=scenario.scenario_id,
        sim_name="m7_construct_clean",
        sim_version="1.0.0",
        seed=0,
        timestamps=np.array(t, copy=True),
        agents=cand,
    )
    return scenario, rollout


_METRICS = [
    PositionErrorMetric(),
    OrientedBoxOverlapRateMetric(),
    WaymaxKinematicInfeasibilityRateMetric(),
]


def _assert_agent_equal(left: Agent, right: Agent) -> None:
    assert left.id == right.id
    assert left.type == right.type
    assert left.length == right.length
    assert left.width == right.width
    for field in ("valid", "x", "y", "heading", "vx", "vy"):
        assert np.array_equal(getattr(left, field), getattr(right, field))


def _assert_scenario_equal(left: Scenario, right: Scenario) -> None:
    assert left.scenario_id == right.scenario_id
    assert left.ego_index == right.ego_index
    assert left.metadata == right.metadata
    assert np.array_equal(left.timestamps, right.timestamps)
    assert len(left.agents) == len(right.agents)
    for left_agent, right_agent in zip(left.agents, right.agents, strict=True):
        _assert_agent_equal(left_agent, right_agent)
    assert len(left.map) == len(right.map)
    for left_feature, right_feature in zip(left.map, right.map, strict=True):
        assert left_feature.type == right_feature.type
        assert np.array_equal(left_feature.xy, right_feature.xy)


def _assert_rollout_equal(left: Rollout, right: Rollout) -> None:
    assert left.scenario_id == right.scenario_id
    assert left.sim_name == right.sim_name
    assert left.sim_version == right.sim_version
    assert left.seed == right.seed
    assert left.perturbation == right.perturbation
    assert left.metadata == right.metadata
    assert np.array_equal(left.timestamps, right.timestamps)
    assert len(left.agents) == len(right.agents)
    for left_agent, right_agent in zip(left.agents, right.agents, strict=True):
        _assert_agent_equal(left_agent, right_agent)


def _assert_rollout_defensive_copy(original: Rollout, transformed: Rollout) -> None:
    assert transformed is not original
    assert transformed.scenario_id == original.scenario_id
    assert transformed.sim_name == original.sim_name
    assert transformed.sim_version == original.sim_version
    assert transformed.seed == original.seed
    assert transformed.perturbation == original.perturbation
    assert transformed.metadata == original.metadata
    assert transformed.metadata is not original.metadata
    assert np.array_equal(transformed.timestamps, original.timestamps)
    assert not np.shares_memory(transformed.timestamps, original.timestamps)
    assert len(transformed.agents) == len(original.agents)
    for transformed_agent in transformed.agents:
        assert all(
            transformed_agent is not original_agent
            for original_agent in original.agents
        )
        for field in ("valid", "x", "y", "heading", "vx", "vy"):
            assert all(
                not np.shares_memory(
                    getattr(transformed_agent, field), getattr(original_agent, field)
                )
                for original_agent in original.agents
            )


def test_permutation_and_translation_preserve_metric_values() -> None:
    cases = [_case(case_index) for case_index in range(3)]
    probes = [
        AgentPermutationProbe(),
        TranslationProbe(dx=5.0, dy=-3.0),
    ]
    results = invariance_matrix(_METRICS, cases, probes, seed=4, tol=1e-6)
    assert len(results) == 3 * 2 * 3
    for r in results:
        assert isinstance(r, InvarianceResult)
        assert r.invariant, (
            f"{r.metric_name} not invariant under {r.probe_name}: delta={r.delta}"
        )


def test_registered_permutation_and_translation_are_nontrivial_exact_transforms() -> None:
    scenario, rollout = _case()
    permuted_scenario, permuted_rollout = AgentPermutationProbe().apply(
        scenario, rollout, seed=4
    )

    expected_ids = [0, 13, 10, 11, 12]
    assert [agent.id for agent in permuted_scenario.agents] == expected_ids
    assert [agent.id for agent in permuted_rollout.agents] == expected_ids
    scenario_by_id = {agent.id: agent for agent in scenario.agents}
    rollout_by_id = {agent.id: agent for agent in rollout.agents}
    for agent in permuted_scenario.agents:
        _assert_agent_equal(agent, scenario_by_id[agent.id])
    for agent in permuted_rollout.agents:
        _assert_agent_equal(agent, rollout_by_id[agent.id])
    _assert_rollout_defensive_copy(rollout, permuted_rollout)

    translated_scenario, translated_rollout = TranslationProbe(
        dx=5.0, dy=-3.0
    ).apply(scenario, rollout, seed=4)
    for original_agents, transformed_agents in (
        (scenario.agents, translated_scenario.agents),
        (rollout.agents, translated_rollout.agents),
    ):
        for original, transformed in zip(
            original_agents, transformed_agents, strict=True
        ):
            assert transformed.id == original.id
            assert transformed.type == original.type
            assert transformed.length == original.length
            assert transformed.width == original.width
            assert np.array_equal(transformed.x, original.x + 5.0)
            assert np.array_equal(transformed.y, original.y - 3.0)
            for field in ("valid", "heading", "vx", "vy"):
                assert np.array_equal(
                    getattr(transformed, field), getattr(original, field)
                )
    _assert_rollout_defensive_copy(rollout, translated_rollout)


def test_harness_flags_semantics_breaking_transform() -> None:
    # Translating ONLY the rollout genuinely changes position error -> must be flagged.
    for case_index in range(3):
        scenario, rollout = _case(case_index)
        result = check_invariance(
            PositionErrorMetric(),
            scenario,
            rollout,
            RolloutOnlyTranslationProbe(dx=5.0, dy=-3.0),
            seed=0,
            tol=1e-6,
        )
        assert result.invariant is False
        assert result.delta > 1e-6


def test_translation_moves_map_geometry_without_mutating_inputs() -> None:
    scenario, rollout = _case()
    lane_xy = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    scenario.map = [MapPolyline(type=MapType.LANE, xy=np.array(lane_xy, copy=True))]
    scenario_before, rollout_before = _case()
    scenario_before.map = [
        MapPolyline(type=MapType.LANE, xy=np.array(lane_xy, copy=True))
    ]

    translated_scenario, translated_rollout = TranslationProbe(
        dx=5.0, dy=-3.0
    ).apply(scenario, rollout, seed=4)

    assert len(translated_scenario.map) == 1
    assert translated_scenario.map[0].type == MapType.LANE
    assert np.array_equal(
        translated_scenario.map[0].xy,
        lane_xy + np.array([5.0, -3.0]),
    )
    assert translated_scenario.map[0] is not scenario.map[0]
    assert not np.shares_memory(translated_scenario.map[0].xy, scenario.map[0].xy)
    _assert_scenario_equal(scenario, scenario_before)
    _assert_rollout_equal(rollout, rollout_before)
    _assert_rollout_defensive_copy(rollout, translated_rollout)


def test_velocity_impulse_changes_kinematic_metric_only_at_registered_frame() -> None:
    for case_index in range(3):
        scenario, rollout = _case(case_index)
        scenario_before, rollout_before = _case(case_index)
        assert scenario.metadata["current_index"] == 1
        assert rollout.num_steps == 6
        expected_frame = 3
        transformed_scenario, transformed_rollout = (
            RolloutOnlyVelocityImpulseProbe(impulse_mps=100.0).apply(
                scenario, rollout, seed=9
            )
        )

        _assert_scenario_equal(transformed_scenario, scenario)
        _assert_rollout_defensive_copy(rollout, transformed_rollout)
        for position, (original, transformed) in enumerate(
            zip(rollout.agents, transformed_rollout.agents, strict=True)
        ):
            for field in ("valid", "x", "y", "heading", "vy"):
                assert np.array_equal(getattr(original, field), getattr(transformed, field))
            expected_vx = np.array(original.vx, copy=True)
            if position == 1:
                expected_vx[expected_frame] += 100.0
            assert np.array_equal(transformed.vx, expected_vx)

        result = check_invariance(
            WaymaxKinematicInfeasibilityRateMetric(),
            scenario,
            rollout,
            RolloutOnlyVelocityImpulseProbe(),
            seed=9,
            tol=1e-6,
        )
        assert result.invariant is False
        assert result.delta > 1e-6
        _assert_scenario_equal(scenario, scenario_before)
        _assert_rollout_equal(rollout, rollout_before)


def test_probes_do_not_mutate_inputs() -> None:
    scenario, rollout = _case()
    scenario_before, rollout_before = _case()
    AgentPermutationProbe().apply(scenario, rollout, seed=1)
    TranslationProbe(dx=9.0, dy=9.0).apply(scenario, rollout, seed=1)
    RolloutOnlyTranslationProbe(dx=9.0, dy=9.0).apply(scenario, rollout, seed=1)
    RolloutOnlyVelocityImpulseProbe().apply(scenario, rollout, seed=1)
    _assert_scenario_equal(scenario, scenario_before)
    _assert_rollout_equal(rollout, rollout_before)
