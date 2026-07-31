"""M7 invariance probes: metrics must not change under semantics-preserving transforms.

A metric whose value moves when world agents are reordered or the whole scene is rigidly
translated is buggy. The harness measures per-metric deltas; the non-tautology test proves
it flags a genuinely semantics-breaking transform (translating only the rollout).
"""
from __future__ import annotations

import copy

import numpy as np

from evalsim import Agent, AgentType, Rollout, Scenario
from evalsim.metrics.m5 import (
    OrientedBoxOverlapRateMetric,
    PositionErrorMetric,
    SpeedErrorMetric,
)
from evalsim.stress.invariance import (
    AgentPermutationProbe,
    InvarianceResult,
    RolloutOnlyTranslationProbe,
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


def _case(n_world: int = 4, steps: int = 6, current_index: int = 1):
    """Ego + world agents; rollout deviates slightly so error metrics are nonzero."""
    t = np.arange(steps, dtype=float) * 0.1
    src = [_agent(0, t, x=-40.0)]
    cand = [_agent(0, t, x=-40.0)]
    for k in range(n_world):
        src.append(_agent(10 + k, t, x=(2.0 + k) * t, y=5.0 * k, vx=2.0 + k))
        cand.append(_agent(10 + k, t, x=(2.0 + k) * t + 0.2, y=5.0 * k, vx=2.0 + k + 0.1))
    scenario = Scenario(
        scenario_id="m7-inv", timestamps=np.array(t, copy=True), agents=src,
        ego_index=0, metadata={"source": "unit", "current_index": current_index},
    )
    rollout = Rollout(
        scenario_id="m7-inv", sim_name="candidate", sim_version="1.0.0", seed=0,
        timestamps=np.array(t, copy=True), agents=cand,
    )
    return scenario, rollout


_METRICS = [PositionErrorMetric(), SpeedErrorMetric(), OrientedBoxOverlapRateMetric()]


def test_permutation_and_translation_preserve_metric_values() -> None:
    scenario, rollout = _case()
    probes = [
        AgentPermutationProbe(),
        TranslationProbe(dx=5.0, dy=-3.0),
    ]
    results = invariance_matrix(_METRICS, [(scenario, rollout)], probes, seed=4)
    assert results  # non-empty
    for r in results:
        assert isinstance(r, InvarianceResult)
        assert r.invariant, (
            f"{r.metric_name} not invariant under {r.probe_name}: delta={r.delta}"
        )


def test_harness_flags_semantics_breaking_transform() -> None:
    # Translating ONLY the rollout genuinely changes position error -> must be flagged.
    scenario, rollout = _case()
    result = check_invariance(
        PositionErrorMetric(),
        scenario,
        rollout,
        RolloutOnlyTranslationProbe(dx=5.0, dy=-3.0),
        seed=0,
    )
    assert result.invariant is False
    assert result.delta > 1e-3


def test_probes_do_not_mutate_inputs() -> None:
    scenario, rollout = _case()
    before = copy.deepcopy(rollout.agents[1].x)
    AgentPermutationProbe().apply(scenario, rollout, seed=1)
    TranslationProbe(dx=9.0, dy=9.0).apply(scenario, rollout, seed=1)
    assert np.array_equal(rollout.agents[1].x, before)
