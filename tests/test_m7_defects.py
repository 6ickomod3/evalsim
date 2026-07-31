"""M7 evaluator red-team: analytic oracles for the defect-generator framework.

These tests encode ground truth independent of the implementation:
- severity 0 is a strict identity (byte-identical rollout);
- generators are pure and deterministic given (severity, seed);
- frozen selection is nested (severity monotone -> superset of affected agents);
- the defect manifest carries only sanitized accounting (no native id/coords);
- the corruption is *detected* by the intended M5 metric with a monotone curve.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from evalsim import Agent, AgentType, Rollout, Scenario
from evalsim.metrics.m5 import PositionErrorMetric
from evalsim.metrics.registry import MetricRegistry
from evalsim.stress.defects import (
    Defect,
    DefectManifest,
    DefectRegistry,
    DefectRegistryError,
    DefectSpec,
    FrozenAgentDefect,
)


def _series(value, count: int, *, dtype=float) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        return np.full(count, array.item(), dtype=dtype)
    return np.array(array, dtype=dtype, copy=True)


def _agent(agent_id, timestamps, *, agent_type=AgentType.VEHICLE, **series) -> Agent:
    count = len(timestamps)
    defaults = {"x": 0.0, "y": 0.0, "heading": 0.0, "vx": 0.0, "vy": 0.0}
    defaults.update(series)
    return Agent(
        id=agent_id,
        type=agent_type,
        valid=_series(series.get("valid", True), count, dtype=bool),
        x=_series(defaults["x"], count),
        y=_series(defaults["y"], count),
        heading=_series(defaults["heading"], count),
        vx=_series(defaults["vx"], count),
        vy=_series(defaults["vy"], count),
        length=2.0,
        width=2.0,
    )


def _moving_pair(n_world: int = 4, steps: int = 6, current_index: int = 1):
    """Ego + n moving world vehicles; rollout is a perfect log-replay of the source."""
    t = np.arange(steps, dtype=float) * 0.1
    source = [_agent(0, t, x=-50.0)]  # exogenous ego, far away
    for k in range(n_world):
        source.append(_agent(10 + k, t, x=(2.0 + k) * t, y=float(k), vx=2.0 + k))
    scenario = Scenario(
        scenario_id="m7-unit",
        timestamps=np.array(t, copy=True),
        agents=source,
        ego_index=0,
        metadata={"source": "unit", "current_index": current_index},
    )
    rollout = Rollout(
        scenario_id=scenario.scenario_id,
        sim_name="candidate",
        sim_version="1.0.0",
        seed=0,
        timestamps=np.array(t, copy=True),
        agents=copy.deepcopy(source),
    )
    return scenario, rollout


def _position_error(scenario, rollout) -> float:
    return MetricRegistry([PositionErrorMetric()]).evaluate(scenario, rollout)[0].value


def _rollouts_equal(a: Rollout, b: Rollout) -> bool:
    # A true severity-0 identity preserves the whole contract, not just agent series.
    if (
        a.scenario_id != b.scenario_id
        or a.sim_name != b.sim_name
        or a.sim_version != b.sim_version
        or a.seed != b.seed
        or a.perturbation != b.perturbation
        or dict(a.metadata) != dict(b.metadata)
        or not np.array_equal(a.timestamps, b.timestamps)
        or len(a.agents) != len(b.agents)
    ):
        return False
    for x, y in zip(a.agents, b.agents, strict=True):
        if x.id != y.id or x.type != y.type or x.length != y.length or x.width != y.width:
            return False
        for field in ("valid", "x", "y", "heading", "vx", "vy"):
            if not np.array_equal(getattr(x, field), getattr(y, field)):
                return False
    return True


# --- DefectSpec validation -------------------------------------------------

def test_defect_spec_rejects_bad_family_and_severity_range() -> None:
    with pytest.raises(ValueError):
        DefectSpec(family="Bad Family", version="v1", severity_min=0.0,
                   severity_max=1.0, severity_unit="fraction", target="rollout",
                   description="x")
    with pytest.raises(ValueError):
        DefectSpec(family="frozen_agent", version="v1", severity_min=1.0,
                   severity_max=0.0, severity_unit="fraction", target="rollout",
                   description="x")


def test_defect_spec_accepts_canonical() -> None:
    spec = DefectSpec(family="frozen_agent", version="v1", severity_min=0.0,
                      severity_max=1.0, severity_unit="fraction", target="rollout",
                      description="freeze a fraction of world agents")
    assert spec.family == "frozen_agent"
    assert spec.severity_max == 1.0


# --- FrozenAgentDefect oracles ---------------------------------------------

def test_frozen_agent_severity_zero_is_identity() -> None:
    scenario, rollout = _moving_pair()
    defect = FrozenAgentDefect()
    corrupted, manifest = defect.apply(scenario, rollout, severity=0.0, seed=7)
    assert _rollouts_equal(corrupted, rollout)
    assert manifest.affected_agent_count == 0
    # source rollout is not mutated in place
    assert _rollouts_equal(rollout, _moving_pair()[1])


def test_frozen_agent_is_deterministic() -> None:
    scenario, rollout = _moving_pair()
    a, _ = FrozenAgentDefect().apply(scenario, rollout, severity=0.5, seed=3)
    b, _ = FrozenAgentDefect().apply(scenario, rollout, severity=0.5, seed=3)
    assert _rollouts_equal(a, b)


def test_frozen_agent_selection_is_nested_in_severity() -> None:
    scenario, rollout = _moving_pair(n_world=4)
    _, low = FrozenAgentDefect().apply(scenario, rollout, severity=0.25, seed=1)
    _, high = FrozenAgentDefect().apply(scenario, rollout, severity=0.75, seed=1)
    assert 0 < low.affected_agent_count < high.affected_agent_count
    assert set(low.affected_agent_ordinals) <= set(high.affected_agent_ordinals)


def test_frozen_agent_manifest_is_sanitized() -> None:
    scenario, rollout = _moving_pair(n_world=4)  # ego id=0, world ids 10..13
    _, half = FrozenAgentDefect().apply(scenario, rollout, severity=0.5, seed=1)
    assert half.family == "frozen_agent"
    assert half.affected_agent_count == 2
    assert half.total_world_agent_count == 4
    # ordinals are 0-based cohort-relative ranks (NOT native ids 10.. or coordinates)
    assert half.affected_agent_ordinals == (0, 1)
    assert all(0 <= o < half.total_world_agent_count for o in half.affected_agent_ordinals)
    # full severity covers exactly the 0-based cohort, never spilling past the count
    _, full = FrozenAgentDefect().apply(scenario, rollout, severity=1.0, seed=1)
    assert full.affected_agent_ordinals == (0, 1, 2, 3)


def test_frozen_agent_holds_state_after_current_index() -> None:
    scenario, rollout = _moving_pair(n_world=1, current_index=1)
    corrupted, _ = FrozenAgentDefect().apply(scenario, rollout, severity=1.0, seed=0)
    world = corrupted.agents[1]
    held_x = world.x[1]
    # every future frame equals the current-index state; velocity zeroed
    assert np.all(world.x[1:] == held_x)
    assert np.all(world.vx[1:] == 0.0)
    # history (frame 0) is untouched
    assert world.x[0] == rollout.agents[1].x[0]


def test_frozen_agent_detected_by_position_error_monotone() -> None:
    scenario, rollout = _moving_pair(n_world=4)
    clean = _position_error(scenario, rollout)
    assert clean == pytest.approx(0.0, abs=1e-9)  # log-replay: zero error
    curve = []
    for sev in (0.25, 0.5, 0.75, 1.0):
        corrupted, _ = FrozenAgentDefect().apply(scenario, rollout, severity=sev, seed=1)
        curve.append(_position_error(scenario, corrupted))
    assert clean < curve[0]
    for lo, hi in zip(curve, curve[1:], strict=False):
        assert lo <= hi + 1e-9  # monotone non-decreasing in severity


# --- DefectRegistry --------------------------------------------------------

def test_registry_one_active_version_and_sorted_iteration() -> None:
    registry = DefectRegistry([FrozenAgentDefect()])
    assert registry.families == ("frozen_agent",)
    assert isinstance(registry.get("frozen_agent"), Defect)
    with pytest.raises(DefectRegistryError):
        registry.register(FrozenAgentDefect())  # duplicate family


def test_registry_rejects_non_defect() -> None:
    with pytest.raises(DefectRegistryError):
        DefectRegistry([object()])  # type: ignore[list-item]


# --- TeleportationDefect + OverlapDefect + default registry -----------------

from evalsim.metrics.m5 import (  # noqa: E402
    OrientedBoxOverlapRateMetric,
    WaymaxKinematicInfeasibilityRateMetric,
)
from evalsim.stress.defects import (  # noqa: E402
    KinematicSpikeDefect,
    OverlapDefect,
    TeleportationDefect,
    default_defect_registry,
)


def _separated_pair(n_world: int = 4, steps: int = 6, current_index: int = 1):
    """Ego + n well-separated constant-velocity world agents; log-replay rollout."""
    t = np.arange(steps, dtype=float) * 0.1
    source = [_agent(0, t, x=-100.0)]
    for k in range(n_world):
        source.append(_agent(10 + k, t, x=2.0 * t, y=50.0 * k, vx=2.0))
    scenario = Scenario(
        scenario_id="m7-sep",
        timestamps=np.array(t, copy=True),
        agents=source,
        ego_index=0,
        metadata={"source": "unit", "current_index": current_index},
    )
    rollout = Rollout(
        scenario_id=scenario.scenario_id, sim_name="candidate", sim_version="1.0.0",
        seed=0, timestamps=np.array(t, copy=True), agents=copy.deepcopy(source),
    )
    return scenario, rollout


def _infeasibility(scenario, rollout) -> float:
    m = WaymaxKinematicInfeasibilityRateMetric()
    return MetricRegistry([m]).evaluate(scenario, rollout)[0].value


def _overlap_rate(scenario, rollout) -> float:
    m = OrientedBoxOverlapRateMetric()
    return MetricRegistry([m]).evaluate(scenario, rollout)[0].value


def test_teleportation_severity_zero_is_identity_and_deterministic() -> None:
    scenario, rollout = _separated_pair()
    a, ma = TeleportationDefect().apply(scenario, rollout, severity=0.0, seed=5)
    assert _rollouts_equal(a, rollout)
    assert ma.affected_agent_count == 0
    b, _ = TeleportationDefect().apply(scenario, rollout, severity=0.75, seed=5)
    c, _ = TeleportationDefect().apply(scenario, rollout, severity=0.75, seed=5)
    assert _rollouts_equal(b, c)


def test_teleportation_is_position_only_and_nested() -> None:
    # A pure position teleport must not touch velocities, and selection is nested.
    scenario, rollout = _separated_pair(n_world=4)
    corrupted, low = TeleportationDefect().apply(scenario, rollout, severity=0.25, seed=2)
    _, high = TeleportationDefect().apply(scenario, rollout, severity=0.75, seed=2)
    assert 0 < low.affected_agent_count < high.affected_agent_count
    assert low.affected_agent_ordinals == (0,)
    assert high.affected_agent_ordinals == (0, 1, 2)
    # velocities and heading are untouched (position-only corruption)
    for a, b in zip(corrupted.agents, rollout.agents, strict=True):
        assert np.array_equal(a.vx, b.vx)
        assert np.array_equal(a.vy, b.vy)
        assert np.array_equal(a.heading, b.heading)


def test_teleportation_detected_by_position_error_monotone() -> None:
    scenario, rollout = _separated_pair(n_world=4)
    clean = _position_error(scenario, rollout)
    assert clean == pytest.approx(0.0, abs=1e-9)
    curve = []
    for sev in (0.25, 0.5, 0.75, 1.0):
        corrupted, _ = TeleportationDefect().apply(scenario, rollout, severity=sev, seed=2)
        curve.append(_position_error(scenario, corrupted))
    assert clean < curve[0]
    for lo, hi in zip(curve, curve[1:], strict=False):
        assert lo <= hi + 1e-9


def test_teleportation_is_a_blind_spot_for_infeasibility() -> None:
    # Honest negative result: the velocity-based metric never finite-diffs position,
    # so a position-only teleport (velocities untouched) reads as perfectly feasible.
    scenario, rollout = _separated_pair(n_world=4)
    clean = _infeasibility(scenario, rollout)
    corrupted, _ = TeleportationDefect().apply(scenario, rollout, severity=1.0, seed=2)
    assert clean == pytest.approx(0.0, abs=1e-9)
    assert _infeasibility(scenario, corrupted) == pytest.approx(clean, abs=1e-9)


def test_kinematic_spike_is_velocity_only_and_nested() -> None:
    scenario, rollout = _separated_pair(n_world=4)
    corrupted, low = KinematicSpikeDefect().apply(scenario, rollout, severity=0.25, seed=3)
    _, high = KinematicSpikeDefect().apply(scenario, rollout, severity=0.75, seed=3)
    assert low.affected_agent_ordinals == (0,)
    assert high.affected_agent_ordinals == (0, 1, 2)
    # positions, heading, and vy are untouched (vx-only corruption)
    for a, b in zip(corrupted.agents, rollout.agents, strict=True):
        assert np.array_equal(a.x, b.x)
        assert np.array_equal(a.y, b.y)
        assert np.array_equal(a.heading, b.heading)
        assert np.array_equal(a.vy, b.vy)


def test_kinematic_spike_severity_zero_is_identity() -> None:
    scenario, rollout = _separated_pair()
    corrupted, manifest = KinematicSpikeDefect().apply(scenario, rollout, severity=0.0, seed=1)
    assert _rollouts_equal(corrupted, rollout)
    assert manifest.affected_agent_count == 0


def test_kinematic_spike_detected_by_infeasibility_monotone() -> None:
    scenario, rollout = _separated_pair(n_world=4)
    clean = _infeasibility(scenario, rollout)
    assert clean == pytest.approx(0.0, abs=1e-9)  # constant velocity is feasible
    curve = []
    for sev in (0.25, 0.5, 0.75, 1.0):
        corrupted, _ = KinematicSpikeDefect().apply(scenario, rollout, severity=sev, seed=3)
        curve.append(_infeasibility(scenario, corrupted))
    assert clean < curve[0]
    for lo, hi in zip(curve, curve[1:], strict=False):
        assert lo <= hi + 1e-9


def test_kinematic_spike_is_a_blind_spot_for_position_error() -> None:
    # Velocity-only impulse leaves positions matching the log -> position error blind.
    scenario, rollout = _separated_pair(n_world=4)
    corrupted, _ = KinematicSpikeDefect().apply(scenario, rollout, severity=1.0, seed=3)
    assert _position_error(scenario, corrupted) == pytest.approx(0.0, abs=1e-9)


def test_overlap_severity_zero_is_identity() -> None:
    scenario, rollout = _separated_pair()
    corrupted, manifest = OverlapDefect().apply(scenario, rollout, severity=0.0, seed=0)
    assert _rollouts_equal(corrupted, rollout)
    assert manifest.affected_agent_count == 0


def test_overlap_detected_by_overlap_rate_monotone() -> None:
    scenario, rollout = _separated_pair(n_world=4)
    clean = _overlap_rate(scenario, rollout)
    assert clean == pytest.approx(0.0, abs=1e-9)  # well separated: no overlap
    curve = []
    for sev in (0.5, 1.0):
        corrupted, _ = OverlapDefect().apply(scenario, rollout, severity=sev, seed=0)
        curve.append(_overlap_rate(scenario, corrupted))
    assert clean < curve[0]
    assert curve[0] <= curve[1] + 1e-9


def test_overlap_selection_is_nested_in_severity() -> None:
    scenario, rollout = _separated_pair(n_world=4)
    _, low = OverlapDefect().apply(scenario, rollout, severity=0.34, seed=0)
    _, high = OverlapDefect().apply(scenario, rollout, severity=1.0, seed=0)
    assert 0 < low.affected_agent_count < high.affected_agent_count
    assert set(low.affected_agent_ordinals) <= set(high.affected_agent_ordinals)


def test_default_registry_exposes_all_families() -> None:
    registry = default_defect_registry()
    assert registry.families == (
        "frozen_agent",
        "kinematic_spike",
        "overlap",
        "teleportation",
    )


def test_future_frame_defects_are_honest_noop_at_terminal_current_index() -> None:
    # current_index at the last frame => no future to corrupt => 0 affected reported,
    # and the rollout is returned unchanged (no manifest over-reporting).
    t = np.arange(3, dtype=float) * 0.1
    src = [_agent(0, t, x=-40.0), _agent(10, t, x=t, vx=1.0), _agent(11, t, x=t + 1, vx=1.0)]
    scenario = Scenario(
        scenario_id="term", timestamps=t, agents=src, ego_index=0,
        metadata={"current_index": 2},
    )
    rollout = Rollout(
        scenario_id="term", sim_name="c", sim_version="1.0.0", seed=0,
        timestamps=t, agents=copy.deepcopy(src),
    )
    for defect in (TeleportationDefect(), KinematicSpikeDefect(), OverlapDefect()):
        corrupted, manifest = defect.apply(scenario, rollout, severity=1.0, seed=0)
        assert manifest.affected_agent_count == 0, defect.spec.family
        assert _rollouts_equal(corrupted, rollout), defect.spec.family
