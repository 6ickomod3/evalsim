from __future__ import annotations

import dataclasses
import functools

import numpy as np
import pytest

from evalsim.contracts import Agent, AgentType, Scenario
from evalsim.simulators.waymax_reference import (
    CompactWaymaxRollout,
    M4_INIT_STEPS,
    WAYMAX_EXACT_LOG_NAME,
    WAYMAX_IDM_NAME,
    WaymaxReferenceError,
    assert_waymax_idm_defaults,
    compact_exact_log_rollout,
    compact_rollout_bytes,
    compact_stock_exact_log_rollout,
    compact_waymax_idm_rollout,
    compact_waymax_to_rollout,
    initialized_overlap_mask_numpy,
    numpy_pairwise_overlaps,
    single_scene_exact_log_kernel,
    single_scene_idm_kernel,
    validate_exact_log_compact,
    validate_stock_equivalence,
    waymax_idm_scalar_acceleration,
)

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
waymax = pytest.importorskip("waymax")


def _synthetic_waymax_state(
    *,
    x_offset: float = 0.0,
    lifecycle: bool = False,
    initial_overlap: bool = False,
):
    from waymax import datatypes

    objects = 128
    steps = 91
    active = 5
    timestep = np.arange(steps, dtype=np.float32)
    shape = (objects, steps)
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

    # SDC log, a following vehicle with a nearby leader, and one free-road
    # vehicle. All are well separated at frame zero for overlap eligibility.
    numeric["x"][0] = x_offset + 10.0 + timestep * 0.5
    numeric["y"][0] = 100.0
    numeric["vel_x"][0] = 5.0
    numeric["x"][1] = x_offset + 100.0 + timestep
    numeric["y"][1] = 0.0
    numeric["vel_x"][1] = 10.0
    numeric["x"][2] = x_offset + 123.0
    numeric["y"][2] = 0.0
    numeric["vel_x"][2] = 0.0
    numeric["x"][3] = x_offset + 200.0 + timestep
    numeric["y"][3] = 40.0
    numeric["vel_x"][3] = 10.0
    numeric["x"][4] = x_offset + 300.0 + timestep * 0.25
    numeric["y"][4] = -40.0
    numeric["vel_x"][4] = 2.5
    if initial_overlap:
        numeric["x"][2, 0] = numeric["x"][1, 0] + 1.0
    if lifecycle:
        valid[3, :M4_INIT_STEPS] = False
        valid[2, M4_INIT_STEPS + 2 :] = False
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
    ids = np.full(objects, -1, dtype=np.int32)
    ids[:active] = (101, 102, 103, 104, 105)
    object_types = np.zeros(objects, dtype=np.int32)
    object_types[:4] = 1
    object_types[4] = 2
    # An invented never-valid vehicle slot makes lifecycle fallback observable
    # without adding a retained EvalSim agent.
    object_types[5] = 1
    is_sdc = np.zeros(objects, dtype=bool)
    is_sdc[0] = True
    is_valid = np.zeros(objects, dtype=bool)
    is_valid[:active] = True
    metadata = datatypes.ObjectMetadata(
        ids=jnp.asarray(ids),
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
    return state


def _scenario_from_synthetic_state(state) -> Scenario:
    trajectory = state.log_trajectory
    retained = np.flatnonzero(np.asarray(state.object_metadata.is_valid))
    valid = np.asarray(trajectory.valid)[retained]
    agents = []
    type_map = {
        1: AgentType.VEHICLE,
        2: AgentType.PEDESTRIAN,
        3: AgentType.CYCLIST,
    }
    for target_index, source_index in enumerate(retained):
        fields = {}
        for target, source in (
            ("x", "x"),
            ("y", "y"),
            ("heading", "yaw"),
            ("vx", "vel_x"),
            ("vy", "vel_y"),
        ):
            values = np.asarray(
                getattr(trajectory, source)[source_index],
                dtype=float,
            )
            if target == "heading":
                values = (values + np.pi) % (2.0 * np.pi) - np.pi
            fields[target] = np.where(valid[target_index], values, 0.0)
        agents.append(
            Agent(
                id=int(np.asarray(state.object_metadata.ids)[source_index]),
                type=type_map.get(
                    int(
                        np.asarray(
                            state.object_metadata.object_types
                        )[source_index]
                    ),
                    AgentType.UNKNOWN,
                ),
                valid=valid[target_index],
                length=4.0,
                width=2.0,
                **fields,
            )
        )
    return Scenario(
        scenario_id="synthetic-m4",
        timestamps=(
            np.arange(91, dtype=np.int64) * 100_000
        ).astype(np.float64)
        * 1e-6,
        agents=agents,
        ego_index=0,
        metadata={
            "current_index": 10,
            "source": "synthetic",
            "source_fingerprint": "invented",
        },
    )


def _block_tree(tree):
    return jax.tree.map(
        lambda value: value.block_until_ready()
        if hasattr(value, "block_until_ready")
        else value,
        tree,
    )


def test_exact_log_compact_matches_direct_log_and_five_step_stock_rollout():
    state = _synthetic_waymax_state()
    compact = _block_tree(compact_exact_log_rollout(state, num_steps=5))
    assert all(validate_exact_log_compact(state, compact).values())

    stock = _block_tree(compact_stock_exact_log_rollout(state, num_steps=5))
    assert all(validate_stock_equivalence(compact, stock).values())


def test_exact_log_timestamp_contradiction_is_not_hidden():
    state = _synthetic_waymax_state()
    compact = compact_exact_log_rollout(state, num_steps=1)
    values = list(compact)
    values[6] = np.asarray(values[6]).copy()
    values[6][0, 0] += 1
    contradicted = CompactWaymaxRollout(*values)
    with pytest.raises(WaymaxReferenceError, match="exact_log_timestamp"):
        validate_exact_log_compact(state, contradicted)


@pytest.mark.parametrize(
    ("field_index", "error_code"),
    [
        (0, "exact_log_x"),
        (1, "exact_log_y"),
        (2, "exact_log_yaw"),
        (3, "exact_log_vx"),
        (4, "exact_log_vy"),
        (5, "exact_log_validity"),
        (6, "exact_log_timestamp"),
        (7, "exact_log_timestep"),
    ],
)
def test_exact_log_each_emitted_field_has_an_independent_contradiction(
    field_index,
    error_code,
):
    state = _synthetic_waymax_state()
    compact = compact_exact_log_rollout(state, num_steps=1)
    values = [np.asarray(value).copy() for value in compact]
    if field_index == 5:
        values[field_index][0, 0] = False
    elif field_index in (6, 7):
        values[field_index].flat[0] += 1
    else:
        values[field_index][0, 0] += 0.01
    contradicted = CompactWaymaxRollout(*values)
    with pytest.raises(WaymaxReferenceError, match=error_code):
        validate_exact_log_compact(state, contradicted)


@pytest.mark.parametrize(
    ("field_index", "replacement"),
    [
        (0, np.float64),
        (5, np.int8),
        (6, np.float32),
        (7, np.float32),
    ],
)
def test_compact_schema_rejects_dtype_drift(field_index, replacement):
    state = _synthetic_waymax_state()
    compact = compact_exact_log_rollout(state, num_steps=1)
    values = list(compact)
    values[field_index] = np.asarray(values[field_index]).astype(replacement)
    with pytest.raises(WaymaxReferenceError, match="compact_dtype"):
        validate_exact_log_compact(state, CompactWaymaxRollout(*values))


def test_exact_log_kernel_supports_jit_vmap_and_permutation():
    first = _synthetic_waymax_state()
    second = _synthetic_waymax_state(x_offset=7.0)
    batch = jax.tree.map(lambda a, b: jnp.stack((a, b)), first, second)
    five_steps = functools.partial(compact_exact_log_rollout, num_steps=5)
    compiled = jax.jit(jax.vmap(five_steps))
    forward = _block_tree(compiled(batch))
    sequential = tuple(
        _block_tree(five_steps(candidate))
        for candidate in (first, second)
    )
    stacked_sequential = jax.tree.map(
        lambda left, right: np.stack(
            (np.asarray(left), np.asarray(right)),
            axis=0,
        ),
        *sequential,
    )
    for compiled_leaf, sequential_leaf in zip(
        jax.tree.leaves(forward),
        jax.tree.leaves(stacked_sequential),
    ):
        np.testing.assert_allclose(
            np.asarray(compiled_leaf),
            np.asarray(sequential_leaf),
            rtol=0.0,
            atol=1e-6,
        )
    reverse_batch = jax.tree.map(lambda value: value[::-1], batch)
    reverse = _block_tree(compiled(reverse_batch))
    for forward_leaf, reverse_leaf in zip(
        jax.tree.leaves(forward),
        jax.tree.leaves(reverse),
    ):
        np.testing.assert_allclose(
            np.asarray(forward_leaf),
            np.asarray(reverse_leaf)[::-1],
            rtol=0.0,
            atol=1e-6,
        )


def test_fixed_public_80_and_20_step_kernels_have_locked_shapes_and_jit():
    first = _synthetic_waymax_state()
    second = _synthetic_waymax_state(x_offset=7.0)
    batch = jax.tree.map(
        lambda left, right: jnp.stack((left, right)),
        first,
        second,
    )
    exact = _block_tree(
        jax.jit(jax.vmap(single_scene_exact_log_kernel))(batch)
    )
    assert np.asarray(exact.x).shape == (2, 80, 128)
    idm = _block_tree(jax.jit(single_scene_idm_kernel)(first))
    assert np.asarray(idm.x).shape == (20, 128)
    assert np.asarray(idm.effective_control).shape == (20, 128)


def test_numpy_overlap_oracle_matches_waymax_without_validity_mask():
    from waymax.utils import geometry

    boxes = np.asarray(
        [
            [0.0, 0.0, 2.0, 2.0, 0.0],
            [1.5, 0.0, 2.0, 2.0, 0.0],
            [3.5, 0.0, 2.0, 2.0, 0.0],  # edge touches the next box
            [5.5, 0.0, 2.0, 2.0, 0.0],
            [0.0, 4.0, 3.0, 1.0, np.pi / 4.0],
            [0.2, 4.0, 3.0, 1.0, -np.pi / 4.0],
            [-1000.0, -1000.0, -1.0, -1.0, -1.0],  # invalid-like
            [-1000.0, -1000.0, -1.0, -1.0, -1.0],  # padding-like
        ],
        dtype=np.float32,
    )
    expected = np.asarray(geometry.compute_pairwise_overlaps(jnp.asarray(boxes)))
    actual = numpy_pairwise_overlaps(boxes)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        initialized_overlap_mask_numpy(boxes),
        np.any(expected, axis=-1),
    )
    assert not actual[2, 3]
    assert actual[6, 7]  # There is intentionally no validity-mask exception.


def test_idm_defaults_scalar_oracle_control_and_repeat_are_nonvacuous():
    from waymax import agents, config, dynamics, env

    assert assert_waymax_idm_defaults()["desired_vel"] == 30.0
    assert waymax_idm_scalar_acceleration(10.0) > 0.0
    assert (
        waymax_idm_scalar_acceleration(
            10.0,
            leader_speed=0.0,
            leader_distance=5.0,
        )
        < 0.0
    )
    state = _synthetic_waymax_state()
    base = env.BaseEnvironment(
        dynamics.StateDynamics(),
        config.EnvironmentConfig(
            init_steps=11,
            max_num_objects=128,
            controlled_object=config.ObjectType.SDC,
            compute_reward=False,
            metrics=config.MetricsConfig(()),
        ),
    )
    reset = base.reset(state)
    policy = agents.IDMRoutePolicy()
    next_speeds, speed_valid = policy.update_speed(reset)
    actual_accelerations = (
        np.asarray(next_speeds) - np.asarray(reset.current_sim_trajectory.speed)[:, 0]
    ) / 0.1
    assert np.asarray(speed_valid)[1]
    assert np.asarray(speed_valid)[3]
    # Vehicle 1's first collision waypoint is independently 10 m ahead;
    # vehicle 3 has no leader on its separated logged path.
    assert actual_accelerations[1] == pytest.approx(
        waymax_idm_scalar_acceleration(
            10.0,
            leader_speed=0.0,
            leader_distance=10.0,
        ),
        abs=2e-6,
    )
    assert actual_accelerations[3] == pytest.approx(
        waymax_idm_scalar_acceleration(10.0),
        abs=2e-6,
    )
    assert actual_accelerations[1] < 0.0 < actual_accelerations[3]

    first = _block_tree(compact_waymax_idm_rollout(state, num_steps=1))
    compiled = jax.jit(
        functools.partial(compact_waymax_idm_rollout, num_steps=1)
    )
    second = _block_tree(compiled(state))
    assert compact_rollout_bytes(first) == compact_rollout_bytes(second)
    assert np.count_nonzero(np.asarray(first.effective_control)) >= 2
    controlled = np.asarray(first.effective_control[0])
    direct_log_x = np.asarray(state.log_trajectory.x[:, M4_INIT_STEPS])
    simulated_x = np.asarray(first.x[0])
    assert np.any(np.abs(simulated_x[controlled] - direct_log_x[controlled]) > 1e-6)
    current_x = np.asarray(reset.current_sim_trajectory.x)[:, 0]
    current_speed = np.asarray(reset.current_sim_trajectory.speed)[:, 0]
    expected_free_x = current_x[3] + 0.05 * (
        current_speed[3] + np.asarray(next_speeds)[3]
    )
    assert simulated_x[3] == pytest.approx(expected_free_x, abs=2e-5)
    assert simulated_x[3] != pytest.approx(direct_log_x[3], abs=1e-6)


def test_compact_conversion_preserves_history_ids_lifecycle_and_provenance():
    state = _synthetic_waymax_state()
    scenario = _scenario_from_synthetic_state(state)
    compact = _block_tree(compact_exact_log_rollout(state, num_steps=5))
    rollout = compact_waymax_to_rollout(
        compact,
        state=state,
        scenario=scenario,
        sim_name=WAYMAX_EXACT_LOG_NAME,
    )
    assert rollout.num_steps == 16
    assert [agent.id for agent in rollout.agents] == [
        101,
        102,
        103,
        104,
        105,
    ]
    assert rollout.metadata["backend_commit"]
    assert rollout.metadata["horizon_transitions"] == 5
    for reference, actual in zip(scenario.agents, rollout.agents):
        np.testing.assert_array_equal(
            actual.valid[:M4_INIT_STEPS],
            reference.valid[:M4_INIT_STEPS],
        )
        np.testing.assert_allclose(
            actual.x[:M4_INIT_STEPS],
            reference.x[:M4_INIT_STEPS],
            rtol=0.0,
            atol=0.0,
        )


def test_conversion_rejects_agent_order_drift():
    state = _synthetic_waymax_state()
    scenario = _scenario_from_synthetic_state(state)
    scenario.agents[0], scenario.agents[1] = (
        scenario.agents[1],
        scenario.agents[0],
    )
    compact = compact_exact_log_rollout(state, num_steps=1)
    with pytest.raises(WaymaxReferenceError, match="agent_order"):
        compact_waymax_to_rollout(
            compact,
            state=state,
            scenario=scenario,
            sim_name=WAYMAX_EXACT_LOG_NAME,
        )


def test_conversion_rejects_timeline_policy_accounting_and_horizon_drift():
    state = _synthetic_waymax_state()
    scenario = _scenario_from_synthetic_state(state)
    compact = compact_exact_log_rollout(state, num_steps=1)

    changed_time = np.array(scenario.timestamps, copy=True)
    changed_time[11] += 7.0
    with pytest.raises(WaymaxReferenceError, match="scenario_time_drift"):
        compact_waymax_to_rollout(
            compact,
            state=state,
            scenario=dataclasses.replace(scenario, timestamps=changed_time),
            sim_name=WAYMAX_EXACT_LOG_NAME,
        )
    with pytest.raises(WaymaxReferenceError, match="sim_name_mismatch"):
        compact_waymax_to_rollout(
            compact,
            state=state,
            scenario=scenario,
            sim_name=WAYMAX_IDM_NAME,
        )
    with pytest.raises(WaymaxReferenceError, match="control_accounting"):
        compact_waymax_to_rollout(
            compact,
            state=state,
            scenario=scenario,
            sim_name=WAYMAX_EXACT_LOG_NAME,
            control_accounting={},
        )
    short_agents = [
        dataclasses.replace(
            agent,
            valid=agent.valid[:-1],
            x=agent.x[:-1],
            y=agent.y[:-1],
            heading=agent.heading[:-1],
            vx=agent.vx[:-1],
            vy=agent.vy[:-1],
        )
        for agent in scenario.agents
    ]
    short = dataclasses.replace(
        scenario,
        timestamps=scenario.timestamps[:-1],
        agents=short_agents,
    )
    with pytest.raises(WaymaxReferenceError, match="scenario_horizon"):
        compact_waymax_to_rollout(
            compact,
            state=state,
            scenario=short,
            sim_name=WAYMAX_EXACT_LOG_NAME,
        )
    wrong_boundary = dataclasses.replace(
        scenario,
        metadata={**scenario.metadata, "current_index": 9},
    )
    with pytest.raises(WaymaxReferenceError, match="current_boundary"):
        compact_waymax_to_rollout(
            compact,
            state=state,
            scenario=wrong_boundary,
            sim_name=WAYMAX_EXACT_LOG_NAME,
        )


def test_conversion_preserves_birth_disappearance_and_normalizes_payload():
    state = _synthetic_waymax_state(lifecycle=True)
    scenario = _scenario_from_synthetic_state(state)
    original = _block_tree(compact_exact_log_rollout(state, num_steps=5))
    values = [np.asarray(value).copy() for value in original]
    invalid = ~values[5]
    values[0][invalid] = np.nan
    values[2][values[5]] += np.float32(2.0 * np.pi)
    compact = CompactWaymaxRollout(*values)
    rollout = compact_waymax_to_rollout(
        compact,
        state=state,
        scenario=scenario,
        sim_name=WAYMAX_EXACT_LOG_NAME,
    )

    born = rollout.agents[3]
    assert not np.any(born.valid[:M4_INIT_STEPS])
    assert born.valid[M4_INIT_STEPS]
    disappeared = rollout.agents[2]
    assert not disappeared.valid[M4_INIT_STEPS + 2]
    for agent in rollout.agents:
        assert np.all(agent.x[~agent.valid] == 0.0)
        assert np.all(agent.heading[~agent.valid] == 0.0)
        assert np.all(agent.heading >= -np.pi)
        assert np.all(agent.heading < np.pi)
    assert rollout.metadata["time_source"] == (
        "direct_waymax_emission_checked_against_log"
    )


def test_idm_conversion_derives_masks_and_rejects_false_fallback_labels():
    state = _synthetic_waymax_state()
    scenario = _scenario_from_synthetic_state(state)
    compact = _block_tree(compact_waymax_idm_rollout(state, num_steps=1))
    rollout = compact_waymax_to_rollout(
        compact,
        state=state,
        scenario=scenario,
        sim_name=WAYMAX_IDM_NAME,
    )
    accounting = rollout.metadata["control_accounting"]
    assert accounting["effective_controlled_transitions"] == int(
        np.count_nonzero(np.asarray(compact.effective_control))
    )
    assert accounting["lifecycle_fallbacks"] >= 1

    contradicted = compact._replace(
        x=np.asarray(compact.x).copy(),
    )
    contradicted.x[0, 5] += 1.0
    with pytest.raises(WaymaxReferenceError, match="fallback_motion"):
        compact_waymax_to_rollout(
            contradicted,
            state=state,
            scenario=scenario,
            sim_name=WAYMAX_IDM_NAME,
        )


def test_idm_initialized_overlap_and_nonvehicle_fallback_are_observable():
    state = _synthetic_waymax_state(initial_overlap=True)
    scenario = _scenario_from_synthetic_state(state)
    compact = _block_tree(compact_waymax_idm_rollout(state, num_steps=1))
    requested = np.asarray(compact.requested_control[0])
    effective = np.asarray(compact.effective_control[0])
    excluded = np.asarray(compact.initialized_overlap_excluded[0])
    lifecycle = np.asarray(compact.lifecycle_fallback[0])
    assert requested[1] and requested[2]
    assert excluded[1] and excluded[2]
    assert not effective[1] and not effective[2]
    assert not requested[4] and not effective[4] and not lifecycle[4]

    rollout = compact_waymax_to_rollout(
        compact,
        state=state,
        scenario=scenario,
        sim_name=WAYMAX_IDM_NAME,
    )
    accounting = rollout.metadata["control_accounting"]
    assert accounting["initialized_overlap_excluded_vehicles"] >= 2
    assert accounting["initialized_overlap_excluded_transitions"] >= 2
    source_x = np.asarray(state.log_trajectory.x)
    assert rollout.agents[1].x[M4_INIT_STEPS] == pytest.approx(
        source_x[1, M4_INIT_STEPS]
    )
    assert rollout.agents[4].x[M4_INIT_STEPS] == pytest.approx(
        source_x[4, M4_INIT_STEPS]
    )

    false_mask = np.asarray(compact.effective_control).copy()
    false_mask[0, 3] = False
    contradicted_mask = compact._replace(effective_control=false_mask)
    with pytest.raises(WaymaxReferenceError, match="control_mask_drift"):
        compact_waymax_to_rollout(
            contradicted_mask,
            state=state,
            scenario=scenario,
            sim_name=WAYMAX_IDM_NAME,
        )
