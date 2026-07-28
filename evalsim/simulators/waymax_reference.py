"""Compact, local-only reference execution through the pinned Waymax runtime.

The public functions in this module keep Waymax, JAX, and TensorFlow optional:
importing :mod:`evalsim` never imports that stack.  Runtime functions accept an
unbatched ``waymax.datatypes.SimulatorState`` and return compact pytrees or the
source-neutral EvalSim :class:`~evalsim.contracts.Rollout` contract.

M4 uses two deliberately different references:

* exact logged state setting with ``BaseEnvironment`` and ``StateDynamics``; and
* Waymax's privileged logged-trajectory waypoint-following ``IDMRoutePolicy``.

The latter follows each controlled vehicle's complete logged trajectory.  It is
not causal, map-route-aware ground truth, or a numerical twin of EvalSim's IDM.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np

from evalsim.contracts import Agent, Rollout, Scenario
from evalsim.sources.waymax import WAYMAX_COMMIT

M4_INIT_STEPS = 11
M4_EXACT_LOG_TRANSITIONS = 80
M4_IDM_TRANSITIONS = 20
M4_MAX_OBJECTS = 128
M4_FLOAT_ATOL = 1e-6
WAYMAX_REFERENCE_VERSION = "0.1.0"

WAYMAX_IDM_NAME = (
    "waymax_privileged_logged_trajectory_waypoint_following_idm"
)
WAYMAX_EXACT_LOG_NAME = "waymax_exact_log_state_dynamics"

WAYMAX_IDM_DEFAULTS: Mapping[str, Any] = {
    "desired_vel": 30.0,
    "min_spacing": 2.0,
    "safe_time_headway": 2.0,
    "max_accel": 2.0,
    "max_decel": 4.0,
    "delta": 4.0,
    "max_lookahead": 10,
    "lookahead_from_current_position": True,
    "additional_lookahead_points": 10,
    "additional_lookahead_distance": 10.0,
    "invalidate_on_end": False,
}


class WaymaxReferenceDependencyError(ImportError):
    """Raised when the optional pinned Waymax runtime is unavailable."""


class WaymaxReferenceError(ValueError):
    """A compact-reference or conversion failure with a stable reason code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class CompactWaymaxRollout(NamedTuple):
    """Post-transition compact frames emitted by a Waymax scan.

    Float, validity, and timestamp leaves have shape ``[steps, objects]``.
    ``timestep`` has shape ``[steps]``.  A ``NamedTuple`` is intentional: JAX
    treats it as a pytree without making JAX an import-time dependency.
    """

    x: Any
    y: Any
    yaw: Any
    vx: Any
    vy: Any
    valid: Any
    timestamp_micros: Any
    timestep: Any


class CompactWaymaxIDMRollout(NamedTuple):
    """Compact IDM frames plus pre-step control/fallback accounting masks."""

    x: Any
    y: Any
    yaw: Any
    vx: Any
    vy: Any
    valid: Any
    timestamp_micros: Any
    timestep: Any
    requested_control: Any
    effective_control: Any
    lifecycle_fallback: Any
    initialized_overlap_excluded: Any


def _require_waymax_runtime() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
        from waymax import agents, config, datatypes, dynamics, env
    except ImportError as exc:
        raise WaymaxReferenceDependencyError(
            "Waymax reference execution is optional; install it with "
            "`uv sync --extra dev --extra waymo`."
        ) from exc
    return jax, jnp, (agents, config), (datatypes, dynamics), env


def _validate_transition_count(
    value: int,
    *,
    maximum: int = M4_EXACT_LOG_TRANSITIONS,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 1 <= int(value) <= maximum
    ):
        raise ValueError(f"num_steps must be an integer in [1, {maximum}]")
    return int(value)


def _validate_unbatched_state(state: Any, *, num_steps: int) -> None:
    shape = tuple(getattr(state, "shape", ()))
    if shape:
        raise WaymaxReferenceError(
            "state_batched",
            "the compact single-scene kernel requires an unbatched state",
        )
    trajectory = getattr(state, "log_trajectory", None)
    if trajectory is None:
        raise WaymaxReferenceError(
            "state_schema",
            "state.log_trajectory is required",
        )
    if int(getattr(trajectory, "num_objects", -1)) != M4_MAX_OBJECTS:
        raise WaymaxReferenceError(
            "object_count",
            f"M4 requires exactly {M4_MAX_OBJECTS} Waymax object slots",
        )
    required_horizon = M4_INIT_STEPS + num_steps
    if int(getattr(trajectory, "num_timesteps", -1)) != (
        M4_INIT_STEPS + M4_EXACT_LOG_TRANSITIONS
    ):
        raise WaymaxReferenceError(
            "horizon",
            "M4 requires exactly 91 Waymax trajectory frames",
        )
    if required_horizon > M4_INIT_STEPS + M4_EXACT_LOG_TRANSITIONS:
        raise WaymaxReferenceError(
            "horizon",
            "the requested transition horizon exceeds the 91-frame state",
        )


def _environment_config(config: Any) -> Any:
    return config.EnvironmentConfig(
        init_steps=M4_INIT_STEPS,
        max_num_objects=M4_MAX_OBJECTS,
        controlled_object=config.ObjectType.SDC,
        compute_reward=False,
        allow_new_objects_after_warmup=True,
        metrics=config.MetricsConfig(metrics_to_run=()),
    )


def exact_log_config_payload() -> dict[str, Any]:
    """Return the JSON-native M4 exact-log environment contract."""

    return {
        "allow_new_objects_after_warmup": True,
        "compute_reward": False,
        "controlled_object": "SDC",
        "init_steps": M4_INIT_STEPS,
        "max_num_objects": M4_MAX_OBJECTS,
        "metrics_to_run": [],
        "state_dynamics": "StateDynamics",
    }


def reference_config_fingerprint() -> str:
    """Fingerprint the compact reference semantics without importing Waymax."""

    payload = {
        "exact_log": exact_log_config_payload(),
        "exact_log_name": WAYMAX_EXACT_LOG_NAME,
        "exact_log_transitions": M4_EXACT_LOG_TRANSITIONS,
        "float_atol": M4_FLOAT_ATOL,
        "idm_defaults": dict(WAYMAX_IDM_DEFAULTS),
        "idm_name": WAYMAX_IDM_NAME,
        "idm_transitions": M4_IDM_TRANSITIONS,
        "reference_version": WAYMAX_REFERENCE_VERSION,
        "waymax_commit": WAYMAX_COMMIT,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_waymax_idm_defaults() -> dict[str, Any]:
    """Fail closed if the installed policy defaults drift from the M4 lock."""

    _, _, (agents, _), _, _ = _require_waymax_runtime()
    policy = agents.IDMRoutePolicy()
    actual = {
        "desired_vel": policy.desired_vel,
        "min_spacing": policy.min_spacing_s0,
        "safe_time_headway": policy.safe_time_headway,
        "max_accel": policy.max_accel,
        "max_decel": policy.max_decel,
        "delta": policy.delta,
        "max_lookahead": policy.max_lookahead,
        "lookahead_from_current_position": (
            policy.lookahead_from_current_position
        ),
        "additional_lookahead_points": policy.additional_headway_points,
        "additional_lookahead_distance": policy.additional_lookahead_distance,
        "invalidate_on_end": policy.invalidate_on_end,
    }
    if actual != dict(WAYMAX_IDM_DEFAULTS):
        raise WaymaxReferenceError(
            "idm_default_drift",
            "the installed IDMRoutePolicy defaults differ from the M4 lock",
        )
    return actual


def _compact_frame(state: Any) -> CompactWaymaxRollout:
    trajectory = state.current_sim_trajectory
    return CompactWaymaxRollout(
        x=trajectory.x[..., 0],
        y=trajectory.y[..., 0],
        yaw=trajectory.yaw[..., 0],
        vx=trajectory.vel_x[..., 0],
        vy=trajectory.vel_y[..., 0],
        valid=trajectory.valid[..., 0],
        timestamp_micros=trajectory.timestamp_micros[..., 0],
        timestep=state.timestep,
    )


def _exact_log_kernel(state: Any, *, num_steps: int) -> CompactWaymaxRollout:
    _validate_unbatched_state(state, num_steps=num_steps)
    jax, _, (agents, config), (_, dynamics), env = _require_waymax_runtime()
    dynamics_model = dynamics.StateDynamics()
    environment = env.BaseEnvironment(
        dynamics_model=dynamics_model,
        config=_environment_config(config),
    )
    expert = agents.create_expert_actor(dynamics_model)
    reset_state = environment.reset(state)

    def step(carry: Any, unused: Any) -> tuple[Any, CompactWaymaxRollout]:
        del unused
        actor_output = expert.select_action(None, carry, None, None)
        next_state = environment.step(carry, actor_output.action)
        return next_state, _compact_frame(next_state)

    _, compact = jax.lax.scan(step, reset_state, xs=None, length=num_steps)
    return compact


def single_scene_exact_log_kernel(state: Any) -> CompactWaymaxRollout:
    """Run the fixed 80-transition M4 exact-log kernel for one scene.

    This exact signature is suitable for
    ``jax.jit(jax.vmap(single_scene_exact_log_kernel))``.
    """

    return _exact_log_kernel(state, num_steps=M4_EXACT_LOG_TRANSITIONS)


def compact_exact_log_rollout(
    state: Any,
    *,
    num_steps: int = M4_EXACT_LOG_TRANSITIONS,
) -> CompactWaymaxRollout:
    """Run a compact exact-log scan without materializing repeated full states."""

    num_steps = _validate_transition_count(num_steps)
    _validate_unbatched_state(state, num_steps=num_steps)
    return _exact_log_kernel(state, num_steps=num_steps)


def compact_stock_exact_log_rollout(
    state: Any,
    *,
    num_steps: int = 5,
) -> CompactWaymaxRollout:
    """Run Waymax's stock rollout and retain post-transition compact frames.

    This memory-heavier path exists only as an API-ordering oracle for a small
    synthetic fixture and the single-transition first-real-scene gate.
    """

    num_steps = _validate_transition_count(num_steps)
    _validate_unbatched_state(state, num_steps=num_steps)
    jax, _, (agents, config), (_, dynamics), env = _require_waymax_runtime()
    dynamics_model = dynamics.StateDynamics()
    environment = env.BaseEnvironment(
        dynamics_model=dynamics_model,
        config=_environment_config(config),
    )
    expert = agents.create_expert_actor(dynamics_model)
    stock = env.rollout(
        state,
        expert,
        environment,
        rng=jax.random.PRNGKey(0),
        rollout_num_steps=num_steps,
    )

    def current_frame(one_state: Any) -> CompactWaymaxRollout:
        return _compact_frame(one_state)

    all_frames = jax.vmap(current_frame)(stock.state)
    return type(all_frames)(*(leaf[1:] for leaf in all_frames))


def _to_numpy_compact(
    compact: CompactWaymaxRollout | CompactWaymaxIDMRollout,
) -> dict[str, np.ndarray]:
    return {
        field: np.asarray(getattr(compact, field))
        for field in CompactWaymaxRollout._fields
    }


def _assert_compact_shape(
    arrays: Mapping[str, np.ndarray],
    *,
    num_steps: int,
) -> None:
    for field in (
        "x",
        "y",
        "yaw",
        "vx",
        "vy",
        "valid",
        "timestamp_micros",
    ):
        if arrays[field].shape != (num_steps, M4_MAX_OBJECTS):
            raise WaymaxReferenceError(
                "compact_shape",
                f"{field} must have shape [{num_steps}, {M4_MAX_OBJECTS}]",
            )
    for field in ("x", "y", "yaw", "vx", "vy"):
        if arrays[field].dtype != np.float32:
            raise WaymaxReferenceError(
                "compact_dtype",
                f"{field} must have float32 dtype",
            )
    if arrays["valid"].dtype != np.bool_:
        raise WaymaxReferenceError(
            "compact_dtype",
            "valid must have boolean dtype",
        )
    for field in ("timestamp_micros", "timestep"):
        if (
            np.issubdtype(arrays[field].dtype, np.bool_)
            or not np.issubdtype(arrays[field].dtype, np.integer)
        ):
            raise WaymaxReferenceError(
                "compact_dtype",
                f"{field} must have integer dtype",
            )
    if arrays["timestep"].shape != (num_steps,):
        raise WaymaxReferenceError(
            "compact_shape",
            f"timestep must have shape [{num_steps}]",
        )


def validate_exact_log_compact(
    state: Any,
    compact: CompactWaymaxRollout,
) -> dict[str, bool]:
    """Independently compare all compact emissions with the direct source log."""

    arrays = _to_numpy_compact(compact)
    num_steps = int(arrays["timestep"].shape[0])
    _validate_transition_count(num_steps)
    _validate_unbatched_state(state, num_steps=num_steps)
    _assert_compact_shape(arrays, num_steps=num_steps)
    expected_steps = np.arange(
        M4_INIT_STEPS,
        M4_INIT_STEPS + num_steps,
        dtype=arrays["timestep"].dtype,
    )
    if not np.array_equal(arrays["timestep"], expected_steps):
        raise WaymaxReferenceError(
            "exact_log_timestep",
            "compact timestep order differs from the direct logged horizon",
        )

    logged = state.log_trajectory
    interval = slice(M4_INIT_STEPS, M4_INIT_STEPS + num_steps)
    expected_valid = np.asarray(logged.valid[:, interval]).T
    actual_valid = np.asarray(arrays["valid"], dtype=bool)
    if not np.array_equal(actual_valid, expected_valid):
        raise WaymaxReferenceError(
            "exact_log_validity",
            "compact validity differs from the direct logged validity",
        )

    expected_timestamps = np.asarray(
        logged.timestamp_micros[:, interval]
    ).T
    if not np.array_equal(
        arrays["timestamp_micros"],
        expected_timestamps,
    ):
        raise WaymaxReferenceError(
            "exact_log_timestamp",
            "compact emitted timestamps differ from the direct source log",
        )

    fields = {
        "x": "x",
        "y": "y",
        "yaw": "yaw",
        "vx": "vel_x",
        "vy": "vel_y",
    }
    for output_name, source_name in fields.items():
        actual = np.asarray(arrays[output_name], dtype=np.float64)
        expected = np.asarray(
            getattr(logged, source_name)[:, interval],
            dtype=np.float64,
        ).T
        if output_name == "yaw":
            actual = (actual + np.pi) % (2.0 * np.pi) - np.pi
            expected = (expected + np.pi) % (2.0 * np.pi) - np.pi
        actual = np.where(actual_valid, actual, 0.0)
        expected = np.where(expected_valid, expected, 0.0)
        if not np.allclose(
            actual,
            expected,
            rtol=0.0,
            atol=M4_FLOAT_ATOL,
        ):
            raise WaymaxReferenceError(
                f"exact_log_{output_name}",
                f"compact {output_name} differs from the direct source log",
            )
    return {
        "fields": True,
        "timestamps": True,
        "timesteps": True,
        "validity": True,
    }


def validate_stock_equivalence(
    compact: CompactWaymaxRollout,
    stock: CompactWaymaxRollout,
) -> dict[str, bool]:
    """Check compact scan emission order and values against stock rollout."""

    compact_arrays = _to_numpy_compact(compact)
    stock_arrays = _to_numpy_compact(stock)
    if compact_arrays.keys() != stock_arrays.keys():
        raise WaymaxReferenceError(
            "stock_schema",
            "compact and stock outputs have different fields",
        )
    exact = {"valid", "timestamp_micros", "timestep"}
    for field in compact_arrays:
        left = compact_arrays[field]
        right = stock_arrays[field]
        equal = (
            np.array_equal(left, right)
            if field in exact
            else np.allclose(
                left,
                right,
                rtol=0.0,
                atol=M4_FLOAT_ATOL,
            )
        )
        if not equal:
            raise WaymaxReferenceError(
                f"stock_{field}",
                f"compact {field} differs from stock Waymax rollout",
            )
    return {field: True for field in compact_arrays}


def _box_corners(box: np.ndarray) -> np.ndarray:
    x, y, length, width, yaw = box
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    lc = length * cosine / np.float32(2.0)
    ls = length * sine / np.float32(2.0)
    wc = width * cosine / np.float32(2.0)
    ws = width * sine / np.float32(2.0)
    return np.asarray(
        (
            (x + lc + ws, y + ls - wc),
            (x + lc - ws, y + ls + wc),
            (x - lc - ws, y - ls + wc),
            (x - lc + ws, y - ls - wc),
        ),
        dtype=np.float32,
    )


def numpy_pairwise_overlaps(boxes: Any) -> np.ndarray:
    """Independent NumPy SAT oracle matching Waymax's strict edge semantics.

    ``boxes`` is ``[objects, 5]`` in ``[x, y, length, width, yaw]`` order.
    No validity mask is consulted.  Self-overlaps are removed.
    """

    values = np.asarray(boxes, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("boxes must have shape [objects, 5]")
    if not np.all(np.isfinite(values)):
        raise ValueError("boxes must contain only finite values")
    corners = tuple(_box_corners(box) for box in values)
    count = values.shape[0]
    result = np.zeros((count, count), dtype=bool)
    for first in range(count):
        for second in range(first + 1, count):
            axes = []
            for index in (first, second):
                cosine = np.cos(values[index, 4])
                sine = np.sin(values[index, 4])
                axes.extend(
                    (
                        np.asarray((cosine, sine), dtype=np.float32),
                        np.asarray((-sine, cosine), dtype=np.float32),
                    )
                )
            overlaps = True
            for axis in axes:
                projection_a = corners[first] @ axis
                projection_b = corners[second] @ axis
                separation = min(
                    float(np.max(projection_a)),
                    float(np.max(projection_b)),
                ) - max(
                    float(np.min(projection_a)),
                    float(np.min(projection_b)),
                )
                if not separation > 0.0:
                    overlaps = False
                    break
            result[first, second] = overlaps
            result[second, first] = overlaps
    return result


def initialized_overlap_mask_numpy(state_or_boxes: Any) -> np.ndarray:
    """Return the frame-zero all-slot overlap exclusion without validity masks."""

    if hasattr(state_or_boxes, "log_trajectory"):
        trajectory = state_or_boxes.log_trajectory
        boxes = np.stack(
            (
                np.asarray(trajectory.x)[:, 0],
                np.asarray(trajectory.y)[:, 0],
                np.asarray(trajectory.length)[:, 0],
                np.asarray(trajectory.width)[:, 0],
                np.asarray(trajectory.yaw)[:, 0],
            ),
            axis=-1,
        )
    else:
        boxes = state_or_boxes
    return np.any(numpy_pairwise_overlaps(boxes), axis=-1)


def _idm_control_mask(state: Any) -> Any:
    """Pinned dynamic actor mask: non-SDC vehicles valid now and next in log."""

    _, _, _, (datatypes, _), _ = _require_waymax_runtime()
    current_valid = datatypes.dynamic_index(
        state.log_trajectory.valid,
        state.timestep,
        axis=-1,
        keepdims=False,
    )
    next_valid = datatypes.dynamic_index(
        state.log_trajectory.valid,
        state.timestep + 1,
        axis=-1,
        keepdims=False,
    )
    return (
        ~state.object_metadata.is_sdc
        & (state.object_metadata.object_types == 1)
        & current_valid
        & next_valid
    )


def _idm_kernel(state: Any, *, num_steps: int) -> CompactWaymaxIDMRollout:
    _validate_unbatched_state(state, num_steps=num_steps)
    jax, jnp, (agents, config), (_, dynamics), env = _require_waymax_runtime()
    assert_waymax_idm_defaults()
    idm_actor = agents.IDMRoutePolicy(is_controlled_func=_idm_control_mask)
    environment = env.PlanningAgentEnvironment(
        dynamics_model=dynamics.StateDynamics(),
        config=_environment_config(config),
        sim_agent_actors=(idm_actor,),
        sim_agent_params=({},),
    )
    expert_sdc = agents.create_expert_actor(environment.dynamics)
    reset_state = environment.reset(state)
    boxes = reset_state.log_trajectory.stack_fields(
        ["x", "y", "length", "width", "yaw"]
    )[:, 0, :]
    from waymax.utils import geometry

    initialized_overlap = jnp.any(
        geometry.compute_pairwise_overlaps(boxes),
        axis=-1,
    )
    non_sdc_vehicle = (
        ~reset_state.object_metadata.is_sdc
        & (reset_state.object_metadata.object_types == 1)
    )

    def step(
        carry: Any,
        unused: Any,
    ) -> tuple[Any, CompactWaymaxIDMRollout]:
        del unused
        requested = _idm_control_mask(carry)
        effective = requested & ~initialized_overlap
        lifecycle_fallback = non_sdc_vehicle & ~requested
        overlap_excluded = requested & initialized_overlap
        expert_output = expert_sdc.select_action(None, carry, None, None)
        next_state = environment.step(carry, expert_output.action)
        frame = _compact_frame(next_state)
        return next_state, CompactWaymaxIDMRollout(
            *frame,
            requested_control=requested,
            effective_control=effective,
            lifecycle_fallback=lifecycle_fallback,
            initialized_overlap_excluded=overlap_excluded,
        )

    _, compact = jax.lax.scan(step, reset_state, xs=None, length=num_steps)
    return compact


def single_scene_idm_kernel(state: Any) -> CompactWaymaxIDMRollout:
    """Run the fixed 20-transition M4 Waymax IDM kernel for one scene."""

    return _idm_kernel(state, num_steps=M4_IDM_TRANSITIONS)


def compact_waymax_idm_rollout(
    state: Any,
    *,
    num_steps: int = M4_IDM_TRANSITIONS,
) -> CompactWaymaxIDMRollout:
    """Run the privileged Waymax waypoint-following IDM reference."""

    num_steps = _validate_transition_count(
        num_steps,
        maximum=M4_IDM_TRANSITIONS,
    )
    _validate_unbatched_state(state, num_steps=num_steps)
    return _idm_kernel(state, num_steps=num_steps)


def waymax_idm_scalar_acceleration(
    speed: float,
    *,
    leader_speed: float | None = None,
    leader_distance: float | None = None,
) -> float:
    """Independent scalar IDM formula for analytic direction checks."""

    speed = float(speed)
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("speed must be finite and non-negative")
    if (leader_speed is None) != (leader_distance is None):
        raise ValueError(
            "leader_speed and leader_distance must be supplied together"
        )
    desired = float(WAYMAX_IDM_DEFAULTS["desired_vel"])
    acceleration = float(WAYMAX_IDM_DEFAULTS["max_accel"])
    free_road = (speed / desired) ** float(WAYMAX_IDM_DEFAULTS["delta"])
    interaction = 0.0
    if leader_speed is not None and leader_distance is not None:
        leader_speed = float(leader_speed)
        leader_distance = float(leader_distance)
        if (
            not math.isfinite(leader_speed)
            or leader_speed < 0.0
            or not math.isfinite(leader_distance)
            or leader_distance < 0.0
        ):
            raise ValueError(
                "leader speed/distance must be finite and non-negative"
            )
        distance = (
            0.1 if leader_distance == 0.0 else leader_distance
        )
        desired_gap = float(WAYMAX_IDM_DEFAULTS["min_spacing"]) + max(
            0.0,
            speed * float(WAYMAX_IDM_DEFAULTS["safe_time_headway"])
            + speed
            * (speed - leader_speed)
            / (
                2.0
                * math.sqrt(
                    acceleration
                    * float(WAYMAX_IDM_DEFAULTS["max_decel"])
                )
            ),
        )
        interaction = (desired_gap / distance) ** 2
    return acceleration * (1.0 - free_road - interaction)


def compact_rollout_bytes(
    compact: CompactWaymaxRollout | CompactWaymaxIDMRollout,
) -> bytes:
    """Canonical bytes for deterministic local repeat checks."""

    chunks: list[bytes] = []
    for field in compact._fields:
        array = np.ascontiguousarray(np.asarray(getattr(compact, field)))
        header = json.dumps(
            {
                "dtype": array.dtype.str,
                "field": field,
                "shape": list(array.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        chunks.extend(
            (
                len(header).to_bytes(8, "big"),
                header,
                array.tobytes(order="C"),
            )
        )
    return b"".join(chunks)


def _normalize_yaw(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _validate_scenario_history(
    state: Any,
    scenario: Scenario,
    retained: np.ndarray,
) -> None:
    logged = state.log_trajectory
    source_fields = {
        "x": "x",
        "y": "y",
        "heading": "yaw",
        "vx": "vel_x",
        "vy": "vel_y",
    }
    source_valid = np.asarray(logged.valid)[retained, :M4_INIT_STEPS]
    for target_index, slot in enumerate(retained):
        agent = scenario.agents[target_index]
        if not np.array_equal(
            agent.valid[:M4_INIT_STEPS],
            source_valid[target_index],
        ):
            raise WaymaxReferenceError(
                "history_validity",
                "scenario history validity differs from the Waymax source",
            )
        for target_name, source_name in source_fields.items():
            actual = np.asarray(getattr(agent, target_name))[:M4_INIT_STEPS]
            expected = np.asarray(
                getattr(logged, source_name)[slot, :M4_INIT_STEPS],
                dtype=np.float64,
            )
            if target_name == "heading":
                expected = _normalize_yaw(expected)
            expected = np.where(
                source_valid[target_index],
                expected,
                0.0,
            )
            if not np.allclose(
                actual,
                expected,
                rtol=0.0,
                atol=M4_FLOAT_ATOL,
            ):
                raise WaymaxReferenceError(
                    "history_mutation",
                    f"scenario history {target_name} differs from source",
                )


def _validate_scenario_timeline(
    state: Any,
    scenario: Scenario,
    retained: np.ndarray,
) -> None:
    source_valid = np.asarray(state.log_trajectory.valid, dtype=bool)[retained]
    source_micros = np.asarray(
        state.log_trajectory.timestamp_micros,
    )[retained]
    canonical = np.empty(
        M4_INIT_STEPS + M4_EXACT_LOG_TRANSITIONS,
        dtype=np.int64,
    )
    for frame in range(canonical.size):
        contributors = source_micros[:, frame][source_valid[:, frame]].astype(
            np.int64,
            copy=False,
        )
        if contributors.size == 0:
            raise WaymaxReferenceError(
                "scenario_time_drift",
                "the source has no valid timestamp contributor",
            )
        if not np.all(contributors == contributors[0]):
            raise WaymaxReferenceError(
                "scenario_time_drift",
                "valid source objects disagree on timestamp",
            )
        canonical[frame] = contributors[0]
    deltas = canonical - canonical[0]
    if np.any(np.diff(deltas) <= 0):
        raise WaymaxReferenceError(
            "scenario_time_drift",
            "source timestamps are not strictly increasing",
        )
    expected = deltas.astype(np.float64) * 1e-6
    if not np.array_equal(np.asarray(scenario.timestamps), expected):
        raise WaymaxReferenceError(
            "scenario_time_drift",
            "Scenario.timestamps differ from normalized source microseconds",
        )


def _validate_idm_control_accounting(
    compact: CompactWaymaxIDMRollout,
    *,
    state: Any,
    arrays: Mapping[str, np.ndarray],
    num_steps: int,
) -> dict[str, int]:
    masks: dict[str, np.ndarray] = {}
    for field in (
        "requested_control",
        "effective_control",
        "lifecycle_fallback",
        "initialized_overlap_excluded",
    ):
        raw = np.asarray(getattr(compact, field))
        if raw.dtype != np.bool_ or raw.shape != (
            num_steps,
            M4_MAX_OBJECTS,
        ):
            raise WaymaxReferenceError(
                "control_mask_shape",
                f"{field} must be boolean [{num_steps}, {M4_MAX_OBJECTS}]",
            )
        masks[field] = raw

    object_metadata = state.object_metadata
    non_sdc_vehicle = (
        ~np.asarray(object_metadata.is_sdc, dtype=bool)
        & (np.asarray(object_metadata.object_types) == 1)
    )
    logged_valid = np.asarray(state.log_trajectory.valid, dtype=bool)
    requested = np.stack(
        [
            non_sdc_vehicle
            & logged_valid[:, frame]
            & logged_valid[:, frame + 1]
            for frame in range(M4_INIT_STEPS - 1, M4_INIT_STEPS - 1 + num_steps)
        ],
        axis=0,
    )
    initialized_overlap = initialized_overlap_mask_numpy(state)
    expected = {
        "requested_control": requested,
        "effective_control": requested & ~initialized_overlap[np.newaxis, :],
        "lifecycle_fallback": non_sdc_vehicle[np.newaxis, :] & ~requested,
        "initialized_overlap_excluded": (
            requested & initialized_overlap[np.newaxis, :]
        ),
    }
    for field, expected_mask in expected.items():
        if not np.array_equal(masks[field], expected_mask):
            raise WaymaxReferenceError(
                "control_mask_drift",
                f"{field} differs from the declared M4 control semantics",
            )

    # Both fallback categories are claims that the environment used the direct
    # next logged frame. Prove that here rather than trusting aggregate labels.
    fallback = (
        masks["lifecycle_fallback"]
        | masks["initialized_overlap_excluded"]
    )
    interval = slice(M4_INIT_STEPS, M4_INIT_STEPS + num_steps)
    logged_fields = {
        "x": "x",
        "y": "y",
        "yaw": "yaw",
        "vx": "vel_x",
        "vy": "vel_y",
    }
    actual_valid = np.asarray(arrays["valid"], dtype=bool)
    expected_valid = np.asarray(
        state.log_trajectory.valid[:, interval],
        dtype=bool,
    ).T
    if not np.array_equal(actual_valid[fallback], expected_valid[fallback]):
        raise WaymaxReferenceError(
            "fallback_validity",
            "declared IDM fallback validity differs from the direct log",
        )
    for output_name, source_name in logged_fields.items():
        actual = np.asarray(arrays[output_name], dtype=np.float64)
        expected_values = np.asarray(
            getattr(state.log_trajectory, source_name)[:, interval],
            dtype=np.float64,
        ).T
        if output_name == "yaw":
            delta = _normalize_yaw(actual - expected_values)
            compared = np.abs(delta[fallback])
        else:
            compared = np.abs((actual - expected_values)[fallback])
        if not np.all(compared <= M4_FLOAT_ATOL):
            raise WaymaxReferenceError(
                "fallback_motion",
                f"declared IDM fallback {output_name} differs from direct log",
            )

    return {
        "effective_controlled_transitions": int(
            np.count_nonzero(masks["effective_control"])
        ),
        "initialized_overlap_excluded_transitions": int(
            np.count_nonzero(masks["initialized_overlap_excluded"])
        ),
        "initialized_overlap_excluded_vehicles": int(
            np.count_nonzero(
                np.any(masks["initialized_overlap_excluded"], axis=0)
            )
        ),
        "lifecycle_fallbacks": int(
            np.count_nonzero(masks["lifecycle_fallback"])
        ),
        "requested_control_transitions": int(
            np.count_nonzero(masks["requested_control"])
        ),
    }


def compact_waymax_to_rollout(
    compact: CompactWaymaxRollout | CompactWaymaxIDMRollout,
    *,
    state: Any,
    scenario: Scenario,
    sim_name: str,
    seed: int = 0,
    control_accounting: Mapping[str, int] | None = None,
) -> Rollout:
    """Convert compact Waymax frames into the ordinary EvalSim Rollout seam."""

    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    if not isinstance(sim_name, str) or not sim_name.strip():
        raise ValueError("sim_name must be a non-empty string")
    if type(compact) is CompactWaymaxIDMRollout:
        expected_sim_name = WAYMAX_IDM_NAME
    elif type(compact) is CompactWaymaxRollout:
        expected_sim_name = WAYMAX_EXACT_LOG_NAME
    else:
        raise TypeError(
            "compact must be CompactWaymaxRollout or CompactWaymaxIDMRollout"
        )
    if sim_name.strip() != expected_sim_name:
        raise WaymaxReferenceError(
            "sim_name_mismatch",
            "sim_name does not match the compact Waymax reference type",
        )
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or not 0 <= int(seed) <= np.iinfo(np.uint32).max
    ):
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    arrays = _to_numpy_compact(compact)
    num_steps = int(arrays["timestep"].shape[0])
    _validate_transition_count(num_steps)
    _validate_unbatched_state(state, num_steps=num_steps)
    _assert_compact_shape(arrays, num_steps=num_steps)
    expected_steps = np.arange(
        M4_INIT_STEPS,
        M4_INIT_STEPS + num_steps,
        dtype=arrays["timestep"].dtype,
    )
    if not np.array_equal(arrays["timestep"], expected_steps):
        raise WaymaxReferenceError(
            "time_drift",
            "compact timesteps are not the declared post-current horizon",
        )
    if scenario.metadata.get("current_index") != M4_INIT_STEPS - 1:
        raise WaymaxReferenceError(
            "current_boundary",
            "scenario current_index must be 10 for the M4 reference",
        )
    output_horizon = M4_INIT_STEPS + num_steps
    if scenario.num_steps != M4_INIT_STEPS + M4_EXACT_LOG_TRANSITIONS:
        raise WaymaxReferenceError(
            "scenario_horizon",
            "M4 requires an exact 91-frame Scenario",
        )

    metadata = state.object_metadata
    source_valid_any = np.any(np.asarray(state.log_trajectory.valid), axis=1)
    declared_valid = np.asarray(metadata.is_valid, dtype=bool)
    if not np.array_equal(source_valid_any, declared_valid):
        raise WaymaxReferenceError(
            "source_validity",
            "Waymax object metadata and trajectory-valid semantics differ",
        )
    retained = np.flatnonzero(source_valid_any)
    if retained.size != scenario.num_agents:
        raise WaymaxReferenceError(
            "agent_count",
            "scenario retained-agent count differs from Waymax slots",
        )
    source_ids = np.asarray(metadata.ids)[retained]
    if len(set(int(value) for value in source_ids)) != retained.size:
        raise WaymaxReferenceError(
            "duplicate_id",
            "retained Waymax object IDs must be unique",
        )
    scenario_ids = np.asarray(
        [int(agent.id) for agent in scenario.agents],
        dtype=source_ids.dtype,
    )
    if not np.array_equal(source_ids, scenario_ids):
        raise WaymaxReferenceError(
            "agent_order",
            "scenario agent order/IDs differ from retained Waymax slots",
        )
    _validate_scenario_timeline(state, scenario, retained)
    _validate_scenario_history(state, scenario, retained)

    interval = slice(M4_INIT_STEPS, output_horizon)
    direct_timestamps = np.asarray(
        state.log_trajectory.timestamp_micros[:, interval]
    ).T
    if not np.array_equal(
        arrays["timestamp_micros"],
        direct_timestamps,
    ):
        raise WaymaxReferenceError(
            "time_drift",
            "emitted timestamps differ from direct Waymax source timestamps",
        )
    valid = np.asarray(arrays["valid"], dtype=bool)[:, retained].T
    numeric = {
        name: np.asarray(arrays[name], dtype=np.float64)[:, retained].T
        for name in ("x", "y", "yaw", "vx", "vy")
    }
    for name, values in numeric.items():
        if not np.all(np.isfinite(values[valid])):
            raise WaymaxReferenceError(
                "nonfinite_valid",
                f"compact {name} contains non-finite valid values",
            )
        values[~valid] = 0.0
    numeric["yaw"] = _normalize_yaw(numeric["yaw"])
    numeric["yaw"][~valid] = 0.0

    agents = []
    for index, reference in enumerate(scenario.agents):
        agents.append(
            Agent(
                id=int(reference.id),
                type=reference.type,
                valid=np.concatenate(
                    (reference.valid[:M4_INIT_STEPS], valid[index])
                ),
                x=np.concatenate(
                    (reference.x[:M4_INIT_STEPS], numeric["x"][index])
                ),
                y=np.concatenate(
                    (reference.y[:M4_INIT_STEPS], numeric["y"][index])
                ),
                heading=np.concatenate(
                    (
                        reference.heading[:M4_INIT_STEPS],
                        numeric["yaw"][index],
                    )
                ),
                vx=np.concatenate(
                    (reference.vx[:M4_INIT_STEPS], numeric["vx"][index])
                ),
                vy=np.concatenate(
                    (reference.vy[:M4_INIT_STEPS], numeric["vy"][index])
                ),
                length=float(reference.length),
                width=float(reference.width),
            )
        )

    derived_accounting: dict[str, int] | None = None
    if isinstance(compact, CompactWaymaxIDMRollout):
        derived_accounting = _validate_idm_control_accounting(
            compact,
            state=state,
            arrays=arrays,
            num_steps=num_steps,
        )

    supplied_accounting: dict[str, int] = {}
    if control_accounting is not None:
        for key, value in sorted(control_accounting.items()):
            if (
                not isinstance(key, str)
                or not key
                or isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
            ):
                raise WaymaxReferenceError(
                    "control_accounting",
                    "control accounting must be non-negative integer counts",
                )
            supplied_accounting[key] = int(value)
    if derived_accounting is not None:
        if (
            control_accounting is not None
            and supplied_accounting != derived_accounting
        ):
            raise WaymaxReferenceError(
                "control_accounting",
                "supplied IDM control accounting differs from compact masks",
            )
        clean_accounting = derived_accounting
    else:
        if control_accounting is not None:
            raise WaymaxReferenceError(
                "control_accounting",
                "exact-log compact output does not accept control accounting",
            )
        clean_accounting = {}
    rollout_metadata = {
        "backend": "waymax",
        "backend_commit": WAYMAX_COMMIT,
        "compact_reference_version": WAYMAX_REFERENCE_VERSION,
        "control_accounting": clean_accounting,
        "horizon_transitions": num_steps,
        "init_steps": M4_INIT_STEPS,
        "invalid_fill": "finite_zero_where_invalid",
        "reference_config_fingerprint": reference_config_fingerprint(),
        "rollout_start_index": M4_INIT_STEPS - 1,
        "scenario_source": scenario.metadata.get("source", "unknown"),
        "scenario_source_fingerprint": scenario.metadata.get(
            "source_fingerprint"
        ),
        "time_source": "direct_waymax_emission_checked_against_log",
    }
    # This is a contract assertion, not merely a serialization convenience.
    try:
        json.dumps(
            rollout_metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise WaymaxReferenceError(
            "provenance_json",
            "generated rollout provenance is not JSON-native",
        ) from exc
    return Rollout(
        scenario_id=scenario.scenario_id,
        sim_name=sim_name.strip(),
        sim_version=WAYMAX_REFERENCE_VERSION,
        seed=int(seed),
        timestamps=np.array(
            scenario.timestamps[:output_horizon],
            copy=True,
        ),
        agents=agents,
        perturbation=None,
        metadata=rollout_metadata,
    )


__all__ = [
    "CompactWaymaxIDMRollout",
    "CompactWaymaxRollout",
    "M4_EXACT_LOG_TRANSITIONS",
    "M4_FLOAT_ATOL",
    "M4_IDM_TRANSITIONS",
    "M4_INIT_STEPS",
    "M4_MAX_OBJECTS",
    "WAYMAX_EXACT_LOG_NAME",
    "WAYMAX_IDM_DEFAULTS",
    "WAYMAX_IDM_NAME",
    "WAYMAX_REFERENCE_VERSION",
    "WaymaxReferenceDependencyError",
    "WaymaxReferenceError",
    "assert_waymax_idm_defaults",
    "compact_exact_log_rollout",
    "compact_rollout_bytes",
    "compact_stock_exact_log_rollout",
    "compact_waymax_idm_rollout",
    "compact_waymax_to_rollout",
    "exact_log_config_payload",
    "initialized_overlap_mask_numpy",
    "numpy_pairwise_overlaps",
    "reference_config_fingerprint",
    "single_scene_exact_log_kernel",
    "single_scene_idm_kernel",
    "validate_exact_log_compact",
    "validate_stock_equivalence",
    "waymax_idm_scalar_acceleration",
]
