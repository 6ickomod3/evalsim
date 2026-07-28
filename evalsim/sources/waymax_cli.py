"""Opt-in, local-only M3 acceptance command."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from evalsim.contracts import (
    AgentFrame,
    AgentType,
    PolicyObservation,
    Scenario,
    scenario_from_parquet,
    scenario_to_parquet,
)
from evalsim.rollout import DynamicsLimits, RolloutEngine
from evalsim.simulators import (
    ConstantVelocityPolicy,
    IDMPolicy,
    LogReplayPolicy,
)
from evalsim.viz import plot_scenario

from .waymax import WAYMAX_COMMIT
from .waymax_loader import (
    DEFAULT_WOMD_VALIDATION_DIR,
    LOCAL_WAYMO_ENV_FLAG,
    WOMD_M3_SEARCH_LIMIT,
    WaymaxSource,
    runtime_summary,
    validate_record_parity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalsim-waymax-smoke",
        description=(
            "Run the opt-in local M3 WOMD/Waymax vertical-slice acceptance. "
            "No raw ID or trajectory value is printed."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "EvalSim Git checkout root (default: current working directory). "
            "The command intentionally writes only to an ignored checkout."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_WOMD_VALIDATION_DIR,
        help="Local ignored WOMD v1.3.1 TFExample validation directory.",
    )
    parser.add_argument(
        "--shard",
        default="00000",
        help="M3 is pre-registered to exact validation shard 00000.",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=WOMD_M3_SEARCH_LIMIT,
        help="Bound for the pre-registered earliest-eligible search (default: 32).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/m3"),
        help="Ignored local output directory; must remain under outputs/m3.",
    )
    return parser


def _project_root(candidate: Path | None = None) -> Path:
    """Resolve an explicit checkout, independent of package installation location."""

    root = (Path.cwd() if candidate is None else candidate).resolve()
    if not (root / "pyproject.toml").is_file() or not (
        root / ".gitignore"
    ).is_file():
        raise ValueError(
            "run from an EvalSim Git checkout or pass --project-root explicitly"
        )
    worktree = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if worktree.returncode != 0:
        raise ValueError("project root must be an EvalSim Git worktree")
    try:
        resolved_worktree = Path(worktree.stdout.strip()).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("could not validate the EvalSim Git worktree") from exc
    if resolved_worktree != root:
        raise ValueError("--project-root must name the Git worktree root exactly")
    return root


def _resolve_project_path(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _prepare_output_directory(path: Path, root: Path) -> Path:
    output = _resolve_project_path(path, root)
    allowed = (root / "outputs" / "m3").resolve()
    if output != allowed and allowed not in output.parents:
        raise ValueError("output directory must be outputs/m3 or its descendant")
    relative = output.relative_to(root)
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(relative)],
        cwd=root,
        check=False,
    )
    if ignored.returncode != 0:
        raise RuntimeError("refusing to write because the M3 output path is not ignored")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _assert_scenarios_equal(left: Scenario, right: Scenario) -> None:
    if (
        left.scenario_id != right.scenario_id
        or left.ego_index != right.ego_index
        or left.metadata != right.metadata
        or not np.array_equal(left.timestamps, right.timestamps)
        or len(left.agents) != len(right.agents)
        or len(left.map) != len(right.map)
    ):
        raise AssertionError("scenario scalar/list fields are not deterministic")
    for left_agent, right_agent in zip(left.agents, right.agents):
        if (
            left_agent.id != right_agent.id
            or left_agent.type != right_agent.type
            or left_agent.length != right_agent.length
            or left_agent.width != right_agent.width
        ):
            raise AssertionError("scenario agent metadata is not deterministic")
        for field in ("valid", "x", "y", "heading", "vx", "vy"):
            if not np.array_equal(
                getattr(left_agent, field),
                getattr(right_agent, field),
            ):
                raise AssertionError(
                    f"scenario agent field {field} is not deterministic"
                )
    for left_feature, right_feature in zip(left.map, right.map):
        if left_feature.type != right_feature.type or not np.array_equal(
            left_feature.xy,
            right_feature.xy,
        ):
            raise AssertionError("scenario map is not deterministic")


def _assert_log_replay_exact(scenario: Scenario, rollout) -> None:
    if rollout.scenario_id != scenario.scenario_id:
        raise AssertionError("log replay changed scenario identity")
    for source_agent, rollout_agent in zip(scenario.agents, rollout.agents):
        for field in ("valid", "x", "y", "heading", "vx", "vy"):
            if not np.array_equal(
                getattr(source_agent, field),
                getattr(rollout_agent, field),
            ):
                raise AssertionError(f"log replay changed agent field {field}")


def _first_policy_observation(scenario: Scenario) -> PolicyObservation:
    current = int(scenario.metadata.get("current_index", 0))
    next_index = current + 1
    return PolicyObservation(
        current_index=current,
        next_index=next_index,
        timestamp=float(scenario.timestamps[current]),
        next_timestamp=float(scenario.timestamps[next_index]),
        dt=float(scenario.timestamps[next_index] - scenario.timestamps[current]),
        frame=AgentFrame.from_scenario(scenario, current),
        next_valid=np.asarray(
            [agent.valid[next_index] for agent in scenario.agents],
            dtype=bool,
        ),
        agent_ids=tuple(int(agent.id) for agent in scenario.agents),
        agent_types=tuple(agent.type for agent in scenario.agents),
        lengths=np.asarray([agent.length for agent in scenario.agents]),
        widths=np.asarray([agent.width for agent in scenario.agents]),
        ego_index=scenario.ego_index,
    )


def _independent_zero_yaw_transition(
    frame: AgentFrame,
    *,
    agent_index: int,
    acceleration: float,
    dt: float,
    limits: DynamicsLimits,
) -> tuple[float, float, float, float, float]:
    """Independent scalar oracle for one engine transition with zero yaw control."""

    raw_speed = float(np.hypot(frame.vx[agent_index], frame.vy[agent_index]))
    current_speed = min(raw_speed, limits.max_speed_mps)
    bounded_acceleration = float(
        np.clip(
            acceleration,
            -limits.max_deceleration_mps2,
            limits.max_acceleration_mps2,
        )
    )
    proposed_speed = current_speed + bounded_acceleration * dt
    next_speed = float(np.clip(proposed_speed, 0.0, limits.max_speed_mps))
    if raw_speed > 1e-12:
        motion_heading = float(
            np.arctan2(frame.vy[agent_index], frame.vx[agent_index])
        )
    else:
        motion_heading = float(frame.heading[agent_index])

    if bounded_acceleration < 0.0 and proposed_speed < 0.0:
        motion_duration = current_speed / -bounded_acceleration
        travel = (
            current_speed * motion_duration
            + 0.5 * bounded_acceleration * motion_duration**2
        )
    elif (
        bounded_acceleration > 0.0
        and proposed_speed > limits.max_speed_mps
    ):
        time_to_limit = (
            limits.max_speed_mps - current_speed
        ) / bounded_acceleration
        travel = (
            current_speed * time_to_limit
            + 0.5 * bounded_acceleration * time_to_limit**2
            + limits.max_speed_mps * (dt - time_to_limit)
        )
    else:
        travel = 0.5 * (current_speed + next_speed) * dt

    if raw_speed <= 1e-12 and next_speed <= 1e-12:
        next_heading = float(frame.heading[agent_index])
    else:
        wrapped = (motion_heading + np.pi) % (2.0 * np.pi) - np.pi
        if math.isclose(wrapped, -np.pi, rel_tol=0.0, abs_tol=1e-15) and (
            motion_heading > 0.0
        ):
            wrapped = np.pi
        next_heading = float(wrapped)
    return (
        float(frame.x[agent_index] + travel * np.cos(motion_heading)),
        float(frame.y[agent_index] + travel * np.sin(motion_heading)),
        next_heading,
        float(next_speed * np.cos(next_heading)),
        float(next_speed * np.sin(next_heading)),
    )


def _rollout_transition_matches(
    rollout,
    *,
    agent_index: int,
    next_index: int,
    expected: tuple[float, float, float, float, float],
) -> bool:
    agent = rollout.agents[agent_index]
    actual = (
        float(agent.x[next_index]),
        float(agent.y[next_index]),
        float(agent.heading[next_index]),
        float(agent.vx[next_index]),
        float(agent.vy[next_index]),
    )
    return all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
        for left, right in zip(actual, expected, strict=True)
    )


def _run_policy_acceptance(scenario: Scenario) -> dict[str, bool]:
    engine = RolloutEngine()
    replay = engine.run(scenario, LogReplayPolicy(), seed=2026)
    _assert_log_replay_exact(scenario, replay)
    cv_policy = ConstantVelocityPolicy()
    idm_policy = IDMPolicy()
    cv = engine.run(scenario, cv_policy, seed=2026)
    idm = engine.run(scenario, idm_policy, seed=2026)

    current = int(scenario.metadata.get("current_index", 0))
    next_index = current + 1
    observation = _first_policy_observation(scenario)
    cv_step = cv_policy.step(
        cv_policy.initialize(scenario, seed=2026),
        observation,
    )
    idm_step = idm_policy.step(
        idm_policy.initialize(scenario, seed=2026),
        observation,
    )

    accepted_index: int | None = None
    for index, agent in enumerate(scenario.agents):
        if not (
            index != scenario.ego_index
            and agent.type == AgentType.VEHICLE
            and agent.valid[current]
            and agent.valid[next_index]
        ):
            continue
        cv_expected = _independent_zero_yaw_transition(
            observation.frame,
            agent_index=index,
            acceleration=float(cv_step.longitudinal_acceleration[index]),
            dt=observation.dt,
            limits=engine.dynamics_limits,
        )
        idm_acceleration = float(idm_step.longitudinal_acceleration[index])
        idm_expected = _independent_zero_yaw_transition(
            observation.frame,
            agent_index=index,
            acceleration=idm_acceleration,
            dt=observation.dt,
            limits=engine.dynamics_limits,
        )
        cv_displacement = float(
            np.hypot(
                cv_expected[0] - observation.frame.x[index],
                cv_expected[1] - observation.frame.y[index],
            )
        )
        policy_separation = any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
            for left, right in zip(cv_expected, idm_expected, strict=True)
        )
        if (
            cv_displacement > 1e-9
            and abs(idm_acceleration) > 1e-9
            and policy_separation
        ):
            if not _rollout_transition_matches(
                cv,
                agent_index=index,
                next_index=next_index,
                expected=cv_expected,
            ):
                raise AssertionError(
                    "constant-velocity numeric first transition failed its "
                    "independent scalar oracle"
                )
            if not _rollout_transition_matches(
                idm,
                agent_index=index,
                next_index=next_index,
                expected=idm_expected,
            ):
                raise AssertionError(
                    "IDM numeric first transition failed its independent scalar oracle"
                )
            accepted_index = index
            break
    if accepted_index is None:
        raise AssertionError(
            "no eligible real vehicle produced both a moving CV transition and a "
            "nonzero, numerically distinct IDM transition"
        )

    eligible_vehicle = scenario.agents[accepted_index]
    if idm.metadata["agent_control_modes"][str(eligible_vehicle.id)] != "idm":
        raise AssertionError("IDM's vehicle-control branch did not execute")
    if cv.metadata["agent_control_modes"][str(eligible_vehicle.id)] != (
        "constant_velocity"
    ):
        raise AssertionError("constant-velocity's world-agent branch did not execute")
    return {
        "log_replay_exact": True,
        "constant_velocity_numeric_transition": True,
        "idm_numeric_vehicle_transition": True,
    }


def run_smoke(args: argparse.Namespace) -> Path:
    if os.environ.get(LOCAL_WAYMO_ENV_FLAG) != "1":
        raise RuntimeError(
            f"set {LOCAL_WAYMO_ENV_FLAG}=1 to opt in to local WOMD access"
        )
    if str(args.shard) != "00000":
        raise ValueError("M3 smoke is pre-registered to exact shard 00000")
    if args.search_limit != WOMD_M3_SEARCH_LIMIT:
        raise ValueError(
            f"M3 smoke is pre-registered to search_limit={WOMD_M3_SEARCH_LIMIT}"
        )

    # Set bounded defaults before the first optional TensorFlow import.
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

    root = _project_root(getattr(args, "project_root", None))
    output_dir = _prepare_output_directory(args.output_dir, root)
    data_dir = _resolve_project_path(args.data_dir, root)
    source = WaymaxSource(
        data_dir=data_dir,
        shard_index=0,
        search_limit=args.search_limit,
    )
    selection = source.load_first_eligible()
    scenario = selection.scenario

    parity = dict(validate_record_parity(selection.record, scenario))
    repeated = source.scenario_from_record(selection.record)
    _assert_scenarios_equal(scenario, repeated)

    parquet_path = output_dir / "waymax_scenario.parquet"
    scenario_to_parquet(scenario, parquet_path)
    round_tripped = scenario_from_parquet(parquet_path)
    _assert_scenarios_equal(scenario, round_tripped)

    figure, _ = plot_scenario(scenario)
    figure.savefig(output_dir / "waymax_scenario.png", dpi=150)
    import matplotlib.pyplot as plt

    plt.close(figure)
    policies = _run_policy_acceptance(scenario)

    runtime = runtime_summary()
    if runtime["jax_backend"] != "cpu":
        raise AssertionError("M3 local acceptance requires the JAX CPU backend")
    import jax
    import jax.numpy as jnp

    jit_result = jax.jit(lambda values: values * 2 + 1)(
        jnp.asarray([1.0, 2.0])
    )
    if not np.array_equal(np.asarray(jit_result), np.asarray([3.0, 5.0])):
        raise AssertionError("JAX JIT smoke returned an unexpected result")

    rejection_counts = Counter(item.code for item in selection.rejections)
    report = {
        "schema_version": "1",
        "accepted": True,
        "purpose": "personal_non_commercial_experimentation",
        "source": {
            "dataset": "Waymo Open Motion Dataset",
            "version": "1.3.1",
            "split": "validation",
            "shard_suffix": "00000",
            "selection_rule": (
                "earliest_supported_map_and_sdc_world_vehicle_current_to_next_"
                "within_first_32"
            ),
            "selected_record_ordinal": selection.record.record_ordinal,
            "pre_selection_rejections": dict(sorted(rejection_counts.items())),
        },
        "checks": {
            **parity,
            "deterministic_conversion": True,
            "parquet_round_trip": True,
            "visualization": True,
            "jax_cpu_jit": True,
            **policies,
        },
        "runtime": runtime,
        "waymax_commit": WAYMAX_COMMIT,
        "artifacts": {
            "parquet": parquet_path.name,
            "visualization": "waymax_scenario.png",
        },
        "privacy": {
            "native_scenario_id_reported": False,
            "coordinates_reported": False,
            "absolute_paths_reported": False,
        },
    }
    report_path = output_dir / "acceptance_summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Fail if a future ignore-rule change makes any output visible to Git.
    for artifact in (parquet_path, output_dir / "waymax_scenario.png", report_path):
        relative = artifact.relative_to(root)
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--", str(relative)],
            cwd=root,
            check=False,
        )
        if ignored.returncode != 0:
            raise RuntimeError("an M3 artifact is not ignored by Git")
    return report_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report_path = run_smoke(args)
    except Exception as exc:
        # Third-party filesystem/data exceptions may embed absolute paths or source
        # values. Keep terminal output to a stable code/type and leave local debugging
        # to an explicitly inspected traceback.
        failure = getattr(exc, "code", type(exc).__name__)
        parser.exit(1, f"M3 local smoke: FAIL ({failure})\n")
    root = _project_root(getattr(args, "project_root", None))
    relative_report = report_path.relative_to(root)
    print("M3 local smoke: PASS")
    print("Native WOMD identity was verified internally and was not printed.")
    print(f"Ignored local report: {relative_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
