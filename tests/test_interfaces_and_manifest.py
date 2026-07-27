"""Interfaces are abstract; RunManifest round-trips through JSON."""
from __future__ import annotations

import numpy as np
import pytest

from evalsim import (
    Metric,
    MetricResult,
    MetricSpec,
    PolicyMetadata,
    RunManifest,
    SimulatorPolicy,
)


def test_simulator_policy_is_abstract():
    with pytest.raises(TypeError):
        SimulatorPolicy()  # cannot instantiate abstract base


def test_metric_is_abstract():
    with pytest.raises(TypeError):
        Metric()


def test_minimal_simulator_and_metric_subclasses():
    class NoopSim(SimulatorPolicy):
        def initialize(self, scenario, seed):
            return {"t": 0}

        def step(self, state, observation):
            return {"t": state["t"] + 1}

        def metadata(self):
            return PolicyMetadata(name="noop", version="0.0.1", deterministic=True)

    class ConstMetric(Metric):
        spec = MetricSpec(name="const", version="0.0.1")

        def validate_inputs(self, scenario, rollout):
            return True

        def compute(self, scenario, rollout):
            return MetricResult("const", "0.0.1", scenario.scenario_id, 1.0)

        def aggregate(self, per_scenario_values):
            return {"mean": float(np.mean(per_scenario_values)), "n": len(per_scenario_values)}

    sim = NoopSim()
    assert sim.step(sim.initialize(None, 0), None)["t"] == 1
    assert sim.metadata().name == "noop"

    m = ConstMetric()
    assert m.validate_inputs(None, None) is True
    agg = m.aggregate([1.0, 1.0, 1.0])
    assert agg["mean"] == pytest.approx(1.0) and agg["n"] == 3


def test_run_manifest_json_roundtrip(tmp_path):
    manifest = RunManifest(
        run_id="run_0001",
        dataset_version="synthetic-v1",
        scenario_manifest="manifests/eval_500.json",
        sim_name="idm",
        sim_version="0.1.0",
        sim_config={"desired_speed_mps": 13.0, "minimum_gap_m": 2.0},
        seeds=[42],
        perturbations=["ego_brake_hard"],
        metric_versions={"collision": "0.1.0"},
        statistics={"bootstrap_samples": 1000, "confidence_level": 0.95},
        code_commit="abc123",
    )
    restored = RunManifest.from_json(manifest.to_json())
    assert restored == manifest

    path = tmp_path / "manifest.json"
    manifest.to_file(path)
    assert RunManifest.from_file(path) == manifest
