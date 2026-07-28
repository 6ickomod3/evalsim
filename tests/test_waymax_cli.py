"""Data-free tests for the M3 local acceptance command."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from evalsim.sources import SyntheticSource
from evalsim.sources import waymax_cli
from evalsim.sources.waymax_loader import _eligibility_rejection


def test_policy_acceptance_checks_numeric_cv_and_idm_transitions() -> None:
    scenario = SyntheticSource(seed=1).generate_one(0, "following")

    assert waymax_cli._run_policy_acceptance(scenario) == {
        "log_replay_exact": True,
        "constant_velocity_numeric_transition": True,
        "idm_numeric_vehicle_transition": True,
    }

    # The pre-registered selector asks only for an SDC current→next transition.
    # Future validity is a downstream rollout acceptance gate, not a reason to scan
    # ahead and select a different record.
    ego_valid = np.array(scenario.ego.valid, copy=True)
    ego_valid[2] = False
    agents = list(scenario.agents)
    agents[scenario.ego_index] = dataclasses.replace(
        scenario.ego,
        valid=ego_valid,
    )
    selection_probe = dataclasses.replace(
        scenario,
        agents=agents,
        metadata={**scenario.metadata, "current_index": 0},
    )
    assert _eligibility_rejection(selection_probe) is None


def test_project_root_does_not_depend_on_installed_module_location(
    monkeypatch,
) -> None:
    checkout = Path.cwd().resolve()
    monkeypatch.setattr(
        waymax_cli,
        "__file__",
        "/isolated/site-packages/evalsim/sources/waymax_cli.py",
    )
    monkeypatch.chdir(checkout)

    assert waymax_cli._project_root() == checkout
