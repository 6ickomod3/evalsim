"""M1 scenario-visualization tests."""
from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from evalsim.sources import SyntheticSource
from evalsim.viz import plot_scenario


def test_all_five_scenario_kinds_render_to_png(tmp_path) -> None:
    scenarios = SyntheticSource(seed=5).generate(5)

    for index, scenario in enumerate(scenarios):
        figure, axes = plot_scenario(scenario)
        path = tmp_path / f"scenario-{index}.png"
        figure.savefig(path, dpi=100)

        assert axes.get_aspect() == 1.0
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 10_000
        plt.close(figure)


def test_plot_uses_nan_gaps_for_invalid_samples_and_accepts_axes() -> None:
    scenario = SyntheticSource(seed=9).generate_one(0)
    figure, supplied_axes = plt.subplots()

    returned_figure, returned_axes = plot_scenario(scenario, ax=supplied_axes)

    assert returned_figure is figure
    assert returned_axes is supplied_axes
    agent_lines = supplied_axes.lines[len(scenario.map) :]
    follower_line = agent_lines[2]
    follower_valid = scenario.agents[2].valid
    plotted_x = np.asarray(follower_line.get_xdata(), dtype=float)
    assert np.all(np.isnan(plotted_x[~follower_valid]))
    assert np.all(np.isfinite(plotted_x[follower_valid]))
    assert any(line.get_label() == "ego" for line in agent_lines)
    plt.close(figure)


def test_plot_tolerates_non_string_optional_kind_metadata() -> None:
    scenario = SyntheticSource().generate_one(0)
    scenario.metadata["scenario_kind"] = 123

    figure, axes = plot_scenario(scenario)

    assert axes.get_title() == scenario.scenario_id
    plt.close(figure)
