"""Static visualizations for EvalSim scenarios."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evalsim.contracts import AgentType, MapType, Scenario

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


_MAP_STYLES = {
    MapType.LANE: {
        "color": "#8a94a6",
        "linestyle": "--",
        "linewidth": 1.2,
        "label": "lane",
    },
    MapType.ROAD_EDGE: {
        "color": "#303846",
        "linestyle": "-",
        "linewidth": 2.0,
        "label": "road edge",
    },
    MapType.CROSSWALK: {
        "color": "#d6a84b",
        "linestyle": "-",
        "linewidth": 1.6,
        "label": "crosswalk",
    },
    MapType.STOP_LINE: {
        "color": "#d66565",
        "linestyle": "-",
        "linewidth": 2.0,
        "label": "stop line",
    },
    MapType.UNKNOWN: {
        "color": "#a0a0a0",
        "linestyle": ":",
        "linewidth": 1.0,
        "label": "map feature",
    },
}

_AGENT_COLORS = {
    AgentType.VEHICLE: "#3979d1",
    AgentType.PEDESTRIAN: "#d14d8b",
    AgentType.CYCLIST: "#35a36f",
    AgentType.UNKNOWN: "#777777",
}


def plot_scenario(
    scenario: Scenario,
    ax: Axes | None = None,
    *,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Render map geometry and valid agent trajectories.

    Invalid trajectory samples become ``NaN`` gaps, preventing Matplotlib from drawing
    a misleading line across periods where an agent was not observed.  The function
    never calls ``show`` and returns both figure and axes so callers control saving.
    """

    import matplotlib.pyplot as plt

    if ax is None:
        figure, ax = plt.subplots(figsize=(9.0, 7.0), constrained_layout=True)
    else:
        figure = ax.figure

    used_labels: set[str] = set()
    for feature in scenario.map:
        style = _MAP_STYLES.get(feature.type, _MAP_STYLES[MapType.UNKNOWN])
        label = str(style["label"])
        ax.plot(
            feature.xy[:, 0],
            feature.xy[:, 1],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=label if label not in used_labels else None,
            zorder=1,
        )
        used_labels.add(label)

    raw_current_index = scenario.metadata.get("current_index")
    current_index = (
        int(raw_current_index)
        if (
            not isinstance(raw_current_index, (bool, np.bool_))
            and isinstance(raw_current_index, (int, np.integer))
            and 0 <= int(raw_current_index) < scenario.num_steps
        )
        else None
    )

    for agent_index, agent in enumerate(scenario.agents):
        finite = (
            np.isfinite(agent.x)
            & np.isfinite(agent.y)
            & np.isfinite(agent.heading)
        )
        visible = agent.valid & finite
        x = np.where(visible, agent.x, np.nan)
        y = np.where(visible, agent.y, np.nan)
        is_ego = agent_index == scenario.ego_index
        if is_ego:
            color = "#f2b134"
            label = "ego"
            linewidth = 3.2
            zorder = 4
        else:
            color = _AGENT_COLORS.get(agent.type, _AGENT_COLORS[AgentType.UNKNOWN])
            label = agent.type.value
            linewidth = 1.8
            zorder = 3

        if current_index is None:
            ax.plot(
                x,
                y,
                color=color,
                linewidth=linewidth,
                label=label if label not in used_labels else None,
                zorder=zorder,
            )
        else:
            history_x = np.array(x, copy=True)
            history_y = np.array(y, copy=True)
            history_x[current_index + 1 :] = np.nan
            history_y[current_index + 1 :] = np.nan
            future_x = np.array(x, copy=True)
            future_y = np.array(y, copy=True)
            future_x[:current_index] = np.nan
            future_y[:current_index] = np.nan
            ax.plot(
                history_x,
                history_y,
                color=color,
                linewidth=linewidth,
                label=label if label not in used_labels else None,
                zorder=zorder,
            )
            ax.plot(
                future_x,
                future_y,
                color=color,
                linewidth=max(1.0, linewidth * 0.75),
                linestyle=":",
                alpha=0.72,
                label=None,
                zorder=zorder,
            )
        used_labels.add(label)

        valid_indices = np.flatnonzero(visible)
        if valid_indices.size:
            first = int(valid_indices[0])
            last = int(valid_indices[-1])
            ax.scatter(
                [agent.x[first]],
                [agent.y[first]],
                color=color,
                marker="o",
                s=32 if is_ego else 20,
                zorder=zorder + 1,
            )
            ax.scatter(
                [agent.x[last]],
                [agent.y[last]],
                color=color,
                marker="s",
                s=64 if is_ego else 40,
                zorder=zorder + 1,
            )
            if (
                current_index is not None
                and is_ego
                and visible[current_index]
            ):
                ax.scatter(
                    [agent.x[current_index]],
                    [agent.y[current_index]],
                    color="#ffffff",
                    edgecolor=color,
                    linewidth=1.5,
                    marker="D",
                    s=72,
                    label="current frame"
                    if "current frame" not in used_labels
                    else None,
                    zorder=zorder + 2,
                )
                used_labels.add("current frame")

    scenario_kind = scenario.metadata.get("scenario_kind")
    default_title = scenario.scenario_id
    if isinstance(scenario_kind, str) and scenario_kind:
        default_title = f"{scenario_kind.replace('_', ' ').title()} — {scenario.scenario_id}"
    ax.set_title(title if title is not None else default_title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(color="#d8dce3", linewidth=0.5, alpha=0.45)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best", framealpha=0.9)
    return figure, ax


__all__ = ["plot_scenario"]
