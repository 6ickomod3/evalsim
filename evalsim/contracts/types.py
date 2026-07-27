"""Shared enums for the EvalSim data contracts."""
from __future__ import annotations

from enum import Enum


class AgentType(str, Enum):
    """Type of a traffic agent. Inherits from ``str`` so values serialize cleanly."""

    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    UNKNOWN = "unknown"


class MapType(str, Enum):
    """Type of a map polyline feature."""

    LANE = "lane"
    ROAD_EDGE = "road_edge"
    CROSSWALK = "crosswalk"
    STOP_LINE = "stop_line"
    UNKNOWN = "unknown"
