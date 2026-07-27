"""Parquet (de)serialization for the ``Scenario`` and ``Rollout`` contracts.

Layout: agents (variable-length time series) are stored as the Parquet *table* (one row
per agent, with list<double> columns). Everything else — scalars, timestamps, map
geometry, metadata — is JSON-encoded into the file-level key/value metadata. The result
is a single self-contained ``.parquet`` file per scenario/rollout that round-trips
losslessly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .scenario import AGENT_SERIES_FIELDS, Agent, MapPolyline, Scenario
from .rollout import Rollout
from .types import AgentType, MapType

_KIND_KEY = b"evalsim.kind"
_PAYLOAD_KEY = b"evalsim.payload"


def _agents_to_table(agents: list[Agent]) -> pa.Table:
    cols: dict[str, list] = {
        "agent_id": [a.id for a in agents],
        "type": [a.type.value for a in agents],
        "length": [a.length for a in agents],
        "width": [a.width for a in agents],
    }
    for f in AGENT_SERIES_FIELDS:
        cols[f] = [np.asarray(getattr(a, f)).tolist() for a in agents]
    return pa.table(cols)


def _agents_from_table(table: pa.Table) -> list[Agent]:
    d = table.to_pydict()
    agents: list[Agent] = []
    for i in range(table.num_rows):
        agents.append(
            Agent(
                id=int(d["agent_id"][i]),
                type=AgentType(d["type"][i]),
                valid=np.asarray(d["valid"][i], dtype=bool),
                x=np.asarray(d["x"][i], dtype=float),
                y=np.asarray(d["y"][i], dtype=float),
                heading=np.asarray(d["heading"][i], dtype=float),
                vx=np.asarray(d["vx"][i], dtype=float),
                vy=np.asarray(d["vy"][i], dtype=float),
                length=float(d["length"][i]),
                width=float(d["width"][i]),
            )
        )
    return agents


def _map_to_json(polylines: list[MapPolyline]) -> list[dict]:
    return [{"type": p.type.value, "xy": np.asarray(p.xy).tolist()} for p in polylines]


def _map_from_json(items: list[dict]) -> list[MapPolyline]:
    return [MapPolyline(type=MapType(it["type"]), xy=np.asarray(it["xy"], dtype=float)) for it in items]


def _write(table: pa.Table, kind: str, payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(table.schema.metadata or {})
    meta[_KIND_KEY] = kind.encode()
    meta[_PAYLOAD_KEY] = json.dumps(payload).encode()
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, path)


def _read(path: str | Path) -> tuple[pa.Table, str, dict]:
    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    kind = meta.get(_KIND_KEY, b"").decode()
    payload = json.loads(meta.get(_PAYLOAD_KEY, b"{}").decode())
    return table, kind, payload


# -- Scenario -----------------------------------------------------------------

def scenario_to_parquet(scenario: Scenario, path: str | Path) -> None:
    payload = {
        "scenario_id": scenario.scenario_id,
        "timestamps": np.asarray(scenario.timestamps).tolist(),
        "ego_index": scenario.ego_index,
        "map": _map_to_json(scenario.map),
        "metadata": scenario.metadata,
    }
    _write(_agents_to_table(scenario.agents), "scenario", payload, path)


def scenario_from_parquet(path: str | Path) -> Scenario:
    table, kind, payload = _read(path)
    if kind != "scenario":
        raise ValueError(f"Expected a scenario parquet, got kind={kind!r}")
    return Scenario(
        scenario_id=payload["scenario_id"],
        timestamps=np.asarray(payload["timestamps"], dtype=float),
        agents=_agents_from_table(table),
        map=_map_from_json(payload["map"]),
        ego_index=int(payload["ego_index"]),
        metadata=payload["metadata"],
    )


# -- Rollout ------------------------------------------------------------------

def rollout_to_parquet(rollout: Rollout, path: str | Path) -> None:
    payload = {
        "scenario_id": rollout.scenario_id,
        "sim_name": rollout.sim_name,
        "sim_version": rollout.sim_version,
        "seed": rollout.seed,
        "perturbation": rollout.perturbation,
        "timestamps": np.asarray(rollout.timestamps).tolist(),
        "metadata": rollout.metadata,
    }
    _write(_agents_to_table(rollout.agents), "rollout", payload, path)


def rollout_from_parquet(path: str | Path) -> Rollout:
    table, kind, payload = _read(path)
    if kind != "rollout":
        raise ValueError(f"Expected a rollout parquet, got kind={kind!r}")
    return Rollout(
        scenario_id=payload["scenario_id"],
        sim_name=payload["sim_name"],
        sim_version=payload["sim_version"],
        seed=int(payload["seed"]),
        timestamps=np.asarray(payload["timestamps"], dtype=float),
        agents=_agents_from_table(table),
        perturbation=payload["perturbation"],
        metadata=payload["metadata"],
    )
