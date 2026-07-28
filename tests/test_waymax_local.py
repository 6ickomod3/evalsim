"""Opt-in M3 acceptance against ignored local WOMD shard 00000."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

from evalsim.sources.waymax_loader import LOCAL_WAYMO_ENV_FLAG

pytestmark = pytest.mark.waymo_local

if os.environ.get(LOCAL_WAYMO_ENV_FLAG) != "1":
    pytest.skip(
        f"set {LOCAL_WAYMO_ENV_FLAG}=1 to opt in to local WOMD access",
        allow_module_level=True,
    )

from evalsim.sources.waymax_cli import run_smoke


def test_local_waymax_vertical_slice() -> None:
    report_path = run_smoke(
        argparse.Namespace(
            data_dir=Path(
                "data/raw/womd/v1.3.1/tf_example/validation"
            ),
            shard="00000",
            search_limit=32,
            output_dir=Path("outputs/m3/pytest"),
        )
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["accepted"] is True
    assert all(report["checks"].values())
    assert report["privacy"] == {
        "absolute_paths_reported": False,
        "coordinates_reported": False,
        "native_scenario_id_reported": False,
    }
