from __future__ import annotations

import os
import sys
from pathlib import Path


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_END_TO_END_ROOT = Path(os.environ.get("SVP_TEXT_END_TO_END_ROOT", "/home/T7/wkq/end-to-end"))


def find_region_route_root() -> Path:
    candidates = [
        Path(os.environ.get("DIECAST_REGION_ROUTE_ROOT", "")),
        DEFAULT_END_TO_END_ROOT / "diecast_region_route",
        SYSTEM_ROOT.parent / "diecast_region_route",
        SYSTEM_ROOT.parent / "diecast_region_route_4B",
        SYSTEM_ROOT.parent / "diecast_region_route",
    ]
    for path in candidates:
        if not str(path):
            continue
        if (path / "e2e_compare_three_modes_graph_4b_v3.py").exists():
            return path
    return DEFAULT_END_TO_END_ROOT / "diecast_region_route"


def ensure_region_route_on_path() -> Path:
    route_root = find_region_route_root()
    route_str = str(route_root)
    if route_str not in sys.path:
        sys.path.insert(0, route_str)
    return route_root

