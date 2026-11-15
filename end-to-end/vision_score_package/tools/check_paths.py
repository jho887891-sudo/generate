#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required = {
        "vision_train_root": cfg["vision_train_root"],
        "test_labelme_dir": cfg["test_labelme_dir"],
    }
    for name, path in required.items():
        p = Path(path)
        print(f"[check] {name}: {p} exists={p.exists()}")
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    for name, path in cfg.get("weights", {}).items():
        p = Path(path)
        status = "OK" if p.exists() and p.stat().st_size > 0 else "MISSING/EMPTY"
        size = p.stat().st_size if p.exists() else 0
        print(f"[check] weight {name:<20} {status:<13} {p} ({size} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
