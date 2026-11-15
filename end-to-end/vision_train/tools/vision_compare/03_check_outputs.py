#!/usr/bin/env python3
"""Check whether expected best weights have been generated."""
from __future__ import annotations

import argparse
from pathlib import Path

from common import ensure_dir, load_config, setup_logger, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/home/T7/wkq/end-to-end/vision_train/configs/vision_compare.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dir(cfg["logs_dir"])
    logger = setup_logger("check_outputs", Path(cfg["logs_dir"]) / "03_check_outputs.log")

    report = {}
    ok = True
    for name, path_str in cfg.get("expected_weights", {}).items():
        p = Path(path_str)
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        nonempty = exists and size > 0
        report[name] = {"path": str(p), "exists": exists, "size": size, "nonempty": nonempty}
        if nonempty:
            logger.info("OK %-18s %s (%d bytes)", name, p, size)
        else:
            logger.error("MISSING/EMPTY %-18s %s", name, p)
            ok = False

    out = Path(cfg["logs_dir"]) / "weights_check_report.json"
    write_json(out, report)
    logger.info("Wrote report: %s", out)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
