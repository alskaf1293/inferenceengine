#!/usr/bin/env python3
"""Run a Yosys full-fabric elaboration check for 1->256->256->23."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default="scripts/synth_pcnet_256_yosys.ys")
    parser.add_argument("--log", default="runs/synth_pcnet_256_yosys.log")
    args = parser.parse_args()

    log_path = ROOT / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["yosys", "-s", args.script],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(result.stdout)
    print(result.stdout[-4000:], end="")
    if result.returncode != 0:
        print(f"\nFAIL yosys returned {result.returncode}; full log: {log_path.relative_to(ROOT)}")
        raise SystemExit(result.returncode)
    print(f"\nPASS full 1->256->256->23 fabric elaborates in Yosys; log: {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
