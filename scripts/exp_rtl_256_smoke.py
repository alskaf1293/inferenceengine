#!/usr/bin/env python3
"""Minimal hidden-256 RTL smoke runner."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = "/home/gregoryv/miniconda3/envs/dsl2/bin/python"


def run(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, end="")
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="runs/four_layer_256_smoke.csv")
    parser.add_argument("--mdir", default="obj_dir/four_layer_256_smoke")
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--verilator-build-jobs", type=int, default=1)
    parser.add_argument("--verilator-verilate-jobs", type=int, default=1)
    parser.add_argument("--verilator-output-split", type=int, default=50000)
    parser.add_argument("--verilator-output-split-cfuncs", type=int, default=50000)
    args = parser.parse_args()

    csv_path = ROOT / args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["VERILATOR_MDIR"] = args.mdir
    env["VERILATOR_REUSE_BINARY"] = "1"
    env["VERILATOR_BUILD_JOBS"] = str(args.verilator_build_jobs)
    env["VERILATOR_VERILATE_JOBS"] = str(args.verilator_verilate_jobs)
    env["VERILATOR_OUTPUT_SPLIT"] = str(args.verilator_output_split)
    env["VERILATOR_OUTPUT_SPLIT_CFUNCS"] = str(args.verilator_output_split_cfuncs)
    (ROOT / args.mdir).mkdir(parents=True, exist_ok=True)

    if not args.reuse_existing:
        result = run([
            "./scripts/run_test.sh",
            "tb/tb_four_layer_256_smoke.sv",
            "tb_four_layer_256_smoke",
            "--",
            f"+CSV={csv_path}",
        ], env=env)
        if result.stdout:
            print(result.stdout, end="")

    result = run([
        PY,
        "scripts/check_four_layer_256_smoke.py",
        "--csv",
        str(csv_path),
        "--tol",
        str(args.tol),
    ])
    print(result.stdout, end="")
    print(f"Saved smoke trace: {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
