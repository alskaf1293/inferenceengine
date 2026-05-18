#!/usr/bin/env python3
"""Run a small RTL/Python primitive dynamics sweep.

This is intentionally below the RL layer: it checks the local neuron dynamics
that the physical inference engine must implement correctly before larger
network or control experiments are meaningful.
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = "/home/gregoryv/miniconda3/envs/dsl2/bin/python"


@dataclass(frozen=True)
class Case:
    name: str
    act: str
    act_id: int
    x_set: bool = False
    clamp_hard: bool = False
    tol: float = 5e-6


CASES = [
    Case("linear_free", "linear", 0),
    Case("linear_soft_xset", "linear", 0, x_set=True),
    Case("linear_hard_xset", "linear", 0, x_set=True, clamp_hard=True),
    Case("relu_free", "relu", 1),
    Case("relu_soft_xset", "relu", 1, x_set=True),
    Case("relu_hard_xset", "relu", 1, x_set=True, clamp_hard=True),
]


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def main() -> None:
    out_dir = ROOT / "runs" / "rtl_primitive_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / "runs" / "rtl_primitive_sweep_summary.csv"

    rows = []
    for case in CASES:
        csv_path = out_dir / f"{case.name}.csv"
        verilator_args = [
            "./scripts/run_test.sh",
            "tb/tb_neuron_tick_trace.sv",
            "tb_neuron_tick_trace",
            f"-GACT_ID={case.act_id}",
            f"-GCLAMP_HARD_PARAM={1 if case.clamp_hard else 0}",
            "--",
            f"+CSV={csv_path}",
        ]
        if case.x_set:
            verilator_args.append("+XSET=1")

        run(verilator_args)

        check_args = [
            PY,
            "scripts/check_neuron_tick_trace.py",
            "--csv",
            str(csv_path),
            "--act",
            case.act,
            "--tol",
            str(case.tol),
        ]
        if case.x_set:
            check_args.append("--x-set")
        if case.clamp_hard:
            check_args.append("--clamp-hard")

        result = run(check_args)
        max_abs = ""
        for line in result.stdout.splitlines():
            if line.startswith("rows="):
                max_abs = line.split("max_abs_error=", 1)[1].split()[0]
                break

        rows.append({
            "case": case.name,
            "act": case.act,
            "x_set": int(case.x_set),
            "clamp_hard": int(case.clamp_hard),
            "tol": case.tol,
            "max_abs_error": max_abs,
            "status": "pass",
            "csv": str(csv_path.relative_to(ROOT)),
        })
        print(f"PASS {case.name} max_abs_error={max_abs}")

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
