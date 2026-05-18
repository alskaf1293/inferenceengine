#!/usr/bin/env python3
"""Sweep deterministic RTL pc_network_nlayer traces against PCNetNLayer."""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = "/home/gregoryv/miniconda3/envs/dsl2/bin/python"
ACT_ID = {"linear": 0, "relu": 1}


@dataclass(frozen=True)
class Case:
    name: str
    act_lut: tuple[str, str, str]
    alpha: float
    gamma: float
    tol: float = 5e-6


CASES = [
    Case("lin_relu_lin_a005_g010", ("linear", "relu", "linear"), 0.05, 0.10),
    Case("lin_relu_lin_a002_g005", ("linear", "relu", "linear"), 0.02, 0.05),
    Case("lin_relu_lin_a010_g020", ("linear", "relu", "linear"), 0.10, 0.20),
    Case("lin_lin_lin_a005_g010", ("linear", "linear", "linear"), 0.05, 0.10),
    Case("relu_relu_lin_a005_g010", ("relu", "relu", "linear"), 0.05, 0.10),
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
    out_dir = ROOT / "runs" / "rtl_network_trace_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / "runs" / "rtl_network_trace_sweep_summary.csv"

    rows = []
    for case in CASES:
        csv_path = out_dir / f"{case.name}.csv"
        act0, act1, act2 = case.act_lut
        verilator_args = [
            "./scripts/run_test.sh",
            "tb/tb_network_tick_trace.sv",
            "tb_network_tick_trace",
            f"-GACT0_ID={ACT_ID[act0]}",
            f"-GACT1_ID={ACT_ID[act1]}",
            f"-GACT2_ID={ACT_ID[act2]}",
            "--",
            f"+CSV={csv_path}",
            f"+ALPHA={case.alpha}",
            f"+GAMMA={case.gamma}",
        ]
        run(verilator_args)

        check_args = [
            PY,
            "scripts/check_network_tick_trace.py",
            "--csv",
            str(csv_path),
            "--tol",
            str(case.tol),
            "--act-lut",
            *case.act_lut,
            "--alpha",
            str(case.alpha),
            "--gamma",
            str(case.gamma),
        ]
        result = run(check_args)
        max_abs = ""
        for line in result.stdout.splitlines():
            if line.startswith("rows="):
                max_abs = line.split("max_abs_error=", 1)[1].split()[0]
                break

        rows.append({
            "case": case.name,
            "act_lut": "/".join(case.act_lut),
            "alpha": case.alpha,
            "gamma": case.gamma,
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
