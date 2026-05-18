#!/usr/bin/env python3
"""Sweep fixed-dataset supervised RTL learning curves against PCNetNLayer."""

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
    epochs: int
    infer_ticks: int
    learn_ticks: int
    eval_ticks: int
    alpha: float
    gamma: float
    tol: float = 5e-5


CASES = [
    Case("base_e3_i10_l2_v20_a005_g010", 3, 10, 2, 20, 0.05, 0.10),
    Case("short_e3_i5_l1_v10_a002_g005", 3, 5, 1, 10, 0.02, 0.05),
    Case("strong_e3_i10_l2_v20_a010_g020", 3, 10, 2, 20, 0.10, 0.20),
    Case("longer_e5_i15_l3_v25_a005_g010", 5, 15, 3, 25, 0.05, 0.10),
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
    out_dir = ROOT / "runs" / "rtl_supervised_fixed_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / "runs" / "rtl_supervised_fixed_sweep_summary.csv"

    rows = []
    for case in CASES:
        csv_path = out_dir / f"{case.name}.csv"
        run([
            "./scripts/run_test.sh",
            "tb/tb_supervised_fixed_trace.sv",
            "tb_supervised_fixed_trace",
            "--",
            f"+CSV={csv_path}",
            f"+EPOCHS={case.epochs}",
            f"+INFER_TICKS={case.infer_ticks}",
            f"+LEARN_TICKS={case.learn_ticks}",
            f"+EVAL_TICKS={case.eval_ticks}",
            f"+ALPHA={case.alpha}",
            f"+GAMMA={case.gamma}",
        ])

        result = run([
            PY,
            "scripts/check_supervised_fixed_trace.py",
            "--csv",
            str(csv_path),
            "--tol",
            str(case.tol),
            "--epochs",
            str(case.epochs),
            "--infer-ticks",
            str(case.infer_ticks),
            "--learn-ticks",
            str(case.learn_ticks),
            "--eval-ticks",
            str(case.eval_ticks),
            "--alpha",
            str(case.alpha),
            "--gamma",
            str(case.gamma),
        ])

        max_abs = ""
        for line in result.stdout.splitlines():
            if line.startswith("rows="):
                max_abs = line.split("max_abs_error=", 1)[1].split()[0]
                break
        final_mse = ""
        with csv_path.open(newline="") as f:
            curve = list(csv.DictReader(f))
            if curve:
                final_mse = curve[-1]["mse"]

        rows.append({
            "case": case.name,
            "epochs": case.epochs,
            "infer_ticks": case.infer_ticks,
            "learn_ticks": case.learn_ticks,
            "eval_ticks": case.eval_ticks,
            "alpha": case.alpha,
            "gamma": case.gamma,
            "max_abs_error": max_abs,
            "final_mse": final_mse,
            "status": "pass",
            "csv": str(csv_path.relative_to(ROOT)),
        })
        print(f"PASS {case.name} max_abs_error={max_abs} final_mse={final_mse}")

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
