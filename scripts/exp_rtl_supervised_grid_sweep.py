#!/usr/bin/env python3
"""Sweep larger fixed-dataset supervised RTL learning curves."""

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
    k0: int
    k1: int
    k2: int
    samples: int
    epochs: int
    infer_ticks: int
    learn_ticks: int
    eval_ticks: int
    alpha: float
    gamma: float
    tol: float = 5e-5


CASES = [
    Case("grid_3_4_2_s8_e2_i8_l2_v12_a005_g010", 3, 4, 2, 8, 2, 8, 2, 12, 0.05, 0.10),
    Case("grid_4_8_4_s12_e2_i8_l2_v12_a004_g008", 4, 8, 4, 12, 2, 8, 2, 12, 0.04, 0.08),
    Case("grid_6_12_6_s16_e2_i10_l2_v16_a003_g006", 6, 12, 6, 16, 2, 10, 2, 16, 0.03, 0.06),
    Case("grid_8_16_8_s16_e1_i10_l2_v16_a002_g005", 8, 16, 8, 16, 1, 10, 2, 16, 0.02, 0.05),
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


def extract_max_abs(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("rows="):
            return line.split("max_abs_error=", 1)[1].split()[0]
    return ""


def final_mse(csv_path: Path) -> str:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1]["mse"] if rows else ""


def main() -> None:
    out_dir = ROOT / "runs" / "rtl_supervised_grid_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / "runs" / "rtl_supervised_grid_sweep_summary.csv"

    rows = []
    for case in CASES:
        csv_path = out_dir / f"{case.name}.csv"
        run([
            "./scripts/run_test.sh",
            "tb/tb_supervised_grid_trace.sv",
            "tb_supervised_grid_trace",
            f"-GK0={case.k0}",
            f"-GK1={case.k1}",
            f"-GK2={case.k2}",
            f"-GNUM_SAMPLES={case.samples}",
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
            "scripts/check_supervised_grid_trace.py",
            "--csv",
            str(csv_path),
            "--k0",
            str(case.k0),
            "--k1",
            str(case.k1),
            "--k2",
            str(case.k2),
            "--samples",
            str(case.samples),
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
            "--tol",
            str(case.tol),
        ])

        max_abs = extract_max_abs(result.stdout)
        mse = final_mse(csv_path)
        rows.append({
            "case": case.name,
            "k0": case.k0,
            "k1": case.k1,
            "k2": case.k2,
            "samples": case.samples,
            "epochs": case.epochs,
            "infer_ticks": case.infer_ticks,
            "learn_ticks": case.learn_ticks,
            "eval_ticks": case.eval_ticks,
            "alpha": case.alpha,
            "gamma": case.gamma,
            "max_abs_error": max_abs,
            "final_mse": mse,
            "status": "pass",
            "csv": str(csv_path.relative_to(ROOT)),
        })
        print(f"PASS {case.name} max_abs_error={max_abs} final_mse={mse}")

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
