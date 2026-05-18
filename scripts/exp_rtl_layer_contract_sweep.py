#!/usr/bin/env python3
"""Composable RTL contract sweep for pc_layer tiles."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = "/home/gregoryv/miniconda3/envs/dsl2/bin/python"


@dataclass(frozen=True)
class Case:
    name: str
    k: int
    n: int
    m: int
    act: str = "linear"


CASES = [
    Case("tile_k2_fanin_256", 2, 256, 1),
    Case("tile_k4_fanin_256", 4, 256, 1),
    Case("tile_k8_fanin_256", 8, 256, 1),
    Case("tile_k2_backflow_256", 2, 1, 256),
    Case("tile_k4_backflow_256", 4, 1, 256),
    Case("tile_k4_balanced_64", 4, 64, 64),
    Case("tile_k4_relu_fanin_256", 4, 256, 1, act="relu"),
]

ACT_ID = {"linear": 0, "relu": 1, "tanh": 2, "sigmoid": 3}


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


def parse_max_abs(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("rows="):
            return line.split("max_abs_error=", 1)[1].split()[0]
    return ""


def run_case(case: Case, out_dir: Path, tol: float, reuse_existing: bool) -> dict[str, object]:
    csv_path = out_dir / f"{case.name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["VERILATOR_MDIR"] = f"obj_dir/layer_contract_{case.name}"
    env["VERILATOR_REUSE_BINARY"] = "1"
    env.setdefault("VERILATOR_BUILD_JOBS", "1")
    env.setdefault("VERILATOR_VERILATE_JOBS", "1")
    (ROOT / env["VERILATOR_MDIR"]).mkdir(parents=True, exist_ok=True)

    if not (reuse_existing and csv_path.exists()):
        run([
            "./scripts/run_test.sh",
            "tb/tb_layer_contract_trace.sv",
            "tb_layer_contract_trace",
            f"-GK={case.k}",
            f"-GN={case.n}",
            f"-GM={case.m}",
            f"-GACT_ID={ACT_ID[case.act]}",
            "--",
            f"+CSV={csv_path}",
        ], env=env)

    result = run([
        PY,
        "scripts/check_layer_contract_trace.py",
        "--csv",
        str(csv_path),
        "--k",
        str(case.k),
        "--n",
        str(case.n),
        "--m",
        str(case.m),
        "--act",
        case.act,
        "--tol",
        str(tol),
    ])
    max_abs = parse_max_abs(result.stdout)
    print(f"PASS {case.name} max_abs_error={max_abs}", flush=True)
    return {
        "case": case.name,
        "k": case.k,
        "n": case.n,
        "m": case.m,
        "act": case.act,
        "max_abs_error": max_abs,
        "tol": tol,
        "csv": str(csv_path.relative_to(ROOT)),
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(case.name for case in CASES))
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    selected = [name.strip() for name in args.cases.split(",") if name.strip()]
    case_by_name = {case.name: case for case in CASES}
    unknown = sorted(set(selected) - set(case_by_name))
    if unknown:
        raise ValueError(f"unknown cases: {unknown}")

    out_dir = ROOT / "runs" / "rtl_layer_contract_sweep"
    rows = [run_case(case_by_name[name], out_dir, args.tol, args.reuse_existing) for name in selected]
    summary_path = ROOT / "runs" / "rtl_layer_contract_sweep_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
