#!/usr/bin/env python3
"""Run narrow/deep pc_network_nlayer scheduler contract traces."""

from __future__ import annotations

import argparse
import csv
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


def parse_max_abs(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("rows="):
            return line.split("max_abs_error=", 1)[1].split()[0]
    return ""


def run_h(h: int, out_dir: Path, tol: float, reuse_existing: bool) -> dict[str, object]:
    csv_path = out_dir / f"deep_scheduler_h{h}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["VERILATOR_MDIR"] = f"obj_dir/deep_scheduler_h{h}"
    env["VERILATOR_REUSE_BINARY"] = "1"
    env.setdefault("VERILATOR_BUILD_JOBS", "1")
    env.setdefault("VERILATOR_VERILATE_JOBS", "1")
    (ROOT / env["VERILATOR_MDIR"]).mkdir(parents=True, exist_ok=True)

    if not (reuse_existing and csv_path.exists()):
        run([
            "./scripts/run_test.sh",
            "tb/tb_deep_scheduler_trace.sv",
            "tb_deep_scheduler_trace",
            f"-GH={h}",
            "--",
            f"+CSV={csv_path}",
        ], env=env)

    result = run([
        PY,
        "scripts/check_deep_scheduler_trace.py",
        "--csv",
        str(csv_path),
        "--h",
        str(h),
        "--tol",
        str(tol),
    ])
    max_abs = parse_max_abs(result.stdout)
    print(f"PASS deep_scheduler_h{h} max_abs_error={max_abs}", flush=True)
    return {
        "case": f"deep_scheduler_h{h}",
        "h": h,
        "layers": 5,
        "shape": f"1->{h}->{h}->{h}->23",
        "max_abs_error": max_abs,
        "tol": tol,
        "csv": str(csv_path.relative_to(ROOT)),
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-widths", default="8,16")
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    widths = [int(v.strip()) for v in args.hidden_widths.split(",") if v.strip()]
    out_dir = ROOT / "runs" / "rtl_deep_scheduler_sweep"
    rows = [run_h(h, out_dir, args.tol, args.reuse_existing) for h in widths]
    summary_path = ROOT / "runs" / "rtl_deep_scheduler_sweep_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
