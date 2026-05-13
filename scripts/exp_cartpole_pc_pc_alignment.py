#!/usr/bin/env python3
"""Short CartPole PC/PC alignment report.

This compares three value/policy PC implementations on the same seeds:

  - CUDA fast bridge
  - Torch/CUDA tick-faithful path
  - NumPy tick-faithful path

Use --run-missing to launch absent CSVs before summarizing.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    filename_template: str
    args: tuple[str, ...]

    def path(self, runs_dir: Path, seed: int, episodes: int) -> Path:
        return runs_dir / self.filename_template.format(seed=seed, episodes=episodes)


RUNS = (
    RunSpec(
        key="pc_pc_fast",
        label="PC/PC CUDA fast bridge",
        filename_template="align_pc_pc_fast_seed{seed}_{episodes}.csv",
        args=(
            "--value-backend", "pc",
            "--policy-backend", "pc",
            "--pc-gradient-mode", "pc_nudge_gated_fast",
            "--pc-policy-gradient-mode", "fast",
            "--device", "cuda",
        ),
    ),
    RunSpec(
        key="pc_pc_torch_tick",
        label="PC/PC Torch tick-faithful",
        filename_template="align_pc_pc_torchtick_seed{seed}_{episodes}.csv",
        args=(
            "--value-backend", "pc",
            "--policy-backend", "pc",
            "--pc-gradient-mode", "pc_nudge_gated_torch_tick",
            "--pc-policy-gradient-mode", "torch_tick",
            "--device", "cuda",
        ),
    ),
    RunSpec(
        key="pc_pc_torch_tick_gated",
        label="PC/PC Torch tick-faithful gated policy",
        filename_template="align_pc_pc_torchtick_gated_seed{seed}_{episodes}.csv",
        args=(
            "--value-backend", "pc",
            "--policy-backend", "pc",
            "--pc-gradient-mode", "pc_nudge_gated_torch_tick",
            "--pc-policy-gradient-mode", "torch_tick_gated",
            "--device", "cuda",
        ),
    ),
    RunSpec(
        key="pc_pc_numpy_tick",
        label="PC/PC NumPy tick-faithful",
        filename_template="align_pc_pc_numpytick_seed{seed}_{episodes}.csv",
        args=(
            "--value-backend", "pc",
            "--policy-backend", "pc",
            "--pc-gradient-mode", "pc_nudge_gated",
            "--pc-policy-gradient-mode", "pc",
            "--pc-nudge-beta", "0.001",
            "--pc-infer", "300",
            "--max-infer-ticks", "300",
            "--no-adaptive-inference",
            "--device", "cpu",
        ),
    ),
)

RUN_BY_KEY = {run.key: run for run in RUNS}


def selected_runs(args: argparse.Namespace) -> tuple[RunSpec, ...]:
    unknown = sorted(set(args.modes) - set(RUN_BY_KEY))
    if unknown:
        raise ValueError(f"Unknown mode(s): {', '.join(unknown)}")
    return tuple(RUN_BY_KEY[key] for key in args.modes)


def run_missing(args: argparse.Namespace) -> None:
    script = ROOT / "python_rtl" / "run_cartpole_millidge_hybrid.py"
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    for spec in selected_runs(args):
        for seed in args.seeds:
            out_csv = spec.path(args.runs_dir, seed, args.episodes)
            if out_csv.exists() and out_csv.stat().st_size > 0 and not args.force:
                continue
            cmd = [
                sys.executable,
                str(script),
                "--episodes", str(args.episodes),
                "--seed", str(seed),
                "--out-csv", str(out_csv),
                "--infotime", str(args.infotime or args.episodes),
                *spec.args,
            ]
            print(f"Running {spec.key} seed {seed}")
            subprocess.run(cmd, check=True)


def summarize(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for spec in selected_runs(args):
        for seed in args.seeds:
            path = spec.path(args.runs_dir, seed, args.episodes)
            if not path.exists() or path.stat().st_size == 0:
                print(f"Missing {path}")
                continue
            df = pd.read_csv(path)
            rewards = df["reward"].to_numpy(dtype=np.float64)
            rows.append({
                "mode": spec.key,
                "label": spec.label,
                "seed": seed,
                "episodes": int(len(rewards)),
                "avg_reward": float(np.mean(rewards)),
                "final_avg50": float(np.mean(rewards[-50:])),
                "best": int(np.max(rewards)),
                "csv": str(path),
            })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary, pd.DataFrame()

    aggregate = (
        summary.groupby(["mode", "label"], sort=False)
        .agg(
            seeds=("seed", "count"),
            mean_avg_reward=("avg_reward", "mean"),
            std_avg_reward=("avg_reward", "std"),
            mean_final_avg50=("final_avg50", "mean"),
            std_final_avg50=("final_avg50", "std"),
            mean_best=("best", "mean"),
            max_best=("best", "max"),
        )
        .reset_index()
    )
    return summary, aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "python_runs")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--infotime", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 21, 42, 84])
    parser.add_argument("--modes", nargs="+", default=[run.key for run in RUNS],
                        choices=[run.key for run in RUNS])
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--run-missing", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_missing:
        run_missing(args)
    summary, aggregate = summarize(args)
    if summary.empty:
        print("No alignment CSVs found. Re-run with --run-missing.")
        return

    suffix = f"_{args.tag}" if args.tag else ""
    summary_path = args.runs_dir / f"pc_pc_alignment_summary_{args.episodes}{suffix}.csv"
    aggregate_path = args.runs_dir / f"pc_pc_alignment_aggregate_{args.episodes}{suffix}.csv"
    summary.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    aggregate.to_csv(aggregate_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {summary_path}")
    print(f"Wrote {aggregate_path}")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
