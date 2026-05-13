#!/usr/bin/env python3
"""CartPole PC bridge comparison report.

This script collects the Millidge-style CartPole bridge runs into one
reproducible artifact:

  - bp/bp reference
  - pc/bp CUDA bridge
  - pc/pc CUDA bridge

Use --run-missing to launch any absent CSVs before summarizing.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS = (1, 7, 21, 42, 84)
EPISODES = 2000


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    filename_template: str
    args: tuple[str, ...]

    def path(self, runs_dir: Path, seed: int) -> Path:
        return runs_dir / self.filename_template.format(seed=seed)


RUNS = (
    RunSpec(
        key="bp_bp",
        label="BP value / BP policy",
        filename_template="sweep_bp_bp_seed{seed}_2000.csv",
        args=("--value-backend", "bp", "--policy-backend", "bp"),
    ),
    RunSpec(
        key="pc_bp_fast",
        label="PC value / BP policy",
        filename_template="sweep_pc_bp_nudgefast_seed{seed}_2000.csv",
        args=(
            "--value-backend", "pc",
            "--policy-backend", "bp",
            "--pc-gradient-mode", "pc_nudge_gated_fast",
        ),
    ),
    RunSpec(
        key="pc_pc_fast",
        label="PC value / PC policy",
        filename_template="sweep_pc_pc_fast_seed{seed}_2000.csv",
        args=(
            "--value-backend", "pc",
            "--policy-backend", "pc",
            "--pc-gradient-mode", "pc_nudge_gated_fast",
            "--pc-policy-gradient-mode", "fast",
        ),
    ),
)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    acc = 0.0
    for idx, value in enumerate(values):
        acc += value
        if idx >= window:
            acc -= values[idx - window]
        out[idx] = acc / min(idx + 1, window)
    return out


def run_missing(args: argparse.Namespace, specs: tuple[RunSpec, ...]) -> None:
    script = Path("python_rtl/run_cartpole_millidge_hybrid.py")
    for spec in specs:
        for seed in args.seeds:
            out_csv = spec.path(args.runs_dir, seed)
            if out_csv.exists() and out_csv.stat().st_size > 0 and not args.force:
                continue
            cmd = [
                sys.executable,
                str(script),
                "--episodes", str(args.episodes),
                "--seed", str(seed),
                "--device", args.device,
                "--out-csv", str(out_csv),
                "--infotime", str(args.infotime),
                *spec.args,
            ]
            print("Running", spec.key, "seed", seed)
            subprocess.run(cmd, check=True)


def summarize(args: argparse.Namespace, specs: tuple[RunSpec, ...]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        for seed in args.seeds:
            path = spec.path(args.runs_dir, seed)
            if not path.exists() or path.stat().st_size == 0:
                print(f"Missing {path}")
                continue
            df = pd.read_csv(path)
            rewards = df["reward"].to_numpy(dtype=np.float64)
            avg50 = float(np.mean(rewards[-50:]))
            avg100 = float(np.mean(rewards[-100:]))
            first_solved = ""
            roll100 = rolling_mean(rewards, 100)
            solved = np.where(roll100 >= 195.0)[0]
            if len(solved):
                first_solved = int(solved[0] + 1)
            rows.append({
                "run_key": spec.key,
                "run_label": spec.label,
                "seed": seed,
                "episodes": len(rewards),
                "final_avg50": avg50,
                "final_avg100": avg100,
                "best": int(np.max(rewards)),
                "first_ep_avg100_ge_195": first_solved,
                "csv": str(path),
            })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.runs_dir / "cartpole_bridge_summary.csv"
    summary.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {summary_path}")

    aggregate = (
        summary.groupby(["run_key", "run_label"], sort=False)
        .agg(
            seeds=("seed", "count"),
            mean_avg50=("final_avg50", "mean"),
            std_avg50=("final_avg50", "std"),
            mean_avg100=("final_avg100", "mean"),
            mean_best=("best", "mean"),
        )
        .reset_index()
    )
    aggregate_path = args.runs_dir / "cartpole_bridge_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {aggregate_path}")
    print(aggregate.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return summary


def plot_curves(args: argparse.Namespace, specs: tuple[RunSpec, ...]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    colors = {
        "bp_bp": "#333333",
        "pc_bp_fast": "#0072B2",
        "pc_pc_fast": "#D55E00",
    }

    for spec in specs:
        curves = []
        episode_axis = None
        for seed in args.seeds:
            path = spec.path(args.runs_dir, seed)
            if not path.exists() or path.stat().st_size == 0:
                continue
            df = pd.read_csv(path)
            rewards = df["reward"].to_numpy(dtype=np.float64)
            curves.append(rolling_mean(rewards, args.window))
            episode_axis = df["episode"].to_numpy(dtype=np.int64)
        if not curves:
            continue
        stacked = np.vstack(curves)
        mean = stacked.mean(axis=0)
        stderr = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0]) if stacked.shape[0] > 1 else np.zeros_like(mean)
        color = colors.get(spec.key)
        ax.plot(episode_axis, mean, label=spec.label, linewidth=2.2, color=color)
        ax.fill_between(episode_axis, mean - stderr, mean + stderr, color=color, alpha=0.16, linewidth=0)

    ax.axhline(195, color="#777777", linestyle="--", linewidth=1.2, label="CartPole solved threshold")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward, rolling mean ({args.window})")
    ax.set_title("CartPole Active-Inference PC Bridge")
    ax.grid(True, linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    out_path = args.out_dir / "cartpole_bridge_learning_curves.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("python_runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--infotime", type=int, default=500)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--run-missing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", choices=[spec.key for spec in RUNS], nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = tuple(spec for spec in RUNS if args.only is None or spec.key in args.only)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    if args.run_missing:
        run_missing(args, specs)
    summarize(args, specs)
    plot_curves(args, specs)


if __name__ == "__main__":
    main()
