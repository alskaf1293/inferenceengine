#!/usr/bin/env python3
"""HalfCheetah TD3 -> BP-AIF -> PC/PC AIF replication sweep.

This is the HalfCheetah analogue of the Reacher sweep:

1. Conventional homebrew TD3 (`q`, BP actor/BP critic)
2. Millidge-style BP-AIF TD3 (`aif`, BP actor/BP critic, G = -Q)
3. PC/PC AIF with BP-equivalent PC critic updates

HalfCheetah is higher-dimensional and normally needs a larger timestep budget
than InvertedPendulum/Reacher, so the default here is 300k steps.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SEEDS = (1, 7, 21, 42, 84)
RUNS_DIR = Path("python_runs")
SCRIPT = Path("python_rtl/run_mujoco_td3.py")


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    extra_args: tuple[str, ...]

    def csv_path(self, runs_dir: Path, seed: int, timesteps: int) -> Path:
        return runs_dir / f"halfcheetah_td3_{self.key}_seed{seed}_{timesteps}.csv"


CONFIGS: tuple[RunSpec, ...] = (
    RunSpec(
        key="bp_q",
        label="BP actor / BP critic (TD3 Q)",
        extra_args=(
            "--actor-backend", "bp",
            "--critic-backend", "bp",
            "--critic-semantics", "q",
        ),
    ),
    RunSpec(
        key="bp_aif",
        label="BP actor / BP critic (AIF, G=-Q)",
        extra_args=(
            "--actor-backend", "bp",
            "--critic-backend", "bp",
            "--critic-semantics", "aif",
        ),
    ),
    RunSpec(
        key="pcpc_aif_bpequiv",
        label="PC actor / PC critic (AIF, BP-equivalent PC update)",
        extra_args=(
            "--actor-backend", "pc",
            "--critic-backend", "pc",
            "--critic-semantics", "aif",
            "--pc-critic-gradient-mode", "bp_equiv",
            "--pc-critic-value-scale", "1",
        ),
    ),
)


def selected_configs(keys: list[str] | None) -> tuple[RunSpec, ...]:
    configs = tuple(c for c in CONFIGS if keys is None or c.key in keys)
    if not configs:
        raise SystemExit(f"No matching configs. Available: {[c.key for c in CONFIGS]}")
    return configs


def run_missing(args: argparse.Namespace, configs: tuple[RunSpec, ...]) -> None:
    RUNS_DIR.mkdir(exist_ok=True)
    for spec in configs:
        for seed in args.seeds:
            out_csv = spec.csv_path(RUNS_DIR, seed, args.total_timesteps)
            if out_csv.exists() and out_csv.stat().st_size > 0 and not args.force:
                print(f"skip existing: {out_csv}", flush=True)
                continue
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--env", args.env,
                "--total-timesteps", str(args.total_timesteps),
                "--seed", str(seed),
                "--hidden", str(args.hidden),
                "--start-steps", str(args.start_steps),
                "--update-after", str(args.update_after),
                "--batch-size", str(args.batch_size),
                "--lr-actor", str(args.lr_actor),
                "--lr-critic", str(args.lr_critic),
                "--exploration-noise", str(args.exploration_noise),
                "--policy-noise", str(args.policy_noise),
                "--noise-clip", str(args.noise_clip),
                "--eval-every", "0",
                "--eval-every-steps", str(args.eval_every_steps),
                "--eval-episodes", str(args.eval_episodes),
                "--final-eval-episodes", str(args.final_eval_episodes),
                "--infotime", str(args.infotime),
                "--device", args.device,
                "--out-csv", str(out_csv),
                *spec.extra_args,
            ]
            if args.print_device:
                cmd.append("--print-device")
            print(f"\nRunning {spec.key} seed={seed}", flush=True)
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


def read_csv_metrics(path: Path) -> dict[str, float | int | str]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    if not rows:
        return {}

    rewards = [float(r["reward"]) for r in rows]
    evals = [float(r["eval_reward"]) for r in rows if r.get("eval_reward")]
    best_evals = [float(r["best_eval"]) for r in rows if r.get("best_eval")]
    return {
        "episodes": len(rows),
        "final_avg10": float(np.mean(rewards[-10:])) if rewards else float("nan"),
        "final_avg50": float(np.mean(rewards[-50:])) if rewards else float("nan"),
        "best_train_reward": float(np.max(rewards)) if rewards else float("nan"),
        "final_eval": evals[-1] if evals else float("nan"),
        "best_eval": best_evals[-1] if best_evals else float("nan"),
    }


def summarize(args: argparse.Namespace, configs: tuple[RunSpec, ...]) -> None:
    RUNS_DIR.mkdir(exist_ok=True)
    summary_path = RUNS_DIR / f"halfcheetah_td3_pcpc_summary_{args.total_timesteps}.csv"
    aggregate_path = RUNS_DIR / f"halfcheetah_td3_pcpc_aggregate_{args.total_timesteps}.csv"

    summary_rows = []
    for spec in configs:
        for seed in args.seeds:
            csv_path = spec.csv_path(RUNS_DIR, seed, args.total_timesteps)
            if not csv_path.exists():
                print(f"missing: {csv_path}", flush=True)
                continue
            summary_rows.append({
                "config": spec.key,
                "label": spec.label,
                "seed": seed,
                "csv": str(csv_path),
                **read_csv_metrics(csv_path),
            })

    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"\nSummary: {summary_path}")

    aggregate_rows = []
    for spec in configs:
        rows = [r for r in summary_rows if r["config"] == spec.key]
        if not rows:
            continue
        final_evals = [r["final_eval"] for r in rows if not np.isnan(r["final_eval"])]
        best_evals = [r["best_eval"] for r in rows if not np.isnan(r["best_eval"])]
        final_avg50s = [r["final_avg50"] for r in rows if not np.isnan(r["final_avg50"])]
        aggregate_rows.append({
            "config": spec.key,
            "label": spec.label,
            "n_seeds": len(rows),
            "mean_final_avg50": float(np.mean(final_avg50s)) if final_avg50s else float("nan"),
            "mean_final_eval": float(np.mean(final_evals)) if final_evals else float("nan"),
            "mean_best_eval": float(np.mean(best_evals)) if best_evals else float("nan"),
            "std_final_eval": float(np.std(final_evals)) if final_evals else float("nan"),
        })
        print(f"\n{spec.label}:")
        print(f"  mean final avg50 = {aggregate_rows[-1]['mean_final_avg50']:.3f}")
        print(f"  mean final eval  = {aggregate_rows[-1]['mean_final_eval']:.3f}")
        print(f"  mean best eval   = {aggregate_rows[-1]['mean_best_eval']:.3f}")

    if aggregate_rows:
        with open(aggregate_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(aggregate_rows)
    print(f"\nAggregate: {aggregate_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HalfCheetah-v5")
    parser.add_argument("--total-timesteps", type=int, default=300000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Subset of: bp_q bp_aif pcpc_aif_bpequiv")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--start-steps", type=int, default=10000)
    parser.add_argument("--update-after", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--eval-every-steps", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--final-eval-episodes", type=int, default=10)
    parser.add_argument("--infotime", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--run-missing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--print-device", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = selected_configs(args.configs)
    if args.run_missing:
        run_missing(args, configs)
    summarize(args, configs)


if __name__ == "__main__":
    main()
