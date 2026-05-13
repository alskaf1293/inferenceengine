#!/usr/bin/env python3
"""Multi-seed TD3-AIF PC/PC InvertedPendulum reproduction sweep.

Runs PC/PC exact-local TD3-AIF across seeds 1, 7, 21, 42, 84 and reports
final-window and best-checkpoint eval. Use --run-missing to launch absent runs
before summarizing.

Winning seed-42 command shape (from CHAT_HANDOFF):
  --actor-backend pc --critic-backend pc --critic-semantics aif
  --pc-critic-value-scale 100 --pc-query 100
  --episodes 600 --start-steps 1000 --update-after 1000 --batch-size 100
  --lr-actor 0.0003 --lr-critic 0.0003 --freeze-actor-after-eval 950
  --device cuda
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

    def csv_path(self, runs_dir: Path, seed: int, episodes: int) -> Path:
        return runs_dir / f"mujoco_td3aif_{self.key}_seed{seed}_{episodes}ep.csv"


CONFIGS: tuple[RunSpec, ...] = (
    RunSpec(
        key="bpbp",
        label="BP actor / BP critic (AIF)",
        extra_args=(
            "--actor-backend", "bp",
            "--critic-backend", "bp",
            "--critic-semantics", "aif",
        ),
    ),
    RunSpec(
        key="pcpc",
        label="PC actor / PC critic (AIF exact-local 2-layer)",
        extra_args=(
            "--actor-backend", "pc",
            "--critic-backend", "pc",
            "--critic-semantics", "aif",
            "--pc-critic-value-scale", "100",
            "--pc-query", "100",
            "--critic-drift-every", "10",
        ),
    ),
    RunSpec(
        key="pcpc_freeze950",
        label="PC actor / PC critic (AIF exact-local 2-layer + freeze@950)",
        extra_args=(
            "--actor-backend", "pc",
            "--critic-backend", "pc",
            "--critic-semantics", "aif",
            "--pc-critic-value-scale", "100",
            "--pc-query", "100",
            "--freeze-actor-after-eval", "950",
            "--critic-drift-every", "10",
        ),
    ),
)


def run_missing(args: argparse.Namespace, configs: tuple[RunSpec, ...]) -> None:
    for spec in configs:
        for seed in args.seeds:
            out_csv = spec.csv_path(RUNS_DIR, seed, args.episodes)
            if out_csv.exists() and out_csv.stat().st_size > 0 and not args.force:
                print(f"  skip existing: {out_csv}", flush=True)
                continue
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--env", args.env,
                "--episodes", str(args.episodes),
                "--seed", str(seed),
                "--start-steps", str(args.start_steps),
                "--update-after", str(args.update_after),
                "--batch-size", str(args.batch_size),
                "--lr-actor", str(args.lr_actor),
                "--lr-critic", str(args.lr_critic),
                "--eval-every", "10",
                "--eval-episodes", "5",
                "--final-eval-episodes", "10",
                "--device", args.device,
                "--out-csv", str(out_csv),
                "--infotime", str(args.infotime),
                *spec.extra_args,
            ]
            print(f"\nRunning: {spec.label} seed={seed} ...", flush=True)
            print("  " + " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


def read_csv_metrics(path: Path) -> dict:
    """Extract metrics from a per-seed CSV."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}
    rewards = [float(r["reward"]) for r in rows]
    evals = [float(r["eval_reward"]) for r in rows if r.get("eval_reward", "") not in ("", None)]
    best_evals = [float(r["best_eval"]) for r in rows if r.get("best_eval", "") not in ("", None)]
    cosines = [float(r["grad_cosine"]) for r in rows if r.get("grad_cosine", "") not in ("", None)]
    final_avg10 = float(np.mean(rewards[-10:])) if rewards else float("nan")
    final_eval = evals[-1] if evals else float("nan")
    best_eval = best_evals[-1] if best_evals else float("nan")
    solved_train = sum(1 for r in rewards if r >= 999.0)
    # Track cosine at start, mid, and end of training to see if equivalence degrades
    min_cosine = float(np.min(cosines)) if cosines else float("nan")
    mean_cosine = float(np.mean(cosines)) if cosines else float("nan")
    final_cosine = cosines[-1] if cosines else float("nan")
    return {
        "final_avg10": final_avg10,
        "final_eval": final_eval,
        "best_eval": best_eval,
        "solved_train_episodes": solved_train,
        "best_eval_solved": best_eval >= 999.0,
        "final_eval_solved": final_eval >= 999.0,
        "min_grad_cosine": min_cosine,
        "mean_grad_cosine": mean_cosine,
        "final_grad_cosine": final_cosine,
    }


def summarize(args: argparse.Namespace, configs: tuple[RunSpec, ...]) -> None:
    summary_path = RUNS_DIR / f"mujoco_td3aif_multiseed_summary_{args.episodes}ep.csv"
    aggregate_path = RUNS_DIR / f"mujoco_td3aif_multiseed_aggregate_{args.episodes}ep.csv"

    summary_rows = []
    for spec in configs:
        for seed in args.seeds:
            csv_path = spec.csv_path(RUNS_DIR, seed, args.episodes)
            if not csv_path.exists():
                print(f"  missing: {csv_path}")
                continue
            m = read_csv_metrics(csv_path)
            summary_rows.append({
                "config": spec.key,
                "label": spec.label,
                "seed": seed,
                "csv": str(csv_path),
                **m,
            })

    with open(summary_path, "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"\nSummary: {summary_path}")

    agg_rows = []
    for spec in configs:
        rows = [r for r in summary_rows if r["config"] == spec.key]
        if not rows:
            continue
        final_avg10s = [r["final_avg10"] for r in rows if not np.isnan(r["final_avg10"])]
        final_evals = [r["final_eval"] for r in rows if not np.isnan(r["final_eval"])]
        best_evals = [r["best_eval"] for r in rows if not np.isnan(r["best_eval"])]
        solved_train_total = sum(r["solved_train_episodes"] for r in rows)
        best_eval_solved_rate = sum(1 for r in rows if r["best_eval_solved"]) / len(rows)
        final_eval_solved_rate = sum(1 for r in rows if r["final_eval_solved"]) / len(rows)
        agg_rows.append({
            "config": spec.key,
            "label": spec.label,
            "n_seeds": len(rows),
            "mean_final_avg10": float(np.mean(final_avg10s)) if final_avg10s else float("nan"),
            "std_final_avg10": float(np.std(final_avg10s)) if final_avg10s else float("nan"),
            "mean_final_eval": float(np.mean(final_evals)) if final_evals else float("nan"),
            "mean_best_eval": float(np.mean(best_evals)) if best_evals else float("nan"),
            "solved_train_total": solved_train_total,
            "best_eval_solved_rate": best_eval_solved_rate,
            "final_eval_solved_rate": final_eval_solved_rate,
        })
        print(f"\n{spec.label} ({len(rows)} seeds):")
        print(f"  mean final avg10  = {agg_rows[-1]['mean_final_avg10']:.2f}")
        print(f"  mean final eval   = {agg_rows[-1]['mean_final_eval']:.2f}")
        print(f"  mean best eval    = {agg_rows[-1]['mean_best_eval']:.2f}")
        print(f"  best-eval solved  = {sum(1 for r in rows if r['best_eval_solved'])}/{len(rows)}")
        print(f"  final-eval solved = {sum(1 for r in rows if r['final_eval_solved'])}/{len(rows)}")

    with open(aggregate_path, "w", newline="") as f:
        if agg_rows:
            writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            writer.writeheader()
            writer.writerows(agg_rows)
    print(f"\nAggregate: {aggregate_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="InvertedPendulum-v5")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Subset of config keys to run (default: all)")
    parser.add_argument("--start-steps", type=int, default=1000)
    parser.add_argument("--update-after", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-missing", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if CSV exists")
    parser.add_argument("--infotime", type=int, default=50)
    args = parser.parse_args()

    selected = tuple(c for c in CONFIGS if args.configs is None or c.key in args.configs)
    if not selected:
        print("No matching configs. Available:", [c.key for c in CONFIGS])
        sys.exit(1)

    RUNS_DIR.mkdir(exist_ok=True)
    if args.run_missing:
        run_missing(args, selected)
    summarize(args, selected)


if __name__ == "__main__":
    main()
