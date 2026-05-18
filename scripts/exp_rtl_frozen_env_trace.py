#!/usr/bin/env python3
"""Frozen CartPole/Pendulum environment-loop traces for RTL vs PCNetNLayer."""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - local env may use old gym
    import gym


ROOT = Path(__file__).resolve().parents[1]
PY = "/home/gregoryv/miniconda3/envs/dsl2/bin/python"


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    k0: int
    k1: int
    k2: int
    samples: int
    seed: int
    epochs: int
    infer_ticks: int
    learn_ticks: int
    eval_ticks: int
    alpha: float
    gamma: float
    tol: float = 5e-5


CASES = [
    Case("cartpole_value_frozen_seed42", "cartpole_value", 2, 16, 4, 24, 42, 2, 10, 2, 16, 0.03, 0.06),
    Case("pendulum_actor_frozen_seed42", "pendulum_actor", 1, 16, 3, 24, 42, 2, 10, 2, 16, 0.03, 0.06),
    Case("pendulum_critic_frozen_seed42", "pendulum_critic", 1, 16, 4, 24, 42, 2, 10, 2, 16, 0.03, 0.06),
]


def reset_env(env, seed: int):
    out = env.reset(seed=seed)
    return out[0] if isinstance(out, tuple) else out


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = out
    return obs, reward, bool(done), info


def collect_cartpole_value(samples: int, seed: int) -> np.ndarray:
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    obs = np.asarray(reset_env(env, seed), dtype=np.float64)
    rows = []
    while len(rows) < samples:
        action = int(rng.integers(0, 2))
        next_obs, reward, done, _ = step_env(env, action)
        target = np.zeros(2, dtype=np.float64)
        target[action] = float(reward)
        rows.append(np.concatenate([obs.astype(np.float64), target]))
        obs = np.asarray(reset_env(env, seed + len(rows)) if done else next_obs, dtype=np.float64)
    env.close()
    return np.asarray(rows, dtype=np.float64)


def collect_pendulum(samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    env = gym.make("Pendulum-v1")
    rng = np.random.default_rng(seed)
    obs = np.asarray(reset_env(env, seed), dtype=np.float64)
    actor_rows = []
    critic_rows = []
    while len(actor_rows) < samples:
        action = rng.uniform(-2.0, 2.0, size=(1,)).astype(np.float64)
        next_obs, reward, done, _ = step_env(env, action.astype(np.float32))

        # Actor targets are normalized actions. Critic targets are one-step
        # reward targets scaled into a hardware-friendly range.
        actor_rows.append(np.concatenate([obs.astype(np.float64), action / 2.0]))
        critic_rows.append(np.concatenate([obs.astype(np.float64), action, [float(reward) / 16.0]]))
        obs = np.asarray(reset_env(env, seed + len(actor_rows)) if done else next_obs, dtype=np.float64)
    env.close()
    return np.asarray(actor_rows, dtype=np.float64), np.asarray(critic_rows, dtype=np.float64)


def write_datasets(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    cartpole = collect_cartpole_value(samples=24, seed=42)
    actor, critic = collect_pendulum(samples=24, seed=42)
    datasets = {
        "cartpole_value_frozen_seed42": cartpole,
        "pendulum_actor_frozen_seed42": actor,
        "pendulum_critic_frozen_seed42": critic,
    }
    for name, data in datasets.items():
        path = out_dir / f"{name}.dat"
        np.savetxt(path, data, fmt="%.9g")
        paths[name] = path
    return paths


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def parse_max_abs(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("rows="):
            return line.split("max_abs_error=", 1)[1].split()[0]
    return ""


def final_mse(csv_path: Path) -> str:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1]["mse"] if rows else ""


def main() -> None:
    out_dir = ROOT / "runs" / "rtl_frozen_env_trace"
    data_dir = out_dir / "data"
    data_paths = write_datasets(data_dir)
    summary_path = ROOT / "runs" / "rtl_frozen_env_trace_summary.csv"

    rows = []
    for case in CASES:
        data_path = data_paths[case.name]
        csv_path = out_dir / f"{case.name}.csv"
        run([
            "./scripts/run_test.sh",
            "tb/tb_supervised_file_trace.sv",
            "tb_supervised_file_trace",
            f"-GK0={case.k0}",
            f"-GK1={case.k1}",
            f"-GK2={case.k2}",
            f"-GNUM_SAMPLES={case.samples}",
            "--",
            f"+DATA={data_path}",
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
            "scripts/check_supervised_file_trace.py",
            "--csv",
            str(csv_path),
            "--data",
            str(data_path),
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

        max_abs = parse_max_abs(result.stdout)
        mse = final_mse(csv_path)
        rows.append({
            "case": case.name,
            "kind": case.kind,
            "k0": case.k0,
            "k1": case.k1,
            "k2": case.k2,
            "samples": case.samples,
            "seed": case.seed,
            "epochs": case.epochs,
            "infer_ticks": case.infer_ticks,
            "learn_ticks": case.learn_ticks,
            "eval_ticks": case.eval_ticks,
            "alpha": case.alpha,
            "gamma": case.gamma,
            "max_abs_error": max_abs,
            "final_mse": mse,
            "status": "pass",
            "data": str(data_path.relative_to(ROOT)),
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
