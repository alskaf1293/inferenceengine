#!/usr/bin/env python3
"""Compare RTL/Python over a sequence of frozen runner-derived updates."""

from __future__ import annotations

import csv
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym

ROOT = Path(__file__).resolve().parents[1]
PY = "/home/gregoryv/miniconda3/envs/dsl2/bin/python"

sys.path.insert(0, str(ROOT / "python_rtl"))
from run_cartpole_millidge_hybrid import BPPolicyModel, BPValueModel  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    k0: int
    k1: int
    k2: int
    samples_per_update: int
    updates: int
    infer_ticks: int = 10
    learn_ticks: int = 2
    eval_ticks: int = 16
    alpha: float = 0.03
    gamma: float = 0.06
    tol: float = 5e-5


CASE = Case("cartpole_value_td_sequence_seed42", 2, 16, 4, 12, 4)


def reset_env(env, seed: int | None = None):
    out = env.reset(seed=seed) if seed is not None else env.reset()
    return out[0] if isinstance(out, tuple) else out


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = out
    return obs, reward, bool(done), info


def collect_cartpole_memory(seed: int, n_transitions: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    value_model = BPValueModel(hidden=16, lr=1e-3, discount=0.99, seed=seed, device=device)
    policy_model = BPPolicyModel(hidden=16, lr=1e-3, seed=seed + 1, device=device)

    env = gym.make("CartPole-v1")
    try:
        env.action_space.seed(seed)
    except AttributeError:
        pass
    obs = np.asarray(reset_env(env, seed), dtype=np.float64)
    memory = []
    while len(memory) < n_transitions:
        action = policy_model.sample_action(obs)
        next_obs, reward, done, _ = step_env(env, action)
        memory.append((
            obs.astype(np.float64),
            int(action),
            float(reward),
            np.asarray(next_obs, dtype=np.float64),
            bool(done),
        ))
        obs = np.asarray(reset_env(env, seed + len(memory)) if done else next_obs, dtype=np.float64)
    env.close()
    return memory, value_model, policy_model


def write_sequence_data(case: Case, out_dir: Path, seed: int = 42) -> tuple[list[Path], Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    memory, value_model, policy_model = collect_cartpole_memory(seed, n_transitions=96)
    rng = random.Random(seed + 100)
    paths = []
    manifest_path = out_dir / f"{case.name}_manifest.txt"
    with manifest_path.open("w") as manifest:
        for update_idx in range(case.updates):
            minibatch = rng.sample(memory, case.samples_per_update)
            states = np.stack([item[0] for item in minibatch]).astype(np.float32)
            next_states = np.stack([item[3] for item in minibatch]).astype(np.float32)
            actions = np.asarray([item[1] for item in minibatch], dtype=np.int64)
            rewards = np.asarray([item[2] for item in minibatch], dtype=np.float32)
            dones = np.asarray([item[4] for item in minibatch], dtype=np.bool_)

            next_policy = policy_model.probs_np(next_states)
            next_values = value_model.predict_np(next_states, target=True)
            targets = rewards + (~dones).astype(np.float32) * 0.99 * np.sum(next_policy * next_values, axis=1)
            y = value_model.predict_np(states, target=False)
            y[np.arange(case.samples_per_update), actions] = targets
            rows = np.concatenate([states.astype(np.float64), y.astype(np.float64)], axis=1)
            data_path = out_dir / f"{case.name}_update{update_idx}.dat"
            np.savetxt(data_path, rows, fmt="%.9g")
            manifest.write(f"{data_path}\n")
            paths.append(data_path)
    return paths, manifest_path


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def final_mse(csv_path: Path) -> str:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1]["mse"] if rows else ""


def parse_max_abs(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("rows="):
            return line.split("max_abs_error=", 1)[1].split()[0]
    return ""


def main() -> None:
    out_dir = ROOT / "runs" / "rtl_runner_update_sequence"
    data_paths, manifest_path = write_sequence_data(CASE, out_dir / "data")
    summary_path = ROOT / "runs" / "rtl_runner_update_sequence_summary.csv"

    rows = []
    for update_idx, data_path in enumerate(data_paths):
        csv_path = out_dir / f"{CASE.name}_update{update_idx}.csv"
        run([
            "./scripts/run_test.sh",
            "tb/tb_supervised_file_trace.sv",
            "tb_supervised_file_trace",
            f"-GK0={CASE.k0}",
            f"-GK1={CASE.k1}",
            f"-GK2={CASE.k2}",
            f"-GNUM_SAMPLES={CASE.samples_per_update}",
            "--",
            f"+DATA={data_path}",
            f"+CSV={csv_path}",
            "+EPOCHS=1",
            f"+INFER_TICKS={CASE.infer_ticks}",
            f"+LEARN_TICKS={CASE.learn_ticks}",
            f"+EVAL_TICKS={CASE.eval_ticks}",
            f"+ALPHA={CASE.alpha}",
            f"+GAMMA={CASE.gamma}",
        ])

        result = run([
            PY,
            "scripts/check_supervised_file_trace.py",
            "--csv",
            str(csv_path),
            "--data",
            str(data_path),
            "--k0",
            str(CASE.k0),
            "--k1",
            str(CASE.k1),
            "--k2",
            str(CASE.k2),
            "--samples",
            str(CASE.samples_per_update),
            "--epochs",
            "1",
            "--infer-ticks",
            str(CASE.infer_ticks),
            "--learn-ticks",
            str(CASE.learn_ticks),
            "--eval-ticks",
            str(CASE.eval_ticks),
            "--alpha",
            str(CASE.alpha),
            "--gamma",
            str(CASE.gamma),
            "--tol",
            str(CASE.tol),
        ])

        max_abs = parse_max_abs(result.stdout)
        mse = final_mse(csv_path)
        rows.append({
            "case": CASE.name,
            "update": update_idx,
            "k0": CASE.k0,
            "k1": CASE.k1,
            "k2": CASE.k2,
            "samples": CASE.samples_per_update,
            "infer_ticks": CASE.infer_ticks,
            "learn_ticks": CASE.learn_ticks,
            "eval_ticks": CASE.eval_ticks,
            "alpha": CASE.alpha,
            "gamma": CASE.gamma,
            "max_abs_error": max_abs,
            "final_mse": mse,
            "status": "pass",
            "data": str(data_path.relative_to(ROOT)),
            "csv": str(csv_path.relative_to(ROOT)),
            "manifest": str(manifest_path.relative_to(ROOT)),
        })
        print(f"PASS update={update_idx} max_abs_error={max_abs} final_mse={mse}")

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
