#!/usr/bin/env python3
"""RTL traces from exact frozen update batches produced by the RL/AIF runners."""

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
from run_cartpole_millidge_hybrid import (  # noqa: E402
    ACTION_SIZE as CARTPOLE_ACTION_SIZE,
    BPPolicyModel,
    BPValueModel,
    softmax_np,
)
from run_pendulum_ddpg import (  # noqa: E402
    ACTION_LIMIT,
    Actor,
    PCCritic,
    Transition,
    set_seed as set_pendulum_seed,
)


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    k0: int
    k1: int
    k2: int
    samples: int
    epochs: int = 2
    infer_ticks: int = 10
    learn_ticks: int = 2
    eval_ticks: int = 16
    alpha: float = 0.03
    gamma: float = 0.06
    tol: float = 5e-5


CASES = [
    Case("cartpole_value_td_batch_seed42", "cartpole_value_td", 2, 16, 4, 24),
    Case("cartpole_policy_aif_batch_seed42", "cartpole_policy_aif", 2, 16, 4, 24),
    Case("pendulum_pc_critic_td_batch_seed42", "pendulum_pc_critic_td", 1, 16, 4, 24),
]


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


def target_logits_from_values(values: np.ndarray, smoothing: float = 0.02, clip: float = 5.0) -> np.ndarray:
    greedy = int(np.argmax(values))
    probs = np.full(CARTPOLE_ACTION_SIZE, smoothing / max(1, CARTPOLE_ACTION_SIZE - 1), dtype=np.float64)
    probs[greedy] = 1.0 - smoothing
    logits = np.log(np.clip(probs, 1e-6, 1.0))
    logits -= np.mean(logits)
    return np.clip(logits, -clip, clip)


def collect_cartpole(seed: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    discount = 0.99
    hidden = 16

    value_model = BPValueModel(hidden=hidden, lr=1e-3, discount=discount, seed=seed, device=device)
    policy_model = BPPolicyModel(hidden=hidden, lr=1e-3, seed=seed + 1, device=device)

    env = gym.make("CartPole-v1")
    try:
        env.action_space.seed(seed)
    except AttributeError:
        pass
    obs = np.asarray(reset_env(env, seed), dtype=np.float64)
    memory = []
    history_states = []
    while len(memory) < max(80, samples):
        history_states.append(obs.copy())
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

    random.seed(seed + 10)
    minibatch = random.sample(memory, samples)
    states = np.stack([item[0] for item in minibatch]).astype(np.float32)
    next_states = np.stack([item[3] for item in minibatch]).astype(np.float32)
    actions = np.asarray([item[1] for item in minibatch], dtype=np.int64)
    rewards = np.asarray([item[2] for item in minibatch], dtype=np.float32)
    dones = np.asarray([item[4] for item in minibatch], dtype=np.bool_)

    next_policy = policy_model.probs_np(next_states)
    next_values = value_model.predict_np(next_states, target=True)
    targets = rewards + (~dones).astype(np.float32) * discount * np.sum(next_policy * next_values, axis=1)
    y = value_model.predict_np(states, target=False)
    y[np.arange(samples), actions] = targets
    value_rows = np.concatenate([states.astype(np.float64), y.astype(np.float64)], axis=1)

    policy_states = np.asarray(history_states[:samples], dtype=np.float32)
    values = value_model.predict_np(policy_states, target=False)
    logits = np.stack([target_logits_from_values(row) for row in values], axis=0)
    policy_rows = np.concatenate([policy_states.astype(np.float64), logits], axis=1)
    return value_rows, policy_rows


def collect_pendulum_pc_critic(seed: int, samples: int) -> np.ndarray:
    set_pendulum_seed(seed)
    device = torch.device("cpu")
    hidden = 16
    discount = 0.99
    q_scale = 10.0
    reward_scale = 0.1

    actor = Actor(hidden).to(device)
    actor_target = Actor(hidden).to(device)
    actor_target.load_state_dict(actor.state_dict())
    critic_target = PCCritic(hidden, gamma_pc=0.1, query_ticks=20, q_scale=q_scale).to(device)
    critic_target.load_state_dict(PCCritic(hidden, gamma_pc=0.1, query_ticks=20, q_scale=q_scale).state_dict())

    env = gym.make("Pendulum-v1")
    try:
        env.action_space.seed(seed)
    except AttributeError:
        pass
    rng = np.random.default_rng(seed)
    obs = np.asarray(reset_env(env, seed), dtype=np.float64)
    replay = []
    while len(replay) < max(80, samples):
        if len(replay) < 20:
            action = rng.uniform(-ACTION_LIMIT, ACTION_LIMIT, size=(1,)).astype(np.float32)
        else:
            state_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = actor(state_t).squeeze(0).cpu().numpy()
            action = np.clip(action, -ACTION_LIMIT, ACTION_LIMIT).astype(np.float32)
        next_obs, reward, done, _ = step_env(env, action)
        replay.append(Transition(
            state=np.asarray(obs, dtype=np.float32),
            action=np.asarray(action, dtype=np.float32),
            reward=reward_scale * float(reward),
            next_state=np.asarray(next_obs, dtype=np.float32),
            done=done,
        ))
        obs = np.asarray(reset_env(env, seed + len(replay)) if done else next_obs, dtype=np.float64)
    env.close()

    random.seed(seed + 20)
    batch = random.sample(replay, samples)
    states = torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=device)
    actions = torch.as_tensor(np.stack([t.action for t in batch]), dtype=torch.float32, device=device)
    rewards = torch.as_tensor([[t.reward] for t in batch], dtype=torch.float32, device=device)
    next_states = torch.as_tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=device)
    dones = torch.as_tensor([[t.done] for t in batch], dtype=torch.float32, device=device)
    with torch.no_grad():
        next_actions = actor_target(next_states)
        target_q = critic_target(next_states, next_actions)
        y = rewards + discount * (1.0 - dones) * target_q
    inputs = torch.cat([states, actions], dim=1).cpu().numpy().astype(np.float64)
    # PCCritic exactlocal_update clamps to target_norm = targets / q_scale.
    targets_norm = (y / q_scale).cpu().numpy().astype(np.float64)
    return np.concatenate([inputs, targets_norm], axis=1)


def write_batches(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cartpole_value, cartpole_policy = collect_cartpole(seed=42, samples=24)
    pendulum_critic = collect_pendulum_pc_critic(seed=42, samples=24)
    datasets = {
        "cartpole_value_td_batch_seed42": cartpole_value,
        "cartpole_policy_aif_batch_seed42": cartpole_policy,
        "pendulum_pc_critic_td_batch_seed42": pendulum_critic,
    }
    paths = {}
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
    out_dir = ROOT / "runs" / "rtl_runner_batch_trace"
    data_paths = write_batches(out_dir / "data")
    summary_path = ROOT / "runs" / "rtl_runner_batch_trace_summary.csv"

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
