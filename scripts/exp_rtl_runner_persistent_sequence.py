#!/usr/bin/env python3
"""Persistent RTL/Python sequence checks for runner-derived update batches."""

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
    samples_per_update: int
    updates: int
    infer_ticks: int = 10
    learn_ticks: int = 2
    eval_ticks: int = 16
    alpha: float = 0.03
    gamma: float = 0.06
    tol: float = 5e-5


CASES = [
    Case("cartpole_value_td_persistent_sequence_seed42", "cartpole_value_td", 2, 16, 4, 12, 4),
    Case("cartpole_policy_aif_persistent_sequence_seed42", "cartpole_policy_aif", 2, 16, 4, 12, 4),
    Case("pendulum_pc_critic_td_persistent_sequence_seed42", "pendulum_pc_critic_td", 1, 16, 4, 12, 4),
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


def cartpole_target_logits(values: np.ndarray, smoothing: float = 0.02, clip: float = 5.0) -> np.ndarray:
    greedy = int(np.argmax(values))
    probs = np.full(CARTPOLE_ACTION_SIZE, smoothing / max(1, CARTPOLE_ACTION_SIZE - 1), dtype=np.float64)
    probs[greedy] = 1.0 - smoothing
    logits = np.log(np.clip(probs, 1e-6, 1.0))
    logits -= np.mean(logits)
    return np.clip(logits, -clip, clip)


def collect_cartpole_context(seed: int, n_transitions: int):
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
    history_states = []
    while len(memory) < n_transitions:
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
    return memory, np.asarray(history_states, dtype=np.float32), value_model, policy_model


def cartpole_value_sequence(case: Case, seed: int) -> np.ndarray:
    memory, _, value_model, policy_model = collect_cartpole_context(seed, n_transitions=128)
    rng = random.Random(seed + 100)
    updates = []
    for _ in range(case.updates):
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
        updates.append(np.concatenate([states.astype(np.float64), y.astype(np.float64)], axis=1))
    return np.vstack(updates)


def cartpole_policy_sequence(case: Case, seed: int) -> np.ndarray:
    _, history_states, value_model, _ = collect_cartpole_context(seed, n_transitions=128)
    updates = []
    cursor = 0
    for _ in range(case.updates):
        states = history_states[cursor:cursor + case.samples_per_update]
        cursor += case.samples_per_update
        values = value_model.predict_np(states, target=False)
        logits = np.stack([cartpole_target_logits(row) for row in values], axis=0)
        updates.append(np.concatenate([states.astype(np.float64), logits], axis=1))
    return np.vstack(updates)


def pendulum_pc_critic_sequence(case: Case, seed: int) -> np.ndarray:
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
    rng_np = np.random.default_rng(seed)
    obs = np.asarray(reset_env(env, seed), dtype=np.float64)
    replay = []
    while len(replay) < 128:
        if len(replay) < 20:
            action = rng_np.uniform(-ACTION_LIMIT, ACTION_LIMIT, size=(1,)).astype(np.float32)
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

    rng = random.Random(seed + 200)
    updates = []
    for _ in range(case.updates):
        batch = rng.sample(replay, case.samples_per_update)
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
        targets_norm = (y / q_scale).cpu().numpy().astype(np.float64)
        updates.append(np.concatenate([inputs, targets_norm], axis=1))
    return np.vstack(updates)


def build_flat_data(case: Case, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if case.kind == "cartpole_value_td":
        data = cartpole_value_sequence(case, seed=42)
    elif case.kind == "cartpole_policy_aif":
        data = cartpole_policy_sequence(case, seed=42)
    elif case.kind == "pendulum_pc_critic_td":
        data = pendulum_pc_critic_sequence(case, seed=42)
    else:
        raise ValueError(f"unknown case kind {case.kind}")
    path = out_dir / f"{case.name}.dat"
    np.savetxt(path, data, fmt="%.9g")
    return path


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


def read_curve(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def run_case(case: Case, out_dir: Path) -> dict[str, object]:
    data_path = build_flat_data(case, out_dir / "data")
    csv_path = out_dir / f"{case.name}.csv"
    run([
        "./scripts/run_test.sh",
        "tb/tb_supervised_sequence_file_trace.sv",
        "tb_supervised_sequence_file_trace",
        f"-GK0={case.k0}",
        f"-GK1={case.k1}",
        f"-GK2={case.k2}",
        f"-GUPDATES={case.updates}",
        f"-GSAMPLES_PER_UPDATE={case.samples_per_update}",
        "--",
        f"+DATA={data_path}",
        f"+CSV={csv_path}",
        f"+INFER_TICKS={case.infer_ticks}",
        f"+LEARN_TICKS={case.learn_ticks}",
        f"+EVAL_TICKS={case.eval_ticks}",
        f"+ALPHA={case.alpha}",
        f"+GAMMA={case.gamma}",
    ])

    result = run([
        PY,
        "scripts/check_supervised_sequence_file_trace.py",
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
        "--updates",
        str(case.updates),
        "--samples-per-update",
        str(case.samples_per_update),
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
    curve = read_curve(csv_path)
    row = {
        "case": case.name,
        "kind": case.kind,
        "k0": case.k0,
        "k1": case.k1,
        "k2": case.k2,
        "updates": case.updates,
        "samples_per_update": case.samples_per_update,
        "infer_ticks": case.infer_ticks,
        "learn_ticks": case.learn_ticks,
        "eval_ticks": case.eval_ticks,
        "alpha": case.alpha,
        "gamma": case.gamma,
        "max_abs_error": max_abs,
        "final_update": curve[-1]["update"] if curve else "",
        "final_mse": curve[-1]["mse"] if curve else "",
        "status": "pass",
        "data": str(data_path.relative_to(ROOT)),
        "csv": str(csv_path.relative_to(ROOT)),
    }
    print(f"PASS {case.name} max_abs_error={max_abs} final_mse={row['final_mse']}")
    return row


def main() -> None:
    out_dir = ROOT / "runs" / "rtl_runner_persistent_sequence"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = ROOT / "runs" / "rtl_runner_persistent_sequence_summary.csv"
    rows = [run_case(case, out_dir) for case in CASES]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
