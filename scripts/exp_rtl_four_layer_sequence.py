#!/usr/bin/env python3
"""Four-layer persistent RTL/Python trace for HalfCheetah-sized input width."""

from __future__ import annotations

import csv
import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PY = "/home/gregoryv/miniconda3/envs/dsl2/bin/python"

sys.path.insert(0, str(ROOT / "python_rtl"))
from run_mujoco_td3 import Actor, PCTickCritic, Transition, set_seed, td3_target  # noqa: E402

K0, K3 = 1, 23
UPDATES = 2
SAMPLES_PER_UPDATE = 6
INFER_TICKS = 8
LEARN_TICKS = 2
EVAL_TICKS = 10
ALPHA = 0.002
GAMMA = 0.05
TOL = 1e-3


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    data_kind: str
    k1: int
    k2: int


DATA_KINDS = {
    "synthetic": "synthetic",
    "halfcheetah_q": "halfcheetah_q",
    "halfcheetah_aif": "halfcheetah_aif",
}


def build_cases(widths: list[int], data_kinds: list[str]) -> list[Case]:
    cases = []
    for width in widths:
        for data_kind in data_kinds:
            if data_kind == "synthetic":
                name = f"halfcheetah_shape_synthetic_1_{width}_{width}_23"
                kind = "synthetic"
            elif data_kind == "halfcheetah_q":
                name = f"halfcheetah_q_td_batch_1_{width}_{width}_23"
                kind = "halfcheetah_q"
            elif data_kind == "halfcheetah_aif":
                name = f"halfcheetah_aif_td_batch_1_{width}_{width}_23"
                kind = "halfcheetah_aif"
            else:
                raise ValueError(f"unknown data kind {data_kind}")
            cases.append(Case(name, kind, data_kind, width, width))
    return cases


def run(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def make_synthetic_data(path: Path) -> None:
    rng = np.random.default_rng(42)
    rows = []
    for update in range(UPDATES):
        x = rng.normal(0.0, 0.35, size=(SAMPLES_PER_UPDATE, K3))
        # Small deterministic target with a changing slice per update.
        base = x[:, update:update + 4].sum(axis=1, keepdims=True)
        y = 0.15 * np.tanh(base)
        rows.append(np.concatenate([x, y], axis=1))
    data = np.vstack(rows).astype(np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, fmt="%.9g")


def make_halfcheetah_data(path: Path, critic_semantics: str, hidden: int) -> None:
    seed = 42
    set_seed(seed)
    device = torch.device("cpu")
    env = gym.make("HalfCheetah-v5")
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    action_scale = action_high
    action_scale_t = torch.as_tensor(action_scale, dtype=torch.float32, device=device).unsqueeze(0)

    actor_target = Actor(obs_dim, act_dim, hidden, action_scale).to(device)
    signed_efe = critic_semantics == "aif"
    value_scale = 1.0
    critic1_target = PCTickCritic(obs_dim, act_dim, hidden, 0.2, 20, value_scale, signed_efe).to(device)
    critic2_target = PCTickCritic(obs_dim, act_dim, hidden, 0.2, 20, value_scale, signed_efe).to(device)

    replay: list[Transition] = []
    while len(replay) < 96:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
        done = bool(terminated or truncated)
        replay.append(Transition(
            state=np.asarray(obs, dtype=np.float32),
            action=np.asarray(action, dtype=np.float32),
            reward=float(reward),
            next_state=np.asarray(next_obs, dtype=np.float32),
            done=done,
        ))
        obs = env.reset(seed=seed + len(replay))[0] if done else next_obs
    env.close()

    args = SimpleNamespace(
        critic_semantics=critic_semantics,
        discount=0.99,
        policy_noise=0.2,
        noise_clip=0.5,
    )
    rng = np.random.default_rng(seed + (300 if signed_efe else 250))
    rows = []
    for _ in range(UPDATES):
        indices = rng.choice(len(replay), size=SAMPLES_PER_UPDATE, replace=False)
        batch = [replay[int(i)] for i in indices]
        states = torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.stack([t.action for t in batch]), dtype=torch.float32, device=device)
        rewards = torch.as_tensor([[t.reward] for t in batch], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=device)
        dones = torch.as_tensor([[t.done] for t in batch], dtype=torch.float32, device=device)
        with torch.no_grad():
            noise = torch.randn_like(actions) * args.policy_noise
            noise = noise.clamp(-args.noise_clip, args.noise_clip)
            next_actions = (actor_target(next_states) + noise).clamp(-action_scale_t, action_scale_t)
            targets = td3_target(args, critic1_target, critic2_target, next_states, next_actions, rewards, dones)
        target_raw = -targets if signed_efe else targets
        target_norm = (target_raw / value_scale).cpu().numpy().astype(np.float64)
        inputs = torch.cat([states, actions], dim=1).cpu().numpy().astype(np.float64)
        rows.append(np.concatenate([inputs, target_norm], axis=1))

    data = np.vstack(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, fmt="%.9g")


def parse_max_abs(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("rows="):
            return line.split("max_abs_error=", 1)[1].split()[0]
    return ""


def make_data(case: Case, data_path: Path) -> None:
    if case.data_kind == "synthetic":
        make_synthetic_data(data_path)
    elif case.data_kind == "halfcheetah_q":
        make_halfcheetah_data(data_path, "q", case.k1)
    elif case.data_kind == "halfcheetah_aif":
        make_halfcheetah_data(data_path, "aif", case.k1)
    else:
        raise ValueError(f"unknown data kind {case.data_kind}")


def run_case(case: Case, out_dir: Path, reuse_existing: bool = False, cache_binaries: bool = True) -> dict[str, object]:
    data_path = out_dir / f"{case.name}.dat"
    csv_path = out_dir / f"{case.name}.csv"
    make_data(case, data_path)
    max_k = max(32, K0, case.k1, case.k2, K3)

    if not (reuse_existing and csv_path.exists()):
        env = os.environ.copy()
        if cache_binaries:
            env["VERILATOR_MDIR"] = (
                f"obj_dir/trace4_k{K0}_{case.k1}_{case.k2}_{K3}"
                f"_max{max_k}_u{UPDATES}_s{SAMPLES_PER_UPDATE}"
            )
            env["VERILATOR_REUSE_BINARY"] = "1"
        run([
            "./scripts/run_test.sh",
            "tb/tb_supervised_sequence_file_trace4.sv",
            "tb_supervised_sequence_file_trace4",
            f"-GK0={K0}",
            f"-GK1={case.k1}",
            f"-GK2={case.k2}",
            f"-GK3={K3}",
            f"-GMAX_K={max_k}",
            f"-GUPDATES={UPDATES}",
            f"-GSAMPLES_PER_UPDATE={SAMPLES_PER_UPDATE}",
            "--",
            f"+DATA={data_path}",
            f"+CSV={csv_path}",
            f"+INFER_TICKS={INFER_TICKS}",
            f"+LEARN_TICKS={LEARN_TICKS}",
            f"+EVAL_TICKS={EVAL_TICKS}",
            f"+ALPHA={ALPHA}",
            f"+GAMMA={GAMMA}",
        ], env=env)

    result = run([
        PY,
        "scripts/check_supervised_sequence_file_trace4.py",
        "--csv",
        str(csv_path),
        "--data",
        str(data_path),
        "--k0",
        str(K0),
        "--k1",
        str(case.k1),
        "--k2",
        str(case.k2),
        "--k3",
        str(K3),
        "--updates",
        str(UPDATES),
        "--samples-per-update",
        str(SAMPLES_PER_UPDATE),
        "--infer-ticks",
        str(INFER_TICKS),
        "--learn-ticks",
        str(LEARN_TICKS),
        "--eval-ticks",
        str(EVAL_TICKS),
        "--alpha",
        str(ALPHA),
        "--gamma",
        str(GAMMA),
        "--tol",
        str(TOL),
    ])
    max_abs = parse_max_abs(result.stdout)
    with csv_path.open(newline="") as f:
        curve = list(csv.DictReader(f))
    row = {
        "case": case.name,
        "kind": case.kind,
        "k0": K0,
        "k1": case.k1,
        "k2": case.k2,
        "k3": K3,
        "updates": UPDATES,
        "samples_per_update": SAMPLES_PER_UPDATE,
        "max_abs_error": max_abs,
        "tol": TOL,
        "final_mse": curve[-1]["mse"] if curve else "",
        "status": "pass",
        "data": str(data_path.relative_to(ROOT)),
        "csv": str(csv_path.relative_to(ROOT)),
    }
    print(f"PASS {row['case']} max_abs_error={max_abs} final_mse={row['final_mse']}", flush=True)
    return row


def main() -> None:
    global UPDATES, SAMPLES_PER_UPDATE, INFER_TICKS, LEARN_TICKS, EVAL_TICKS, ALPHA, GAMMA, TOL

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden-widths",
        default="16,32,64",
        help="comma-separated hidden widths to test; use 64 or 256 as targeted longer runs",
    )
    parser.add_argument(
        "--data-kinds",
        default="synthetic,halfcheetah_q,halfcheetah_aif",
        help=f"comma-separated data kinds from {','.join(DATA_KINDS)}",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse existing CSV traces and only rerun the Python checker/summary",
    )
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--samples-per-update", type=int, default=SAMPLES_PER_UPDATE)
    parser.add_argument("--infer-ticks", type=int, default=INFER_TICKS)
    parser.add_argument("--learn-ticks", type=int, default=LEARN_TICKS)
    parser.add_argument("--eval-ticks", type=int, default=EVAL_TICKS)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--tol", type=float, default=TOL)
    parser.add_argument(
        "--no-cache-binaries",
        action="store_true",
        help="disable per-width Verilator binary cache",
    )
    parser.add_argument(
        "--verilator-build-jobs",
        type=int,
        default=None,
        help="set VERILATOR_BUILD_JOBS for parallel C++ build, e.g. 16 on this 32-core host",
    )
    parser.add_argument(
        "--verilator-verilate-jobs",
        type=int,
        default=None,
        help="set VERILATOR_VERILATE_JOBS for parallel Verilator front-end work",
    )
    parser.add_argument(
        "--verilator-output-split",
        type=int,
        default=None,
        help="set VERILATOR_OUTPUT_SPLIT for large generated models",
    )
    parser.add_argument(
        "--verilator-output-split-cfuncs",
        type=int,
        default=None,
        help="set VERILATOR_OUTPUT_SPLIT_CFUNCS for large generated functions",
    )
    args = parser.parse_args()

    UPDATES = args.updates
    SAMPLES_PER_UPDATE = args.samples_per_update
    INFER_TICKS = args.infer_ticks
    LEARN_TICKS = args.learn_ticks
    EVAL_TICKS = args.eval_ticks
    ALPHA = args.alpha
    GAMMA = args.gamma
    TOL = args.tol

    if args.verilator_build_jobs is not None:
        os.environ["VERILATOR_BUILD_JOBS"] = str(args.verilator_build_jobs)
    if args.verilator_verilate_jobs is not None:
        os.environ["VERILATOR_VERILATE_JOBS"] = str(args.verilator_verilate_jobs)
    if args.verilator_output_split is not None:
        os.environ["VERILATOR_OUTPUT_SPLIT"] = str(args.verilator_output_split)
    if args.verilator_output_split_cfuncs is not None:
        os.environ["VERILATOR_OUTPUT_SPLIT_CFUNCS"] = str(args.verilator_output_split_cfuncs)

    widths = [int(v.strip()) for v in args.hidden_widths.split(",") if v.strip()]
    data_kinds = [v.strip() for v in args.data_kinds.split(",") if v.strip()]
    unknown = sorted(set(data_kinds) - set(DATA_KINDS))
    if unknown:
        raise ValueError(f"unknown data kinds: {unknown}")
    cases = build_cases(widths, data_kinds)

    out_dir = ROOT / "runs" / "rtl_four_layer_sequence"
    summary_path = ROOT / "runs" / "rtl_four_layer_sequence_summary.csv"
    rows = [
        run_case(
            case,
            out_dir,
            reuse_existing=args.reuse_existing,
            cache_binaries=not args.no_cache_binaries,
        )
        for case in cases
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
