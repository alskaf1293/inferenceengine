#!/usr/bin/env python3
"""Diagnose fast PC bridge alignment with tick-faithful PC updates.

The CartPole bridge works in CUDA-fast mode. This script asks the next
scientific question: how close are the fast bridge, nudged tick update, and
original full-clamp PC update on fixed batches and short CartPole probes?
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_rtl"))

from run_cartpole_millidge_hybrid import (  # noqa: E402
    ACTION_SIZE,
    STATE_SIZE,
    BPMLP,
    PCValueModel,
)


@dataclass(frozen=True)
class ModeSpec:
    key: str
    label: str
    gradient_mode: str
    device: str


MODES = (
    ModeSpec("pc", "full-clamp PC", "pc", "cpu"),
    ModeSpec("pc_nudge_gated", "nudged gated PC ticks", "pc_nudge_gated", "cpu"),
    ModeSpec("pc_nudge_gated_torch_tick", "nudged gated Torch ticks", "pc_nudge_gated_torch_tick", "cuda"),
    ModeSpec("pc_nudge_gated_torch_backvec", "nudged gated Torch back-vector", "pc_nudge_gated_torch_backvec", "cuda"),
    ModeSpec("pc_nudge_gated_torch_exactlocal", "Torch exact local back-vector", "pc_nudge_gated_torch_exactlocal", "cuda"),
    ModeSpec("pc_nudge_gated_fast", "nudged gated CUDA bridge", "pc_nudge_gated_fast", "cuda"),
    ModeSpec("bp_equiv_fast", "exact BP bridge", "bp_equiv_fast", "cuda"),
)


def flatten_grads(grads: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([grads[key].reshape(-1) for key in ("W0", "b0", "W1", "b1")])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (denom + 1e-12))


def relerr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-12))


def make_pc(mode: ModeSpec, args: argparse.Namespace) -> PCValueModel:
    device = torch.device(mode.device if mode.device == "cuda" and torch.cuda.is_available() else "cpu")
    return PCValueModel(
        hidden=args.hidden,
        lr=args.lr,
        discount=0.99,
        seed=args.seed,
        gamma_pc=args.gamma_pc,
        n_infer=args.infer_ticks,
        n_learn=10,
        n_query=args.query_ticks,
        adaptive_inference=False,
        settle_tol=1e-6,
        max_infer_ticks=args.infer_ticks,
        max_query_ticks=args.query_ticks,
        value_scale=1.0,
        init="from_bp",
        device=device,
        optimizer="adam",
        gradient_mode=mode.gradient_mode,
        nudge_beta=args.nudge_beta,
    )


def copy_pc_to_bp(pc: PCValueModel, hidden: int) -> BPMLP:
    bp = BPMLP(STATE_SIZE, hidden, ACTION_SIZE)
    with torch.no_grad():
        bp.fc1.weight.copy_(torch.as_tensor(pc.net.layer1.W, dtype=torch.float32))
        bp.fc1.bias.copy_(torch.as_tensor(pc.net.layer1.bias, dtype=torch.float32))
        bp.fc2.weight.copy_(torch.as_tensor(pc.net.layer0.W, dtype=torch.float32))
        bp.fc2.bias.copy_(torch.as_tensor(pc.net.layer0.bias, dtype=torch.float32))
    return bp


def bp_batch_grads(bp: BPMLP, states: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    x = torch.as_tensor(states, dtype=torch.float32)
    y = torch.as_tensor(targets, dtype=torch.float32)
    loss = F.mse_loss(bp(x), y)
    bp.zero_grad()
    loss.backward()
    return {
        "W0": bp.fc2.weight.grad.detach().cpu().numpy().astype(np.float64),
        "b0": bp.fc2.bias.grad.detach().cpu().numpy().astype(np.float64),
        "W1": bp.fc1.weight.grad.detach().cpu().numpy().astype(np.float64),
        "b1": bp.fc1.bias.grad.detach().cpu().numpy().astype(np.float64),
    }


def pc_batch_grads(pc: PCValueModel, states: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    if pc.torch_tick:
        states_t = torch.as_tensor(states, dtype=torch.float32, device=pc.device)
        targets_t = torch.as_tensor(targets, dtype=torch.float32, device=pc.device)
        pred = pc._torch_tick_query(states_t, target=False, raw_units=True).detach()
        pred_norm = pred / pc.value_scale
        target_norm = targets_t / pc.value_scale
        if pc.gradient_mode == "pc_nudge_gated_torch_exactlocal":
            out_delta = pred_norm - target_norm
            batch = states_t.shape[0]
            x1 = torch.full((batch, pc.tW1.shape[0]), 0.001, dtype=states_t.dtype, device=pc.device)
            x0 = torch.full((batch, pc.tW0.shape[0]), 0.001, dtype=states_t.dtype, device=pc.device)
            back0 = torch.zeros((batch, pc.tW1.shape[0]), dtype=states_t.dtype, device=pc.device)
            with torch.no_grad():
                for _ in range(pc.n_query):
                    mu1 = F.linear(states_t, pc.tW1, pc.tb1)
                    eps1 = x1 - mu1
                    x1 = x1 + pc.gamma_pc * (back0 - eps1)

                    hidden_phi = F.relu(x1)
                    mu0 = F.linear(hidden_phi, pc.tW0, pc.tb0)
                    eps0 = x0 - mu0
                    back0 = eps0 @ pc.tW0
                    x0 = x0 - pc.gamma_pc * eps0

                hidden_prime = (x1 > 0.0).float()
                hidden_delta = (out_delta @ pc.tW0) * hidden_prime
                return {
                    "W0": ((out_delta.T @ hidden_phi) / batch).detach().cpu().numpy().astype(np.float64),
                    "b0": out_delta.mean(dim=0).detach().cpu().numpy().astype(np.float64),
                    "W1": ((hidden_delta.T @ states_t) / batch).detach().cpu().numpy().astype(np.float64),
                    "b1": hidden_delta.mean(dim=0).detach().cpu().numpy().astype(np.float64),
                }

        beta = max(float(pc.nudge_beta), 1e-12)
        nudged = pred_norm + beta * (target_norm - pred_norm)

        batch = states_t.shape[0]
        x1 = torch.full((batch, pc.tW1.shape[0]), 0.001, dtype=states_t.dtype, device=pc.device)
        back0 = torch.zeros((batch, pc.tW1.shape[0]), dtype=states_t.dtype, device=pc.device)
        eps0 = torch.zeros((batch, pc.tW0.shape[0]), dtype=states_t.dtype, device=pc.device)
        eps1 = torch.zeros((batch, pc.tW1.shape[0]), dtype=states_t.dtype, device=pc.device)
        with torch.no_grad():
            for _ in range(pc.n_infer):
                mu1 = F.linear(states_t, pc.tW1, pc.tb1)
                eps1 = x1 - mu1
                x1 = x1 + pc.gamma_pc * (back0 - eps1)

                phi1 = F.relu(x1)
                mu0 = F.linear(phi1, pc.tW0, pc.tb0)
                eps0 = nudged - mu0
                back0 = eps0 @ pc.tW0

            hidden_phi = F.relu(x1)
            hidden_prime = (x1 > 0.0).float()
            hidden_signal = back0 if pc.gradient_mode == "pc_nudge_gated_torch_backvec" else eps1
            gated_eps1 = hidden_signal * hidden_prime
            scale = 1.0 / beta
            return {
                "W0": (-scale * (eps0.T @ hidden_phi) / batch).detach().cpu().numpy().astype(np.float64),
                "b0": (-scale * eps0.mean(dim=0)).detach().cpu().numpy().astype(np.float64),
                "W1": (-scale * (gated_eps1.T @ states_t) / batch).detach().cpu().numpy().astype(np.float64),
                "b1": (-scale * gated_eps1.mean(dim=0)).detach().cpu().numpy().astype(np.float64),
            }

    if pc.torch_fast:
        states_t = torch.as_tensor(states, dtype=torch.float32, device=pc.device)
        targets_t = torch.as_tensor(targets, dtype=torch.float32, device=pc.device)
        pred = pc._torch_forward(states_t, target=False, raw_units=True)
        if pc.gradient_mode == "bp_equiv_fast":
            loss = F.mse_loss(pred, targets_t)
        else:
            pred_norm = pred / pc.value_scale
            target_norm = targets_t / pc.value_scale
            loss = 0.5 * ((pred_norm - target_norm) ** 2).sum(dim=1).mean()
        params = (pc.tW0, pc.tb0, pc.tW1, pc.tb1)
        for param in params:
            if param.grad is not None:
                param.grad = None
        loss.backward()
        return {
            "W0": pc.tW0.grad.detach().cpu().numpy().astype(np.float64),
            "b0": pc.tb0.grad.detach().cpu().numpy().astype(np.float64),
            "W1": pc.tW1.grad.detach().cpu().numpy().astype(np.float64),
            "b1": pc.tb1.grad.detach().cpu().numpy().astype(np.float64),
        }

    grads = {
        "W0": np.zeros_like(pc.net.layer0.W),
        "b0": np.zeros_like(pc.net.layer0.bias),
        "W1": np.zeros_like(pc.net.layer1.W),
        "b1": np.zeros_like(pc.net.layer1.bias),
    }
    for state, target in zip(states, targets):
        sample = pc._pc_gradient_single(state, target)
        for key in grads:
            grads[key] += sample[key]
    for key in grads:
        grads[key] /= len(states)
    return grads


def run_frozen_batch(args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    states = (rng.standard_normal((args.batch_size, STATE_SIZE)) * args.state_scale).astype(np.float64)
    targets = (rng.standard_normal((args.batch_size, ACTION_SIZE)) * args.target_scale).astype(np.float64)

    reference_pc = make_pc(MODES[0], args)
    bp = copy_pc_to_bp(reference_pc, args.hidden)
    bp_grads = bp_batch_grads(bp, states, targets)
    bp_flat = flatten_grads(bp_grads)
    with torch.no_grad():
        bp_forward = bp(torch.as_tensor(states, dtype=torch.float32)).cpu().numpy()

    rows = []
    for mode in MODES:
        pc = make_pc(mode, args)
        start = time.perf_counter()
        pred = pc.predict_np(states, target=False)
        forward_seconds = time.perf_counter() - start

        start = time.perf_counter()
        grads = pc_batch_grads(pc, states, targets)
        grad_seconds = time.perf_counter() - start

        grad_flat = flatten_grads(grads)
        optimal_scale = float(np.dot(bp_flat, grad_flat) / (np.dot(grad_flat, grad_flat) + 1e-12))
        scaled_grad_flat = optimal_scale * grad_flat
        row = {
            "mode": mode.key,
            "label": mode.label,
            "device": str(pc.device),
            "batch_size": args.batch_size,
            "hidden": args.hidden,
            "query_ticks": args.query_ticks,
            "infer_ticks": args.infer_ticks,
            "nudge_beta": args.nudge_beta,
            "forward_mse_vs_bp": float(np.mean((pred - bp_forward) ** 2)),
            "forward_max_abs_vs_bp": float(np.max(np.abs(pred - bp_forward))),
            "grad_cosine_vs_bp": cosine(bp_flat, grad_flat),
            "grad_relerr_vs_bp": relerr(bp_flat, grad_flat),
            "grad_norm_ratio_vs_bp": float(np.linalg.norm(grad_flat) / (np.linalg.norm(bp_flat) + 1e-12)),
            "optimal_grad_scale_vs_bp": optimal_scale,
            "scaled_grad_relerr_vs_bp": relerr(bp_flat, scaled_grad_flat),
            "forward_seconds": forward_seconds,
            "grad_seconds": grad_seconds,
        }
        for key in ("W0", "b0", "W1", "b1"):
            row[f"{key}_cosine_vs_bp"] = cosine(bp_grads[key].reshape(-1), grads[key].reshape(-1))
            row[f"{key}_relerr_vs_bp"] = relerr(bp_grads[key].reshape(-1), grads[key].reshape(-1))
        rows.append(row)

    df = pd.DataFrame(rows)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    out = args.runs_dir / "pc_bridge_equivalence.csv"
    df.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {out}")
    print(df[["mode", "grad_cosine_vs_bp", "grad_relerr_vs_bp", "grad_norm_ratio_vs_bp",
              "optimal_grad_scale_vs_bp", "scaled_grad_relerr_vs_bp", "forward_mse_vs_bp",
              "grad_seconds"]].to_string(index=False))
    return df


def plot_alignment(args: argparse.Namespace, df: pd.DataFrame) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    colors = ["#999999", "#0072B2", "#D55E00", "#333333"]
    ax.bar(df["mode"], df["grad_cosine_vs_bp"], color=colors[:len(df)])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Gradient cosine vs BP")
    ax.set_title("PC Bridge Gradient Alignment")
    ax.grid(axis="y", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelrotation=18)
    fig.tight_layout()
    out = args.out_dir / "pc_bridge_gradient_alignment.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")


def cartpole_mode_args(mode: str) -> tuple[str, ...]:
    if mode == "pc":
        return ("--pc-gradient-mode", "pc", "--device", "cpu")
    if mode == "pc_nudge_gated":
        return (
            "--pc-gradient-mode", "pc_nudge_gated",
            "--pc-nudge-beta", "0.001",
            "--pc-infer", "300",
            "--max-infer-ticks", "300",
            "--no-adaptive-inference",
            "--device", "cpu",
        )
    if mode == "pc_nudge_gated_fast":
        return ("--pc-gradient-mode", "pc_nudge_gated_fast", "--device", "cuda")
    raise ValueError(mode)


def run_short_cartpole(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    script = ROOT / "python_rtl" / "run_cartpole_millidge_hybrid.py"
    modes = ("pc", "pc_nudge_gated", "pc_nudge_gated_fast")
    for mode in modes:
        for seed in args.cartpole_seeds:
            out_csv = args.runs_dir / f"short_cartpole_{mode}_seed{seed}_{args.cartpole_episodes}.csv"
            if args.run_cartpole:
                cmd = [
                    sys.executable,
                    str(script),
                    "--episodes", str(args.cartpole_episodes),
                    "--seed", str(seed),
                    "--value-backend", "pc",
                    "--policy-backend", "bp",
                    "--out-csv", str(out_csv),
                    "--infotime", str(args.cartpole_episodes),
                    *cartpole_mode_args(mode),
                ]
                print("Running short CartPole", mode, "seed", seed)
                subprocess.run(cmd, check=True)
            if not out_csv.exists():
                continue
            data = pd.read_csv(out_csv)
            rewards = data["reward"].to_numpy(dtype=np.float64)
            rows.append({
                "mode": mode,
                "seed": seed,
                "episodes": len(rewards),
                "avg_reward": float(np.mean(rewards)),
                "final_avg50": float(np.mean(rewards[-50:])),
                "best": int(np.max(rewards)),
                "csv": str(out_csv),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        out = args.runs_dir / "pc_bridge_short_cartpole.csv"
        df.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Wrote {out}")
        print(df.to_string(index=False))
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "python_runs")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-ticks", type=int, default=300)
    parser.add_argument("--infer-ticks", type=int, default=300)
    parser.add_argument("--gamma-pc", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--nudge-beta", type=float, default=0.001)
    parser.add_argument("--state-scale", type=float, default=0.5)
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--run-cartpole", action="store_true")
    parser.add_argument("--cartpole-episodes", type=int, default=50)
    parser.add_argument("--cartpole-seeds", type=int, nargs="+", default=[42])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_frozen_batch(args)
    plot_alignment(args, df)
    run_short_cartpole(args)


if __name__ == "__main__":
    main()
