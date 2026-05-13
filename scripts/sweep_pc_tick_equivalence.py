#!/usr/bin/env python3
"""Sweep Torch tick-faithful critic settings on frozen batches.

The fast PC bridge reproduces CartPole, while the RTL-faithful Torch tick
critic is stable but underpowered. This script searches the local tick
hyperparameters for settings that better match the BP critic gradient before
spending time on RL runs.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diagnose_pc_bridge_equivalence import (  # noqa: E402
    MODES,
    STATE_SIZE,
    ACTION_SIZE,
    bp_batch_grads,
    copy_pc_to_bp,
    cosine,
    flatten_grads,
    make_pc,
    pc_batch_grads,
    relerr,
)


ROOT = Path(__file__).resolve().parents[1]
TORCH_TICK_MODE = next(mode for mode in MODES if mode.key == "pc_nudge_gated_torch_tick")
TORCH_BACKVEC_MODE = next(mode for mode in MODES if mode.key == "pc_nudge_gated_torch_backvec")


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item]


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item]


def run_sweep(args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    states = (rng.standard_normal((args.batch_size, STATE_SIZE)) * args.state_scale).astype(np.float64)
    targets = (rng.standard_normal((args.batch_size, ACTION_SIZE)) * args.target_scale).astype(np.float64)

    rows = []
    for gamma_pc in args.gamma_pcs:
        for infer_ticks in args.infer_ticks:
            for query_ticks in args.query_ticks:
                for nudge_beta in args.nudge_betas:
                    local_args = argparse.Namespace(
                        hidden=args.hidden,
                        lr=args.lr,
                        seed=args.seed,
                        gamma_pc=gamma_pc,
                        infer_ticks=infer_ticks,
                        query_ticks=query_ticks,
                        nudge_beta=nudge_beta,
                    )
                    mode = TORCH_BACKVEC_MODE if args.backvec else TORCH_TICK_MODE
                    reference_pc = make_pc(mode, local_args)
                    bp = copy_pc_to_bp(reference_pc, args.hidden)
                    bp_grads = bp_batch_grads(bp, states, targets)
                    bp_flat = flatten_grads(bp_grads)

                    pred = reference_pc.predict_np(states, target=False)
                    with torch.no_grad():
                        bp_forward = bp(torch.as_tensor(states, dtype=torch.float32)).cpu().numpy()

                    grads = pc_batch_grads(reference_pc, states, targets)
                    grad_flat = flatten_grads(grads)
                    optimal_scale = float(np.dot(bp_flat, grad_flat) / (np.dot(grad_flat, grad_flat) + 1e-12))
                    scaled = optimal_scale * grad_flat

                    rows.append({
                        "gamma_pc": gamma_pc,
                        "infer_ticks": infer_ticks,
                        "query_ticks": query_ticks,
                        "nudge_beta": nudge_beta,
                        "grad_cosine_vs_bp": cosine(bp_flat, grad_flat),
                        "grad_relerr_vs_bp": relerr(bp_flat, grad_flat),
                        "grad_norm_ratio_vs_bp": float(np.linalg.norm(grad_flat) / (np.linalg.norm(bp_flat) + 1e-12)),
                        "optimal_grad_scale_vs_bp": optimal_scale,
                        "scaled_grad_relerr_vs_bp": relerr(bp_flat, scaled),
                        "forward_mse_vs_bp": float(np.mean((pred - bp_forward) ** 2)),
                        "forward_max_abs_vs_bp": float(np.max(np.abs(pred - bp_forward))),
                    })
                    print(rows[-1])

    df = pd.DataFrame(rows).sort_values(
        ["scaled_grad_relerr_vs_bp", "grad_cosine_vs_bp"],
        ascending=[True, False],
    )
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    out = args.runs_dir / args.out_name
    df.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {out}")
    print(df.head(args.top).to_string(index=False))
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "python_runs")
    parser.add_argument("--out-name", default="pc_tick_equivalence_sweep.csv")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--state-scale", type=float, default=0.5)
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--gamma-pcs", type=parse_float_list, default=parse_float_list("0.02,0.05,0.1,0.2,0.4"))
    parser.add_argument("--infer-ticks", type=parse_int_list, default=parse_int_list("50,100,200,300,600"))
    parser.add_argument("--query-ticks", type=parse_int_list, default=parse_int_list("300"))
    parser.add_argument("--nudge-betas", type=parse_float_list, default=parse_float_list("0.0001,0.001,0.01"))
    parser.add_argument("--backvec", action="store_true")
    parser.add_argument("--top", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    run_sweep(parse_args())


if __name__ == "__main__":
    main()
