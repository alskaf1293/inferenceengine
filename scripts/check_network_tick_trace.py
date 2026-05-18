#!/usr/bin/env python3
"""Compare tb_network_tick_trace RTL output against PCNetNLayer."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "python_rtl")
from pc_network import PCNetNLayer  # noqa: E402


FIELDS = [
    "x0_0", "x0_1", "x0_2",
    "x1_0", "x1_1", "x1_2", "x1_3",
    "x2_0", "x2_1",
    "l0_w00", "l0_w01", "l0_b0",
    "l1_w00", "l1_w01", "l1_b0",
    "l0_back_0_0",
]


def make_net(act_lut: list[str], alpha: float, gamma: float) -> PCNetNLayer:
    return PCNetNLayer(
        k_lut=[3, 4, 2],
        act_lut=act_lut,
        gamma=gamma,
        alpha=alpha,
        seed=0,
        rtl_init=True,
        gen_k_lut=[8, 64, 64],
        bias_init_scale=0.0,
        top_rtl_width=True,
    )


def snapshot(net: PCNetNLayer, tick: int) -> dict[str, float]:
    l0, l1, l2 = net.layers
    return {
        "tick": float(tick),
        "x0_0": float(l0.x_state[0]),
        "x0_1": float(l0.x_state[1]),
        "x0_2": float(l0.x_state[2]),
        "x1_0": float(l1.x_state[0]),
        "x1_1": float(l1.x_state[1]),
        "x1_2": float(l1.x_state[2]),
        "x1_3": float(l1.x_state[3]),
        "x2_0": float(l2.x_state[0]),
        "x2_1": float(l2.x_state[1]),
        "l0_w00": float(l0.W[0, 0]),
        "l0_w01": float(l0.W[0, 1]),
        "l0_b0": float(l0.bias[0]),
        "l1_w00": float(l1.W[0, 0]),
        "l1_w01": float(l1.W[0, 1]),
        "l1_b0": float(l1.bias[0]),
        "l0_back_0_0": float(l0.back_nk[0, 0]),
    }


def expected_rows(act_lut: list[str], alpha: float, gamma: float) -> list[dict[str, float]]:
    net = make_net(act_lut, alpha, gamma)
    sequence = [
        (np.array([0.7, -0.2]), np.array([0.1, -0.3, 0.2]), True, alpha),
        (np.array([0.7, -0.2]), np.array([0.1, -0.3, 0.2]), True, alpha),
        (np.array([-0.4, 0.9]), np.array([-0.2, 0.05, 0.3]), True, alpha),
        (np.array([-0.4, 0.9]), np.array([-0.2, 0.05, 0.3]), False, 0.0),
    ]
    rows = []
    for tick, (x_top, y_bottom, clamp_bottom, alpha) in enumerate(sequence, start=1):
        net.set_rates(alpha=alpha, gamma=gamma)
        net.tick_parallel(x_top, y_bottom, clamp_top=True, clamp_bottom=clamp_bottom)
        rows.append(snapshot(net, tick))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="runs/network_tick_trace.csv")
    parser.add_argument("--tol", type=float, default=5e-6)
    parser.add_argument("--act-lut", nargs=3, default=["linear", "relu", "linear"],
                        choices=["linear", "relu", "tanh", "sigmoid"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.10)
    args = parser.parse_args()

    with Path(args.csv).open(newline="") as f:
        got_rows = list(csv.DictReader(f))
    want_rows = expected_rows(args.act_lut, args.alpha, args.gamma)

    failures = []
    max_abs = 0.0
    for got, want in zip(got_rows, want_rows):
        tick = int(got["tick"])
        for field in FIELDS:
            g = float(got[field])
            w = float(want[field])
            err = abs(g - w)
            max_abs = max(max_abs, err)
            if err > args.tol:
                failures.append((tick, field, g, w, err))

    print(f"rows={len(got_rows)} max_abs_error={max_abs:.9g} tol={args.tol:g}")
    if failures:
        for tick, field, got, want, err in failures[:40]:
            print(f"FAIL tick={tick} field={field} got={got:.9g} expected={want:.9g} abs={err:.9g}")
        if len(failures) > 40:
            print(f"... {len(failures) - 40} more failures")
        raise SystemExit(1)
    print("PASS network tick RTL matches PCNetNLayer")


if __name__ == "__main__":
    main()
