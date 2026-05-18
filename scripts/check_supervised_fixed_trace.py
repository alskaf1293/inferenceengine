#!/usr/bin/env python3
"""Compare fixed supervised RTL learning trace against PCNetNLayer."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "python_rtl")
from pc_network import PCNetNLayer  # noqa: E402


X = np.array([
    [0.70, -0.20],
    [-0.40, 0.90],
    [0.20, 0.30],
    [-0.80, -0.50],
], dtype=np.float64)

Y = np.array([
    [0.10, -0.30, 0.20],
    [-0.20, 0.05, 0.30],
    [0.25, -0.10, 0.00],
    [-0.15, 0.20, -0.25],
], dtype=np.float64)


def make_net(alpha: float, gamma: float) -> PCNetNLayer:
    return PCNetNLayer(
        k_lut=[3, 4, 2],
        act_lut=["linear", "relu", "linear"],
        gamma=gamma,
        alpha=alpha,
        seed=0,
        rtl_init=True,
        gen_k_lut=[8, 64, 64],
        bias_init_scale=0.0,
        top_rtl_width=True,
    )


def eval_mse(net: PCNetNLayer, gamma: float, eval_ticks: int) -> float:
    net.set_rates(alpha=0.0, gamma=gamma)
    acc = 0.0
    for x, y in zip(X, Y):
        for _ in range(eval_ticks):
            net.tick_parallel(x, None, clamp_top=True, clamp_bottom=False)
        diff = net.x0 - y
        acc += float(np.dot(diff, diff)) / Y.shape[1]
    return acc / len(X)


def expected_rows(epochs: int, infer_ticks: int, learn_ticks: int,
                  eval_ticks: int, alpha: float, gamma: float) -> list[tuple[int, float]]:
    net = make_net(alpha, gamma)
    rows = [(0, eval_mse(net, gamma, eval_ticks))]
    for ep in range(epochs):
        for x, y in zip(X, Y):
            net.set_rates(alpha=0.0, gamma=gamma)
            for _ in range(infer_ticks):
                net.tick_parallel(x, y, clamp_top=True, clamp_bottom=True)
            net.set_rates(alpha=alpha, gamma=gamma)
            for _ in range(learn_ticks):
                net.tick_parallel(x, y, clamp_top=True, clamp_bottom=True)
        rows.append((ep + 1, eval_mse(net, gamma, eval_ticks)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="runs/supervised_fixed_trace.csv")
    parser.add_argument("--tol", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--infer-ticks", type=int, default=10)
    parser.add_argument("--learn-ticks", type=int, default=2)
    parser.add_argument("--eval-ticks", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.10)
    args = parser.parse_args()

    with Path(args.csv).open(newline="") as f:
        got = [(int(r["epoch"]), float(r["mse"])) for r in csv.DictReader(f)]
    want = expected_rows(
        args.epochs,
        args.infer_ticks,
        args.learn_ticks,
        args.eval_ticks,
        args.alpha,
        args.gamma,
    )

    max_abs = 0.0
    failures = []
    for (ge, gm), (we, wm) in zip(got, want):
        err = abs(gm - wm)
        max_abs = max(max_abs, err)
        if ge != we or err > args.tol:
            failures.append((ge, gm, we, wm, err))

    print(f"rows={len(got)} max_abs_error={max_abs:.9g} tol={args.tol:g}")
    if failures:
        for ge, gm, we, wm, err in failures:
            print(f"FAIL got_epoch={ge} got_mse={gm:.9g} expected_epoch={we} expected_mse={wm:.9g} abs={err:.9g}")
        raise SystemExit(1)
    print("PASS supervised fixed RTL learning curve matches PCNetNLayer")


if __name__ == "__main__":
    main()
