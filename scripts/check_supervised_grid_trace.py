#!/usr/bin/env python3
"""Compare parameterized supervised RTL grid trace against PCNetNLayer."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "python_rtl")
from pc_network import PCNetNLayer  # noqa: E402


def build_teacher(k0: int, k1: int, k2: int):
    b = np.zeros((k1, k2), dtype=np.float64)
    a = np.zeros((k0, k1), dtype=np.float64)
    for i in range(k1):
        row_type = i % 4
        j0 = 0 if k2 == 1 else i % k2
        j1 = 0 if k2 == 1 else (i + 1) % k2
        if row_type == 0:
            b[i, j0] = 1.00
            if k2 > 1:
                b[i, j1] = -0.20
        elif row_type == 1:
            b[i, j0] = -0.15
            if k2 > 1:
                b[i, j1] = 0.95
        elif row_type == 2:
            b[i, j0] = 0.70
            if k2 > 1:
                b[i, j1] = 0.25
        else:
            b[i, j0] = 0.20
            if k2 > 1:
                b[i, j1] = 0.80

    for o in range(k0):
        row_type = o % 3
        base = 0 if k1 <= 4 else (2 * o) % k1
        if row_type == 0:
            vals = [(0, 0.90), (1, -0.45), (2, 0.30)]
        elif row_type == 1:
            vals = [(0, -0.70), (1, 0.85), (3, 0.25)]
        else:
            vals = [(0, 0.50), (1, 0.60), (2, -0.20), (3, 0.35)]
        for off, val in vals:
            if base + off < k1:
                a[o, base + off] = val
    return a, b


def build_dataset(k0: int, k1: int, k2: int, n_samples: int):
    a, b = build_teacher(k0, k1, k2)
    x = np.zeros((n_samples, k2), dtype=np.float64)
    for s in range(n_samples):
        for j in range(k2):
            raw = ((s * 17 + j * 11) % 25) - 12
            x[s, j] = raw / 10.0
    h = np.maximum(0.0, x @ b.T)
    y = h @ a.T
    return x, y


def make_net(k0: int, k1: int, k2: int, alpha: float, gamma: float) -> PCNetNLayer:
    return PCNetNLayer(
        k_lut=[k0, k1, k2],
        act_lut=["linear", "relu", "linear"],
        gamma=gamma,
        alpha=alpha,
        seed=0,
        rtl_init=True,
        gen_k_lut=[8, 64, 64],
        bias_init_scale=0.0,
        top_rtl_width=True,
    )


def eval_mse(net: PCNetNLayer, x, y, gamma: float, eval_ticks: int) -> float:
    net.set_rates(alpha=0.0, gamma=gamma)
    acc = 0.0
    for xs, ys in zip(x, y):
        for _ in range(eval_ticks):
            net.tick_parallel(xs, None, clamp_top=True, clamp_bottom=False)
        diff = net.x0 - ys
        acc += float(np.dot(diff, diff)) / y.shape[1]
    return acc / len(x)


def expected_rows(args) -> list[tuple[int, float]]:
    x, y = build_dataset(args.k0, args.k1, args.k2, args.samples)
    net = make_net(args.k0, args.k1, args.k2, args.alpha, args.gamma)
    rows = [(0, eval_mse(net, x, y, args.gamma, args.eval_ticks))]
    for ep in range(args.epochs):
        for xs, ys in zip(x, y):
            net.set_rates(alpha=0.0, gamma=args.gamma)
            for _ in range(args.infer_ticks):
                net.tick_parallel(xs, ys, clamp_top=True, clamp_bottom=True)
            net.set_rates(alpha=args.alpha, gamma=args.gamma)
            for _ in range(args.learn_ticks):
                net.tick_parallel(xs, ys, clamp_top=True, clamp_bottom=True)
        rows.append((ep + 1, eval_mse(net, x, y, args.gamma, args.eval_ticks)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="runs/supervised_grid_trace.csv")
    parser.add_argument("--k0", type=int, default=3)
    parser.add_argument("--k1", type=int, default=4)
    parser.add_argument("--k2", type=int, default=2)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--infer-ticks", type=int, default=10)
    parser.add_argument("--learn-ticks", type=int, default=2)
    parser.add_argument("--eval-ticks", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--tol", type=float, default=5e-5)
    args = parser.parse_args()

    with Path(args.csv).open(newline="") as f:
        got = [(int(r["epoch"]), float(r["mse"])) for r in csv.DictReader(f)]
    want = expected_rows(args)

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
    print("PASS supervised grid RTL learning curve matches PCNetNLayer")


if __name__ == "__main__":
    main()
