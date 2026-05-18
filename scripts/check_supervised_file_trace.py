#!/usr/bin/env python3
"""Compare file-driven supervised RTL trace against PCNetNLayer."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "python_rtl")
from pc_network import PCNetNLayer  # noqa: E402


def load_dataset(path: str, k2: int, k0: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    expected_cols = k2 + k0
    if data.shape[1] != expected_cols:
        raise ValueError(f"{path} has {data.shape[1]} columns, expected {expected_cols}")
    return data[:, :k2], data[:, k2:]


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


def eval_mse(net: PCNetNLayer, x: np.ndarray, y: np.ndarray, gamma: float, eval_ticks: int) -> float:
    net.set_rates(alpha=0.0, gamma=gamma)
    acc = 0.0
    for xs, ys in zip(x, y):
        for _ in range(eval_ticks):
            net.tick_parallel(xs, None, clamp_top=True, clamp_bottom=False)
        diff = net.x0 - ys
        acc += float(np.dot(diff, diff)) / y.shape[1]
    return acc / len(x)


def expected_rows(args) -> list[tuple[int, float]]:
    x, y = load_dataset(args.data, args.k2, args.k0)
    if len(x) != args.samples:
        raise ValueError(f"{args.data} has {len(x)} rows, expected {args.samples}")
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
    parser.add_argument("--csv", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--k0", type=int, required=True)
    parser.add_argument("--k1", type=int, required=True)
    parser.add_argument("--k2", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--infer-ticks", type=int, default=10)
    parser.add_argument("--learn-ticks", type=int, default=2)
    parser.add_argument("--eval-ticks", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--gamma", type=float, default=0.06)
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
    print("PASS file-driven supervised RTL learning curve matches PCNetNLayer")


if __name__ == "__main__":
    main()
