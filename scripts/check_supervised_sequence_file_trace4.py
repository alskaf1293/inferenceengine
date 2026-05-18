#!/usr/bin/env python3
"""Compare four-layer persistent RTL sequence trace against PCNetNLayer."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "python_rtl")
from pc_network import PCNetNLayer  # noqa: E402


def load_sequence(path: str, k3: int, k0: int, updates: int, samples_per_update: int):
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    expected = (updates * samples_per_update, k3 + k0)
    if data.shape != expected:
        raise ValueError(f"{path} has shape {data.shape}, expected {expected}")
    x = data[:, :k3].reshape(updates, samples_per_update, k3)
    y = data[:, k3:].reshape(updates, samples_per_update, k0)
    return x, y


def make_net(args) -> PCNetNLayer:
    return PCNetNLayer(
        k_lut=[args.k0, args.k1, args.k2, args.k3],
        act_lut=["linear", "relu", "relu", "linear"],
        gamma=args.gamma,
        alpha=args.alpha,
        seed=0,
        rtl_init=True,
        gen_k_lut=[8, 256, 256, 32],
        bias_init_scale=0.0,
        top_rtl_width=True,
    )


def eval_update(net: PCNetNLayer, x: np.ndarray, y: np.ndarray, gamma: float, eval_ticks: int) -> float:
    net.set_rates(alpha=0.0, gamma=gamma)
    acc = 0.0
    for xs, ys in zip(x, y):
        for _ in range(eval_ticks):
            net.tick_parallel(xs, None, clamp_top=True, clamp_bottom=False)
        diff = net.x0 - ys
        acc += float(np.dot(diff, diff)) / y.shape[1]
    return acc / len(x)


def expected_rows(args) -> list[tuple[int, float]]:
    x, y = load_sequence(args.data, args.k3, args.k0, args.updates, args.samples_per_update)
    net = make_net(args)
    rows = [(-1, eval_update(net, x[0], y[0], args.gamma, args.eval_ticks))]
    for update_idx in range(args.updates):
        for xs, ys in zip(x[update_idx], y[update_idx]):
            net.set_rates(alpha=0.0, gamma=args.gamma)
            for _ in range(args.infer_ticks):
                net.tick_parallel(xs, ys, clamp_top=True, clamp_bottom=True)
            net.set_rates(alpha=args.alpha, gamma=args.gamma)
            for _ in range(args.learn_ticks):
                net.tick_parallel(xs, ys, clamp_top=True, clamp_bottom=True)
        rows.append((update_idx, eval_update(net, x[update_idx], y[update_idx], args.gamma, args.eval_ticks)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--k0", type=int, default=1)
    parser.add_argument("--k1", type=int, default=16)
    parser.add_argument("--k2", type=int, default=16)
    parser.add_argument("--k3", type=int, default=23)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--samples-per-update", type=int, default=6)
    parser.add_argument("--infer-ticks", type=int, default=8)
    parser.add_argument("--learn-ticks", type=int, default=2)
    parser.add_argument("--eval-ticks", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--tol", type=float, default=5e-5)
    args = parser.parse_args()

    with Path(args.csv).open(newline="") as f:
        got = [(int(r["update"]), float(r["mse"])) for r in csv.DictReader(f)]
    want = expected_rows(args)

    max_abs = 0.0
    failures = []
    for (gu, gm), (wu, wm) in zip(got, want):
        err = abs(gm - wm)
        max_abs = max(max_abs, err)
        if gu != wu or err > args.tol:
            failures.append((gu, gm, wu, wm, err))

    print(f"rows={len(got)} max_abs_error={max_abs:.9g} tol={args.tol:g}")
    if failures:
        for gu, gm, wu, wm, err in failures:
            print(f"FAIL got_update={gu} got_mse={gm:.9g} expected_update={wu} expected_mse={wm:.9g} abs={err:.9g}")
        raise SystemExit(1)
    print("PASS four-layer persistent RTL learning curve matches PCNetNLayer")


if __name__ == "__main__":
    main()
