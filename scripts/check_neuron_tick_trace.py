#!/usr/bin/env python3
"""Compare tb_neuron_tick_trace RTL output against a small Python oracle."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def f32(x: float) -> np.float32:
    return np.float32(x)


def phi_pair(x, act: str):
    x = np.asarray(x, dtype=np.float32)
    if act == "linear":
        return x.astype(np.float32), np.ones_like(x, dtype=np.float32)
    if act == "relu":
        return np.maximum(x, f32(0.0)).astype(np.float32), (x > f32(0.0)).astype(np.float32)
    if act == "tanh":
        t = np.tanh(x).astype(np.float32)
        return t, (f32(1.0) - t * t).astype(np.float32)
    if act == "sigmoid":
        s = (f32(1.0) / (f32(1.0) + np.exp(-x))).astype(np.float32)
        return s, (s * (f32(1.0) - s)).astype(np.float32)
    raise ValueError(f"unknown activation: {act}")


def step(state, x_vec, back_in, alpha=f32(0.05), gamma=f32(0.10),
         act: str = "linear", x_set: bool = False, clamp_hard: bool = False, x_obs=f32(0.6)):
    theta = state["theta"].copy()
    bias = f32(state["bias"])
    x = f32(state["x"])

    phi_xup, _ = phi_pair(np.array([f32(x_vec[0]), f32(x_vec[1])], dtype=np.float32), act)
    x_eff = f32(x_obs) if x_set else x
    mu = f32(f32(theta[0] * phi_xup[0]) + f32(theta[1] * phi_xup[1]))
    mu = f32(mu + bias)
    eps = f32(x_eff - mu)
    backsum = f32(f32(back_in[0]) + f32(back_in[1]))
    _, phi_prime = phi_pair(np.array([x_eff], dtype=np.float32), act)
    back_eff = f32(phi_prime[0] * backsum)
    back0 = f32(theta[0] * eps)
    back1 = f32(theta[1] * eps)

    theta[0] = f32(theta[0] + f32(f32(alpha * eps) * phi_xup[0]))
    theta[1] = f32(theta[1] + f32(f32(alpha * eps) * phi_xup[1]))
    bias = f32(bias + f32(alpha * eps))
    x = f32(x_obs) if (clamp_hard and x_set) else f32(x + f32(gamma * f32(back_eff - eps)))

    state["theta"] = theta
    state["bias"] = bias
    state["x"] = x
    return {
        "mu": mu,
        "eps": eps,
        "backsum": backsum,
        "back0": back0,
        "back1": back1,
        "theta0": theta[0],
        "theta1": theta[1],
        "bias": bias,
        "x_state": x,
    }


def expected_rows(act: str, x_set: bool, clamp_hard: bool):
    state = {
        "theta": np.array([f32(0.25), f32(-0.40)], dtype=np.float32),
        "bias": f32(0.1),
        "x": f32(0.3),
    }
    inputs = [
        ([f32(0.7), f32(-0.2)], [f32(0.15), f32(-0.05)]),
        ([f32(-0.4), f32(0.9)], [f32(-0.10), f32(0.05)]),
    ]
    return [step(state, x_vec, back_in, act=act, x_set=x_set, clamp_hard=clamp_hard)
            for x_vec, back_in in inputs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="runs/neuron_tick_trace.csv")
    parser.add_argument("--tol", type=float, default=5e-6)
    parser.add_argument("--act", choices=["linear", "relu", "tanh", "sigmoid"], default="linear")
    parser.add_argument("--x-set", action="store_true")
    parser.add_argument("--clamp-hard", action="store_true")
    args = parser.parse_args()

    rows = []
    with Path(args.csv).open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    exp = expected_rows(args.act, args.x_set, args.clamp_hard)
    fields = ["mu", "eps", "backsum", "back0", "back1", "theta0", "theta1", "bias", "x_state"]
    max_abs = 0.0
    failures = []
    for idx, (got, want) in enumerate(zip(rows, exp), start=1):
        for field in fields:
            g = float(got[field])
            w = float(want[field])
            err = abs(g - w)
            max_abs = max(max_abs, err)
            if err > args.tol:
                failures.append((idx, field, g, w, err))

    print(f"rows={len(rows)} max_abs_error={max_abs:.9g} tol={args.tol:g}")
    if failures:
        for tick, field, got, want, err in failures:
            print(f"FAIL tick={tick} field={field} got={got:.9g} expected={want:.9g} abs={err:.9g}")
        raise SystemExit(1)
    print("PASS neuron tick RTL matches Python oracle")


if __name__ == "__main__":
    main()
