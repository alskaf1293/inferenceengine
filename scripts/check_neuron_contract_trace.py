#!/usr/bin/env python3
"""Check parameterized neuron contract traces against a Python oracle."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def f32(x: float) -> np.float32:
    return np.float32(x)


def theta_value(j: int) -> np.float32:
    return f32(0.001 * float((j % 17) - 8))


def x_value(j: int) -> np.float32:
    return f32(0.01 * float((j % 23) - 11))


def back_value(j: int) -> np.float32:
    return f32(0.002 * float((j % 19) - 9))


def phi_pair(x: np.ndarray, act: str) -> tuple[np.ndarray, np.ndarray]:
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
    raise ValueError(f"unknown act: {act}")


def step(state: dict[str, np.ndarray | np.float32], x_vec: np.ndarray, back_in: np.ndarray,
         act: str, x_set: bool, clamp_hard: bool) -> dict[str, np.float32]:
    theta = np.asarray(state["theta"], dtype=np.float32).copy()
    bias = f32(float(state["bias"]))
    x = f32(float(state["x"]))
    alpha = f32(0.05)
    gamma = f32(0.10)
    x_obs = f32(0.6)

    phi_xup, _ = phi_pair(x_vec, act)
    x_eff = x_obs if x_set else x
    mu = f32(0.0)
    for t, xu in zip(theta, phi_xup):
        mu = f32(mu + f32(t * xu))
    mu = f32(mu + bias)
    eps = f32(x_eff - mu)

    backsum = f32(0.0)
    for b in back_in:
        backsum = f32(backsum + f32(b))
    _, phi_prime = phi_pair(np.asarray([x_eff], dtype=np.float32), act)
    back_eff = f32(phi_prime[0] * backsum)

    back_vec = np.asarray([f32(t * eps) for t in theta], dtype=np.float32)
    for j in range(len(theta)):
        theta[j] = f32(theta[j] + f32(f32(alpha * eps) * phi_xup[j]))
    bias = f32(bias + f32(alpha * eps))
    x = x_obs if (clamp_hard and x_set) else f32(x + f32(gamma * f32(back_eff - eps)))

    state["theta"] = theta
    state["bias"] = bias
    state["x"] = x
    return {
        "mu": mu,
        "eps": eps,
        "backsum": backsum,
        "back0": back_vec[0],
        "back_last": back_vec[-1],
        "theta0": theta[0],
        "theta_last": theta[-1],
        "bias": bias,
        "x_state": x,
        "x_i": x_eff,
    }


def expected_rows(n: int, m: int, act: str, x_set: bool, clamp_hard: bool) -> list[dict[str, np.float32]]:
    state: dict[str, np.ndarray | np.float32] = {
        "theta": np.asarray([theta_value(j) for j in range(n)], dtype=np.float32),
        "bias": f32(0.1),
        "x": f32(0.3),
    }
    rows = []
    x_vec = np.asarray([x_value(j) for j in range(n)], dtype=np.float32)
    back_in = np.asarray([back_value(j) for j in range(m)], dtype=np.float32)
    rows.append(step(state, x_vec, back_in, act, x_set, clamp_hard))
    rows.append(step(state, f32(-0.5) * x_vec, f32(-0.25) * back_in, act, x_set, clamp_hard))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--act", choices=["linear", "relu", "tanh", "sigmoid"], default="linear")
    parser.add_argument("--x-set", action="store_true")
    parser.add_argument("--clamp-hard", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    with Path(args.csv).open(newline="") as f:
        got_rows = list(csv.DictReader(f))
    want_rows = expected_rows(args.n, args.m, args.act, args.x_set, args.clamp_hard)

    fields = ["mu", "eps", "backsum", "back0", "back_last", "theta0", "theta_last", "bias", "x_state"]
    failures = []
    max_abs = 0.0
    for idx, (got, want) in enumerate(zip(got_rows, want_rows), start=1):
        for field in fields:
            g = float(got[field])
            w = float(want[field])
            err = abs(g - w)
            max_abs = max(max_abs, err)
            if err > args.tol:
                failures.append((idx, field, g, w, err))

    print(f"rows={len(got_rows)} max_abs_error={max_abs:.9g} tol={args.tol:g}")
    if len(got_rows) != len(want_rows):
        print(f"FAIL row count got={len(got_rows)} expected={len(want_rows)}")
        raise SystemExit(1)
    if failures:
        for tick, field, got, want, err in failures[:40]:
            print(f"FAIL tick={tick} field={field} got={got:.9g} expected={want:.9g} abs={err:.9g}")
        if len(failures) > 40:
            print(f"... {len(failures) - 40} more failures")
        raise SystemExit(1)
    print("PASS neuron contract RTL matches Python oracle")


if __name__ == "__main__":
    main()
