#!/usr/bin/env python3
"""Check pc_layer tile contract traces against a Python oracle."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def f32(x: float) -> np.float32:
    return np.float32(x)


def x_value(j: int) -> np.float32:
    return f32(0.01 * float((j % 23) - 11))


def back_value(r: int, i: int) -> np.float32:
    return f32(0.002 * float(((r * 7 + i * 3) % 19) - 9))


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


def step(state: dict[str, np.ndarray], x_up: np.ndarray, back_down: np.ndarray,
         act: str) -> dict[str, float]:
    w = state["w"].copy()
    bias = state["bias"].copy()
    x = state["x"].copy()
    alpha = f32(0.05)
    gamma = f32(0.10)

    phi_xup, _ = phi_pair(x_up, act)
    mu = np.zeros(w.shape[0], dtype=np.float32)
    for i in range(w.shape[0]):
      acc = f32(0.0)
      for j in range(w.shape[1]):
          acc = f32(acc + f32(w[i, j] * phi_xup[j]))
      mu[i] = f32(acc + bias[i])

    eps = (x - mu).astype(np.float32)
    _, phi_prime = phi_pair(x, act)
    back_eff = np.zeros(w.shape[0], dtype=np.float32)
    for i in range(w.shape[0]):
        acc = f32(0.0)
        for r in range(back_down.shape[0]):
            acc = f32(acc + back_down[r, i])
        back_eff[i] = f32(phi_prime[i] * acc)

    back_kn = (w * eps[:, np.newaxis]).astype(np.float32)
    for i in range(w.shape[0]):
        mi = f32(alpha * eps[i])
        for j in range(w.shape[1]):
            w[i, j] = f32(w[i, j] + f32(mi * phi_xup[j]))
        bias[i] = f32(bias[i] + mi)
        x[i] = f32(x[i] + f32(gamma * f32(back_eff[i] - eps[i])))

    state["w"] = w
    state["bias"] = bias
    state["x"] = x
    return snapshot(state, back_kn)


def snapshot(state: dict[str, np.ndarray], back_kn: np.ndarray) -> dict[str, float]:
    k = state["w"].shape[0]
    n = state["w"].shape[1]
    return {
        "x0": float(state["x"][0]),
        "x_last": float(state["x"][k - 1]),
        "back00": float(back_kn[0, 0]),
        "back0_last": float(back_kn[0, n - 1]),
        "backlast0": float(back_kn[k - 1, 0]),
        "backlast_last": float(back_kn[k - 1, n - 1]),
        "backnk00": float(back_kn.T[0, 0]),
        "backnk_last_last": float(back_kn.T[n - 1, k - 1]),
        "theta00": float(state["w"][0, 0]),
        "theta_last": float(state["w"][k - 1, n - 1]),
        "bias0": float(state["bias"][0]),
        "bias_last": float(state["bias"][k - 1]),
    }


def expected_rows(k: int, n: int, m: int, act: str) -> list[dict[str, float]]:
    state = {
        "w": np.zeros((k, n), dtype=np.float32),
        "bias": np.zeros(k, dtype=np.float32),
        "x": np.zeros(k, dtype=np.float32),
    }
    x_up = np.asarray([x_value(j) for j in range(n)], dtype=np.float32)
    back_down = np.asarray([[back_value(r, i) for i in range(k)] for r in range(m)], dtype=np.float32)
    rows = [step(state, x_up, back_down, act)]
    rows.append(step(state, f32(-0.5) * x_up, f32(-0.25) * back_down, act))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--act", choices=["linear", "relu", "tanh", "sigmoid"], default="linear")
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    with Path(args.csv).open(newline="") as f:
        got_rows = list(csv.DictReader(f))
    want_rows = expected_rows(args.k, args.n, args.m, args.act)

    fields = [
        "x0", "x_last", "back00", "back0_last", "backlast0", "backlast_last",
        "backnk00", "backnk_last_last", "theta00", "theta_last", "bias0", "bias_last",
    ]
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
    print("PASS layer tile RTL matches Python oracle")


if __name__ == "__main__":
    main()
