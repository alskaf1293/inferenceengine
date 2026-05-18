#!/usr/bin/env python3
"""Compare the minimal 256-wide RTL smoke trace against PCNetNLayer."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "python_rtl")
from pc_network import PCNetNLayer  # noqa: E402


def x_top() -> np.ndarray:
    return np.asarray([0.05 * float(j - 11) for j in range(23)], dtype=np.float64)


def y_bottom() -> np.ndarray:
    return np.asarray([0.15], dtype=np.float64)


def make_net() -> PCNetNLayer:
    return PCNetNLayer(
        k_lut=[1, 256, 256, 23],
        act_lut=["linear", "relu", "relu", "linear"],
        gamma=0.05,
        alpha=0.002,
        seed=0,
        rtl_init=True,
        gen_k_lut=[8, 256, 256, 32],
        bias_init_scale=0.0,
        top_rtl_width=True,
    )


def mse(net: PCNetNLayer) -> float:
    d = net.x0 - y_bottom()
    return float(np.dot(d, d))


def expected_rows() -> list[tuple[str, float, float]]:
    net = make_net()
    xt = x_top()
    yb = y_bottom()

    net.set_rates(alpha=0.0, gamma=0.05)
    net.tick_parallel(xt, None, clamp_top=True, clamp_bottom=False)
    rows = [("eval_before", mse(net), float(net.x0[0]))]

    net.set_rates(alpha=0.0, gamma=0.05)
    net.tick_parallel(xt, yb, clamp_top=True, clamp_bottom=True)

    net.set_rates(alpha=0.002, gamma=0.05)
    net.tick_parallel(xt, yb, clamp_top=True, clamp_bottom=True)

    net.set_rates(alpha=0.0, gamma=0.05)
    net.tick_parallel(xt, None, clamp_top=True, clamp_bottom=False)
    rows.append(("eval_after", mse(net), float(net.x0[0])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    with Path(args.csv).open(newline="") as f:
        got = [(r["phase"], float(r["mse"]), float(r["x0"])) for r in csv.DictReader(f)]
    want = expected_rows()

    max_abs = 0.0
    failures = []
    for (gp, gm, gx), (wp, wm, wx) in zip(got, want):
        mse_err = abs(gm - wm)
        x0_err = abs(gx - wx)
        max_abs = max(max_abs, mse_err, x0_err)
        if gp != wp or mse_err > args.tol or x0_err > args.tol:
            failures.append((gp, gm, gx, wp, wm, wx, mse_err, x0_err))

    print(f"rows={len(got)} max_abs_error={max_abs:.9g} tol={args.tol:g}")
    if len(got) != len(want):
        print(f"FAIL row count got={len(got)} expected={len(want)}")
        raise SystemExit(1)
    if failures:
        for gp, gm, gx, wp, wm, wx, me, xe in failures:
            print(
                "FAIL "
                f"got_phase={gp} got_mse={gm:.9g} got_x0={gx:.9g} "
                f"expected_phase={wp} expected_mse={wm:.9g} expected_x0={wx:.9g} "
                f"mse_abs={me:.9g} x0_abs={xe:.9g}"
            )
        raise SystemExit(1)
    print("PASS minimal 256 RTL smoke matches PCNetNLayer")


if __name__ == "__main__":
    main()
