#!/usr/bin/env python3
"""Check narrow/deep pc_network_nlayer scheduler traces."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "python_rtl")
from pc_network import PCNetNLayer  # noqa: E402


FIELDS = [
    "x0", "x1_0", "x1_mid", "x2_0", "x2_mid", "x3_0", "x3_mid",
    "x4_0", "back0_0_0", "l0_w00", "l1_w00", "l2_w00",
]


def x_top() -> np.ndarray:
    return np.asarray([0.03 * float((j % 11) - 5) for j in range(23)], dtype=np.float64)


def make_net(h: int) -> PCNetNLayer:
    net = PCNetNLayer(
        k_lut=[1, h, h, h, 23],
        act_lut=["linear", "relu", "relu", "relu", "linear"],
        gamma=0.10,
        alpha=0.05,
        seed=0,
        rtl_init=True,
        gen_k_lut=[8, 256, 256, 32, 23],
        bias_init_scale=0.0,
        top_rtl_width=True,
    )
    # RTL only routes nonzero presets through L2 for >3-layer networks; deeper
    # unsupported layers/top keep zero theta presets.
    net.layers[3].W.fill(0.0)
    net.layers[4].W.fill(0.0)
    return net


def snapshot(net: PCNetNLayer, tick: int, h: int) -> dict[str, float]:
    mid = h // 2
    return {
        "tick": float(tick),
        "x0": float(net.layers[0].x_state[0]),
        "x1_0": float(net.layers[1].x_state[0]),
        "x1_mid": float(net.layers[1].x_state[mid]),
        "x2_0": float(net.layers[2].x_state[0]),
        "x2_mid": float(net.layers[2].x_state[mid]),
        "x3_0": float(net.layers[3].x_state[0]),
        "x3_mid": float(net.layers[3].x_state[mid]),
        "x4_0": float(net.layers[4].x_state[0]),
        "back0_0_0": float(net.layers[0].back_nk[0, 0]),
        "l0_w00": float(net.layers[0].W[0, 0]),
        "l1_w00": float(net.layers[1].W[0, 0]),
        "l2_w00": float(net.layers[2].W[0, 0]),
    }


def expected_rows(h: int) -> list[dict[str, float]]:
    net = make_net(h)
    rows = []
    seq = [
        (0.05, True),
        (0.05, True),
        (0.0, False),
    ]
    for tick, (alpha, clamp_bottom) in enumerate(seq, start=1):
        net.set_rates(alpha=alpha, gamma=0.10)
        net.tick_parallel(x_top(), np.asarray([0.12]), clamp_top=True, clamp_bottom=clamp_bottom)
        rows.append(snapshot(net, tick, h))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    with Path(args.csv).open(newline="") as f:
        got_rows = list(csv.DictReader(f))
    want_rows = expected_rows(args.h)

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
    if len(got_rows) != len(want_rows):
        print(f"FAIL row count got={len(got_rows)} expected={len(want_rows)}")
        raise SystemExit(1)
    if failures:
        for tick, field, got, want, err in failures[:40]:
            print(f"FAIL tick={tick} field={field} got={got:.9g} expected={want:.9g} abs={err:.9g}")
        if len(failures) > 40:
            print(f"... {len(failures) - 40} more failures")
        raise SystemExit(1)
    print("PASS deep scheduler RTL matches PCNetNLayer")


if __name__ == "__main__":
    main()
