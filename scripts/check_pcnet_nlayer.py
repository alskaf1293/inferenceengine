#!/usr/bin/env python3
"""Smoke checks for the general PCNetNLayer reference."""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "python_rtl")
from pc_network import PCNet3Layer, PCNetNLayer  # noqa: E402


def compare_three_layer_parallel() -> None:
    kwargs = dict(
        k_lut=[3, 4, 2],
        act_lut=["linear", "relu", "linear"],
        gamma=0.1,
        alpha=0.05,
        seed=0,
        rtl_init=True,
        gen_k_lut=[8, 64, 64],
        bias_init_scale=0.0,
    )
    net3 = PCNet3Layer(**kwargs)
    netn = PCNetNLayer(**kwargs, top_rtl_width=False)

    x_top = np.array([0.7, -0.2], dtype=np.float64)
    y_bottom = np.array([0.1, -0.3, 0.2], dtype=np.float64)
    for _ in range(8):
        net3.tick_parallel(x_top, y_bottom, clamp_top=True, clamp_bottom=True)
        netn.tick_parallel(x_top, y_bottom, clamp_top=True, clamp_bottom=True)

    checks = {
        "x0": (net3.x0, netn.x0),
        "layer0.W": (net3.layer0.W, netn.layers[0].W),
        "layer1.W": (net3.layer1.W, netn.layers[1].W),
        "layer2.x": (net3.layer2.x_state, netn.layers[2].x_state),
    }
    for name, (a, b) in checks.items():
        max_abs = float(np.max(np.abs(a - b))) if a.size else 0.0
        print(f"{name}: max_abs={max_abs:.9g}")
        if not np.allclose(a, b):
            raise AssertionError(f"PCNetNLayer diverged from PCNet3Layer for {name}")


def smoke_four_layer() -> None:
    net = PCNetNLayer(
        k_lut=[2, 3, 4, 2],
        act_lut=["linear", "relu", "relu", "linear"],
        gamma=0.05,
        alpha=0.02,
        seed=1,
        rtl_init=True,
    )
    x_top = np.array([0.2, -0.4], dtype=np.float64)
    y_bottom = np.array([0.1, -0.2], dtype=np.float64)
    for _ in range(4):
        net.tick_parallel(x_top, y_bottom, clamp_top=True, clamp_bottom=True)
    assert net.x0.shape == (2,)
    assert len(net.layers) == 4
    print(f"four_layer_x0={net.x0}")


def main() -> None:
    compare_three_layer_parallel()
    smoke_four_layer()
    print("PASS PCNetNLayer checks")


if __name__ == "__main__":
    main()
