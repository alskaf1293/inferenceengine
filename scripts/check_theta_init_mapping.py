#!/usr/bin/env python3
"""Verify theta_init_pkg selected indices against the Python RNG recipe."""

from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "rtl" / "includes" / "theta_init_pkg.sv"


def f32_bits(x: np.float32) -> str:
    return f"{struct.unpack('>I', struct.pack('>f', float(x)))[0]:08X}"


def expected_tables() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    tables = {}
    for name, shape in [("THETA_L0", (8, 256)), ("THETA_L1", (256, 256)), ("THETA_L2", (256, 32))]:
        raw = rng.standard_normal(shape)
        tables[name] = (raw.astype(np.float32) * np.float32(0.1)).astype(np.float32)
    return tables


def parse_pkg_bits(text: str, table: str, i: int, j: int) -> str:
    row_match = re.search(rf"{i}:\s*'\{{([^}}]+)\}}", table_block(text, table), flags=re.S)
    if not row_match:
        raise ValueError(f"missing {table}[{i}]")
    values = re.findall(r"32'h([0-9A-Fa-f]{8})", row_match.group(1))
    if j >= len(values):
        raise ValueError(f"missing {table}[{i}][{j}]")
    return values[j].upper()


def table_block(text: str, table: str) -> str:
    start = text.index(f"{table} ")
    next_match = re.search(r"\n\s*localparam logic \[31:0\] THETA_L", text[start + 1:])
    end = len(text) if next_match is None else start + 1 + next_match.start()
    return text[start:end]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol", type=float, default=0.0, help="unused; bits are compared exactly")
    args = parser.parse_args()
    del args

    text = PKG.read_text()
    tables = expected_tables()
    probes = [
        ("THETA_L0", 0, 0),
        ("THETA_L0", 0, 127),
        ("THETA_L0", 0, 255),
        ("THETA_L0", 7, 0),
        ("THETA_L0", 7, 255),
        ("THETA_L1", 0, 0),
        ("THETA_L1", 0, 127),
        ("THETA_L1", 127, 127),
        ("THETA_L1", 255, 0),
        ("THETA_L1", 255, 255),
        ("THETA_L2", 0, 0),
        ("THETA_L2", 0, 31),
        ("THETA_L2", 127, 31),
        ("THETA_L2", 255, 0),
        ("THETA_L2", 255, 31),
    ]
    failures = []
    for table, i, j in probes:
        got = parse_pkg_bits(text, table, i, j)
        want = f32_bits(tables[table][i, j])
        if got != want:
            failures.append((table, i, j, got, want))
    print(f"probes={len(probes)} failures={len(failures)}")
    if failures:
        for table, i, j, got, want in failures:
            print(f"FAIL {table}[{i}][{j}] got=32'h{got} expected=32'h{want}")
        raise SystemExit(1)
    print("PASS theta_init_pkg selected indices match Python RNG mapping")


if __name__ == "__main__":
    main()
