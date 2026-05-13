#!/usr/bin/env python3
"""Check NVIDIA/PyTorch CUDA availability for experiment runs."""
from __future__ import annotations

import subprocess
import sys


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    out = (proc.stdout + proc.stderr).strip()
    return proc.returncode, out


def main() -> int:
    code, out = run(["nvidia-smi"])
    print("=== nvidia-smi ===")
    print(out or f"nvidia-smi exited with code {code}")
    if code != 0:
        return code

    print("\n=== torch cuda ===")
    try:
        import torch
    except ImportError as exc:
        print(f"PyTorch import failed: {exc}")
        return 1

    print(f"torch: {torch.__version__}")
    print(f"torch cuda build: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"device count: {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        return 1

    for idx in range(torch.cuda.device_count()):
        print(f"device {idx}: {torch.cuda.get_device_name(idx)}")

    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"matmul smoke: shape={tuple(y.shape)} device={y.device}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
