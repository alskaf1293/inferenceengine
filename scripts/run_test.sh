#!/usr/bin/env bash
set -euo pipefail

TB="${1:?usage: ./scripts/run_test.sh tb/<test>.sv <top_module> [verilator args...] -- [sim args...]}"
TOP="${2:?usage: ./scripts/run_test.sh tb/<test>.sv <top_module> [verilator args...] -- [sim args...]}"
shift 2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/check_env.sh"
cd "$ROOT"

MDIR="${VERILATOR_MDIR:-obj_dir}"
BIN="$ROOT/$MDIR/V${TOP}"

VERILATOR_EXTRA=()
SIM_EXTRA=()
SEEN_SEP=0

for arg in "$@"; do
  if [[ "$arg" == "--" ]]; then
    SEEN_SEP=1
    continue
  fi

  if [[ "$SEEN_SEP" -eq 0 ]]; then
    VERILATOR_EXTRA+=("$arg")
  else
    SIM_EXTRA+=("$arg")
  fi
done

COMMON_FLAGS=(
  -sv --binary --timing
  -Wno-fatal
  -Wno-TIMESCALEMOD -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-DECLFILENAME
  -Wno-GENUNNAMED -Wno-VARHIDDEN -Wno-PROCASSINIT -Wno-UNUSEDSIGNAL
  -Wno-UNOPTFLAT -Wno-UNSIGNED -Wno-UNDRIVEN -Wno-LATCH -Wno-WIDTHCONCAT
  -I"$ROOT/HardFloat/source" -I"$ROOT/HardFloat/source/8086-SSE"
  --Mdir "$ROOT/$MDIR"
)

if [[ -n "${VERILATOR_BUILD_JOBS:-}" ]]; then
  COMMON_FLAGS+=(--build-jobs "$VERILATOR_BUILD_JOBS")
fi

if [[ -n "${VERILATOR_VERILATE_JOBS:-}" ]]; then
  COMMON_FLAGS+=(--verilate-jobs "$VERILATOR_VERILATE_JOBS")
fi

if [[ -n "${VERILATOR_THREADS:-}" ]]; then
  COMMON_FLAGS+=(--threads "$VERILATOR_THREADS")
fi

if [[ -n "${VERILATOR_OUTPUT_SPLIT:-}" ]]; then
  COMMON_FLAGS+=(--output-split "$VERILATOR_OUTPUT_SPLIT")
fi

if [[ -n "${VERILATOR_OUTPUT_SPLIT_CFUNCS:-}" ]]; then
  COMMON_FLAGS+=(--output-split-cfuncs "$VERILATOR_OUTPUT_SPLIT_CFUNCS")
fi

COMMON_SRCS=(
  "$ROOT/HardFloat/source/HardFloat_primitives.v"
  "$ROOT/HardFloat/source/HardFloat_consts.vi"
  "$ROOT/HardFloat/source/HardFloat_localFuncs.vi"
  "$ROOT/HardFloat/source/HardFloat_rawFN.v"
  "$ROOT/HardFloat/source/addRecFN.v"
  "$ROOT/HardFloat/source/compareRecFN.v"
  "$ROOT/HardFloat/source/divSqrtRecFN_small.v"
  "$ROOT/HardFloat/source/fNToRecFN.v"
  "$ROOT/HardFloat/source/iNToRecFN.v"
  "$ROOT/HardFloat/source/isSigNaNRecFN.v"
  "$ROOT/HardFloat/source/mulAddRecFN.v"
  "$ROOT/HardFloat/source/mulRecFN.v"
  "$ROOT/HardFloat/source/recFNToFN.v"
  "$ROOT/HardFloat/source/recFNToIN.v"
  "$ROOT/HardFloat/source/recFNToRecFN.v"
  "$ROOT/HardFloat/source/8086-SSE/HardFloat_specialize.v"

  "$ROOT/rtl/hf_mac32.sv"
  "$ROOT/rtl/activation_relu32.sv"
  "$ROOT/rtl/nc_neuralcomputer.sv"
)

DPI="$ROOT/tb/dpi_casts.cc"

if [[ "${VERILATOR_REUSE_BINARY:-0}" != "1" || ! -x "$BIN" ]]; then
  verilator "${COMMON_FLAGS[@]}" \
    "${VERILATOR_EXTRA[@]}" \
    "${COMMON_SRCS[@]}" \
    "$ROOT/$TB" \
    --top-module "$TOP" \
    --exe "$DPI"
fi

"$BIN" "${SIM_EXTRA[@]}"
