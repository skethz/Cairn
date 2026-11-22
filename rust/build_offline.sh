#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
: "${LIBTORCH:?Set LIBTORCH to a verified local PyTorch/LibTorch 2.11.0 native tree}"
: "${CUDA_HOME:?Set CUDA_HOME to the local CUDA 12.8 toolkit}"
if [ -d "$TASK_ROOT/toolchain/bin" ]; then export PATH="$TASK_ROOT/toolchain/bin:$PATH"; fi
if [ -d "$TASK_ROOT/cargo-home" ]; then export CARGO_HOME="$TASK_ROOT/cargo-home"; fi
export CARGO_BUILD_JOBS=4
export CXXFLAGS='-g0'
cd "$TASK_ROOT"
exec cargo build --release --locked --offline
