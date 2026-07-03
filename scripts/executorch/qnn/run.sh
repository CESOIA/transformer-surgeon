#!/usr/bin/env bash
# Launcher for exporter_function_test.py.
# Sets LD_LIBRARY_PATH before Python starts (glibc caches it at process init,
# so os.environ changes inside Python are too late for dlopen to see them).
# After running `conda activate py310-executorch-qnn` these vars are set
# correctly by the conda env; this script is a fallback for the current session.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

QNN_LIB="/workspace/qairt/2.37.0.250724/lib/x86_64-linux-clang"
NDK_LIB="/workspace/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/lib"

export LD_LIBRARY_PATH="${QNN_LIB}:${NDK_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec python3 "${SCRIPT_DIR}/exporter_function_test.py" "$@"
