#!/usr/bin/env bash
# Install the Qualcomm AI Runtime (QNN) SDK that the active environment's ExecuTorch was built against, and make
# the environment use it: `conda activate` then sets QNN_SDK_ROOT and puts the SDK's x86 libraries (plus the libc++
# they need) on LD_LIBRARY_PATH; `conda deactivate` restores both. Exports with backend="qnn" need this.
#
#     conda activate cesoia                                # environment.yml
#     bash scripts/executorch/qnn/install_qnn_sdk.sh       # SDK in ~/qairt/<version>; or pass another root
#     conda deactivate && conda activate cesoia
#
# Downloads about 1.4 GB from Qualcomm (the SDK comes with Qualcomm's license) and, once, the 600 MB LLVM release
# whose libc++, libc++abi and libunwind ExecuTorch pairs with the SDK (only those three are unpacked).
# Running a QNN program still needs a Snapdragon device; this only enables exporting.

set -euo pipefail

: "${CONDA_PREFIX:?activate the conda environment first (conda activate cesoia)}"
VERSION="$(python -c 'from executorch.backends.qualcomm.scripts.download_qnn_sdk import QNN_VERSION; print(QNN_VERSION)')"
SDK_ROOT="${1:-$HOME/qairt}/$VERSION"

# The SDK (skipped when already there).
python -m executorch.backends.qualcomm.scripts.download_qnn_sdk --dst-folder "$SDK_ROOT"
test -f "$SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so"

# The SDK's x86 libraries link against LLVM's libc++, libc++abi and libunwind, which neither the SDK nor conda-forge
# ship (conda-forge's libunwind is another library): take them from the LLVM release ExecuTorch stages them from,
# into a folder of their own so that LD_LIBRARY_PATH gets only them.
LIBCXX_DIR="$CONDA_PREFIX/lib/qnn-libcxx"
LIBCXX_LIBS="libc++.so.1 libc++abi.so.1 libunwind.so.1"
if ! (cd "$LIBCXX_DIR" 2>/dev/null && ls $LIBCXX_LIBS >/dev/null 2>&1); then
    LLVM_URL="$(python -c 'from executorch.backends.qualcomm.scripts.download_qnn_sdk import LLVM_URL; print(LLVM_URL)')"
    TMP="$(mktemp -d)"
    trap 'rm -rf "$TMP"' EXIT
    curl -fL --retry 3 -o "$TMP/llvm.tar.xz" "$LLVM_URL"
    tar -xJf "$TMP/llvm.tar.xz" -C "$TMP" --wildcards "*/lib/x86_64-unknown-linux-gnu/libc++.so.1*" \
        "*/lib/x86_64-unknown-linux-gnu/libc++abi.so.1*" "*/lib/x86_64-unknown-linux-gnu/libunwind.so.1*"
    mkdir -p "$LIBCXX_DIR"
    for lib in $LIBCXX_LIBS; do
        rm -f "$LIBCXX_DIR/$lib"
        cp -L "$TMP"/*/lib/x86_64-unknown-linux-gnu/"$lib" "$LIBCXX_DIR/$lib"
    done
fi

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/qnn_sdk.sh" <<EOF
# Written by transformer-surgeon's scripts/executorch/qnn/install_qnn_sdk.sh: the QNN SDK for ExecuTorch.
export _QNN_SAVED_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export QNN_SDK_ROOT="$SDK_ROOT"
export LD_LIBRARY_PATH="\$QNN_SDK_ROOT/lib/x86_64-linux-clang:\$CONDA_PREFIX/lib/qnn-libcxx\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/qnn_sdk.sh" <<'EOF'
# Written by transformer-surgeon's scripts/executorch/qnn/install_qnn_sdk.sh: undoes activate.d/qnn_sdk.sh.
if [ -n "${_QNN_SAVED_LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="$_QNN_SAVED_LD_LIBRARY_PATH"
else
    unset LD_LIBRARY_PATH
fi
unset QNN_SDK_ROOT _QNN_SAVED_LD_LIBRARY_PATH
EOF

echo "QNN SDK $VERSION installed at $SDK_ROOT. Reactivate the environment to use it:"
echo "    conda deactivate && conda activate $(basename "$CONDA_PREFIX")"
