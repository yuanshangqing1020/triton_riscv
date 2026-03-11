#!/bin/bash
set -e

# Set paths
TRITON_OPT="build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt"
LLVM_BIN="/home/jichao.dong/.triton/llvm/llvm-20902f0b-ubuntu-x64/bin"
LLC="$LLVM_BIN/llc"

# Input/Output files
INPUT_MLIR=$1
BASENAME=$(basename "$INPUT_MLIR" .mlir)
LLVM_IR="${BASENAME}.ll"
ASM_FILE="${BASENAME}.s"
OBJ_FILE="${BASENAME}.o"

if [ -z "$INPUT_MLIR" ]; then
    echo "Usage: $0 <input.mlir>"
    exit 1
fi

echo "[1/4] Lowering Triton MLIR to LLVM IR..."
$TRITON_OPT "$INPUT_MLIR" \
    -convert-triton-to-rvv \
    -convert-vector-to-llvm \
    -convert-arith-to-llvm \
    -finalize-memref-to-llvm \
    -convert-func-to-llvm \
    -reconcile-unrealized-casts \
    -o "$LLVM_IR"
echo "Generated LLVM IR: $LLVM_IR"

# Check if LLC supports RISC-V
if $LLC --version | grep -q "riscv64"; then
    echo "[2/4] Compiling LLVM IR to Assembly (RISC-V Vector)..."
    $LLC -O3 -march=riscv64 -mattr=+m,+f,+d,+c,+v \
        -filetype=asm \
        "$LLVM_IR" -o "$ASM_FILE"

    echo "[3/4] Compiling Assembly to Object File..."
    $LLC -O3 -march=riscv64 -mattr=+m,+f,+d,+c,+v \
        -filetype=obj \
        -relocation-model=pic \
        "$LLVM_IR" -o "$OBJ_FILE"

    echo "Success! Generated: $LLVM_IR, $ASM_FILE, $OBJ_FILE"
else
    echo "WARNING: The current 'llc' does not support RISC-V target."
    echo "Current targets: $($LLC --version | grep "Registered Targets:" -A 10)"
    echo "Skipping compilation to machine code."
    echo "You can manually compile '$LLVM_IR' on a machine with RISC-V LLVM support using:"
    echo "  llc -march=riscv64 -mattr=+m,+f,+d,+c,+v -filetype=obj $LLVM_IR"
fi
