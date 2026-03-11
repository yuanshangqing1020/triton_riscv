
#!/bin/bash
set -e

# --- Configuration ---
# 1. Path to mlir-translate
# Now we use the one from our custom RISC-V LLVM build
MLIR_TRANSLATE="mlir-translate"

# 2. RISC-V LLVM Environment
export RISCV_HOME=/opt/riscv_llvm
export PATH=$RISCV_HOME/bin:$PATH

# --- Compilation Steps ---

echo "[Step 1] Converting MLIR (LLVM Dialect) to LLVM IR (.ll)"
$MLIR_TRANSLATE --mlir-to-llvmir test_kernel.mlir -o test_kernel.ll
echo "Generated: test_kernel.ll"

echo "[Step 2] Compiling LLVM IR to Object File (.o)"
llc -march=riscv64 -mattr=+m,+f,+d,+c,+v \
    -filetype=obj \
    -relocation-model=pic \
    test_kernel.ll -o test_kernel.o
echo "Generated: test_kernel.o"

echo "[Step 3] Linking Object File to Shared Library (.so)"
riscv64-unknown-linux-gnu-gcc -shared -o test_kernel.so test_kernel.o
echo "Success! Generated: test_kernel.so"

# Verify file existence
ls -l test_kernel.so
