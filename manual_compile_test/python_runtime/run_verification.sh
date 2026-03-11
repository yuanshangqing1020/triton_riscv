
#!/bin/bash
set -e

# Configuration
RISCV_HOME="/opt/riscv_llvm"
PATH="$RISCV_HOME/bin:$PATH"
MLIR_TRANSLATE="mlir-translate"
LLC="llc"
GCC="riscv64-unknown-linux-gnu-gcc"
SPIKE="spike"
PK="/opt/riscv_llvm/riscv64-unknown-elf/bin/pk64" # Use pk64 if available, otherwise pk

# Adjust PK path if needed
if [ ! -f "$PK" ]; then
    PK="/opt/riscv_llvm/riscv64-unknown-elf/bin/pk"
fi

echo "=== 1. Compile Kernel (MLIR -> SO) ==="
# Reuse the kernel from parent directory
cp ../test_kernel.mlir .

$MLIR_TRANSLATE --mlir-to-llvmir test_kernel.mlir -o test_kernel.ll
$LLC -march=riscv64 -mattr=+m,+f,+d,+c,+v -filetype=obj -relocation-model=pic test_kernel.ll -o test_kernel.o
$GCC -shared -o libkernel.so test_kernel.o
echo "Generated libkernel.so"

echo "=== 2. Compile Runner (C -> RISC-V Executable) ==="
# Link statically with the kernel object file
# Note: dlopen requires dynamic linking which is hard with baremetal PK.
# We switched runner.c to use static linking for verification.
$GCC -static -o runner runner.c test_kernel.o
echo "Generated runner (static)"

echo "=== 3. Run with Spike ==="
# Run the static binary directly with PK
$SPIKE --isa=rv64gcv $PK ./runner
