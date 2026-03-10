#ifndef TRITON_CONVERSION_TRITONTORVV_PASSES
#define TRITON_CONVERSION_TRITONTORVV_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToRVV : Pass<"convert-triton-to-rvv", "mlir::ModuleOp"> {
    let summary = "Convert Triton to RISC-V Vector (RVV) compatible MLIR";
    let description = [{
        This pass converts Triton dialect operations directly to standard MLIR dialects
        (Vector, SCF, Arith, MemRef, Func) suitable for lowering to LLVM IR with RISC-V Vector extension.
        
        It performs:
        1. Type conversion (pointers to memrefs).
        2. Block strip-mining (mapping large block sizes to loops + vector ops).
        3. Op conversion (tt.load -> vector.transfer_read, tt.add -> arith.addf, etc.).
    }];

    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::scf::SCFDialect",
        "mlir::vector::VectorDialect",
        "mlir::memref::MemRefDialect",
        "mlir::func::FuncDialect",
        "mlir::triton::TritonDialect"
    ];

    // No options for now, maybe VLEN later
}

#endif
