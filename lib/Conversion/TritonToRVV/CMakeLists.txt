add_triton_library(TritonToRVV
    TritonToRVVPass.cpp

    DEPENDS
    TritonToRVVPassIncGen

    LINK_LIBS PUBLIC
    MLIRVectorDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRFuncDialect
    TritonIR
)
