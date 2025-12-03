set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CherryOpt)
add_public_tablegen_target(CherryPassesIncGen)
add_dependencies(mlir-headers CherryPassesIncGen)
