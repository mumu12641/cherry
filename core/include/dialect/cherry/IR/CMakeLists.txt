# set(LLVM_TARGET_DEFINITIONS CherryDialect.td)
# set(LLVM_TARGET_DEFINITIONS CherryTypes.td)

set(LLVM_TARGET_DEFINITIONS CherryOps.td)

mlir_tablegen(CherryDialect.h.inc --gen-dialect-decls -dialect=cherry)
mlir_tablegen(CherryDialect.cpp.inc --gen-dialect-defs -dialect=cherry)
mlir_tablegen(CherryTypes.h.inc -gen-typedef-decls -dialect=cherry)
mlir_tablegen(CherryTypes.cpp.inc -gen-typedef-defs -dialect=cherry)
mlir_tablegen(CherryOps.h.inc -gen-op-decls -dialect=cherry)
mlir_tablegen(CherryOps.cpp.inc -gen-op-defs -dialect=cherry)

add_public_tablegen_target(CherryDialectIncGen)


