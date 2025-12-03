set(LLVM_TARGET_DEFINITIONS CherryInterface.td)
mlir_tablegen(CherryInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(CherryInterface.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(CherryDialectInterfacesIncGen)

