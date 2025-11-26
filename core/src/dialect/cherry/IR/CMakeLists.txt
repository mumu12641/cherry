add_mlir_dialect_library(CherryDialect
    CherryDialect.cpp
    CherryTypes.cpp
    CherryOps.cpp

    DEPENDS
    CherryDialectIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRTensorDialect
)

