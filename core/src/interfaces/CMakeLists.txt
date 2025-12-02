add_mlir_library(CherryDialectInterfaces
    CherryInterface.cpp

    DEPENDS
    CherryDialectInterfacesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)

