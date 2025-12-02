
add_mlir_dialect_library(CherryPasses
  ShapeInferencePass.cpp

  DEPENDS
  CherryPassesIncGen

  LINK_LIBS PUBLIC
  CherryDialect
)
