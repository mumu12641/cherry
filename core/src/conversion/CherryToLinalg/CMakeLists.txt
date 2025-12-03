add_mlir_conversion_library(MLIRCherryToLinalg
  CherryToLinalg.cpp

  DEPENDS
  MLIRCherryConversionPassesIncGen

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  CherryDialect
  MLIRPass
  MLIRTransformUtils
  )

