add_mlir_conversion_library(CherryToLinalg
  CherryToLinalg.cpp

  DEPENDS
  CherryConversionPassesIncGen

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRFuncDialect
  CherryDialect
  MLIRPass
  MLIRTransformUtils
  )

