add_mlir_conversion_library(CherryToLinalg
  CherryToLinalg.cpp
  LinalgTiling.cpp

  DEPENDS
  CherryConversionPassesIncGen

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRTensorDialect
  CherryDialect
  MLIRPass
  MLIRIR      
  MLIRTransforms            
  MLIRTransformUtils
)

