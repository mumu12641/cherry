add_mlir_conversion_library(CherryGPU
  InsertHostRegister.cpp

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
  MLIRBufferizationDialect
  MLIRMemRefDialect
  CherryDialect
  MLIRPass
  MLIRIR      
  MLIRTransforms            
  MLIRTransformUtils
  
  MLIRGPUDialect
)

