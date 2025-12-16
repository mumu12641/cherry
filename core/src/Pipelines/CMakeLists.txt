add_mlir_library(CherryPipelines
  Pipelines.cpp
  
  DEPENDS
  LINK_LIBS PUBLIC
  
  CherryDialect 
  CherryPasses
  CherryToLinalg
  MLIRFuncAllExtensions
  
  MLIRLinalgTransforms
  
  MLIRMemRefDialect
  MLIRAffineDialect
  
  MLIRBufferizationTransforms
  MLIRBufferizationToMemRef
  
  MLIRSCFToControlFlow
  MLIRMemRefToLLVM
  MLIRMathToLLVM
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRLLVMDialect
  
  MLIRAffineToStandard
  MLIRTargetLLVM
  MLIRTargetLLVMIRExport
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRIndexToLLVM
)

