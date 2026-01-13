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
  MLIRSCFDialect
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
  
  MLIRVectorDialect               # Vector dialect
  MLIRVectorTransforms            # Vector transforms
  MLIRVectorToLLVM                # Vector to LLVM conversion patterns
)

