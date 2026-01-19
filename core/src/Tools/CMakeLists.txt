set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}

  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR

  MLIRToLLVMIRTranslationRegistration
  CherryDialect
  CherryPipelines
  CherryToLinalg
  CherryGPU
  CherryDialectInterfaces
  CherryPasses
  MLIRCAPIDebug
)
add_mlir_tool(cherry-opt
    CherryOpt.cpp

  DEPENDS
  ${LIBS}
)

target_link_libraries("cherry-opt" PRIVATE ${LIBS})
llvm_update_compile_flags(cherry-opt)

mlir_check_all_link_libraries(cherry-opt)
export_executable_symbols_for_plugins(cherry-opt)
