include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(src)
add_executable(cherry "main.cpp")
target_link_libraries(cherry 
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
        
)
