include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(src)
add_executable(cherry "main.cpp")
target_link_libraries(cherry 
        CherryDialect 
        MLIRGPUDialect 
        MLIRFuncDialect 
        MLIRAnalysis
        MLIRCastInterfaces
        MLIRCallInterfaces
        MLIRFunctionInterfaces
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRTransforms)
