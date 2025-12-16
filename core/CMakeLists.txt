include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(src)
add_executable(cherry "main.cpp")
target_link_libraries(cherry
        CherryDialect
        CherryPasses
        CherryToLinalg
        CherryPipelines
        LLVMCore
        mlir_c_runner_utils
        mlir_runner_utils
)
