include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(src)

add_library(CherryCore INTERFACE)
target_link_libraries(CherryCore INTERFACE
    CherryDialect
    CherryPasses
    CherryToLinalg
    CherryPipelines
)
target_include_directories(CherryCore INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
)
add_executable(cherry "main.cpp")
target_link_libraries(cherry
        # CherryDialect
        # CherryPasses
        # CherryToLinalg
        # CherryPipelines
        CherryCore
        LLVMCore
        mlir_c_runner_utils
        mlir_runner_utils
        # cherry-opt
)
