find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
  OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYBIND11_FIND_RESULT
)


add_library(cherry_py MODULE bindings.cpp)

target_compile_options(cherry_py PRIVATE -frtti)

target_include_directories(cherry_py PRIVATE
  ${PYBIND11_INCLUDE_DIR}
  ${Python3_INCLUDE_DIRS}
)

target_link_libraries(cherry_py PRIVATE
  CherryCore
  MLIRIR
  cherry_runtime
)

set_target_properties(cherry_py PROPERTIES
    PREFIX ""
    OUTPUT_NAME "core"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cherry
)
