#!/bin/bash
/home/nx/ycy/pb/cherry/build/core/cherry /home/nx/ycy/pb/cherry/tests/llama/cherry.mlir
/home/nx/ycy/pb/cherry/build/third_party/llvm-project/llvm/bin/clang++ /home/nx/ycy/pb/cherry/tests/llama/main.cpp /home/nx/ycy/pb/cherry/tests/llama/cherry_output.ll -o llama -O3 -L/home/nx/ycy/pb/cherry/build/third_party/llvm-project/llvm/lib -lmlir_c_runner_utils -lmlir_runner_utils -Wl,-rpath, /home/nx/ycy/pb/cherry/build/third_party/llvm-project/llvm/lib
./llama