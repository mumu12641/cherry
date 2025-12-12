#!/bin/bash
../../build/core/cherry /home/nx/ycy/pb/cherry/tests/llama/cherry.mlir
../../build/third_party/llvm-project/llvm/bin/clang++ main.cpp cherry_output.ll -o llama.out -O3 -L../../build/third_party/llvm-project/llvm/lib -lmlir_c_runner_utils -lmlir_runner_utils -Wl,-rpath, ../../build/third_party/llvm-project/llvm/lib -L/home/nx/ycy/pb/cherry/build/runtime -lcherry_runtime
./llama.out