#!/bin/bash

mlir-opt test.mlir -one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map" -canonicalize \
-convert-linalg-to-parallel-loops -canonicalize -gpu-map-parallel-loops  -convert-parallel-loops-to-gpu -lower-affine -o lower.mlir
cherry-opt lower.mlir --cherry-insert-host-register -o cherry_lower.mlir

LLVM_BUILD="/home/nx/ycy/pb/cherry/build/third_party/llvm-project/llvm"

$LLVM_BUILD/bin/mlir-opt cherry_lower.mlir \
    -gpu-lower-to-nvvm-pipeline="cubin-format=fatbin" \
    -gpu-to-llvm \
    -convert-scf-to-cf \
    -convert-arith-to-llvm \
    -finalize-memref-to-llvm \
    -convert-func-to-llvm \
    -convert-index-to-llvm \
    --convert-arith-to-llvm \
    --convert-math-to-llvm \
    -reconcile-unrealized-casts \
    | $LLVM_BUILD/bin/mlir-translate --mlir-to-llvmir > output.ll

$LLVM_BUILD/bin/clang++ output.ll -o my_gpu_app \
        -O3 \
        -L${LLVM_BUILD}/lib \
        -lmlir_cuda_runtime \
        -lmlir_runner_utils \
        -lmlir_c_runner_utils \
        -Wl,-rpath,${LLVM_BUILD}/lib \
        -lm -ldl -lpthread
