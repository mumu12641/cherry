#!/bin/bash

mlir-opt llama/llama.mlir 
-one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map" -canonicalize \
-convert-linalg-to-parallel-loops \
-canonicalize -gpu-map-parallel-loops  -convert-parallel-loops-to-gpu -lower-affine -o llama/llama_lower.mlir 
# cherry-opt llama/llama_lower.mlir --cherry-insert-host-register -o llama/llama_lower.mlir



# LLVM_BUILD="/home/nx/ycy/pb/cherry/build/third_party/llvm-project/llvm"

# $LLVM_BUILD/bin/mlir-opt llama/lower.mlir \
#     -gpu-lower-to-nvvm-pipeline="cubin-format=fatbin" \
#     -gpu-to-llvm \
#     -convert-scf-to-cf \
#     -convert-arith-to-llvm \
#     -finalize-memref-to-llvm \
#     -convert-func-to-llvm \
#     -reconcile-unrealized-casts \
#     | $LLVM_BUILD/bin/mlir-translate --mlir-to-llvmir > llama/output.ll
    
# $LLVM_BUILD/bin/clang llama/output.ll -o llama/my_gpu_app \
#         -O3 \
#         -L${LLVM_BUILD}/lib \
#         -lmlir_cuda_runtime \
#         -lmlir_runner_utils \
#         -lmlir_c_runner_utils \
#         -Wl,-rpath,${LLVM_BUILD}/lib \
#         -lm -ldl -lpthread
        
# -lower-affine -gpu-decompose-memrefs -expand-strided-metadata -normalize-memrefs -convert-gpu-to-nvvm="index-bitwidth=0 use-bare-ptr-memref-call-conv" \
# -nvvm-attach-target="chip=sm_60 features=+ptx70 O=3" -convert-nvvm-to-llvm -reconcile-unrealized-casts \
# -gpu-to-llvm="use-bare-pointers-for-host use-bare-pointers-for-kernels" \
# -o lower.mlir

# mlir-translate --mlir-to-llvmir lower.mlir -o lower.ll
# llc   -mtriple=nvptx64-nvidia-cuda   -mcpu=sm_60   -filetype=asm   lower.ll -o lower.ptx

# mlir-translate --mlir-to-llvmir kernel.mlir -o kernel.ll
# llc --march=nvptx64 --mcpu=sm_60 kernel.ll -o kernel.ptx