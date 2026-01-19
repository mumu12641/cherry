#!/bin/bash
export PATH=$PATH:/home/nx/ycy/pb/cherry/build/core/src/Tools:/home/nx/ycy/pb/cherry/build/core:/home/nx/ycy/pb/cherry/build/third_party/llvm-project/llvm/bin
cherry-opt \
./test_llama.mlir \
--cherry-lowering-pipeline \
--mlir-print-ir-after-all 2>&1 | tee run_llama_opt.log