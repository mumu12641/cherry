#!/bin/bash
# /home/nx/ycy/pb/cherry/build/core/src/Tools/cherry-opt --cherry-lowering-pipeline  --mlir-print-ir-after-all /home/nx/ycy/pb/cherry/tests/llama/cherry_python.mlir -o ./test_cherry_python_output.mlir
export PATH=$PATH:/home/nx/ycy/pb/cherry/build/core/src/Tools:/home/nx/ycy/pb/cherry/build/core:/home/nx/ycy/pb/cherry/build/third_party/llvm-project/llvm/bin
/home/nx/ycy/pb/cherry/build/core/src/Tools/cherry-opt \
--cherry-lowering-pipeline  \
--mlir-print-ir-after-all \
test_matmul.mlir
