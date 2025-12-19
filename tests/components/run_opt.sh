#!/bin/bash
/home/nx/ycy/pb/cherry/build/core/src/Tools/cherry-opt --cherry-lowering-pipeline  --mlir-print-ir-after-all /home/nx/ycy/pb/cherry/tests/components/test_matmul.mlir -o ./test_matmul_output.mlir
