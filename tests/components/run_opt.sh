#!/bin/bash
/home/nx/ycy/pb/cherry/build/core/src/Tools/cherry-opt --convert-cherry-to-linalg --cherry-linalg-tiling --mlir-print-ir-after-all ./test_matmul.mlir -o ./test_matmul_output.mlir
