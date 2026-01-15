#!/bin/bash
cherry-opt \
./test_llama.mlir \
--cherry-lowering-pipeline \
--mlir-print-ir-after-all 2>&1 | tee run_llama_opt.log