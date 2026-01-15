#!/bin/bash
cherry-opt \
./test_attention.mlir \
-linalg-generalize-named-ops \
-linalg-fuse-elementwise-ops \
--one-shot-bufferize \
-convert-linalg-to-loops \
--mlir-print-ir-after-all