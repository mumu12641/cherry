#!/bin/bash

BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

BUILD_DIR="$SCRIPT_DIR/../build"
echo -e "${BLUE} 🔨 Starting Compile core.so${NC}"
cd "$BUILD_DIR"

ninja cherry
ninja cherry_py

CHERRY_BIN="$BUILD_DIR/core/cherry"
CLANG_BIN="$BUILD_DIR/third_party/llvm-project/llvm/bin/clang++"
LLVM_LIB="$BUILD_DIR/third_party/llvm-project/llvm/lib"
RUNTIME_LIB="$BUILD_DIR/runtime"

INPUT_MLIR="test.mlir"
OUTPUT_LL="output.ll"
OUTPUT_EXE="llama.out"

echo -e "${YELLOW} 🏃 Running py...${NC}"
cd "$SCRIPT_DIR"

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
python test.py

echo -e "${YELLOW} 🔨 Compiling MLIR to LLVM IR (Cherry)...${NC}"

$CHERRY_BIN "$INPUT_MLIR"

echo -e "${YELLOW} 🔧 Compiling and Linking with Clang++...${NC}"

$CLANG_BIN "$OUTPUT_LL" \
    -o "$OUTPUT_EXE" \
    -O3 -ffast-math \
    -L"$LLVM_LIB" \
    -lmlir_c_runner_utils -lmlir_runner_utils \
    -Wl,-rpath,"$LLVM_LIB" \
    -L"$RUNTIME_LIB" \
    -lcherry_runtime

echo -e "${BLUE}   -> Built executable: $OUTPUT_EXE${NC}"

echo -e "${YELLOW} 🏃 Running Llama Inference...${NC}"
echo "-----------------------------------------"

run_start=$(date +%s%N)

./"$OUTPUT_EXE"

run_end=$(date +%s%N)
run_duration=$(( (run_end - run_start) / 1000000 ))

echo "-----------------------------------------"
echo -e "${BLUE}✅ Done! Execution finished in ${run_duration}ms.${NC}"
