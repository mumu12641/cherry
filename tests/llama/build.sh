#!/bin/bash

# ================= 配置区域 =================
# 定义颜色，让日志更清晰
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 路径变量 (提取出来方便修改)
CHERRY_BIN="../../build/core/cherry"
CLANG_BIN="../../build/third_party/llvm-project/llvm/bin/clang++"
LLVM_LIB="../../build/third_party/llvm-project/llvm/lib"
RUNTIME_LIB="/home/nx/ycy/pb/cherry/build/runtime"

INPUT_MLIR="/home/nx/ycy/pb/cherry/tests/llama/cherry.mlir"
OUTPUT_LL="cherry_output.ll"
DRIVER_CPP="main.cpp"
OUTPUT_EXE="llama.out"

# ================= 脚本逻辑 =================

# 遇到错误立即停止
set -e

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}🚀 Starting Llama Build & Run Pipeline${NC}"
echo -e "${BLUE}=========================================${NC}"

# 1. 清理旧文件
if [ -f "$OUTPUT_EXE" ]; then
    rm "$OUTPUT_EXE"
fi

# 2. 运行 Cherry 编译器
echo -e "${YELLOW}[1/3] 🔨 Compiling MLIR to LLVM IR (Cherry)...${NC}"
start_time=$(date +%s)

$CHERRY_BIN "$INPUT_MLIR"

if [ ! -f "$OUTPUT_LL" ]; then
    echo -e "${RED}❌ Error: $OUTPUT_LL was not generated!${NC}"
    exit 1
fi
echo -e "${GREEN}   -> Generated $OUTPUT_LL${NC}"

# 3. 运行 Clang++ 编译链接
echo -e "${YELLOW}[2/3] 🔧 Compiling and Linking with Clang++...${NC}"

$CLANG_BIN "$DRIVER_CPP" "$OUTPUT_LL" \
    -o "$OUTPUT_EXE" \
    -O3 \
    -L"$LLVM_LIB" \
    -lmlir_c_runner_utils -lmlir_runner_utils \
    -Wl,-rpath,"$LLVM_LIB" \
    -L"$RUNTIME_LIB" \
    -lcherry_runtime

echo -e "${GREEN}   -> Built executable: $OUTPUT_EXE${NC}"

# 计算编译耗时
end_time=$(date +%s)
duration=$((end_time - start_time))
echo -e "${BLUE}   (Build took ${duration}s)${NC}"

# 4. 运行程序
echo -e "${YELLOW}[3/3] 🏃 Running Llama Inference...${NC}"
echo "-----------------------------------------"

# 记录运行开始时间
run_start=$(date +%s%N)

./"$OUTPUT_EXE"

run_end=$(date +%s%N)
# 计算毫秒
run_duration=$(( (run_end - run_start) / 1000000 ))

echo "-----------------------------------------"
echo -e "${GREEN}✅ Done! Execution finished in ${run_duration}ms.${NC}"
