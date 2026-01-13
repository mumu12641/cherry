#!/bin/bash

# ================= 配置区域 =================
# 项目根目录 (根据你的路径 /home/nx/ycy/pb/cherry 推断)
PROJECT_ROOT="/home/nx/ycy/pb/cherry"
BUILD_DIR="${PROJECT_ROOT}/build"

# 工具路径
CHERRY_BIN="${BUILD_DIR}/core/cherry"
CLANG_BIN="${BUILD_DIR}/third_party/llvm-project/llvm/bin/clang++"

# 库路径
LLVM_LIB_DIR="${BUILD_DIR}/third_party/llvm-project/llvm/lib"
RUNTIME_LIB_DIR="${BUILD_DIR}/runtime"

# ===========================================

# 错误处理：任何命令失败则立即退出
set -e

# 1. 检查参数
if [ "$#" -lt 1 ]; then
    echo "❌ Usage: $0 <test_name_without_extension> [driver_cpp_path]"
    echo "Example: $0 test_matmul"
    exit 1
fi

TEST_NAME=$1

# 构建文件路径
INPUT_MLIR="${PROJECT_ROOT}/tests/components/${TEST_NAME}.mlir"
OUTPUT_LL="output.ll"
OUTPUT_BIN="${TEST_NAME}.out"

# 2. 检查输入文件是否存在
if [ ! -f "$INPUT_MLIR" ]; then
    echo "❌ Error: Input file not found: $INPUT_MLIR"
    exit 1
fi

echo "=========================================="
echo "🧪 Test: $TEST_NAME"
echo "📄 Input: $INPUT_MLIR"
echo "=========================================="

# 3. 运行 Cherry (MLIR -> LLVM IR)
echo -e "\n🔨 [1/3] Running Cherry Compiler..."
$CHERRY_BIN "$INPUT_MLIR"

# 检查 Cherry 是否成功生成了 .ll 文件
if [ ! -f "$OUTPUT_LL" ]; then
    echo "❌ Error: Expected output file '$OUTPUT_LL' was not generated."
    exit 1
fi

# 4. 运行 Clang++ (Link -> Executable)
echo -e "\n🔧 [2/3] Compiling with Clang++..."
$CLANG_BIN "$OUTPUT_LL" \
    -o "$OUTPUT_BIN" \
    -O3 \
    -L"$LLVM_LIB_DIR" \
    -lmlir_c_runner_utils -lmlir_runner_utils \
    -Wl,-rpath,"$LLVM_LIB_DIR" \
    -L"$RUNTIME_LIB_DIR" \
    -lcherry_runtime

# 5. 运行生成的可执行文件
echo -e "\n🚀 [3/3] Running Executable..."
echo "------------------------------------------"
./"$OUTPUT_BIN"
echo "------------------------------------------"
echo "✅ Test Finished."

# 可选：清理中间文件 (如果需要保留，请注释掉下面这行)
# rm "$OUTPUT_LL" "$OUTPUT_BIN"
