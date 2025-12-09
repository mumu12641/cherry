#include <chrono>
#include <iostream>
// 声明 kernel.ll 中的 host 函数。
// 注意：虽然 host 返回一个巨大的结构体，但在 C++ 中调用时，
// 为了避免复杂的结构体定义和 ABI 问题，我们可以声明它返回 void。
// 在 x86-64 Linux 上，对于这种大结构体返回，编译器会自动处理隐藏参数，
// 只要我们不尝试读取返回值，直接调用通常是安全的。
extern "C" void host();
extern "C" void print_memref_f32(int64_t rank, void* descriptor)
{
    std::cout << "Unranked Memref base@ = " << descriptor << " rank = " << rank << " offset = " << 0
              << " sizes = [";

    // 解析描述符
    // Layout: { allocatedPtr, alignedPtr, offset, sizes[], strides[] }
    char*    descBytes  = static_cast<char*>(descriptor);
    float*   alignedPtr = *reinterpret_cast<float**>(descBytes + 8);     // offset 8
    int64_t  offset     = *reinterpret_cast<int64_t*>(descBytes + 16);   // offset 16
    int64_t* sizes      = reinterpret_cast<int64_t*>(descBytes + 24);    // offset 24
    int64_t* strides    = reinterpret_cast<int64_t*>(descBytes + 24 + rank * 8);

    // 打印 Sizes
    for (int i = 0; i < rank; ++i) {
        std::cout << sizes[i];
        if (i < rank - 1) std::cout << ", ";
    }
    std::cout << "] strides = [";

    // 打印 Strides
    for (int i = 0; i < rank; ++i) {
        std::cout << strides[i];
        if (i < rank - 1) std::cout << ", ";
    }
    std::cout << "] data = " << std::endl;

    // 打印数据 (仅针对 Rank 2 的特定实现，适配你的 kernel.ll)
    if (rank == 2) {
        std::cout << "[";
        for (int64_t i = 0; i < sizes[0]; ++i) {
            if (i > 0) std::cout << " ";
            std::cout << "[";
            for (int64_t j = 0; j < sizes[1]; ++j) {
                // 计算索引: offset + i*stride0 + j*stride1
                int64_t idx = offset + i * strides[0] + j * strides[1];
                std::cout << alignedPtr[idx];
                if (j < sizes[1] - 1) std::cout << ", ";
            }
            std::cout << "]";
            if (i < sizes[0] - 1) std::cout << "," << std::endl;
        }
        std::cout << "]" << std::endl;
    }
    else {
        std::cout << "(Printing for rank " << rank << " not fully implemented in this stub)"
                  << std::endl;
    }
}
int main()
{
    std::cout << "Running kernel..." << std::endl;

    // 1. 获取开始时间点
    auto start = std::chrono::high_resolution_clock::now();

    // 2. 运行 host 函数
    host();
    std::cout << "Kernel finished successfully." << std::endl;

    // 3. 获取结束时间点
    auto end = std::chrono::high_resolution_clock::now();

    // 4. 计算持续时间
    // 使用 duration_cast 转换为毫秒 (milliseconds) 或微秒 (microseconds)
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Kernel finished successfully." << std::endl;

    // 5. 打印时间
    std::cout << "Execution time: " << duration_ms.count() << " ms (" << duration_us.count()
              << " us)" << std::endl;

    return 0;
}

// /home/nx/ycy/pb/cherry/build/third_party/llvm-project/llvm/bin/clang++ /home/nx/ycy/pb/cherry/tests/llama/main.cpp /home/nx/ycy/pb/cherry/tests/llama/cherry_output.ll -o runner \
//   -Lthird_party/llvm-project/llvm/lib \
//   -lmlir_c_runner_utils \
//   -Wl,-rpath,$(pwd)/third_party/llvm-project/llvm/lib
