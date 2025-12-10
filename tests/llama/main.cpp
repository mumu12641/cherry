#include <chrono>
#include <iostream>
extern "C" void host();
int             main()
{
    std::cout << "Running kernel..." << std::endl;

    // 1. 获取开始时间点
    auto start = std::chrono::high_resolution_clock::now();

    // 2. 运行 host 函数
    host();

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

