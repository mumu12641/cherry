#include <chrono>
#include <iomanip>
#include <iostream>

static std::chrono::high_resolution_clock::time_point g_start_time;

extern "C" {

void start()
{
    g_start_time = std::chrono::high_resolution_clock::now();
}

void end(int num)
{
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - g_start_time);
    auto duration_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - g_start_time);

    std::chrono::duration<double> duration_sec = end_time - g_start_time;
    double                        seconds      = duration_sec.count();

    double tokens_per_second = 0.0;
    if (seconds > 0.0) {
        tokens_per_second = num / seconds;
    }

    std::cout << "\n\n[Performance Metrics]" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Tokens processed : " << num << std::endl;
    std::cout << "Execution time   : " << duration_ms.count() << " ms (" << duration_us.count()
              << " us)" << std::endl;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Throughput       : " << tokens_per_second << " tokens/s" << std::endl;
    std::cout << "Latency          : " << (seconds * 1000.0 / num) << " ms/token" << std::endl;
    std::cout << "--------------------------------" << std::endl;
}
}
