# GPU

跑通了 mlir 的 tests cuda 样例

要 ninja mlir-cpu-runner 和 mlir_cuda_runtime

mlir-opt -gpu-lower-to-nvvm-pipeline="cubin-format=fatbin" ./standard.mlir | mlir-cpu-runner --shared-libs=/home/nx/ycy/pb/cherry/build/third_party/llvm-project/llvm/lib/libmlir_cuda_runtime.so --shared-libs=/home/nx/ycy/pb/cherry/build/third_party/llvm-project/llvm/lib/libmlir_runner_utils.so  --shared-libs=/home/nx/ycy/pb/cherry/build/third_party/llvm-project/llvm/lib/libmlir_c_runner_utils.so --entry-point-result=void

mlir-opt -gpu-lower-to-nvvm-pipeline="cubin-format=fatbin" ./lower.mlir | mlir-cpu-runner --shared-libs=/home/nx/ycy/pb/cherry/build/third_party/llvm-project/llvm/lib/libmlir_cuda_runtime.so --shared-libs=/home/nx/ycy/pb/cherry/build/third_party/llvm-project/llvm/lib/libmlir_runner_utils.so  --shared-libs=/home/nx/ycy/pb/cherry/build/third_party/llvm-project/llvm/lib/libmlir_c_runner_utils.so --entry-point-result=void

module attributes {gpu.container_module} {
    gpu.module @kernels {
        gpu.func @hello() kernel {
            %0 = gpu.thread_id x
            %csti8 = arith.constant 2 : i8
            %cstf32 = arith.constant 3.0 : f32
            gpu.printf "Hello from %lld, %d, %f\n" %0, %csti8, %cstf32  : index, i8, f32
            gpu.return
        }
    }

    func.func @main() {
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        gpu.launch_func @kernels::@hello
            blocks in (%c1, %c1, %c1)
            threads in (%c2, %c1, %c1)
        return
    }
}
