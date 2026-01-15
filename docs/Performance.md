# Performance

Stories110M baseline : 0.52 tokens/s
0.63 tokens/s 2026/1/13
2.01 tokens/s 2026/1/15 好像是把 fusion 做好了, 以及减少了很多 reshape 的操作。

下一步的优化方向应该是：

1、很多 op 例如 matmul op 先要 fill 一个 result tensor，然后进行 matmul
// 循环 1: 遍历整个 Tensor 做 Fill
%1 = scf.for ... { linalg.fill ... }

// 循环 2: 遍历整个 Tensor 做 Matmul
%2 = scf.for ... { linalg.matmul ... }

这意味着 CPU 先把数据从内存拉进 Cache 填 0，写回内存；然后再从内存拉进 Cache 做乘法。对于大 Tensor，这导致了 Cache Thrashing。

你应该做 Fusion（融合）： 把 Fill 和 Matmul 融合成一个循环嵌套。mlir::linalg::tileConsumerAndFuseProducers



这是一个非常标准的 MLIR 编译流程。虽然你的路径（Custom -> Linalg -> MemRef -> SCF -> LLVM）已经打通，但在**Linalg 层面**和**从 Linalg 到 MemRef/Vector 的转换过程**中，蕴含着巨大的性能优化空间。

以下我按照编译流程的各个阶段，为你梳理主要的优化点：

### 1. Linalg 层面的变换 (High-Level Optimizations)
这是 MLIR 优化的核心区域。Linalg 提供了结构化的信息，此时做变换收益最大。

*   **Tiling (分块)**:
    *   **目的**: 提高 Cache 命中率（Locality）。
    *   **操作**: 将大的矩阵运算（如 Matmul, Conv）切分成适应 L1/L2/L3 Cache 大小的块。
    *   **策略**: 多级 Tiling（Multi-level tiling）。
*   **Fusion (算子融合)**:
    *   **目的**: 减少内存带宽消耗，避免中间结果写回内存。
    *   **操作**:
        *   **Elementwise Fusion**: 将连续的逐元素操作（如 `add` -> `relu`）融合进同一个循环。
        *   **Tile and Fuse**: 在 Tiling 的同时进行融合，这是 Linalg 最强大的功能之一。让生产者（Producer）和消费者（Consumer）在同一个 Tile 内执行。
*   **Generalization & Interchange (泛化与交换)**:
    *   **Interchange**: 改变循环顺序（例如将 reduction loop 放到最内层或最外层），以优化内存访问模式或向量化潜力。
    *   **Padding**: 对 Tensor 进行 Padding 以消除边界检查，使生成的 Kernel 更规整，利于向量化。
*   **Packing (Data Layout Transformation)**:
    *   为了利用特定硬件指令（如 AVX-512 的 Blocked 访问或 Tensor Core），需要在 Linalg 层做数据的 Layout 变换（如 NCHW -> NC/xHWx）。

### 2. 向量化 (Vectorization)
这是获得高性能（尤其是 CPU 上）的关键步骤。不要直接从 Linalg 降级到 Scalar 的 SCF，而应该先经过 Vector Dialect。

*   **Linalg to Vector**:
    *   使用 `mlir-opt` 的策略将 Linalg op 转换为 `vector` dialect 的操作。
    *   利用 `vector.transfer_read` / `vector.transfer_write` 进行高效内存读写。
    *   **Vector Contract**: 将 Matmul 等操作映射到 `vector.contract`，这更容易进一步映射到硬件指令（如 FMA 或 AMX/Tensor Cores）。
*   **Scalable Vectorization**:
    *   如果是面向 ARM SVE 或 RISC-V V 扩展，考虑使用可变长向量（Scalable Vectors）。

### 3. Bufferization (Buffer 化策略)
从 Tensor (Value Semantics) 到 MemRef (Memory Semantics) 的转换是性能杀手潜伏的地方。

*   **One-Shot Bufferize**:
    *   **痛点**: 传统的 Bufferization 可能会产生大量的 `memref.copy` 和 `bufferization.clone`。
    *   **优化**: 务必使用 `One-Shot Bufferize` pass。它通过全图分析（Whole-graph analysis）来判断哪些 Tensor 可以 In-place update（原位更新），从而消除不必要的内存拷贝。
*   **Allocation Optimization**:
    *   尽量将小的临时 Buffer 分配在栈上（`alloca`）而不是堆上（`alloc`），或者使用内存池。

### 4. 循环优化 (SCF/Affine Level)
当降级到 `scf` 或 `affine` dialect 后，关注控制流和指令流水。

*   **Loop Unrolling & Jamming**: 展开循环以减少分支预测开销，增加指令级并行。
*   **Software Pipelining**: 在 `scf.for` 中进行软件流水线优化，隐藏内存延迟。
*   **Parallelization**:
    *   将 `scf.parallel` 映射到线程级并行。
    *   CPU 端通常降级到 `OpenMP` 或 `Async` dialect。
    *   GPU 端映射到 `gpu.thread/block`。
*   **Affine Optimizations**:
    *   如果内存访问模式是仿射的（Affine），利用 `affine` dialect 的 pass 做依赖分析、死代码消除和简化索引计算。

### 5. LLVM IR 层面
最后一步，但依然重要。

*   **Data Layout**: 确保生成的 LLVM IR 里的 Data Layout 字符串与目标机器匹配。
*   **Intrinsics**: 确保 MLIR 的 Vector 操作正确映射到了目标机器的 Intrinsics（如 x86 的 AVX2/AVX-512）。
*   **LLVM Passes**: 在 MLIR 导出 LLVM IR 后，通常还需要运行标准的 LLVM 优化管道（如 `-O3`），特别是：
    *   SLP Vectorizer
    *   Loop Vectorizer (如果前面 MLIR 没做彻底)
    *   Instruction Combining

### 总结建议的 Pass Pipeline 顺序

一个比较现代且高性能的 Pipeline 可能会长这样：

1.  **High-Level Linalg**: Tiling -> Fusion -> Padding.
2.  **Vectorization**: Linalg -> Vector (Strategy based).
3.  **Bufferization**: One-Shot Bufferize (Eliminate copies).
4.  **Lowering to SCF/Affine**: Handle control flow.
5.  **Loop Opts**: Unrolling / Parallel reduction.
6.  **Lowering to LLVM**: Convert dialects to LLVM dialect.

**既然你已经跑通了流程，我建议你重点检查以下两点，通常能带来最大的性能提升：**
1.  **有没有做 Tile and Fuse？** (Cache 效率)
2.  **有没有做显式的 Vectorization？** (计算密度)

如果你能提供具体的算子类型（比如是做 Matmul 还是做 Conv，或者是 Elementwise），我可以给出更针对性的 Pass 建议。