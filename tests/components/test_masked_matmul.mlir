module {
  func.func @host() {
    // 1. 构造 LHS (1x2)
    // [1.0, 2.0]
    %lhs = cherry.create_tensor dense<[[1.0, 2.0]]> : tensor<1x2xf32> -> !cherry.cherry_tensor<[1x2xf32]>

    // 2. 构造 RHS (2x2)
    // [1.0, 10.0]
    // [1.0, 10.0]
    %rhs = cherry.create_tensor dense<[[1.0, 10.0], [1.0, 10.0]]> : tensor<2x2xf32> -> !cherry.cherry_tensor<[2x2xf32]>

    // 3. 构造 Mask (1x2)
    // [1.0, 0.0] -> 第一个保留，第二个屏蔽
    // %mask = cherry.create_tensor dense<[[1.0, 0.0]]> : tensor<1x2xf32> -> !cherry.cherry_tensor<[1x2xf32]>
    %pos = cherry.constant (0 : i64): i64
    %mask = cherry.generate_mask %pos, [1, 2] : !cherry.cherry_tensor<[1x2xf32]>

    // 4. 执行 Masked MatMul
    // 注意：这里为了测试方便，直接使用 tensor 类型，实际你的 Op 可能要求 !cherry.cherry_tensor
    %result = cherry.masked_matmul %lhs, %rhs, %mask 
        : (!cherry.cherry_tensor<[1x2xf32]>, !cherry.cherry_tensor<[2x2xf32]>, !cherry.cherry_tensor<[1x2xf32]>) -> !cherry.cherry_tensor<[1x2xf32]>

    // 5. 打印结果
    // 预期输出: dense<[[3.000000e+00, -1.000000e+09]]>
    cherry.print %result : !cherry.cherry_tensor<[1x2xf32]>
    
    // 为了验证，我们可以用 vector.print 或简单的 return 来检查
    return
  }
}
