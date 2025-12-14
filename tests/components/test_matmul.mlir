module {
  cherry.func @host() {
    // 1. 准备 LHS (1x4)
    // 包含正数、负数、典型小数
    %lhs = cherry.create_tensor dense<[[0.5, -1.2, 3.14, 0.01]]> : tensor<1x4xf32> -> !cherry.cherry_tensor<[1x4xf32]>

    // 2. 准备 RHS (4x4)
    // 包含各种随机分布的小数
    %rhs = cherry.create_tensor dense<[
      [ 0.1,  0.2,  0.3,   0.4 ],
      [-0.5,  1.0, -1.5,   2.0 ],
      [ 1.1,  2.2,  3.3,   4.4 ],
      [ 0.0, -0.1,  0.05, -0.05]
    ]> : tensor<4x4xf32> -> !cherry.cherry_tensor<[4x4xf32]>

    // 3. 调用 MatMul Op
    // 
    // --- 预期计算结果 (Expected Result) ---
    // Col 0: (0.5*0.1) + (-1.2*-0.5) + (3.14*1.1) + (0.01*0.0)   = 0.05 + 0.6 + 3.454 + 0      = 4.104
    // Col 1: (0.5*0.2) + (-1.2*1.0)  + (3.14*2.2) + (0.01*-0.1)  = 0.1 - 1.2 + 6.908 - 0.001   = 5.807
    // Col 2: (0.5*0.3) + (-1.2*-1.5) + (3.14*3.3) + (0.01*0.05)  = 0.15 + 1.8 + 10.362 + 0.0005 = 12.3125
    // Col 3: (0.5*0.4) + (-1.2*2.0)  + (3.14*4.4) + (0.01*-0.05) = 0.2 - 2.4 + 13.816 - 0.0005  = 11.6155
    //
    // 最终结果应该是: [4.104, 5.807, 12.3125, 11.6155]
    // -------------------------------------
    
    %result = cherry.matmul %lhs, %rhs : (!cherry.cherry_tensor<[1x4xf32]>, !cherry.cherry_tensor<[4x4xf32]>) -> !cherry.cherry_tensor<[1x4xf32]>
    
    cherry.print %result : !cherry.cherry_tensor<[1x4xf32]>
    
    cherry.return 
  }
}
