// 原始 IR
func.func @fc_relu(%input: tensor<128x256xf32>, %weights: tensor<256x64xf32>,
                   %bias: tensor<64xf32>, %output: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %c0 = arith.constant dense<0.0> : tensor<128x64xf32>
  
  // matmul
  %0 = linalg.matmul ins(%input, %weights : tensor<128x256xf32>, tensor<256x64xf32> ) outs(%output : tensor<128x64xf32>) -> tensor<128x64xf32>
  
  // broadcast bias
  %1 = linalg.broadcast ins(%bias : tensor<64xf32>) outs(%output : tensor<128x64xf32>) dimensions = [0]
  
  // add
  %2 = linalg.add ins(%0, %1 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%output : tensor<128x64xf32>) -> tensor<128x64xf32>
  
  // relu (max with 0)
  %3 = linalg.max ins(%2, %c0 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%output : tensor<128x64xf32>) -> tensor<128x64xf32>
  
  return %3 : tensor<128x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    
    transform.sequence failures(propagate) {
    ^bb0(%arg0: !transform.any_op):
      
      // ========== 1. 匹配目标 ==========
      %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
      %add = transform.structured.match ops{["linalg.add"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
      %max = transform.structured.match ops{["linalg.max"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
      
      // ========== 2. Tiling：从输出端开始 ==========
      %tiled_max, %loop = transform.structured.tile_using_forall %max 
        tile_sizes [32, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      
      // ========== 3. Fusion：把上游拉进循环 ==========
      %fused_add, %loop_1 = transform.structured.fuse_into_containing_op %add into %loop
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      
      %fused_matmul, %loop_2 = transform.structured.fuse_into_containing_op %matmul into %loop_1
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      
      // ========== 4. 二次 Tiling：更细粒度 ==========
      %tiled_inner, %inner_loop = transform.structured.tile_using_forall %fused_matmul 
        tile_sizes [8, 8]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      
      // ========== 5. 向量化 ==========
      %func = transform.structured.match ops{["func.func"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
      
      transform.structured.vectorize_children_and_apply_patterns %func
        : (!transform.any_op) -> !transform.any_op
      
      // ========== 6. 清理 ==========
      transform.apply_patterns to %func {
        transform.apply_patterns.canonicalization
      } : !transform.any_op
      
      transform.yield
    }
    
    transform.yield
  }
}