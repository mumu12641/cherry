func.func @attention(
    %Q: tensor<128x64xf32>,
    %K: tensor<128x64xf32>,
    %V: tensor<128x64xf32>
) -> tensor<128x64xf32> {
  
  // 常量定义
  %cst_0 = arith.constant 0.0 : f32
  %cst_neg_inf = arith.constant 0xFF800000 : f32 // -inf

  // ==========================================
  // 1. K_transpose = Transpose(K)
  // 形状: [128, 64] -> [64, 128]
  // ==========================================
  %init_KT = tensor.empty() : tensor<64x128xf32>
  %KT = linalg.transpose
      ins(%K : tensor<128x64xf32>)
      outs(%init_KT : tensor<64x128xf32>)
      permutation = [1, 0]

  // ==========================================
  // 2. S = Matmul(Q, KT)
  // 形状: [128, 64] x [64, 128] -> [128, 128]
  // ==========================================
  %init_S = tensor.empty() : tensor<128x128xf32>
  %filled_S = linalg.fill ins(%cst_0 : f32) outs(%init_S : tensor<128x128xf32>) -> tensor<128x128xf32>
  %S = linalg.matmul
      ins(%Q, %KT : tensor<128x64xf32>, tensor<64x128xf32>)
      outs(%filled_S : tensor<128x128xf32>) -> tensor<128x128xf32>

  // ==========================================
  // 3. Softmax: max_val = ReduceMax(S)
  // 形状: [128, 128] -> [128]
  // ==========================================
  %init_max = tensor.empty() : tensor<128xf32>
  %filled_max = linalg.fill ins(%cst_neg_inf : f32) outs(%init_max : tensor<128xf32>) -> tensor<128xf32>
  %max_val = linalg.reduce
      ins(%S : tensor<128x128xf32>)
      outs(%filled_max : tensor<128xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %0 = arith.maximumf %out, %in : f32
        linalg.yield %0 : f32
      }

  // ==========================================
  // 4. Softmax: Broadcast max back to NxN
  // 形状: [128] -> [128, 128]
  // ==========================================
  %init_bcast_max = tensor.empty() : tensor<128x128xf32>
  %max_bcast = linalg.broadcast
      ins(%max_val : tensor<128xf32>)
      outs(%init_bcast_max : tensor<128x128xf32>)
      dimensions = [1]

  // ==========================================
  // 5. Softmax: S_shifted = S - max_bcast
  // 形状: [128, 128]
  // ==========================================
  %init_sub = tensor.empty() : tensor<128x128xf32>
  %sub = linalg.sub 
      ins(%S, %max_bcast : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%init_sub : tensor<128x128xf32>) -> tensor<128x128xf32>

  // ==========================================
  // 6. Softmax: Exp = exp(S_shifted)
  // 形状: [128, 128]
  // ==========================================
  %init_exp = tensor.empty() : tensor<128x128xf32>
  %exp = linalg.exp 
      ins(%sub : tensor<128x128xf32>)
      outs(%init_exp : tensor<128x128xf32>) -> tensor<128x128xf32>

  // ==========================================
  // 7. Softmax: sum_exp = ReduceSum(Exp)
  // 形状: [128, 128] -> [128]
  // ==========================================
  %init_sum = tensor.empty() : tensor<128xf32>
  %filled_sum = linalg.fill ins(%cst_0 : f32) outs(%init_sum : tensor<128xf32>) -> tensor<128xf32>
  %sum_val = linalg.reduce
      ins(%exp : tensor<128x128xf32>)
      outs(%filled_sum : tensor<128xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %out, %in : f32
        linalg.yield %0 : f32
      }

  // ==========================================
  // 8. Softmax: Broadcast sum back to NxN
  // 形状: [128] -> [128, 128]
  // ==========================================
  %init_bcast_sum = tensor.empty() : tensor<128x128xf32>
  %sum_bcast = linalg.broadcast
      ins(%sum_val : tensor<128xf32>)
      outs(%init_bcast_sum : tensor<128x128xf32>)
      dimensions = [1]

  // ==========================================
  // 9. Softmax: P = Exp / sum_bcast
  // 形状: [128, 128]
  // ==========================================
  %init_P = tensor.empty() : tensor<128x128xf32>
  %P = linalg.div 
      ins(%exp, %sum_bcast : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%init_P : tensor<128x128xf32>) -> tensor<128x128xf32>

  // ==========================================
  // 10. O = Matmul(P, V)
  // 形状: [128, 128] x [128, 64] -> [128, 64]
  // ==========================================
  %init_O = tensor.empty() : tensor<128x64xf32>
  %filled_O = linalg.fill ins(%cst_0 : f32) outs(%init_O : tensor<128x64xf32>) -> tensor<128x64xf32>
  %O = linalg.matmul
      ins(%P, %V : tensor<128x128xf32>, tensor<128x64xf32>)
      outs(%filled_O : tensor<128x64xf32>) -> tensor<128x64xf32>

  return %O : tensor<128x64xf32>
}
