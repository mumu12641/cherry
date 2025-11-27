module {
  // 维度定义: Batch=1, Seq=4, Dim=8, FF=32
  
  func.func @transformer_block(
    %x: !cherry.tensor<[1, 4, 8], f32>,
    // Attention Weights (简化为 Tensor 输入)
    %w_q: !cherry.tensor<[8, 8], f32>,
    %w_k: !cherry.tensor<[8, 8], f32>,
    %w_v: !cherry.tensor<[8, 8], f32>,
    // FFN Weights
    %w_ff1: !cherry.tensor<[8, 32], f32>,
    %w_ff2: !cherry.tensor<[32, 8], f32>,
    // LayerNorm Parameters (Gamma, Beta)
    %ln_gamma: !cherry.tensor<[8], f32>,
    %ln_beta:  !cherry.tensor<[8], f32>
  ) -> !cherry.tensor<[1, 4, 8], f32> {

    // ==========================================
    // 1. Self-Attention
    // ==========================================

    // 1.1 Projections (Q, K, V)
    // 注意：PyTorch Linear 是 x @ W.T，这里简化为 x @ W
    %q = cherry.matmul %x, %w_q : (!cherry.tensor<[1, 4, 8], f32>, !cherry.tensor<[8, 8], f32>) -> !cherry.tensor<[1, 4, 8], f32>
    %k = cherry.matmul %x, %w_k : (!cherry.tensor<[1, 4, 8], f32>, !cherry.tensor<[8, 8], f32>) -> !cherry.tensor<[1, 4, 8], f32>
    %v = cherry.matmul %x, %w_v : (!cherry.tensor<[1, 4, 8], f32>, !cherry.tensor<[8, 8], f32>) -> !cherry.tensor<[1, 4, 8], f32>

    // 1.2 Transpose K -> [1, 8, 4]
    // Permutation: [0, 2, 1]
    %p0 = cherry.constant 0 : i64
    %p1 = cherry.constant 2 : i64
    %p2 = cherry.constant 1 : i64
    %k_t = cherry.transpose %k, %p0, %p1, %p2 : (!cherry.tensor<[1, 4, 8], f32>, i64, i64, i64) -> !cherry.tensor<[1, 8, 4], f32>

    // 1.3 Score = Q @ K.T -> [1, 4, 4]
    %score_raw = cherry.matmul %q, %k_t : (!cherry.tensor<[1, 4, 8], f32>, !cherry.tensor<[1, 8, 4], f32>) -> !cherry.tensor<[1, 4, 4], f32>

    // 1.4 Scale (Divide by sqrt(8) = 2.828)
    // 创建一个包含 2.828 的 1x1x1 Tensor
    %sqrt_dk = cherry.constant dense<2.828> : tensor<1x1x1xf32>
    // 广播到 [1, 4, 4]
    %d0 = cherry.constant 1 : i64
    %d1 = cherry.constant 4 : i64
    %scale_tensor = cherry.broadcast %sqrt_dk, %d0, %d1, %d1 : (!cherry.tensor<[1, 1, 1], f32>, i64, i64, i64) -> !cherry.tensor<[1, 4, 4], f32>
    // 除法
    %score_scaled = cherry.tensor_div %score_raw, %scale_tensor : (!cherry.tensor<[1, 4, 4], f32>, !cherry.tensor<[1, 4, 4], f32>) -> !cherry.tensor<[1, 4, 4], f32>

    // 1.5 Softmax (axis = 2, last dim)
    %attn_probs = cherry.softmax %score_scaled axis 2 : (!cherry.tensor<[1, 4, 4], f32>) -> !cherry.tensor<[1, 4, 4], f32>

    // 1.6 Output = Probs @ V -> [1, 4, 8]
    %attn_out = cherry.matmul %attn_probs, %v : (!cherry.tensor<[1, 4, 4], f32>, !cherry.tensor<[1, 4, 8], f32>) -> !cherry.tensor<[1, 4, 8], f32>

    // ==========================================
    // 2. Add & Norm 1
    // ==========================================
    %res1 = cherry.tensor_add %x, %attn_out : (!cherry.tensor<[1, 4, 8], f32>, !cherry.tensor<[1, 4, 8], f32>) -> !cherry.tensor<[1, 4, 8], f32>
    
    // LayerNorm (eps=1e-5)
    %norm1 = cherry.layernorm %res1, %ln_gamma, %ln_beta eps 1.0e-5 : (!cherry.tensor<[1, 4, 8], f32>, !cherry.tensor<[8], f32>, !cherry.tensor<[8], f32>) -> !cherry.tensor<[1, 4, 8], f32>

    // ==========================================
    // 3. Feed Forward Network
    // ==========================================
    
    // 3.1 Linear 1: [1, 4, 8] @ [8, 32] -> [1, 4, 32]
    %ff1 = cherry.matmul %norm1, %w_ff1 : (!cherry.tensor<[1, 4, 8], f32>, !cherry.tensor<[8, 32], f32>) -> !cherry.tensor<[1, 4, 32], f32>
    
    // 3.2 Relu
    %ff_relu = cherry.tensor_relu %ff1 : (!cherry.tensor<[1, 4, 32], f32>) -> !cherry.tensor<[1, 4, 32], f32>
    
    // 3.3 Linear 2: [1, 4, 32] @ [32, 8] -> [1, 4, 8]
    %ff2 = cherry.matmul %ff_relu, %w_ff2 : (!cherry.tensor<[1, 4, 32], f32>, !cherry.tensor<[32, 8], f32>) -> !cherry.tensor<[1, 4, 8], f32>

    // ==========================================
    // 4. Add & Norm 2
    // ==========================================
    %res2 = cherry.tensor_add %norm1, %ff2 : (!cherry.tensor<[1, 4, 8], f32>, !cherry.tensor<[1, 4, 8], f32>) -> !cherry.tensor<[1, 4, 8], f32>
    
    %final_output = cherry.layernorm %res2, %ln_gamma, %ln_beta eps 1.0e-5 : (!cherry.tensor<[1, 4, 8], f32>, !cherry.tensor<[8], f32>, !cherry.tensor<[8], f32>) -> !cherry.tensor<[1, 4, 8], f32>

    return %final_output : !cherry.tensor<[1, 4, 8], f32>
  }
}
