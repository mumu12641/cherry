 module {
  // 定义常量
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c12 = arith.constant 12 : index // 12 layers
  %c0_i64 = cherry.constant 0 : i64

  cherry.func private @llama_forward(
      // Inputs
      %token_id: i64, 
      %pos: i64,
      %kv_cache: !cherry.cherry_tensor<[12x2048x768xf32]>, // [n_layers, max_seq, dim]
      
      // Embeddings
      %embedding_table: !cherry.cherry_tensor<[32000x768xf32]>,
      
      // Weights (Flattened or stacked for easier access by index)
      // Attention Weights [n_layers, dim, dim]
      %rms_att_weight: !cherry.cherry_tensor<[12x768xf32]>,
      %wq: !cherry.cherry_tensor<[12x768x768xf32]>,
      %wk: !cherry.cherry_tensor<[12x768x768xf32]>,
      %wv: !cherry.cherry_tensor<[12x768x768xf32]>,
      %wo: !cherry.cherry_tensor<[12x768x768xf32]>,
      
      // FFN Weights [n_layers, ...]
      %rms_ffn_weight: !cherry.cherry_tensor<[12x768xf32]>,
      %w1: !cherry.cherry_tensor<[12x768x2048xf32]>, // Gate
      %w2: !cherry.cherry_tensor<[12x2048x768xf32]>, // Down
      %w3: !cherry.cherry_tensor<[12x768x2048xf32]>, // Up
      
      // Final Weights
      %rms_final_weight: !cherry.cherry_tensor<[1x768xf32]>,
      %wcls: !cherry.cherry_tensor<[32000x768xf32]>

  ) -> (!cherry.cherry_tensor<[1x32000xf32]>, !cherry.cherry_tensor<[12x2048x768xf32]>) {

    // 1. Embedding Lookup: x = token_embedding_table[token]
    // Output shape: [1, 768]
    %x = cherry.tensor_slice %embedding_table[%token_id, %c0_i64] sizes [1, 768] 
         : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>

    // 2. Transformer Layers Loop (0 to 11)
    // iter_args: current hidden state (%curr_x), current kv_cache (%curr_cache)
    %x_final, %cache_final = scf.for %i = %c0 to %c12 step %c1 
      iter_args(%curr_x = %x, %curr_cache = %kv_cache) -> (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x2048x768xf32]>) {
        
        // Convert index to i64 for cherry ops if needed, or use index directly depending on op def
        %i_i64 = arith.index_cast %i : index to i64

        // =================================================================
        // Part A: Attention Block
        // =================================================================
        
        // A.1 RMS Norm
        // Slice rms_att_weight[i] -> [1, 768]
        %rms_att_w_layer = cherry.tensor_slice %rms_att_weight[%i_i64, %c0_i64] sizes [1, 768]
            : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        
        %xb = cherry.rmsnorm %curr_x, %rms_att_w_layer 
            : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>

        // A.2 Q, K, V Projections
        // Slice weights for layer i
        %wq_layer = cherry.tensor_slice %wq[%i_i64, %c0_i64, %c0_i64] sizes [1, 768, 768]
            : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %wk_layer = cherry.tensor_slice %wk[%i_i64, %c0_i64, %c0_i64] sizes [1, 768, 768]
            : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %wv_layer = cherry.tensor_slice %wv[%i_i64, %c0_i64, %c0_i64] sizes [1, 768, 768]
            : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>

        // MatMul: [1, 768] x [768, 768] -> [1, 768]
        // Note: Assuming MatMulOp handles broadcasting or squeezing of the first dim of weights
        %q = cherry.matmul %xb, %wq_layer : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %k = cherry.matmul %xb, %wk_layer : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %v = cherry.matmul %xb, %wv_layer : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>

        // A.3 RoPE (Rotary Positional Embedding)
        %q_rope, %k_rope = cherry.rope %q, %k, %pos 
            : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>, i64) 
            -> (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>)

        // A.4 Update KV Cache
        // Insert k_rope and v into cache at [layer, pos]
        // Note: Using standard tensor.insert_slice as cherry.tensor_update is not defined in provided ops
        // Assuming cherry_tensor is compatible with tensor ops for this example
        // %next_cache = tensor.insert_slice %k_rope into %curr_cache[%i, %pos, 0] [1, 1, 768] [1, 1, 1] ...
        // For simplicity in this cherry.mlir, we assume a placeholder update or pass through
        // In real impl, you would use:
        // %cache_k_updated = cherry.update_cache %curr_cache, %k_rope, %i, %pos
        
        // A.5 Multi-Head Attention (Simplified as flat attention for example)
        // score = Q * K^T
        // att = softmax(score)
        // x_att = att * V
        // Here we simplify to a direct placeholder calculation as full MHA requires reshape/transpose ops not in cherryops.td
        %att_out = cherry.matmul %q_rope, %k_rope : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>

        // A.6 Output Projection
        %wo_layer = cherry.tensor_slice %wo[%i_i64, %c0_i64, %c0_i64] sizes [1, 768, 768]
            : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        
        %xb2 = cherry.matmul %att_out, %wo_layer 
            : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>

        // A.7 Residual Connection: x = x + xb2
        %x_resid_1 = cherry.add %curr_x, %xb2 
            : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>

        // =================================================================
        // Part B: FFN Block
        // =================================================================

        // B.1 RMS Norm
        %rms_ffn_w_layer = cherry.tensor_slice %rms_ffn_weight[%i_i64, %c0_i64] sizes [1, 768]
            : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        
        %xb_ffn = cherry.rmsnorm %x_resid_1, %rms_ffn_w_layer
            : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>

        // B.2 Projections (Gate w1, Up w3)
        // w1: [768, 2048], w3: [768, 2048]
        %w1_layer = cherry.tensor_slice %w1[%i_i64, %c0_i64, %c0_i64] sizes [1, 768, 2048]
             : (!cherry.cherry_tensor<[12x768x2048xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x2048xf32]>
        %w3_layer = cherry.tensor_slice %w3[%i_i64, %c0_i64, %c0_i64] sizes [1, 768, 2048]
             : (!cherry.cherry_tensor<[12x768x2048xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x2048xf32]>

        %hb = cherry.matmul %xb_ffn, %w1_layer 
            : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
        %hb2 = cherry.matmul %xb_ffn, %w3_layer 
            : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>

        // B.3 Activation (SiLU)
        %hb_silu = cherry.silu %hb : (!cherry.cherry_tensor<[1x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>

        // B.4 Element-wise Mul
        %hb_mul = cherry.mul %hb_silu, %hb2 
            : (!cherry.cherry_tensor<[1x2048xf32]>, !cherry.cherry_tensor<[1x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>

        // B.5 Down Projection (w2)
        // w2: [2048, 768]
        %w2_layer = cherry.tensor_slice %w2[%i_i64, %c0_i64, %c0_i64] sizes [1, 2048, 768]
             : (!cherry.cherry_tensor<[12x2048x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x2048x768xf32]>
        
        %ffn_out = cherry.matmul %hb_mul, %w2_layer
            : (!cherry.cherry_tensor<[1x2048xf32]>, !cherry.cherry_tensor<[1x2048x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>

        // B.6 Residual Connection
        %x_next = cherry.add %x_resid_1, %ffn_out
            : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>

        scf.yield %x_next, %curr_cache : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x2048x768xf32]>
    }

    // 3. Final Block
    // x = rmsnorm(x, rms_final_weight)
    // logits = matmul(x, wcls)
    
    %x_norm = cherry.rmsnorm %x_final, %rms_final_weight
        : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>

    // wcls is [32000, 768], we might need transpose or assume matmul handles [1, 768] * [32000, 768]^T
    // Assuming wcls is stored as [768, 32000] for direct matmul, or matmul supports this.
    // Based on example.mlir, wcls is [vocab, dim].
    // Let's assume we output [1, 32000]
    
    // Note: Since cherry.matmul behavior on dimensions depends on implementation, 
    // we assume here it produces the logits.
    %logits = cherry.matmul %x_norm, %wcls
        : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[32000x768xf32]>) -> !cherry.cherry_tensor<[1x32000xf32]>

    cherry.return %logits, %cache_final : !cherry.cherry_tensor<[1x32000xf32]>, !cherry.cherry_tensor<[12x2048x768xf32]>
  }
}


 func.func @main() {
    // ==========================================
    // 1. 初始化常量与配置
    // ==========================================
    %max_len = arith.constant 128 : i64
    %start_token = cherry.constant 1 : i64 // BOS token
    %start_pos = cherry.constant 0 : i64
    
    // ==========================================
    // 2. 分配权重 (Mock Data)
    // 在实际推理中，这些数据通常通过 memref.global 加载或 mmap
    // 这里使用 create_tensor dense<0.0> 作为占位符
    // ==========================================
    
    // Embedding
    %embedding = cherry.create_tensor dense<0.0> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
    
    // Attention Weights (12 layers)
    %rms_att = cherry.create_tensor dense<0.0> : tensor<12x768xf32> -> !cherry.cherry_tensor<[12x768xf32]>
    %wq = cherry.create_tensor dense<0.0> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %wk = cherry.create_tensor dense<0.0> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %wv = cherry.create_tensor dense<0.0> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %wo = cherry.create_tensor dense<0.0> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    
    // FFN Weights (12 layers)
    %rms_ffn = cherry.create_tensor dense<0.0> : tensor<12x768xf32> -> !cherry.cherry_tensor<[12x768xf32]>
    %w1 = cherry.create_tensor dense<0.0> : tensor<12x768x2048xf32> -> !cherry.cherry_tensor<[12x768x2048xf32]>
    %w2 = cherry.create_tensor dense<0.0> : tensor<12x2048x768xf32> -> !cherry.cherry_tensor<[12x2048x768xf32]>
    %w3 = cherry.create_tensor dense<0.0> : tensor<12x768x2048xf32> -> !cherry.cherry_tensor<[12x768x2048xf32]>
    
    // Final Weights
    %rms_final = cherry.create_tensor dense<0.0> : tensor<1x768xf32> -> !cherry.cherry_tensor<[1x768xf32]>
    %wcls = cherry.create_tensor dense<0.0> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>

    // KV Cache (Initialize with Zeros)
    %kv_cache_init = cherry.create_tensor dense<0.0> : tensor<12x2048x768xf32> -> !cherry.cherry_tensor<[12x2048x768xf32]>

    // ==========================================
    // 3. 推理循环 (Generation Loop)
    // ==========================================
    %final_token, %final_pos, %final_cache = scf.while (%arg_token = %start_token, %arg_pos = %start_pos, %arg_cache = %kv_cache_init) 
          : (i64, i64, !cherry.cherry_tensor<[12x2048x768xf32]>) -> (i64, i64, !cherry.cherry_tensor<[12x2048x768xf32]>) {
        
        // Condition: pos < max_len
        %cond = arith.cmpi slt, %arg_pos, %max_len : i64
        scf.condition(%cond) %arg_token, %arg_pos, %arg_cache : i64, i64, !cherry.cherry_tensor<[12x2048x768xf32]>
      
      } do {
        ^bb0(%curr_token: i64, %curr_pos: i64, %curr_cache: !cherry.cherry_tensor<[12x2048x768xf32]>):
        
        // 3.1 Call Forward Function
        // 传入当前 Token, Pos, Cache 以及所有权重
        %logits, %next_cache = cherry.call @llama_forward(
            %curr_token, %curr_pos, %curr_cache,
            %embedding,
            %rms_att, %wq, %wk, %wv, %wo,
            %rms_ffn, %w1, %w2, %w3,
            %rms_final, %wcls
        ) : (i64, i64, !cherry.cherry_tensor<[12x2048x768xf32]>, 
             !cherry.cherry_tensor<[32000x768xf32]>,
             !cherry.cherry_tensor<[12x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>,
             !cherry.cherry_tensor<[12x768xf32]>, !cherry.cherry_tensor<[12x768x2048xf32]>, !cherry.cherry_tensor<[12x2048x768xf32]>, !cherry.cherry_tensor<[12x768x2048xf32]>,
             !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[32000x768xf32]>
            ) -> (!cherry.cherry_tensor<[1x32000xf32]>, !cherry.cherry_tensor<[12x2048x768xf32]>)
            
        // 3.2 Greedy Sampling (ArgMax)
        // logits shape: [1, 32000]
        %arg_max = cherry.argmax %logits dim 1 : (!cherry.cherry_tensor<[1x32000xf32]>) -> (!cherry.cherry_tensor<[1xf32]>)
        
        // 3.3 Extract Token ID
        %zero_idx = cherry.constant 0 : i64
        %next_token_f32 = cherry.tensor_get %arg_max [%zero_idx] : (!cherry.cherry_tensor<[1xf32]>, i64) -> f32
        %next_token = arith.fptosi %next_token_f32 : f32 to i64
        
        // Optional: Print generated token
        // cherry.print %arg_max : !cherry.cherry_tensor<[1xf32]>

        // 3.4 Update Position
        %c1 = cherry.constant 1 : i64
        %next_pos = arith.addi %curr_pos, %c1 : i64
        
        // 3.5 Yield to next iteration
        scf.yield %next_token, %next_pos, %next_cache : i64, i64, !cherry.cherry_tensor<[12x2048x768xf32]>
      }

    return
  }