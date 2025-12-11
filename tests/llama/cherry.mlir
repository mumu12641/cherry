module {

  
  cherry.func private @llama_forward(
       // Inputs
      %token_id: i64, 
      %pos: i64,
      %k_cache: !cherry.cherry_tensor<[12x1024x768xf32]>, // [n_layers, max_seq, dim]
      %v_cache: !cherry.cherry_tensor<[12x1024x768xf32]>, // [n_layers, max_seq, dim]
      %mask: !cherry.cherry_tensor<[1x1024xf32]>, // [1, max_seq]
      
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
      %rms_final_weight: !cherry.cherry_tensor<[768xf32]>,
      %wcls: !cherry.cherry_tensor<[32000x768xf32]>
      
  ) -> (!cherry.cherry_tensor<[1x32000xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) {   
  
    // 1. Embedding Lookup: x = token_embedding_table[token]
    // Output shape: [1, 768]
    %c0_i64 = cherry.constant (0 : i64) : i64
    %x = cherry.tensor_slice %embedding_table[%token_id, %c0_i64] sizes [1, 768] 
         : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
         
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %constant_1 = cherry.constant (1 : i64) : i64
    %n_layers = arith.constant 12 : index // 12 layers
    %dim = cherry.constant (768 : i64) : i64
    %ffn_dim = cherry.constant (2048 : i64) : i64
    %head_num_index = arith.constant 12 : index // 12 heads
    %head_num = cherry.constant (12 : i64) : i64 // head_dim
    %head_dim = cherry.constant (64 : i64) : i64 // head_dim
    %scale_val = cherry.constant (0.125 : f32) : f32 // 1 / sqrt(64)
    %max_len = cherry.constant (1024 : i64) : i64
    %vocab_size = cherry.constant (32000 : i64) : i64

    // 2. Transformer Layers Loop (0 to 11)
    // iter_args: current hidden state (%curr_x), current kv_cache (%curr_cache)
    %x_final, %k_cache_final, %v_cache_final = scf.for %i = %c0 to %n_layers step %c1 
      iter_args(%curr_x = %x, %curr_k_cache = %k_cache, %curr_v_cache = %v_cache) -> (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) {
        
        %layer = arith.index_cast %i : index to i64
    
        // =================================================================
        // Part A: Attention Block
        // =================================================================

        // A.1 RMS Norm
        // Slice rms_att_weight[i] -> [1, 768]
        %rms_att_w_layer_1 = cherry.tensor_slice %rms_att_weight[%layer, %c0_i64] sizes [1, 768]
            : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %rms_att_w_layer = cherry.reshape %rms_att_w_layer_1, %dim : (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>
            
        %xb = cherry.rmsnorm %curr_x scale %rms_att_w_layer eps 1.000000e-05
            : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
        
        // A.2 Q, K, V Projections
        // Slice weights for layer i
        %wq_layer_1 = cherry.tensor_slice %wq[%layer, %c0_i64, %c0_i64] sizes [1, 768, 768]
            : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %wq_layer = cherry.reshape %wq_layer_1, %dim, %dim : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        
        %wk_layer_1 = cherry.tensor_slice %wk[%layer, %c0_i64, %c0_i64] sizes [1, 768, 768]
            : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %wk_layer = cherry.reshape %wk_layer_1, %dim, %dim : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>

        %wv_layer_1 = cherry.tensor_slice %wv[%layer, %c0_i64, %c0_i64] sizes [1, 768, 768]
            : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %wv_layer = cherry.reshape %wv_layer_1, %dim, %dim : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>

            
        // MatMul: [1, 768] x [768, 768] -> [1, 768]
        // Note: Assuming MatMulOp handles broadcasting or squeezing of the first dim of weights
        %q_raw = cherry.matmul %xb, %wq_layer : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %k_raw = cherry.matmul %xb, %wk_layer : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %v_raw = cherry.matmul %xb, %wv_layer : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>

        // A.3 RoPE (Rotary Positional Embedding)
        // TODO
        %q_raw_heads = cherry.reshape %q_raw, %constant_1, %head_num, %head_dim : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %q_rope = cherry.rope %q_raw_heads, %pos : (!cherry.cherry_tensor<[1x12x64xf32]>, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %q = cherry.reshape %q_rope, %constant_1, %dim : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>

        %k_raw_heads = cherry.reshape %k_raw, %constant_1, %head_num, %head_dim : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %k_rope = cherry.rope %k_raw_heads, %pos : (!cherry.cherry_tensor<[1x12x64xf32]>, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %k = cherry.reshape %k_rope, %constant_1, %dim : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>

        %v_raw_heads = cherry.reshape %v_raw, %constant_1, %head_num, %head_dim : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %v_rope = cherry.rope %v_raw_heads, %pos : (!cherry.cherry_tensor<[1x12x64xf32]>, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
        %v = cherry.reshape %v_rope, %constant_1, %dim : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>

        // A.4 Update KV Cache
        %k_1 = cherry.reshape %k, %constant_1, %constant_1, %dim : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x768xf32]>
        %next_k_cache = cherry.tensor_set_slice %curr_k_cache[%layer, %pos], %k_1 
            : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64 ) 
            -> !cherry.cherry_tensor<[12x1024x768xf32]>
        
        %v_1 = cherry.reshape %v, %constant_1, %constant_1, %dim : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x768xf32]>
        %next_v_cache = cherry.tensor_set_slice %curr_v_cache[%layer, %pos], %v_1 
            : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64 ) 
            -> !cherry.cherry_tensor<[12x1024x768xf32]>
        
        
        // A.5 Multi-Head Attention 
        %k_layer_full_1 = cherry.tensor_slice %next_k_cache[%layer, %c0_i64, %c0_i64] sizes [1, 1024, 768]
            : (!cherry.cherry_tensor<[12x1024x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1024x768xf32]>
        %k_layer_full = cherry.reshape %k_layer_full_1, %max_len, %dim
            : (!cherry.cherry_tensor<[1x1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x768xf32]>
        
        %v_layer_full_1 = cherry.tensor_slice %next_v_cache[%layer, %c0_i64, %c0_i64] sizes [1, 1024, 768]
            : (!cherry.cherry_tensor<[12x1024x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1024x768xf32]>
        %v_layer_full = cherry.reshape %v_layer_full_1, %max_len, %dim
            : (!cherry.cherry_tensor<[1x1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x768xf32]>
            
        %att_out_init = cherry.create_tensor dense<0.0> : tensor<1x12x64xf32> -> !cherry.cherry_tensor<[1x12x64xf32]>
        %att_out_final_heads = scf.for %h = %c0 to %head_num_index step %c1 
          iter_args(%curr_att_out = %att_out_init) -> (!cherry.cherry_tensor<[1x12x64xf32]>) {
          
            %h_i64 = arith.index_cast %h : index to i64
            %offset = arith.muli %h_i64, %head_dim : i64
            
            // ---------------------------------------------------------
            //  Q Head [1, 64]
            // ---------------------------------------------------------
            %q_head = cherry.tensor_slice %q[%c0_i64, %offset] sizes [1, 64]
                : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x64xf32]>

            // ---------------------------------------------------------
            //  K Head [1024, 64] -> [64, 1024]
            // ---------------------------------------------------------
            // start_indices: [0, offset]
            %k_head = cherry.tensor_slice %k_layer_full[%c0_i64, %offset] sizes [1024, 64]
                : (!cherry.cherry_tensor<[1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x64xf32]>
            
            %k_head_T = cherry.transpose %k_head perm [1, 0]
                : (!cherry.cherry_tensor<[1024x64xf32]>) -> !cherry.cherry_tensor<[64x1024xf32]>
            
            // ---------------------------------------------------------
            //  Masked Attention Score 
            // ---------------------------------------------------------
            // Q: [1, 64]
            // K_T: [64, 1024]
            // Mask: [1, 1024]
            %score = cherry.masked_matmul %q_head, %k_head_T, %mask
                : (!cherry.cherry_tensor<[1x64xf32]>, !cherry.cherry_tensor<[64x1024xf32]>, !cherry.cherry_tensor<[1x1024xf32]>) 
                -> !cherry.cherry_tensor<[1x1024xf32]>
            
            %score_scaled = cherry.tensor_mul_scalar %score, %scale_val
                : (!cherry.cherry_tensor<[1x1024xf32]>, f32) -> !cherry.cherry_tensor<[1x1024xf32]>
            
            %probs = cherry.softmax %score_scaled axis 1
                : (!cherry.cherry_tensor<[1x1024xf32]>) -> !cherry.cherry_tensor<[1x1024xf32]>
            
            %v_head = cherry.tensor_slice %v_layer_full[%c0_i64, %offset] sizes [1024, 64]
                : (!cherry.cherry_tensor<[1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x64xf32]>
            
            // Probs: [1, 1024]
            // V_head: [1024, 64]
            // Result: [1, 64]
            %context_head = cherry.matmul %probs, %v_head
                : (!cherry.cherry_tensor<[1x1024xf32]>, !cherry.cherry_tensor<[1024x64xf32]>) -> !cherry.cherry_tensor<[1x64xf32]>

            %context_head_reshape = cherry.reshape %context_head, %constant_1, %constant_1, %head_dim 
                : (!cherry.cherry_tensor<[1x64xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x64xf32]>

            %next_att_out = cherry.tensor_set_slice %curr_att_out[%c0_i64, %h_i64], %context_head_reshape
                : (!cherry.cherry_tensor<[1x12x64xf32]>, !cherry.cherry_tensor<[1x1x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x12x64xf32]>
                
            scf.yield %next_att_out : !cherry.cherry_tensor<[1x12x64xf32]>
        }
        %att_out_final = cherry.reshape %att_out_final_heads, %constant_1, %dim 
                : (!cherry.cherry_tensor<[1x12x64xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>

        // A.6 Output Projection
        %wo_layer_1 = cherry.tensor_slice %wo[%layer, %c0_i64, %c0_i64] sizes [1, 768, 768]
            : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %wo_layer = cherry.reshape %wo_layer_1, %dim, %dim : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>

        %xb2 = cherry.matmul %att_out_final, %wo_layer 
            : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        
        // A.7 Residual Connection: x = x + xb2
        %x_resid_1 = cherry.tensor_add %curr_x, %xb2 
            : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        
        // =================================================================
        // Part B: FFN Block
        // =================================================================
        
        // B.1 RMS Norm
        %rms_ffn_w_layer_1 = cherry.tensor_slice %rms_ffn_weight[%layer, %c0_i64] sizes [1, 768]
            : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %rms_ffn_w_layer = cherry.reshape %rms_ffn_w_layer_1, %dim : (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>
        
        %xb_ffn = cherry.rmsnorm %x_resid_1 scale %rms_ffn_w_layer eps 1.000000e-05
            : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
        
        // B.2 Projections (Gate w1, Up w3)
        // w1: [768, 2048], w3: [768, 2048]
        %w1_layer_1 = cherry.tensor_slice %w1[%layer, %c0_i64, %c0_i64] sizes [1, 768, 2048]
             : (!cherry.cherry_tensor<[12x768x2048xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x2048xf32]>
        %w1_layer = cherry.reshape %w1_layer_1, %dim, %ffn_dim : (!cherry.cherry_tensor<[1x768x2048xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x2048xf32]>
        
        %w3_layer_1 = cherry.tensor_slice %w3[%layer, %c0_i64, %c0_i64] sizes [1, 768, 2048]
             : (!cherry.cherry_tensor<[12x768x2048xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x2048xf32]>
        %w3_layer = cherry.reshape %w3_layer_1, %dim, %ffn_dim : (!cherry.cherry_tensor<[1x768x2048xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x2048xf32]>
        
        %hb = cherry.matmul %xb_ffn, %w1_layer 
            : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
        %hb2 = cherry.matmul %xb_ffn, %w3_layer 
            : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
            
        // B.3 Activation (SiLU)
        %hb_silu = cherry.tensor_silu %hb : (!cherry.cherry_tensor<[1x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
        
        // B.4 Element-wise Mul
        %hb_mul = cherry.tensor_mul %hb_silu, %hb2 
            : (!cherry.cherry_tensor<[1x2048xf32]>, !cherry.cherry_tensor<[1x2048xf32]>) -> !cherry.cherry_tensor<[1x2048xf32]>
        
        // B.5 Down Projection (w2)
        // w2: [2048, 768]
        %w2_layer_1 = cherry.tensor_slice %w2[%layer, %c0_i64, %c0_i64] sizes [1, 2048, 768]
             : (!cherry.cherry_tensor<[12x2048x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x2048x768xf32]>
        %w2_layer = cherry.reshape %w2_layer_1, %ffn_dim, %dim : (!cherry.cherry_tensor<[1x2048x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[2048x768xf32]>

        %ffn_out = cherry.matmul %hb_mul, %w2_layer
            : (!cherry.cherry_tensor<[1x2048xf32]>, !cherry.cherry_tensor<[2048x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>

        // B.6 Residual Connection
        %x_next = cherry.tensor_add %x_resid_1, %ffn_out
            : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>

        scf.yield %x_next, %next_k_cache, %next_v_cache : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
    }
    
    // 3. Final Block
    // x = rmsnorm(x, rms_final_weight)
    // logits = matmul(x, wcls)
    %x_norm = cherry.rmsnorm %x_final scale %rms_final_weight eps 1.000000e-05
        : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>

    %wcls_reshape = cherry.reshape %wcls, %dim, %vocab_size
        : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x32000xf32]>
    %logits = cherry.matmul %x_norm, %wcls_reshape
        : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x32000xf32]>) -> !cherry.cherry_tensor<[1x32000xf32]>

    cherry.return %logits, %k_cache_final, %v_cache_final : !cherry.cherry_tensor<[1x32000xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
  }

  cherry.func @host(){
    // Embedding
    %embedding = cherry.create_tensor dense<2.0> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
    
    // Attention Weights (12 layers)
    %rms_att = cherry.create_tensor dense<3.0> : tensor<12x768xf32> -> !cherry.cherry_tensor<[12x768xf32]>
    %wq = cherry.create_tensor dense<4.0> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %wk = cherry.create_tensor dense<5.0> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %wv = cherry.create_tensor dense<6.0> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %wo = cherry.create_tensor dense<7.0> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    
    // FFN Weights (12 layers)
    %rms_ffn = cherry.create_tensor dense<8.0> : tensor<12x768xf32> -> !cherry.cherry_tensor<[12x768xf32]>
    %w1 = cherry.create_tensor dense<9.0> : tensor<12x768x2048xf32> -> !cherry.cherry_tensor<[12x768x2048xf32]>
    %w2 = cherry.create_tensor dense<10.0> : tensor<12x2048x768xf32> -> !cherry.cherry_tensor<[12x2048x768xf32]>
    %w3 = cherry.create_tensor dense<11.0> : tensor<12x768x2048xf32> -> !cherry.cherry_tensor<[12x768x2048xf32]>
    
    // Final Weights
    %rms_final = cherry.create_tensor dense<12.0> : tensor<768xf32> -> !cherry.cherry_tensor<[768xf32]>
    %wcls = cherry.create_tensor dense<13.0> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
    
    // KV Cache (Initialize with Zeros)
    %k_cache_init = cherry.create_tensor dense<0.0> : tensor<12x1024x768xf32> -> !cherry.cherry_tensor<[12x1024x768xf32]>
    %v_cache_init = cherry.create_tensor dense<0.0> : tensor<12x1024x768xf32> -> !cherry.cherry_tensor<[12x1024x768xf32]>
    
    %start_token = cherry.constant (1 : i64) : i64
    %start_pos = cherry.constant (0 : i64) : i64
    %max_len = cherry.constant (10 : i64) : i64   
    
    %final_token, %final_pos, %final_k_cache, %final_v_cache = scf.while (%arg_token = %start_token, %arg_pos = %start_pos, %arg_k_cache = %k_cache_init, %arg_v_cache = %v_cache_init) 
          : (i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) -> (i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) {
        %cond = arith.cmpi slt, %arg_pos, %max_len : i64
        scf.condition(%cond) %arg_token, %arg_pos, %arg_k_cache, %arg_v_cache : i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
      } do {
        ^bb0(%curr_token: i64, %curr_pos: i64, %curr_k_cache: !cherry.cherry_tensor<[12x1024x768xf32]>, %curr_v_cache: !cherry.cherry_tensor<[12x1024x768xf32]>):
        
        %mask = cherry.generate_mask %curr_pos, [1, 1024] : !cherry.cherry_tensor<[1x1024xf32]>

        %logits, %next_k_cache, %next_v_cache = cherry.call @llama_forward(
            %curr_token, %curr_pos, %curr_k_cache, %curr_v_cache, %mask,
            %embedding,
            %rms_att, %wq, %wk, %wv, %wo,
            %rms_ffn, %w1, %w2, %w3,
            %rms_final, %wcls
        ) : (i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>,!cherry.cherry_tensor<[1x1024xf32]>,
             !cherry.cherry_tensor<[32000x768xf32]>,
             !cherry.cherry_tensor<[12x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>,
             !cherry.cherry_tensor<[12x768xf32]>, !cherry.cherry_tensor<[12x768x2048xf32]>, !cherry.cherry_tensor<[12x2048x768xf32]>, !cherry.cherry_tensor<[12x768x2048xf32]>,
             !cherry.cherry_tensor<[768xf32]>, !cherry.cherry_tensor<[32000x768xf32]>) 
            -> (!cherry.cherry_tensor<[1x32000xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>)
            
        %arg_max = cherry.argmax %logits dim 1 : (!cherry.cherry_tensor<[1x32000xf32]>) -> (!cherry.cherry_tensor<[1xi64]>)
        cherry.print %logits : !cherry.cherry_tensor<[1x32000xf32]>
        // cherry.print %arg_max : !cherry.cherry_tensor<[1xi64]>
        
        %zero = cherry.constant (0 : i64) : i64
        %next_token = cherry.tensor_get %arg_max [%zero] : (!cherry.cherry_tensor<[1xi64]>, i64) -> i64
        
        %c1 = cherry.constant (1 : i64) : i64
        %next_pos = arith.addi %curr_pos, %c1 : i64
        
        scf.yield %next_token, %next_pos, %next_k_cache, %next_v_cache : i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
      }
    //cherry.print %final_k_cache : !cherry.cherry_tensor<[12x1024x768xf32]>
    //cherry.print %final_v_cache : !cherry.cherry_tensor<[12x1024x768xf32]>
    cherry.return 
  }
}
