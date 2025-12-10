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
      %rms_final_weight: !cherry.cherry_tensor<[1x768xf32]>,
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
    %head_num = cherry.constant (12 : i64) : i64
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
        %q = cherry.matmul %xb, %wq_layer : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %k = cherry.matmul %xb, %wk_layer : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %v = cherry.matmul %xb, %wv_layer : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>

        // A.3 RoPE (Rotary Positional Embedding)
        // TODO
        
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
        
        scf.yield %q, %next_k_cache, %next_v_cache : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
    }
    %wcls_reshape = cherry.reshape %wcls, %dim, %vocab_size
        : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x32000xf32]>
    %logits = cherry.matmul %x_final, %wcls_reshape
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
    %rms_final = cherry.create_tensor dense<12.0> : tensor<1x768xf32> -> !cherry.cherry_tensor<[1x768xf32]>
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
            %curr_token, %curr_pos, %curr_k_cache, %curr_v_cache, %mask
            %embedding,
            %rms_att, %wq, %wk, %wv, %wo,
            %rms_ffn, %w1, %w2, %w3,
            %rms_final, %wcls
        ) : (i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>,!cherry.cherry_tensor<[1x1024xf32]>,
             !cherry.cherry_tensor<[32000x768xf32]>,
             !cherry.cherry_tensor<[12x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>,
             !cherry.cherry_tensor<[12x768xf32]>, !cherry.cherry_tensor<[12x768x2048xf32]>, !cherry.cherry_tensor<[12x2048x768xf32]>, !cherry.cherry_tensor<[12x768x2048xf32]>,
             !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[32000x768xf32]>) 
            -> (!cherry.cherry_tensor<[1x32000xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>)
            
        %arg_max = cherry.argmax %logits dim 1 : (!cherry.cherry_tensor<[1x32000xf32]>) -> (!cherry.cherry_tensor<[1xi64]>)
        cherry.print %arg_max : !cherry.cherry_tensor<[1xi64]>
        
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
