module {
  cherry.func private @llama_forward(
      %token_id: i64, 
      %pos: i64,
      %kv_cache: !cherry.cherry_tensor<[32x2048x128xf32]>, 
      %embedding_table: !cherry.cherry_tensor<[32000x768xf32]>
  ) -> (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) {    
    %zero = cherry.constant (0 : i64) : i64
    
    %token_embedding = cherry.tensor_slice %embedding_table[%token_id, %zero] sizes[1, 768] : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    %logits_weight = cherry.create_tensor dense<3214.0> : tensor<768x32000xf32> -> !cherry.cherry_tensor<[768x32000xf32]>
    %logits = cherry.matmul %token_embedding, %logits_weight : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[768x32000xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    // %kv_cache_new = cherry.create_tensor dense<0.0> : tensor<32x2048x128xf32> -> !cherry.cherry_tensor<[32x2048x128xf32]>
    
    %kv_cache_weight = cherry.create_tensor dense<12342.0> : tensor<128x128xf32> -> !cherry.cherry_tensor<[128x128xf32]>
    %kv_cache_new = cherry.matmul %kv_cache, %kv_cache_weight : (!cherry.cherry_tensor<[32x2048x128xf32]>, !cherry.cherry_tensor<[128x128xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    
    cherry.return  %logits, %kv_cache_new : !cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>
  }

  cherry.func @host(){
    %token_embedding_table = cherry.create_tensor dense<5.000000e-01> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
    %kv_cache_init = cherry.create_tensor dense<0.0> : tensor<32x2048x128xf32> -> !cherry.cherry_tensor<[32x2048x128xf32]>
    
    %start_token = cherry.constant (1 : i64) : i64 // BOS token
    %start_pos = cherry.constant (0 : i64) : i64
    %max_len = cherry.constant (10 : i64) : i64   // 生成 10 个 token
    
    %final_token, %final_pos, %final_cache = scf.while (%arg_token = %start_token, %arg_pos = %start_pos, %arg_cache = %kv_cache_init) 
          : (i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>) -> (i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>) {
        %cond = arith.cmpi slt, %arg_pos, %max_len : i64
        scf.condition(%cond) %arg_token, %arg_pos, %arg_cache : i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>
      } do {
        ^bb0(%curr_token: i64, %curr_pos: i64, %curr_cache: !cherry.cherry_tensor<[32x2048x128xf32]>):
        %logits, %next_cache = cherry.call @llama_forward(%curr_token, %curr_pos, %curr_cache, %token_embedding_table)
            : (i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>, !cherry.cherry_tensor<[32000x768xf32]>) 
            -> (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[32x2048x128xf32]>)
        
        // cherry.print %logits : !cherry.cherry_tensor<[?xf32]>
            
        %arg_max = cherry.argmax %logits dim 1 : (!cherry.cherry_tensor<[?xf32]>) -> (!cherry.cherry_tensor<[?xi64]>)
        
        %zero = cherry.constant (0 : i64) : i64
        %next_token = cherry.tensor_get %arg_max [%zero] : (!cherry.cherry_tensor<[?xi64]>, i64) -> i64
        
        %c1 = cherry.constant (1 : i64) : i64
        %next_pos = arith.addi %curr_pos, %c1 : i64
        scf.yield %next_token, %next_pos, %next_cache : i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>
      }
    cherry.print %final_cache : !cherry.cherry_tensor<[32x2048x128xf32]>
    cherry.return 
  }
}
