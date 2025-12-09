module {
  // ===========================================================================
  // Kernel: 执行单步推理
  // 输入: 
  //   %token_id: 当前输入的 token (i64)
  //   %pos: 当前的位置索引 (i64) - 用于 KV Cache 和 RoPE
  //   %kv_cache: 预分配好的 KV Cache (传递引用/tensor)
  //   %weights: 权重表
  // 输出:
  //   %logits: 下一个 token 的概率分布 (1x32000)
  //   %new_kv_cache: 更新后的 KV Cache (如果是函数式语义)
  // ===========================================================================
  cherry.func private @llama_forward(
      %token_id: i64, 
      %pos: i64,
      %kv_cache: !cherry.cherry_tensor<[32x2048x128xf32]>, 
      %embedding_table: !cherry.cherry_tensor<[32000x768xf32]>
  ) -> (!cherry.cherry_tensor<[1x32000xf32]>) {
    
    // Constants
    %c0 = cherry.constant (0 : i64) : i64
    %c1 = cherry.constant (1 : i64) : i64
    %dim = cherry.constant (768 : i64) : i64
    
    // 1. Embedding Lookup
    // slice table[token_id, 0] size [1, 768]
    // 这里的 token_id 是动态 offset，但 size 是静态的
    %x = cherry.tensor_slice %embedding_table[%token_id, %c1, %c0, %dim] 
         : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>

    // ... 这里会有 Transformer Layers ...
    // ... Attention 计算中会用到 %pos 来读取/写入 %kv_cache ...
    
    // 假设我们计算出了新的 K 值 %new_k (1x1x128)
    // 我们需要把它写入 KV Cache 的 [layer_id, pos, :] 位置
    // %updated_cache = cherry.tensor_set_slice %kv_cache, %new_k, %layer_id, %pos
    
    // ... 最后计算 Logits ...
    // 模拟输出 logits
    %logits = cherry.create_tensor dense<0.0> : tensor<1x32000xf32> -> !cherry.cherry_tensor<[1x32000xf32]>
    
    // 返回 logits 和 更新后的 cache
    cherry.return  %kv_cache : !cherry.cherry_tensor<[32x2048x128xf32]>
  }

  // ===========================================================================
  // Host: 模拟推理循环
  // ===========================================================================
  cherry.func @host() -> !cherry.cherry_tensor<[1xf32]> {
      // 1. 加载/初始化权重
      %token_embedding_table = cherry.create_tensor dense<0.5> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
      
      // 2. 初始化 KV Cache (全 0, 最大长度 2048)
      // 这是一个巨大的静态 Tensor
      %kv_cache_init = cherry.create_tensor dense<0.0> : tensor<32x2048x128xf32> -> !cherry.cherry_tensor<[32x2048x128xf32]>

      // 3. 初始状态
      %start_token = cherry.constant (1 : i64) : i64 // BOS token
      %start_pos = cherry.constant (0 : i64) : i64
      %max_len = cherry.constant (10 : i64) : i64   // 生成 10 个 token

      // 4. 自回归循环 (While Loop)
      // iter_args: [current_token, current_pos, current_kv_cache]
      %final_token, %final_pos, %final_cache = scf.while (%arg_token = %start_token, %arg_pos = %start_pos, %arg_cache = %kv_cache_init) 
          : (i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>) -> (i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>) {
        
        // Condition: pos < max_len
        %cond = arith.cmpi slt, %arg_pos, %max_len : i64
        scf.condition(%cond) %arg_token, %arg_pos, %arg_cache : i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>
      
      } do {
      ^bb0(%curr_token: i64, %curr_pos: i64, %curr_cache: !cherry.cherry_tensor<[32x2048x128xf32]>):
        
        // A. 调用 Kernel
        // 注意：这里传入了 curr_pos，Kernel 内部会利用它做 tensor_slice/set_slice
        %logits, %next_cache = cherry.call @llama_forward(%curr_token, %curr_pos, %curr_cache, %token_embedding_table) 
            : (i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>, !cherry.cherry_tensor<[32000x768xf32]>) 
            -> !cherry.cherry_tensor<[32x2048x128xf32]>

        // B. 采样 (Sampling) - 这里简化为 Argmax
        // 实际中这里可能是一个 cherry.argmax op
        // 假设我们得到了下一个 token ID
        %next_token = cherry.constant (999 : i64) : i64 // Mock next token
        
        // C. 更新位置
        %c1 = cherry.constant (1 : i64) : i64
        %next_pos = arith.addi %curr_pos, %c1 : i64

        // D. 传递给下一次迭代
        scf.yield %next_token, %next_pos, %next_cache : i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>
      }

      // 这里为了演示返回空，实际可能返回生成的 token 序列
      %result = cherry.create_tensor dense<0.0> : tensor<1xf32> -> !cherry.cherry_tensor<[1xf32]>
      cherry.return %result : !cherry.cherry_tensor<[1xf32]>
  }
}
