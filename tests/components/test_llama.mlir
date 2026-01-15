module {


  cherry.func @llama_forward(
      %pos: i64,
      %q: !cherry.cherry_tensor<[1x768xf32]>,
      %k_layer_full: !cherry.cherry_tensor<[1024x768xf32]>,
      %v_layer_full: !cherry.cherry_tensor<[1024x768xf32]>
      
  ) -> () {
        %c0_i64 = cherry.constant (0 : i64) : i64
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
        %valid_len = cherry.scalar_add %pos, %constant_1 : (i64, i64) -> i64
        %h_i64 = cherry.constant (0 : i64) : i64
    
            %offset = arith.muli %h_i64, %head_dim : i64

            // ---------------------------------------------------------
            //  Q Head [1, 64]
            // ---------------------------------------------------------
            %q_head = cherry.tensor_slice %q[%c0_i64, %offset] sizes [1, 64] {squeeze = false}
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
            %score = cherry.masked_matmul %q_head, %k_head_T, %valid_len
                : (!cherry.cherry_tensor<[1x64xf32]>, !cherry.cherry_tensor<[64x1024xf32]>, i64)
                -> !cherry.cherry_tensor<[1x1024xf32]>

            %score_scaled = cherry.tensor_mul_scalar %score, %scale_val
                : (!cherry.cherry_tensor<[1x1024xf32]>, f32) -> !cherry.cherry_tensor<[1x1024xf32]>

            %probs = cherry.softmax %score_scaled axis 1
                : (!cherry.cherry_tensor<[1x1024xf32]>) -> !cherry.cherry_tensor<[1x1024xf32]>

            %v_head = cherry.tensor_slice %v_layer_full[%c0_i64, %offset] sizes [1024, 64]
                : (!cherry.cherry_tensor<[1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x64xf32]>

            %context_head = cherry.matmul %probs, %v_head
                : (!cherry.cherry_tensor<[1x1024xf32]>, !cherry.cherry_tensor<[1024x64xf32]>) -> !cherry.cherry_tensor<[1x64xf32]>

            %context_head_reshape = cherry.reshape %context_head shape [1, 1, 64]
                : (!cherry.cherry_tensor<[1x64xf32]>) -> !cherry.cherry_tensor<[1x1x64xf32]>

    cherry.print %context_head_reshape : !cherry.cherry_tensor<[1x1x64xf32]>

    cherry.return
  }
}
