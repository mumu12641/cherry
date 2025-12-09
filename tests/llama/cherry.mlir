module {
  cherry.func private @llama_forward(%token_embedding_table: !cherry.cherry_tensor<[32000x768xf32]>, %x: i64) -> !cherry.cherry_tensor<[?xf32]> {
    %zero = cherry.constant (0 : i64) : i64
    %one = cherry.constant (1 : i64) : i64
    %dim = cherry.constant (768 : i64) : i64
    %token_embedding = cherry.tensor_slice %token_embedding_table[%x, %one, %zero, %dim] : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %token_embedding : !cherry.cherry_tensor<[?xf32]>
  }

  cherry.func @host() -> !cherry.cherry_tensor<[?xf32]>{
    %token_embedding_table = cherry.create_tensor dense<5.000000e-01> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
    %kv_cache_init = cherry.create_tensor dense<0.0> : tensor<32x2048x128xf32> -> !cherry.cherry_tensor<[32x2048x128xf32]>
    %x = cherry.constant (985 : i64) : i64
    %embedding = cherry.call @llama_forward(%token_embedding_table, %x) : (!cherry.cherry_tensor<[32000x768xf32]>, i64) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %embedding : !cherry.cherry_tensor<[?xf32]>
  }
}
