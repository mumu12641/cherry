
module {
  cherry.func @test_tensor_get_set() -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.constant(0 : i64) : i64
    %1 = cherry.constant(1 : i64) : i64
    %2 = cherry.constant(2 : i64) : i64
    %input = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %get = cherry.tensor_get %input[%0, %1, %2] : (!cherry.cherry_tensor<[1x4x8xf32]>, i64, i64, i64) -> f32
    %set = cherry.tensor_set %input[%0, %1, %2], %get : (!cherry.cherry_tensor<[1x4x8xf32]>, i64, i64, i64, f32) -> !cherry.cherry_tensor<[1x4x8xf32]>
    cherry.return %set : !cherry.cherry_tensor<[1x4x8xf32]>
  }
}
