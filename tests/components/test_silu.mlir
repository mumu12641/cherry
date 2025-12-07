module {
  cherry.func @test_silu() -> !cherry.cherry_tensor<[?xf32]> {
    %input = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %output = cherry.tensor_silu %input : (!cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %output : !cherry.cherry_tensor<[?xf32]>
  }
}
