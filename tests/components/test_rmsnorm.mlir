module {
  cherry.func @test_rmsnorm() -> !cherry.cherry_tensor<[?xf32]> {
    %input = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %scale = cherry.create_tensor dense<1.0> : tensor<8xf32> -> !cherry.cherry_tensor<[8xf32]>
    %output = cherry.rmsnorm %input scale %scale eps 1.000000e-05 : !cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8xf32]> -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %output : !cherry.cherry_tensor<[?xf32]>
  }
}
