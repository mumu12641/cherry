module {
  cherry.func @test_rmsnorm() -> !cherry.cherry_tensor<[?xf32]> {
    %input = cherry.create_tensor dense<5.000000e-01> : tensor<1x768xf32> -> !cherry.cherry_tensor<[1x768xf32]>
    %scale_1 = cherry.create_tensor dense<1.0> : tensor<1x768xf32> -> !cherry.cherry_tensor<[1x768xf32]>
    %d0 = cherry.constant (768 : i64) : i64
    %scale = cherry.reshape %scale_1, %d0: (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>

    %output = cherry.rmsnorm %input scale %scale eps 1.000000e-05 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %output : !cherry.cherry_tensor<[?xf32]>
  }
}
