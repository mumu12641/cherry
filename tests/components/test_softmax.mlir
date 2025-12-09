module {
  cherry.func private @test_softmax(%input: !cherry.cherry_tensor<[1x1024x768xf32]>) -> (!cherry.cherry_tensor<[?xi64]>, !cherry.cherry_tensor<[?xf32]>) {
    %0 = cherry.argmax %input dim 1 : (!cherry.cherry_tensor<[1x1024x768xf32]>) -> !cherry.cherry_tensor<[?xi64]>
    %1 = cherry.softmax %input axis 1 : (!cherry.cherry_tensor<[1x1024x768xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %0, %1 : !cherry.cherry_tensor<[?xi64]>, !cherry.cherry_tensor<[?xf32]>
  }

  cherry.func @host() -> (!cherry.cherry_tensor<[?xi64]>, !cherry.cherry_tensor<[?xf32]>){
    %input = cherry.create_tensor dense<3.000000e-01> : tensor<1x1024x768xf32> -> !cherry.cherry_tensor<[1x1024x768xf32]>

    %result, %softmax = cherry.call @test_softmax(%input)
                : (!cherry.cherry_tensor<[1x1024x768xf32]>) -> (!cherry.cherry_tensor<[?xi64]>, !cherry.cherry_tensor<[?xf32]>)

    cherry.return %result, %softmax : !cherry.cherry_tensor<[?xi64]>, !cherry.cherry_tensor<[?xf32]>
  }
}
