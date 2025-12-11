module {
  func.func @host() {
    %input = cherry.create_tensor dense<[[1.0, 2.0, 3.0, 4.0]]> : tensor<1x4xf32> -> !cherry.cherry_tensor<[1x4xf32]>

    %pos = cherry.constant (1 : i64) : i64

    %output = cherry.rope %input, %pos 
        : (!cherry.cherry_tensor<[1x4xf32]>, i64) -> !cherry.cherry_tensor<[1x4xf32]>

    cherry.print %output : !cherry.cherry_tensor<[1x4xf32]>
    return
  }
}
