// Original IR loaded from file
module {
  cherry.func private @test_slice(%arg0: !cherry.cherry_tensor<[4x4xf32]>, %arg1: i64) -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.constant(0 : i64) : i64
    %1 = cherry.constant(2 : i64) : i64
    %2 = cherry.tensor_slice %arg0[%0, %arg1] sizes [2, 2] : (!cherry.cherry_tensor<[4x4xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %2 : !cherry.cherry_tensor<[?xf32]>
  }
  cherry.func @host() -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<4x4xf32> -> !cherry.cherry_tensor<[4x4xf32]>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %1 = arith.index_cast %c0 : index to i64
    %2 = cherry.call @test_slice(%0, %1) : (!cherry.cherry_tensor<[4x4xf32]>, i64) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %2 : !cherry.cherry_tensor<[?xf32]>
  }
}

// ==========================================
// Phase: Inliner
// ==========================================
module {
  cherry.func @host() -> !cherry.cherry_tensor<[?xf32]> {
    %c0_i64 = arith.constant 0 : i64
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<4x4xf32> -> !cherry.cherry_tensor<[4x4xf32]>
    %1 = cherry.constant(0 : i64) : i64
    %2 = cherry.tensor_slice %0[%1, %c0_i64] sizes [2, 2] : (!cherry.cherry_tensor<[4x4xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %2 : !cherry.cherry_tensor<[?xf32]>
  }
}

// ==========================================
// Phase: Canonicalizer
// ==========================================
module {
  cherry.func @host() -> !cherry.cherry_tensor<[?xf32]> {
    %c0_i64 = arith.constant 0 : i64
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<4x4xf32> -> !cherry.cherry_tensor<[4x4xf32]>
    %1 = cherry.constant(0 : i64) : i64
    %2 = cherry.tensor_slice %0[%1, %c0_i64] sizes [2, 2] : (!cherry.cherry_tensor<[4x4xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %2 : !cherry.cherry_tensor<[?xf32]>
  }
}
