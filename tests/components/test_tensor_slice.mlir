module {
  cherry.func private @test_slice(%arg0: !cherry.cherry_tensor<[4x4xf32]>, %offset_x: i64) -> !cherry.cherry_tensor<[?xf32]> {
    %c0 = cherry.constant(0 : i64) : i64
    %c2 = cherry.constant(2 : i64) : i64

    %0 = cherry.tensor_slice %arg0[%c0, %c2, %offset_x, %c2]
         : (!cherry.cherry_tensor<[4x4xf32]>, i64, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>

    cherry.return %0 : !cherry.cherry_tensor<[?xf32]>
  }

  cherry.func @host() -> !cherry.cherry_tensor<[?xf32]>{
    %input = cherry.create_tensor dense<5.000000e-01> : tensor<4x4xf32> -> !cherry.cherry_tensor<[4x4xf32]>

    %c0_idx = arith.constant 0 : index
    %c2_idx = arith.constant 2 : index
    %c4_idx = arith.constant 4 : index
    
      
      %offset_i64 = arith.index_cast %c0_idx : index to i64
      
      %result = cherry.call @test_slice(%input, %offset_i64) 
                : (!cherry.cherry_tensor<[4x4xf32]>, i64) -> !cherry.cherry_tensor<[?xf32]>
    
    cherry.return %result : !cherry.cherry_tensor<[?xf32]>
  }
}
