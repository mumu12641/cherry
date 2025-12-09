module {
  cherry.func private @test_set_slice(%dest: !cherry.cherry_tensor<[12x1024x768xf32]>,%source: !cherry.cherry_tensor<[1x1x768xf32]>, %i: i64, %j: i64) -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.tensor_set_slice %dest[%i, %j], %source : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %0 : !cherry.cherry_tensor<[?xf32]>
  }

  cherry.func @host() -> !cherry.cherry_tensor<[?xf32]>{
    %dest = cherry.create_tensor dense<5.000000e-01> : tensor<12x1024x768xf32> -> !cherry.cherry_tensor<[12x1024x768xf32]>
    %source = cherry.create_tensor dense<3.000000e-01> : tensor<1x1x768xf32> -> !cherry.cherry_tensor<[1x1x768xf32]>

    %i = cherry.constant(2 : i64) : i64
    %j = cherry.constant(512 : i64) : i64
    
    %result = cherry.call @test_set_slice(%dest, %source, %i, %j) 
                : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    
    cherry.return %result : !cherry.cherry_tensor<[?xf32]>
  }
}
