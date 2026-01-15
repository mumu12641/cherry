module {
  cherry.func @host(%arg0: !cherry.cherry_tensor<[1x768xf32]>, %arg1: !cherry.cherry_tensor<[1x768xf32]>, %arg2: !cherry.cherry_tensor<[1x768xf32]>, %arg3: !cherry.cherry_tensor<[768x768xf32]>) {
    // %lhs = cherry.create_tensor dense<0.5> : tensor<1x768xf32> -> !cherry.cherry_tensor<[1x768xf32]>
    %add = cherry.tensor_add %arg0, %arg1 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
    %sub = cherry.tensor_sub %add, %arg2 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
// 
    // %rhs = cherry.create_tensor dense<
    //   0.1
    // > : tensor<768x768xf32> -> !cherry.cherry_tensor<[768x768xf32]>

    
    %result = cherry.matmul %sub, %arg3 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
    
    cherry.print %result : !cherry.cherry_tensor<[1x768xf32]>
    
    cherry.return 
  }
}
