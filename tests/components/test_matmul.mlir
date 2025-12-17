module {
  cherry.func @host() {
    %lhs = cherry.create_tensor dense<0.5> : tensor<1x768xf32> -> !cherry.cherry_tensor<[1x768xf32]>
    %lhs_1 = cherry.create_tensor dense<0.7> : tensor<1x768xf32> -> !cherry.cherry_tensor<[1x768xf32]>
    %lhs_2 = cherry.create_tensor dense<0.1> : tensor<1x768xf32> -> !cherry.cherry_tensor<[1x768xf32]>
    %add = cherry.tensor_add %lhs, %lhs_1 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
    %sub = cherry.tensor_sub %add, %lhs_2 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[1x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>

    %rhs = cherry.create_tensor dense<
      0.1
    > : tensor<768x768xf32> -> !cherry.cherry_tensor<[768x768xf32]>

    
    %result = cherry.matmul %sub, %rhs : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
    
    cherry.print %result : !cherry.cherry_tensor<[1x768xf32]>
    
    cherry.return 
  }
}
