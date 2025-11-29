module {
  func.func @simple_transformer_block(%arg0: !cherry.cherry_tensor<[1x4x8xf32]>, %arg1: !cherry.cherry_tensor<[8x8xf32]>, %arg2: !cherry.cherry_tensor<[8x8xf32]>, %arg3: !cherry.cherry_tensor<[8x8xf32]>, %arg4: !cherry.cherry_tensor<[8x32xf32]>, %arg5: !cherry.cherry_tensor<[32x8xf32]>, %arg6: !cherry.cherry_tensor<[8xf32]>, %arg7: !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]> {
    %0 = cherry.matmul %arg0, %arg1 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %1 = cherry.matmul %arg0, %arg2 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %2 = cherry.matmul %arg0, %arg3 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %3 = cherry.constant(0 : i64) : i64
    %4 = cherry.constant(2 : i64) : i64
    %5 = cherry.constant(1 : i64) : i64
    %6 = cherry.transpose %1, %3, %4, %5 : (!cherry.cherry_tensor<[1x4x8xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x8x4xf32]>
    %7 = cherry.matmul %0, %6 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[1x8x4xf32]>) -> !cherry.cherry_tensor<[1x4x4xf32]>
    %8 = cherry.create_tensor dense<2.828400e+00> : tensor<1xf32> -> !cherry.cherry_tensor<[1xf32]>
    %9 = cherry.create_tensor dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32> -> !cherry.cherry_tensor<[2x3xf32]>
    %10 = cherry.constant(1 : i64) : i64
    %11 = cherry.constant(4 : i64) : i64
    %12 = cherry.constant(4 : i64) : i64
    %13 = cherry.broadcast %8, %10, %11, %12 : (!cherry.cherry_tensor<[1xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x4x4xf32]>
    %14 = cherry.tensor_div %7, %13 : (!cherry.cherry_tensor<[1x4x4xf32]>, !cherry.cherry_tensor<[1x4x4xf32]>) -> !cherry.cherry_tensor<[1x4x4xf32]>
    %15 = cherry.softmax %14 axis 2 : (!cherry.cherry_tensor<[1x4x4xf32]>) -> !cherry.cherry_tensor<[1x4x4xf32]>
    %16 = cherry.matmul %15, %2 : (!cherry.cherry_tensor<[1x4x4xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %17 = cherry.tensor_add %arg0, %16 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %18 = cherry.layernorm %17, %arg6, %arg7 eps 9.99999974E-6 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8xf32]>, !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %19 = cherry.matmul %18, %arg4 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8x32xf32]>) -> !cherry.cherry_tensor<[1x4x32xf32]>
    %20 = cherry.tensor_relu %19 : (!cherry.cherry_tensor<[1x4x32xf32]>) -> !cherry.cherry_tensor<[1x4x32xf32]>
    %21 = cherry.matmul %20, %arg5 : (!cherry.cherry_tensor<[1x4x32xf32]>, !cherry.cherry_tensor<[32x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %22 = cherry.tensor_add %18, %21 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %23 = cherry.layernorm %22, %arg6, %arg7 eps 9.99999974E-6 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8xf32]>, !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    return %23 : !cherry.cherry_tensor<[1x4x8xf32]>
  }
}
