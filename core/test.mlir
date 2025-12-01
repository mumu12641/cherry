module {
  cherry.func @simple_transformer_block(%arg0: !cherry.cherry_tensor<[?x?x8xf32]>, %arg1: !cherry.cherry_tensor<[8x8xf32]>, %arg2: !cherry.cherry_tensor<[8x8xf32]>, %arg3: !cherry.cherry_tensor<[8x8xf32]>, %arg4: !cherry.cherry_tensor<[8x32xf32]>, %arg5: !cherry.cherry_tensor<[32x8xf32]>, %arg6: !cherry.cherry_tensor<[8xf32]>, %arg7: !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.matmul %arg0, %arg1 : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %1 = cherry.matmul %arg0, %arg2 : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %2 = cherry.matmul %arg0, %arg3 : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %3 = cherry.constant(0 : i64) : i64
    %4 = cherry.constant(2 : i64) : i64
    %5 = cherry.constant(1 : i64) : i64
    %6 = cherry.transpose %1, %3, %4, %5 : (!cherry.cherry_tensor<[?x?x8xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?x8x?xf32]>
    %7 = cherry.matmul %0, %6 : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[?x8x?xf32]>) -> !cherry.cherry_tensor<[?x?x?xf32]>
    %8 = cherry.create_tensor dense<2.828400e+00> : tensor<1xf32> -> !cherry.cherry_tensor<[1xf32]>
    %9 = cherry.constant(1 : i64) : i64
    %10 = cherry.constant(4 : i64) : i64
    %11 = cherry.broadcast %8, %9, %10, %10 : (!cherry.cherry_tensor<[1xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?x?x?xf32]>
    %12 = cherry.tensor_div %7, %11 : (!cherry.cherry_tensor<[?x?x?xf32]>, !cherry.cherry_tensor<[?x?x?xf32]>) -> !cherry.cherry_tensor<[?x?x?xf32]>
    %13 = cherry.softmax %12 axis 2 : (!cherry.cherry_tensor<[?x?x?xf32]>) -> !cherry.cherry_tensor<[?x?x?xf32]>
    %14 = cherry.matmul %13, %2 : (!cherry.cherry_tensor<[?x?x?xf32]>, !cherry.cherry_tensor<[?x?x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %15 = cherry.tensor_add %arg0, %14 : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[?x?x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %16 = cherry.layernorm %15, %arg6, %arg7 eps 9.99999974E-6 : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8xf32]>, !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %17 = cherry.matmul %16, %arg4 : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8x32xf32]>) -> !cherry.cherry_tensor<[?x?x32xf32]>
    %18 = cherry.tensor_relu %17 : (!cherry.cherry_tensor<[?x?x32xf32]>) -> !cherry.cherry_tensor<[?x?x32xf32]>
    %19 = cherry.matmul %18, %arg5 : (!cherry.cherry_tensor<[?x?x32xf32]>, !cherry.cherry_tensor<[32x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %20 = cherry.tensor_add %16, %19 : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[?x?x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %21 = cherry.layernorm %20, %arg6, %arg7 eps 9.99999974E-6 : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8xf32]>, !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    cherry.return %21 : !cherry.cherry_tensor<[?x?x8xf32]>
  }
  cherry.func @main() {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %1 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %2 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %3 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %4 = cherry.create_tensor dense<2.000000e-01> : tensor<8x32xf32> -> !cherry.cherry_tensor<[8x32xf32]>
    %5 = cherry.create_tensor dense<2.000000e-01> : tensor<32x8xf32> -> !cherry.cherry_tensor<[32x8xf32]>
    %6 = cherry.create_tensor dense<1.000000e+00> : tensor<8xf32> -> !cherry.cherry_tensor<[8xf32]>
    %7 = cherry.create_tensor dense<0.000000e+00> : tensor<8xf32> -> !cherry.cherry_tensor<[8xf32]>
    %8 = cherry.call @simple_transformer_block(%0, %1, %2, %3, %4, %5, %6, %7) : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>, !cherry.cherry_tensor<[8x32xf32]>, !cherry.cherry_tensor<[32x8xf32]>, !cherry.cherry_tensor<[8xf32]>, !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return
  }
}
