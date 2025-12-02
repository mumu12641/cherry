module {
  cherry.func @main() -> !cherry.cherry_tensor<[?x?x8xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %1 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %2 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %3 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %4 = cherry.create_tensor dense<2.000000e-01> : tensor<8x32xf32> -> !cherry.cherry_tensor<[8x32xf32]>
    %5 = cherry.create_tensor dense<2.000000e-01> : tensor<32x8xf32> -> !cherry.cherry_tensor<[32x8xf32]>
    %6 = cherry.create_tensor dense<1.000000e+00> : tensor<8xf32> -> !cherry.cherry_tensor<[8xf32]>
    %7 = cherry.create_tensor dense<0.000000e+00> : tensor<8xf32> -> !cherry.cherry_tensor<[8xf32]>
    %8 = cherry.cast %0 : !cherry.cherry_tensor<[1x4x8xf32]> to !cherry.cherry_tensor<[?x?x8xf32]>
    %9 = cherry.matmul %8, %1 : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %10 = cherry.matmul %8, %2 : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %11 = cherry.matmul %8, %3 : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %12 = cherry.constant(0 : i64) : i64
    %13 = cherry.constant(2 : i64) : i64
    %14 = cherry.constant(1 : i64) : i64
    %15 = cherry.transpose %10, %12, %13, %14 : (!cherry.cherry_tensor<[?x?x8xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?x8x?xf32]>
    %16 = cherry.matmul %9, %15 : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[?x8x?xf32]>) -> !cherry.cherry_tensor<[?x?x?xf32]>
    %17 = cherry.create_tensor dense<2.828400e+00> : tensor<1xf32> -> !cherry.cherry_tensor<[1xf32]>
    %18 = cherry.constant(1 : i64) : i64
    %19 = cherry.constant(4 : i64) : i64
    %20 = cherry.broadcast %17, %18, %19, %19 : (!cherry.cherry_tensor<[1xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?x?x?xf32]>
    %21 = cherry.tensor_div %16, %20 : (!cherry.cherry_tensor<[?x?x?xf32]>, !cherry.cherry_tensor<[?x?x?xf32]>) -> !cherry.cherry_tensor<[?x?x?xf32]>
    %22 = cherry.softmax %21 axis 2 : (!cherry.cherry_tensor<[?x?x?xf32]>) -> !cherry.cherry_tensor<[?x?x?xf32]>
    %23 = cherry.matmul %22, %11 : (!cherry.cherry_tensor<[?x?x?xf32]>, !cherry.cherry_tensor<[?x?x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %24 = cherry.tensor_add %8, %23 : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[?x?x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %25 = cherry.layernorm %24, %6, %7 eps 9.99999974E-6 : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8xf32]>, !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %26 = cherry.matmul %25, %4 : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8x32xf32]>) -> !cherry.cherry_tensor<[?x?x32xf32]>
    %27 = cherry.tensor_relu %26 : (!cherry.cherry_tensor<[?x?x32xf32]>) -> !cherry.cherry_tensor<[?x?x32xf32]>
    %28 = cherry.matmul %27, %5 : (!cherry.cherry_tensor<[?x?x32xf32]>, !cherry.cherry_tensor<[32x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %29 = cherry.tensor_add %25, %28 : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[?x?x8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    %30 = cherry.layernorm %29, %6, %7 eps 9.99999974E-6 : (!cherry.cherry_tensor<[?x?x8xf32]>, !cherry.cherry_tensor<[8xf32]>, !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[?x?x8xf32]>
    cherry.return %30 : !cherry.cherry_tensor<[?x?x8xf32]>
  }
}
