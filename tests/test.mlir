module {
cherry.func @test_float_scalar(%arg0: f32, %arg1: f32) -> () {
    %0 = cherry.scalar_add %arg0, %arg1 : (f32, f32) -> f32
    %1 = cherry.scalar_sub %0, %arg1 : (f32, f32) -> f32
    %2 = cherry.scalar_mul %1, %arg0 : (f32, f32) -> f32
    %3 = cherry.scalar_div %2, %arg1 : (f32, f32) -> f32
    cherry.return
  }

  cherry.func @test_int_scalar(%arg0: i32, %arg1: i32) -> () {
    %0 = cherry.scalar_add %arg0, %arg1 : (i32, i32) -> i32
    %1 = cherry.scalar_sub %0, %arg1 : (i32, i32) -> i32
    %2 = cherry.scalar_mul %1, %arg0 : (i32, i32) -> i32
    %3 = cherry.scalar_div %2, %arg1 : (i32, i32) -> i32

    cherry.return
  }
  cherry.func @main() -> !cherry.cherry_tensor<[8xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %1 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %2 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %3 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %4 = cherry.create_tensor dense<2.000000e-01> : tensor<8x32xf32> -> !cherry.cherry_tensor<[8x32xf32]>
    %5 = cherry.create_tensor dense<2.000000e-01> : tensor<32x8xf32> -> !cherry.cherry_tensor<[32x8xf32]>
    %6 = cherry.create_tensor dense<1.000000e+00> : tensor<8xf32> -> !cherry.cherry_tensor<[8xf32]>
    %7 = cherry.create_tensor dense<0.000000e+00> : tensor<8xf32> -> !cherry.cherry_tensor<[8xf32]>
    %8 = cherry.tensor_add %6, %7 : (!cherry.cherry_tensor<[8xf32]>, !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[8xf32]>
    %9 = cherry.tensor_sub %6, %7 : (!cherry.cherry_tensor<[8xf32]>, !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[8xf32]>
    %10 = cherry.tensor_mul %6, %7 : (!cherry.cherry_tensor<[8xf32]>, !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[8xf32]>
    %11 = cherry.tensor_div %6, %7 : (!cherry.cherry_tensor<[8xf32]>, !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[8xf32]>

    %12 = cherry.create_tensor dense<[-1.0, 0.0, 1.0, 2.0]> : tensor<4xf32> -> !cherry.cherry_tensor<[4xf32]>
    %13 = cherry.tensor_neg %12 : (!cherry.cherry_tensor<[4xf32]>) -> !cherry.cherry_tensor<[4xf32]>
    %14 = cherry.tensor_exp %12 : (!cherry.cherry_tensor<[4xf32]>) -> !cherry.cherry_tensor<[4xf32]>
    %15 = cherry.tensor_tanh %12 : (!cherry.cherry_tensor<[4xf32]>) -> !cherry.cherry_tensor<[4xf32]>
    %16 = cherry.tensor_relu %0 : (!cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %17 = cherry.tensor_sigmoid %0 : (!cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>

    %18 = cherry.constant(0 : i64) : i64
    %19 = cherry.constant(2 : i64) : i64
    %20 = cherry.constant(1 : i64) : i64
    %21 = cherry.transpose %0, %18, %19, %20 : (!cherry.cherry_tensor<[1x4x8xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x8x4xf32]>

    %22 = cherry.constant(2 : i64) : i64
    %23 = cherry.constant(32 : i64) : i64
    %24 = cherry.reshape %1, %22, %23 : (!cherry.cherry_tensor<[8x8xf32]>, i64, i64) -> !cherry.cherry_tensor<[2x32xf32]>

    %25 = cherry.matmul %0, %1 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    
    %26 = cherry.create_tensor dense<1.000000e-01> : tensor<1x2x3x4x5x6x7x8xf32> -> !cherry.cherry_tensor<[1x2x3x4x5x6x7x8xf32]>
    %27 = cherry.create_tensor dense<1.000000e-01> : tensor<4x5x6x8x9xf32> -> !cherry.cherry_tensor<[4x5x6x8x9xf32]>
    %28 = cherry.matmul %26, %27 : (!cherry.cherry_tensor<[1x2x3x4x5x6x7x8xf32]>, !cherry.cherry_tensor<[4x5x6x8x9xf32]>) -> !cherry.cherry_tensor<[1x2x3x4x5x6x7x9xf32]>
    %29 = cherry.softmax %0 axis 2 : (!cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    
    %30= cherry.create_tensor dense<2.828400e+00> : tensor<2x2xf32> -> !cherry.cherry_tensor<[2x2xf32]>
    %31 = cherry.constant(2 : i64) : i64
    %32 = cherry.constant(4 : i64) : i64
    %33 = cherry.broadcast %30, %31, %31, %32, %32 : (!cherry.cherry_tensor<[2x2xf32]>, i64, i64, i64, i64) -> !cherry.cherry_tensor<[2x2x4x4xf32]>
    
    cherry.return %8 : !cherry.cherry_tensor<[8xf32]>
  }
}
