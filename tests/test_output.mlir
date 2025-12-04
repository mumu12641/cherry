// Original IR loaded from file
module {
  cherry.func @test_float_scalar(%arg0: f32, %arg1: f32) {
    %0 = cherry.scalar_add %arg0, %arg1 : (f32, f32) -> f32
    %1 = cherry.scalar_sub %0, %arg1 : (f32, f32) -> f32
    %2 = cherry.scalar_mul %1, %arg0 : (f32, f32) -> f32
    %3 = cherry.scalar_div %2, %arg1 : (f32, f32) -> f32
    cherry.return
  }
  cherry.func @test_int_scalar(%arg0: i32, %arg1: i32) {
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
    %12 = cherry.create_tensor dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<4xf32> -> !cherry.cherry_tensor<[4xf32]>
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
    %30 = cherry.create_tensor dense<2.828400e+00> : tensor<2x2xf32> -> !cherry.cherry_tensor<[2x2xf32]>
    %31 = cherry.constant(2 : i64) : i64
    %32 = cherry.constant(4 : i64) : i64
    %33 = cherry.broadcast %30, %31, %31, %32, %32 : (!cherry.cherry_tensor<[2x2xf32]>, i64, i64, i64, i64) -> !cherry.cherry_tensor<[2x2x4x4xf32]>
    cherry.return %8 : !cherry.cherry_tensor<[8xf32]>
  }
}

// ==========================================
// Phase: Convert to Linalg
// ==========================================
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5, d6, d8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d3, d4, d5, d8, d7)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
module {
  func.func @test_float_scalar(%arg0: f32, %arg1: f32) {
    %0 = arith.addf %arg0, %arg1 : f32
    %1 = arith.subf %0, %arg1 : f32
    %2 = arith.mulf %1, %arg0 : f32
    %3 = arith.divf %2, %arg1 : f32
    return
  }
  func.func @test_int_scalar(%arg0: i32, %arg1: i32) {
    %0 = arith.addi %arg0, %arg1 : i32
    %1 = arith.subi %0, %arg1 : i32
    %2 = arith.muli %1, %arg0 : i32
    %3 = arith.divsi %2, %arg1 : i32
    return
  }
  func.func @main() -> tensor<8xf32> {
    %cst = arith.constant dense<5.000000e-01> : tensor<1x4x8xf32>
    %cst_0 = arith.constant dense<1.000000e-01> : tensor<8x8xf32>
    %cst_1 = arith.constant dense<1.000000e-01> : tensor<8x8xf32>
    %cst_2 = arith.constant dense<1.000000e-01> : tensor<8x8xf32>
    %cst_3 = arith.constant dense<2.000000e-01> : tensor<8x32xf32>
    %cst_4 = arith.constant dense<2.000000e-01> : tensor<32x8xf32>
    %cst_5 = arith.constant dense<1.000000e+00> : tensor<8xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %0 = tensor.empty() : tensor<8xf32>
    %1 = linalg.add ins(%cst_5, %cst_6 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) -> tensor<8xf32>
    %2 = tensor.empty() : tensor<8xf32>
    %3 = linalg.sub ins(%cst_5, %cst_6 : tensor<8xf32>, tensor<8xf32>) outs(%2 : tensor<8xf32>) -> tensor<8xf32>
    %4 = tensor.empty() : tensor<8xf32>
    %5 = linalg.mul ins(%cst_5, %cst_6 : tensor<8xf32>, tensor<8xf32>) outs(%4 : tensor<8xf32>) -> tensor<8xf32>
    %6 = tensor.empty() : tensor<8xf32>
    %7 = linalg.div ins(%cst_5, %cst_6 : tensor<8xf32>, tensor<8xf32>) outs(%6 : tensor<8xf32>) -> tensor<8xf32>
    %cst_7 = arith.constant dense<[-1.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<4xf32>
    %8 = tensor.empty() : tensor<4xf32>
    %9 = linalg.negf ins(%cst_7 : tensor<4xf32>) outs(%8 : tensor<4xf32>) -> tensor<4xf32>
    %10 = tensor.empty() : tensor<4xf32>
    %11 = linalg.exp ins(%cst_7 : tensor<4xf32>) outs(%10 : tensor<4xf32>) -> tensor<4xf32>
    %12 = tensor.empty() : tensor<4xf32>
    %13 = linalg.tanh ins(%cst_7 : tensor<4xf32>) outs(%12 : tensor<4xf32>) -> tensor<4xf32>
    %14 = tensor.empty() : tensor<1x4x8xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : tensor<1x4x8xf32>) outs(%14 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_17 = arith.constant 0.000000e+00 : f32
      %28 = arith.maximumf %in, %cst_17 : f32
      linalg.yield %28 : f32
    } -> tensor<1x4x8xf32>
    %16 = tensor.empty() : tensor<1x4x8xf32>
    %17 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : tensor<1x4x8xf32>) outs(%16 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_17 = arith.constant 1.000000e+00 : f32
      %28 = arith.negf %in : f32
      %29 = math.exp %28 : f32
      %30 = arith.addf %cst_17, %29 : f32
      %31 = arith.divf %cst_17, %30 : f32
      linalg.yield %31 : f32
    } -> tensor<1x4x8xf32>
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %18 = tensor.empty() : tensor<1x8x4xf32>
    %transposed = linalg.transpose ins(%cst : tensor<1x4x8xf32>) outs(%18 : tensor<1x8x4xf32>) permutation = [0, 2, 1] 
    %c2_i64_8 = arith.constant 2 : i64
    %c32_i64 = arith.constant 32 : i64
    %c2_i64_9 = arith.constant 2 : i64
    %c32_i64_10 = arith.constant 32 : i64
    %from_elements = tensor.from_elements %c2_i64_9, %c32_i64_10 : tensor<2xi64>
    %reshape = tensor.reshape %cst_0(%from_elements) : (tensor<8x8xf32>, tensor<2xi64>) -> tensor<2x32xf32>
    %19 = tensor.empty() : tensor<1x4x8xf32>
    %cst_11 = arith.constant 0.000000e+00 : f32
    %20 = linalg.fill ins(%cst_11 : f32) outs(%19 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %21 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%cst, %cst_0 : tensor<1x4x8xf32>, tensor<8x8xf32>) outs(%20 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_17: f32, %out: f32):
      %28 = arith.mulf %in, %in_17 : f32
      %29 = arith.addf %out, %28 : f32
      linalg.yield %29 : f32
    } -> tensor<1x4x8xf32>
    %cst_12 = arith.constant dense<1.000000e-01> : tensor<1x2x3x4x5x6x7x8xf32>
    %cst_13 = arith.constant dense<1.000000e-01> : tensor<4x5x6x8x9xf32>
    %22 = tensor.empty() : tensor<1x2x3x4x5x6x7x9xf32>
    %cst_14 = arith.constant 0.000000e+00 : f32
    %23 = linalg.fill ins(%cst_14 : f32) outs(%22 : tensor<1x2x3x4x5x6x7x9xf32>) -> tensor<1x2x3x4x5x6x7x9xf32>
    %24 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%cst_12, %cst_13 : tensor<1x2x3x4x5x6x7x8xf32>, tensor<4x5x6x8x9xf32>) outs(%23 : tensor<1x2x3x4x5x6x7x9xf32>) {
    ^bb0(%in: f32, %in_17: f32, %out: f32):
      %28 = arith.mulf %in, %in_17 : f32
      %29 = arith.addf %out, %28 : f32
      linalg.yield %29 : f32
    } -> tensor<1x2x3x4x5x6x7x9xf32>
    %25 = tensor.empty() : tensor<1x4x8xf32>
    %26 = linalg.softmax dimension(2) ins(%cst : tensor<1x4x8xf32>) outs(%25 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %cst_15 = arith.constant dense<2.828400e+00> : tensor<2x2xf32>
    %c2_i64_16 = arith.constant 2 : i64
    %c4_i64 = arith.constant 4 : i64
    %27 = tensor.empty() : tensor<2x2x4x4xf32>
    %broadcasted = linalg.broadcast ins(%cst_15 : tensor<2x2xf32>) outs(%27 : tensor<2x2x4x4xf32>) dimensions = [2, 3] 
    return %1 : tensor<8xf32>
  }
}
