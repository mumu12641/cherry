// Original IR loaded from file
module {
  cherry.func private @simple_transformer_block(%arg0: !cherry.cherry_tensor<[?xf32]>, %arg1: !cherry.cherry_tensor<[8x8xf32]>, %arg2: !cherry.cherry_tensor<[8x8xf32]>, %arg3: !cherry.cherry_tensor<[8x8xf32]>, %arg4: !cherry.cherry_tensor<[8x32xf32]>, %arg5: !cherry.cherry_tensor<[32x8xf32]>, %arg6: !cherry.cherry_tensor<[8xf32]>, %arg7: !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.matmul %arg0, %arg1 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %1 = cherry.matmul %arg0, %arg2 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %2 = cherry.matmul %arg0, %arg3 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %3 = cherry.constant(0 : i64) : i64
    %4 = cherry.constant(2 : i64) : i64
    %5 = cherry.constant(1 : i64) : i64
    %6 = cherry.transpose %1, %3, %4, %5 : (!cherry.cherry_tensor<[?xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    %7 = cherry.matmul %0, %6 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %8 = cherry.create_tensor dense<2.828400e+00> : tensor<1xf32> -> !cherry.cherry_tensor<[1xf32]>
    %9 = cherry.constant(1 : i64) : i64
    %10 = cherry.constant(4 : i64) : i64
    %11 = cherry.broadcast %8, %9, %10, %10 : (!cherry.cherry_tensor<[1xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    %12 = cherry.tensor_div %7, %11 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %13 = cherry.softmax %12 axis 2 : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %14 = cherry.matmul %13, %2 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %15 = cherry.tensor_add %arg0, %14 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %16 = cherry.matmul %15, %arg4 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[8x32xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %17 = cherry.tensor_relu %16 : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %18 = cherry.matmul %17, %arg5 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[32x8xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %19 = cherry.tensor_add %15, %18 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %19 : !cherry.cherry_tensor<[?xf32]>
  }
  cherry.func @main() -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %1 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %2 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %3 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %4 = cherry.create_tensor dense<2.000000e-01> : tensor<8x32xf32> -> !cherry.cherry_tensor<[8x32xf32]>
    %5 = cherry.create_tensor dense<2.000000e-01> : tensor<32x8xf32> -> !cherry.cherry_tensor<[32x8xf32]>
    %6 = cherry.create_tensor dense<1.000000e+00> : tensor<8xf32> -> !cherry.cherry_tensor<[8xf32]>
    %7 = cherry.create_tensor dense<0.000000e+00> : tensor<8xf32> -> !cherry.cherry_tensor<[8xf32]>
    %8 = cherry.call @simple_transformer_block(%0, %1, %2, %3, %4, %5, %6, %7) : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>, !cherry.cherry_tensor<[8x32xf32]>, !cherry.cherry_tensor<[32x8xf32]>, !cherry.cherry_tensor<[8xf32]>, !cherry.cherry_tensor<[8xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %9 = cherry.tensor_mul %8, %8 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %10 = cherry.tensor_add %9, %9 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %10 : !cherry.cherry_tensor<[?xf32]>
  }
}

// ==========================================
// Phase: Inliner
// ==========================================
module {
  cherry.func @main() -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %1 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %2 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %3 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %4 = cherry.create_tensor dense<2.000000e-01> : tensor<8x32xf32> -> !cherry.cherry_tensor<[8x32xf32]>
    %5 = cherry.create_tensor dense<2.000000e-01> : tensor<32x8xf32> -> !cherry.cherry_tensor<[32x8xf32]>
    %6 = cherry.cast %0 : !cherry.cherry_tensor<[1x4x8xf32]> to !cherry.cherry_tensor<[?xf32]>
    %7 = cherry.matmul %6, %1 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %8 = cherry.matmul %6, %2 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %9 = cherry.matmul %6, %3 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %10 = cherry.constant(0 : i64) : i64
    %11 = cherry.constant(2 : i64) : i64
    %12 = cherry.constant(1 : i64) : i64
    %13 = cherry.transpose %8, %10, %11, %12 : (!cherry.cherry_tensor<[?xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    %14 = cherry.matmul %7, %13 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %15 = cherry.create_tensor dense<2.828400e+00> : tensor<1xf32> -> !cherry.cherry_tensor<[1xf32]>
    %16 = cherry.constant(1 : i64) : i64
    %17 = cherry.constant(4 : i64) : i64
    %18 = cherry.broadcast %15, %16, %17, %17 : (!cherry.cherry_tensor<[1xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    %19 = cherry.tensor_div %14, %18 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %20 = cherry.softmax %19 axis 2 : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %21 = cherry.matmul %20, %9 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %22 = cherry.tensor_add %6, %21 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %23 = cherry.matmul %22, %4 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[8x32xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %24 = cherry.tensor_relu %23 : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %25 = cherry.matmul %24, %5 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[32x8xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %26 = cherry.tensor_add %22, %25 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %27 = cherry.tensor_mul %26, %26 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %28 = cherry.tensor_add %27, %27 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %28 : !cherry.cherry_tensor<[?xf32]>
  }
}

// ==========================================
// Phase: Shape Inference
// ==========================================
module {
  cherry.func @main() -> !cherry.cherry_tensor<[1x4x8xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %1 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %2 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %3 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %4 = cherry.create_tensor dense<2.000000e-01> : tensor<8x32xf32> -> !cherry.cherry_tensor<[8x32xf32]>
    %5 = cherry.create_tensor dense<2.000000e-01> : tensor<32x8xf32> -> !cherry.cherry_tensor<[32x8xf32]>
    %6 = cherry.cast %0 : !cherry.cherry_tensor<[1x4x8xf32]> to !cherry.cherry_tensor<[1x4x8xf32]>
    %7 = cherry.matmul %6, %1 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %8 = cherry.matmul %6, %2 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %9 = cherry.matmul %6, %3 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %10 = cherry.constant(0 : i64) : i64
    %11 = cherry.constant(2 : i64) : i64
    %12 = cherry.constant(1 : i64) : i64
    %13 = cherry.transpose %8, %10, %11, %12 : (!cherry.cherry_tensor<[1x4x8xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x8x4xf32]>
    %14 = cherry.matmul %7, %13 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[1x8x4xf32]>) -> !cherry.cherry_tensor<[1x4x4xf32]>
    %15 = cherry.create_tensor dense<2.828400e+00> : tensor<1xf32> -> !cherry.cherry_tensor<[1xf32]>
    %16 = cherry.constant(1 : i64) : i64
    %17 = cherry.constant(4 : i64) : i64
    %18 = cherry.broadcast %15, %16, %17, %17 : (!cherry.cherry_tensor<[1xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x4x4xf32]>
    %19 = cherry.tensor_div %14, %18 : (!cherry.cherry_tensor<[1x4x4xf32]>, !cherry.cherry_tensor<[1x4x4xf32]>) -> !cherry.cherry_tensor<[1x4x4xf32]>
    %20 = cherry.softmax %19 axis 2 : (!cherry.cherry_tensor<[1x4x4xf32]>) -> !cherry.cherry_tensor<[1x4x4xf32]>
    %21 = cherry.matmul %20, %9 : (!cherry.cherry_tensor<[1x4x4xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %22 = cherry.tensor_add %6, %21 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %23 = cherry.matmul %22, %4 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8x32xf32]>) -> !cherry.cherry_tensor<[1x4x32xf32]>
    %24 = cherry.tensor_relu %23 : (!cherry.cherry_tensor<[1x4x32xf32]>) -> !cherry.cherry_tensor<[1x4x32xf32]>
    %25 = cherry.matmul %24, %5 : (!cherry.cherry_tensor<[1x4x32xf32]>, !cherry.cherry_tensor<[32x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %26 = cherry.tensor_add %22, %25 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %27 = cherry.tensor_mul %26, %26 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %28 = cherry.tensor_add %27, %27 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    cherry.return %28 : !cherry.cherry_tensor<[1x4x8xf32]>
  }
}

// ==========================================
// Phase: Canonicalizer
// ==========================================
module {
  cherry.func @main() -> !cherry.cherry_tensor<[1x4x8xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %1 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %2 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %3 = cherry.create_tensor dense<1.000000e-01> : tensor<8x8xf32> -> !cherry.cherry_tensor<[8x8xf32]>
    %4 = cherry.create_tensor dense<2.000000e-01> : tensor<8x32xf32> -> !cherry.cherry_tensor<[8x32xf32]>
    %5 = cherry.create_tensor dense<2.000000e-01> : tensor<32x8xf32> -> !cherry.cherry_tensor<[32x8xf32]>
    %6 = cherry.matmul %0, %1 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %7 = cherry.matmul %0, %2 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %8 = cherry.matmul %0, %3 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %9 = cherry.constant(0 : i64) : i64
    %10 = cherry.constant(2 : i64) : i64
    %11 = cherry.constant(1 : i64) : i64
    %12 = cherry.transpose %7, %9, %10, %11 : (!cherry.cherry_tensor<[1x4x8xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x8x4xf32]>
    %13 = cherry.matmul %6, %12 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[1x8x4xf32]>) -> !cherry.cherry_tensor<[1x4x4xf32]>
    %14 = cherry.create_tensor dense<2.828400e+00> : tensor<1xf32> -> !cherry.cherry_tensor<[1xf32]>
    %15 = cherry.constant(1 : i64) : i64
    %16 = cherry.constant(4 : i64) : i64
    %17 = cherry.broadcast %14, %15, %16, %16 : (!cherry.cherry_tensor<[1xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x4x4xf32]>
    %18 = cherry.tensor_div %13, %17 : (!cherry.cherry_tensor<[1x4x4xf32]>, !cherry.cherry_tensor<[1x4x4xf32]>) -> !cherry.cherry_tensor<[1x4x4xf32]>
    %19 = cherry.softmax %18 axis 2 : (!cherry.cherry_tensor<[1x4x4xf32]>) -> !cherry.cherry_tensor<[1x4x4xf32]>
    %20 = cherry.matmul %19, %8 : (!cherry.cherry_tensor<[1x4x4xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %21 = cherry.tensor_add %0, %20 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %22 = cherry.matmul %21, %4 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8x32xf32]>) -> !cherry.cherry_tensor<[1x4x32xf32]>
    %23 = cherry.tensor_relu %22 : (!cherry.cherry_tensor<[1x4x32xf32]>) -> !cherry.cherry_tensor<[1x4x32xf32]>
    %24 = cherry.matmul %23, %5 : (!cherry.cherry_tensor<[1x4x32xf32]>, !cherry.cherry_tensor<[32x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %25 = cherry.tensor_add %21, %24 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %26 = cherry.tensor_mul %25, %25 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %27 = cherry.tensor_add %26, %26 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    cherry.return %27 : !cherry.cherry_tensor<[1x4x8xf32]>
  }
}

// ==========================================
// Phase: Convert to Linalg
// ==========================================
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main() -> tensor<1x4x8xf32> {
    %cst = arith.constant dense<5.000000e-01> : tensor<1x4x8xf32>
    %cst_0 = arith.constant dense<1.000000e-01> : tensor<8x8xf32>
    %cst_1 = arith.constant dense<1.000000e-01> : tensor<8x8xf32>
    %cst_2 = arith.constant dense<1.000000e-01> : tensor<8x8xf32>
    %cst_3 = arith.constant dense<2.000000e-01> : tensor<8x32xf32>
    %cst_4 = arith.constant dense<2.000000e-01> : tensor<32x8xf32>
    %0 = tensor.empty() : tensor<1x4x8xf32>
    %cst_5 = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst_5 : f32) outs(%0 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%cst, %cst_0 : tensor<1x4x8xf32>, tensor<8x8xf32>) outs(%1 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %37 = arith.mulf %in, %in_14 : f32
      %38 = arith.addf %out, %37 : f32
      linalg.yield %38 : f32
    } -> tensor<1x4x8xf32>
    %3 = tensor.empty() : tensor<1x4x8xf32>
    %cst_6 = arith.constant 0.000000e+00 : f32
    %4 = linalg.fill ins(%cst_6 : f32) outs(%3 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%cst, %cst_1 : tensor<1x4x8xf32>, tensor<8x8xf32>) outs(%4 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %37 = arith.mulf %in, %in_14 : f32
      %38 = arith.addf %out, %37 : f32
      linalg.yield %38 : f32
    } -> tensor<1x4x8xf32>
    %6 = tensor.empty() : tensor<1x4x8xf32>
    %cst_7 = arith.constant 0.000000e+00 : f32
    %7 = linalg.fill ins(%cst_7 : f32) outs(%6 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%cst, %cst_2 : tensor<1x4x8xf32>, tensor<8x8xf32>) outs(%7 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %37 = arith.mulf %in, %in_14 : f32
      %38 = arith.addf %out, %37 : f32
      linalg.yield %38 : f32
    } -> tensor<1x4x8xf32>
    %c0_i64 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %9 = tensor.empty() : tensor<1x8x4xf32>
    %transposed = linalg.transpose ins(%5 : tensor<1x4x8xf32>) outs(%9 : tensor<1x8x4xf32>) permutation = [0, 2, 1] 
    %10 = tensor.empty() : tensor<1x4x4xf32>
    %cst_8 = arith.constant 0.000000e+00 : f32
    %11 = linalg.fill ins(%cst_8 : f32) outs(%10 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2, %transposed : tensor<1x4x8xf32>, tensor<1x8x4xf32>) outs(%11 : tensor<1x4x4xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %37 = arith.mulf %in, %in_14 : f32
      %38 = arith.addf %out, %37 : f32
      linalg.yield %38 : f32
    } -> tensor<1x4x4xf32>
    %cst_9 = arith.constant dense<2.828400e+00> : tensor<1xf32>
    %c1_i64_10 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %13 = tensor.empty() : tensor<1x4x4xf32>
    %broadcasted = linalg.broadcast ins(%cst_9 : tensor<1xf32>) outs(%13 : tensor<1x4x4xf32>) dimensions = [1, 2] 
    %14 = tensor.empty() : tensor<1x4x4xf32>
    %15 = linalg.div ins(%12, %broadcasted : tensor<1x4x4xf32>, tensor<1x4x4xf32>) outs(%14 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %16 = tensor.empty() : tensor<1x4x4xf32>
    %17 = linalg.softmax dimension(2) ins(%15 : tensor<1x4x4xf32>) outs(%16 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %18 = tensor.empty() : tensor<1x4x8xf32>
    %cst_11 = arith.constant 0.000000e+00 : f32
    %19 = linalg.fill ins(%cst_11 : f32) outs(%18 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%17, %8 : tensor<1x4x4xf32>, tensor<1x4x8xf32>) outs(%19 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %37 = arith.mulf %in, %in_14 : f32
      %38 = arith.addf %out, %37 : f32
      linalg.yield %38 : f32
    } -> tensor<1x4x8xf32>
    %21 = tensor.empty() : tensor<1x4x8xf32>
    %22 = linalg.add ins(%cst, %20 : tensor<1x4x8xf32>, tensor<1x4x8xf32>) outs(%21 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %23 = tensor.empty() : tensor<1x4x32xf32>
    %cst_12 = arith.constant 0.000000e+00 : f32
    %24 = linalg.fill ins(%cst_12 : f32) outs(%23 : tensor<1x4x32xf32>) -> tensor<1x4x32xf32>
    %25 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%22, %cst_3 : tensor<1x4x8xf32>, tensor<8x32xf32>) outs(%24 : tensor<1x4x32xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %37 = arith.mulf %in, %in_14 : f32
      %38 = arith.addf %out, %37 : f32
      linalg.yield %38 : f32
    } -> tensor<1x4x32xf32>
    %26 = tensor.empty() : tensor<1x4x32xf32>
    %27 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25 : tensor<1x4x32xf32>) outs(%26 : tensor<1x4x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_14 = arith.constant 0.000000e+00 : f32
      %37 = arith.maximumf %in, %cst_14 : f32
      linalg.yield %37 : f32
    } -> tensor<1x4x32xf32>
    %28 = tensor.empty() : tensor<1x4x8xf32>
    %cst_13 = arith.constant 0.000000e+00 : f32
    %29 = linalg.fill ins(%cst_13 : f32) outs(%28 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %30 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%27, %cst_4 : tensor<1x4x32xf32>, tensor<32x8xf32>) outs(%29 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %37 = arith.mulf %in, %in_14 : f32
      %38 = arith.addf %out, %37 : f32
      linalg.yield %38 : f32
    } -> tensor<1x4x8xf32>
    %31 = tensor.empty() : tensor<1x4x8xf32>
    %32 = linalg.add ins(%22, %30 : tensor<1x4x8xf32>, tensor<1x4x8xf32>) outs(%31 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %33 = tensor.empty() : tensor<1x4x8xf32>
    %34 = linalg.mul ins(%32, %32 : tensor<1x4x8xf32>, tensor<1x4x8xf32>) outs(%33 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %35 = tensor.empty() : tensor<1x4x8xf32>
    %36 = linalg.add ins(%34, %34 : tensor<1x4x8xf32>, tensor<1x4x8xf32>) outs(%35 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    return %36 : tensor<1x4x8xf32>
  }
}

// ==========================================
// Phase: Linalg Tiling
// ==========================================
#map = affine_map<(d0) -> (-d0 + 1, 8)>
#map1 = affine_map<(d0) -> (-d0 + 4, 8)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
module {
  func.func @main() -> tensor<1x4x8xf32> {
    %c8 = arith.constant 8 : index
    %c8_0 = arith.constant 8 : index
    %c8_1 = arith.constant 8 : index
    %c8_2 = arith.constant 8 : index
    %c8_3 = arith.constant 8 : index
    %c8_4 = arith.constant 8 : index
    %c8_5 = arith.constant 8 : index
    %c8_6 = arith.constant 8 : index
    %c8_7 = arith.constant 8 : index
    %c8_8 = arith.constant 8 : index
    %c8_9 = arith.constant 8 : index
    %c8_10 = arith.constant 8 : index
    %c8_11 = arith.constant 8 : index
    %c8_12 = arith.constant 8 : index
    %c8_13 = arith.constant 8 : index
    %c8_14 = arith.constant 8 : index
    %c8_15 = arith.constant 8 : index
    %c8_16 = arith.constant 8 : index
    %c8_17 = arith.constant 8 : index
    %c8_18 = arith.constant 8 : index
    %c8_19 = arith.constant 8 : index
    %c8_20 = arith.constant 8 : index
    %c8_21 = arith.constant 8 : index
    %c8_22 = arith.constant 8 : index
    %c8_23 = arith.constant 8 : index
    %c8_24 = arith.constant 8 : index
    %c8_25 = arith.constant 8 : index
    %c8_26 = arith.constant 8 : index
    %c8_27 = arith.constant 8 : index
    %c8_28 = arith.constant 8 : index
    %c8_29 = arith.constant 8 : index
    %c8_30 = arith.constant 8 : index
    %c8_31 = arith.constant 8 : index
    %c8_32 = arith.constant 8 : index
    %c8_33 = arith.constant 8 : index
    %c8_34 = arith.constant 8 : index
    %c8_35 = arith.constant 8 : index
    %c8_36 = arith.constant 8 : index
    %c8_37 = arith.constant 8 : index
    %c8_38 = arith.constant 8 : index
    %c8_39 = arith.constant 8 : index
    %c8_40 = arith.constant 8 : index
    %c8_41 = arith.constant 8 : index
    %c8_42 = arith.constant 8 : index
    %c8_43 = arith.constant 8 : index
    %c8_44 = arith.constant 8 : index
    %c8_45 = arith.constant 8 : index
    %c8_46 = arith.constant 8 : index
    %c8_47 = arith.constant 8 : index
    %c8_48 = arith.constant 8 : index
    %c8_49 = arith.constant 8 : index
    %cst = arith.constant 2.000000e-01 : f32
    %cst_50 = arith.constant 2.828400e+00 : f32
    %cst_51 = arith.constant 5.000000e-01 : f32
    %cst_52 = arith.constant 0.000000e+00 : f32
    %cst_53 = arith.constant dense<1.000000e-01> : tensor<8x8xf32>
    %0 = tensor.empty() : tensor<1x4x8xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_54 = arith.constant 8 : index
    %c0_55 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8_56 = arith.constant 8 : index
    %c0_57 = arith.constant 0 : index
    %c8_58 = arith.constant 8 : index
    %c8_59 = arith.constant 8 : index
    %1 = scf.for %arg0 = %c0 to %c1 step %c8_54 iter_args(%arg1 = %0) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_55 to %c4 step %c8_56 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_57 to %c8_58 step %c8_59 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_52 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %c0_60 = arith.constant 0 : index
    %c1_61 = arith.constant 1 : index
    %c8_62 = arith.constant 8 : index
    %c0_63 = arith.constant 0 : index
    %c4_64 = arith.constant 4 : index
    %c8_65 = arith.constant 8 : index
    %c0_66 = arith.constant 0 : index
    %c8_67 = arith.constant 8 : index
    %c8_68 = arith.constant 8 : index
    %2 = scf.for %arg0 = %c0_60 to %c1_61 step %c8_62 iter_args(%arg1 = %1) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_63 to %c4_64 step %c8_65 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_66 to %c8_67 step %c8_68 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %cst_53[0, %arg4] [8, 8] [1, 1] : tensor<8x8xf32> to tensor<8x8xf32>
          %extracted_slice_203 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<8x8xf32>) outs(%extracted_slice_203 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %34 = arith.mulf %in, %cst_51 : f32
            %35 = arith.addf %out, %34 : f32
            linalg.yield %35 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %3 = tensor.empty() : tensor<1x4x8xf32>
    %c0_69 = arith.constant 0 : index
    %c1_70 = arith.constant 1 : index
    %c8_71 = arith.constant 8 : index
    %c0_72 = arith.constant 0 : index
    %c4_73 = arith.constant 4 : index
    %c8_74 = arith.constant 8 : index
    %c0_75 = arith.constant 0 : index
    %c8_76 = arith.constant 8 : index
    %c8_77 = arith.constant 8 : index
    %4 = scf.for %arg0 = %c0_69 to %c1_70 step %c8_71 iter_args(%arg1 = %3) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_72 to %c4_73 step %c8_74 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_75 to %c8_76 step %c8_77 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_52 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %c0_78 = arith.constant 0 : index
    %c1_79 = arith.constant 1 : index
    %c8_80 = arith.constant 8 : index
    %c0_81 = arith.constant 0 : index
    %c4_82 = arith.constant 4 : index
    %c8_83 = arith.constant 8 : index
    %c0_84 = arith.constant 0 : index
    %c8_85 = arith.constant 8 : index
    %c8_86 = arith.constant 8 : index
    %5 = scf.for %arg0 = %c0_78 to %c1_79 step %c8_80 iter_args(%arg1 = %4) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_81 to %c4_82 step %c8_83 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_84 to %c8_85 step %c8_86 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %cst_53[0, %arg4] [8, 8] [1, 1] : tensor<8x8xf32> to tensor<8x8xf32>
          %extracted_slice_203 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<8x8xf32>) outs(%extracted_slice_203 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %34 = arith.mulf %in, %cst_51 : f32
            %35 = arith.addf %out, %34 : f32
            linalg.yield %35 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %6 = tensor.empty() : tensor<1x4x8xf32>
    %c0_87 = arith.constant 0 : index
    %c1_88 = arith.constant 1 : index
    %c8_89 = arith.constant 8 : index
    %c0_90 = arith.constant 0 : index
    %c4_91 = arith.constant 4 : index
    %c8_92 = arith.constant 8 : index
    %c0_93 = arith.constant 0 : index
    %c8_94 = arith.constant 8 : index
    %c8_95 = arith.constant 8 : index
    %7 = scf.for %arg0 = %c0_87 to %c1_88 step %c8_89 iter_args(%arg1 = %6) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_90 to %c4_91 step %c8_92 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_93 to %c8_94 step %c8_95 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_52 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %c0_96 = arith.constant 0 : index
    %c1_97 = arith.constant 1 : index
    %c8_98 = arith.constant 8 : index
    %c0_99 = arith.constant 0 : index
    %c4_100 = arith.constant 4 : index
    %c8_101 = arith.constant 8 : index
    %c0_102 = arith.constant 0 : index
    %c8_103 = arith.constant 8 : index
    %c8_104 = arith.constant 8 : index
    %8 = scf.for %arg0 = %c0_96 to %c1_97 step %c8_98 iter_args(%arg1 = %7) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_99 to %c4_100 step %c8_101 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_102 to %c8_103 step %c8_104 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %cst_53[0, %arg4] [8, 8] [1, 1] : tensor<8x8xf32> to tensor<8x8xf32>
          %extracted_slice_203 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<8x8xf32>) outs(%extracted_slice_203 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %34 = arith.mulf %in, %cst_51 : f32
            %35 = arith.addf %out, %34 : f32
            linalg.yield %35 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %9 = tensor.empty() : tensor<1x4x4xf32>
    %c0_105 = arith.constant 0 : index
    %c1_106 = arith.constant 1 : index
    %c8_107 = arith.constant 8 : index
    %c0_108 = arith.constant 0 : index
    %c4_109 = arith.constant 4 : index
    %c8_110 = arith.constant 8 : index
    %c0_111 = arith.constant 0 : index
    %c4_112 = arith.constant 4 : index
    %c8_113 = arith.constant 8 : index
    %10 = scf.for %arg0 = %c0_105 to %c1_106 step %c8_107 iter_args(%arg1 = %9) -> (tensor<1x4x4xf32>) {
      %29 = scf.for %arg2 = %c0_108 to %c4_109 step %c8_110 iter_args(%arg3 = %arg1) -> (tensor<1x4x4xf32>) {
        %30 = scf.for %arg4 = %c0_111 to %c4_112 step %c8_113 iter_args(%arg5 = %arg3) -> (tensor<1x4x4xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %33 = affine.min #map1(%arg4)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, %33] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x?xf32>
          %34 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x?xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_52 : f32
          } -> tensor<?x?x?xf32>
          %inserted_slice = tensor.insert_slice %34 into %arg5[%arg0, %arg2, %arg4] [%31, %32, %33] [1, 1, 1] : tensor<?x?x?xf32> into tensor<1x4x4xf32>
          scf.yield %inserted_slice : tensor<1x4x4xf32>
        }
        scf.yield %30 : tensor<1x4x4xf32>
      }
      scf.yield %29 : tensor<1x4x4xf32>
    }
    %c0_114 = arith.constant 0 : index
    %c1_115 = arith.constant 1 : index
    %c8_116 = arith.constant 8 : index
    %c0_117 = arith.constant 0 : index
    %c4_118 = arith.constant 4 : index
    %c8_119 = arith.constant 8 : index
    %c0_120 = arith.constant 0 : index
    %c4_121 = arith.constant 4 : index
    %c8_122 = arith.constant 8 : index
    %11 = scf.for %arg0 = %c0_114 to %c1_115 step %c8_116 iter_args(%arg1 = %10) -> (tensor<1x4x4xf32>) {
      %29 = scf.for %arg2 = %c0_117 to %c4_118 step %c8_119 iter_args(%arg3 = %arg1) -> (tensor<1x4x4xf32>) {
        %30 = scf.for %arg4 = %c0_120 to %c4_121 step %c8_122 iter_args(%arg5 = %arg3) -> (tensor<1x4x4xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %33 = affine.min #map(%arg0)
          %34 = affine.min #map1(%arg4)
          %35 = affine.min #map(%arg0)
          %36 = affine.min #map1(%arg2)
          %37 = affine.min #map1(%arg4)
          %extracted_slice = tensor.extract_slice %2[%arg0, %arg2, 0] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_203 = tensor.extract_slice %5[%arg0, %arg4, 0] [%33, %34, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_204 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%35, %36, %37] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x?xf32>
          %38 = linalg.generic {indexing_maps = [#map5, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_203 : tensor<?x?x8xf32>, tensor<?x?x8xf32>) outs(%extracted_slice_204 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %in_205: f32, %out: f32):
            %39 = arith.mulf %in, %in_205 : f32
            %40 = arith.addf %out, %39 : f32
            linalg.yield %40 : f32
          } -> tensor<?x?x?xf32>
          %inserted_slice = tensor.insert_slice %38 into %arg5[%arg0, %arg2, %arg4] [%35, %36, %37] [1, 1, 1] : tensor<?x?x?xf32> into tensor<1x4x4xf32>
          scf.yield %inserted_slice : tensor<1x4x4xf32>
        }
        scf.yield %30 : tensor<1x4x4xf32>
      }
      scf.yield %29 : tensor<1x4x4xf32>
    }
    %12 = tensor.empty() : tensor<1x4x4xf32>
    %c0_123 = arith.constant 0 : index
    %c1_124 = arith.constant 1 : index
    %c8_125 = arith.constant 8 : index
    %c0_126 = arith.constant 0 : index
    %c4_127 = arith.constant 4 : index
    %c8_128 = arith.constant 8 : index
    %c0_129 = arith.constant 0 : index
    %c4_130 = arith.constant 4 : index
    %c8_131 = arith.constant 8 : index
    %13 = scf.for %arg0 = %c0_123 to %c1_124 step %c8_125 iter_args(%arg1 = %12) -> (tensor<1x4x4xf32>) {
      %29 = scf.for %arg2 = %c0_126 to %c4_127 step %c8_128 iter_args(%arg3 = %arg1) -> (tensor<1x4x4xf32>) {
        %30 = scf.for %arg4 = %c0_129 to %c4_130 step %c8_131 iter_args(%arg5 = %arg3) -> (tensor<1x4x4xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %33 = affine.min #map1(%arg4)
          %34 = affine.min #map(%arg0)
          %35 = affine.min #map1(%arg2)
          %36 = affine.min #map1(%arg4)
          %extracted_slice = tensor.extract_slice %11[%arg0, %arg2, %arg4] [%31, %32, %33] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x?xf32>
          %extracted_slice_203 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%34, %35, %36] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x?xf32>
          %37 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<?x?x?xf32>) outs(%extracted_slice_203 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %38 = arith.divf %in, %cst_50 : f32
            linalg.yield %38 : f32
          } -> tensor<?x?x?xf32>
          %inserted_slice = tensor.insert_slice %37 into %arg5[%arg0, %arg2, %arg4] [%34, %35, %36] [1, 1, 1] : tensor<?x?x?xf32> into tensor<1x4x4xf32>
          scf.yield %inserted_slice : tensor<1x4x4xf32>
        }
        scf.yield %30 : tensor<1x4x4xf32>
      }
      scf.yield %29 : tensor<1x4x4xf32>
    }
    %14 = tensor.empty() : tensor<1x4x4xf32>
    %15 = linalg.softmax dimension(2) ins(%13 : tensor<1x4x4xf32>) outs(%14 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %16 = tensor.empty() : tensor<1x4x8xf32>
    %c0_132 = arith.constant 0 : index
    %c1_133 = arith.constant 1 : index
    %c8_134 = arith.constant 8 : index
    %c0_135 = arith.constant 0 : index
    %c4_136 = arith.constant 4 : index
    %c8_137 = arith.constant 8 : index
    %c0_138 = arith.constant 0 : index
    %c8_139 = arith.constant 8 : index
    %c8_140 = arith.constant 8 : index
    %17 = scf.for %arg0 = %c0_132 to %c1_133 step %c8_134 iter_args(%arg1 = %16) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_135 to %c4_136 step %c8_137 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_138 to %c8_139 step %c8_140 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_52 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %c0_141 = arith.constant 0 : index
    %c1_142 = arith.constant 1 : index
    %c8_143 = arith.constant 8 : index
    %c0_144 = arith.constant 0 : index
    %c4_145 = arith.constant 4 : index
    %c8_146 = arith.constant 8 : index
    %c0_147 = arith.constant 0 : index
    %c8_148 = arith.constant 8 : index
    %c8_149 = arith.constant 8 : index
    %18 = scf.for %arg0 = %c0_141 to %c1_142 step %c8_143 iter_args(%arg1 = %17) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_144 to %c4_145 step %c8_146 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_147 to %c8_148 step %c8_149 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %33 = affine.min #map(%arg0)
          %34 = affine.min #map(%arg0)
          %35 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %15[%arg0, %arg2, 0] [%31, %32, 4] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x4xf32>
          %extracted_slice_203 = tensor.extract_slice %8[%arg0, 0, %arg4] [%33, 4, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x4x8xf32>
          %extracted_slice_204 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%34, %35, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %36 = linalg.generic {indexing_maps = [#map5, #map7, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_203 : tensor<?x?x4xf32>, tensor<?x4x8xf32>) outs(%extracted_slice_204 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %in_205: f32, %out: f32):
            %37 = arith.mulf %in, %in_205 : f32
            %38 = arith.addf %out, %37 : f32
            linalg.yield %38 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %36 into %arg5[%arg0, %arg2, %arg4] [%34, %35, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %19 = tensor.empty() : tensor<1x4x8xf32>
    %c0_150 = arith.constant 0 : index
    %c1_151 = arith.constant 1 : index
    %c8_152 = arith.constant 8 : index
    %c0_153 = arith.constant 0 : index
    %c4_154 = arith.constant 4 : index
    %c8_155 = arith.constant 8 : index
    %c0_156 = arith.constant 0 : index
    %c8_157 = arith.constant 8 : index
    %c8_158 = arith.constant 8 : index
    %20 = scf.for %arg0 = %c0_150 to %c1_151 step %c8_152 iter_args(%arg1 = %19) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_153 to %c4_154 step %c8_155 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_156 to %c8_157 step %c8_158 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %33 = affine.min #map(%arg0)
          %34 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %18[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_203 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%33, %34, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %35 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<?x?x8xf32>) outs(%extracted_slice_203 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %36 = arith.addf %in, %cst_51 : f32
            linalg.yield %36 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %35 into %arg5[%arg0, %arg2, %arg4] [%33, %34, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %21 = tensor.empty() : tensor<1x4x32xf32>
    %c0_159 = arith.constant 0 : index
    %c1_160 = arith.constant 1 : index
    %c8_161 = arith.constant 8 : index
    %c0_162 = arith.constant 0 : index
    %c4_163 = arith.constant 4 : index
    %c8_164 = arith.constant 8 : index
    %c0_165 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c8_166 = arith.constant 8 : index
    %22 = scf.for %arg0 = %c0_159 to %c1_160 step %c8_161 iter_args(%arg1 = %21) -> (tensor<1x4x32xf32>) {
      %29 = scf.for %arg2 = %c0_162 to %c4_163 step %c8_164 iter_args(%arg3 = %arg1) -> (tensor<1x4x32xf32>) {
        %30 = scf.for %arg4 = %c0_165 to %c32 step %c8_166 iter_args(%arg5 = %arg3) -> (tensor<1x4x32xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x32xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_52 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x32xf32>
          scf.yield %inserted_slice : tensor<1x4x32xf32>
        }
        scf.yield %30 : tensor<1x4x32xf32>
      }
      scf.yield %29 : tensor<1x4x32xf32>
    }
    %c0_167 = arith.constant 0 : index
    %c1_168 = arith.constant 1 : index
    %c8_169 = arith.constant 8 : index
    %c0_170 = arith.constant 0 : index
    %c4_171 = arith.constant 4 : index
    %c8_172 = arith.constant 8 : index
    %c0_173 = arith.constant 0 : index
    %c32_174 = arith.constant 32 : index
    %c8_175 = arith.constant 8 : index
    %23 = scf.for %arg0 = %c0_167 to %c1_168 step %c8_169 iter_args(%arg1 = %22) -> (tensor<1x4x32xf32>) {
      %29 = scf.for %arg2 = %c0_170 to %c4_171 step %c8_172 iter_args(%arg3 = %arg1) -> (tensor<1x4x32xf32>) {
        %30 = scf.for %arg4 = %c0_173 to %c32_174 step %c8_175 iter_args(%arg5 = %arg3) -> (tensor<1x4x32xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %33 = affine.min #map(%arg0)
          %34 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %20[%arg0, %arg2, 0] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_203 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%33, %34, 8] [1, 1, 1] : tensor<1x4x32xf32> to tensor<?x?x8xf32>
          %35 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<?x?x8xf32>) outs(%extracted_slice_203 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %36 = arith.mulf %in, %cst : f32
            %37 = arith.addf %out, %36 : f32
            linalg.yield %37 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %35 into %arg5[%arg0, %arg2, %arg4] [%33, %34, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x32xf32>
          scf.yield %inserted_slice : tensor<1x4x32xf32>
        }
        scf.yield %30 : tensor<1x4x32xf32>
      }
      scf.yield %29 : tensor<1x4x32xf32>
    }
    %24 = tensor.empty() : tensor<1x4x8xf32>
    %c0_176 = arith.constant 0 : index
    %c1_177 = arith.constant 1 : index
    %c8_178 = arith.constant 8 : index
    %c0_179 = arith.constant 0 : index
    %c4_180 = arith.constant 4 : index
    %c8_181 = arith.constant 8 : index
    %c0_182 = arith.constant 0 : index
    %c8_183 = arith.constant 8 : index
    %c8_184 = arith.constant 8 : index
    %25 = scf.for %arg0 = %c0_176 to %c1_177 step %c8_178 iter_args(%arg1 = %24) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_179 to %c4_180 step %c8_181 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_182 to %c8_183 step %c8_184 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_52 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %c0_185 = arith.constant 0 : index
    %c1_186 = arith.constant 1 : index
    %c8_187 = arith.constant 8 : index
    %c0_188 = arith.constant 0 : index
    %c4_189 = arith.constant 4 : index
    %c8_190 = arith.constant 8 : index
    %c0_191 = arith.constant 0 : index
    %c8_192 = arith.constant 8 : index
    %c8_193 = arith.constant 8 : index
    %26 = scf.for %arg0 = %c0_185 to %c1_186 step %c8_187 iter_args(%arg1 = %25) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_188 to %c4_189 step %c8_190 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_191 to %c8_192 step %c8_193 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %33 = affine.min #map(%arg0)
          %34 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %23[%arg0, %arg2, 0] [%31, %32, 32] [1, 1, 1] : tensor<1x4x32xf32> to tensor<?x?x32xf32>
          %extracted_slice_203 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%33, %34, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %35 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<?x?x32xf32>) outs(%extracted_slice_203 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %36 = arith.maximumf %in, %cst_52 : f32
            %37 = arith.mulf %36, %cst : f32
            %38 = arith.addf %out, %37 : f32
            linalg.yield %38 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %35 into %arg5[%arg0, %arg2, %arg4] [%33, %34, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %27 = tensor.empty() : tensor<1x4x8xf32>
    %c0_194 = arith.constant 0 : index
    %c1_195 = arith.constant 1 : index
    %c8_196 = arith.constant 8 : index
    %c0_197 = arith.constant 0 : index
    %c4_198 = arith.constant 4 : index
    %c8_199 = arith.constant 8 : index
    %c0_200 = arith.constant 0 : index
    %c8_201 = arith.constant 8 : index
    %c8_202 = arith.constant 8 : index
    %28 = scf.for %arg0 = %c0_194 to %c1_195 step %c8_196 iter_args(%arg1 = %27) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_197 to %c4_198 step %c8_199 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_200 to %c8_201 step %c8_202 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %33 = affine.min #map(%arg0)
          %34 = affine.min #map1(%arg2)
          %35 = affine.min #map(%arg0)
          %36 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %20[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_203 = tensor.extract_slice %26[%arg0, %arg2, %arg4] [%33, %34, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_204 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%35, %36, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %37 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_203 : tensor<?x?x8xf32>, tensor<?x?x8xf32>) outs(%extracted_slice_204 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %in_205: f32, %out: f32):
            %38 = arith.addf %in, %in_205 : f32
            %39 = arith.mulf %38, %38 : f32
            %40 = arith.addf %39, %39 : f32
            linalg.yield %40 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %37 into %arg5[%arg0, %arg2, %arg4] [%35, %36, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    return %28 : tensor<1x4x8xf32>
  }
}

// ==========================================
// Phase: test
// ==========================================
#map = affine_map<(d0) -> (-d0 + 1, 8)>
#map1 = affine_map<(d0) -> (-d0 + 4, 8)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
module {
  func.func @main() -> tensor<1x4x8xf32> {
    %c8 = arith.constant 8 : index
    %c8_0 = arith.constant 8 : index
    %c8_1 = arith.constant 8 : index
    %c8_2 = arith.constant 8 : index
    %c8_3 = arith.constant 8 : index
    %c8_4 = arith.constant 8 : index
    %c8_5 = arith.constant 8 : index
    %c8_6 = arith.constant 8 : index
    %c8_7 = arith.constant 8 : index
    %c8_8 = arith.constant 8 : index
    %c8_9 = arith.constant 8 : index
    %c8_10 = arith.constant 8 : index
    %c8_11 = arith.constant 8 : index
    %c8_12 = arith.constant 8 : index
    %c8_13 = arith.constant 8 : index
    %c8_14 = arith.constant 8 : index
    %c8_15 = arith.constant 8 : index
    %c8_16 = arith.constant 8 : index
    %c8_17 = arith.constant 8 : index
    %c8_18 = arith.constant 8 : index
    %c8_19 = arith.constant 8 : index
    %c8_20 = arith.constant 8 : index
    %c8_21 = arith.constant 8 : index
    %c8_22 = arith.constant 8 : index
    %c8_23 = arith.constant 8 : index
    %c8_24 = arith.constant 8 : index
    %c8_25 = arith.constant 8 : index
    %c8_26 = arith.constant 8 : index
    %c8_27 = arith.constant 8 : index
    %c8_28 = arith.constant 8 : index
    %c8_29 = arith.constant 8 : index
    %c8_30 = arith.constant 8 : index
    %c8_31 = arith.constant 8 : index
    %c8_32 = arith.constant 8 : index
    %c8_33 = arith.constant 8 : index
    %c8_34 = arith.constant 8 : index
    %c8_35 = arith.constant 8 : index
    %c8_36 = arith.constant 8 : index
    %c8_37 = arith.constant 8 : index
    %c8_38 = arith.constant 8 : index
    %c8_39 = arith.constant 8 : index
    %c8_40 = arith.constant 8 : index
    %c8_41 = arith.constant 8 : index
    %c8_42 = arith.constant 8 : index
    %c8_43 = arith.constant 8 : index
    %c8_44 = arith.constant 8 : index
    %c8_45 = arith.constant 8 : index
    %c8_46 = arith.constant 8 : index
    %c8_47 = arith.constant 8 : index
    %c8_48 = arith.constant 8 : index
    %c8_49 = arith.constant 8 : index
    %cst = arith.constant 2.000000e-01 : f32
    %cst_50 = arith.constant 2.828400e+00 : f32
    %cst_51 = arith.constant 5.000000e-01 : f32
    %cst_52 = arith.constant 0.000000e+00 : f32
    %cst_53 = arith.constant dense<1.000000e-01> : tensor<8x8xf32>
    %0 = tensor.empty() : tensor<1x4x8xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_54 = arith.constant 8 : index
    %c0_55 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8_56 = arith.constant 8 : index
    %c0_57 = arith.constant 0 : index
    %c8_58 = arith.constant 8 : index
    %c8_59 = arith.constant 8 : index
    %1 = scf.for %arg0 = %c0 to %c1 step %c8_54 iter_args(%arg1 = %0) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_55 to %c4 step %c8_56 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_57 to %c8_58 step %c8_59 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_52 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %c0_60 = arith.constant 0 : index
    %c1_61 = arith.constant 1 : index
    %c8_62 = arith.constant 8 : index
    %c0_63 = arith.constant 0 : index
    %c4_64 = arith.constant 4 : index
    %c8_65 = arith.constant 8 : index
    %c0_66 = arith.constant 0 : index
    %c8_67 = arith.constant 8 : index
    %c8_68 = arith.constant 8 : index
    %2 = scf.for %arg0 = %c0_60 to %c1_61 step %c8_62 iter_args(%arg1 = %1) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_63 to %c4_64 step %c8_65 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_66 to %c8_67 step %c8_68 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %cst_53[0, %arg4] [8, 8] [1, 1] : tensor<8x8xf32> to tensor<8x8xf32>
          %extracted_slice_203 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<8x8xf32>) outs(%extracted_slice_203 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %34 = arith.mulf %in, %cst_51 : f32
            %35 = arith.addf %out, %34 : f32
            linalg.yield %35 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %3 = tensor.empty() : tensor<1x4x8xf32>
    %c0_69 = arith.constant 0 : index
    %c1_70 = arith.constant 1 : index
    %c8_71 = arith.constant 8 : index
    %c0_72 = arith.constant 0 : index
    %c4_73 = arith.constant 4 : index
    %c8_74 = arith.constant 8 : index
    %c0_75 = arith.constant 0 : index
    %c8_76 = arith.constant 8 : index
    %c8_77 = arith.constant 8 : index
    %4 = scf.for %arg0 = %c0_69 to %c1_70 step %c8_71 iter_args(%arg1 = %3) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_72 to %c4_73 step %c8_74 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_75 to %c8_76 step %c8_77 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_52 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %c0_78 = arith.constant 0 : index
    %c1_79 = arith.constant 1 : index
    %c8_80 = arith.constant 8 : index
    %c0_81 = arith.constant 0 : index
    %c4_82 = arith.constant 4 : index
    %c8_83 = arith.constant 8 : index
    %c0_84 = arith.constant 0 : index
    %c8_85 = arith.constant 8 : index
    %c8_86 = arith.constant 8 : index
    %5 = scf.for %arg0 = %c0_78 to %c1_79 step %c8_80 iter_args(%arg1 = %4) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_81 to %c4_82 step %c8_83 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_84 to %c8_85 step %c8_86 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %cst_53[0, %arg4] [8, 8] [1, 1] : tensor<8x8xf32> to tensor<8x8xf32>
          %extracted_slice_203 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<8x8xf32>) outs(%extracted_slice_203 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %34 = arith.mulf %in, %cst_51 : f32
            %35 = arith.addf %out, %34 : f32
            linalg.yield %35 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %6 = tensor.empty() : tensor<1x4x8xf32>
    %c0_87 = arith.constant 0 : index
    %c1_88 = arith.constant 1 : index
    %c8_89 = arith.constant 8 : index
    %c0_90 = arith.constant 0 : index
    %c4_91 = arith.constant 4 : index
    %c8_92 = arith.constant 8 : index
    %c0_93 = arith.constant 0 : index
    %c8_94 = arith.constant 8 : index
    %c8_95 = arith.constant 8 : index
    %7 = scf.for %arg0 = %c0_87 to %c1_88 step %c8_89 iter_args(%arg1 = %6) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_90 to %c4_91 step %c8_92 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_93 to %c8_94 step %c8_95 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_52 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %c0_96 = arith.constant 0 : index
    %c1_97 = arith.constant 1 : index
    %c8_98 = arith.constant 8 : index
    %c0_99 = arith.constant 0 : index
    %c4_100 = arith.constant 4 : index
    %c8_101 = arith.constant 8 : index
    %c0_102 = arith.constant 0 : index
    %c8_103 = arith.constant 8 : index
    %c8_104 = arith.constant 8 : index
    %8 = scf.for %arg0 = %c0_96 to %c1_97 step %c8_98 iter_args(%arg1 = %7) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_99 to %c4_100 step %c8_101 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_102 to %c8_103 step %c8_104 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %cst_53[0, %arg4] [8, 8] [1, 1] : tensor<8x8xf32> to tensor<8x8xf32>
          %extracted_slice_203 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<8x8xf32>) outs(%extracted_slice_203 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %34 = arith.mulf %in, %cst_51 : f32
            %35 = arith.addf %out, %34 : f32
            linalg.yield %35 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %9 = tensor.empty() : tensor<1x4x4xf32>
    %c0_105 = arith.constant 0 : index
    %c1_106 = arith.constant 1 : index
    %c8_107 = arith.constant 8 : index
    %c0_108 = arith.constant 0 : index
    %c4_109 = arith.constant 4 : index
    %c8_110 = arith.constant 8 : index
    %c0_111 = arith.constant 0 : index
    %c4_112 = arith.constant 4 : index
    %c8_113 = arith.constant 8 : index
    %10 = scf.for %arg0 = %c0_105 to %c1_106 step %c8_107 iter_args(%arg1 = %9) -> (tensor<1x4x4xf32>) {
      %29 = scf.for %arg2 = %c0_108 to %c4_109 step %c8_110 iter_args(%arg3 = %arg1) -> (tensor<1x4x4xf32>) {
        %30 = scf.for %arg4 = %c0_111 to %c4_112 step %c8_113 iter_args(%arg5 = %arg3) -> (tensor<1x4x4xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %33 = affine.min #map1(%arg4)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, %33] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x?xf32>
          %34 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x?xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_52 : f32
          } -> tensor<?x?x?xf32>
          %inserted_slice = tensor.insert_slice %34 into %arg5[%arg0, %arg2, %arg4] [%31, %32, %33] [1, 1, 1] : tensor<?x?x?xf32> into tensor<1x4x4xf32>
          scf.yield %inserted_slice : tensor<1x4x4xf32>
        }
        scf.yield %30 : tensor<1x4x4xf32>
      }
      scf.yield %29 : tensor<1x4x4xf32>
    }
    %c0_114 = arith.constant 0 : index
    %c1_115 = arith.constant 1 : index
    %c8_116 = arith.constant 8 : index
    %c0_117 = arith.constant 0 : index
    %c4_118 = arith.constant 4 : index
    %c8_119 = arith.constant 8 : index
    %c0_120 = arith.constant 0 : index
    %c4_121 = arith.constant 4 : index
    %c8_122 = arith.constant 8 : index
    %11 = scf.for %arg0 = %c0_114 to %c1_115 step %c8_116 iter_args(%arg1 = %10) -> (tensor<1x4x4xf32>) {
      %29 = scf.for %arg2 = %c0_117 to %c4_118 step %c8_119 iter_args(%arg3 = %arg1) -> (tensor<1x4x4xf32>) {
        %30 = scf.for %arg4 = %c0_120 to %c4_121 step %c8_122 iter_args(%arg5 = %arg3) -> (tensor<1x4x4xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %33 = affine.min #map(%arg0)
          %34 = affine.min #map1(%arg4)
          %35 = affine.min #map(%arg0)
          %36 = affine.min #map1(%arg2)
          %37 = affine.min #map1(%arg4)
          %extracted_slice = tensor.extract_slice %2[%arg0, %arg2, 0] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_203 = tensor.extract_slice %5[%arg0, %arg4, 0] [%33, %34, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_204 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%35, %36, %37] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x?xf32>
          %38 = linalg.generic {indexing_maps = [#map5, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_203 : tensor<?x?x8xf32>, tensor<?x?x8xf32>) outs(%extracted_slice_204 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %in_205: f32, %out: f32):
            %39 = arith.mulf %in, %in_205 : f32
            %40 = arith.addf %out, %39 : f32
            linalg.yield %40 : f32
          } -> tensor<?x?x?xf32>
          %inserted_slice = tensor.insert_slice %38 into %arg5[%arg0, %arg2, %arg4] [%35, %36, %37] [1, 1, 1] : tensor<?x?x?xf32> into tensor<1x4x4xf32>
          scf.yield %inserted_slice : tensor<1x4x4xf32>
        }
        scf.yield %30 : tensor<1x4x4xf32>
      }
      scf.yield %29 : tensor<1x4x4xf32>
    }
    %12 = tensor.empty() : tensor<1x4x4xf32>
    %c0_123 = arith.constant 0 : index
    %c1_124 = arith.constant 1 : index
    %c8_125 = arith.constant 8 : index
    %c0_126 = arith.constant 0 : index
    %c4_127 = arith.constant 4 : index
    %c8_128 = arith.constant 8 : index
    %c0_129 = arith.constant 0 : index
    %c4_130 = arith.constant 4 : index
    %c8_131 = arith.constant 8 : index
    %13 = scf.for %arg0 = %c0_123 to %c1_124 step %c8_125 iter_args(%arg1 = %12) -> (tensor<1x4x4xf32>) {
      %29 = scf.for %arg2 = %c0_126 to %c4_127 step %c8_128 iter_args(%arg3 = %arg1) -> (tensor<1x4x4xf32>) {
        %30 = scf.for %arg4 = %c0_129 to %c4_130 step %c8_131 iter_args(%arg5 = %arg3) -> (tensor<1x4x4xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %33 = affine.min #map1(%arg4)
          %34 = affine.min #map(%arg0)
          %35 = affine.min #map1(%arg2)
          %36 = affine.min #map1(%arg4)
          %extracted_slice = tensor.extract_slice %11[%arg0, %arg2, %arg4] [%31, %32, %33] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x?xf32>
          %extracted_slice_203 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%34, %35, %36] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x?xf32>
          %37 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<?x?x?xf32>) outs(%extracted_slice_203 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %38 = arith.divf %in, %cst_50 : f32
            linalg.yield %38 : f32
          } -> tensor<?x?x?xf32>
          %inserted_slice = tensor.insert_slice %37 into %arg5[%arg0, %arg2, %arg4] [%34, %35, %36] [1, 1, 1] : tensor<?x?x?xf32> into tensor<1x4x4xf32>
          scf.yield %inserted_slice : tensor<1x4x4xf32>
        }
        scf.yield %30 : tensor<1x4x4xf32>
      }
      scf.yield %29 : tensor<1x4x4xf32>
    }
    %14 = tensor.empty() : tensor<1x4x4xf32>
    %15 = linalg.softmax dimension(2) ins(%13 : tensor<1x4x4xf32>) outs(%14 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %16 = tensor.empty() : tensor<1x4x8xf32>
    %c0_132 = arith.constant 0 : index
    %c1_133 = arith.constant 1 : index
    %c8_134 = arith.constant 8 : index
    %c0_135 = arith.constant 0 : index
    %c4_136 = arith.constant 4 : index
    %c8_137 = arith.constant 8 : index
    %c0_138 = arith.constant 0 : index
    %c8_139 = arith.constant 8 : index
    %c8_140 = arith.constant 8 : index
    %17 = scf.for %arg0 = %c0_132 to %c1_133 step %c8_134 iter_args(%arg1 = %16) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_135 to %c4_136 step %c8_137 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_138 to %c8_139 step %c8_140 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_52 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %c0_141 = arith.constant 0 : index
    %c1_142 = arith.constant 1 : index
    %c8_143 = arith.constant 8 : index
    %c0_144 = arith.constant 0 : index
    %c4_145 = arith.constant 4 : index
    %c8_146 = arith.constant 8 : index
    %c0_147 = arith.constant 0 : index
    %c8_148 = arith.constant 8 : index
    %c8_149 = arith.constant 8 : index
    %18 = scf.for %arg0 = %c0_141 to %c1_142 step %c8_143 iter_args(%arg1 = %17) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_144 to %c4_145 step %c8_146 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_147 to %c8_148 step %c8_149 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %33 = affine.min #map(%arg0)
          %34 = affine.min #map(%arg0)
          %35 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %15[%arg0, %arg2, 0] [%31, %32, 4] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x4xf32>
          %extracted_slice_203 = tensor.extract_slice %8[%arg0, 0, %arg4] [%33, 4, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x4x8xf32>
          %extracted_slice_204 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%34, %35, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %36 = linalg.generic {indexing_maps = [#map5, #map7, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_203 : tensor<?x?x4xf32>, tensor<?x4x8xf32>) outs(%extracted_slice_204 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %in_205: f32, %out: f32):
            %37 = arith.mulf %in, %in_205 : f32
            %38 = arith.addf %out, %37 : f32
            linalg.yield %38 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %36 into %arg5[%arg0, %arg2, %arg4] [%34, %35, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %19 = tensor.empty() : tensor<1x4x8xf32>
    %c0_150 = arith.constant 0 : index
    %c1_151 = arith.constant 1 : index
    %c8_152 = arith.constant 8 : index
    %c0_153 = arith.constant 0 : index
    %c4_154 = arith.constant 4 : index
    %c8_155 = arith.constant 8 : index
    %c0_156 = arith.constant 0 : index
    %c8_157 = arith.constant 8 : index
    %c8_158 = arith.constant 8 : index
    %20 = scf.for %arg0 = %c0_150 to %c1_151 step %c8_152 iter_args(%arg1 = %19) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_153 to %c4_154 step %c8_155 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_156 to %c8_157 step %c8_158 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %33 = affine.min #map(%arg0)
          %34 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %18[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_203 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%33, %34, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %35 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<?x?x8xf32>) outs(%extracted_slice_203 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %36 = arith.addf %in, %cst_51 : f32
            linalg.yield %36 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %35 into %arg5[%arg0, %arg2, %arg4] [%33, %34, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %21 = tensor.empty() : tensor<1x4x32xf32>
    %c0_159 = arith.constant 0 : index
    %c1_160 = arith.constant 1 : index
    %c8_161 = arith.constant 8 : index
    %c0_162 = arith.constant 0 : index
    %c4_163 = arith.constant 4 : index
    %c8_164 = arith.constant 8 : index
    %c0_165 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c8_166 = arith.constant 8 : index
    %22 = scf.for %arg0 = %c0_159 to %c1_160 step %c8_161 iter_args(%arg1 = %21) -> (tensor<1x4x32xf32>) {
      %29 = scf.for %arg2 = %c0_162 to %c4_163 step %c8_164 iter_args(%arg3 = %arg1) -> (tensor<1x4x32xf32>) {
        %30 = scf.for %arg4 = %c0_165 to %c32 step %c8_166 iter_args(%arg5 = %arg3) -> (tensor<1x4x32xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x32xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_52 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x32xf32>
          scf.yield %inserted_slice : tensor<1x4x32xf32>
        }
        scf.yield %30 : tensor<1x4x32xf32>
      }
      scf.yield %29 : tensor<1x4x32xf32>
    }
    %c0_167 = arith.constant 0 : index
    %c1_168 = arith.constant 1 : index
    %c8_169 = arith.constant 8 : index
    %c0_170 = arith.constant 0 : index
    %c4_171 = arith.constant 4 : index
    %c8_172 = arith.constant 8 : index
    %c0_173 = arith.constant 0 : index
    %c32_174 = arith.constant 32 : index
    %c8_175 = arith.constant 8 : index
    %23 = scf.for %arg0 = %c0_167 to %c1_168 step %c8_169 iter_args(%arg1 = %22) -> (tensor<1x4x32xf32>) {
      %29 = scf.for %arg2 = %c0_170 to %c4_171 step %c8_172 iter_args(%arg3 = %arg1) -> (tensor<1x4x32xf32>) {
        %30 = scf.for %arg4 = %c0_173 to %c32_174 step %c8_175 iter_args(%arg5 = %arg3) -> (tensor<1x4x32xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %33 = affine.min #map(%arg0)
          %34 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %20[%arg0, %arg2, 0] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_203 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%33, %34, 8] [1, 1, 1] : tensor<1x4x32xf32> to tensor<?x?x8xf32>
          %35 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<?x?x8xf32>) outs(%extracted_slice_203 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %36 = arith.mulf %in, %cst : f32
            %37 = arith.addf %out, %36 : f32
            linalg.yield %37 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %35 into %arg5[%arg0, %arg2, %arg4] [%33, %34, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x32xf32>
          scf.yield %inserted_slice : tensor<1x4x32xf32>
        }
        scf.yield %30 : tensor<1x4x32xf32>
      }
      scf.yield %29 : tensor<1x4x32xf32>
    }
    %24 = tensor.empty() : tensor<1x4x8xf32>
    %c0_176 = arith.constant 0 : index
    %c1_177 = arith.constant 1 : index
    %c8_178 = arith.constant 8 : index
    %c0_179 = arith.constant 0 : index
    %c4_180 = arith.constant 4 : index
    %c8_181 = arith.constant 8 : index
    %c0_182 = arith.constant 0 : index
    %c8_183 = arith.constant 8 : index
    %c8_184 = arith.constant 8 : index
    %25 = scf.for %arg0 = %c0_176 to %c1_177 step %c8_178 iter_args(%arg1 = %24) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_179 to %c4_180 step %c8_181 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_182 to %c8_183 step %c8_184 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_52 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %c0_185 = arith.constant 0 : index
    %c1_186 = arith.constant 1 : index
    %c8_187 = arith.constant 8 : index
    %c0_188 = arith.constant 0 : index
    %c4_189 = arith.constant 4 : index
    %c8_190 = arith.constant 8 : index
    %c0_191 = arith.constant 0 : index
    %c8_192 = arith.constant 8 : index
    %c8_193 = arith.constant 8 : index
    %26 = scf.for %arg0 = %c0_185 to %c1_186 step %c8_187 iter_args(%arg1 = %25) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_188 to %c4_189 step %c8_190 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_191 to %c8_192 step %c8_193 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %33 = affine.min #map(%arg0)
          %34 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %23[%arg0, %arg2, 0] [%31, %32, 32] [1, 1, 1] : tensor<1x4x32xf32> to tensor<?x?x32xf32>
          %extracted_slice_203 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%33, %34, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %35 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<?x?x32xf32>) outs(%extracted_slice_203 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %36 = arith.maximumf %in, %cst_52 : f32
            %37 = arith.mulf %36, %cst : f32
            %38 = arith.addf %out, %37 : f32
            linalg.yield %38 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %35 into %arg5[%arg0, %arg2, %arg4] [%33, %34, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    %27 = tensor.empty() : tensor<1x4x8xf32>
    %c0_194 = arith.constant 0 : index
    %c1_195 = arith.constant 1 : index
    %c8_196 = arith.constant 8 : index
    %c0_197 = arith.constant 0 : index
    %c4_198 = arith.constant 4 : index
    %c8_199 = arith.constant 8 : index
    %c0_200 = arith.constant 0 : index
    %c8_201 = arith.constant 8 : index
    %c8_202 = arith.constant 8 : index
    %28 = scf.for %arg0 = %c0_194 to %c1_195 step %c8_196 iter_args(%arg1 = %27) -> (tensor<1x4x8xf32>) {
      %29 = scf.for %arg2 = %c0_197 to %c4_198 step %c8_199 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %30 = scf.for %arg4 = %c0_200 to %c8_201 step %c8_202 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %33 = affine.min #map(%arg0)
          %34 = affine.min #map1(%arg2)
          %35 = affine.min #map(%arg0)
          %36 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %20[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_203 = tensor.extract_slice %26[%arg0, %arg2, %arg4] [%33, %34, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_204 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%35, %36, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %37 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_203 : tensor<?x?x8xf32>, tensor<?x?x8xf32>) outs(%extracted_slice_204 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %in_205: f32, %out: f32):
            %38 = arith.addf %in, %in_205 : f32
            %39 = arith.mulf %38, %38 : f32
            %40 = arith.addf %39, %39 : f32
            linalg.yield %40 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %37 into %arg5[%arg0, %arg2, %arg4] [%35, %36, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %30 : tensor<1x4x8xf32>
      }
      scf.yield %29 : tensor<1x4x8xf32>
    }
    return %28 : tensor<1x4x8xf32>
  }
}
