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
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
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
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %45 = arith.mulf %in, %in_16 : f32
      %46 = arith.addf %out, %45 : f32
      linalg.yield %46 : f32
    } -> tensor<1x4x8xf32>
    %3 = tensor.empty() : tensor<1x4x8xf32>
    %cst_6 = arith.constant 0.000000e+00 : f32
    %4 = linalg.fill ins(%cst_6 : f32) outs(%3 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%cst, %cst_1 : tensor<1x4x8xf32>, tensor<8x8xf32>) outs(%4 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %45 = arith.mulf %in, %in_16 : f32
      %46 = arith.addf %out, %45 : f32
      linalg.yield %46 : f32
    } -> tensor<1x4x8xf32>
    %6 = tensor.empty() : tensor<1x4x8xf32>
    %cst_7 = arith.constant 0.000000e+00 : f32
    %7 = linalg.fill ins(%cst_7 : f32) outs(%6 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%cst, %cst_2 : tensor<1x4x8xf32>, tensor<8x8xf32>) outs(%7 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %45 = arith.mulf %in, %in_16 : f32
      %46 = arith.addf %out, %45 : f32
      linalg.yield %46 : f32
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
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %45 = arith.mulf %in, %in_16 : f32
      %46 = arith.addf %out, %45 : f32
      linalg.yield %46 : f32
    } -> tensor<1x4x4xf32>
    %cst_9 = arith.constant dense<2.828400e+00> : tensor<1xf32>
    %c1_i64_10 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %13 = tensor.empty() : tensor<1x4x4xf32>
    %broadcasted = linalg.broadcast ins(%cst_9 : tensor<1xf32>) outs(%13 : tensor<1x4x4xf32>) dimensions = [1, 2] 
    %14 = tensor.empty() : tensor<1x4x4xf32>
    %15 = linalg.div ins(%12, %broadcasted : tensor<1x4x4xf32>, tensor<1x4x4xf32>) outs(%14 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %16 = tensor.empty() : tensor<1x4xf32>
    %cst_11 = arith.constant 0xFF800000 : f32
    %17 = linalg.fill ins(%cst_11 : f32) outs(%16 : tensor<1x4xf32>) -> tensor<1x4xf32>
    %18 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : tensor<1x4x4xf32>) outs(%17 : tensor<1x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %45 = arith.maxnumf %in, %out : f32
      linalg.yield %45 : f32
    } -> tensor<1x4xf32>
    %19 = tensor.empty() : tensor<1x4x4xf32>
    %20 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %18 : tensor<1x4x4xf32>, tensor<1x4xf32>) outs(%19 : tensor<1x4x4xf32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %45 = arith.subf %in, %in_16 : f32
      %46 = math.exp %45 : f32
      linalg.yield %46 : f32
    } -> tensor<1x4x4xf32>
    %21 = tensor.empty() : tensor<1x4xf32>
    %cst_12 = arith.constant 0.000000e+00 : f32
    %22 = linalg.fill ins(%cst_12 : f32) outs(%21 : tensor<1x4xf32>) -> tensor<1x4xf32>
    %23 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20 : tensor<1x4x4xf32>) outs(%22 : tensor<1x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %45 = arith.addf %in, %out : f32
      linalg.yield %45 : f32
    } -> tensor<1x4xf32>
    %24 = tensor.empty() : tensor<1x4x4xf32>
    %25 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20, %23 : tensor<1x4x4xf32>, tensor<1x4xf32>) outs(%24 : tensor<1x4x4xf32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %45 = arith.divf %in, %in_16 : f32
      linalg.yield %45 : f32
    } -> tensor<1x4x4xf32>
    %26 = tensor.empty() : tensor<1x4x8xf32>
    %cst_13 = arith.constant 0.000000e+00 : f32
    %27 = linalg.fill ins(%cst_13 : f32) outs(%26 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %28 = linalg.generic {indexing_maps = [#map, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%25, %8 : tensor<1x4x4xf32>, tensor<1x4x8xf32>) outs(%27 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %45 = arith.mulf %in, %in_16 : f32
      %46 = arith.addf %out, %45 : f32
      linalg.yield %46 : f32
    } -> tensor<1x4x8xf32>
    %29 = tensor.empty() : tensor<1x4x8xf32>
    %30 = linalg.add ins(%cst, %28 : tensor<1x4x8xf32>, tensor<1x4x8xf32>) outs(%29 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %31 = tensor.empty() : tensor<1x4x32xf32>
    %cst_14 = arith.constant 0.000000e+00 : f32
    %32 = linalg.fill ins(%cst_14 : f32) outs(%31 : tensor<1x4x32xf32>) -> tensor<1x4x32xf32>
    %33 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%30, %cst_3 : tensor<1x4x8xf32>, tensor<8x32xf32>) outs(%32 : tensor<1x4x32xf32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %45 = arith.mulf %in, %in_16 : f32
      %46 = arith.addf %out, %45 : f32
      linalg.yield %46 : f32
    } -> tensor<1x4x32xf32>
    %34 = tensor.empty() : tensor<1x4x32xf32>
    %35 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33 : tensor<1x4x32xf32>) outs(%34 : tensor<1x4x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_16 = arith.constant 0.000000e+00 : f32
      %45 = arith.maximumf %in, %cst_16 : f32
      linalg.yield %45 : f32
    } -> tensor<1x4x32xf32>
    %36 = tensor.empty() : tensor<1x4x8xf32>
    %cst_15 = arith.constant 0.000000e+00 : f32
    %37 = linalg.fill ins(%cst_15 : f32) outs(%36 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %38 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%35, %cst_4 : tensor<1x4x32xf32>, tensor<32x8xf32>) outs(%37 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %45 = arith.mulf %in, %in_16 : f32
      %46 = arith.addf %out, %45 : f32
      linalg.yield %46 : f32
    } -> tensor<1x4x8xf32>
    %39 = tensor.empty() : tensor<1x4x8xf32>
    %40 = linalg.add ins(%30, %38 : tensor<1x4x8xf32>, tensor<1x4x8xf32>) outs(%39 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %41 = tensor.empty() : tensor<1x4x8xf32>
    %42 = linalg.mul ins(%40, %40 : tensor<1x4x8xf32>, tensor<1x4x8xf32>) outs(%41 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %43 = tensor.empty() : tensor<1x4x8xf32>
    %44 = linalg.add ins(%42, %42 : tensor<1x4x8xf32>, tensor<1x4x8xf32>) outs(%43 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    return %44 : tensor<1x4x8xf32>
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
#map7 = affine_map<(d0, d1) -> (d0, d1)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
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
    %c8_50 = arith.constant 8 : index
    %c8_51 = arith.constant 8 : index
    %c8_52 = arith.constant 8 : index
    %c8_53 = arith.constant 8 : index
    %c8_54 = arith.constant 8 : index
    %c8_55 = arith.constant 8 : index
    %c8_56 = arith.constant 8 : index
    %c8_57 = arith.constant 8 : index
    %c8_58 = arith.constant 8 : index
    %c8_59 = arith.constant 8 : index
    %c8_60 = arith.constant 8 : index
    %c8_61 = arith.constant 8 : index
    %c8_62 = arith.constant 8 : index
    %c8_63 = arith.constant 8 : index
    %c8_64 = arith.constant 8 : index
    %cst = arith.constant 2.000000e-01 : f32
    %cst_65 = arith.constant 0xFF800000 : f32
    %cst_66 = arith.constant 2.828400e+00 : f32
    %cst_67 = arith.constant 5.000000e-01 : f32
    %cst_68 = arith.constant 0.000000e+00 : f32
    %cst_69 = arith.constant dense<1.000000e-01> : tensor<8x8xf32>
    %0 = tensor.empty() : tensor<1x4x8xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_70 = arith.constant 8 : index
    %c0_71 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8_72 = arith.constant 8 : index
    %c0_73 = arith.constant 0 : index
    %c8_74 = arith.constant 8 : index
    %c8_75 = arith.constant 8 : index
    %1 = scf.for %arg0 = %c0 to %c1 step %c8_70 iter_args(%arg1 = %0) -> (tensor<1x4x8xf32>) {
      %35 = scf.for %arg2 = %c0_71 to %c4 step %c8_72 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %36 = scf.for %arg4 = %c0_73 to %c8_74 step %c8_75 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %39 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_68 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %39 into %arg5[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %36 : tensor<1x4x8xf32>
      }
      scf.yield %35 : tensor<1x4x8xf32>
    }
    %c0_76 = arith.constant 0 : index
    %c1_77 = arith.constant 1 : index
    %c8_78 = arith.constant 8 : index
    %c0_79 = arith.constant 0 : index
    %c4_80 = arith.constant 4 : index
    %c8_81 = arith.constant 8 : index
    %c0_82 = arith.constant 0 : index
    %c8_83 = arith.constant 8 : index
    %c8_84 = arith.constant 8 : index
    %2 = scf.for %arg0 = %c0_76 to %c1_77 step %c8_78 iter_args(%arg1 = %1) -> (tensor<1x4x8xf32>) {
      %35 = scf.for %arg2 = %c0_79 to %c4_80 step %c8_81 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %36 = scf.for %arg4 = %c0_82 to %c8_83 step %c8_84 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %cst_69[0, %arg4] [8, 8] [1, 1] : tensor<8x8xf32> to tensor<8x8xf32>
          %extracted_slice_258 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %39 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<8x8xf32>) outs(%extracted_slice_258 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %40 = arith.mulf %in, %cst_67 : f32
            %41 = arith.addf %out, %40 : f32
            linalg.yield %41 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %39 into %arg5[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %36 : tensor<1x4x8xf32>
      }
      scf.yield %35 : tensor<1x4x8xf32>
    }
    %3 = tensor.empty() : tensor<1x4x8xf32>
    %c0_85 = arith.constant 0 : index
    %c1_86 = arith.constant 1 : index
    %c8_87 = arith.constant 8 : index
    %c0_88 = arith.constant 0 : index
    %c4_89 = arith.constant 4 : index
    %c8_90 = arith.constant 8 : index
    %c0_91 = arith.constant 0 : index
    %c8_92 = arith.constant 8 : index
    %c8_93 = arith.constant 8 : index
    %4 = scf.for %arg0 = %c0_85 to %c1_86 step %c8_87 iter_args(%arg1 = %3) -> (tensor<1x4x8xf32>) {
      %35 = scf.for %arg2 = %c0_88 to %c4_89 step %c8_90 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %36 = scf.for %arg4 = %c0_91 to %c8_92 step %c8_93 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %39 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_68 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %39 into %arg5[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %36 : tensor<1x4x8xf32>
      }
      scf.yield %35 : tensor<1x4x8xf32>
    }
    %c0_94 = arith.constant 0 : index
    %c1_95 = arith.constant 1 : index
    %c8_96 = arith.constant 8 : index
    %c0_97 = arith.constant 0 : index
    %c4_98 = arith.constant 4 : index
    %c8_99 = arith.constant 8 : index
    %c0_100 = arith.constant 0 : index
    %c8_101 = arith.constant 8 : index
    %c8_102 = arith.constant 8 : index
    %5 = scf.for %arg0 = %c0_94 to %c1_95 step %c8_96 iter_args(%arg1 = %4) -> (tensor<1x4x8xf32>) {
      %35 = scf.for %arg2 = %c0_97 to %c4_98 step %c8_99 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %36 = scf.for %arg4 = %c0_100 to %c8_101 step %c8_102 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %cst_69[0, %arg4] [8, 8] [1, 1] : tensor<8x8xf32> to tensor<8x8xf32>
          %extracted_slice_258 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %39 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<8x8xf32>) outs(%extracted_slice_258 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %40 = arith.mulf %in, %cst_67 : f32
            %41 = arith.addf %out, %40 : f32
            linalg.yield %41 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %39 into %arg5[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %36 : tensor<1x4x8xf32>
      }
      scf.yield %35 : tensor<1x4x8xf32>
    }
    %6 = tensor.empty() : tensor<1x4x8xf32>
    %c0_103 = arith.constant 0 : index
    %c1_104 = arith.constant 1 : index
    %c8_105 = arith.constant 8 : index
    %c0_106 = arith.constant 0 : index
    %c4_107 = arith.constant 4 : index
    %c8_108 = arith.constant 8 : index
    %c0_109 = arith.constant 0 : index
    %c8_110 = arith.constant 8 : index
    %c8_111 = arith.constant 8 : index
    %7 = scf.for %arg0 = %c0_103 to %c1_104 step %c8_105 iter_args(%arg1 = %6) -> (tensor<1x4x8xf32>) {
      %35 = scf.for %arg2 = %c0_106 to %c4_107 step %c8_108 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %36 = scf.for %arg4 = %c0_109 to %c8_110 step %c8_111 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %39 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_68 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %39 into %arg5[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %36 : tensor<1x4x8xf32>
      }
      scf.yield %35 : tensor<1x4x8xf32>
    }
    %c0_112 = arith.constant 0 : index
    %c1_113 = arith.constant 1 : index
    %c8_114 = arith.constant 8 : index
    %c0_115 = arith.constant 0 : index
    %c4_116 = arith.constant 4 : index
    %c8_117 = arith.constant 8 : index
    %c0_118 = arith.constant 0 : index
    %c8_119 = arith.constant 8 : index
    %c8_120 = arith.constant 8 : index
    %8 = scf.for %arg0 = %c0_112 to %c1_113 step %c8_114 iter_args(%arg1 = %7) -> (tensor<1x4x8xf32>) {
      %35 = scf.for %arg2 = %c0_115 to %c4_116 step %c8_117 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %36 = scf.for %arg4 = %c0_118 to %c8_119 step %c8_120 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %cst_69[0, %arg4] [8, 8] [1, 1] : tensor<8x8xf32> to tensor<8x8xf32>
          %extracted_slice_258 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %39 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<8x8xf32>) outs(%extracted_slice_258 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %40 = arith.mulf %in, %cst_67 : f32
            %41 = arith.addf %out, %40 : f32
            linalg.yield %41 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %39 into %arg5[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %36 : tensor<1x4x8xf32>
      }
      scf.yield %35 : tensor<1x4x8xf32>
    }
    %9 = tensor.empty() : tensor<1x4x4xf32>
    %c0_121 = arith.constant 0 : index
    %c1_122 = arith.constant 1 : index
    %c8_123 = arith.constant 8 : index
    %c0_124 = arith.constant 0 : index
    %c4_125 = arith.constant 4 : index
    %c8_126 = arith.constant 8 : index
    %c0_127 = arith.constant 0 : index
    %c4_128 = arith.constant 4 : index
    %c8_129 = arith.constant 8 : index
    %10 = scf.for %arg0 = %c0_121 to %c1_122 step %c8_123 iter_args(%arg1 = %9) -> (tensor<1x4x4xf32>) {
      %35 = scf.for %arg2 = %c0_124 to %c4_125 step %c8_126 iter_args(%arg3 = %arg1) -> (tensor<1x4x4xf32>) {
        %36 = scf.for %arg4 = %c0_127 to %c4_128 step %c8_129 iter_args(%arg5 = %arg3) -> (tensor<1x4x4xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %39 = affine.min #map1(%arg4)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%37, %38, %39] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x?xf32>
          %40 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x?xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_68 : f32
          } -> tensor<?x?x?xf32>
          %inserted_slice = tensor.insert_slice %40 into %arg5[%arg0, %arg2, %arg4] [%37, %38, %39] [1, 1, 1] : tensor<?x?x?xf32> into tensor<1x4x4xf32>
          scf.yield %inserted_slice : tensor<1x4x4xf32>
        }
        scf.yield %36 : tensor<1x4x4xf32>
      }
      scf.yield %35 : tensor<1x4x4xf32>
    }
    %c0_130 = arith.constant 0 : index
    %c1_131 = arith.constant 1 : index
    %c8_132 = arith.constant 8 : index
    %c0_133 = arith.constant 0 : index
    %c4_134 = arith.constant 4 : index
    %c8_135 = arith.constant 8 : index
    %c0_136 = arith.constant 0 : index
    %c4_137 = arith.constant 4 : index
    %c8_138 = arith.constant 8 : index
    %11 = scf.for %arg0 = %c0_130 to %c1_131 step %c8_132 iter_args(%arg1 = %10) -> (tensor<1x4x4xf32>) {
      %35 = scf.for %arg2 = %c0_133 to %c4_134 step %c8_135 iter_args(%arg3 = %arg1) -> (tensor<1x4x4xf32>) {
        %36 = scf.for %arg4 = %c0_136 to %c4_137 step %c8_138 iter_args(%arg5 = %arg3) -> (tensor<1x4x4xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %39 = affine.min #map(%arg0)
          %40 = affine.min #map1(%arg4)
          %41 = affine.min #map(%arg0)
          %42 = affine.min #map1(%arg2)
          %43 = affine.min #map1(%arg4)
          %extracted_slice = tensor.extract_slice %2[%arg0, %arg2, 0] [%37, %38, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_258 = tensor.extract_slice %5[%arg0, %arg4, 0] [%39, %40, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_259 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%41, %42, %43] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x?xf32>
          %44 = linalg.generic {indexing_maps = [#map5, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_258 : tensor<?x?x8xf32>, tensor<?x?x8xf32>) outs(%extracted_slice_259 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %in_260: f32, %out: f32):
            %45 = arith.mulf %in, %in_260 : f32
            %46 = arith.addf %out, %45 : f32
            linalg.yield %46 : f32
          } -> tensor<?x?x?xf32>
          %inserted_slice = tensor.insert_slice %44 into %arg5[%arg0, %arg2, %arg4] [%41, %42, %43] [1, 1, 1] : tensor<?x?x?xf32> into tensor<1x4x4xf32>
          scf.yield %inserted_slice : tensor<1x4x4xf32>
        }
        scf.yield %36 : tensor<1x4x4xf32>
      }
      scf.yield %35 : tensor<1x4x4xf32>
    }
    %12 = tensor.empty() : tensor<1x4x4xf32>
    %c0_139 = arith.constant 0 : index
    %c1_140 = arith.constant 1 : index
    %c8_141 = arith.constant 8 : index
    %c0_142 = arith.constant 0 : index
    %c4_143 = arith.constant 4 : index
    %c8_144 = arith.constant 8 : index
    %c0_145 = arith.constant 0 : index
    %c4_146 = arith.constant 4 : index
    %c8_147 = arith.constant 8 : index
    %13 = scf.for %arg0 = %c0_139 to %c1_140 step %c8_141 iter_args(%arg1 = %12) -> (tensor<1x4x4xf32>) {
      %35 = scf.for %arg2 = %c0_142 to %c4_143 step %c8_144 iter_args(%arg3 = %arg1) -> (tensor<1x4x4xf32>) {
        %36 = scf.for %arg4 = %c0_145 to %c4_146 step %c8_147 iter_args(%arg5 = %arg3) -> (tensor<1x4x4xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %39 = affine.min #map1(%arg4)
          %40 = affine.min #map(%arg0)
          %41 = affine.min #map1(%arg2)
          %42 = affine.min #map1(%arg4)
          %extracted_slice = tensor.extract_slice %11[%arg0, %arg2, %arg4] [%37, %38, %39] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x?xf32>
          %extracted_slice_258 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%40, %41, %42] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x?xf32>
          %43 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<?x?x?xf32>) outs(%extracted_slice_258 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %44 = arith.divf %in, %cst_66 : f32
            linalg.yield %44 : f32
          } -> tensor<?x?x?xf32>
          %inserted_slice = tensor.insert_slice %43 into %arg5[%arg0, %arg2, %arg4] [%40, %41, %42] [1, 1, 1] : tensor<?x?x?xf32> into tensor<1x4x4xf32>
          scf.yield %inserted_slice : tensor<1x4x4xf32>
        }
        scf.yield %36 : tensor<1x4x4xf32>
      }
      scf.yield %35 : tensor<1x4x4xf32>
    }
    %14 = tensor.empty() : tensor<1x4xf32>
    %c0_148 = arith.constant 0 : index
    %c1_149 = arith.constant 1 : index
    %c8_150 = arith.constant 8 : index
    %c0_151 = arith.constant 0 : index
    %c4_152 = arith.constant 4 : index
    %c8_153 = arith.constant 8 : index
    %15 = scf.for %arg0 = %c0_148 to %c1_149 step %c8_150 iter_args(%arg1 = %14) -> (tensor<1x4xf32>) {
      %35 = scf.for %arg2 = %c0_151 to %c4_152 step %c8_153 iter_args(%arg3 = %arg1) -> (tensor<1x4xf32>) {
        %36 = affine.min #map(%arg0)
        %37 = affine.min #map1(%arg2)
        %extracted_slice = tensor.extract_slice %arg3[%arg0, %arg2] [%36, %37] [1, 1] : tensor<1x4xf32> to tensor<?x?xf32>
        %38 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice : tensor<?x?xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_65 : f32
        } -> tensor<?x?xf32>
        %inserted_slice = tensor.insert_slice %38 into %arg3[%arg0, %arg2] [%36, %37] [1, 1] : tensor<?x?xf32> into tensor<1x4xf32>
        scf.yield %inserted_slice : tensor<1x4xf32>
      }
      scf.yield %35 : tensor<1x4xf32>
    }
    %c0_154 = arith.constant 0 : index
    %c1_155 = arith.constant 1 : index
    %c8_156 = arith.constant 8 : index
    %c0_157 = arith.constant 0 : index
    %c4_158 = arith.constant 4 : index
    %c8_159 = arith.constant 8 : index
    %c0_160 = arith.constant 0 : index
    %c4_161 = arith.constant 4 : index
    %c8_162 = arith.constant 8 : index
    %16 = scf.for %arg0 = %c0_154 to %c1_155 step %c8_156 iter_args(%arg1 = %15) -> (tensor<1x4xf32>) {
      %35 = scf.for %arg2 = %c0_157 to %c4_158 step %c8_159 iter_args(%arg3 = %arg1) -> (tensor<1x4xf32>) {
        %36 = scf.for %arg4 = %c0_160 to %c4_161 step %c8_162 iter_args(%arg5 = %arg3) -> (tensor<1x4xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %39 = affine.min #map1(%arg4)
          %40 = affine.min #map(%arg0)
          %41 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %13[%arg0, %arg2, %arg4] [%37, %38, %39] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x?xf32>
          %extracted_slice_258 = tensor.extract_slice %arg5[%arg0, %arg2] [%40, %41] [1, 1] : tensor<1x4xf32> to tensor<?x?xf32>
          %42 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<?x?x?xf32>) outs(%extracted_slice_258 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %43 = arith.maxnumf %in, %out : f32
            linalg.yield %43 : f32
          } -> tensor<?x?xf32>
          %inserted_slice = tensor.insert_slice %42 into %arg5[%arg0, %arg2] [%40, %41] [1, 1] : tensor<?x?xf32> into tensor<1x4xf32>
          scf.yield %inserted_slice : tensor<1x4xf32>
        }
        scf.yield %36 : tensor<1x4xf32>
      }
      scf.yield %35 : tensor<1x4xf32>
    }
    %17 = tensor.empty() : tensor<1x4x4xf32>
    %c0_163 = arith.constant 0 : index
    %c1_164 = arith.constant 1 : index
    %c8_165 = arith.constant 8 : index
    %c0_166 = arith.constant 0 : index
    %c4_167 = arith.constant 4 : index
    %c8_168 = arith.constant 8 : index
    %c0_169 = arith.constant 0 : index
    %c4_170 = arith.constant 4 : index
    %c8_171 = arith.constant 8 : index
    %18 = scf.for %arg0 = %c0_163 to %c1_164 step %c8_165 iter_args(%arg1 = %17) -> (tensor<1x4x4xf32>) {
      %35 = scf.for %arg2 = %c0_166 to %c4_167 step %c8_168 iter_args(%arg3 = %arg1) -> (tensor<1x4x4xf32>) {
        %36 = scf.for %arg4 = %c0_169 to %c4_170 step %c8_171 iter_args(%arg5 = %arg3) -> (tensor<1x4x4xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %39 = affine.min #map1(%arg4)
          %40 = affine.min #map(%arg0)
          %41 = affine.min #map1(%arg2)
          %42 = affine.min #map(%arg0)
          %43 = affine.min #map1(%arg2)
          %44 = affine.min #map1(%arg4)
          %extracted_slice = tensor.extract_slice %13[%arg0, %arg2, %arg4] [%37, %38, %39] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x?xf32>
          %extracted_slice_258 = tensor.extract_slice %16[%arg0, %arg2] [%40, %41] [1, 1] : tensor<1x4xf32> to tensor<?x?xf32>
          %extracted_slice_259 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%42, %43, %44] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x?xf32>
          %45 = linalg.generic {indexing_maps = [#map2, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_258 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_259 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %in_260: f32, %out: f32):
            %46 = arith.subf %in, %in_260 : f32
            %47 = math.exp %46 : f32
            linalg.yield %47 : f32
          } -> tensor<?x?x?xf32>
          %inserted_slice = tensor.insert_slice %45 into %arg5[%arg0, %arg2, %arg4] [%42, %43, %44] [1, 1, 1] : tensor<?x?x?xf32> into tensor<1x4x4xf32>
          scf.yield %inserted_slice : tensor<1x4x4xf32>
        }
        scf.yield %36 : tensor<1x4x4xf32>
      }
      scf.yield %35 : tensor<1x4x4xf32>
    }
    %19 = tensor.empty() : tensor<1x4xf32>
    %c0_172 = arith.constant 0 : index
    %c1_173 = arith.constant 1 : index
    %c8_174 = arith.constant 8 : index
    %c0_175 = arith.constant 0 : index
    %c4_176 = arith.constant 4 : index
    %c8_177 = arith.constant 8 : index
    %20 = scf.for %arg0 = %c0_172 to %c1_173 step %c8_174 iter_args(%arg1 = %19) -> (tensor<1x4xf32>) {
      %35 = scf.for %arg2 = %c0_175 to %c4_176 step %c8_177 iter_args(%arg3 = %arg1) -> (tensor<1x4xf32>) {
        %36 = affine.min #map(%arg0)
        %37 = affine.min #map1(%arg2)
        %extracted_slice = tensor.extract_slice %arg3[%arg0, %arg2] [%36, %37] [1, 1] : tensor<1x4xf32> to tensor<?x?xf32>
        %38 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice : tensor<?x?xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_68 : f32
        } -> tensor<?x?xf32>
        %inserted_slice = tensor.insert_slice %38 into %arg3[%arg0, %arg2] [%36, %37] [1, 1] : tensor<?x?xf32> into tensor<1x4xf32>
        scf.yield %inserted_slice : tensor<1x4xf32>
      }
      scf.yield %35 : tensor<1x4xf32>
    }
    %c0_178 = arith.constant 0 : index
    %c1_179 = arith.constant 1 : index
    %c8_180 = arith.constant 8 : index
    %c0_181 = arith.constant 0 : index
    %c4_182 = arith.constant 4 : index
    %c8_183 = arith.constant 8 : index
    %c0_184 = arith.constant 0 : index
    %c4_185 = arith.constant 4 : index
    %c8_186 = arith.constant 8 : index
    %21 = scf.for %arg0 = %c0_178 to %c1_179 step %c8_180 iter_args(%arg1 = %20) -> (tensor<1x4xf32>) {
      %35 = scf.for %arg2 = %c0_181 to %c4_182 step %c8_183 iter_args(%arg3 = %arg1) -> (tensor<1x4xf32>) {
        %36 = scf.for %arg4 = %c0_184 to %c4_185 step %c8_186 iter_args(%arg5 = %arg3) -> (tensor<1x4xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %39 = affine.min #map1(%arg4)
          %40 = affine.min #map(%arg0)
          %41 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %18[%arg0, %arg2, %arg4] [%37, %38, %39] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x?xf32>
          %extracted_slice_258 = tensor.extract_slice %arg5[%arg0, %arg2] [%40, %41] [1, 1] : tensor<1x4xf32> to tensor<?x?xf32>
          %42 = linalg.generic {indexing_maps = [#map2, #map8], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<?x?x?xf32>) outs(%extracted_slice_258 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %43 = arith.addf %in, %out : f32
            linalg.yield %43 : f32
          } -> tensor<?x?xf32>
          %inserted_slice = tensor.insert_slice %42 into %arg5[%arg0, %arg2] [%40, %41] [1, 1] : tensor<?x?xf32> into tensor<1x4xf32>
          scf.yield %inserted_slice : tensor<1x4xf32>
        }
        scf.yield %36 : tensor<1x4xf32>
      }
      scf.yield %35 : tensor<1x4xf32>
    }
    %22 = tensor.empty() : tensor<1x4x8xf32>
    %c0_187 = arith.constant 0 : index
    %c1_188 = arith.constant 1 : index
    %c8_189 = arith.constant 8 : index
    %c0_190 = arith.constant 0 : index
    %c4_191 = arith.constant 4 : index
    %c8_192 = arith.constant 8 : index
    %c0_193 = arith.constant 0 : index
    %c8_194 = arith.constant 8 : index
    %c8_195 = arith.constant 8 : index
    %23 = scf.for %arg0 = %c0_187 to %c1_188 step %c8_189 iter_args(%arg1 = %22) -> (tensor<1x4x8xf32>) {
      %35 = scf.for %arg2 = %c0_190 to %c4_191 step %c8_192 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %36 = scf.for %arg4 = %c0_193 to %c8_194 step %c8_195 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %39 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_68 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %39 into %arg5[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %36 : tensor<1x4x8xf32>
      }
      scf.yield %35 : tensor<1x4x8xf32>
    }
    %c0_196 = arith.constant 0 : index
    %c1_197 = arith.constant 1 : index
    %c8_198 = arith.constant 8 : index
    %c0_199 = arith.constant 0 : index
    %c4_200 = arith.constant 4 : index
    %c8_201 = arith.constant 8 : index
    %c0_202 = arith.constant 0 : index
    %c8_203 = arith.constant 8 : index
    %c8_204 = arith.constant 8 : index
    %24 = scf.for %arg0 = %c0_196 to %c1_197 step %c8_198 iter_args(%arg1 = %23) -> (tensor<1x4x8xf32>) {
      %35 = scf.for %arg2 = %c0_199 to %c4_200 step %c8_201 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %36 = scf.for %arg4 = %c0_202 to %c8_203 step %c8_204 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %39 = affine.min #map(%arg0)
          %40 = affine.min #map1(%arg2)
          %41 = affine.min #map(%arg0)
          %42 = affine.min #map(%arg0)
          %43 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %18[%arg0, %arg2, 0] [%37, %38, 4] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x4xf32>
          %extracted_slice_258 = tensor.extract_slice %21[%arg0, %arg2] [%39, %40] [1, 1] : tensor<1x4xf32> to tensor<?x?xf32>
          %extracted_slice_259 = tensor.extract_slice %8[%arg0, 0, %arg4] [%41, 4, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x4x8xf32>
          %extracted_slice_260 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%42, %43, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %44 = linalg.generic {indexing_maps = [#map5, #map9, #map10, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_258, %extracted_slice_259 : tensor<?x?x4xf32>, tensor<?x?xf32>, tensor<?x4x8xf32>) outs(%extracted_slice_260 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %in_261: f32, %in_262: f32, %out: f32):
            %45 = arith.divf %in, %in_261 : f32
            %46 = arith.mulf %45, %in_262 : f32
            %47 = arith.addf %out, %46 : f32
            linalg.yield %47 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %44 into %arg5[%arg0, %arg2, %arg4] [%42, %43, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %36 : tensor<1x4x8xf32>
      }
      scf.yield %35 : tensor<1x4x8xf32>
    }
    %25 = tensor.empty() : tensor<1x4x8xf32>
    %c0_205 = arith.constant 0 : index
    %c1_206 = arith.constant 1 : index
    %c8_207 = arith.constant 8 : index
    %c0_208 = arith.constant 0 : index
    %c4_209 = arith.constant 4 : index
    %c8_210 = arith.constant 8 : index
    %c0_211 = arith.constant 0 : index
    %c8_212 = arith.constant 8 : index
    %c8_213 = arith.constant 8 : index
    %26 = scf.for %arg0 = %c0_205 to %c1_206 step %c8_207 iter_args(%arg1 = %25) -> (tensor<1x4x8xf32>) {
      %35 = scf.for %arg2 = %c0_208 to %c4_209 step %c8_210 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %36 = scf.for %arg4 = %c0_211 to %c8_212 step %c8_213 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %39 = affine.min #map(%arg0)
          %40 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %24[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_258 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%39, %40, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %41 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<?x?x8xf32>) outs(%extracted_slice_258 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %42 = arith.addf %in, %cst_67 : f32
            linalg.yield %42 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %41 into %arg5[%arg0, %arg2, %arg4] [%39, %40, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %36 : tensor<1x4x8xf32>
      }
      scf.yield %35 : tensor<1x4x8xf32>
    }
    %27 = tensor.empty() : tensor<1x4x32xf32>
    %c0_214 = arith.constant 0 : index
    %c1_215 = arith.constant 1 : index
    %c8_216 = arith.constant 8 : index
    %c0_217 = arith.constant 0 : index
    %c4_218 = arith.constant 4 : index
    %c8_219 = arith.constant 8 : index
    %c0_220 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c8_221 = arith.constant 8 : index
    %28 = scf.for %arg0 = %c0_214 to %c1_215 step %c8_216 iter_args(%arg1 = %27) -> (tensor<1x4x32xf32>) {
      %35 = scf.for %arg2 = %c0_217 to %c4_218 step %c8_219 iter_args(%arg3 = %arg1) -> (tensor<1x4x32xf32>) {
        %36 = scf.for %arg4 = %c0_220 to %c32 step %c8_221 iter_args(%arg5 = %arg3) -> (tensor<1x4x32xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<1x4x32xf32> to tensor<?x?x8xf32>
          %39 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_68 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %39 into %arg5[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x32xf32>
          scf.yield %inserted_slice : tensor<1x4x32xf32>
        }
        scf.yield %36 : tensor<1x4x32xf32>
      }
      scf.yield %35 : tensor<1x4x32xf32>
    }
    %c0_222 = arith.constant 0 : index
    %c1_223 = arith.constant 1 : index
    %c8_224 = arith.constant 8 : index
    %c0_225 = arith.constant 0 : index
    %c4_226 = arith.constant 4 : index
    %c8_227 = arith.constant 8 : index
    %c0_228 = arith.constant 0 : index
    %c32_229 = arith.constant 32 : index
    %c8_230 = arith.constant 8 : index
    %29 = scf.for %arg0 = %c0_222 to %c1_223 step %c8_224 iter_args(%arg1 = %28) -> (tensor<1x4x32xf32>) {
      %35 = scf.for %arg2 = %c0_225 to %c4_226 step %c8_227 iter_args(%arg3 = %arg1) -> (tensor<1x4x32xf32>) {
        %36 = scf.for %arg4 = %c0_228 to %c32_229 step %c8_230 iter_args(%arg5 = %arg3) -> (tensor<1x4x32xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %39 = affine.min #map(%arg0)
          %40 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %26[%arg0, %arg2, 0] [%37, %38, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_258 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%39, %40, 8] [1, 1, 1] : tensor<1x4x32xf32> to tensor<?x?x8xf32>
          %41 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<?x?x8xf32>) outs(%extracted_slice_258 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %42 = arith.mulf %in, %cst : f32
            %43 = arith.addf %out, %42 : f32
            linalg.yield %43 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %41 into %arg5[%arg0, %arg2, %arg4] [%39, %40, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x32xf32>
          scf.yield %inserted_slice : tensor<1x4x32xf32>
        }
        scf.yield %36 : tensor<1x4x32xf32>
      }
      scf.yield %35 : tensor<1x4x32xf32>
    }
    %30 = tensor.empty() : tensor<1x4x8xf32>
    %c0_231 = arith.constant 0 : index
    %c1_232 = arith.constant 1 : index
    %c8_233 = arith.constant 8 : index
    %c0_234 = arith.constant 0 : index
    %c4_235 = arith.constant 4 : index
    %c8_236 = arith.constant 8 : index
    %c0_237 = arith.constant 0 : index
    %c8_238 = arith.constant 8 : index
    %c8_239 = arith.constant 8 : index
    %31 = scf.for %arg0 = %c0_231 to %c1_232 step %c8_233 iter_args(%arg1 = %30) -> (tensor<1x4x8xf32>) {
      %35 = scf.for %arg2 = %c0_234 to %c4_235 step %c8_236 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %36 = scf.for %arg4 = %c0_237 to %c8_238 step %c8_239 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %39 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_68 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %39 into %arg5[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %36 : tensor<1x4x8xf32>
      }
      scf.yield %35 : tensor<1x4x8xf32>
    }
    %c0_240 = arith.constant 0 : index
    %c1_241 = arith.constant 1 : index
    %c8_242 = arith.constant 8 : index
    %c0_243 = arith.constant 0 : index
    %c4_244 = arith.constant 4 : index
    %c8_245 = arith.constant 8 : index
    %c0_246 = arith.constant 0 : index
    %c8_247 = arith.constant 8 : index
    %c8_248 = arith.constant 8 : index
    %32 = scf.for %arg0 = %c0_240 to %c1_241 step %c8_242 iter_args(%arg1 = %31) -> (tensor<1x4x8xf32>) {
      %35 = scf.for %arg2 = %c0_243 to %c4_244 step %c8_245 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %36 = scf.for %arg4 = %c0_246 to %c8_247 step %c8_248 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %39 = affine.min #map(%arg0)
          %40 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %29[%arg0, %arg2, 0] [%37, %38, 32] [1, 1, 1] : tensor<1x4x32xf32> to tensor<?x?x32xf32>
          %extracted_slice_258 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%39, %40, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %41 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<?x?x32xf32>) outs(%extracted_slice_258 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %42 = arith.maximumf %in, %cst_68 : f32
            %43 = arith.mulf %42, %cst : f32
            %44 = arith.addf %out, %43 : f32
            linalg.yield %44 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %41 into %arg5[%arg0, %arg2, %arg4] [%39, %40, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %36 : tensor<1x4x8xf32>
      }
      scf.yield %35 : tensor<1x4x8xf32>
    }
    %33 = tensor.empty() : tensor<1x4x8xf32>
    %c0_249 = arith.constant 0 : index
    %c1_250 = arith.constant 1 : index
    %c8_251 = arith.constant 8 : index
    %c0_252 = arith.constant 0 : index
    %c4_253 = arith.constant 4 : index
    %c8_254 = arith.constant 8 : index
    %c0_255 = arith.constant 0 : index
    %c8_256 = arith.constant 8 : index
    %c8_257 = arith.constant 8 : index
    %34 = scf.for %arg0 = %c0_249 to %c1_250 step %c8_251 iter_args(%arg1 = %33) -> (tensor<1x4x8xf32>) {
      %35 = scf.for %arg2 = %c0_252 to %c4_253 step %c8_254 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %36 = scf.for %arg4 = %c0_255 to %c8_256 step %c8_257 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %37 = affine.min #map(%arg0)
          %38 = affine.min #map1(%arg2)
          %39 = affine.min #map(%arg0)
          %40 = affine.min #map1(%arg2)
          %41 = affine.min #map(%arg0)
          %42 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %26[%arg0, %arg2, %arg4] [%37, %38, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_258 = tensor.extract_slice %32[%arg0, %arg2, %arg4] [%39, %40, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_259 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%41, %42, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %43 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_258 : tensor<?x?x8xf32>, tensor<?x?x8xf32>) outs(%extracted_slice_259 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %in_260: f32, %out: f32):
            %44 = arith.addf %in, %in_260 : f32
            %45 = arith.mulf %44, %44 : f32
            %46 = arith.addf %45, %45 : f32
            linalg.yield %46 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %43 into %arg5[%arg0, %arg2, %arg4] [%41, %42, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %36 : tensor<1x4x8xf32>
      }
      scf.yield %35 : tensor<1x4x8xf32>
    }
    return %34 : tensor<1x4x8xf32>
  }
}

// ==========================================
// Phase: Bufferization
// ==========================================
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
module {
  memref.global "private" constant @__constant_8x8xf32 : memref<8x8xf32> = dense<1.000000e-01> {alignment = 64 : i64}
  func.func @main() -> memref<1x4x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 2.828400e+00 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %cst_3 = arith.constant 2.000000e-01 : f32
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %0 = memref.get_global @__constant_8x8xf32 : memref<8x8xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloc : memref<1x4x8xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0 : memref<8x8xf32>) outs(%alloc : memref<1x4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.mulf %in, %cst_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloc_4 : memref<1x4x8xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0 : memref<8x8xf32>) outs(%alloc_4 : memref<1x4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.mulf %in, %cst_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloc_5 : memref<1x4x8xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0 : memref<8x8xf32>) outs(%alloc_5 : memref<1x4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.mulf %in, %cst_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x4x4xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloc_6 : memref<1x4x4xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    linalg.generic {indexing_maps = [#map3, #map4, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%alloc, %alloc_4 : memref<1x4x8xf32>, memref<1x4x8xf32>) outs(%alloc_6 : memref<1x4x4xf32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %1 = arith.mulf %in, %in_16 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x4x4xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_6 : memref<1x4x4xf32>) outs(%alloc_7 : memref<1x4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.divf %in, %cst_1 : f32
      linalg.yield %1 : f32
    }
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x4xf32>
    linalg.generic {indexing_maps = [#map5], iterator_types = ["parallel", "parallel"]} outs(%alloc_8 : memref<1x4xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_2 : f32
    }
    linalg.generic {indexing_maps = [#map, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%alloc_7 : memref<1x4x4xf32>) outs(%alloc_8 : memref<1x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.maxnumf %in, %out : f32
      linalg.yield %1 : f32
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x4x4xf32>
    linalg.generic {indexing_maps = [#map, #map6, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_7, %alloc_8 : memref<1x4x4xf32>, memref<1x4xf32>) outs(%alloc_9 : memref<1x4x4xf32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %1 = arith.subf %in, %in_16 : f32
      %2 = math.exp %1 : f32
      linalg.yield %2 : f32
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x4xf32>
    linalg.generic {indexing_maps = [#map5], iterator_types = ["parallel", "parallel"]} outs(%alloc_10 : memref<1x4xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    linalg.generic {indexing_maps = [#map, #map6], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_9 : memref<1x4x4xf32>) outs(%alloc_10 : memref<1x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %out : f32
      linalg.yield %1 : f32
    }
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloc_11 : memref<1x4x8xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    linalg.generic {indexing_maps = [#map3, #map7, #map8, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%alloc_9, %alloc_10, %alloc_5 : memref<1x4x4xf32>, memref<1x4xf32>, memref<1x4x8xf32>) outs(%alloc_11 : memref<1x4x8xf32>) {
    ^bb0(%in: f32, %in_16: f32, %in_17: f32, %out: f32):
      %1 = arith.divf %in, %in_16 : f32
      %2 = arith.mulf %1, %in_17 : f32
      %3 = arith.addf %out, %2 : f32
      linalg.yield %3 : f32
    }
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_11 : memref<1x4x8xf32>) outs(%alloc_12 : memref<1x4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %cst_0 : f32
      linalg.yield %1 : f32
    }
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x4x32xf32>
    scf.for %arg0 = %c0 to %c32 step %c8 {
      %subview = memref.subview %alloc_13[0, 0, %arg0] [1, 4, 8] [1, 1, 1] : memref<1x4x32xf32> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      }
      memref.copy %subview, %subview : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
    }
    scf.for %arg0 = %c0 to %c32 step %c8 {
      %subview = memref.subview %alloc_13[0, 0, %arg0] [1, 4, 8] [1, 1, 1] : memref<1x4x32xf32> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%alloc_12 : memref<1x4x8xf32>) outs(%subview : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        %1 = arith.mulf %in, %cst_3 : f32
        %2 = arith.addf %out, %1 : f32
        linalg.yield %2 : f32
      }
      memref.copy %subview, %subview : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
    }
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloc_14 : memref<1x4x8xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%alloc_13 : memref<1x4x32xf32>) outs(%alloc_14 : memref<1x4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.maximumf %in, %cst : f32
      %2 = arith.mulf %1, %cst_3 : f32
      %3 = arith.addf %out, %2 : f32
      linalg.yield %3 : f32
    }
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_12, %alloc_14 : memref<1x4x8xf32>, memref<1x4x8xf32>) outs(%alloc_15 : memref<1x4x8xf32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %1 = arith.addf %in, %in_16 : f32
      %2 = arith.mulf %1, %1 : f32
      %3 = arith.addf %2, %2 : f32
      linalg.yield %3 : f32
    }
    return %alloc_15 : memref<1x4x8xf32>
  }
}

// ==========================================
// Phase: linalg to scf
// ==========================================
module {
  memref.global "private" constant @__constant_8x8xf32 : memref<8x8xf32> = dense<1.000000e-01> {alignment = 64 : i64}
  func.func @main() -> memref<1x4x8xf32> {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 2.828400e+00 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %cst_3 = arith.constant 2.000000e-01 : f32
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %0 = memref.get_global @__constant_8x8xf32 : memref<8x8xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          memref.store %cst, %alloc[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
        }
      }
    }
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            %1 = memref.load %0[%arg3, %arg2] : memref<8x8xf32>
            %2 = memref.load %alloc[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
            %3 = arith.mulf %1, %cst_0 : f32
            %4 = arith.addf %2, %3 : f32
            memref.store %4, %alloc[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
          }
        }
      }
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          memref.store %cst, %alloc_4[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
        }
      }
    }
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            %1 = memref.load %0[%arg3, %arg2] : memref<8x8xf32>
            %2 = memref.load %alloc_4[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
            %3 = arith.mulf %1, %cst_0 : f32
            %4 = arith.addf %2, %3 : f32
            memref.store %4, %alloc_4[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
          }
        }
      }
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          memref.store %cst, %alloc_5[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
        }
      }
    }
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            %1 = memref.load %0[%arg3, %arg2] : memref<8x8xf32>
            %2 = memref.load %alloc_5[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
            %3 = arith.mulf %1, %cst_0 : f32
            %4 = arith.addf %2, %3 : f32
            memref.store %4, %alloc_5[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
          }
        }
      }
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x4x4xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c4 step %c1 {
          memref.store %cst, %alloc_6[%arg0, %arg1, %arg2] : memref<1x4x4xf32>
        }
      }
    }
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c4 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            %1 = memref.load %alloc[%arg0, %arg1, %arg3] : memref<1x4x8xf32>
            %2 = memref.load %alloc_4[%arg0, %arg2, %arg3] : memref<1x4x8xf32>
            %3 = memref.load %alloc_6[%arg0, %arg1, %arg2] : memref<1x4x4xf32>
            %4 = arith.mulf %1, %2 : f32
            %5 = arith.addf %3, %4 : f32
            memref.store %5, %alloc_6[%arg0, %arg1, %arg2] : memref<1x4x4xf32>
          }
        }
      }
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x4x4xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c4 step %c1 {
          %1 = memref.load %alloc_6[%arg0, %arg1, %arg2] : memref<1x4x4xf32>
          %2 = arith.divf %1, %cst_1 : f32
          memref.store %2, %alloc_7[%arg0, %arg1, %arg2] : memref<1x4x4xf32>
        }
      }
    }
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x4xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        memref.store %cst_2, %alloc_8[%arg0, %arg1] : memref<1x4xf32>
      }
    }
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c4 step %c1 {
          %1 = memref.load %alloc_7[%arg0, %arg1, %arg2] : memref<1x4x4xf32>
          %2 = memref.load %alloc_8[%arg0, %arg1] : memref<1x4xf32>
          %3 = arith.maxnumf %1, %2 : f32
          memref.store %3, %alloc_8[%arg0, %arg1] : memref<1x4xf32>
        }
      }
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x4x4xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c4 step %c1 {
          %1 = memref.load %alloc_7[%arg0, %arg1, %arg2] : memref<1x4x4xf32>
          %2 = memref.load %alloc_8[%arg0, %arg1] : memref<1x4xf32>
          %3 = arith.subf %1, %2 : f32
          %4 = math.exp %3 : f32
          memref.store %4, %alloc_9[%arg0, %arg1, %arg2] : memref<1x4x4xf32>
        }
      }
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x4xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        memref.store %cst, %alloc_10[%arg0, %arg1] : memref<1x4xf32>
      }
    }
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c4 step %c1 {
          %1 = memref.load %alloc_9[%arg0, %arg1, %arg2] : memref<1x4x4xf32>
          %2 = memref.load %alloc_10[%arg0, %arg1] : memref<1x4xf32>
          %3 = arith.addf %1, %2 : f32
          memref.store %3, %alloc_10[%arg0, %arg1] : memref<1x4xf32>
        }
      }
    }
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          memref.store %cst, %alloc_11[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
        }
      }
    }
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          scf.for %arg3 = %c0 to %c4 step %c1 {
            %1 = memref.load %alloc_9[%arg0, %arg1, %arg3] : memref<1x4x4xf32>
            %2 = memref.load %alloc_10[%arg0, %arg1] : memref<1x4xf32>
            %3 = memref.load %alloc_5[%arg0, %arg3, %arg2] : memref<1x4x8xf32>
            %4 = memref.load %alloc_11[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
            %5 = arith.divf %1, %2 : f32
            %6 = arith.mulf %5, %3 : f32
            %7 = arith.addf %4, %6 : f32
            memref.store %7, %alloc_11[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
          }
        }
      }
    }
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          %1 = memref.load %alloc_11[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
          %2 = arith.addf %1, %cst_0 : f32
          memref.store %2, %alloc_12[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
        }
      }
    }
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1x4x32xf32>
    scf.for %arg0 = %c0 to %c32 step %c8 {
      %subview = memref.subview %alloc_13[0, 0, %arg0] [1, 4, 8] [1, 1, 1] : memref<1x4x32xf32> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c4 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            memref.store %cst, %subview[%arg1, %arg2, %arg3] : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
          }
        }
      }
      memref.copy %subview, %subview : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
    }
    scf.for %arg0 = %c0 to %c32 step %c8 {
      %subview = memref.subview %alloc_13[0, 0, %arg0] [1, 4, 8] [1, 1, 1] : memref<1x4x32xf32> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c4 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            scf.for %arg4 = %c0 to %c8 step %c1 {
              %1 = memref.load %alloc_12[%arg1, %arg2, %arg4] : memref<1x4x8xf32>
              %2 = memref.load %subview[%arg1, %arg2, %arg3] : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
              %3 = arith.mulf %1, %cst_3 : f32
              %4 = arith.addf %2, %3 : f32
              memref.store %4, %subview[%arg1, %arg2, %arg3] : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
            }
          }
        }
      }
      memref.copy %subview, %subview : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
    }
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          memref.store %cst, %alloc_14[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
        }
      }
    }
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          scf.for %arg3 = %c0 to %c32 step %c1 {
            %1 = memref.load %alloc_13[%arg0, %arg1, %arg3] : memref<1x4x32xf32>
            %2 = memref.load %alloc_14[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
            %3 = arith.maximumf %1, %cst : f32
            %4 = arith.mulf %3, %cst_3 : f32
            %5 = arith.addf %2, %4 : f32
            memref.store %5, %alloc_14[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
          }
        }
      }
    }
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          %1 = memref.load %alloc_12[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
          %2 = memref.load %alloc_14[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
          %3 = arith.addf %1, %2 : f32
          %4 = arith.mulf %3, %3 : f32
          %5 = arith.addf %4, %4 : f32
          memref.store %5, %alloc_15[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
        }
      }
    }
    return %alloc_15 : memref<1x4x8xf32>
  }
}

// ==========================================
// Phase: lower to llvm
// ==========================================
module {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_8x8xf32(dense<1.000000e-01> : tensor<8x8xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<8 x array<8 x f32>>
  llvm.func @main() -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> {
    %0 = llvm.mlir.constant(4 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %3 = llvm.mlir.constant(5.000000e-01 : f32) : f32
    %4 = llvm.mlir.constant(2.828400e+00 : f32) : f32
    %5 = llvm.mlir.constant(0xFF800000 : f32) : f32
    %6 = llvm.mlir.constant(2.000000e-01 : f32) : f32
    %7 = llvm.mlir.constant(8 : index) : i64
    %8 = llvm.mlir.constant(0 : index) : i64
    %9 = llvm.mlir.constant(32 : index) : i64
    %10 = llvm.mlir.constant(8 : index) : i64
    %11 = llvm.mlir.constant(8 : index) : i64
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.mlir.constant(64 : index) : i64
    %14 = llvm.mlir.zero : !llvm.ptr
    %15 = llvm.getelementptr %14[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %16 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %17 = llvm.mlir.addressof @__constant_8x8xf32 : !llvm.ptr
    %18 = llvm.getelementptr %17[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x array<8 x f32>>
    %19 = llvm.mlir.constant(3735928559 : index) : i64
    %20 = llvm.inttoptr %19 : i64 to !llvm.ptr
    %21 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %22 = llvm.insertvalue %20, %21[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %18, %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(0 : index) : i64
    %25 = llvm.insertvalue %24, %23[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.insertvalue %10, %25[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %11, %26[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %11, %27[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.insertvalue %12, %28[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.mlir.constant(1 : index) : i64
    %31 = llvm.mlir.constant(4 : index) : i64
    %32 = llvm.mlir.constant(8 : index) : i64
    %33 = llvm.mlir.constant(1 : index) : i64
    %34 = llvm.mlir.constant(32 : index) : i64
    %35 = llvm.mlir.constant(32 : index) : i64
    %36 = llvm.mlir.zero : !llvm.ptr
    %37 = llvm.getelementptr %36[%35] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %38 = llvm.ptrtoint %37 : !llvm.ptr to i64
    %39 = llvm.mlir.constant(64 : index) : i64
    %40 = llvm.add %38, %39 : i64
    %41 = llvm.call @malloc(%40) : (i64) -> !llvm.ptr
    %42 = llvm.ptrtoint %41 : !llvm.ptr to i64
    %43 = llvm.mlir.constant(1 : index) : i64
    %44 = llvm.sub %39, %43 : i64
    %45 = llvm.add %42, %44 : i64
    %46 = llvm.urem %45, %39  : i64
    %47 = llvm.sub %45, %46 : i64
    %48 = llvm.inttoptr %47 : i64 to !llvm.ptr
    %49 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %50 = llvm.insertvalue %41, %49[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %51 = llvm.insertvalue %48, %50[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %52 = llvm.mlir.constant(0 : index) : i64
    %53 = llvm.insertvalue %52, %51[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %54 = llvm.insertvalue %30, %53[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %55 = llvm.insertvalue %31, %54[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %56 = llvm.insertvalue %32, %55[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %57 = llvm.insertvalue %34, %56[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %58 = llvm.insertvalue %32, %57[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %59 = llvm.insertvalue %33, %58[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb1(%8 : i64)
  ^bb1(%60: i64):  // 2 preds: ^bb0, ^bb8
    %61 = llvm.icmp "slt" %60, %1 : i64
    llvm.cond_br %61, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%8 : i64)
  ^bb3(%62: i64):  // 2 preds: ^bb2, ^bb7
    %63 = llvm.icmp "slt" %62, %0 : i64
    llvm.cond_br %63, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%8 : i64)
  ^bb5(%64: i64):  // 2 preds: ^bb4, ^bb6
    %65 = llvm.icmp "slt" %64, %7 : i64
    llvm.cond_br %65, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %66 = llvm.extractvalue %59[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %67 = llvm.mlir.constant(32 : index) : i64
    %68 = llvm.mul %60, %67 : i64
    %69 = llvm.mlir.constant(8 : index) : i64
    %70 = llvm.mul %62, %69 : i64
    %71 = llvm.add %68, %70 : i64
    %72 = llvm.add %71, %64 : i64
    %73 = llvm.getelementptr %66[%72] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %73 : f32, !llvm.ptr
    %74 = llvm.add %64, %1 : i64
    llvm.br ^bb5(%74 : i64)
  ^bb7:  // pred: ^bb5
    %75 = llvm.add %62, %1 : i64
    llvm.br ^bb3(%75 : i64)
  ^bb8:  // pred: ^bb3
    %76 = llvm.add %60, %1 : i64
    llvm.br ^bb1(%76 : i64)
  ^bb9:  // pred: ^bb1
    llvm.br ^bb10(%8 : i64)
  ^bb10(%77: i64):  // 2 preds: ^bb9, ^bb20
    %78 = llvm.icmp "slt" %77, %1 : i64
    llvm.cond_br %78, ^bb11, ^bb21
  ^bb11:  // pred: ^bb10
    llvm.br ^bb12(%8 : i64)
  ^bb12(%79: i64):  // 2 preds: ^bb11, ^bb19
    %80 = llvm.icmp "slt" %79, %0 : i64
    llvm.cond_br %80, ^bb13, ^bb20
  ^bb13:  // pred: ^bb12
    llvm.br ^bb14(%8 : i64)
  ^bb14(%81: i64):  // 2 preds: ^bb13, ^bb18
    %82 = llvm.icmp "slt" %81, %7 : i64
    llvm.cond_br %82, ^bb15, ^bb19
  ^bb15:  // pred: ^bb14
    llvm.br ^bb16(%8 : i64)
  ^bb16(%83: i64):  // 2 preds: ^bb15, ^bb17
    %84 = llvm.icmp "slt" %83, %7 : i64
    llvm.cond_br %84, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %85 = llvm.extractvalue %29[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %86 = llvm.mlir.constant(8 : index) : i64
    %87 = llvm.mul %83, %86 : i64
    %88 = llvm.add %87, %81 : i64
    %89 = llvm.getelementptr %85[%88] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %90 = llvm.load %89 : !llvm.ptr -> f32
    %91 = llvm.extractvalue %59[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %92 = llvm.mlir.constant(32 : index) : i64
    %93 = llvm.mul %77, %92 : i64
    %94 = llvm.mlir.constant(8 : index) : i64
    %95 = llvm.mul %79, %94 : i64
    %96 = llvm.add %93, %95 : i64
    %97 = llvm.add %96, %81 : i64
    %98 = llvm.getelementptr %91[%97] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %99 = llvm.load %98 : !llvm.ptr -> f32
    %100 = llvm.fmul %90, %3  : f32
    %101 = llvm.fadd %99, %100  : f32
    %102 = llvm.extractvalue %59[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %103 = llvm.mlir.constant(32 : index) : i64
    %104 = llvm.mul %77, %103 : i64
    %105 = llvm.mlir.constant(8 : index) : i64
    %106 = llvm.mul %79, %105 : i64
    %107 = llvm.add %104, %106 : i64
    %108 = llvm.add %107, %81 : i64
    %109 = llvm.getelementptr %102[%108] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %101, %109 : f32, !llvm.ptr
    %110 = llvm.add %83, %1 : i64
    llvm.br ^bb16(%110 : i64)
  ^bb18:  // pred: ^bb16
    %111 = llvm.add %81, %1 : i64
    llvm.br ^bb14(%111 : i64)
  ^bb19:  // pred: ^bb14
    %112 = llvm.add %79, %1 : i64
    llvm.br ^bb12(%112 : i64)
  ^bb20:  // pred: ^bb12
    %113 = llvm.add %77, %1 : i64
    llvm.br ^bb10(%113 : i64)
  ^bb21:  // pred: ^bb10
    %114 = llvm.mlir.constant(1 : index) : i64
    %115 = llvm.mlir.constant(4 : index) : i64
    %116 = llvm.mlir.constant(8 : index) : i64
    %117 = llvm.mlir.constant(1 : index) : i64
    %118 = llvm.mlir.constant(32 : index) : i64
    %119 = llvm.mlir.constant(32 : index) : i64
    %120 = llvm.mlir.zero : !llvm.ptr
    %121 = llvm.getelementptr %120[%119] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %122 = llvm.ptrtoint %121 : !llvm.ptr to i64
    %123 = llvm.mlir.constant(64 : index) : i64
    %124 = llvm.add %122, %123 : i64
    %125 = llvm.call @malloc(%124) : (i64) -> !llvm.ptr
    %126 = llvm.ptrtoint %125 : !llvm.ptr to i64
    %127 = llvm.mlir.constant(1 : index) : i64
    %128 = llvm.sub %123, %127 : i64
    %129 = llvm.add %126, %128 : i64
    %130 = llvm.urem %129, %123  : i64
    %131 = llvm.sub %129, %130 : i64
    %132 = llvm.inttoptr %131 : i64 to !llvm.ptr
    %133 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %134 = llvm.insertvalue %125, %133[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %135 = llvm.insertvalue %132, %134[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %136 = llvm.mlir.constant(0 : index) : i64
    %137 = llvm.insertvalue %136, %135[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %138 = llvm.insertvalue %114, %137[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %139 = llvm.insertvalue %115, %138[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %140 = llvm.insertvalue %116, %139[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %141 = llvm.insertvalue %118, %140[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %142 = llvm.insertvalue %116, %141[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %143 = llvm.insertvalue %117, %142[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb22(%8 : i64)
  ^bb22(%144: i64):  // 2 preds: ^bb21, ^bb29
    %145 = llvm.icmp "slt" %144, %1 : i64
    llvm.cond_br %145, ^bb23, ^bb30
  ^bb23:  // pred: ^bb22
    llvm.br ^bb24(%8 : i64)
  ^bb24(%146: i64):  // 2 preds: ^bb23, ^bb28
    %147 = llvm.icmp "slt" %146, %0 : i64
    llvm.cond_br %147, ^bb25, ^bb29
  ^bb25:  // pred: ^bb24
    llvm.br ^bb26(%8 : i64)
  ^bb26(%148: i64):  // 2 preds: ^bb25, ^bb27
    %149 = llvm.icmp "slt" %148, %7 : i64
    llvm.cond_br %149, ^bb27, ^bb28
  ^bb27:  // pred: ^bb26
    %150 = llvm.extractvalue %143[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %151 = llvm.mlir.constant(32 : index) : i64
    %152 = llvm.mul %144, %151 : i64
    %153 = llvm.mlir.constant(8 : index) : i64
    %154 = llvm.mul %146, %153 : i64
    %155 = llvm.add %152, %154 : i64
    %156 = llvm.add %155, %148 : i64
    %157 = llvm.getelementptr %150[%156] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %157 : f32, !llvm.ptr
    %158 = llvm.add %148, %1 : i64
    llvm.br ^bb26(%158 : i64)
  ^bb28:  // pred: ^bb26
    %159 = llvm.add %146, %1 : i64
    llvm.br ^bb24(%159 : i64)
  ^bb29:  // pred: ^bb24
    %160 = llvm.add %144, %1 : i64
    llvm.br ^bb22(%160 : i64)
  ^bb30:  // pred: ^bb22
    llvm.br ^bb31(%8 : i64)
  ^bb31(%161: i64):  // 2 preds: ^bb30, ^bb41
    %162 = llvm.icmp "slt" %161, %1 : i64
    llvm.cond_br %162, ^bb32, ^bb42
  ^bb32:  // pred: ^bb31
    llvm.br ^bb33(%8 : i64)
  ^bb33(%163: i64):  // 2 preds: ^bb32, ^bb40
    %164 = llvm.icmp "slt" %163, %0 : i64
    llvm.cond_br %164, ^bb34, ^bb41
  ^bb34:  // pred: ^bb33
    llvm.br ^bb35(%8 : i64)
  ^bb35(%165: i64):  // 2 preds: ^bb34, ^bb39
    %166 = llvm.icmp "slt" %165, %7 : i64
    llvm.cond_br %166, ^bb36, ^bb40
  ^bb36:  // pred: ^bb35
    llvm.br ^bb37(%8 : i64)
  ^bb37(%167: i64):  // 2 preds: ^bb36, ^bb38
    %168 = llvm.icmp "slt" %167, %7 : i64
    llvm.cond_br %168, ^bb38, ^bb39
  ^bb38:  // pred: ^bb37
    %169 = llvm.extractvalue %29[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %170 = llvm.mlir.constant(8 : index) : i64
    %171 = llvm.mul %167, %170 : i64
    %172 = llvm.add %171, %165 : i64
    %173 = llvm.getelementptr %169[%172] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %174 = llvm.load %173 : !llvm.ptr -> f32
    %175 = llvm.extractvalue %143[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %176 = llvm.mlir.constant(32 : index) : i64
    %177 = llvm.mul %161, %176 : i64
    %178 = llvm.mlir.constant(8 : index) : i64
    %179 = llvm.mul %163, %178 : i64
    %180 = llvm.add %177, %179 : i64
    %181 = llvm.add %180, %165 : i64
    %182 = llvm.getelementptr %175[%181] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %183 = llvm.load %182 : !llvm.ptr -> f32
    %184 = llvm.fmul %174, %3  : f32
    %185 = llvm.fadd %183, %184  : f32
    %186 = llvm.extractvalue %143[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %187 = llvm.mlir.constant(32 : index) : i64
    %188 = llvm.mul %161, %187 : i64
    %189 = llvm.mlir.constant(8 : index) : i64
    %190 = llvm.mul %163, %189 : i64
    %191 = llvm.add %188, %190 : i64
    %192 = llvm.add %191, %165 : i64
    %193 = llvm.getelementptr %186[%192] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %185, %193 : f32, !llvm.ptr
    %194 = llvm.add %167, %1 : i64
    llvm.br ^bb37(%194 : i64)
  ^bb39:  // pred: ^bb37
    %195 = llvm.add %165, %1 : i64
    llvm.br ^bb35(%195 : i64)
  ^bb40:  // pred: ^bb35
    %196 = llvm.add %163, %1 : i64
    llvm.br ^bb33(%196 : i64)
  ^bb41:  // pred: ^bb33
    %197 = llvm.add %161, %1 : i64
    llvm.br ^bb31(%197 : i64)
  ^bb42:  // pred: ^bb31
    %198 = llvm.mlir.constant(1 : index) : i64
    %199 = llvm.mlir.constant(4 : index) : i64
    %200 = llvm.mlir.constant(8 : index) : i64
    %201 = llvm.mlir.constant(1 : index) : i64
    %202 = llvm.mlir.constant(32 : index) : i64
    %203 = llvm.mlir.constant(32 : index) : i64
    %204 = llvm.mlir.zero : !llvm.ptr
    %205 = llvm.getelementptr %204[%203] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %206 = llvm.ptrtoint %205 : !llvm.ptr to i64
    %207 = llvm.mlir.constant(64 : index) : i64
    %208 = llvm.add %206, %207 : i64
    %209 = llvm.call @malloc(%208) : (i64) -> !llvm.ptr
    %210 = llvm.ptrtoint %209 : !llvm.ptr to i64
    %211 = llvm.mlir.constant(1 : index) : i64
    %212 = llvm.sub %207, %211 : i64
    %213 = llvm.add %210, %212 : i64
    %214 = llvm.urem %213, %207  : i64
    %215 = llvm.sub %213, %214 : i64
    %216 = llvm.inttoptr %215 : i64 to !llvm.ptr
    %217 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %218 = llvm.insertvalue %209, %217[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %219 = llvm.insertvalue %216, %218[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %220 = llvm.mlir.constant(0 : index) : i64
    %221 = llvm.insertvalue %220, %219[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %222 = llvm.insertvalue %198, %221[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %223 = llvm.insertvalue %199, %222[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %224 = llvm.insertvalue %200, %223[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %225 = llvm.insertvalue %202, %224[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %226 = llvm.insertvalue %200, %225[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %227 = llvm.insertvalue %201, %226[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb43(%8 : i64)
  ^bb43(%228: i64):  // 2 preds: ^bb42, ^bb50
    %229 = llvm.icmp "slt" %228, %1 : i64
    llvm.cond_br %229, ^bb44, ^bb51
  ^bb44:  // pred: ^bb43
    llvm.br ^bb45(%8 : i64)
  ^bb45(%230: i64):  // 2 preds: ^bb44, ^bb49
    %231 = llvm.icmp "slt" %230, %0 : i64
    llvm.cond_br %231, ^bb46, ^bb50
  ^bb46:  // pred: ^bb45
    llvm.br ^bb47(%8 : i64)
  ^bb47(%232: i64):  // 2 preds: ^bb46, ^bb48
    %233 = llvm.icmp "slt" %232, %7 : i64
    llvm.cond_br %233, ^bb48, ^bb49
  ^bb48:  // pred: ^bb47
    %234 = llvm.extractvalue %227[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %235 = llvm.mlir.constant(32 : index) : i64
    %236 = llvm.mul %228, %235 : i64
    %237 = llvm.mlir.constant(8 : index) : i64
    %238 = llvm.mul %230, %237 : i64
    %239 = llvm.add %236, %238 : i64
    %240 = llvm.add %239, %232 : i64
    %241 = llvm.getelementptr %234[%240] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %241 : f32, !llvm.ptr
    %242 = llvm.add %232, %1 : i64
    llvm.br ^bb47(%242 : i64)
  ^bb49:  // pred: ^bb47
    %243 = llvm.add %230, %1 : i64
    llvm.br ^bb45(%243 : i64)
  ^bb50:  // pred: ^bb45
    %244 = llvm.add %228, %1 : i64
    llvm.br ^bb43(%244 : i64)
  ^bb51:  // pred: ^bb43
    llvm.br ^bb52(%8 : i64)
  ^bb52(%245: i64):  // 2 preds: ^bb51, ^bb62
    %246 = llvm.icmp "slt" %245, %1 : i64
    llvm.cond_br %246, ^bb53, ^bb63
  ^bb53:  // pred: ^bb52
    llvm.br ^bb54(%8 : i64)
  ^bb54(%247: i64):  // 2 preds: ^bb53, ^bb61
    %248 = llvm.icmp "slt" %247, %0 : i64
    llvm.cond_br %248, ^bb55, ^bb62
  ^bb55:  // pred: ^bb54
    llvm.br ^bb56(%8 : i64)
  ^bb56(%249: i64):  // 2 preds: ^bb55, ^bb60
    %250 = llvm.icmp "slt" %249, %7 : i64
    llvm.cond_br %250, ^bb57, ^bb61
  ^bb57:  // pred: ^bb56
    llvm.br ^bb58(%8 : i64)
  ^bb58(%251: i64):  // 2 preds: ^bb57, ^bb59
    %252 = llvm.icmp "slt" %251, %7 : i64
    llvm.cond_br %252, ^bb59, ^bb60
  ^bb59:  // pred: ^bb58
    %253 = llvm.extractvalue %29[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %254 = llvm.mlir.constant(8 : index) : i64
    %255 = llvm.mul %251, %254 : i64
    %256 = llvm.add %255, %249 : i64
    %257 = llvm.getelementptr %253[%256] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %258 = llvm.load %257 : !llvm.ptr -> f32
    %259 = llvm.extractvalue %227[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %260 = llvm.mlir.constant(32 : index) : i64
    %261 = llvm.mul %245, %260 : i64
    %262 = llvm.mlir.constant(8 : index) : i64
    %263 = llvm.mul %247, %262 : i64
    %264 = llvm.add %261, %263 : i64
    %265 = llvm.add %264, %249 : i64
    %266 = llvm.getelementptr %259[%265] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %267 = llvm.load %266 : !llvm.ptr -> f32
    %268 = llvm.fmul %258, %3  : f32
    %269 = llvm.fadd %267, %268  : f32
    %270 = llvm.extractvalue %227[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %271 = llvm.mlir.constant(32 : index) : i64
    %272 = llvm.mul %245, %271 : i64
    %273 = llvm.mlir.constant(8 : index) : i64
    %274 = llvm.mul %247, %273 : i64
    %275 = llvm.add %272, %274 : i64
    %276 = llvm.add %275, %249 : i64
    %277 = llvm.getelementptr %270[%276] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %269, %277 : f32, !llvm.ptr
    %278 = llvm.add %251, %1 : i64
    llvm.br ^bb58(%278 : i64)
  ^bb60:  // pred: ^bb58
    %279 = llvm.add %249, %1 : i64
    llvm.br ^bb56(%279 : i64)
  ^bb61:  // pred: ^bb56
    %280 = llvm.add %247, %1 : i64
    llvm.br ^bb54(%280 : i64)
  ^bb62:  // pred: ^bb54
    %281 = llvm.add %245, %1 : i64
    llvm.br ^bb52(%281 : i64)
  ^bb63:  // pred: ^bb52
    %282 = llvm.mlir.constant(1 : index) : i64
    %283 = llvm.mlir.constant(4 : index) : i64
    %284 = llvm.mlir.constant(4 : index) : i64
    %285 = llvm.mlir.constant(1 : index) : i64
    %286 = llvm.mlir.constant(16 : index) : i64
    %287 = llvm.mlir.constant(16 : index) : i64
    %288 = llvm.mlir.zero : !llvm.ptr
    %289 = llvm.getelementptr %288[%287] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %290 = llvm.ptrtoint %289 : !llvm.ptr to i64
    %291 = llvm.mlir.constant(64 : index) : i64
    %292 = llvm.add %290, %291 : i64
    %293 = llvm.call @malloc(%292) : (i64) -> !llvm.ptr
    %294 = llvm.ptrtoint %293 : !llvm.ptr to i64
    %295 = llvm.mlir.constant(1 : index) : i64
    %296 = llvm.sub %291, %295 : i64
    %297 = llvm.add %294, %296 : i64
    %298 = llvm.urem %297, %291  : i64
    %299 = llvm.sub %297, %298 : i64
    %300 = llvm.inttoptr %299 : i64 to !llvm.ptr
    %301 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %302 = llvm.insertvalue %293, %301[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %303 = llvm.insertvalue %300, %302[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %304 = llvm.mlir.constant(0 : index) : i64
    %305 = llvm.insertvalue %304, %303[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %306 = llvm.insertvalue %282, %305[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %307 = llvm.insertvalue %283, %306[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %308 = llvm.insertvalue %284, %307[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %309 = llvm.insertvalue %286, %308[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %310 = llvm.insertvalue %284, %309[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %311 = llvm.insertvalue %285, %310[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb64(%8 : i64)
  ^bb64(%312: i64):  // 2 preds: ^bb63, ^bb71
    %313 = llvm.icmp "slt" %312, %1 : i64
    llvm.cond_br %313, ^bb65, ^bb72
  ^bb65:  // pred: ^bb64
    llvm.br ^bb66(%8 : i64)
  ^bb66(%314: i64):  // 2 preds: ^bb65, ^bb70
    %315 = llvm.icmp "slt" %314, %0 : i64
    llvm.cond_br %315, ^bb67, ^bb71
  ^bb67:  // pred: ^bb66
    llvm.br ^bb68(%8 : i64)
  ^bb68(%316: i64):  // 2 preds: ^bb67, ^bb69
    %317 = llvm.icmp "slt" %316, %0 : i64
    llvm.cond_br %317, ^bb69, ^bb70
  ^bb69:  // pred: ^bb68
    %318 = llvm.extractvalue %311[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %319 = llvm.mlir.constant(16 : index) : i64
    %320 = llvm.mul %312, %319 : i64
    %321 = llvm.mlir.constant(4 : index) : i64
    %322 = llvm.mul %314, %321 : i64
    %323 = llvm.add %320, %322 : i64
    %324 = llvm.add %323, %316 : i64
    %325 = llvm.getelementptr %318[%324] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %325 : f32, !llvm.ptr
    %326 = llvm.add %316, %1 : i64
    llvm.br ^bb68(%326 : i64)
  ^bb70:  // pred: ^bb68
    %327 = llvm.add %314, %1 : i64
    llvm.br ^bb66(%327 : i64)
  ^bb71:  // pred: ^bb66
    %328 = llvm.add %312, %1 : i64
    llvm.br ^bb64(%328 : i64)
  ^bb72:  // pred: ^bb64
    llvm.br ^bb73(%8 : i64)
  ^bb73(%329: i64):  // 2 preds: ^bb72, ^bb83
    %330 = llvm.icmp "slt" %329, %1 : i64
    llvm.cond_br %330, ^bb74, ^bb84
  ^bb74:  // pred: ^bb73
    llvm.br ^bb75(%8 : i64)
  ^bb75(%331: i64):  // 2 preds: ^bb74, ^bb82
    %332 = llvm.icmp "slt" %331, %0 : i64
    llvm.cond_br %332, ^bb76, ^bb83
  ^bb76:  // pred: ^bb75
    llvm.br ^bb77(%8 : i64)
  ^bb77(%333: i64):  // 2 preds: ^bb76, ^bb81
    %334 = llvm.icmp "slt" %333, %0 : i64
    llvm.cond_br %334, ^bb78, ^bb82
  ^bb78:  // pred: ^bb77
    llvm.br ^bb79(%8 : i64)
  ^bb79(%335: i64):  // 2 preds: ^bb78, ^bb80
    %336 = llvm.icmp "slt" %335, %7 : i64
    llvm.cond_br %336, ^bb80, ^bb81
  ^bb80:  // pred: ^bb79
    %337 = llvm.extractvalue %59[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %338 = llvm.mlir.constant(32 : index) : i64
    %339 = llvm.mul %329, %338 : i64
    %340 = llvm.mlir.constant(8 : index) : i64
    %341 = llvm.mul %331, %340 : i64
    %342 = llvm.add %339, %341 : i64
    %343 = llvm.add %342, %335 : i64
    %344 = llvm.getelementptr %337[%343] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %345 = llvm.load %344 : !llvm.ptr -> f32
    %346 = llvm.extractvalue %143[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %347 = llvm.mlir.constant(32 : index) : i64
    %348 = llvm.mul %329, %347 : i64
    %349 = llvm.mlir.constant(8 : index) : i64
    %350 = llvm.mul %333, %349 : i64
    %351 = llvm.add %348, %350 : i64
    %352 = llvm.add %351, %335 : i64
    %353 = llvm.getelementptr %346[%352] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %354 = llvm.load %353 : !llvm.ptr -> f32
    %355 = llvm.extractvalue %311[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %356 = llvm.mlir.constant(16 : index) : i64
    %357 = llvm.mul %329, %356 : i64
    %358 = llvm.mlir.constant(4 : index) : i64
    %359 = llvm.mul %331, %358 : i64
    %360 = llvm.add %357, %359 : i64
    %361 = llvm.add %360, %333 : i64
    %362 = llvm.getelementptr %355[%361] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %363 = llvm.load %362 : !llvm.ptr -> f32
    %364 = llvm.fmul %345, %354  : f32
    %365 = llvm.fadd %363, %364  : f32
    %366 = llvm.extractvalue %311[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %367 = llvm.mlir.constant(16 : index) : i64
    %368 = llvm.mul %329, %367 : i64
    %369 = llvm.mlir.constant(4 : index) : i64
    %370 = llvm.mul %331, %369 : i64
    %371 = llvm.add %368, %370 : i64
    %372 = llvm.add %371, %333 : i64
    %373 = llvm.getelementptr %366[%372] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %365, %373 : f32, !llvm.ptr
    %374 = llvm.add %335, %1 : i64
    llvm.br ^bb79(%374 : i64)
  ^bb81:  // pred: ^bb79
    %375 = llvm.add %333, %1 : i64
    llvm.br ^bb77(%375 : i64)
  ^bb82:  // pred: ^bb77
    %376 = llvm.add %331, %1 : i64
    llvm.br ^bb75(%376 : i64)
  ^bb83:  // pred: ^bb75
    %377 = llvm.add %329, %1 : i64
    llvm.br ^bb73(%377 : i64)
  ^bb84:  // pred: ^bb73
    %378 = llvm.mlir.constant(1 : index) : i64
    %379 = llvm.mlir.constant(4 : index) : i64
    %380 = llvm.mlir.constant(4 : index) : i64
    %381 = llvm.mlir.constant(1 : index) : i64
    %382 = llvm.mlir.constant(16 : index) : i64
    %383 = llvm.mlir.constant(16 : index) : i64
    %384 = llvm.mlir.zero : !llvm.ptr
    %385 = llvm.getelementptr %384[%383] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %386 = llvm.ptrtoint %385 : !llvm.ptr to i64
    %387 = llvm.mlir.constant(64 : index) : i64
    %388 = llvm.add %386, %387 : i64
    %389 = llvm.call @malloc(%388) : (i64) -> !llvm.ptr
    %390 = llvm.ptrtoint %389 : !llvm.ptr to i64
    %391 = llvm.mlir.constant(1 : index) : i64
    %392 = llvm.sub %387, %391 : i64
    %393 = llvm.add %390, %392 : i64
    %394 = llvm.urem %393, %387  : i64
    %395 = llvm.sub %393, %394 : i64
    %396 = llvm.inttoptr %395 : i64 to !llvm.ptr
    %397 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %398 = llvm.insertvalue %389, %397[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %399 = llvm.insertvalue %396, %398[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %400 = llvm.mlir.constant(0 : index) : i64
    %401 = llvm.insertvalue %400, %399[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %402 = llvm.insertvalue %378, %401[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %403 = llvm.insertvalue %379, %402[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %404 = llvm.insertvalue %380, %403[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %405 = llvm.insertvalue %382, %404[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %406 = llvm.insertvalue %380, %405[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %407 = llvm.insertvalue %381, %406[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb85(%8 : i64)
  ^bb85(%408: i64):  // 2 preds: ^bb84, ^bb92
    %409 = llvm.icmp "slt" %408, %1 : i64
    llvm.cond_br %409, ^bb86, ^bb93
  ^bb86:  // pred: ^bb85
    llvm.br ^bb87(%8 : i64)
  ^bb87(%410: i64):  // 2 preds: ^bb86, ^bb91
    %411 = llvm.icmp "slt" %410, %0 : i64
    llvm.cond_br %411, ^bb88, ^bb92
  ^bb88:  // pred: ^bb87
    llvm.br ^bb89(%8 : i64)
  ^bb89(%412: i64):  // 2 preds: ^bb88, ^bb90
    %413 = llvm.icmp "slt" %412, %0 : i64
    llvm.cond_br %413, ^bb90, ^bb91
  ^bb90:  // pred: ^bb89
    %414 = llvm.extractvalue %311[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %415 = llvm.mlir.constant(16 : index) : i64
    %416 = llvm.mul %408, %415 : i64
    %417 = llvm.mlir.constant(4 : index) : i64
    %418 = llvm.mul %410, %417 : i64
    %419 = llvm.add %416, %418 : i64
    %420 = llvm.add %419, %412 : i64
    %421 = llvm.getelementptr %414[%420] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %422 = llvm.load %421 : !llvm.ptr -> f32
    %423 = llvm.fdiv %422, %4  : f32
    %424 = llvm.extractvalue %407[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %425 = llvm.mlir.constant(16 : index) : i64
    %426 = llvm.mul %408, %425 : i64
    %427 = llvm.mlir.constant(4 : index) : i64
    %428 = llvm.mul %410, %427 : i64
    %429 = llvm.add %426, %428 : i64
    %430 = llvm.add %429, %412 : i64
    %431 = llvm.getelementptr %424[%430] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %423, %431 : f32, !llvm.ptr
    %432 = llvm.add %412, %1 : i64
    llvm.br ^bb89(%432 : i64)
  ^bb91:  // pred: ^bb89
    %433 = llvm.add %410, %1 : i64
    llvm.br ^bb87(%433 : i64)
  ^bb92:  // pred: ^bb87
    %434 = llvm.add %408, %1 : i64
    llvm.br ^bb85(%434 : i64)
  ^bb93:  // pred: ^bb85
    %435 = llvm.mlir.constant(1 : index) : i64
    %436 = llvm.mlir.constant(4 : index) : i64
    %437 = llvm.mlir.constant(1 : index) : i64
    %438 = llvm.mlir.constant(4 : index) : i64
    %439 = llvm.mlir.zero : !llvm.ptr
    %440 = llvm.getelementptr %439[%438] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %441 = llvm.ptrtoint %440 : !llvm.ptr to i64
    %442 = llvm.mlir.constant(64 : index) : i64
    %443 = llvm.add %441, %442 : i64
    %444 = llvm.call @malloc(%443) : (i64) -> !llvm.ptr
    %445 = llvm.ptrtoint %444 : !llvm.ptr to i64
    %446 = llvm.mlir.constant(1 : index) : i64
    %447 = llvm.sub %442, %446 : i64
    %448 = llvm.add %445, %447 : i64
    %449 = llvm.urem %448, %442  : i64
    %450 = llvm.sub %448, %449 : i64
    %451 = llvm.inttoptr %450 : i64 to !llvm.ptr
    %452 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %453 = llvm.insertvalue %444, %452[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %454 = llvm.insertvalue %451, %453[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %455 = llvm.mlir.constant(0 : index) : i64
    %456 = llvm.insertvalue %455, %454[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %457 = llvm.insertvalue %435, %456[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %458 = llvm.insertvalue %436, %457[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %459 = llvm.insertvalue %436, %458[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %460 = llvm.insertvalue %437, %459[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb94(%8 : i64)
  ^bb94(%461: i64):  // 2 preds: ^bb93, ^bb98
    %462 = llvm.icmp "slt" %461, %1 : i64
    llvm.cond_br %462, ^bb95, ^bb99
  ^bb95:  // pred: ^bb94
    llvm.br ^bb96(%8 : i64)
  ^bb96(%463: i64):  // 2 preds: ^bb95, ^bb97
    %464 = llvm.icmp "slt" %463, %0 : i64
    llvm.cond_br %464, ^bb97, ^bb98
  ^bb97:  // pred: ^bb96
    %465 = llvm.extractvalue %460[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %466 = llvm.mlir.constant(4 : index) : i64
    %467 = llvm.mul %461, %466 : i64
    %468 = llvm.add %467, %463 : i64
    %469 = llvm.getelementptr %465[%468] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %5, %469 : f32, !llvm.ptr
    %470 = llvm.add %463, %1 : i64
    llvm.br ^bb96(%470 : i64)
  ^bb98:  // pred: ^bb96
    %471 = llvm.add %461, %1 : i64
    llvm.br ^bb94(%471 : i64)
  ^bb99:  // pred: ^bb94
    llvm.br ^bb100(%8 : i64)
  ^bb100(%472: i64):  // 2 preds: ^bb99, ^bb107
    %473 = llvm.icmp "slt" %472, %1 : i64
    llvm.cond_br %473, ^bb101, ^bb108
  ^bb101:  // pred: ^bb100
    llvm.br ^bb102(%8 : i64)
  ^bb102(%474: i64):  // 2 preds: ^bb101, ^bb106
    %475 = llvm.icmp "slt" %474, %0 : i64
    llvm.cond_br %475, ^bb103, ^bb107
  ^bb103:  // pred: ^bb102
    llvm.br ^bb104(%8 : i64)
  ^bb104(%476: i64):  // 2 preds: ^bb103, ^bb105
    %477 = llvm.icmp "slt" %476, %0 : i64
    llvm.cond_br %477, ^bb105, ^bb106
  ^bb105:  // pred: ^bb104
    %478 = llvm.extractvalue %407[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %479 = llvm.mlir.constant(16 : index) : i64
    %480 = llvm.mul %472, %479 : i64
    %481 = llvm.mlir.constant(4 : index) : i64
    %482 = llvm.mul %474, %481 : i64
    %483 = llvm.add %480, %482 : i64
    %484 = llvm.add %483, %476 : i64
    %485 = llvm.getelementptr %478[%484] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %486 = llvm.load %485 : !llvm.ptr -> f32
    %487 = llvm.extractvalue %460[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %488 = llvm.mlir.constant(4 : index) : i64
    %489 = llvm.mul %472, %488 : i64
    %490 = llvm.add %489, %474 : i64
    %491 = llvm.getelementptr %487[%490] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %492 = llvm.load %491 : !llvm.ptr -> f32
    %493 = llvm.intr.maxnum(%486, %492)  : (f32, f32) -> f32
    %494 = llvm.extractvalue %460[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %495 = llvm.mlir.constant(4 : index) : i64
    %496 = llvm.mul %472, %495 : i64
    %497 = llvm.add %496, %474 : i64
    %498 = llvm.getelementptr %494[%497] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %493, %498 : f32, !llvm.ptr
    %499 = llvm.add %476, %1 : i64
    llvm.br ^bb104(%499 : i64)
  ^bb106:  // pred: ^bb104
    %500 = llvm.add %474, %1 : i64
    llvm.br ^bb102(%500 : i64)
  ^bb107:  // pred: ^bb102
    %501 = llvm.add %472, %1 : i64
    llvm.br ^bb100(%501 : i64)
  ^bb108:  // pred: ^bb100
    %502 = llvm.mlir.constant(1 : index) : i64
    %503 = llvm.mlir.constant(4 : index) : i64
    %504 = llvm.mlir.constant(4 : index) : i64
    %505 = llvm.mlir.constant(1 : index) : i64
    %506 = llvm.mlir.constant(16 : index) : i64
    %507 = llvm.mlir.constant(16 : index) : i64
    %508 = llvm.mlir.zero : !llvm.ptr
    %509 = llvm.getelementptr %508[%507] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %510 = llvm.ptrtoint %509 : !llvm.ptr to i64
    %511 = llvm.mlir.constant(64 : index) : i64
    %512 = llvm.add %510, %511 : i64
    %513 = llvm.call @malloc(%512) : (i64) -> !llvm.ptr
    %514 = llvm.ptrtoint %513 : !llvm.ptr to i64
    %515 = llvm.mlir.constant(1 : index) : i64
    %516 = llvm.sub %511, %515 : i64
    %517 = llvm.add %514, %516 : i64
    %518 = llvm.urem %517, %511  : i64
    %519 = llvm.sub %517, %518 : i64
    %520 = llvm.inttoptr %519 : i64 to !llvm.ptr
    %521 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %522 = llvm.insertvalue %513, %521[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %523 = llvm.insertvalue %520, %522[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %524 = llvm.mlir.constant(0 : index) : i64
    %525 = llvm.insertvalue %524, %523[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %526 = llvm.insertvalue %502, %525[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %527 = llvm.insertvalue %503, %526[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %528 = llvm.insertvalue %504, %527[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %529 = llvm.insertvalue %506, %528[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %530 = llvm.insertvalue %504, %529[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %531 = llvm.insertvalue %505, %530[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb109(%8 : i64)
  ^bb109(%532: i64):  // 2 preds: ^bb108, ^bb116
    %533 = llvm.icmp "slt" %532, %1 : i64
    llvm.cond_br %533, ^bb110, ^bb117
  ^bb110:  // pred: ^bb109
    llvm.br ^bb111(%8 : i64)
  ^bb111(%534: i64):  // 2 preds: ^bb110, ^bb115
    %535 = llvm.icmp "slt" %534, %0 : i64
    llvm.cond_br %535, ^bb112, ^bb116
  ^bb112:  // pred: ^bb111
    llvm.br ^bb113(%8 : i64)
  ^bb113(%536: i64):  // 2 preds: ^bb112, ^bb114
    %537 = llvm.icmp "slt" %536, %0 : i64
    llvm.cond_br %537, ^bb114, ^bb115
  ^bb114:  // pred: ^bb113
    %538 = llvm.extractvalue %407[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %539 = llvm.mlir.constant(16 : index) : i64
    %540 = llvm.mul %532, %539 : i64
    %541 = llvm.mlir.constant(4 : index) : i64
    %542 = llvm.mul %534, %541 : i64
    %543 = llvm.add %540, %542 : i64
    %544 = llvm.add %543, %536 : i64
    %545 = llvm.getelementptr %538[%544] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %546 = llvm.load %545 : !llvm.ptr -> f32
    %547 = llvm.extractvalue %460[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %548 = llvm.mlir.constant(4 : index) : i64
    %549 = llvm.mul %532, %548 : i64
    %550 = llvm.add %549, %534 : i64
    %551 = llvm.getelementptr %547[%550] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %552 = llvm.load %551 : !llvm.ptr -> f32
    %553 = llvm.fsub %546, %552  : f32
    %554 = llvm.intr.exp(%553)  : (f32) -> f32
    %555 = llvm.extractvalue %531[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %556 = llvm.mlir.constant(16 : index) : i64
    %557 = llvm.mul %532, %556 : i64
    %558 = llvm.mlir.constant(4 : index) : i64
    %559 = llvm.mul %534, %558 : i64
    %560 = llvm.add %557, %559 : i64
    %561 = llvm.add %560, %536 : i64
    %562 = llvm.getelementptr %555[%561] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %554, %562 : f32, !llvm.ptr
    %563 = llvm.add %536, %1 : i64
    llvm.br ^bb113(%563 : i64)
  ^bb115:  // pred: ^bb113
    %564 = llvm.add %534, %1 : i64
    llvm.br ^bb111(%564 : i64)
  ^bb116:  // pred: ^bb111
    %565 = llvm.add %532, %1 : i64
    llvm.br ^bb109(%565 : i64)
  ^bb117:  // pred: ^bb109
    %566 = llvm.mlir.constant(1 : index) : i64
    %567 = llvm.mlir.constant(4 : index) : i64
    %568 = llvm.mlir.constant(1 : index) : i64
    %569 = llvm.mlir.constant(4 : index) : i64
    %570 = llvm.mlir.zero : !llvm.ptr
    %571 = llvm.getelementptr %570[%569] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %572 = llvm.ptrtoint %571 : !llvm.ptr to i64
    %573 = llvm.mlir.constant(64 : index) : i64
    %574 = llvm.add %572, %573 : i64
    %575 = llvm.call @malloc(%574) : (i64) -> !llvm.ptr
    %576 = llvm.ptrtoint %575 : !llvm.ptr to i64
    %577 = llvm.mlir.constant(1 : index) : i64
    %578 = llvm.sub %573, %577 : i64
    %579 = llvm.add %576, %578 : i64
    %580 = llvm.urem %579, %573  : i64
    %581 = llvm.sub %579, %580 : i64
    %582 = llvm.inttoptr %581 : i64 to !llvm.ptr
    %583 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %584 = llvm.insertvalue %575, %583[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %585 = llvm.insertvalue %582, %584[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %586 = llvm.mlir.constant(0 : index) : i64
    %587 = llvm.insertvalue %586, %585[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %588 = llvm.insertvalue %566, %587[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %589 = llvm.insertvalue %567, %588[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %590 = llvm.insertvalue %567, %589[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %591 = llvm.insertvalue %568, %590[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb118(%8 : i64)
  ^bb118(%592: i64):  // 2 preds: ^bb117, ^bb122
    %593 = llvm.icmp "slt" %592, %1 : i64
    llvm.cond_br %593, ^bb119, ^bb123
  ^bb119:  // pred: ^bb118
    llvm.br ^bb120(%8 : i64)
  ^bb120(%594: i64):  // 2 preds: ^bb119, ^bb121
    %595 = llvm.icmp "slt" %594, %0 : i64
    llvm.cond_br %595, ^bb121, ^bb122
  ^bb121:  // pred: ^bb120
    %596 = llvm.extractvalue %591[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %597 = llvm.mlir.constant(4 : index) : i64
    %598 = llvm.mul %592, %597 : i64
    %599 = llvm.add %598, %594 : i64
    %600 = llvm.getelementptr %596[%599] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %600 : f32, !llvm.ptr
    %601 = llvm.add %594, %1 : i64
    llvm.br ^bb120(%601 : i64)
  ^bb122:  // pred: ^bb120
    %602 = llvm.add %592, %1 : i64
    llvm.br ^bb118(%602 : i64)
  ^bb123:  // pred: ^bb118
    llvm.br ^bb124(%8 : i64)
  ^bb124(%603: i64):  // 2 preds: ^bb123, ^bb131
    %604 = llvm.icmp "slt" %603, %1 : i64
    llvm.cond_br %604, ^bb125, ^bb132
  ^bb125:  // pred: ^bb124
    llvm.br ^bb126(%8 : i64)
  ^bb126(%605: i64):  // 2 preds: ^bb125, ^bb130
    %606 = llvm.icmp "slt" %605, %0 : i64
    llvm.cond_br %606, ^bb127, ^bb131
  ^bb127:  // pred: ^bb126
    llvm.br ^bb128(%8 : i64)
  ^bb128(%607: i64):  // 2 preds: ^bb127, ^bb129
    %608 = llvm.icmp "slt" %607, %0 : i64
    llvm.cond_br %608, ^bb129, ^bb130
  ^bb129:  // pred: ^bb128
    %609 = llvm.extractvalue %531[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %610 = llvm.mlir.constant(16 : index) : i64
    %611 = llvm.mul %603, %610 : i64
    %612 = llvm.mlir.constant(4 : index) : i64
    %613 = llvm.mul %605, %612 : i64
    %614 = llvm.add %611, %613 : i64
    %615 = llvm.add %614, %607 : i64
    %616 = llvm.getelementptr %609[%615] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %617 = llvm.load %616 : !llvm.ptr -> f32
    %618 = llvm.extractvalue %591[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %619 = llvm.mlir.constant(4 : index) : i64
    %620 = llvm.mul %603, %619 : i64
    %621 = llvm.add %620, %605 : i64
    %622 = llvm.getelementptr %618[%621] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %623 = llvm.load %622 : !llvm.ptr -> f32
    %624 = llvm.fadd %617, %623  : f32
    %625 = llvm.extractvalue %591[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %626 = llvm.mlir.constant(4 : index) : i64
    %627 = llvm.mul %603, %626 : i64
    %628 = llvm.add %627, %605 : i64
    %629 = llvm.getelementptr %625[%628] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %624, %629 : f32, !llvm.ptr
    %630 = llvm.add %607, %1 : i64
    llvm.br ^bb128(%630 : i64)
  ^bb130:  // pred: ^bb128
    %631 = llvm.add %605, %1 : i64
    llvm.br ^bb126(%631 : i64)
  ^bb131:  // pred: ^bb126
    %632 = llvm.add %603, %1 : i64
    llvm.br ^bb124(%632 : i64)
  ^bb132:  // pred: ^bb124
    %633 = llvm.mlir.constant(1 : index) : i64
    %634 = llvm.mlir.constant(4 : index) : i64
    %635 = llvm.mlir.constant(8 : index) : i64
    %636 = llvm.mlir.constant(1 : index) : i64
    %637 = llvm.mlir.constant(32 : index) : i64
    %638 = llvm.mlir.constant(32 : index) : i64
    %639 = llvm.mlir.zero : !llvm.ptr
    %640 = llvm.getelementptr %639[%638] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %641 = llvm.ptrtoint %640 : !llvm.ptr to i64
    %642 = llvm.mlir.constant(64 : index) : i64
    %643 = llvm.add %641, %642 : i64
    %644 = llvm.call @malloc(%643) : (i64) -> !llvm.ptr
    %645 = llvm.ptrtoint %644 : !llvm.ptr to i64
    %646 = llvm.mlir.constant(1 : index) : i64
    %647 = llvm.sub %642, %646 : i64
    %648 = llvm.add %645, %647 : i64
    %649 = llvm.urem %648, %642  : i64
    %650 = llvm.sub %648, %649 : i64
    %651 = llvm.inttoptr %650 : i64 to !llvm.ptr
    %652 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %653 = llvm.insertvalue %644, %652[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %654 = llvm.insertvalue %651, %653[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %655 = llvm.mlir.constant(0 : index) : i64
    %656 = llvm.insertvalue %655, %654[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %657 = llvm.insertvalue %633, %656[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %658 = llvm.insertvalue %634, %657[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %659 = llvm.insertvalue %635, %658[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %660 = llvm.insertvalue %637, %659[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %661 = llvm.insertvalue %635, %660[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %662 = llvm.insertvalue %636, %661[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb133(%8 : i64)
  ^bb133(%663: i64):  // 2 preds: ^bb132, ^bb140
    %664 = llvm.icmp "slt" %663, %1 : i64
    llvm.cond_br %664, ^bb134, ^bb141
  ^bb134:  // pred: ^bb133
    llvm.br ^bb135(%8 : i64)
  ^bb135(%665: i64):  // 2 preds: ^bb134, ^bb139
    %666 = llvm.icmp "slt" %665, %0 : i64
    llvm.cond_br %666, ^bb136, ^bb140
  ^bb136:  // pred: ^bb135
    llvm.br ^bb137(%8 : i64)
  ^bb137(%667: i64):  // 2 preds: ^bb136, ^bb138
    %668 = llvm.icmp "slt" %667, %7 : i64
    llvm.cond_br %668, ^bb138, ^bb139
  ^bb138:  // pred: ^bb137
    %669 = llvm.extractvalue %662[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %670 = llvm.mlir.constant(32 : index) : i64
    %671 = llvm.mul %663, %670 : i64
    %672 = llvm.mlir.constant(8 : index) : i64
    %673 = llvm.mul %665, %672 : i64
    %674 = llvm.add %671, %673 : i64
    %675 = llvm.add %674, %667 : i64
    %676 = llvm.getelementptr %669[%675] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %676 : f32, !llvm.ptr
    %677 = llvm.add %667, %1 : i64
    llvm.br ^bb137(%677 : i64)
  ^bb139:  // pred: ^bb137
    %678 = llvm.add %665, %1 : i64
    llvm.br ^bb135(%678 : i64)
  ^bb140:  // pred: ^bb135
    %679 = llvm.add %663, %1 : i64
    llvm.br ^bb133(%679 : i64)
  ^bb141:  // pred: ^bb133
    llvm.br ^bb142(%8 : i64)
  ^bb142(%680: i64):  // 2 preds: ^bb141, ^bb152
    %681 = llvm.icmp "slt" %680, %1 : i64
    llvm.cond_br %681, ^bb143, ^bb153
  ^bb143:  // pred: ^bb142
    llvm.br ^bb144(%8 : i64)
  ^bb144(%682: i64):  // 2 preds: ^bb143, ^bb151
    %683 = llvm.icmp "slt" %682, %0 : i64
    llvm.cond_br %683, ^bb145, ^bb152
  ^bb145:  // pred: ^bb144
    llvm.br ^bb146(%8 : i64)
  ^bb146(%684: i64):  // 2 preds: ^bb145, ^bb150
    %685 = llvm.icmp "slt" %684, %7 : i64
    llvm.cond_br %685, ^bb147, ^bb151
  ^bb147:  // pred: ^bb146
    llvm.br ^bb148(%8 : i64)
  ^bb148(%686: i64):  // 2 preds: ^bb147, ^bb149
    %687 = llvm.icmp "slt" %686, %0 : i64
    llvm.cond_br %687, ^bb149, ^bb150
  ^bb149:  // pred: ^bb148
    %688 = llvm.extractvalue %531[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %689 = llvm.mlir.constant(16 : index) : i64
    %690 = llvm.mul %680, %689 : i64
    %691 = llvm.mlir.constant(4 : index) : i64
    %692 = llvm.mul %682, %691 : i64
    %693 = llvm.add %690, %692 : i64
    %694 = llvm.add %693, %686 : i64
    %695 = llvm.getelementptr %688[%694] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %696 = llvm.load %695 : !llvm.ptr -> f32
    %697 = llvm.extractvalue %591[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %698 = llvm.mlir.constant(4 : index) : i64
    %699 = llvm.mul %680, %698 : i64
    %700 = llvm.add %699, %682 : i64
    %701 = llvm.getelementptr %697[%700] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %702 = llvm.load %701 : !llvm.ptr -> f32
    %703 = llvm.extractvalue %227[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %704 = llvm.mlir.constant(32 : index) : i64
    %705 = llvm.mul %680, %704 : i64
    %706 = llvm.mlir.constant(8 : index) : i64
    %707 = llvm.mul %686, %706 : i64
    %708 = llvm.add %705, %707 : i64
    %709 = llvm.add %708, %684 : i64
    %710 = llvm.getelementptr %703[%709] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %711 = llvm.load %710 : !llvm.ptr -> f32
    %712 = llvm.extractvalue %662[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %713 = llvm.mlir.constant(32 : index) : i64
    %714 = llvm.mul %680, %713 : i64
    %715 = llvm.mlir.constant(8 : index) : i64
    %716 = llvm.mul %682, %715 : i64
    %717 = llvm.add %714, %716 : i64
    %718 = llvm.add %717, %684 : i64
    %719 = llvm.getelementptr %712[%718] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %720 = llvm.load %719 : !llvm.ptr -> f32
    %721 = llvm.fdiv %696, %702  : f32
    %722 = llvm.fmul %721, %711  : f32
    %723 = llvm.fadd %720, %722  : f32
    %724 = llvm.extractvalue %662[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %725 = llvm.mlir.constant(32 : index) : i64
    %726 = llvm.mul %680, %725 : i64
    %727 = llvm.mlir.constant(8 : index) : i64
    %728 = llvm.mul %682, %727 : i64
    %729 = llvm.add %726, %728 : i64
    %730 = llvm.add %729, %684 : i64
    %731 = llvm.getelementptr %724[%730] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %723, %731 : f32, !llvm.ptr
    %732 = llvm.add %686, %1 : i64
    llvm.br ^bb148(%732 : i64)
  ^bb150:  // pred: ^bb148
    %733 = llvm.add %684, %1 : i64
    llvm.br ^bb146(%733 : i64)
  ^bb151:  // pred: ^bb146
    %734 = llvm.add %682, %1 : i64
    llvm.br ^bb144(%734 : i64)
  ^bb152:  // pred: ^bb144
    %735 = llvm.add %680, %1 : i64
    llvm.br ^bb142(%735 : i64)
  ^bb153:  // pred: ^bb142
    %736 = llvm.mlir.constant(1 : index) : i64
    %737 = llvm.mlir.constant(4 : index) : i64
    %738 = llvm.mlir.constant(8 : index) : i64
    %739 = llvm.mlir.constant(1 : index) : i64
    %740 = llvm.mlir.constant(32 : index) : i64
    %741 = llvm.mlir.constant(32 : index) : i64
    %742 = llvm.mlir.zero : !llvm.ptr
    %743 = llvm.getelementptr %742[%741] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %744 = llvm.ptrtoint %743 : !llvm.ptr to i64
    %745 = llvm.mlir.constant(64 : index) : i64
    %746 = llvm.add %744, %745 : i64
    %747 = llvm.call @malloc(%746) : (i64) -> !llvm.ptr
    %748 = llvm.ptrtoint %747 : !llvm.ptr to i64
    %749 = llvm.mlir.constant(1 : index) : i64
    %750 = llvm.sub %745, %749 : i64
    %751 = llvm.add %748, %750 : i64
    %752 = llvm.urem %751, %745  : i64
    %753 = llvm.sub %751, %752 : i64
    %754 = llvm.inttoptr %753 : i64 to !llvm.ptr
    %755 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %756 = llvm.insertvalue %747, %755[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %757 = llvm.insertvalue %754, %756[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %758 = llvm.mlir.constant(0 : index) : i64
    %759 = llvm.insertvalue %758, %757[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %760 = llvm.insertvalue %736, %759[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %761 = llvm.insertvalue %737, %760[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %762 = llvm.insertvalue %738, %761[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %763 = llvm.insertvalue %740, %762[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %764 = llvm.insertvalue %738, %763[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %765 = llvm.insertvalue %739, %764[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb154(%8 : i64)
  ^bb154(%766: i64):  // 2 preds: ^bb153, ^bb161
    %767 = llvm.icmp "slt" %766, %1 : i64
    llvm.cond_br %767, ^bb155, ^bb162
  ^bb155:  // pred: ^bb154
    llvm.br ^bb156(%8 : i64)
  ^bb156(%768: i64):  // 2 preds: ^bb155, ^bb160
    %769 = llvm.icmp "slt" %768, %0 : i64
    llvm.cond_br %769, ^bb157, ^bb161
  ^bb157:  // pred: ^bb156
    llvm.br ^bb158(%8 : i64)
  ^bb158(%770: i64):  // 2 preds: ^bb157, ^bb159
    %771 = llvm.icmp "slt" %770, %7 : i64
    llvm.cond_br %771, ^bb159, ^bb160
  ^bb159:  // pred: ^bb158
    %772 = llvm.extractvalue %662[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %773 = llvm.mlir.constant(32 : index) : i64
    %774 = llvm.mul %766, %773 : i64
    %775 = llvm.mlir.constant(8 : index) : i64
    %776 = llvm.mul %768, %775 : i64
    %777 = llvm.add %774, %776 : i64
    %778 = llvm.add %777, %770 : i64
    %779 = llvm.getelementptr %772[%778] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %780 = llvm.load %779 : !llvm.ptr -> f32
    %781 = llvm.fadd %780, %3  : f32
    %782 = llvm.extractvalue %765[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %783 = llvm.mlir.constant(32 : index) : i64
    %784 = llvm.mul %766, %783 : i64
    %785 = llvm.mlir.constant(8 : index) : i64
    %786 = llvm.mul %768, %785 : i64
    %787 = llvm.add %784, %786 : i64
    %788 = llvm.add %787, %770 : i64
    %789 = llvm.getelementptr %782[%788] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %781, %789 : f32, !llvm.ptr
    %790 = llvm.add %770, %1 : i64
    llvm.br ^bb158(%790 : i64)
  ^bb160:  // pred: ^bb158
    %791 = llvm.add %768, %1 : i64
    llvm.br ^bb156(%791 : i64)
  ^bb161:  // pred: ^bb156
    %792 = llvm.add %766, %1 : i64
    llvm.br ^bb154(%792 : i64)
  ^bb162:  // pred: ^bb154
    %793 = llvm.mlir.constant(1 : index) : i64
    %794 = llvm.mlir.constant(4 : index) : i64
    %795 = llvm.mlir.constant(32 : index) : i64
    %796 = llvm.mlir.constant(1 : index) : i64
    %797 = llvm.mlir.constant(128 : index) : i64
    %798 = llvm.mlir.constant(128 : index) : i64
    %799 = llvm.mlir.zero : !llvm.ptr
    %800 = llvm.getelementptr %799[%798] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %801 = llvm.ptrtoint %800 : !llvm.ptr to i64
    %802 = llvm.mlir.constant(64 : index) : i64
    %803 = llvm.add %801, %802 : i64
    %804 = llvm.call @malloc(%803) : (i64) -> !llvm.ptr
    %805 = llvm.ptrtoint %804 : !llvm.ptr to i64
    %806 = llvm.mlir.constant(1 : index) : i64
    %807 = llvm.sub %802, %806 : i64
    %808 = llvm.add %805, %807 : i64
    %809 = llvm.urem %808, %802  : i64
    %810 = llvm.sub %808, %809 : i64
    %811 = llvm.inttoptr %810 : i64 to !llvm.ptr
    %812 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %813 = llvm.insertvalue %804, %812[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %814 = llvm.insertvalue %811, %813[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %815 = llvm.mlir.constant(0 : index) : i64
    %816 = llvm.insertvalue %815, %814[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %817 = llvm.insertvalue %793, %816[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %818 = llvm.insertvalue %794, %817[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %819 = llvm.insertvalue %795, %818[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %820 = llvm.insertvalue %797, %819[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %821 = llvm.insertvalue %795, %820[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %822 = llvm.insertvalue %796, %821[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %823 = builtin.unrealized_conversion_cast %822 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<1x4x32xf32>
    llvm.br ^bb163(%8 : i64)
  ^bb163(%824: i64):  // 2 preds: ^bb162, ^bb173
    %825 = builtin.unrealized_conversion_cast %824 : i64 to index
    %826 = llvm.icmp "slt" %824, %9 : i64
    llvm.cond_br %826, ^bb164, ^bb174
  ^bb164:  // pred: ^bb163
    %subview = memref.subview %823[0, 0, %825] [1, 4, 8] [1, 1, 1] : memref<1x4x32xf32> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
    %827 = builtin.unrealized_conversion_cast %subview : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    llvm.br ^bb165(%8 : i64)
  ^bb165(%828: i64):  // 2 preds: ^bb164, ^bb172
    %829 = llvm.icmp "slt" %828, %1 : i64
    llvm.cond_br %829, ^bb166, ^bb173
  ^bb166:  // pred: ^bb165
    llvm.br ^bb167(%8 : i64)
  ^bb167(%830: i64):  // 2 preds: ^bb166, ^bb171
    %831 = llvm.icmp "slt" %830, %0 : i64
    llvm.cond_br %831, ^bb168, ^bb172
  ^bb168:  // pred: ^bb167
    llvm.br ^bb169(%8 : i64)
  ^bb169(%832: i64):  // 2 preds: ^bb168, ^bb170
    %833 = llvm.icmp "slt" %832, %7 : i64
    llvm.cond_br %833, ^bb170, ^bb171
  ^bb170:  // pred: ^bb169
    %834 = llvm.extractvalue %827[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %835 = llvm.extractvalue %827[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %836 = llvm.getelementptr %834[%835] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %837 = llvm.mlir.constant(128 : index) : i64
    %838 = llvm.mul %828, %837 : i64
    %839 = llvm.mlir.constant(32 : index) : i64
    %840 = llvm.mul %830, %839 : i64
    %841 = llvm.add %838, %840 : i64
    %842 = llvm.add %841, %832 : i64
    %843 = llvm.getelementptr %836[%842] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %843 : f32, !llvm.ptr
    %844 = llvm.add %832, %1 : i64
    llvm.br ^bb169(%844 : i64)
  ^bb171:  // pred: ^bb169
    %845 = llvm.add %830, %1 : i64
    llvm.br ^bb167(%845 : i64)
  ^bb172:  // pred: ^bb167
    %846 = llvm.add %828, %1 : i64
    llvm.br ^bb165(%846 : i64)
  ^bb173:  // pred: ^bb165
    %847 = llvm.intr.stacksave : !llvm.ptr
    %848 = llvm.mlir.constant(3 : i64) : i64
    %849 = llvm.mlir.constant(1 : index) : i64
    %850 = llvm.alloca %849 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %827, %850 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %851 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %852 = llvm.insertvalue %848, %851[0] : !llvm.struct<(i64, ptr)> 
    %853 = llvm.insertvalue %850, %852[1] : !llvm.struct<(i64, ptr)> 
    %854 = llvm.mlir.constant(3 : i64) : i64
    %855 = llvm.mlir.constant(1 : index) : i64
    %856 = llvm.alloca %855 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %827, %856 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %857 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %858 = llvm.insertvalue %854, %857[0] : !llvm.struct<(i64, ptr)> 
    %859 = llvm.insertvalue %856, %858[1] : !llvm.struct<(i64, ptr)> 
    %860 = llvm.mlir.constant(1 : index) : i64
    %861 = llvm.alloca %860 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %853, %861 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %862 = llvm.alloca %860 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %859, %862 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %863 = llvm.mlir.zero : !llvm.ptr
    %864 = llvm.getelementptr %863[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %865 = llvm.ptrtoint %864 : !llvm.ptr to i64
    llvm.call @memrefCopy(%865, %861, %862) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %847 : !llvm.ptr
    %866 = llvm.add %824, %7 : i64
    llvm.br ^bb163(%866 : i64)
  ^bb174:  // pred: ^bb163
    llvm.br ^bb175(%8 : i64)
  ^bb175(%867: i64):  // 2 preds: ^bb174, ^bb188
    %868 = builtin.unrealized_conversion_cast %867 : i64 to index
    %869 = llvm.icmp "slt" %867, %9 : i64
    llvm.cond_br %869, ^bb176, ^bb189
  ^bb176:  // pred: ^bb175
    %subview_0 = memref.subview %823[0, 0, %868] [1, 4, 8] [1, 1, 1] : memref<1x4x32xf32> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
    %870 = builtin.unrealized_conversion_cast %subview_0 : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    llvm.br ^bb177(%8 : i64)
  ^bb177(%871: i64):  // 2 preds: ^bb176, ^bb187
    %872 = llvm.icmp "slt" %871, %1 : i64
    llvm.cond_br %872, ^bb178, ^bb188
  ^bb178:  // pred: ^bb177
    llvm.br ^bb179(%8 : i64)
  ^bb179(%873: i64):  // 2 preds: ^bb178, ^bb186
    %874 = llvm.icmp "slt" %873, %0 : i64
    llvm.cond_br %874, ^bb180, ^bb187
  ^bb180:  // pred: ^bb179
    llvm.br ^bb181(%8 : i64)
  ^bb181(%875: i64):  // 2 preds: ^bb180, ^bb185
    %876 = llvm.icmp "slt" %875, %7 : i64
    llvm.cond_br %876, ^bb182, ^bb186
  ^bb182:  // pred: ^bb181
    llvm.br ^bb183(%8 : i64)
  ^bb183(%877: i64):  // 2 preds: ^bb182, ^bb184
    %878 = llvm.icmp "slt" %877, %7 : i64
    llvm.cond_br %878, ^bb184, ^bb185
  ^bb184:  // pred: ^bb183
    %879 = llvm.extractvalue %765[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %880 = llvm.mlir.constant(32 : index) : i64
    %881 = llvm.mul %871, %880 : i64
    %882 = llvm.mlir.constant(8 : index) : i64
    %883 = llvm.mul %873, %882 : i64
    %884 = llvm.add %881, %883 : i64
    %885 = llvm.add %884, %877 : i64
    %886 = llvm.getelementptr %879[%885] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %887 = llvm.load %886 : !llvm.ptr -> f32
    %888 = llvm.extractvalue %870[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %889 = llvm.extractvalue %870[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %890 = llvm.getelementptr %888[%889] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %891 = llvm.mlir.constant(128 : index) : i64
    %892 = llvm.mul %871, %891 : i64
    %893 = llvm.mlir.constant(32 : index) : i64
    %894 = llvm.mul %873, %893 : i64
    %895 = llvm.add %892, %894 : i64
    %896 = llvm.add %895, %875 : i64
    %897 = llvm.getelementptr %890[%896] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %898 = llvm.load %897 : !llvm.ptr -> f32
    %899 = llvm.fmul %887, %6  : f32
    %900 = llvm.fadd %898, %899  : f32
    %901 = llvm.extractvalue %870[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %902 = llvm.extractvalue %870[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %903 = llvm.getelementptr %901[%902] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %904 = llvm.mlir.constant(128 : index) : i64
    %905 = llvm.mul %871, %904 : i64
    %906 = llvm.mlir.constant(32 : index) : i64
    %907 = llvm.mul %873, %906 : i64
    %908 = llvm.add %905, %907 : i64
    %909 = llvm.add %908, %875 : i64
    %910 = llvm.getelementptr %903[%909] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %900, %910 : f32, !llvm.ptr
    %911 = llvm.add %877, %1 : i64
    llvm.br ^bb183(%911 : i64)
  ^bb185:  // pred: ^bb183
    %912 = llvm.add %875, %1 : i64
    llvm.br ^bb181(%912 : i64)
  ^bb186:  // pred: ^bb181
    %913 = llvm.add %873, %1 : i64
    llvm.br ^bb179(%913 : i64)
  ^bb187:  // pred: ^bb179
    %914 = llvm.add %871, %1 : i64
    llvm.br ^bb177(%914 : i64)
  ^bb188:  // pred: ^bb177
    %915 = llvm.intr.stacksave : !llvm.ptr
    %916 = llvm.mlir.constant(3 : i64) : i64
    %917 = llvm.mlir.constant(1 : index) : i64
    %918 = llvm.alloca %917 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %870, %918 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %919 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %920 = llvm.insertvalue %916, %919[0] : !llvm.struct<(i64, ptr)> 
    %921 = llvm.insertvalue %918, %920[1] : !llvm.struct<(i64, ptr)> 
    %922 = llvm.mlir.constant(3 : i64) : i64
    %923 = llvm.mlir.constant(1 : index) : i64
    %924 = llvm.alloca %923 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %870, %924 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %925 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %926 = llvm.insertvalue %922, %925[0] : !llvm.struct<(i64, ptr)> 
    %927 = llvm.insertvalue %924, %926[1] : !llvm.struct<(i64, ptr)> 
    %928 = llvm.mlir.constant(1 : index) : i64
    %929 = llvm.alloca %928 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %921, %929 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %930 = llvm.alloca %928 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %927, %930 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %931 = llvm.mlir.zero : !llvm.ptr
    %932 = llvm.getelementptr %931[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %933 = llvm.ptrtoint %932 : !llvm.ptr to i64
    llvm.call @memrefCopy(%933, %929, %930) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %915 : !llvm.ptr
    %934 = llvm.add %867, %7 : i64
    llvm.br ^bb175(%934 : i64)
  ^bb189:  // pred: ^bb175
    %935 = llvm.mlir.constant(1 : index) : i64
    %936 = llvm.mlir.constant(4 : index) : i64
    %937 = llvm.mlir.constant(8 : index) : i64
    %938 = llvm.mlir.constant(1 : index) : i64
    %939 = llvm.mlir.constant(32 : index) : i64
    %940 = llvm.mlir.constant(32 : index) : i64
    %941 = llvm.mlir.zero : !llvm.ptr
    %942 = llvm.getelementptr %941[%940] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %943 = llvm.ptrtoint %942 : !llvm.ptr to i64
    %944 = llvm.mlir.constant(64 : index) : i64
    %945 = llvm.add %943, %944 : i64
    %946 = llvm.call @malloc(%945) : (i64) -> !llvm.ptr
    %947 = llvm.ptrtoint %946 : !llvm.ptr to i64
    %948 = llvm.mlir.constant(1 : index) : i64
    %949 = llvm.sub %944, %948 : i64
    %950 = llvm.add %947, %949 : i64
    %951 = llvm.urem %950, %944  : i64
    %952 = llvm.sub %950, %951 : i64
    %953 = llvm.inttoptr %952 : i64 to !llvm.ptr
    %954 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %955 = llvm.insertvalue %946, %954[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %956 = llvm.insertvalue %953, %955[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %957 = llvm.mlir.constant(0 : index) : i64
    %958 = llvm.insertvalue %957, %956[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %959 = llvm.insertvalue %935, %958[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %960 = llvm.insertvalue %936, %959[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %961 = llvm.insertvalue %937, %960[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %962 = llvm.insertvalue %939, %961[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %963 = llvm.insertvalue %937, %962[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %964 = llvm.insertvalue %938, %963[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb190(%8 : i64)
  ^bb190(%965: i64):  // 2 preds: ^bb189, ^bb197
    %966 = llvm.icmp "slt" %965, %1 : i64
    llvm.cond_br %966, ^bb191, ^bb198
  ^bb191:  // pred: ^bb190
    llvm.br ^bb192(%8 : i64)
  ^bb192(%967: i64):  // 2 preds: ^bb191, ^bb196
    %968 = llvm.icmp "slt" %967, %0 : i64
    llvm.cond_br %968, ^bb193, ^bb197
  ^bb193:  // pred: ^bb192
    llvm.br ^bb194(%8 : i64)
  ^bb194(%969: i64):  // 2 preds: ^bb193, ^bb195
    %970 = llvm.icmp "slt" %969, %7 : i64
    llvm.cond_br %970, ^bb195, ^bb196
  ^bb195:  // pred: ^bb194
    %971 = llvm.extractvalue %964[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %972 = llvm.mlir.constant(32 : index) : i64
    %973 = llvm.mul %965, %972 : i64
    %974 = llvm.mlir.constant(8 : index) : i64
    %975 = llvm.mul %967, %974 : i64
    %976 = llvm.add %973, %975 : i64
    %977 = llvm.add %976, %969 : i64
    %978 = llvm.getelementptr %971[%977] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %978 : f32, !llvm.ptr
    %979 = llvm.add %969, %1 : i64
    llvm.br ^bb194(%979 : i64)
  ^bb196:  // pred: ^bb194
    %980 = llvm.add %967, %1 : i64
    llvm.br ^bb192(%980 : i64)
  ^bb197:  // pred: ^bb192
    %981 = llvm.add %965, %1 : i64
    llvm.br ^bb190(%981 : i64)
  ^bb198:  // pred: ^bb190
    llvm.br ^bb199(%8 : i64)
  ^bb199(%982: i64):  // 2 preds: ^bb198, ^bb209
    %983 = llvm.icmp "slt" %982, %1 : i64
    llvm.cond_br %983, ^bb200, ^bb210
  ^bb200:  // pred: ^bb199
    llvm.br ^bb201(%8 : i64)
  ^bb201(%984: i64):  // 2 preds: ^bb200, ^bb208
    %985 = llvm.icmp "slt" %984, %0 : i64
    llvm.cond_br %985, ^bb202, ^bb209
  ^bb202:  // pred: ^bb201
    llvm.br ^bb203(%8 : i64)
  ^bb203(%986: i64):  // 2 preds: ^bb202, ^bb207
    %987 = llvm.icmp "slt" %986, %7 : i64
    llvm.cond_br %987, ^bb204, ^bb208
  ^bb204:  // pred: ^bb203
    llvm.br ^bb205(%8 : i64)
  ^bb205(%988: i64):  // 2 preds: ^bb204, ^bb206
    %989 = llvm.icmp "slt" %988, %9 : i64
    llvm.cond_br %989, ^bb206, ^bb207
  ^bb206:  // pred: ^bb205
    %990 = llvm.extractvalue %822[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %991 = llvm.mlir.constant(128 : index) : i64
    %992 = llvm.mul %982, %991 : i64
    %993 = llvm.mlir.constant(32 : index) : i64
    %994 = llvm.mul %984, %993 : i64
    %995 = llvm.add %992, %994 : i64
    %996 = llvm.add %995, %988 : i64
    %997 = llvm.getelementptr %990[%996] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %998 = llvm.load %997 : !llvm.ptr -> f32
    %999 = llvm.extractvalue %964[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1000 = llvm.mlir.constant(32 : index) : i64
    %1001 = llvm.mul %982, %1000 : i64
    %1002 = llvm.mlir.constant(8 : index) : i64
    %1003 = llvm.mul %984, %1002 : i64
    %1004 = llvm.add %1001, %1003 : i64
    %1005 = llvm.add %1004, %986 : i64
    %1006 = llvm.getelementptr %999[%1005] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1007 = llvm.load %1006 : !llvm.ptr -> f32
    %1008 = llvm.intr.maximum(%998, %2)  : (f32, f32) -> f32
    %1009 = llvm.fmul %1008, %6  : f32
    %1010 = llvm.fadd %1007, %1009  : f32
    %1011 = llvm.extractvalue %964[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1012 = llvm.mlir.constant(32 : index) : i64
    %1013 = llvm.mul %982, %1012 : i64
    %1014 = llvm.mlir.constant(8 : index) : i64
    %1015 = llvm.mul %984, %1014 : i64
    %1016 = llvm.add %1013, %1015 : i64
    %1017 = llvm.add %1016, %986 : i64
    %1018 = llvm.getelementptr %1011[%1017] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %1010, %1018 : f32, !llvm.ptr
    %1019 = llvm.add %988, %1 : i64
    llvm.br ^bb205(%1019 : i64)
  ^bb207:  // pred: ^bb205
    %1020 = llvm.add %986, %1 : i64
    llvm.br ^bb203(%1020 : i64)
  ^bb208:  // pred: ^bb203
    %1021 = llvm.add %984, %1 : i64
    llvm.br ^bb201(%1021 : i64)
  ^bb209:  // pred: ^bb201
    %1022 = llvm.add %982, %1 : i64
    llvm.br ^bb199(%1022 : i64)
  ^bb210:  // pred: ^bb199
    %1023 = llvm.mlir.constant(1 : index) : i64
    %1024 = llvm.mlir.constant(4 : index) : i64
    %1025 = llvm.mlir.constant(8 : index) : i64
    %1026 = llvm.mlir.constant(1 : index) : i64
    %1027 = llvm.mlir.constant(32 : index) : i64
    %1028 = llvm.mlir.constant(32 : index) : i64
    %1029 = llvm.mlir.zero : !llvm.ptr
    %1030 = llvm.getelementptr %1029[%1028] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1031 = llvm.ptrtoint %1030 : !llvm.ptr to i64
    %1032 = llvm.mlir.constant(64 : index) : i64
    %1033 = llvm.add %1031, %1032 : i64
    %1034 = llvm.call @malloc(%1033) : (i64) -> !llvm.ptr
    %1035 = llvm.ptrtoint %1034 : !llvm.ptr to i64
    %1036 = llvm.mlir.constant(1 : index) : i64
    %1037 = llvm.sub %1032, %1036 : i64
    %1038 = llvm.add %1035, %1037 : i64
    %1039 = llvm.urem %1038, %1032  : i64
    %1040 = llvm.sub %1038, %1039 : i64
    %1041 = llvm.inttoptr %1040 : i64 to !llvm.ptr
    %1042 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %1043 = llvm.insertvalue %1034, %1042[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1044 = llvm.insertvalue %1041, %1043[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1045 = llvm.mlir.constant(0 : index) : i64
    %1046 = llvm.insertvalue %1045, %1044[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1047 = llvm.insertvalue %1023, %1046[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1048 = llvm.insertvalue %1024, %1047[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1049 = llvm.insertvalue %1025, %1048[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1050 = llvm.insertvalue %1027, %1049[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1051 = llvm.insertvalue %1025, %1050[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1052 = llvm.insertvalue %1026, %1051[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb211(%8 : i64)
  ^bb211(%1053: i64):  // 2 preds: ^bb210, ^bb218
    %1054 = llvm.icmp "slt" %1053, %1 : i64
    llvm.cond_br %1054, ^bb212, ^bb219
  ^bb212:  // pred: ^bb211
    llvm.br ^bb213(%8 : i64)
  ^bb213(%1055: i64):  // 2 preds: ^bb212, ^bb217
    %1056 = llvm.icmp "slt" %1055, %0 : i64
    llvm.cond_br %1056, ^bb214, ^bb218
  ^bb214:  // pred: ^bb213
    llvm.br ^bb215(%8 : i64)
  ^bb215(%1057: i64):  // 2 preds: ^bb214, ^bb216
    %1058 = llvm.icmp "slt" %1057, %7 : i64
    llvm.cond_br %1058, ^bb216, ^bb217
  ^bb216:  // pred: ^bb215
    %1059 = llvm.extractvalue %765[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1060 = llvm.mlir.constant(32 : index) : i64
    %1061 = llvm.mul %1053, %1060 : i64
    %1062 = llvm.mlir.constant(8 : index) : i64
    %1063 = llvm.mul %1055, %1062 : i64
    %1064 = llvm.add %1061, %1063 : i64
    %1065 = llvm.add %1064, %1057 : i64
    %1066 = llvm.getelementptr %1059[%1065] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1067 = llvm.load %1066 : !llvm.ptr -> f32
    %1068 = llvm.extractvalue %964[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1069 = llvm.mlir.constant(32 : index) : i64
    %1070 = llvm.mul %1053, %1069 : i64
    %1071 = llvm.mlir.constant(8 : index) : i64
    %1072 = llvm.mul %1055, %1071 : i64
    %1073 = llvm.add %1070, %1072 : i64
    %1074 = llvm.add %1073, %1057 : i64
    %1075 = llvm.getelementptr %1068[%1074] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %1076 = llvm.load %1075 : !llvm.ptr -> f32
    %1077 = llvm.fadd %1067, %1076  : f32
    %1078 = llvm.fmul %1077, %1077  : f32
    %1079 = llvm.fadd %1078, %1078  : f32
    %1080 = llvm.extractvalue %1052[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %1081 = llvm.mlir.constant(32 : index) : i64
    %1082 = llvm.mul %1053, %1081 : i64
    %1083 = llvm.mlir.constant(8 : index) : i64
    %1084 = llvm.mul %1055, %1083 : i64
    %1085 = llvm.add %1082, %1084 : i64
    %1086 = llvm.add %1085, %1057 : i64
    %1087 = llvm.getelementptr %1080[%1086] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %1079, %1087 : f32, !llvm.ptr
    %1088 = llvm.add %1057, %1 : i64
    llvm.br ^bb215(%1088 : i64)
  ^bb217:  // pred: ^bb215
    %1089 = llvm.add %1055, %1 : i64
    llvm.br ^bb213(%1089 : i64)
  ^bb218:  // pred: ^bb213
    %1090 = llvm.add %1053, %1 : i64
    llvm.br ^bb211(%1090 : i64)
  ^bb219:  // pred: ^bb211
    llvm.return %1052 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  }
}
