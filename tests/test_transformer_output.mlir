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
    %13 = cherry.matmul %12, %2 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %14 = cherry.tensor_add %arg0, %13 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %15 = cherry.matmul %14, %arg4 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[8x32xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %16 = cherry.tensor_relu %15 : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %17 = cherry.matmul %16, %arg5 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[32x8xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %18 = cherry.tensor_add %14, %17 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %18 : !cherry.cherry_tensor<[?xf32]>
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
    %20 = cherry.matmul %19, %9 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %21 = cherry.tensor_add %6, %20 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %22 = cherry.matmul %21, %4 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[8x32xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %23 = cherry.tensor_relu %22 : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %24 = cherry.matmul %23, %5 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[32x8xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %25 = cherry.tensor_add %21, %24 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %26 = cherry.tensor_mul %25, %25 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %27 = cherry.tensor_add %26, %26 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %27 : !cherry.cherry_tensor<[?xf32]>
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
    %20 = cherry.matmul %19, %9 : (!cherry.cherry_tensor<[1x4x4xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %21 = cherry.tensor_add %6, %20 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
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
    %19 = cherry.matmul %18, %8 : (!cherry.cherry_tensor<[1x4x4xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %20 = cherry.tensor_add %0, %19 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %21 = cherry.matmul %20, %4 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8x32xf32]>) -> !cherry.cherry_tensor<[1x4x32xf32]>
    %22 = cherry.tensor_relu %21 : (!cherry.cherry_tensor<[1x4x32xf32]>) -> !cherry.cherry_tensor<[1x4x32xf32]>
    %23 = cherry.matmul %22, %5 : (!cherry.cherry_tensor<[1x4x32xf32]>, !cherry.cherry_tensor<[32x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %24 = cherry.tensor_add %20, %23 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %25 = cherry.tensor_mul %24, %24 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    %26 = cherry.tensor_add %25, %25 : (!cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    cherry.return %26 : !cherry.cherry_tensor<[1x4x8xf32]>
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
      %35 = arith.mulf %in, %in_14 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    } -> tensor<1x4x8xf32>
    %3 = tensor.empty() : tensor<1x4x8xf32>
    %cst_6 = arith.constant 0.000000e+00 : f32
    %4 = linalg.fill ins(%cst_6 : f32) outs(%3 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%cst, %cst_1 : tensor<1x4x8xf32>, tensor<8x8xf32>) outs(%4 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %35 = arith.mulf %in, %in_14 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    } -> tensor<1x4x8xf32>
    %6 = tensor.empty() : tensor<1x4x8xf32>
    %cst_7 = arith.constant 0.000000e+00 : f32
    %7 = linalg.fill ins(%cst_7 : f32) outs(%6 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%cst, %cst_2 : tensor<1x4x8xf32>, tensor<8x8xf32>) outs(%7 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %35 = arith.mulf %in, %in_14 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
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
      %35 = arith.mulf %in, %in_14 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    } -> tensor<1x4x4xf32>
    %cst_9 = arith.constant dense<2.828400e+00> : tensor<1xf32>
    %c1_i64_10 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %13 = tensor.empty() : tensor<1x4x4xf32>
    %broadcasted = linalg.broadcast ins(%cst_9 : tensor<1xf32>) outs(%13 : tensor<1x4x4xf32>) dimensions = [1, 2] 
    %14 = tensor.empty() : tensor<1x4x4xf32>
    %15 = linalg.div ins(%12, %broadcasted : tensor<1x4x4xf32>, tensor<1x4x4xf32>) outs(%14 : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
    %16 = tensor.empty() : tensor<1x4x8xf32>
    %cst_11 = arith.constant 0.000000e+00 : f32
    %17 = linalg.fill ins(%cst_11 : f32) outs(%16 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%15, %8 : tensor<1x4x4xf32>, tensor<1x4x8xf32>) outs(%17 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %35 = arith.mulf %in, %in_14 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    } -> tensor<1x4x8xf32>
    %19 = tensor.empty() : tensor<1x4x8xf32>
    %20 = linalg.add ins(%cst, %18 : tensor<1x4x8xf32>, tensor<1x4x8xf32>) outs(%19 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %21 = tensor.empty() : tensor<1x4x32xf32>
    %cst_12 = arith.constant 0.000000e+00 : f32
    %22 = linalg.fill ins(%cst_12 : f32) outs(%21 : tensor<1x4x32xf32>) -> tensor<1x4x32xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%20, %cst_3 : tensor<1x4x8xf32>, tensor<8x32xf32>) outs(%22 : tensor<1x4x32xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %35 = arith.mulf %in, %in_14 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    } -> tensor<1x4x32xf32>
    %24 = tensor.empty() : tensor<1x4x32xf32>
    %25 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23 : tensor<1x4x32xf32>) outs(%24 : tensor<1x4x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_14 = arith.constant 0.000000e+00 : f32
      %35 = arith.maximumf %in, %cst_14 : f32
      linalg.yield %35 : f32
    } -> tensor<1x4x32xf32>
    %26 = tensor.empty() : tensor<1x4x8xf32>
    %cst_13 = arith.constant 0.000000e+00 : f32
    %27 = linalg.fill ins(%cst_13 : f32) outs(%26 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %28 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%25, %cst_4 : tensor<1x4x32xf32>, tensor<32x8xf32>) outs(%27 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %35 = arith.mulf %in, %in_14 : f32
      %36 = arith.addf %out, %35 : f32
      linalg.yield %36 : f32
    } -> tensor<1x4x8xf32>
    %29 = tensor.empty() : tensor<1x4x8xf32>
    %30 = linalg.add ins(%20, %28 : tensor<1x4x8xf32>, tensor<1x4x8xf32>) outs(%29 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %31 = tensor.empty() : tensor<1x4x8xf32>
    %32 = linalg.mul ins(%30, %30 : tensor<1x4x8xf32>, tensor<1x4x8xf32>) outs(%31 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %33 = tensor.empty() : tensor<1x4x8xf32>
    %34 = linalg.add ins(%32, %32 : tensor<1x4x8xf32>, tensor<1x4x8xf32>) outs(%33 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    return %34 : tensor<1x4x8xf32>
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
    %cst = arith.constant 2.000000e-01 : f32
    %cst_47 = arith.constant 2.828400e+00 : f32
    %cst_48 = arith.constant 5.000000e-01 : f32
    %cst_49 = arith.constant 0.000000e+00 : f32
    %cst_50 = arith.constant dense<1.000000e-01> : tensor<8x8xf32>
    %0 = tensor.empty() : tensor<1x4x8xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_51 = arith.constant 8 : index
    %c0_52 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8_53 = arith.constant 8 : index
    %c0_54 = arith.constant 0 : index
    %c8_55 = arith.constant 8 : index
    %c8_56 = arith.constant 8 : index
    %1 = scf.for %arg0 = %c0 to %c1 step %c8_51 iter_args(%arg1 = %0) -> (tensor<1x4x8xf32>) {
      %25 = scf.for %arg2 = %c0_52 to %c4 step %c8_53 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %26 = scf.for %arg4 = %c0_54 to %c8_55 step %c8_56 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %27 = affine.min #map(%arg0)
          %28 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %29 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_49 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %29 into %arg5[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %26 : tensor<1x4x8xf32>
      }
      scf.yield %25 : tensor<1x4x8xf32>
    }
    %c0_57 = arith.constant 0 : index
    %c1_58 = arith.constant 1 : index
    %c8_59 = arith.constant 8 : index
    %c0_60 = arith.constant 0 : index
    %c4_61 = arith.constant 4 : index
    %c8_62 = arith.constant 8 : index
    %c0_63 = arith.constant 0 : index
    %c8_64 = arith.constant 8 : index
    %c8_65 = arith.constant 8 : index
    %2 = scf.for %arg0 = %c0_57 to %c1_58 step %c8_59 iter_args(%arg1 = %1) -> (tensor<1x4x8xf32>) {
      %25 = scf.for %arg2 = %c0_60 to %c4_61 step %c8_62 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %26 = scf.for %arg4 = %c0_63 to %c8_64 step %c8_65 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %27 = affine.min #map(%arg0)
          %28 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %cst_50[0, %arg4] [8, 8] [1, 1] : tensor<8x8xf32> to tensor<8x8xf32>
          %extracted_slice_191 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %29 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<8x8xf32>) outs(%extracted_slice_191 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %30 = arith.mulf %in, %cst_48 : f32
            %31 = arith.addf %out, %30 : f32
            linalg.yield %31 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %29 into %arg5[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %26 : tensor<1x4x8xf32>
      }
      scf.yield %25 : tensor<1x4x8xf32>
    }
    %3 = tensor.empty() : tensor<1x4x8xf32>
    %c0_66 = arith.constant 0 : index
    %c1_67 = arith.constant 1 : index
    %c8_68 = arith.constant 8 : index
    %c0_69 = arith.constant 0 : index
    %c4_70 = arith.constant 4 : index
    %c8_71 = arith.constant 8 : index
    %c0_72 = arith.constant 0 : index
    %c8_73 = arith.constant 8 : index
    %c8_74 = arith.constant 8 : index
    %4 = scf.for %arg0 = %c0_66 to %c1_67 step %c8_68 iter_args(%arg1 = %3) -> (tensor<1x4x8xf32>) {
      %25 = scf.for %arg2 = %c0_69 to %c4_70 step %c8_71 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %26 = scf.for %arg4 = %c0_72 to %c8_73 step %c8_74 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %27 = affine.min #map(%arg0)
          %28 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %29 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_49 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %29 into %arg5[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %26 : tensor<1x4x8xf32>
      }
      scf.yield %25 : tensor<1x4x8xf32>
    }
    %c0_75 = arith.constant 0 : index
    %c1_76 = arith.constant 1 : index
    %c8_77 = arith.constant 8 : index
    %c0_78 = arith.constant 0 : index
    %c4_79 = arith.constant 4 : index
    %c8_80 = arith.constant 8 : index
    %c0_81 = arith.constant 0 : index
    %c8_82 = arith.constant 8 : index
    %c8_83 = arith.constant 8 : index
    %5 = scf.for %arg0 = %c0_75 to %c1_76 step %c8_77 iter_args(%arg1 = %4) -> (tensor<1x4x8xf32>) {
      %25 = scf.for %arg2 = %c0_78 to %c4_79 step %c8_80 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %26 = scf.for %arg4 = %c0_81 to %c8_82 step %c8_83 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %27 = affine.min #map(%arg0)
          %28 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %cst_50[0, %arg4] [8, 8] [1, 1] : tensor<8x8xf32> to tensor<8x8xf32>
          %extracted_slice_191 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %29 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<8x8xf32>) outs(%extracted_slice_191 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %30 = arith.mulf %in, %cst_48 : f32
            %31 = arith.addf %out, %30 : f32
            linalg.yield %31 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %29 into %arg5[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %26 : tensor<1x4x8xf32>
      }
      scf.yield %25 : tensor<1x4x8xf32>
    }
    %6 = tensor.empty() : tensor<1x4x8xf32>
    %c0_84 = arith.constant 0 : index
    %c1_85 = arith.constant 1 : index
    %c8_86 = arith.constant 8 : index
    %c0_87 = arith.constant 0 : index
    %c4_88 = arith.constant 4 : index
    %c8_89 = arith.constant 8 : index
    %c0_90 = arith.constant 0 : index
    %c8_91 = arith.constant 8 : index
    %c8_92 = arith.constant 8 : index
    %7 = scf.for %arg0 = %c0_84 to %c1_85 step %c8_86 iter_args(%arg1 = %6) -> (tensor<1x4x8xf32>) {
      %25 = scf.for %arg2 = %c0_87 to %c4_88 step %c8_89 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %26 = scf.for %arg4 = %c0_90 to %c8_91 step %c8_92 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %27 = affine.min #map(%arg0)
          %28 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %29 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_49 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %29 into %arg5[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %26 : tensor<1x4x8xf32>
      }
      scf.yield %25 : tensor<1x4x8xf32>
    }
    %c0_93 = arith.constant 0 : index
    %c1_94 = arith.constant 1 : index
    %c8_95 = arith.constant 8 : index
    %c0_96 = arith.constant 0 : index
    %c4_97 = arith.constant 4 : index
    %c8_98 = arith.constant 8 : index
    %c0_99 = arith.constant 0 : index
    %c8_100 = arith.constant 8 : index
    %c8_101 = arith.constant 8 : index
    %8 = scf.for %arg0 = %c0_93 to %c1_94 step %c8_95 iter_args(%arg1 = %7) -> (tensor<1x4x8xf32>) {
      %25 = scf.for %arg2 = %c0_96 to %c4_97 step %c8_98 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %26 = scf.for %arg4 = %c0_99 to %c8_100 step %c8_101 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %27 = affine.min #map(%arg0)
          %28 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %cst_50[0, %arg4] [8, 8] [1, 1] : tensor<8x8xf32> to tensor<8x8xf32>
          %extracted_slice_191 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %29 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<8x8xf32>) outs(%extracted_slice_191 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %30 = arith.mulf %in, %cst_48 : f32
            %31 = arith.addf %out, %30 : f32
            linalg.yield %31 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %29 into %arg5[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %26 : tensor<1x4x8xf32>
      }
      scf.yield %25 : tensor<1x4x8xf32>
    }
    %9 = tensor.empty() : tensor<1x4x4xf32>
    %c0_102 = arith.constant 0 : index
    %c1_103 = arith.constant 1 : index
    %c8_104 = arith.constant 8 : index
    %c0_105 = arith.constant 0 : index
    %c4_106 = arith.constant 4 : index
    %c8_107 = arith.constant 8 : index
    %c0_108 = arith.constant 0 : index
    %c4_109 = arith.constant 4 : index
    %c8_110 = arith.constant 8 : index
    %10 = scf.for %arg0 = %c0_102 to %c1_103 step %c8_104 iter_args(%arg1 = %9) -> (tensor<1x4x4xf32>) {
      %25 = scf.for %arg2 = %c0_105 to %c4_106 step %c8_107 iter_args(%arg3 = %arg1) -> (tensor<1x4x4xf32>) {
        %26 = scf.for %arg4 = %c0_108 to %c4_109 step %c8_110 iter_args(%arg5 = %arg3) -> (tensor<1x4x4xf32>) {
          %27 = affine.min #map(%arg0)
          %28 = affine.min #map1(%arg2)
          %29 = affine.min #map1(%arg4)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%27, %28, %29] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x?xf32>
          %30 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x?xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_49 : f32
          } -> tensor<?x?x?xf32>
          %inserted_slice = tensor.insert_slice %30 into %arg5[%arg0, %arg2, %arg4] [%27, %28, %29] [1, 1, 1] : tensor<?x?x?xf32> into tensor<1x4x4xf32>
          scf.yield %inserted_slice : tensor<1x4x4xf32>
        }
        scf.yield %26 : tensor<1x4x4xf32>
      }
      scf.yield %25 : tensor<1x4x4xf32>
    }
    %c0_111 = arith.constant 0 : index
    %c1_112 = arith.constant 1 : index
    %c8_113 = arith.constant 8 : index
    %c0_114 = arith.constant 0 : index
    %c4_115 = arith.constant 4 : index
    %c8_116 = arith.constant 8 : index
    %c0_117 = arith.constant 0 : index
    %c4_118 = arith.constant 4 : index
    %c8_119 = arith.constant 8 : index
    %11 = scf.for %arg0 = %c0_111 to %c1_112 step %c8_113 iter_args(%arg1 = %10) -> (tensor<1x4x4xf32>) {
      %25 = scf.for %arg2 = %c0_114 to %c4_115 step %c8_116 iter_args(%arg3 = %arg1) -> (tensor<1x4x4xf32>) {
        %26 = scf.for %arg4 = %c0_117 to %c4_118 step %c8_119 iter_args(%arg5 = %arg3) -> (tensor<1x4x4xf32>) {
          %27 = affine.min #map(%arg0)
          %28 = affine.min #map1(%arg2)
          %29 = affine.min #map(%arg0)
          %30 = affine.min #map1(%arg4)
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %33 = affine.min #map1(%arg4)
          %extracted_slice = tensor.extract_slice %2[%arg0, %arg2, 0] [%27, %28, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_191 = tensor.extract_slice %5[%arg0, %arg4, 0] [%29, %30, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_192 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, %33] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x?xf32>
          %34 = linalg.generic {indexing_maps = [#map5, #map6, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_191 : tensor<?x?x8xf32>, tensor<?x?x8xf32>) outs(%extracted_slice_192 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %in_193: f32, %out: f32):
            %35 = arith.mulf %in, %in_193 : f32
            %36 = arith.addf %out, %35 : f32
            linalg.yield %36 : f32
          } -> tensor<?x?x?xf32>
          %inserted_slice = tensor.insert_slice %34 into %arg5[%arg0, %arg2, %arg4] [%31, %32, %33] [1, 1, 1] : tensor<?x?x?xf32> into tensor<1x4x4xf32>
          scf.yield %inserted_slice : tensor<1x4x4xf32>
        }
        scf.yield %26 : tensor<1x4x4xf32>
      }
      scf.yield %25 : tensor<1x4x4xf32>
    }
    %12 = tensor.empty() : tensor<1x4x8xf32>
    %c0_120 = arith.constant 0 : index
    %c1_121 = arith.constant 1 : index
    %c8_122 = arith.constant 8 : index
    %c0_123 = arith.constant 0 : index
    %c4_124 = arith.constant 4 : index
    %c8_125 = arith.constant 8 : index
    %c0_126 = arith.constant 0 : index
    %c8_127 = arith.constant 8 : index
    %c8_128 = arith.constant 8 : index
    %13 = scf.for %arg0 = %c0_120 to %c1_121 step %c8_122 iter_args(%arg1 = %12) -> (tensor<1x4x8xf32>) {
      %25 = scf.for %arg2 = %c0_123 to %c4_124 step %c8_125 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %26 = scf.for %arg4 = %c0_126 to %c8_127 step %c8_128 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %27 = affine.min #map(%arg0)
          %28 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %29 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_49 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %29 into %arg5[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %26 : tensor<1x4x8xf32>
      }
      scf.yield %25 : tensor<1x4x8xf32>
    }
    %c0_129 = arith.constant 0 : index
    %c1_130 = arith.constant 1 : index
    %c8_131 = arith.constant 8 : index
    %c0_132 = arith.constant 0 : index
    %c4_133 = arith.constant 4 : index
    %c8_134 = arith.constant 8 : index
    %c0_135 = arith.constant 0 : index
    %c8_136 = arith.constant 8 : index
    %c8_137 = arith.constant 8 : index
    %14 = scf.for %arg0 = %c0_129 to %c1_130 step %c8_131 iter_args(%arg1 = %13) -> (tensor<1x4x8xf32>) {
      %25 = scf.for %arg2 = %c0_132 to %c4_133 step %c8_134 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %26 = scf.for %arg4 = %c0_135 to %c8_136 step %c8_137 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %27 = affine.min #map(%arg0)
          %28 = affine.min #map1(%arg2)
          %29 = affine.min #map(%arg0)
          %30 = affine.min #map(%arg0)
          %31 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %11[%arg0, %arg2, 0] [%27, %28, 4] [1, 1, 1] : tensor<1x4x4xf32> to tensor<?x?x4xf32>
          %extracted_slice_191 = tensor.extract_slice %8[%arg0, 0, %arg4] [%29, 4, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x4x8xf32>
          %extracted_slice_192 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%30, %31, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %32 = linalg.generic {indexing_maps = [#map5, #map7, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_191 : tensor<?x?x4xf32>, tensor<?x4x8xf32>) outs(%extracted_slice_192 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %in_193: f32, %out: f32):
            %33 = arith.divf %in, %cst_47 : f32
            %34 = arith.mulf %33, %in_193 : f32
            %35 = arith.addf %out, %34 : f32
            linalg.yield %35 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %32 into %arg5[%arg0, %arg2, %arg4] [%30, %31, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %26 : tensor<1x4x8xf32>
      }
      scf.yield %25 : tensor<1x4x8xf32>
    }
    %15 = tensor.empty() : tensor<1x4x8xf32>
    %c0_138 = arith.constant 0 : index
    %c1_139 = arith.constant 1 : index
    %c8_140 = arith.constant 8 : index
    %c0_141 = arith.constant 0 : index
    %c4_142 = arith.constant 4 : index
    %c8_143 = arith.constant 8 : index
    %c0_144 = arith.constant 0 : index
    %c8_145 = arith.constant 8 : index
    %c8_146 = arith.constant 8 : index
    %16 = scf.for %arg0 = %c0_138 to %c1_139 step %c8_140 iter_args(%arg1 = %15) -> (tensor<1x4x8xf32>) {
      %25 = scf.for %arg2 = %c0_141 to %c4_142 step %c8_143 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %26 = scf.for %arg4 = %c0_144 to %c8_145 step %c8_146 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %27 = affine.min #map(%arg0)
          %28 = affine.min #map1(%arg2)
          %29 = affine.min #map(%arg0)
          %30 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %14[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_191 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%29, %30, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %31 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<?x?x8xf32>) outs(%extracted_slice_191 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %32 = arith.addf %in, %cst_48 : f32
            linalg.yield %32 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %31 into %arg5[%arg0, %arg2, %arg4] [%29, %30, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %26 : tensor<1x4x8xf32>
      }
      scf.yield %25 : tensor<1x4x8xf32>
    }
    %17 = tensor.empty() : tensor<1x4x32xf32>
    %c0_147 = arith.constant 0 : index
    %c1_148 = arith.constant 1 : index
    %c8_149 = arith.constant 8 : index
    %c0_150 = arith.constant 0 : index
    %c4_151 = arith.constant 4 : index
    %c8_152 = arith.constant 8 : index
    %c0_153 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c8_154 = arith.constant 8 : index
    %18 = scf.for %arg0 = %c0_147 to %c1_148 step %c8_149 iter_args(%arg1 = %17) -> (tensor<1x4x32xf32>) {
      %25 = scf.for %arg2 = %c0_150 to %c4_151 step %c8_152 iter_args(%arg3 = %arg1) -> (tensor<1x4x32xf32>) {
        %26 = scf.for %arg4 = %c0_153 to %c32 step %c8_154 iter_args(%arg5 = %arg3) -> (tensor<1x4x32xf32>) {
          %27 = affine.min #map(%arg0)
          %28 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<1x4x32xf32> to tensor<?x?x8xf32>
          %29 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_49 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %29 into %arg5[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x32xf32>
          scf.yield %inserted_slice : tensor<1x4x32xf32>
        }
        scf.yield %26 : tensor<1x4x32xf32>
      }
      scf.yield %25 : tensor<1x4x32xf32>
    }
    %c0_155 = arith.constant 0 : index
    %c1_156 = arith.constant 1 : index
    %c8_157 = arith.constant 8 : index
    %c0_158 = arith.constant 0 : index
    %c4_159 = arith.constant 4 : index
    %c8_160 = arith.constant 8 : index
    %c0_161 = arith.constant 0 : index
    %c32_162 = arith.constant 32 : index
    %c8_163 = arith.constant 8 : index
    %19 = scf.for %arg0 = %c0_155 to %c1_156 step %c8_157 iter_args(%arg1 = %18) -> (tensor<1x4x32xf32>) {
      %25 = scf.for %arg2 = %c0_158 to %c4_159 step %c8_160 iter_args(%arg3 = %arg1) -> (tensor<1x4x32xf32>) {
        %26 = scf.for %arg4 = %c0_161 to %c32_162 step %c8_163 iter_args(%arg5 = %arg3) -> (tensor<1x4x32xf32>) {
          %27 = affine.min #map(%arg0)
          %28 = affine.min #map1(%arg2)
          %29 = affine.min #map(%arg0)
          %30 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %16[%arg0, %arg2, 0] [%27, %28, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_191 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%29, %30, 8] [1, 1, 1] : tensor<1x4x32xf32> to tensor<?x?x8xf32>
          %31 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<?x?x8xf32>) outs(%extracted_slice_191 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %32 = arith.mulf %in, %cst : f32
            %33 = arith.addf %out, %32 : f32
            linalg.yield %33 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %31 into %arg5[%arg0, %arg2, %arg4] [%29, %30, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x32xf32>
          scf.yield %inserted_slice : tensor<1x4x32xf32>
        }
        scf.yield %26 : tensor<1x4x32xf32>
      }
      scf.yield %25 : tensor<1x4x32xf32>
    }
    %20 = tensor.empty() : tensor<1x4x8xf32>
    %c0_164 = arith.constant 0 : index
    %c1_165 = arith.constant 1 : index
    %c8_166 = arith.constant 8 : index
    %c0_167 = arith.constant 0 : index
    %c4_168 = arith.constant 4 : index
    %c8_169 = arith.constant 8 : index
    %c0_170 = arith.constant 0 : index
    %c8_171 = arith.constant 8 : index
    %c8_172 = arith.constant 8 : index
    %21 = scf.for %arg0 = %c0_164 to %c1_165 step %c8_166 iter_args(%arg1 = %20) -> (tensor<1x4x8xf32>) {
      %25 = scf.for %arg2 = %c0_167 to %c4_168 step %c8_169 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %26 = scf.for %arg4 = %c0_170 to %c8_171 step %c8_172 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %27 = affine.min #map(%arg0)
          %28 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %29 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_49 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %29 into %arg5[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %26 : tensor<1x4x8xf32>
      }
      scf.yield %25 : tensor<1x4x8xf32>
    }
    %c0_173 = arith.constant 0 : index
    %c1_174 = arith.constant 1 : index
    %c8_175 = arith.constant 8 : index
    %c0_176 = arith.constant 0 : index
    %c4_177 = arith.constant 4 : index
    %c8_178 = arith.constant 8 : index
    %c0_179 = arith.constant 0 : index
    %c8_180 = arith.constant 8 : index
    %c8_181 = arith.constant 8 : index
    %22 = scf.for %arg0 = %c0_173 to %c1_174 step %c8_175 iter_args(%arg1 = %21) -> (tensor<1x4x8xf32>) {
      %25 = scf.for %arg2 = %c0_176 to %c4_177 step %c8_178 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %26 = scf.for %arg4 = %c0_179 to %c8_180 step %c8_181 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %27 = affine.min #map(%arg0)
          %28 = affine.min #map1(%arg2)
          %29 = affine.min #map(%arg0)
          %30 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %19[%arg0, %arg2, 0] [%27, %28, 32] [1, 1, 1] : tensor<1x4x32xf32> to tensor<?x?x32xf32>
          %extracted_slice_191 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%29, %30, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %31 = linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<?x?x32xf32>) outs(%extracted_slice_191 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %32 = arith.maximumf %in, %cst_49 : f32
            %33 = arith.mulf %32, %cst : f32
            %34 = arith.addf %out, %33 : f32
            linalg.yield %34 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %31 into %arg5[%arg0, %arg2, %arg4] [%29, %30, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %26 : tensor<1x4x8xf32>
      }
      scf.yield %25 : tensor<1x4x8xf32>
    }
    %23 = tensor.empty() : tensor<1x4x8xf32>
    %c0_182 = arith.constant 0 : index
    %c1_183 = arith.constant 1 : index
    %c8_184 = arith.constant 8 : index
    %c0_185 = arith.constant 0 : index
    %c4_186 = arith.constant 4 : index
    %c8_187 = arith.constant 8 : index
    %c0_188 = arith.constant 0 : index
    %c8_189 = arith.constant 8 : index
    %c8_190 = arith.constant 8 : index
    %24 = scf.for %arg0 = %c0_182 to %c1_183 step %c8_184 iter_args(%arg1 = %23) -> (tensor<1x4x8xf32>) {
      %25 = scf.for %arg2 = %c0_185 to %c4_186 step %c8_187 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %26 = scf.for %arg4 = %c0_188 to %c8_189 step %c8_190 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %27 = affine.min #map(%arg0)
          %28 = affine.min #map1(%arg2)
          %29 = affine.min #map(%arg0)
          %30 = affine.min #map1(%arg2)
          %31 = affine.min #map(%arg0)
          %32 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %16[%arg0, %arg2, %arg4] [%27, %28, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_191 = tensor.extract_slice %22[%arg0, %arg2, %arg4] [%29, %30, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_192 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %33 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_191 : tensor<?x?x8xf32>, tensor<?x?x8xf32>) outs(%extracted_slice_192 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %in_193: f32, %out: f32):
            %34 = arith.addf %in, %in_193 : f32
            %35 = arith.mulf %34, %34 : f32
            %36 = arith.addf %35, %35 : f32
            linalg.yield %36 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %33 into %arg5[%arg0, %arg2, %arg4] [%31, %32, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %26 : tensor<1x4x8xf32>
      }
      scf.yield %25 : tensor<1x4x8xf32>
    }
    return %24 : tensor<1x4x8xf32>
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
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
module {
  memref.global "private" constant @__constant_8x8xf32 : memref<8x8xf32> = dense<1.000000e-01> {alignment = 64 : i64}
  func.func @main() -> memref<1x4x8xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 2.828400e+00 : f32
    %cst_2 = arith.constant 2.000000e-01 : f32
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
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloc_3 : memref<1x4x8xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%0 : memref<8x8xf32>) outs(%alloc_3 : memref<1x4x8xf32>) {
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
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x4x4xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloc_5 : memref<1x4x4xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    linalg.generic {indexing_maps = [#map3, #map4, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%alloc, %alloc_3 : memref<1x4x8xf32>, memref<1x4x8xf32>) outs(%alloc_5 : memref<1x4x4xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %1 = arith.mulf %in, %in_11 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloc_6 : memref<1x4x8xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    linalg.generic {indexing_maps = [#map3, #map5, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%alloc_5, %alloc_4 : memref<1x4x4xf32>, memref<1x4x8xf32>) outs(%alloc_6 : memref<1x4x8xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %1 = arith.divf %in, %cst_1 : f32
      %2 = arith.mulf %1, %in_11 : f32
      %3 = arith.addf %out, %2 : f32
      linalg.yield %3 : f32
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_6 : memref<1x4x8xf32>) outs(%alloc_7 : memref<1x4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %cst_0 : f32
      linalg.yield %1 : f32
    }
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x4x32xf32>
    scf.for %arg0 = %c0 to %c32 step %c8 {
      %subview = memref.subview %alloc_8[0, 0, %arg0] [1, 4, 8] [1, 1, 1] : memref<1x4x32xf32> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      }
      memref.copy %subview, %subview : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
    }
    scf.for %arg0 = %c0 to %c32 step %c8 {
      %subview = memref.subview %alloc_8[0, 0, %arg0] [1, 4, 8] [1, 1, 1] : memref<1x4x32xf32> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%alloc_7 : memref<1x4x8xf32>) outs(%subview : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        %1 = arith.mulf %in, %cst_2 : f32
        %2 = arith.addf %out, %1 : f32
        linalg.yield %2 : f32
      }
      memref.copy %subview, %subview : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloc_9 : memref<1x4x8xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%alloc_8 : memref<1x4x32xf32>) outs(%alloc_9 : memref<1x4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.maximumf %in, %cst : f32
      %2 = arith.mulf %1, %cst_2 : f32
      %3 = arith.addf %out, %2 : f32
      linalg.yield %3 : f32
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_7, %alloc_9 : memref<1x4x8xf32>, memref<1x4x8xf32>) outs(%alloc_10 : memref<1x4x8xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %1 = arith.addf %in, %in_11 : f32
      %2 = arith.mulf %1, %1 : f32
      %3 = arith.addf %2, %2 : f32
      linalg.yield %3 : f32
    }
    return %alloc_10 : memref<1x4x8xf32>
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
    %cst_2 = arith.constant 2.000000e-01 : f32
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
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          memref.store %cst, %alloc_3[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
        }
      }
    }
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            %1 = memref.load %0[%arg3, %arg2] : memref<8x8xf32>
            %2 = memref.load %alloc_3[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
            %3 = arith.mulf %1, %cst_0 : f32
            %4 = arith.addf %2, %3 : f32
            memref.store %4, %alloc_3[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
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
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x4x4xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c4 step %c1 {
          memref.store %cst, %alloc_5[%arg0, %arg1, %arg2] : memref<1x4x4xf32>
        }
      }
    }
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c4 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            %1 = memref.load %alloc[%arg0, %arg1, %arg3] : memref<1x4x8xf32>
            %2 = memref.load %alloc_3[%arg0, %arg2, %arg3] : memref<1x4x8xf32>
            %3 = memref.load %alloc_5[%arg0, %arg1, %arg2] : memref<1x4x4xf32>
            %4 = arith.mulf %1, %2 : f32
            %5 = arith.addf %3, %4 : f32
            memref.store %5, %alloc_5[%arg0, %arg1, %arg2] : memref<1x4x4xf32>
          }
        }
      }
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          memref.store %cst, %alloc_6[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
        }
      }
    }
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          scf.for %arg3 = %c0 to %c4 step %c1 {
            %1 = memref.load %alloc_5[%arg0, %arg1, %arg3] : memref<1x4x4xf32>
            %2 = memref.load %alloc_4[%arg0, %arg3, %arg2] : memref<1x4x8xf32>
            %3 = memref.load %alloc_6[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
            %4 = arith.divf %1, %cst_1 : f32
            %5 = arith.mulf %4, %2 : f32
            %6 = arith.addf %3, %5 : f32
            memref.store %6, %alloc_6[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
          }
        }
      }
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          %1 = memref.load %alloc_6[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
          %2 = arith.addf %1, %cst_0 : f32
          memref.store %2, %alloc_7[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
        }
      }
    }
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x4x32xf32>
    scf.for %arg0 = %c0 to %c32 step %c8 {
      %subview = memref.subview %alloc_8[0, 0, %arg0] [1, 4, 8] [1, 1, 1] : memref<1x4x32xf32> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
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
      %subview = memref.subview %alloc_8[0, 0, %arg0] [1, 4, 8] [1, 1, 1] : memref<1x4x32xf32> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c4 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            scf.for %arg4 = %c0 to %c8 step %c1 {
              %1 = memref.load %alloc_7[%arg1, %arg2, %arg4] : memref<1x4x8xf32>
              %2 = memref.load %subview[%arg1, %arg2, %arg3] : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
              %3 = arith.mulf %1, %cst_2 : f32
              %4 = arith.addf %2, %3 : f32
              memref.store %4, %subview[%arg1, %arg2, %arg3] : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
            }
          }
        }
      }
      memref.copy %subview, %subview : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          memref.store %cst, %alloc_9[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
        }
      }
    }
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          scf.for %arg3 = %c0 to %c32 step %c1 {
            %1 = memref.load %alloc_8[%arg0, %arg1, %arg3] : memref<1x4x32xf32>
            %2 = memref.load %alloc_9[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
            %3 = arith.maximumf %1, %cst : f32
            %4 = arith.mulf %3, %cst_2 : f32
            %5 = arith.addf %2, %4 : f32
            memref.store %5, %alloc_9[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
          }
        }
      }
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          %1 = memref.load %alloc_7[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
          %2 = memref.load %alloc_9[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
          %3 = arith.addf %1, %2 : f32
          %4 = arith.mulf %3, %3 : f32
          %5 = arith.addf %4, %4 : f32
          memref.store %5, %alloc_10[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
        }
      }
    }
    return %alloc_10 : memref<1x4x8xf32>
  }
}

// ==========================================
// Phase: linalg to scf
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
    %5 = llvm.mlir.constant(2.000000e-01 : f32) : f32
    %6 = llvm.mlir.constant(8 : index) : i64
    %7 = llvm.mlir.constant(0 : index) : i64
    %8 = llvm.mlir.constant(32 : index) : i64
    %9 = llvm.mlir.constant(8 : index) : i64
    %10 = llvm.mlir.constant(8 : index) : i64
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.mlir.constant(64 : index) : i64
    %13 = llvm.mlir.zero : !llvm.ptr
    %14 = llvm.getelementptr %13[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %15 = llvm.ptrtoint %14 : !llvm.ptr to i64
    %16 = llvm.mlir.addressof @__constant_8x8xf32 : !llvm.ptr
    %17 = llvm.getelementptr %16[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x array<8 x f32>>
    %18 = llvm.mlir.constant(3735928559 : index) : i64
    %19 = llvm.inttoptr %18 : i64 to !llvm.ptr
    %20 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %21 = llvm.insertvalue %19, %20[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %17, %21[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.mlir.constant(0 : index) : i64
    %24 = llvm.insertvalue %23, %22[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.insertvalue %9, %24[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.insertvalue %10, %25[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %10, %26[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %11, %27[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.mlir.constant(1 : index) : i64
    %30 = llvm.mlir.constant(4 : index) : i64
    %31 = llvm.mlir.constant(8 : index) : i64
    %32 = llvm.mlir.constant(1 : index) : i64
    %33 = llvm.mlir.constant(32 : index) : i64
    %34 = llvm.mlir.constant(32 : index) : i64
    %35 = llvm.mlir.zero : !llvm.ptr
    %36 = llvm.getelementptr %35[%34] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %37 = llvm.ptrtoint %36 : !llvm.ptr to i64
    %38 = llvm.mlir.constant(64 : index) : i64
    %39 = llvm.add %37, %38 : i64
    %40 = llvm.call @malloc(%39) : (i64) -> !llvm.ptr
    %41 = llvm.ptrtoint %40 : !llvm.ptr to i64
    %42 = llvm.mlir.constant(1 : index) : i64
    %43 = llvm.sub %38, %42 : i64
    %44 = llvm.add %41, %43 : i64
    %45 = llvm.urem %44, %38  : i64
    %46 = llvm.sub %44, %45 : i64
    %47 = llvm.inttoptr %46 : i64 to !llvm.ptr
    %48 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %49 = llvm.insertvalue %40, %48[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %50 = llvm.insertvalue %47, %49[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %51 = llvm.mlir.constant(0 : index) : i64
    %52 = llvm.insertvalue %51, %50[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %53 = llvm.insertvalue %29, %52[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %54 = llvm.insertvalue %30, %53[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %55 = llvm.insertvalue %31, %54[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %56 = llvm.insertvalue %33, %55[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %57 = llvm.insertvalue %31, %56[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %58 = llvm.insertvalue %32, %57[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb1(%7 : i64)
  ^bb1(%59: i64):  // 2 preds: ^bb0, ^bb8
    %60 = llvm.icmp "slt" %59, %1 : i64
    llvm.cond_br %60, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%7 : i64)
  ^bb3(%61: i64):  // 2 preds: ^bb2, ^bb7
    %62 = llvm.icmp "slt" %61, %0 : i64
    llvm.cond_br %62, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%7 : i64)
  ^bb5(%63: i64):  // 2 preds: ^bb4, ^bb6
    %64 = llvm.icmp "slt" %63, %6 : i64
    llvm.cond_br %64, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %65 = llvm.extractvalue %58[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %66 = llvm.mlir.constant(32 : index) : i64
    %67 = llvm.mul %59, %66 : i64
    %68 = llvm.mlir.constant(8 : index) : i64
    %69 = llvm.mul %61, %68 : i64
    %70 = llvm.add %67, %69 : i64
    %71 = llvm.add %70, %63 : i64
    %72 = llvm.getelementptr %65[%71] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %72 : f32, !llvm.ptr
    %73 = llvm.add %63, %1 : i64
    llvm.br ^bb5(%73 : i64)
  ^bb7:  // pred: ^bb5
    %74 = llvm.add %61, %1 : i64
    llvm.br ^bb3(%74 : i64)
  ^bb8:  // pred: ^bb3
    %75 = llvm.add %59, %1 : i64
    llvm.br ^bb1(%75 : i64)
  ^bb9:  // pred: ^bb1
    llvm.br ^bb10(%7 : i64)
  ^bb10(%76: i64):  // 2 preds: ^bb9, ^bb20
    %77 = llvm.icmp "slt" %76, %1 : i64
    llvm.cond_br %77, ^bb11, ^bb21
  ^bb11:  // pred: ^bb10
    llvm.br ^bb12(%7 : i64)
  ^bb12(%78: i64):  // 2 preds: ^bb11, ^bb19
    %79 = llvm.icmp "slt" %78, %0 : i64
    llvm.cond_br %79, ^bb13, ^bb20
  ^bb13:  // pred: ^bb12
    llvm.br ^bb14(%7 : i64)
  ^bb14(%80: i64):  // 2 preds: ^bb13, ^bb18
    %81 = llvm.icmp "slt" %80, %6 : i64
    llvm.cond_br %81, ^bb15, ^bb19
  ^bb15:  // pred: ^bb14
    llvm.br ^bb16(%7 : i64)
  ^bb16(%82: i64):  // 2 preds: ^bb15, ^bb17
    %83 = llvm.icmp "slt" %82, %6 : i64
    llvm.cond_br %83, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %84 = llvm.extractvalue %28[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %85 = llvm.mlir.constant(8 : index) : i64
    %86 = llvm.mul %82, %85 : i64
    %87 = llvm.add %86, %80 : i64
    %88 = llvm.getelementptr %84[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %89 = llvm.load %88 : !llvm.ptr -> f32
    %90 = llvm.extractvalue %58[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %91 = llvm.mlir.constant(32 : index) : i64
    %92 = llvm.mul %76, %91 : i64
    %93 = llvm.mlir.constant(8 : index) : i64
    %94 = llvm.mul %78, %93 : i64
    %95 = llvm.add %92, %94 : i64
    %96 = llvm.add %95, %80 : i64
    %97 = llvm.getelementptr %90[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %98 = llvm.load %97 : !llvm.ptr -> f32
    %99 = llvm.fmul %89, %3  : f32
    %100 = llvm.fadd %98, %99  : f32
    %101 = llvm.extractvalue %58[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %102 = llvm.mlir.constant(32 : index) : i64
    %103 = llvm.mul %76, %102 : i64
    %104 = llvm.mlir.constant(8 : index) : i64
    %105 = llvm.mul %78, %104 : i64
    %106 = llvm.add %103, %105 : i64
    %107 = llvm.add %106, %80 : i64
    %108 = llvm.getelementptr %101[%107] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %100, %108 : f32, !llvm.ptr
    %109 = llvm.add %82, %1 : i64
    llvm.br ^bb16(%109 : i64)
  ^bb18:  // pred: ^bb16
    %110 = llvm.add %80, %1 : i64
    llvm.br ^bb14(%110 : i64)
  ^bb19:  // pred: ^bb14
    %111 = llvm.add %78, %1 : i64
    llvm.br ^bb12(%111 : i64)
  ^bb20:  // pred: ^bb12
    %112 = llvm.add %76, %1 : i64
    llvm.br ^bb10(%112 : i64)
  ^bb21:  // pred: ^bb10
    %113 = llvm.mlir.constant(1 : index) : i64
    %114 = llvm.mlir.constant(4 : index) : i64
    %115 = llvm.mlir.constant(8 : index) : i64
    %116 = llvm.mlir.constant(1 : index) : i64
    %117 = llvm.mlir.constant(32 : index) : i64
    %118 = llvm.mlir.constant(32 : index) : i64
    %119 = llvm.mlir.zero : !llvm.ptr
    %120 = llvm.getelementptr %119[%118] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %121 = llvm.ptrtoint %120 : !llvm.ptr to i64
    %122 = llvm.mlir.constant(64 : index) : i64
    %123 = llvm.add %121, %122 : i64
    %124 = llvm.call @malloc(%123) : (i64) -> !llvm.ptr
    %125 = llvm.ptrtoint %124 : !llvm.ptr to i64
    %126 = llvm.mlir.constant(1 : index) : i64
    %127 = llvm.sub %122, %126 : i64
    %128 = llvm.add %125, %127 : i64
    %129 = llvm.urem %128, %122  : i64
    %130 = llvm.sub %128, %129 : i64
    %131 = llvm.inttoptr %130 : i64 to !llvm.ptr
    %132 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %133 = llvm.insertvalue %124, %132[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %134 = llvm.insertvalue %131, %133[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %135 = llvm.mlir.constant(0 : index) : i64
    %136 = llvm.insertvalue %135, %134[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %137 = llvm.insertvalue %113, %136[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %138 = llvm.insertvalue %114, %137[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %139 = llvm.insertvalue %115, %138[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %140 = llvm.insertvalue %117, %139[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %141 = llvm.insertvalue %115, %140[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %142 = llvm.insertvalue %116, %141[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb22(%7 : i64)
  ^bb22(%143: i64):  // 2 preds: ^bb21, ^bb29
    %144 = llvm.icmp "slt" %143, %1 : i64
    llvm.cond_br %144, ^bb23, ^bb30
  ^bb23:  // pred: ^bb22
    llvm.br ^bb24(%7 : i64)
  ^bb24(%145: i64):  // 2 preds: ^bb23, ^bb28
    %146 = llvm.icmp "slt" %145, %0 : i64
    llvm.cond_br %146, ^bb25, ^bb29
  ^bb25:  // pred: ^bb24
    llvm.br ^bb26(%7 : i64)
  ^bb26(%147: i64):  // 2 preds: ^bb25, ^bb27
    %148 = llvm.icmp "slt" %147, %6 : i64
    llvm.cond_br %148, ^bb27, ^bb28
  ^bb27:  // pred: ^bb26
    %149 = llvm.extractvalue %142[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %150 = llvm.mlir.constant(32 : index) : i64
    %151 = llvm.mul %143, %150 : i64
    %152 = llvm.mlir.constant(8 : index) : i64
    %153 = llvm.mul %145, %152 : i64
    %154 = llvm.add %151, %153 : i64
    %155 = llvm.add %154, %147 : i64
    %156 = llvm.getelementptr %149[%155] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %156 : f32, !llvm.ptr
    %157 = llvm.add %147, %1 : i64
    llvm.br ^bb26(%157 : i64)
  ^bb28:  // pred: ^bb26
    %158 = llvm.add %145, %1 : i64
    llvm.br ^bb24(%158 : i64)
  ^bb29:  // pred: ^bb24
    %159 = llvm.add %143, %1 : i64
    llvm.br ^bb22(%159 : i64)
  ^bb30:  // pred: ^bb22
    llvm.br ^bb31(%7 : i64)
  ^bb31(%160: i64):  // 2 preds: ^bb30, ^bb41
    %161 = llvm.icmp "slt" %160, %1 : i64
    llvm.cond_br %161, ^bb32, ^bb42
  ^bb32:  // pred: ^bb31
    llvm.br ^bb33(%7 : i64)
  ^bb33(%162: i64):  // 2 preds: ^bb32, ^bb40
    %163 = llvm.icmp "slt" %162, %0 : i64
    llvm.cond_br %163, ^bb34, ^bb41
  ^bb34:  // pred: ^bb33
    llvm.br ^bb35(%7 : i64)
  ^bb35(%164: i64):  // 2 preds: ^bb34, ^bb39
    %165 = llvm.icmp "slt" %164, %6 : i64
    llvm.cond_br %165, ^bb36, ^bb40
  ^bb36:  // pred: ^bb35
    llvm.br ^bb37(%7 : i64)
  ^bb37(%166: i64):  // 2 preds: ^bb36, ^bb38
    %167 = llvm.icmp "slt" %166, %6 : i64
    llvm.cond_br %167, ^bb38, ^bb39
  ^bb38:  // pred: ^bb37
    %168 = llvm.extractvalue %28[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %169 = llvm.mlir.constant(8 : index) : i64
    %170 = llvm.mul %166, %169 : i64
    %171 = llvm.add %170, %164 : i64
    %172 = llvm.getelementptr %168[%171] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %173 = llvm.load %172 : !llvm.ptr -> f32
    %174 = llvm.extractvalue %142[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %175 = llvm.mlir.constant(32 : index) : i64
    %176 = llvm.mul %160, %175 : i64
    %177 = llvm.mlir.constant(8 : index) : i64
    %178 = llvm.mul %162, %177 : i64
    %179 = llvm.add %176, %178 : i64
    %180 = llvm.add %179, %164 : i64
    %181 = llvm.getelementptr %174[%180] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %182 = llvm.load %181 : !llvm.ptr -> f32
    %183 = llvm.fmul %173, %3  : f32
    %184 = llvm.fadd %182, %183  : f32
    %185 = llvm.extractvalue %142[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %186 = llvm.mlir.constant(32 : index) : i64
    %187 = llvm.mul %160, %186 : i64
    %188 = llvm.mlir.constant(8 : index) : i64
    %189 = llvm.mul %162, %188 : i64
    %190 = llvm.add %187, %189 : i64
    %191 = llvm.add %190, %164 : i64
    %192 = llvm.getelementptr %185[%191] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %184, %192 : f32, !llvm.ptr
    %193 = llvm.add %166, %1 : i64
    llvm.br ^bb37(%193 : i64)
  ^bb39:  // pred: ^bb37
    %194 = llvm.add %164, %1 : i64
    llvm.br ^bb35(%194 : i64)
  ^bb40:  // pred: ^bb35
    %195 = llvm.add %162, %1 : i64
    llvm.br ^bb33(%195 : i64)
  ^bb41:  // pred: ^bb33
    %196 = llvm.add %160, %1 : i64
    llvm.br ^bb31(%196 : i64)
  ^bb42:  // pred: ^bb31
    %197 = llvm.mlir.constant(1 : index) : i64
    %198 = llvm.mlir.constant(4 : index) : i64
    %199 = llvm.mlir.constant(8 : index) : i64
    %200 = llvm.mlir.constant(1 : index) : i64
    %201 = llvm.mlir.constant(32 : index) : i64
    %202 = llvm.mlir.constant(32 : index) : i64
    %203 = llvm.mlir.zero : !llvm.ptr
    %204 = llvm.getelementptr %203[%202] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %205 = llvm.ptrtoint %204 : !llvm.ptr to i64
    %206 = llvm.mlir.constant(64 : index) : i64
    %207 = llvm.add %205, %206 : i64
    %208 = llvm.call @malloc(%207) : (i64) -> !llvm.ptr
    %209 = llvm.ptrtoint %208 : !llvm.ptr to i64
    %210 = llvm.mlir.constant(1 : index) : i64
    %211 = llvm.sub %206, %210 : i64
    %212 = llvm.add %209, %211 : i64
    %213 = llvm.urem %212, %206  : i64
    %214 = llvm.sub %212, %213 : i64
    %215 = llvm.inttoptr %214 : i64 to !llvm.ptr
    %216 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %217 = llvm.insertvalue %208, %216[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %218 = llvm.insertvalue %215, %217[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %219 = llvm.mlir.constant(0 : index) : i64
    %220 = llvm.insertvalue %219, %218[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %221 = llvm.insertvalue %197, %220[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %222 = llvm.insertvalue %198, %221[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %223 = llvm.insertvalue %199, %222[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %224 = llvm.insertvalue %201, %223[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %225 = llvm.insertvalue %199, %224[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %226 = llvm.insertvalue %200, %225[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb43(%7 : i64)
  ^bb43(%227: i64):  // 2 preds: ^bb42, ^bb50
    %228 = llvm.icmp "slt" %227, %1 : i64
    llvm.cond_br %228, ^bb44, ^bb51
  ^bb44:  // pred: ^bb43
    llvm.br ^bb45(%7 : i64)
  ^bb45(%229: i64):  // 2 preds: ^bb44, ^bb49
    %230 = llvm.icmp "slt" %229, %0 : i64
    llvm.cond_br %230, ^bb46, ^bb50
  ^bb46:  // pred: ^bb45
    llvm.br ^bb47(%7 : i64)
  ^bb47(%231: i64):  // 2 preds: ^bb46, ^bb48
    %232 = llvm.icmp "slt" %231, %6 : i64
    llvm.cond_br %232, ^bb48, ^bb49
  ^bb48:  // pred: ^bb47
    %233 = llvm.extractvalue %226[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %234 = llvm.mlir.constant(32 : index) : i64
    %235 = llvm.mul %227, %234 : i64
    %236 = llvm.mlir.constant(8 : index) : i64
    %237 = llvm.mul %229, %236 : i64
    %238 = llvm.add %235, %237 : i64
    %239 = llvm.add %238, %231 : i64
    %240 = llvm.getelementptr %233[%239] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %240 : f32, !llvm.ptr
    %241 = llvm.add %231, %1 : i64
    llvm.br ^bb47(%241 : i64)
  ^bb49:  // pred: ^bb47
    %242 = llvm.add %229, %1 : i64
    llvm.br ^bb45(%242 : i64)
  ^bb50:  // pred: ^bb45
    %243 = llvm.add %227, %1 : i64
    llvm.br ^bb43(%243 : i64)
  ^bb51:  // pred: ^bb43
    llvm.br ^bb52(%7 : i64)
  ^bb52(%244: i64):  // 2 preds: ^bb51, ^bb62
    %245 = llvm.icmp "slt" %244, %1 : i64
    llvm.cond_br %245, ^bb53, ^bb63
  ^bb53:  // pred: ^bb52
    llvm.br ^bb54(%7 : i64)
  ^bb54(%246: i64):  // 2 preds: ^bb53, ^bb61
    %247 = llvm.icmp "slt" %246, %0 : i64
    llvm.cond_br %247, ^bb55, ^bb62
  ^bb55:  // pred: ^bb54
    llvm.br ^bb56(%7 : i64)
  ^bb56(%248: i64):  // 2 preds: ^bb55, ^bb60
    %249 = llvm.icmp "slt" %248, %6 : i64
    llvm.cond_br %249, ^bb57, ^bb61
  ^bb57:  // pred: ^bb56
    llvm.br ^bb58(%7 : i64)
  ^bb58(%250: i64):  // 2 preds: ^bb57, ^bb59
    %251 = llvm.icmp "slt" %250, %6 : i64
    llvm.cond_br %251, ^bb59, ^bb60
  ^bb59:  // pred: ^bb58
    %252 = llvm.extractvalue %28[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %253 = llvm.mlir.constant(8 : index) : i64
    %254 = llvm.mul %250, %253 : i64
    %255 = llvm.add %254, %248 : i64
    %256 = llvm.getelementptr %252[%255] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %257 = llvm.load %256 : !llvm.ptr -> f32
    %258 = llvm.extractvalue %226[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %259 = llvm.mlir.constant(32 : index) : i64
    %260 = llvm.mul %244, %259 : i64
    %261 = llvm.mlir.constant(8 : index) : i64
    %262 = llvm.mul %246, %261 : i64
    %263 = llvm.add %260, %262 : i64
    %264 = llvm.add %263, %248 : i64
    %265 = llvm.getelementptr %258[%264] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %266 = llvm.load %265 : !llvm.ptr -> f32
    %267 = llvm.fmul %257, %3  : f32
    %268 = llvm.fadd %266, %267  : f32
    %269 = llvm.extractvalue %226[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %270 = llvm.mlir.constant(32 : index) : i64
    %271 = llvm.mul %244, %270 : i64
    %272 = llvm.mlir.constant(8 : index) : i64
    %273 = llvm.mul %246, %272 : i64
    %274 = llvm.add %271, %273 : i64
    %275 = llvm.add %274, %248 : i64
    %276 = llvm.getelementptr %269[%275] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %268, %276 : f32, !llvm.ptr
    %277 = llvm.add %250, %1 : i64
    llvm.br ^bb58(%277 : i64)
  ^bb60:  // pred: ^bb58
    %278 = llvm.add %248, %1 : i64
    llvm.br ^bb56(%278 : i64)
  ^bb61:  // pred: ^bb56
    %279 = llvm.add %246, %1 : i64
    llvm.br ^bb54(%279 : i64)
  ^bb62:  // pred: ^bb54
    %280 = llvm.add %244, %1 : i64
    llvm.br ^bb52(%280 : i64)
  ^bb63:  // pred: ^bb52
    %281 = llvm.mlir.constant(1 : index) : i64
    %282 = llvm.mlir.constant(4 : index) : i64
    %283 = llvm.mlir.constant(4 : index) : i64
    %284 = llvm.mlir.constant(1 : index) : i64
    %285 = llvm.mlir.constant(16 : index) : i64
    %286 = llvm.mlir.constant(16 : index) : i64
    %287 = llvm.mlir.zero : !llvm.ptr
    %288 = llvm.getelementptr %287[%286] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %289 = llvm.ptrtoint %288 : !llvm.ptr to i64
    %290 = llvm.mlir.constant(64 : index) : i64
    %291 = llvm.add %289, %290 : i64
    %292 = llvm.call @malloc(%291) : (i64) -> !llvm.ptr
    %293 = llvm.ptrtoint %292 : !llvm.ptr to i64
    %294 = llvm.mlir.constant(1 : index) : i64
    %295 = llvm.sub %290, %294 : i64
    %296 = llvm.add %293, %295 : i64
    %297 = llvm.urem %296, %290  : i64
    %298 = llvm.sub %296, %297 : i64
    %299 = llvm.inttoptr %298 : i64 to !llvm.ptr
    %300 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %301 = llvm.insertvalue %292, %300[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %302 = llvm.insertvalue %299, %301[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %303 = llvm.mlir.constant(0 : index) : i64
    %304 = llvm.insertvalue %303, %302[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %305 = llvm.insertvalue %281, %304[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %306 = llvm.insertvalue %282, %305[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %307 = llvm.insertvalue %283, %306[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %308 = llvm.insertvalue %285, %307[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %309 = llvm.insertvalue %283, %308[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %310 = llvm.insertvalue %284, %309[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb64(%7 : i64)
  ^bb64(%311: i64):  // 2 preds: ^bb63, ^bb71
    %312 = llvm.icmp "slt" %311, %1 : i64
    llvm.cond_br %312, ^bb65, ^bb72
  ^bb65:  // pred: ^bb64
    llvm.br ^bb66(%7 : i64)
  ^bb66(%313: i64):  // 2 preds: ^bb65, ^bb70
    %314 = llvm.icmp "slt" %313, %0 : i64
    llvm.cond_br %314, ^bb67, ^bb71
  ^bb67:  // pred: ^bb66
    llvm.br ^bb68(%7 : i64)
  ^bb68(%315: i64):  // 2 preds: ^bb67, ^bb69
    %316 = llvm.icmp "slt" %315, %0 : i64
    llvm.cond_br %316, ^bb69, ^bb70
  ^bb69:  // pred: ^bb68
    %317 = llvm.extractvalue %310[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %318 = llvm.mlir.constant(16 : index) : i64
    %319 = llvm.mul %311, %318 : i64
    %320 = llvm.mlir.constant(4 : index) : i64
    %321 = llvm.mul %313, %320 : i64
    %322 = llvm.add %319, %321 : i64
    %323 = llvm.add %322, %315 : i64
    %324 = llvm.getelementptr %317[%323] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %324 : f32, !llvm.ptr
    %325 = llvm.add %315, %1 : i64
    llvm.br ^bb68(%325 : i64)
  ^bb70:  // pred: ^bb68
    %326 = llvm.add %313, %1 : i64
    llvm.br ^bb66(%326 : i64)
  ^bb71:  // pred: ^bb66
    %327 = llvm.add %311, %1 : i64
    llvm.br ^bb64(%327 : i64)
  ^bb72:  // pred: ^bb64
    llvm.br ^bb73(%7 : i64)
  ^bb73(%328: i64):  // 2 preds: ^bb72, ^bb83
    %329 = llvm.icmp "slt" %328, %1 : i64
    llvm.cond_br %329, ^bb74, ^bb84
  ^bb74:  // pred: ^bb73
    llvm.br ^bb75(%7 : i64)
  ^bb75(%330: i64):  // 2 preds: ^bb74, ^bb82
    %331 = llvm.icmp "slt" %330, %0 : i64
    llvm.cond_br %331, ^bb76, ^bb83
  ^bb76:  // pred: ^bb75
    llvm.br ^bb77(%7 : i64)
  ^bb77(%332: i64):  // 2 preds: ^bb76, ^bb81
    %333 = llvm.icmp "slt" %332, %0 : i64
    llvm.cond_br %333, ^bb78, ^bb82
  ^bb78:  // pred: ^bb77
    llvm.br ^bb79(%7 : i64)
  ^bb79(%334: i64):  // 2 preds: ^bb78, ^bb80
    %335 = llvm.icmp "slt" %334, %6 : i64
    llvm.cond_br %335, ^bb80, ^bb81
  ^bb80:  // pred: ^bb79
    %336 = llvm.extractvalue %58[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %337 = llvm.mlir.constant(32 : index) : i64
    %338 = llvm.mul %328, %337 : i64
    %339 = llvm.mlir.constant(8 : index) : i64
    %340 = llvm.mul %330, %339 : i64
    %341 = llvm.add %338, %340 : i64
    %342 = llvm.add %341, %334 : i64
    %343 = llvm.getelementptr %336[%342] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %344 = llvm.load %343 : !llvm.ptr -> f32
    %345 = llvm.extractvalue %142[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %346 = llvm.mlir.constant(32 : index) : i64
    %347 = llvm.mul %328, %346 : i64
    %348 = llvm.mlir.constant(8 : index) : i64
    %349 = llvm.mul %332, %348 : i64
    %350 = llvm.add %347, %349 : i64
    %351 = llvm.add %350, %334 : i64
    %352 = llvm.getelementptr %345[%351] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %353 = llvm.load %352 : !llvm.ptr -> f32
    %354 = llvm.extractvalue %310[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %355 = llvm.mlir.constant(16 : index) : i64
    %356 = llvm.mul %328, %355 : i64
    %357 = llvm.mlir.constant(4 : index) : i64
    %358 = llvm.mul %330, %357 : i64
    %359 = llvm.add %356, %358 : i64
    %360 = llvm.add %359, %332 : i64
    %361 = llvm.getelementptr %354[%360] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %362 = llvm.load %361 : !llvm.ptr -> f32
    %363 = llvm.fmul %344, %353  : f32
    %364 = llvm.fadd %362, %363  : f32
    %365 = llvm.extractvalue %310[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %366 = llvm.mlir.constant(16 : index) : i64
    %367 = llvm.mul %328, %366 : i64
    %368 = llvm.mlir.constant(4 : index) : i64
    %369 = llvm.mul %330, %368 : i64
    %370 = llvm.add %367, %369 : i64
    %371 = llvm.add %370, %332 : i64
    %372 = llvm.getelementptr %365[%371] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %364, %372 : f32, !llvm.ptr
    %373 = llvm.add %334, %1 : i64
    llvm.br ^bb79(%373 : i64)
  ^bb81:  // pred: ^bb79
    %374 = llvm.add %332, %1 : i64
    llvm.br ^bb77(%374 : i64)
  ^bb82:  // pred: ^bb77
    %375 = llvm.add %330, %1 : i64
    llvm.br ^bb75(%375 : i64)
  ^bb83:  // pred: ^bb75
    %376 = llvm.add %328, %1 : i64
    llvm.br ^bb73(%376 : i64)
  ^bb84:  // pred: ^bb73
    %377 = llvm.mlir.constant(1 : index) : i64
    %378 = llvm.mlir.constant(4 : index) : i64
    %379 = llvm.mlir.constant(8 : index) : i64
    %380 = llvm.mlir.constant(1 : index) : i64
    %381 = llvm.mlir.constant(32 : index) : i64
    %382 = llvm.mlir.constant(32 : index) : i64
    %383 = llvm.mlir.zero : !llvm.ptr
    %384 = llvm.getelementptr %383[%382] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %385 = llvm.ptrtoint %384 : !llvm.ptr to i64
    %386 = llvm.mlir.constant(64 : index) : i64
    %387 = llvm.add %385, %386 : i64
    %388 = llvm.call @malloc(%387) : (i64) -> !llvm.ptr
    %389 = llvm.ptrtoint %388 : !llvm.ptr to i64
    %390 = llvm.mlir.constant(1 : index) : i64
    %391 = llvm.sub %386, %390 : i64
    %392 = llvm.add %389, %391 : i64
    %393 = llvm.urem %392, %386  : i64
    %394 = llvm.sub %392, %393 : i64
    %395 = llvm.inttoptr %394 : i64 to !llvm.ptr
    %396 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %397 = llvm.insertvalue %388, %396[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %398 = llvm.insertvalue %395, %397[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %399 = llvm.mlir.constant(0 : index) : i64
    %400 = llvm.insertvalue %399, %398[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %401 = llvm.insertvalue %377, %400[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %402 = llvm.insertvalue %378, %401[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %403 = llvm.insertvalue %379, %402[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %404 = llvm.insertvalue %381, %403[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %405 = llvm.insertvalue %379, %404[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %406 = llvm.insertvalue %380, %405[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb85(%7 : i64)
  ^bb85(%407: i64):  // 2 preds: ^bb84, ^bb92
    %408 = llvm.icmp "slt" %407, %1 : i64
    llvm.cond_br %408, ^bb86, ^bb93
  ^bb86:  // pred: ^bb85
    llvm.br ^bb87(%7 : i64)
  ^bb87(%409: i64):  // 2 preds: ^bb86, ^bb91
    %410 = llvm.icmp "slt" %409, %0 : i64
    llvm.cond_br %410, ^bb88, ^bb92
  ^bb88:  // pred: ^bb87
    llvm.br ^bb89(%7 : i64)
  ^bb89(%411: i64):  // 2 preds: ^bb88, ^bb90
    %412 = llvm.icmp "slt" %411, %6 : i64
    llvm.cond_br %412, ^bb90, ^bb91
  ^bb90:  // pred: ^bb89
    %413 = llvm.extractvalue %406[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %414 = llvm.mlir.constant(32 : index) : i64
    %415 = llvm.mul %407, %414 : i64
    %416 = llvm.mlir.constant(8 : index) : i64
    %417 = llvm.mul %409, %416 : i64
    %418 = llvm.add %415, %417 : i64
    %419 = llvm.add %418, %411 : i64
    %420 = llvm.getelementptr %413[%419] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %420 : f32, !llvm.ptr
    %421 = llvm.add %411, %1 : i64
    llvm.br ^bb89(%421 : i64)
  ^bb91:  // pred: ^bb89
    %422 = llvm.add %409, %1 : i64
    llvm.br ^bb87(%422 : i64)
  ^bb92:  // pred: ^bb87
    %423 = llvm.add %407, %1 : i64
    llvm.br ^bb85(%423 : i64)
  ^bb93:  // pred: ^bb85
    llvm.br ^bb94(%7 : i64)
  ^bb94(%424: i64):  // 2 preds: ^bb93, ^bb104
    %425 = llvm.icmp "slt" %424, %1 : i64
    llvm.cond_br %425, ^bb95, ^bb105
  ^bb95:  // pred: ^bb94
    llvm.br ^bb96(%7 : i64)
  ^bb96(%426: i64):  // 2 preds: ^bb95, ^bb103
    %427 = llvm.icmp "slt" %426, %0 : i64
    llvm.cond_br %427, ^bb97, ^bb104
  ^bb97:  // pred: ^bb96
    llvm.br ^bb98(%7 : i64)
  ^bb98(%428: i64):  // 2 preds: ^bb97, ^bb102
    %429 = llvm.icmp "slt" %428, %6 : i64
    llvm.cond_br %429, ^bb99, ^bb103
  ^bb99:  // pred: ^bb98
    llvm.br ^bb100(%7 : i64)
  ^bb100(%430: i64):  // 2 preds: ^bb99, ^bb101
    %431 = llvm.icmp "slt" %430, %0 : i64
    llvm.cond_br %431, ^bb101, ^bb102
  ^bb101:  // pred: ^bb100
    %432 = llvm.extractvalue %310[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %433 = llvm.mlir.constant(16 : index) : i64
    %434 = llvm.mul %424, %433 : i64
    %435 = llvm.mlir.constant(4 : index) : i64
    %436 = llvm.mul %426, %435 : i64
    %437 = llvm.add %434, %436 : i64
    %438 = llvm.add %437, %430 : i64
    %439 = llvm.getelementptr %432[%438] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %440 = llvm.load %439 : !llvm.ptr -> f32
    %441 = llvm.extractvalue %226[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %442 = llvm.mlir.constant(32 : index) : i64
    %443 = llvm.mul %424, %442 : i64
    %444 = llvm.mlir.constant(8 : index) : i64
    %445 = llvm.mul %430, %444 : i64
    %446 = llvm.add %443, %445 : i64
    %447 = llvm.add %446, %428 : i64
    %448 = llvm.getelementptr %441[%447] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %449 = llvm.load %448 : !llvm.ptr -> f32
    %450 = llvm.extractvalue %406[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %451 = llvm.mlir.constant(32 : index) : i64
    %452 = llvm.mul %424, %451 : i64
    %453 = llvm.mlir.constant(8 : index) : i64
    %454 = llvm.mul %426, %453 : i64
    %455 = llvm.add %452, %454 : i64
    %456 = llvm.add %455, %428 : i64
    %457 = llvm.getelementptr %450[%456] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %458 = llvm.load %457 : !llvm.ptr -> f32
    %459 = llvm.fdiv %440, %4  : f32
    %460 = llvm.fmul %459, %449  : f32
    %461 = llvm.fadd %458, %460  : f32
    %462 = llvm.extractvalue %406[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %463 = llvm.mlir.constant(32 : index) : i64
    %464 = llvm.mul %424, %463 : i64
    %465 = llvm.mlir.constant(8 : index) : i64
    %466 = llvm.mul %426, %465 : i64
    %467 = llvm.add %464, %466 : i64
    %468 = llvm.add %467, %428 : i64
    %469 = llvm.getelementptr %462[%468] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %461, %469 : f32, !llvm.ptr
    %470 = llvm.add %430, %1 : i64
    llvm.br ^bb100(%470 : i64)
  ^bb102:  // pred: ^bb100
    %471 = llvm.add %428, %1 : i64
    llvm.br ^bb98(%471 : i64)
  ^bb103:  // pred: ^bb98
    %472 = llvm.add %426, %1 : i64
    llvm.br ^bb96(%472 : i64)
  ^bb104:  // pred: ^bb96
    %473 = llvm.add %424, %1 : i64
    llvm.br ^bb94(%473 : i64)
  ^bb105:  // pred: ^bb94
    %474 = llvm.mlir.constant(1 : index) : i64
    %475 = llvm.mlir.constant(4 : index) : i64
    %476 = llvm.mlir.constant(8 : index) : i64
    %477 = llvm.mlir.constant(1 : index) : i64
    %478 = llvm.mlir.constant(32 : index) : i64
    %479 = llvm.mlir.constant(32 : index) : i64
    %480 = llvm.mlir.zero : !llvm.ptr
    %481 = llvm.getelementptr %480[%479] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %482 = llvm.ptrtoint %481 : !llvm.ptr to i64
    %483 = llvm.mlir.constant(64 : index) : i64
    %484 = llvm.add %482, %483 : i64
    %485 = llvm.call @malloc(%484) : (i64) -> !llvm.ptr
    %486 = llvm.ptrtoint %485 : !llvm.ptr to i64
    %487 = llvm.mlir.constant(1 : index) : i64
    %488 = llvm.sub %483, %487 : i64
    %489 = llvm.add %486, %488 : i64
    %490 = llvm.urem %489, %483  : i64
    %491 = llvm.sub %489, %490 : i64
    %492 = llvm.inttoptr %491 : i64 to !llvm.ptr
    %493 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %494 = llvm.insertvalue %485, %493[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %495 = llvm.insertvalue %492, %494[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %496 = llvm.mlir.constant(0 : index) : i64
    %497 = llvm.insertvalue %496, %495[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %498 = llvm.insertvalue %474, %497[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %499 = llvm.insertvalue %475, %498[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %500 = llvm.insertvalue %476, %499[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %501 = llvm.insertvalue %478, %500[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %502 = llvm.insertvalue %476, %501[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %503 = llvm.insertvalue %477, %502[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb106(%7 : i64)
  ^bb106(%504: i64):  // 2 preds: ^bb105, ^bb113
    %505 = llvm.icmp "slt" %504, %1 : i64
    llvm.cond_br %505, ^bb107, ^bb114
  ^bb107:  // pred: ^bb106
    llvm.br ^bb108(%7 : i64)
  ^bb108(%506: i64):  // 2 preds: ^bb107, ^bb112
    %507 = llvm.icmp "slt" %506, %0 : i64
    llvm.cond_br %507, ^bb109, ^bb113
  ^bb109:  // pred: ^bb108
    llvm.br ^bb110(%7 : i64)
  ^bb110(%508: i64):  // 2 preds: ^bb109, ^bb111
    %509 = llvm.icmp "slt" %508, %6 : i64
    llvm.cond_br %509, ^bb111, ^bb112
  ^bb111:  // pred: ^bb110
    %510 = llvm.extractvalue %406[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %511 = llvm.mlir.constant(32 : index) : i64
    %512 = llvm.mul %504, %511 : i64
    %513 = llvm.mlir.constant(8 : index) : i64
    %514 = llvm.mul %506, %513 : i64
    %515 = llvm.add %512, %514 : i64
    %516 = llvm.add %515, %508 : i64
    %517 = llvm.getelementptr %510[%516] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %518 = llvm.load %517 : !llvm.ptr -> f32
    %519 = llvm.fadd %518, %3  : f32
    %520 = llvm.extractvalue %503[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %521 = llvm.mlir.constant(32 : index) : i64
    %522 = llvm.mul %504, %521 : i64
    %523 = llvm.mlir.constant(8 : index) : i64
    %524 = llvm.mul %506, %523 : i64
    %525 = llvm.add %522, %524 : i64
    %526 = llvm.add %525, %508 : i64
    %527 = llvm.getelementptr %520[%526] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %519, %527 : f32, !llvm.ptr
    %528 = llvm.add %508, %1 : i64
    llvm.br ^bb110(%528 : i64)
  ^bb112:  // pred: ^bb110
    %529 = llvm.add %506, %1 : i64
    llvm.br ^bb108(%529 : i64)
  ^bb113:  // pred: ^bb108
    %530 = llvm.add %504, %1 : i64
    llvm.br ^bb106(%530 : i64)
  ^bb114:  // pred: ^bb106
    %531 = llvm.mlir.constant(1 : index) : i64
    %532 = llvm.mlir.constant(4 : index) : i64
    %533 = llvm.mlir.constant(32 : index) : i64
    %534 = llvm.mlir.constant(1 : index) : i64
    %535 = llvm.mlir.constant(128 : index) : i64
    %536 = llvm.mlir.constant(128 : index) : i64
    %537 = llvm.mlir.zero : !llvm.ptr
    %538 = llvm.getelementptr %537[%536] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %539 = llvm.ptrtoint %538 : !llvm.ptr to i64
    %540 = llvm.mlir.constant(64 : index) : i64
    %541 = llvm.add %539, %540 : i64
    %542 = llvm.call @malloc(%541) : (i64) -> !llvm.ptr
    %543 = llvm.ptrtoint %542 : !llvm.ptr to i64
    %544 = llvm.mlir.constant(1 : index) : i64
    %545 = llvm.sub %540, %544 : i64
    %546 = llvm.add %543, %545 : i64
    %547 = llvm.urem %546, %540  : i64
    %548 = llvm.sub %546, %547 : i64
    %549 = llvm.inttoptr %548 : i64 to !llvm.ptr
    %550 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %551 = llvm.insertvalue %542, %550[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %552 = llvm.insertvalue %549, %551[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %553 = llvm.mlir.constant(0 : index) : i64
    %554 = llvm.insertvalue %553, %552[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %555 = llvm.insertvalue %531, %554[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %556 = llvm.insertvalue %532, %555[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %557 = llvm.insertvalue %533, %556[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %558 = llvm.insertvalue %535, %557[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %559 = llvm.insertvalue %533, %558[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %560 = llvm.insertvalue %534, %559[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %561 = builtin.unrealized_conversion_cast %560 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<1x4x32xf32>
    llvm.br ^bb115(%7 : i64)
  ^bb115(%562: i64):  // 2 preds: ^bb114, ^bb125
    %563 = builtin.unrealized_conversion_cast %562 : i64 to index
    %564 = llvm.icmp "slt" %562, %8 : i64
    llvm.cond_br %564, ^bb116, ^bb126
  ^bb116:  // pred: ^bb115
    %subview = memref.subview %561[0, 0, %563] [1, 4, 8] [1, 1, 1] : memref<1x4x32xf32> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
    %565 = builtin.unrealized_conversion_cast %subview : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    llvm.br ^bb117(%7 : i64)
  ^bb117(%566: i64):  // 2 preds: ^bb116, ^bb124
    %567 = llvm.icmp "slt" %566, %1 : i64
    llvm.cond_br %567, ^bb118, ^bb125
  ^bb118:  // pred: ^bb117
    llvm.br ^bb119(%7 : i64)
  ^bb119(%568: i64):  // 2 preds: ^bb118, ^bb123
    %569 = llvm.icmp "slt" %568, %0 : i64
    llvm.cond_br %569, ^bb120, ^bb124
  ^bb120:  // pred: ^bb119
    llvm.br ^bb121(%7 : i64)
  ^bb121(%570: i64):  // 2 preds: ^bb120, ^bb122
    %571 = llvm.icmp "slt" %570, %6 : i64
    llvm.cond_br %571, ^bb122, ^bb123
  ^bb122:  // pred: ^bb121
    %572 = llvm.extractvalue %565[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %573 = llvm.extractvalue %565[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %574 = llvm.getelementptr %572[%573] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %575 = llvm.mlir.constant(128 : index) : i64
    %576 = llvm.mul %566, %575 : i64
    %577 = llvm.mlir.constant(32 : index) : i64
    %578 = llvm.mul %568, %577 : i64
    %579 = llvm.add %576, %578 : i64
    %580 = llvm.add %579, %570 : i64
    %581 = llvm.getelementptr %574[%580] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %581 : f32, !llvm.ptr
    %582 = llvm.add %570, %1 : i64
    llvm.br ^bb121(%582 : i64)
  ^bb123:  // pred: ^bb121
    %583 = llvm.add %568, %1 : i64
    llvm.br ^bb119(%583 : i64)
  ^bb124:  // pred: ^bb119
    %584 = llvm.add %566, %1 : i64
    llvm.br ^bb117(%584 : i64)
  ^bb125:  // pred: ^bb117
    %585 = llvm.intr.stacksave : !llvm.ptr
    %586 = llvm.mlir.constant(3 : i64) : i64
    %587 = llvm.mlir.constant(1 : index) : i64
    %588 = llvm.alloca %587 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %565, %588 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %589 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %590 = llvm.insertvalue %586, %589[0] : !llvm.struct<(i64, ptr)> 
    %591 = llvm.insertvalue %588, %590[1] : !llvm.struct<(i64, ptr)> 
    %592 = llvm.mlir.constant(3 : i64) : i64
    %593 = llvm.mlir.constant(1 : index) : i64
    %594 = llvm.alloca %593 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %565, %594 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %595 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %596 = llvm.insertvalue %592, %595[0] : !llvm.struct<(i64, ptr)> 
    %597 = llvm.insertvalue %594, %596[1] : !llvm.struct<(i64, ptr)> 
    %598 = llvm.mlir.constant(1 : index) : i64
    %599 = llvm.alloca %598 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %591, %599 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %600 = llvm.alloca %598 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %597, %600 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %601 = llvm.mlir.zero : !llvm.ptr
    %602 = llvm.getelementptr %601[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %603 = llvm.ptrtoint %602 : !llvm.ptr to i64
    llvm.call @memrefCopy(%603, %599, %600) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %585 : !llvm.ptr
    %604 = llvm.add %562, %6 : i64
    llvm.br ^bb115(%604 : i64)
  ^bb126:  // pred: ^bb115
    llvm.br ^bb127(%7 : i64)
  ^bb127(%605: i64):  // 2 preds: ^bb126, ^bb140
    %606 = builtin.unrealized_conversion_cast %605 : i64 to index
    %607 = llvm.icmp "slt" %605, %8 : i64
    llvm.cond_br %607, ^bb128, ^bb141
  ^bb128:  // pred: ^bb127
    %subview_0 = memref.subview %561[0, 0, %606] [1, 4, 8] [1, 1, 1] : memref<1x4x32xf32> to memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>>
    %608 = builtin.unrealized_conversion_cast %subview_0 : memref<1x4x8xf32, strided<[128, 32, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    llvm.br ^bb129(%7 : i64)
  ^bb129(%609: i64):  // 2 preds: ^bb128, ^bb139
    %610 = llvm.icmp "slt" %609, %1 : i64
    llvm.cond_br %610, ^bb130, ^bb140
  ^bb130:  // pred: ^bb129
    llvm.br ^bb131(%7 : i64)
  ^bb131(%611: i64):  // 2 preds: ^bb130, ^bb138
    %612 = llvm.icmp "slt" %611, %0 : i64
    llvm.cond_br %612, ^bb132, ^bb139
  ^bb132:  // pred: ^bb131
    llvm.br ^bb133(%7 : i64)
  ^bb133(%613: i64):  // 2 preds: ^bb132, ^bb137
    %614 = llvm.icmp "slt" %613, %6 : i64
    llvm.cond_br %614, ^bb134, ^bb138
  ^bb134:  // pred: ^bb133
    llvm.br ^bb135(%7 : i64)
  ^bb135(%615: i64):  // 2 preds: ^bb134, ^bb136
    %616 = llvm.icmp "slt" %615, %6 : i64
    llvm.cond_br %616, ^bb136, ^bb137
  ^bb136:  // pred: ^bb135
    %617 = llvm.extractvalue %503[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %618 = llvm.mlir.constant(32 : index) : i64
    %619 = llvm.mul %609, %618 : i64
    %620 = llvm.mlir.constant(8 : index) : i64
    %621 = llvm.mul %611, %620 : i64
    %622 = llvm.add %619, %621 : i64
    %623 = llvm.add %622, %615 : i64
    %624 = llvm.getelementptr %617[%623] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %625 = llvm.load %624 : !llvm.ptr -> f32
    %626 = llvm.extractvalue %608[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %627 = llvm.extractvalue %608[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %628 = llvm.getelementptr %626[%627] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %629 = llvm.mlir.constant(128 : index) : i64
    %630 = llvm.mul %609, %629 : i64
    %631 = llvm.mlir.constant(32 : index) : i64
    %632 = llvm.mul %611, %631 : i64
    %633 = llvm.add %630, %632 : i64
    %634 = llvm.add %633, %613 : i64
    %635 = llvm.getelementptr %628[%634] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %636 = llvm.load %635 : !llvm.ptr -> f32
    %637 = llvm.fmul %625, %5  : f32
    %638 = llvm.fadd %636, %637  : f32
    %639 = llvm.extractvalue %608[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %640 = llvm.extractvalue %608[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %641 = llvm.getelementptr %639[%640] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %642 = llvm.mlir.constant(128 : index) : i64
    %643 = llvm.mul %609, %642 : i64
    %644 = llvm.mlir.constant(32 : index) : i64
    %645 = llvm.mul %611, %644 : i64
    %646 = llvm.add %643, %645 : i64
    %647 = llvm.add %646, %613 : i64
    %648 = llvm.getelementptr %641[%647] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %638, %648 : f32, !llvm.ptr
    %649 = llvm.add %615, %1 : i64
    llvm.br ^bb135(%649 : i64)
  ^bb137:  // pred: ^bb135
    %650 = llvm.add %613, %1 : i64
    llvm.br ^bb133(%650 : i64)
  ^bb138:  // pred: ^bb133
    %651 = llvm.add %611, %1 : i64
    llvm.br ^bb131(%651 : i64)
  ^bb139:  // pred: ^bb131
    %652 = llvm.add %609, %1 : i64
    llvm.br ^bb129(%652 : i64)
  ^bb140:  // pred: ^bb129
    %653 = llvm.intr.stacksave : !llvm.ptr
    %654 = llvm.mlir.constant(3 : i64) : i64
    %655 = llvm.mlir.constant(1 : index) : i64
    %656 = llvm.alloca %655 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %608, %656 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %657 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %658 = llvm.insertvalue %654, %657[0] : !llvm.struct<(i64, ptr)> 
    %659 = llvm.insertvalue %656, %658[1] : !llvm.struct<(i64, ptr)> 
    %660 = llvm.mlir.constant(3 : i64) : i64
    %661 = llvm.mlir.constant(1 : index) : i64
    %662 = llvm.alloca %661 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %608, %662 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %663 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %664 = llvm.insertvalue %660, %663[0] : !llvm.struct<(i64, ptr)> 
    %665 = llvm.insertvalue %662, %664[1] : !llvm.struct<(i64, ptr)> 
    %666 = llvm.mlir.constant(1 : index) : i64
    %667 = llvm.alloca %666 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %659, %667 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %668 = llvm.alloca %666 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %665, %668 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %669 = llvm.mlir.zero : !llvm.ptr
    %670 = llvm.getelementptr %669[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %671 = llvm.ptrtoint %670 : !llvm.ptr to i64
    llvm.call @memrefCopy(%671, %667, %668) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %653 : !llvm.ptr
    %672 = llvm.add %605, %6 : i64
    llvm.br ^bb127(%672 : i64)
  ^bb141:  // pred: ^bb127
    %673 = llvm.mlir.constant(1 : index) : i64
    %674 = llvm.mlir.constant(4 : index) : i64
    %675 = llvm.mlir.constant(8 : index) : i64
    %676 = llvm.mlir.constant(1 : index) : i64
    %677 = llvm.mlir.constant(32 : index) : i64
    %678 = llvm.mlir.constant(32 : index) : i64
    %679 = llvm.mlir.zero : !llvm.ptr
    %680 = llvm.getelementptr %679[%678] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %681 = llvm.ptrtoint %680 : !llvm.ptr to i64
    %682 = llvm.mlir.constant(64 : index) : i64
    %683 = llvm.add %681, %682 : i64
    %684 = llvm.call @malloc(%683) : (i64) -> !llvm.ptr
    %685 = llvm.ptrtoint %684 : !llvm.ptr to i64
    %686 = llvm.mlir.constant(1 : index) : i64
    %687 = llvm.sub %682, %686 : i64
    %688 = llvm.add %685, %687 : i64
    %689 = llvm.urem %688, %682  : i64
    %690 = llvm.sub %688, %689 : i64
    %691 = llvm.inttoptr %690 : i64 to !llvm.ptr
    %692 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %693 = llvm.insertvalue %684, %692[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %694 = llvm.insertvalue %691, %693[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %695 = llvm.mlir.constant(0 : index) : i64
    %696 = llvm.insertvalue %695, %694[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %697 = llvm.insertvalue %673, %696[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %698 = llvm.insertvalue %674, %697[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %699 = llvm.insertvalue %675, %698[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %700 = llvm.insertvalue %677, %699[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %701 = llvm.insertvalue %675, %700[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %702 = llvm.insertvalue %676, %701[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb142(%7 : i64)
  ^bb142(%703: i64):  // 2 preds: ^bb141, ^bb149
    %704 = llvm.icmp "slt" %703, %1 : i64
    llvm.cond_br %704, ^bb143, ^bb150
  ^bb143:  // pred: ^bb142
    llvm.br ^bb144(%7 : i64)
  ^bb144(%705: i64):  // 2 preds: ^bb143, ^bb148
    %706 = llvm.icmp "slt" %705, %0 : i64
    llvm.cond_br %706, ^bb145, ^bb149
  ^bb145:  // pred: ^bb144
    llvm.br ^bb146(%7 : i64)
  ^bb146(%707: i64):  // 2 preds: ^bb145, ^bb147
    %708 = llvm.icmp "slt" %707, %6 : i64
    llvm.cond_br %708, ^bb147, ^bb148
  ^bb147:  // pred: ^bb146
    %709 = llvm.extractvalue %702[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %710 = llvm.mlir.constant(32 : index) : i64
    %711 = llvm.mul %703, %710 : i64
    %712 = llvm.mlir.constant(8 : index) : i64
    %713 = llvm.mul %705, %712 : i64
    %714 = llvm.add %711, %713 : i64
    %715 = llvm.add %714, %707 : i64
    %716 = llvm.getelementptr %709[%715] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %716 : f32, !llvm.ptr
    %717 = llvm.add %707, %1 : i64
    llvm.br ^bb146(%717 : i64)
  ^bb148:  // pred: ^bb146
    %718 = llvm.add %705, %1 : i64
    llvm.br ^bb144(%718 : i64)
  ^bb149:  // pred: ^bb144
    %719 = llvm.add %703, %1 : i64
    llvm.br ^bb142(%719 : i64)
  ^bb150:  // pred: ^bb142
    llvm.br ^bb151(%7 : i64)
  ^bb151(%720: i64):  // 2 preds: ^bb150, ^bb161
    %721 = llvm.icmp "slt" %720, %1 : i64
    llvm.cond_br %721, ^bb152, ^bb162
  ^bb152:  // pred: ^bb151
    llvm.br ^bb153(%7 : i64)
  ^bb153(%722: i64):  // 2 preds: ^bb152, ^bb160
    %723 = llvm.icmp "slt" %722, %0 : i64
    llvm.cond_br %723, ^bb154, ^bb161
  ^bb154:  // pred: ^bb153
    llvm.br ^bb155(%7 : i64)
  ^bb155(%724: i64):  // 2 preds: ^bb154, ^bb159
    %725 = llvm.icmp "slt" %724, %6 : i64
    llvm.cond_br %725, ^bb156, ^bb160
  ^bb156:  // pred: ^bb155
    llvm.br ^bb157(%7 : i64)
  ^bb157(%726: i64):  // 2 preds: ^bb156, ^bb158
    %727 = llvm.icmp "slt" %726, %8 : i64
    llvm.cond_br %727, ^bb158, ^bb159
  ^bb158:  // pred: ^bb157
    %728 = llvm.extractvalue %560[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %729 = llvm.mlir.constant(128 : index) : i64
    %730 = llvm.mul %720, %729 : i64
    %731 = llvm.mlir.constant(32 : index) : i64
    %732 = llvm.mul %722, %731 : i64
    %733 = llvm.add %730, %732 : i64
    %734 = llvm.add %733, %726 : i64
    %735 = llvm.getelementptr %728[%734] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %736 = llvm.load %735 : !llvm.ptr -> f32
    %737 = llvm.extractvalue %702[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %738 = llvm.mlir.constant(32 : index) : i64
    %739 = llvm.mul %720, %738 : i64
    %740 = llvm.mlir.constant(8 : index) : i64
    %741 = llvm.mul %722, %740 : i64
    %742 = llvm.add %739, %741 : i64
    %743 = llvm.add %742, %724 : i64
    %744 = llvm.getelementptr %737[%743] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %745 = llvm.load %744 : !llvm.ptr -> f32
    %746 = llvm.intr.maximum(%736, %2)  : (f32, f32) -> f32
    %747 = llvm.fmul %746, %5  : f32
    %748 = llvm.fadd %745, %747  : f32
    %749 = llvm.extractvalue %702[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %750 = llvm.mlir.constant(32 : index) : i64
    %751 = llvm.mul %720, %750 : i64
    %752 = llvm.mlir.constant(8 : index) : i64
    %753 = llvm.mul %722, %752 : i64
    %754 = llvm.add %751, %753 : i64
    %755 = llvm.add %754, %724 : i64
    %756 = llvm.getelementptr %749[%755] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %748, %756 : f32, !llvm.ptr
    %757 = llvm.add %726, %1 : i64
    llvm.br ^bb157(%757 : i64)
  ^bb159:  // pred: ^bb157
    %758 = llvm.add %724, %1 : i64
    llvm.br ^bb155(%758 : i64)
  ^bb160:  // pred: ^bb155
    %759 = llvm.add %722, %1 : i64
    llvm.br ^bb153(%759 : i64)
  ^bb161:  // pred: ^bb153
    %760 = llvm.add %720, %1 : i64
    llvm.br ^bb151(%760 : i64)
  ^bb162:  // pred: ^bb151
    %761 = llvm.mlir.constant(1 : index) : i64
    %762 = llvm.mlir.constant(4 : index) : i64
    %763 = llvm.mlir.constant(8 : index) : i64
    %764 = llvm.mlir.constant(1 : index) : i64
    %765 = llvm.mlir.constant(32 : index) : i64
    %766 = llvm.mlir.constant(32 : index) : i64
    %767 = llvm.mlir.zero : !llvm.ptr
    %768 = llvm.getelementptr %767[%766] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %769 = llvm.ptrtoint %768 : !llvm.ptr to i64
    %770 = llvm.mlir.constant(64 : index) : i64
    %771 = llvm.add %769, %770 : i64
    %772 = llvm.call @malloc(%771) : (i64) -> !llvm.ptr
    %773 = llvm.ptrtoint %772 : !llvm.ptr to i64
    %774 = llvm.mlir.constant(1 : index) : i64
    %775 = llvm.sub %770, %774 : i64
    %776 = llvm.add %773, %775 : i64
    %777 = llvm.urem %776, %770  : i64
    %778 = llvm.sub %776, %777 : i64
    %779 = llvm.inttoptr %778 : i64 to !llvm.ptr
    %780 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %781 = llvm.insertvalue %772, %780[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %782 = llvm.insertvalue %779, %781[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %783 = llvm.mlir.constant(0 : index) : i64
    %784 = llvm.insertvalue %783, %782[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %785 = llvm.insertvalue %761, %784[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %786 = llvm.insertvalue %762, %785[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %787 = llvm.insertvalue %763, %786[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %788 = llvm.insertvalue %765, %787[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %789 = llvm.insertvalue %763, %788[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %790 = llvm.insertvalue %764, %789[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb163(%7 : i64)
  ^bb163(%791: i64):  // 2 preds: ^bb162, ^bb170
    %792 = llvm.icmp "slt" %791, %1 : i64
    llvm.cond_br %792, ^bb164, ^bb171
  ^bb164:  // pred: ^bb163
    llvm.br ^bb165(%7 : i64)
  ^bb165(%793: i64):  // 2 preds: ^bb164, ^bb169
    %794 = llvm.icmp "slt" %793, %0 : i64
    llvm.cond_br %794, ^bb166, ^bb170
  ^bb166:  // pred: ^bb165
    llvm.br ^bb167(%7 : i64)
  ^bb167(%795: i64):  // 2 preds: ^bb166, ^bb168
    %796 = llvm.icmp "slt" %795, %6 : i64
    llvm.cond_br %796, ^bb168, ^bb169
  ^bb168:  // pred: ^bb167
    %797 = llvm.extractvalue %503[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %798 = llvm.mlir.constant(32 : index) : i64
    %799 = llvm.mul %791, %798 : i64
    %800 = llvm.mlir.constant(8 : index) : i64
    %801 = llvm.mul %793, %800 : i64
    %802 = llvm.add %799, %801 : i64
    %803 = llvm.add %802, %795 : i64
    %804 = llvm.getelementptr %797[%803] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %805 = llvm.load %804 : !llvm.ptr -> f32
    %806 = llvm.extractvalue %702[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %807 = llvm.mlir.constant(32 : index) : i64
    %808 = llvm.mul %791, %807 : i64
    %809 = llvm.mlir.constant(8 : index) : i64
    %810 = llvm.mul %793, %809 : i64
    %811 = llvm.add %808, %810 : i64
    %812 = llvm.add %811, %795 : i64
    %813 = llvm.getelementptr %806[%812] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %814 = llvm.load %813 : !llvm.ptr -> f32
    %815 = llvm.fadd %805, %814  : f32
    %816 = llvm.fmul %815, %815  : f32
    %817 = llvm.fadd %816, %816  : f32
    %818 = llvm.extractvalue %790[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %819 = llvm.mlir.constant(32 : index) : i64
    %820 = llvm.mul %791, %819 : i64
    %821 = llvm.mlir.constant(8 : index) : i64
    %822 = llvm.mul %793, %821 : i64
    %823 = llvm.add %820, %822 : i64
    %824 = llvm.add %823, %795 : i64
    %825 = llvm.getelementptr %818[%824] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %817, %825 : f32, !llvm.ptr
    %826 = llvm.add %795, %1 : i64
    llvm.br ^bb167(%826 : i64)
  ^bb169:  // pred: ^bb167
    %827 = llvm.add %793, %1 : i64
    llvm.br ^bb165(%827 : i64)
  ^bb170:  // pred: ^bb165
    %828 = llvm.add %791, %1 : i64
    llvm.br ^bb163(%828 : i64)
  ^bb171:  // pred: ^bb163
    llvm.return %790 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  }
}
