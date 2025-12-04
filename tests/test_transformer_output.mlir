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
    cherry.return %8 : !cherry.cherry_tensor<[?xf32]>
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
    cherry.return %26 : !cherry.cherry_tensor<[?xf32]>
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
    cherry.return %26 : !cherry.cherry_tensor<[1x4x8xf32]>
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
    cherry.return %25 : !cherry.cherry_tensor<[1x4x8xf32]>
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
      %33 = arith.mulf %in, %in_14 : f32
      %34 = arith.addf %out, %33 : f32
      linalg.yield %34 : f32
    } -> tensor<1x4x8xf32>
    %3 = tensor.empty() : tensor<1x4x8xf32>
    %cst_6 = arith.constant 0.000000e+00 : f32
    %4 = linalg.fill ins(%cst_6 : f32) outs(%3 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%cst, %cst_1 : tensor<1x4x8xf32>, tensor<8x8xf32>) outs(%4 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %33 = arith.mulf %in, %in_14 : f32
      %34 = arith.addf %out, %33 : f32
      linalg.yield %34 : f32
    } -> tensor<1x4x8xf32>
    %6 = tensor.empty() : tensor<1x4x8xf32>
    %cst_7 = arith.constant 0.000000e+00 : f32
    %7 = linalg.fill ins(%cst_7 : f32) outs(%6 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%cst, %cst_2 : tensor<1x4x8xf32>, tensor<8x8xf32>) outs(%7 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %33 = arith.mulf %in, %in_14 : f32
      %34 = arith.addf %out, %33 : f32
      linalg.yield %34 : f32
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
      %33 = arith.mulf %in, %in_14 : f32
      %34 = arith.addf %out, %33 : f32
      linalg.yield %34 : f32
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
      %33 = arith.mulf %in, %in_14 : f32
      %34 = arith.addf %out, %33 : f32
      linalg.yield %34 : f32
    } -> tensor<1x4x8xf32>
    %21 = tensor.empty() : tensor<1x4x8xf32>
    %22 = linalg.add ins(%cst, %20 : tensor<1x4x8xf32>, tensor<1x4x8xf32>) outs(%21 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %23 = tensor.empty() : tensor<1x4x32xf32>
    %cst_12 = arith.constant 0.000000e+00 : f32
    %24 = linalg.fill ins(%cst_12 : f32) outs(%23 : tensor<1x4x32xf32>) -> tensor<1x4x32xf32>
    %25 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%22, %cst_3 : tensor<1x4x8xf32>, tensor<8x32xf32>) outs(%24 : tensor<1x4x32xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %33 = arith.mulf %in, %in_14 : f32
      %34 = arith.addf %out, %33 : f32
      linalg.yield %34 : f32
    } -> tensor<1x4x32xf32>
    %26 = tensor.empty() : tensor<1x4x32xf32>
    %27 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25 : tensor<1x4x32xf32>) outs(%26 : tensor<1x4x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_14 = arith.constant 0.000000e+00 : f32
      %33 = arith.maximumf %in, %cst_14 : f32
      linalg.yield %33 : f32
    } -> tensor<1x4x32xf32>
    %28 = tensor.empty() : tensor<1x4x8xf32>
    %cst_13 = arith.constant 0.000000e+00 : f32
    %29 = linalg.fill ins(%cst_13 : f32) outs(%28 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    %30 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%27, %cst_4 : tensor<1x4x32xf32>, tensor<32x8xf32>) outs(%29 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %33 = arith.mulf %in, %in_14 : f32
      %34 = arith.addf %out, %33 : f32
      linalg.yield %34 : f32
    } -> tensor<1x4x8xf32>
    %31 = tensor.empty() : tensor<1x4x8xf32>
    %32 = linalg.add ins(%22, %30 : tensor<1x4x8xf32>, tensor<1x4x8xf32>) outs(%31 : tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    return %32 : tensor<1x4x8xf32>
  }
}
