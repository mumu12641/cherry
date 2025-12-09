// Original IR loaded from file
module {
  cherry.func private @llama_forward(%arg0: i64, %arg1: i64, %arg2: !cherry.cherry_tensor<[32x2048x128xf32]>, %arg3: !cherry.cherry_tensor<[32000x768xf32]>) -> (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>) {
    %0 = cherry.constant(0 : i64) : i64
    %1 = cherry.tensor_slice %arg3[%arg0, %0] sizes [1, 768] : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    %2 = cherry.create_tensor dense<3.214000e+03> : tensor<768x32000xf32> -> !cherry.cherry_tensor<[768x32000xf32]>
    %3 = cherry.matmul %1, %2 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[768x32000xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    %4 = cherry.create_tensor dense<1.234200e+04> : tensor<128x128xf32> -> !cherry.cherry_tensor<[128x128xf32]>
    %5 = cherry.matmul %arg2, %4 : (!cherry.cherry_tensor<[32x2048x128xf32]>, !cherry.cherry_tensor<[128x128xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %3, %5 : !cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[?xf32]>
  }
  cherry.func @host() -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
    %1 = cherry.create_tensor dense<0.000000e+00> : tensor<32x2048x128xf32> -> !cherry.cherry_tensor<[32x2048x128xf32]>
    %2 = cherry.constant(1 : i64) : i64
    %3 = cherry.constant(0 : i64) : i64
    %4 = cherry.constant(10 : i64) : i64
    %5:3 = scf.while (%arg0 = %2, %arg1 = %3, %arg2 = %1) : (i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>) -> (i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>) {
      %6 = arith.cmpi slt, %arg1, %4 : i64
      scf.condition(%6) %arg0, %arg1, %arg2 : i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>
    } do {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: !cherry.cherry_tensor<[32x2048x128xf32]>):
      %6:2 = cherry.call @llama_forward(%arg0, %arg1, %arg2, %0) : (i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>, !cherry.cherry_tensor<[32000x768xf32]>) -> (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[32x2048x128xf32]>)
      cherry.print %6#0 : !cherry.cherry_tensor<[?xf32]>
      %7 = cherry.argmax %6#0 dim 1 : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xi64]>
      %8 = cherry.constant(0 : i64) : i64
      %9 = cherry.tensor_get %7[%8] : (!cherry.cherry_tensor<[?xi64]>, i64) -> i64
      %10 = cherry.constant(1 : i64) : i64
      %11 = arith.addi %arg1, %10 : i64
      scf.yield %9, %11, %6#1 : i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>
    }
    cherry.return %5#2 : !cherry.cherry_tensor<[32x2048x128xf32]>
  }
}

// ==========================================
// Phase: Inliner
// ==========================================
module {
  cherry.func @host() -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
    %1 = cherry.create_tensor dense<0.000000e+00> : tensor<32x2048x128xf32> -> !cherry.cherry_tensor<[32x2048x128xf32]>
    %2 = cherry.constant(1 : i64) : i64
    %3 = cherry.constant(0 : i64) : i64
    %4 = cherry.constant(10 : i64) : i64
    %5:3 = scf.while (%arg0 = %2, %arg1 = %3, %arg2 = %1) : (i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>) -> (i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>) {
      %6 = arith.cmpi slt, %arg1, %4 : i64
      scf.condition(%6) %arg0, %arg1, %arg2 : i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>
    } do {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: !cherry.cherry_tensor<[32x2048x128xf32]>):
      %6 = cherry.constant(0 : i64) : i64
      %7 = cherry.tensor_slice %0[%arg0, %6] sizes [1, 768] : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
      %8 = cherry.create_tensor dense<3.214000e+03> : tensor<768x32000xf32> -> !cherry.cherry_tensor<[768x32000xf32]>
      %9 = cherry.matmul %7, %8 : (!cherry.cherry_tensor<[?xf32]>, !cherry.cherry_tensor<[768x32000xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %10 = cherry.create_tensor dense<1.234200e+04> : tensor<128x128xf32> -> !cherry.cherry_tensor<[128x128xf32]>
      %11 = cherry.matmul %arg2, %10 : (!cherry.cherry_tensor<[32x2048x128xf32]>, !cherry.cherry_tensor<[128x128xf32]>) -> !cherry.cherry_tensor<[?xf32]>
      %12 = cherry.cast %11 : !cherry.cherry_tensor<[?xf32]> to !cherry.cherry_tensor<[32x2048x128xf32]>
      cherry.print %9 : !cherry.cherry_tensor<[?xf32]>
      %13 = cherry.argmax %9 dim 1 : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xi64]>
      %14 = cherry.constant(0 : i64) : i64
      %15 = cherry.tensor_get %13[%14] : (!cherry.cherry_tensor<[?xi64]>, i64) -> i64
      %16 = cherry.constant(1 : i64) : i64
      %17 = arith.addi %arg1, %16 : i64
      scf.yield %15, %17, %12 : i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>
    }
    cherry.return %5#2 : !cherry.cherry_tensor<[32x2048x128xf32]>
  }
}

// ==========================================
// Phase: Shape Inference
// ==========================================
module {
  cherry.func @host() -> !cherry.cherry_tensor<[32x2048x128xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
    %1 = cherry.create_tensor dense<0.000000e+00> : tensor<32x2048x128xf32> -> !cherry.cherry_tensor<[32x2048x128xf32]>
    %2 = cherry.constant(1 : i64) : i64
    %3 = cherry.constant(0 : i64) : i64
    %4 = cherry.constant(10 : i64) : i64
    %5:3 = scf.while (%arg0 = %2, %arg1 = %3, %arg2 = %1) : (i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>) -> (i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>) {
      %6 = arith.cmpi slt, %arg1, %4 : i64
      scf.condition(%6) %arg0, %arg1, %arg2 : i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>
    } do {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: !cherry.cherry_tensor<[32x2048x128xf32]>):
      %6 = cherry.constant(0 : i64) : i64
      %7 = cherry.tensor_slice %0[%arg0, %6] sizes [1, 768] : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
      %8 = cherry.create_tensor dense<3.214000e+03> : tensor<768x32000xf32> -> !cherry.cherry_tensor<[768x32000xf32]>
      %9 = cherry.matmul %7, %8 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x32000xf32]>) -> !cherry.cherry_tensor<[1x32000xf32]>
      %10 = cherry.create_tensor dense<1.234200e+04> : tensor<128x128xf32> -> !cherry.cherry_tensor<[128x128xf32]>
      %11 = cherry.matmul %arg2, %10 : (!cherry.cherry_tensor<[32x2048x128xf32]>, !cherry.cherry_tensor<[128x128xf32]>) -> !cherry.cherry_tensor<[32x2048x128xf32]>
      %12 = cherry.cast %11 : !cherry.cherry_tensor<[32x2048x128xf32]> to !cherry.cherry_tensor<[32x2048x128xf32]>
      cherry.print %9 : !cherry.cherry_tensor<[1x32000xf32]>
      %13 = cherry.argmax %9 dim 1 : (!cherry.cherry_tensor<[1x32000xf32]>) -> !cherry.cherry_tensor<[1xi64]>
      %14 = cherry.constant(0 : i64) : i64
      %15 = cherry.tensor_get %13[%14] : (!cherry.cherry_tensor<[1xi64]>, i64) -> i64
      %16 = cherry.constant(1 : i64) : i64
      %17 = arith.addi %arg1, %16 : i64
      scf.yield %15, %17, %12 : i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>
    }
    cherry.return %5#2 : !cherry.cherry_tensor<[32x2048x128xf32]>
  }
}

// ==========================================
// Phase: Canonicalizer
// ==========================================
module {
  cherry.func @host() -> !cherry.cherry_tensor<[32x2048x128xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
    %1 = cherry.create_tensor dense<0.000000e+00> : tensor<32x2048x128xf32> -> !cherry.cherry_tensor<[32x2048x128xf32]>
    %2 = cherry.constant(1 : i64) : i64
    %3 = cherry.constant(0 : i64) : i64
    %4 = cherry.constant(10 : i64) : i64
    %5:3 = scf.while (%arg0 = %2, %arg1 = %3, %arg2 = %1) : (i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>) -> (i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>) {
      %6 = arith.cmpi slt, %arg1, %4 : i64
      scf.condition(%6) %arg0, %arg1, %arg2 : i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>
    } do {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: !cherry.cherry_tensor<[32x2048x128xf32]>):
      %6 = cherry.constant(0 : i64) : i64
      %7 = cherry.tensor_slice %0[%arg0, %6] sizes [1, 768] : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
      %8 = cherry.create_tensor dense<3.214000e+03> : tensor<768x32000xf32> -> !cherry.cherry_tensor<[768x32000xf32]>
      %9 = cherry.matmul %7, %8 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x32000xf32]>) -> !cherry.cherry_tensor<[1x32000xf32]>
      %10 = cherry.create_tensor dense<1.234200e+04> : tensor<128x128xf32> -> !cherry.cherry_tensor<[128x128xf32]>
      %11 = cherry.matmul %arg2, %10 : (!cherry.cherry_tensor<[32x2048x128xf32]>, !cherry.cherry_tensor<[128x128xf32]>) -> !cherry.cherry_tensor<[32x2048x128xf32]>
      cherry.print %9 : !cherry.cherry_tensor<[1x32000xf32]>
      %12 = cherry.argmax %9 dim 1 : (!cherry.cherry_tensor<[1x32000xf32]>) -> !cherry.cherry_tensor<[1xi64]>
      %13 = cherry.constant(0 : i64) : i64
      %14 = cherry.tensor_get %12[%13] : (!cherry.cherry_tensor<[1xi64]>, i64) -> i64
      %15 = cherry.constant(1 : i64) : i64
      %16 = arith.addi %arg1, %15 : i64
      scf.yield %14, %16, %11 : i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>
    }
    cherry.return %5#2 : !cherry.cherry_tensor<[32x2048x128xf32]>
  }
}

// ==========================================
// Phase: Convert to Linalg
// ==========================================
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d0)>
module {
  func.func private @print_memref_f32(tensor<*xf32> {bufferization.access = "read"})
  func.func @host() -> tensor<32x2048x128xf32> {
    %cst = arith.constant dense<5.000000e-01> : tensor<32000x768xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x2048x128xf32>
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c10_i64 = arith.constant 10 : i64
    %0:3 = scf.while (%arg0 = %c1_i64, %arg1 = %c0_i64, %arg2 = %cst_0) : (i64, i64, tensor<32x2048x128xf32>) -> (i64, i64, tensor<32x2048x128xf32>) {
      %1 = arith.cmpi slt, %arg1, %c10_i64 : i64
      scf.condition(%1) %arg0, %arg1, %arg2 : i64, i64, tensor<32x2048x128xf32>
    } do {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: tensor<32x2048x128xf32>):
      %c0_i64_1 = arith.constant 0 : i64
      %1 = arith.index_cast %arg0 : i64 to index
      %2 = arith.index_cast %c0_i64_1 : i64 to index
      %extracted_slice = tensor.extract_slice %cst[%1, %2] [1, 768] [1, 1] : tensor<32000x768xf32> to tensor<1x768xf32>
      %cst_2 = arith.constant dense<3.214000e+03> : tensor<768x32000xf32>
      %3 = tensor.empty() : tensor<1x32000xf32>
      %cst_3 = arith.constant 0.000000e+00 : f32
      %4 = linalg.fill ins(%cst_3 : f32) outs(%3 : tensor<1x32000xf32>) -> tensor<1x32000xf32>
      %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %cst_2 : tensor<1x768xf32>, tensor<768x32000xf32>) outs(%4 : tensor<1x32000xf32>) {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %16 = arith.mulf %in, %in_10 : f32
        %17 = arith.addf %out, %16 : f32
        linalg.yield %17 : f32
      } -> tensor<1x32000xf32>
      %cst_4 = arith.constant dense<1.234200e+04> : tensor<128x128xf32>
      %6 = tensor.empty() : tensor<32x2048x128xf32>
      %cst_5 = arith.constant 0.000000e+00 : f32
      %7 = linalg.fill ins(%cst_5 : f32) outs(%6 : tensor<32x2048x128xf32>) -> tensor<32x2048x128xf32>
      %8 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg2, %cst_4 : tensor<32x2048x128xf32>, tensor<128x128xf32>) outs(%7 : tensor<32x2048x128xf32>) {
      ^bb0(%in: f32, %in_10: f32, %out: f32):
        %16 = arith.mulf %in, %in_10 : f32
        %17 = arith.addf %out, %16 : f32
        linalg.yield %17 : f32
      } -> tensor<32x2048x128xf32>
      %cast = tensor.cast %5 : tensor<1x32000xf32> to tensor<*xf32>
      func.call @print_memref_f32(%cast) : (tensor<*xf32>) -> ()
      %cst_6 = arith.constant 0xFF800000 : f32
      %9 = tensor.empty() : tensor<1xf32>
      %10 = linalg.fill ins(%cst_6 : f32) outs(%9 : tensor<1xf32>) -> tensor<1xf32>
      %c0_i64_7 = arith.constant 0 : i64
      %11 = tensor.empty() : tensor<1xi64>
      %12 = linalg.fill ins(%c0_i64_7 : i64) outs(%11 : tensor<1xi64>) -> tensor<1xi64>
      %13:2 = linalg.generic {indexing_maps = [#map6, #map7, #map7], iterator_types = ["parallel", "reduction"]} ins(%5 : tensor<1x32000xf32>) outs(%10, %12 : tensor<1xf32>, tensor<1xi64>) {
      ^bb0(%in: f32, %out: f32, %out_10: i64):
        %16 = linalg.index 1 : index
        %17 = arith.index_cast %16 : index to i64
        %18 = arith.cmpf ogt, %in, %out : f32
        %19 = arith.select %18, %in, %out : f32
        %20 = arith.select %18, %17, %out_10 : i64
        linalg.yield %19, %20 : f32, i64
      } -> (tensor<1xf32>, tensor<1xi64>)
      %c0_i64_8 = arith.constant 0 : i64
      %14 = arith.index_cast %c0_i64_8 : i64 to index
      %extracted = tensor.extract %13#1[%14] : tensor<1xi64>
      %c1_i64_9 = arith.constant 1 : i64
      %15 = arith.addi %arg1, %c1_i64_9 : i64
      scf.yield %extracted, %15, %8 : i64, i64, tensor<32x2048x128xf32>
    }
    return %0#2 : tensor<32x2048x128xf32>
  }
}

// ==========================================
// Phase: Linalg Tiling
// ==========================================
#map = affine_map<(d0) -> (-d0 + 1, 8)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map7 = affine_map<(d0) -> (d0)>
#map8 = affine_map<(d0, d1) -> (d0)>
#map9 = affine_map<(d0, d1) -> (d0 + d1)>
module {
  func.func private @print_memref_f32(tensor<*xf32> {bufferization.access = "read"})
  func.func @host() -> tensor<32x2048x128xf32> {
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
    %cst = arith.constant 0xFF800000 : f32
    %cst_20 = arith.constant 1.234200e+04 : f32
    %cst_21 = arith.constant 5.000000e-01 : f32
    %cst_22 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %cst_23 = arith.constant dense<3.214000e+03> : tensor<768x32000xf32>
    %cst_24 = arith.constant dense<0.000000e+00> : tensor<32x2048x128xf32>
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c10_i64 = arith.constant 10 : i64
    %0:3 = scf.while (%arg0 = %c1_i64, %arg1 = %c0_i64, %arg2 = %cst_24) : (i64, i64, tensor<32x2048x128xf32>) -> (i64, i64, tensor<32x2048x128xf32>) {
      %1 = arith.cmpi slt, %arg1, %c10_i64 : i64
      scf.condition(%1) %arg0, %arg1, %arg2 : i64, i64, tensor<32x2048x128xf32>
    } do {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: tensor<32x2048x128xf32>):
      %1 = tensor.empty() : tensor<1x32000xf32>
      %c0_25 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8_26 = arith.constant 8 : index
      %c0_27 = arith.constant 0 : index
      %c32000 = arith.constant 32000 : index
      %c8_28 = arith.constant 8 : index
      %2 = scf.for %arg3 = %c0_25 to %c1 step %c8_26 iter_args(%arg4 = %1) -> (tensor<1x32000xf32>) {
        %13 = scf.for %arg5 = %c0_27 to %c32000 step %c8_28 iter_args(%arg6 = %arg4) -> (tensor<1x32000xf32>) {
          %14 = affine.min #map(%arg3)
          %extracted_slice = tensor.extract_slice %arg6[%arg3, %arg5] [%14, 8] [1, 1] : tensor<1x32000xf32> to tensor<?x8xf32>
          %15 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice : tensor<?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_22 : f32
          } -> tensor<?x8xf32>
          %inserted_slice = tensor.insert_slice %15 into %arg6[%arg3, %arg5] [%14, 8] [1, 1] : tensor<?x8xf32> into tensor<1x32000xf32>
          scf.yield %inserted_slice : tensor<1x32000xf32>
        }
        scf.yield %13 : tensor<1x32000xf32>
      }
      %c0_29 = arith.constant 0 : index
      %c1_30 = arith.constant 1 : index
      %c8_31 = arith.constant 8 : index
      %c0_32 = arith.constant 0 : index
      %c32000_33 = arith.constant 32000 : index
      %c8_34 = arith.constant 8 : index
      %c0_35 = arith.constant 0 : index
      %c768 = arith.constant 768 : index
      %c8_36 = arith.constant 8 : index
      %3 = scf.for %arg3 = %c0_29 to %c1_30 step %c8_31 iter_args(%arg4 = %2) -> (tensor<1x32000xf32>) {
        %13 = scf.for %arg5 = %c0_32 to %c32000_33 step %c8_34 iter_args(%arg6 = %arg4) -> (tensor<1x32000xf32>) {
          %14 = scf.for %arg7 = %c0_35 to %c768 step %c8_36 iter_args(%arg8 = %arg6) -> (tensor<1x32000xf32>) {
            %15 = affine.min #map(%arg3)
            %extracted_slice = tensor.extract_slice %cst_23[%arg7, %arg5] [8, 8] [1, 1] : tensor<768x32000xf32> to tensor<8x8xf32>
            %extracted_slice_64 = tensor.extract_slice %arg8[%arg3, %arg5] [%15, 8] [1, 1] : tensor<1x32000xf32> to tensor<?x8xf32>
            %16 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<8x8xf32>) outs(%extracted_slice_64 : tensor<?x8xf32>) {
            ^bb0(%in: f32, %out: f32):
              %17 = arith.mulf %in, %cst_21 : f32
              %18 = arith.addf %out, %17 : f32
              linalg.yield %18 : f32
            } -> tensor<?x8xf32>
            %inserted_slice = tensor.insert_slice %16 into %arg8[%arg3, %arg5] [%15, 8] [1, 1] : tensor<?x8xf32> into tensor<1x32000xf32>
            scf.yield %inserted_slice : tensor<1x32000xf32>
          }
          scf.yield %14 : tensor<1x32000xf32>
        }
        scf.yield %13 : tensor<1x32000xf32>
      }
      %4 = tensor.empty() : tensor<32x2048x128xf32>
      %c0_37 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c8_38 = arith.constant 8 : index
      %c0_39 = arith.constant 0 : index
      %c2048 = arith.constant 2048 : index
      %c8_40 = arith.constant 8 : index
      %c0_41 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c8_42 = arith.constant 8 : index
      %5 = scf.for %arg3 = %c0_37 to %c32 step %c8_38 iter_args(%arg4 = %4) -> (tensor<32x2048x128xf32>) {
        %13 = scf.for %arg5 = %c0_39 to %c2048 step %c8_40 iter_args(%arg6 = %arg4) -> (tensor<32x2048x128xf32>) {
          %14 = scf.for %arg7 = %c0_41 to %c128 step %c8_42 iter_args(%arg8 = %arg6) -> (tensor<32x2048x128xf32>) {
            %extracted_slice = tensor.extract_slice %arg8[%arg3, %arg5, %arg7] [8, 8, 8] [1, 1, 1] : tensor<32x2048x128xf32> to tensor<8x8x8xf32>
            %15 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<8x8x8xf32>) {
            ^bb0(%out: f32):
              linalg.yield %cst_22 : f32
            } -> tensor<8x8x8xf32>
            %inserted_slice = tensor.insert_slice %15 into %arg8[%arg3, %arg5, %arg7] [8, 8, 8] [1, 1, 1] : tensor<8x8x8xf32> into tensor<32x2048x128xf32>
            scf.yield %inserted_slice : tensor<32x2048x128xf32>
          }
          scf.yield %14 : tensor<32x2048x128xf32>
        }
        scf.yield %13 : tensor<32x2048x128xf32>
      }
      %c0_43 = arith.constant 0 : index
      %c32_44 = arith.constant 32 : index
      %c8_45 = arith.constant 8 : index
      %c0_46 = arith.constant 0 : index
      %c2048_47 = arith.constant 2048 : index
      %c8_48 = arith.constant 8 : index
      %c0_49 = arith.constant 0 : index
      %c128_50 = arith.constant 128 : index
      %c8_51 = arith.constant 8 : index
      %6 = scf.for %arg3 = %c0_43 to %c32_44 step %c8_45 iter_args(%arg4 = %5) -> (tensor<32x2048x128xf32>) {
        %13 = scf.for %arg5 = %c0_46 to %c2048_47 step %c8_48 iter_args(%arg6 = %arg4) -> (tensor<32x2048x128xf32>) {
          %14 = scf.for %arg7 = %c0_49 to %c128_50 step %c8_51 iter_args(%arg8 = %arg6) -> (tensor<32x2048x128xf32>) {
            %extracted_slice = tensor.extract_slice %arg2[%arg3, %arg5, 0] [8, 8, 128] [1, 1, 1] : tensor<32x2048x128xf32> to tensor<8x8x128xf32>
            %extracted_slice_64 = tensor.extract_slice %arg8[%arg3, %arg5, %arg7] [8, 8, 8] [1, 1, 1] : tensor<32x2048x128xf32> to tensor<8x8x8xf32>
            %15 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<8x8x128xf32>) outs(%extracted_slice_64 : tensor<8x8x8xf32>) {
            ^bb0(%in: f32, %out: f32):
              %16 = arith.mulf %in, %cst_20 : f32
              %17 = arith.addf %out, %16 : f32
              linalg.yield %17 : f32
            } -> tensor<8x8x8xf32>
            %inserted_slice = tensor.insert_slice %15 into %arg8[%arg3, %arg5, %arg7] [8, 8, 8] [1, 1, 1] : tensor<8x8x8xf32> into tensor<32x2048x128xf32>
            scf.yield %inserted_slice : tensor<32x2048x128xf32>
          }
          scf.yield %14 : tensor<32x2048x128xf32>
        }
        scf.yield %13 : tensor<32x2048x128xf32>
      }
      %cast = tensor.cast %3 : tensor<1x32000xf32> to tensor<*xf32>
      func.call @print_memref_f32(%cast) : (tensor<*xf32>) -> ()
      %7 = tensor.empty() : tensor<1xf32>
      %c0_52 = arith.constant 0 : index
      %c1_53 = arith.constant 1 : index
      %c8_54 = arith.constant 8 : index
      %8 = scf.for %arg3 = %c0_52 to %c1_53 step %c8_54 iter_args(%arg4 = %7) -> (tensor<1xf32>) {
        %13 = affine.min #map(%arg3)
        %extracted_slice = tensor.extract_slice %arg4[%arg3] [%13] [1] : tensor<1xf32> to tensor<?xf32>
        %14 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel"]} outs(%extracted_slice : tensor<?xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<?xf32>
        %inserted_slice = tensor.insert_slice %14 into %arg4[%arg3] [%13] [1] : tensor<?xf32> into tensor<1xf32>
        scf.yield %inserted_slice : tensor<1xf32>
      }
      %9 = tensor.empty() : tensor<1xi64>
      %c0_55 = arith.constant 0 : index
      %c1_56 = arith.constant 1 : index
      %c8_57 = arith.constant 8 : index
      %10 = scf.for %arg3 = %c0_55 to %c1_56 step %c8_57 iter_args(%arg4 = %9) -> (tensor<1xi64>) {
        %13 = affine.min #map(%arg3)
        %extracted_slice = tensor.extract_slice %arg4[%arg3] [%13] [1] : tensor<1xi64> to tensor<?xi64>
        %14 = linalg.generic {indexing_maps = [#map7], iterator_types = ["parallel"]} outs(%extracted_slice : tensor<?xi64>) {
        ^bb0(%out: i64):
          linalg.yield %c0_i64 : i64
        } -> tensor<?xi64>
        %inserted_slice = tensor.insert_slice %14 into %arg4[%arg3] [%13] [1] : tensor<?xi64> into tensor<1xi64>
        scf.yield %inserted_slice : tensor<1xi64>
      }
      %c0_58 = arith.constant 0 : index
      %c1_59 = arith.constant 1 : index
      %c8_60 = arith.constant 8 : index
      %c0_61 = arith.constant 0 : index
      %c32000_62 = arith.constant 32000 : index
      %c8_63 = arith.constant 8 : index
      %11:2 = scf.for %arg3 = %c0_58 to %c1_59 step %c8_60 iter_args(%arg4 = %8, %arg5 = %10) -> (tensor<1xf32>, tensor<1xi64>) {
        %13:2 = scf.for %arg6 = %c0_61 to %c32000_62 step %c8_63 iter_args(%arg7 = %arg4, %arg8 = %arg5) -> (tensor<1xf32>, tensor<1xi64>) {
          %14 = affine.min #map(%arg3)
          %15 = affine.min #map(%arg3)
          %16 = affine.min #map(%arg3)
          %extracted_slice = tensor.extract_slice %3[%arg3, %arg6] [%14, 8] [1, 1] : tensor<1x32000xf32> to tensor<?x8xf32>
          %extracted_slice_64 = tensor.extract_slice %arg7[%arg3] [%15] [1] : tensor<1xf32> to tensor<?xf32>
          %extracted_slice_65 = tensor.extract_slice %arg8[%arg3] [%16] [1] : tensor<1xi64> to tensor<?xi64>
          %17:2 = linalg.generic {indexing_maps = [#map1, #map8, #map8], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice : tensor<?x8xf32>) outs(%extracted_slice_64, %extracted_slice_65 : tensor<?xf32>, tensor<?xi64>) {
          ^bb0(%in: f32, %out: f32, %out_67: i64):
            %18 = linalg.index 1 : index
            %19 = affine.apply #map9(%18, %arg6)
            %20 = arith.index_cast %19 : index to i64
            %21 = arith.cmpf ogt, %in, %out : f32
            %22 = arith.select %21, %in, %out : f32
            %23 = arith.select %21, %20, %out_67 : i64
            linalg.yield %22, %23 : f32, i64
          } -> (tensor<?xf32>, tensor<?xi64>)
          %inserted_slice = tensor.insert_slice %17#0 into %arg7[%arg3] [%15] [1] : tensor<?xf32> into tensor<1xf32>
          %inserted_slice_66 = tensor.insert_slice %17#1 into %arg8[%arg3] [%16] [1] : tensor<?xi64> into tensor<1xi64>
          scf.yield %inserted_slice, %inserted_slice_66 : tensor<1xf32>, tensor<1xi64>
        }
        scf.yield %13#0, %13#1 : tensor<1xf32>, tensor<1xi64>
      }
      %extracted = tensor.extract %11#1[%c0] : tensor<1xi64>
      %12 = arith.addi %arg1, %c1_i64 : i64
      scf.yield %extracted, %12, %6 : i64, i64, tensor<32x2048x128xf32>
    }
    return %0#2 : tensor<32x2048x128xf32>
  }
}

// ==========================================
// Phase: Bufferization
// ==========================================
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map6 = affine_map<(d0) -> (d0)>
#map7 = affine_map<(d0, d1) -> (d0)>
#map8 = affine_map<(d0, d1) -> (d0 + d1)>
module {
  memref.global "private" constant @__constant_32x2048x128xf32 : memref<32x2048x128xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8x8xf32 : memref<8x8xf32> = dense<3.214000e+03> {alignment = 64 : i64}
  func.func private @print_memref_f32(memref<*xf32> {bufferization.access = "read"})
  func.func @host() -> memref<32x2048x128xf32> {
    %c10_i64 = arith.constant 10 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 1.234200e+04 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %c8 = arith.constant 8 : index
    %c32000 = arith.constant 32000 : index
    %c768 = arith.constant 768 : index
    %c32 = arith.constant 32 : index
    %c2048 = arith.constant 2048 : index
    %c128 = arith.constant 128 : index
    %0 = memref.get_global @__constant_8x8xf32 : memref<8x8xf32>
    %1 = memref.get_global @__constant_32x2048x128xf32 : memref<32x2048x128xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x2048x128xf32>
    memref.copy %1, %alloc : memref<32x2048x128xf32> to memref<32x2048x128xf32>
    %2:2 = scf.while (%arg0 = %c0_i64, %arg1 = %alloc) : (i64, memref<32x2048x128xf32>) -> (i64, memref<32x2048x128xf32>) {
      %3 = arith.cmpi slt, %arg0, %c10_i64 : i64
      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32x2048x128xf32>
      memref.copy %arg1, %alloc_3 : memref<32x2048x128xf32> to memref<32x2048x128xf32>
      scf.condition(%3) %arg0, %alloc_3 : i64, memref<32x2048x128xf32>
    } do {
    ^bb0(%arg0: i64, %arg1: memref<32x2048x128xf32>):
      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x32000xf32>
      scf.for %arg2 = %c0 to %c32000 step %c8 {
        %subview = memref.subview %alloc_3[0, %arg2] [1, 8] [1, 1] : memref<1x32000xf32> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%subview : memref<1x8xf32, strided<[32000, 1], offset: ?>>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        }
        memref.copy %subview, %subview : memref<1x8xf32, strided<[32000, 1], offset: ?>> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
      }
      scf.for %arg2 = %c0 to %c32000 step %c8 {
        scf.for %arg3 = %c0 to %c768 step %c8 {
          %subview = memref.subview %alloc_3[0, %arg2] [1, 8] [1, 1] : memref<1x32000xf32> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0 : memref<8x8xf32>) outs(%subview : memref<1x8xf32, strided<[32000, 1], offset: ?>>) {
          ^bb0(%in: f32, %out: f32):
            %4 = arith.mulf %in, %cst_0 : f32
            %5 = arith.addf %out, %4 : f32
            linalg.yield %5 : f32
          }
          memref.copy %subview, %subview : memref<1x8xf32, strided<[32000, 1], offset: ?>> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
        }
      }
      %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x2048x128xf32>
      scf.for %arg2 = %c0 to %c32 step %c8 {
        scf.for %arg3 = %c0 to %c2048 step %c8 {
          scf.for %arg4 = %c0 to %c128 step %c8 {
            %subview = memref.subview %alloc_4[%arg2, %arg3, %arg4] [8, 8, 8] [1, 1, 1] : memref<32x2048x128xf32> to memref<8x8x8xf32, strided<[262144, 128, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel", "parallel"]} outs(%subview : memref<8x8x8xf32, strided<[262144, 128, 1], offset: ?>>) {
            ^bb0(%out: f32):
              linalg.yield %cst : f32
            }
            memref.copy %subview, %subview : memref<8x8x8xf32, strided<[262144, 128, 1], offset: ?>> to memref<8x8x8xf32, strided<[262144, 128, 1], offset: ?>>
          }
        }
      }
      scf.for %arg2 = %c0 to %c32 step %c8 {
        scf.for %arg3 = %c0 to %c2048 step %c8 {
          scf.for %arg4 = %c0 to %c128 step %c8 {
            %subview = memref.subview %arg1[%arg2, %arg3, 0] [8, 8, 128] [1, 1, 1] : memref<32x2048x128xf32> to memref<8x8x128xf32, strided<[262144, 128, 1], offset: ?>>
            %subview_7 = memref.subview %alloc_4[%arg2, %arg3, %arg4] [8, 8, 8] [1, 1, 1] : memref<32x2048x128xf32> to memref<8x8x8xf32, strided<[262144, 128, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%subview : memref<8x8x128xf32, strided<[262144, 128, 1], offset: ?>>) outs(%subview_7 : memref<8x8x8xf32, strided<[262144, 128, 1], offset: ?>>) {
            ^bb0(%in: f32, %out: f32):
              %4 = arith.mulf %in, %cst_1 : f32
              %5 = arith.addf %out, %4 : f32
              linalg.yield %5 : f32
            }
            memref.copy %subview_7, %subview_7 : memref<8x8x8xf32, strided<[262144, 128, 1], offset: ?>> to memref<8x8x8xf32, strided<[262144, 128, 1], offset: ?>>
          }
        }
      }
      %cast = memref.cast %alloc_3 : memref<1x32000xf32> to memref<*xf32>
      func.call @print_memref_f32(%cast) : (memref<*xf32>) -> ()
      %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      linalg.generic {indexing_maps = [#map6], iterator_types = ["parallel"]} outs(%alloc_5 : memref<1xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_2 : f32
      }
      %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1xi64>
      linalg.generic {indexing_maps = [#map6], iterator_types = ["parallel"]} outs(%alloc_6 : memref<1xi64>) {
      ^bb0(%out: i64):
        linalg.yield %c0_i64 : i64
      }
      scf.for %arg2 = %c0 to %c32000 step %c8 {
        %subview = memref.subview %alloc_3[0, %arg2] [1, 8] [1, 1] : memref<1x32000xf32> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map, #map7, #map7], iterator_types = ["parallel", "reduction"]} ins(%subview : memref<1x8xf32, strided<[32000, 1], offset: ?>>) outs(%alloc_5, %alloc_6 : memref<1xf32>, memref<1xi64>) {
        ^bb0(%in: f32, %out: f32, %out_7: i64):
          %4 = linalg.index 1 : index
          %5 = affine.apply #map8(%4, %arg2)
          %6 = arith.index_cast %5 : index to i64
          %7 = arith.cmpf ogt, %in, %out : f32
          %8 = arith.select %7, %in, %out : f32
          %9 = arith.select %7, %6, %out_7 : i64
          linalg.yield %8, %9 : f32, i64
        }
      }
      %3 = arith.addi %arg0, %c1_i64 : i64
      scf.yield %3, %alloc_4 : i64, memref<32x2048x128xf32>
    }
    return %2#1 : memref<32x2048x128xf32>
  }
}

// ==========================================
// Phase: linalg to scf
// ==========================================
#map = affine_map<(d0, d1) -> (d0 + d1)>
module {
  memref.global "private" constant @__constant_32x2048x128xf32 : memref<32x2048x128xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8x8xf32 : memref<8x8xf32> = dense<3.214000e+03> {alignment = 64 : i64}
  func.func private @print_memref_f32(memref<*xf32> {bufferization.access = "read"})
  func.func @host() -> memref<32x2048x128xf32> {
    %c1 = arith.constant 1 : index
    %c10_i64 = arith.constant 10 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 1.234200e+04 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %c8 = arith.constant 8 : index
    %c32000 = arith.constant 32000 : index
    %c768 = arith.constant 768 : index
    %c32 = arith.constant 32 : index
    %c2048 = arith.constant 2048 : index
    %c128 = arith.constant 128 : index
    %0 = memref.get_global @__constant_8x8xf32 : memref<8x8xf32>
    %1 = memref.get_global @__constant_32x2048x128xf32 : memref<32x2048x128xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x2048x128xf32>
    memref.copy %1, %alloc : memref<32x2048x128xf32> to memref<32x2048x128xf32>
    %2:2 = scf.while (%arg0 = %c0_i64, %arg1 = %alloc) : (i64, memref<32x2048x128xf32>) -> (i64, memref<32x2048x128xf32>) {
      %3 = arith.cmpi slt, %arg0, %c10_i64 : i64
      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32x2048x128xf32>
      memref.copy %arg1, %alloc_3 : memref<32x2048x128xf32> to memref<32x2048x128xf32>
      scf.condition(%3) %arg0, %alloc_3 : i64, memref<32x2048x128xf32>
    } do {
    ^bb0(%arg0: i64, %arg1: memref<32x2048x128xf32>):
      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x32000xf32>
      scf.for %arg2 = %c0 to %c32000 step %c8 {
        %subview = memref.subview %alloc_3[0, %arg2] [1, 8] [1, 1] : memref<1x32000xf32> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
        scf.for %arg3 = %c0 to %c1 step %c1 {
          scf.for %arg4 = %c0 to %c8 step %c1 {
            memref.store %cst, %subview[%arg3, %arg4] : memref<1x8xf32, strided<[32000, 1], offset: ?>>
          }
        }
        memref.copy %subview, %subview : memref<1x8xf32, strided<[32000, 1], offset: ?>> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
      }
      scf.for %arg2 = %c0 to %c32000 step %c8 {
        scf.for %arg3 = %c0 to %c768 step %c8 {
          %subview = memref.subview %alloc_3[0, %arg2] [1, 8] [1, 1] : memref<1x32000xf32> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
          scf.for %arg4 = %c0 to %c1 step %c1 {
            scf.for %arg5 = %c0 to %c8 step %c1 {
              scf.for %arg6 = %c0 to %c8 step %c1 {
                %4 = memref.load %0[%arg6, %arg5] : memref<8x8xf32>
                %5 = memref.load %subview[%arg4, %arg5] : memref<1x8xf32, strided<[32000, 1], offset: ?>>
                %6 = arith.mulf %4, %cst_0 : f32
                %7 = arith.addf %5, %6 : f32
                memref.store %7, %subview[%arg4, %arg5] : memref<1x8xf32, strided<[32000, 1], offset: ?>>
              }
            }
          }
          memref.copy %subview, %subview : memref<1x8xf32, strided<[32000, 1], offset: ?>> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
        }
      }
      %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x2048x128xf32>
      scf.for %arg2 = %c0 to %c32 step %c8 {
        scf.for %arg3 = %c0 to %c2048 step %c8 {
          scf.for %arg4 = %c0 to %c128 step %c8 {
            %subview = memref.subview %alloc_4[%arg2, %arg3, %arg4] [8, 8, 8] [1, 1, 1] : memref<32x2048x128xf32> to memref<8x8x8xf32, strided<[262144, 128, 1], offset: ?>>
            scf.for %arg5 = %c0 to %c8 step %c1 {
              scf.for %arg6 = %c0 to %c8 step %c1 {
                scf.for %arg7 = %c0 to %c8 step %c1 {
                  memref.store %cst, %subview[%arg5, %arg6, %arg7] : memref<8x8x8xf32, strided<[262144, 128, 1], offset: ?>>
                }
              }
            }
            memref.copy %subview, %subview : memref<8x8x8xf32, strided<[262144, 128, 1], offset: ?>> to memref<8x8x8xf32, strided<[262144, 128, 1], offset: ?>>
          }
        }
      }
      scf.for %arg2 = %c0 to %c32 step %c8 {
        scf.for %arg3 = %c0 to %c2048 step %c8 {
          scf.for %arg4 = %c0 to %c128 step %c8 {
            %subview = memref.subview %arg1[%arg2, %arg3, 0] [8, 8, 128] [1, 1, 1] : memref<32x2048x128xf32> to memref<8x8x128xf32, strided<[262144, 128, 1], offset: ?>>
            %subview_7 = memref.subview %alloc_4[%arg2, %arg3, %arg4] [8, 8, 8] [1, 1, 1] : memref<32x2048x128xf32> to memref<8x8x8xf32, strided<[262144, 128, 1], offset: ?>>
            scf.for %arg5 = %c0 to %c8 step %c1 {
              scf.for %arg6 = %c0 to %c8 step %c1 {
                scf.for %arg7 = %c0 to %c8 step %c1 {
                  scf.for %arg8 = %c0 to %c128 step %c1 {
                    %4 = memref.load %subview[%arg5, %arg6, %arg8] : memref<8x8x128xf32, strided<[262144, 128, 1], offset: ?>>
                    %5 = memref.load %subview_7[%arg5, %arg6, %arg7] : memref<8x8x8xf32, strided<[262144, 128, 1], offset: ?>>
                    %6 = arith.mulf %4, %cst_1 : f32
                    %7 = arith.addf %5, %6 : f32
                    memref.store %7, %subview_7[%arg5, %arg6, %arg7] : memref<8x8x8xf32, strided<[262144, 128, 1], offset: ?>>
                  }
                }
              }
            }
            memref.copy %subview_7, %subview_7 : memref<8x8x8xf32, strided<[262144, 128, 1], offset: ?>> to memref<8x8x8xf32, strided<[262144, 128, 1], offset: ?>>
          }
        }
      }
      %cast = memref.cast %alloc_3 : memref<1x32000xf32> to memref<*xf32>
      func.call @print_memref_f32(%cast) : (memref<*xf32>) -> ()
      %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      scf.for %arg2 = %c0 to %c1 step %c1 {
        memref.store %cst_2, %alloc_5[%arg2] : memref<1xf32>
      }
      %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1xi64>
      scf.for %arg2 = %c0 to %c1 step %c1 {
        memref.store %c0_i64, %alloc_6[%arg2] : memref<1xi64>
      }
      scf.for %arg2 = %c0 to %c32000 step %c8 {
        %subview = memref.subview %alloc_3[0, %arg2] [1, 8] [1, 1] : memref<1x32000xf32> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
        scf.for %arg3 = %c0 to %c1 step %c1 {
          scf.for %arg4 = %c0 to %c8 step %c1 {
            %4 = memref.load %subview[%arg3, %arg4] : memref<1x8xf32, strided<[32000, 1], offset: ?>>
            %5 = memref.load %alloc_5[%arg3] : memref<1xf32>
            %6 = memref.load %alloc_6[%arg3] : memref<1xi64>
            %7 = affine.apply #map(%arg4, %arg2)
            %8 = arith.index_cast %7 : index to i64
            %9 = arith.cmpf ogt, %4, %5 : f32
            %10 = arith.select %9, %4, %5 : f32
            %11 = arith.select %9, %8, %6 : i64
            memref.store %10, %alloc_5[%arg3] : memref<1xf32>
            memref.store %11, %alloc_6[%arg3] : memref<1xi64>
          }
        }
      }
      %3 = arith.addi %arg0, %c1_i64 : i64
      scf.yield %3, %alloc_4 : i64, memref<32x2048x128xf32>
    }
    return %2#1 : memref<32x2048x128xf32>
  }
}

// ==========================================
// Phase: lower to llvm
// ==========================================
module {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_32x2048x128xf32(dense<0.000000e+00> : tensor<32x2048x128xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<32 x array<2048 x array<128 x f32>>>
  llvm.mlir.global private constant @__constant_8x8xf32(dense<3.214000e+03> : tensor<8x8xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<8 x array<8 x f32>>
  llvm.func @print_memref_f32(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @host() -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> {
    %0 = llvm.mlir.constant(2 : index) : i64
    %1 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %2 = llvm.mlir.constant(3 : i64) : i64
    %3 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %4 = llvm.mlir.addressof @__constant_32x2048x128xf32 : !llvm.ptr
    %5 = llvm.mlir.constant(262144 : index) : i64
    %6 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.mlir.addressof @__constant_8x8xf32 : !llvm.ptr
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(10 : i64) : i64
    %10 = llvm.mlir.constant(0 : i64) : i64
    %11 = llvm.mlir.constant(1 : i64) : i64
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %14 = llvm.mlir.constant(5.000000e-01 : f32) : f32
    %15 = llvm.mlir.constant(1.234200e+04 : f32) : f32
    %16 = llvm.mlir.constant(0xFF800000 : f32) : f32
    %17 = llvm.mlir.constant(8 : index) : i64
    %18 = llvm.mlir.constant(32000 : index) : i64
    %19 = llvm.mlir.constant(768 : index) : i64
    %20 = llvm.mlir.constant(32 : index) : i64
    %21 = llvm.mlir.constant(2048 : index) : i64
    %22 = llvm.mlir.constant(128 : index) : i64
    %23 = llvm.mlir.constant(64 : index) : i64
    %24 = llvm.mlir.zero : !llvm.ptr
    %25 = llvm.getelementptr %7[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x array<8 x f32>>
    %26 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<2048 x array<128 x f32>>>
    %27 = llvm.getelementptr %24[8388608] : (!llvm.ptr) -> !llvm.ptr, f32
    %28 = llvm.ptrtoint %27 : !llvm.ptr to i64
    %29 = llvm.add %28, %23 : i64
    %30 = llvm.call @malloc(%29) : (i64) -> !llvm.ptr
    %31 = llvm.ptrtoint %30 : !llvm.ptr to i64
    %32 = llvm.sub %23, %8 : i64
    %33 = llvm.add %31, %32 : i64
    %34 = llvm.urem %33, %23  : i64
    %35 = llvm.sub %33, %34 : i64
    %36 = llvm.inttoptr %35 : i64 to !llvm.ptr
    %37 = llvm.insertvalue %30, %3[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %38 = llvm.insertvalue %36, %37[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %39 = llvm.insertvalue %12, %38[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %40 = llvm.insertvalue %20, %39[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %41 = llvm.insertvalue %21, %40[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %42 = llvm.insertvalue %22, %41[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %43 = llvm.insertvalue %5, %42[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %44 = llvm.insertvalue %22, %43[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %45 = llvm.insertvalue %8, %44[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %46 = llvm.mul %20, %8 : i64
    %47 = llvm.mul %46, %21 : i64
    %48 = llvm.mul %47, %22 : i64
    %49 = llvm.getelementptr %24[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %50 = llvm.ptrtoint %49 : !llvm.ptr to i64
    %51 = llvm.mul %48, %50 : i64
    "llvm.intr.memcpy"(%36, %26, %51) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb1(%10, %45 : i64, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)
  ^bb1(%52: i64, %53: !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>):  // 2 preds: ^bb0, ^bb80
    %54 = llvm.icmp "slt" %52, %9 : i64
    %55 = llvm.call @malloc(%29) : (i64) -> !llvm.ptr
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.add %56, %32 : i64
    %58 = llvm.urem %57, %23  : i64
    %59 = llvm.sub %57, %58 : i64
    %60 = llvm.inttoptr %59 : i64 to !llvm.ptr
    %61 = llvm.insertvalue %55, %3[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %62 = llvm.insertvalue %60, %61[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %63 = llvm.insertvalue %12, %62[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %64 = llvm.insertvalue %20, %63[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %65 = llvm.insertvalue %21, %64[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %66 = llvm.insertvalue %22, %65[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %67 = llvm.insertvalue %5, %66[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %68 = llvm.insertvalue %22, %67[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %69 = llvm.insertvalue %8, %68[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %70 = llvm.extractvalue %53[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %71 = llvm.mul %70, %8 : i64
    %72 = llvm.extractvalue %53[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %73 = llvm.mul %71, %72 : i64
    %74 = llvm.extractvalue %53[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %75 = llvm.mul %73, %74 : i64
    %76 = llvm.mul %75, %50 : i64
    %77 = llvm.extractvalue %53[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %78 = llvm.extractvalue %53[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %79 = llvm.getelementptr %77[%78] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%60, %79, %76) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.cond_br %54, ^bb2(%52, %69 : i64, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>), ^bb81
  ^bb2(%80: i64, %81: !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>):  // pred: ^bb1
    %82 = llvm.getelementptr %24[32000] : (!llvm.ptr) -> !llvm.ptr, f32
    %83 = llvm.ptrtoint %82 : !llvm.ptr to i64
    %84 = llvm.add %83, %23 : i64
    %85 = llvm.call @malloc(%84) : (i64) -> !llvm.ptr
    %86 = llvm.ptrtoint %85 : !llvm.ptr to i64
    %87 = llvm.add %86, %32 : i64
    %88 = llvm.urem %87, %23  : i64
    %89 = llvm.sub %87, %88 : i64
    %90 = llvm.inttoptr %89 : i64 to !llvm.ptr
    %91 = llvm.insertvalue %85, %6[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.insertvalue %90, %91[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.insertvalue %12, %92[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.insertvalue %8, %93[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %95 = llvm.insertvalue %18, %94[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.insertvalue %18, %95[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %97 = llvm.insertvalue %8, %96[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb3(%12 : i64)
  ^bb3(%98: i64):  // 2 preds: ^bb2, ^bb10
    %99 = llvm.icmp "slt" %98, %18 : i64
    llvm.cond_br %99, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%12 : i64)
  ^bb5(%100: i64):  // 2 preds: ^bb4, ^bb9
    %101 = llvm.icmp "slt" %100, %8 : i64
    llvm.cond_br %101, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%12 : i64)
  ^bb7(%102: i64):  // 2 preds: ^bb6, ^bb8
    %103 = llvm.icmp "slt" %102, %17 : i64
    llvm.cond_br %103, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %104 = llvm.getelementptr %90[%98] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %105 = llvm.mul %100, %18 : i64
    %106 = llvm.add %105, %102 : i64
    %107 = llvm.getelementptr %104[%106] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %13, %107 : f32, !llvm.ptr
    %108 = llvm.add %102, %8 : i64
    llvm.br ^bb7(%108 : i64)
  ^bb9:  // pred: ^bb7
    %109 = llvm.add %100, %8 : i64
    llvm.br ^bb5(%109 : i64)
  ^bb10:  // pred: ^bb5
    %110 = llvm.mul %8, %8 : i64
    %111 = llvm.mul %110, %17 : i64
    %112 = llvm.mul %111, %50 : i64
    %113 = llvm.getelementptr %90[%98] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%113, %113, %112) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %114 = llvm.add %98, %17 : i64
    llvm.br ^bb3(%114 : i64)
  ^bb11:  // pred: ^bb3
    llvm.br ^bb12(%12 : i64)
  ^bb12(%115: i64):  // 2 preds: ^bb11, ^bb25
    %116 = llvm.icmp "slt" %115, %18 : i64
    llvm.cond_br %116, ^bb13, ^bb26
  ^bb13:  // pred: ^bb12
    llvm.br ^bb14(%12 : i64)
  ^bb14(%117: i64):  // 2 preds: ^bb13, ^bb24
    %118 = llvm.icmp "slt" %117, %19 : i64
    llvm.cond_br %118, ^bb15, ^bb25
  ^bb15:  // pred: ^bb14
    llvm.br ^bb16(%12 : i64)
  ^bb16(%119: i64):  // 2 preds: ^bb15, ^bb23
    %120 = llvm.icmp "slt" %119, %8 : i64
    llvm.cond_br %120, ^bb17, ^bb24
  ^bb17:  // pred: ^bb16
    llvm.br ^bb18(%12 : i64)
  ^bb18(%121: i64):  // 2 preds: ^bb17, ^bb22
    %122 = llvm.icmp "slt" %121, %17 : i64
    llvm.cond_br %122, ^bb19, ^bb23
  ^bb19:  // pred: ^bb18
    llvm.br ^bb20(%12 : i64)
  ^bb20(%123: i64):  // 2 preds: ^bb19, ^bb21
    %124 = llvm.icmp "slt" %123, %17 : i64
    llvm.cond_br %124, ^bb21, ^bb22
  ^bb21:  // pred: ^bb20
    %125 = llvm.mul %123, %17 : i64
    %126 = llvm.add %125, %121 : i64
    %127 = llvm.getelementptr %25[%126] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %128 = llvm.load %127 : !llvm.ptr -> f32
    %129 = llvm.getelementptr %90[%115] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %130 = llvm.mul %119, %18 : i64
    %131 = llvm.add %130, %121 : i64
    %132 = llvm.getelementptr %129[%131] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %133 = llvm.load %132 : !llvm.ptr -> f32
    %134 = llvm.fmul %128, %14  : f32
    %135 = llvm.fadd %133, %134  : f32
    llvm.store %135, %132 : f32, !llvm.ptr
    %136 = llvm.add %123, %8 : i64
    llvm.br ^bb20(%136 : i64)
  ^bb22:  // pred: ^bb20
    %137 = llvm.add %121, %8 : i64
    llvm.br ^bb18(%137 : i64)
  ^bb23:  // pred: ^bb18
    %138 = llvm.add %119, %8 : i64
    llvm.br ^bb16(%138 : i64)
  ^bb24:  // pred: ^bb16
    %139 = llvm.mul %8, %8 : i64
    %140 = llvm.mul %139, %17 : i64
    %141 = llvm.mul %140, %50 : i64
    %142 = llvm.getelementptr %90[%115] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%142, %142, %141) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %143 = llvm.add %117, %17 : i64
    llvm.br ^bb14(%143 : i64)
  ^bb25:  // pred: ^bb14
    %144 = llvm.add %115, %17 : i64
    llvm.br ^bb12(%144 : i64)
  ^bb26:  // pred: ^bb12
    %145 = llvm.call @malloc(%29) : (i64) -> !llvm.ptr
    %146 = llvm.ptrtoint %145 : !llvm.ptr to i64
    %147 = llvm.add %146, %32 : i64
    %148 = llvm.urem %147, %23  : i64
    %149 = llvm.sub %147, %148 : i64
    %150 = llvm.inttoptr %149 : i64 to !llvm.ptr
    %151 = llvm.insertvalue %145, %3[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %152 = llvm.insertvalue %150, %151[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %153 = llvm.insertvalue %12, %152[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %154 = llvm.insertvalue %20, %153[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %155 = llvm.insertvalue %21, %154[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %156 = llvm.insertvalue %22, %155[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %157 = llvm.insertvalue %5, %156[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %158 = llvm.insertvalue %22, %157[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %159 = llvm.insertvalue %8, %158[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb27(%12 : i64)
  ^bb27(%160: i64):  // 2 preds: ^bb26, ^bb43
    %161 = llvm.icmp "slt" %160, %20 : i64
    llvm.cond_br %161, ^bb28, ^bb44
  ^bb28:  // pred: ^bb27
    llvm.br ^bb29(%12 : i64)
  ^bb29(%162: i64):  // 2 preds: ^bb28, ^bb42
    %163 = llvm.icmp "slt" %162, %21 : i64
    llvm.cond_br %163, ^bb30, ^bb43
  ^bb30:  // pred: ^bb29
    llvm.br ^bb31(%12 : i64)
  ^bb31(%164: i64):  // 2 preds: ^bb30, ^bb41
    %165 = llvm.icmp "slt" %164, %22 : i64
    llvm.cond_br %165, ^bb32, ^bb42
  ^bb32:  // pred: ^bb31
    %166 = llvm.mul %160, %5 : i64
    %167 = llvm.mul %162, %22 : i64
    %168 = llvm.add %166, %167 : i64
    %169 = llvm.add %168, %164 : i64
    %170 = llvm.insertvalue %169, %152[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %171 = llvm.insertvalue %17, %170[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %172 = llvm.insertvalue %5, %171[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %173 = llvm.insertvalue %17, %172[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %174 = llvm.insertvalue %22, %173[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %175 = llvm.insertvalue %17, %174[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %176 = llvm.insertvalue %8, %175[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb33(%12 : i64)
  ^bb33(%177: i64):  // 2 preds: ^bb32, ^bb40
    %178 = llvm.icmp "slt" %177, %17 : i64
    llvm.cond_br %178, ^bb34, ^bb41
  ^bb34:  // pred: ^bb33
    llvm.br ^bb35(%12 : i64)
  ^bb35(%179: i64):  // 2 preds: ^bb34, ^bb39
    %180 = llvm.icmp "slt" %179, %17 : i64
    llvm.cond_br %180, ^bb36, ^bb40
  ^bb36:  // pred: ^bb35
    llvm.br ^bb37(%12 : i64)
  ^bb37(%181: i64):  // 2 preds: ^bb36, ^bb38
    %182 = llvm.icmp "slt" %181, %17 : i64
    llvm.cond_br %182, ^bb38, ^bb39
  ^bb38:  // pred: ^bb37
    %183 = llvm.getelementptr %150[%169] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %184 = llvm.mul %177, %5 : i64
    %185 = llvm.mul %179, %22 : i64
    %186 = llvm.add %184, %185 : i64
    %187 = llvm.add %186, %181 : i64
    %188 = llvm.getelementptr %183[%187] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %13, %188 : f32, !llvm.ptr
    %189 = llvm.add %181, %8 : i64
    llvm.br ^bb37(%189 : i64)
  ^bb39:  // pred: ^bb37
    %190 = llvm.add %179, %8 : i64
    llvm.br ^bb35(%190 : i64)
  ^bb40:  // pred: ^bb35
    %191 = llvm.add %177, %8 : i64
    llvm.br ^bb33(%191 : i64)
  ^bb41:  // pred: ^bb33
    %192 = llvm.intr.stacksave : !llvm.ptr
    %193 = llvm.alloca %8 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %176, %193 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %194 = llvm.insertvalue %2, %1[0] : !llvm.struct<(i64, ptr)> 
    %195 = llvm.insertvalue %193, %194[1] : !llvm.struct<(i64, ptr)> 
    %196 = llvm.alloca %8 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %176, %196 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %197 = llvm.insertvalue %196, %194[1] : !llvm.struct<(i64, ptr)> 
    %198 = llvm.alloca %8 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %195, %198 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %199 = llvm.alloca %8 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %197, %199 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%50, %198, %199) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %192 : !llvm.ptr
    %200 = llvm.add %164, %17 : i64
    llvm.br ^bb31(%200 : i64)
  ^bb42:  // pred: ^bb31
    %201 = llvm.add %162, %17 : i64
    llvm.br ^bb29(%201 : i64)
  ^bb43:  // pred: ^bb29
    %202 = llvm.add %160, %17 : i64
    llvm.br ^bb27(%202 : i64)
  ^bb44:  // pred: ^bb27
    llvm.br ^bb45(%12 : i64)
  ^bb45(%203: i64):  // 2 preds: ^bb44, ^bb64
    %204 = llvm.icmp "slt" %203, %20 : i64
    llvm.cond_br %204, ^bb46, ^bb65
  ^bb46:  // pred: ^bb45
    llvm.br ^bb47(%12 : i64)
  ^bb47(%205: i64):  // 2 preds: ^bb46, ^bb63
    %206 = llvm.icmp "slt" %205, %21 : i64
    llvm.cond_br %206, ^bb48, ^bb64
  ^bb48:  // pred: ^bb47
    llvm.br ^bb49(%12 : i64)
  ^bb49(%207: i64):  // 2 preds: ^bb48, ^bb62
    %208 = llvm.icmp "slt" %207, %22 : i64
    llvm.cond_br %208, ^bb50, ^bb63
  ^bb50:  // pred: ^bb49
    %209 = llvm.extractvalue %81[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %210 = llvm.mul %203, %5 : i64
    %211 = llvm.mul %205, %22 : i64
    %212 = llvm.add %210, %211 : i64
    %213 = llvm.add %212, %207 : i64
    %214 = llvm.insertvalue %213, %152[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %215 = llvm.insertvalue %17, %214[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %216 = llvm.insertvalue %5, %215[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %217 = llvm.insertvalue %17, %216[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %218 = llvm.insertvalue %22, %217[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %219 = llvm.insertvalue %17, %218[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %220 = llvm.insertvalue %8, %219[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb51(%12 : i64)
  ^bb51(%221: i64):  // 2 preds: ^bb50, ^bb61
    %222 = llvm.icmp "slt" %221, %17 : i64
    llvm.cond_br %222, ^bb52, ^bb62
  ^bb52:  // pred: ^bb51
    llvm.br ^bb53(%12 : i64)
  ^bb53(%223: i64):  // 2 preds: ^bb52, ^bb60
    %224 = llvm.icmp "slt" %223, %17 : i64
    llvm.cond_br %224, ^bb54, ^bb61
  ^bb54:  // pred: ^bb53
    llvm.br ^bb55(%12 : i64)
  ^bb55(%225: i64):  // 2 preds: ^bb54, ^bb59
    %226 = llvm.icmp "slt" %225, %17 : i64
    llvm.cond_br %226, ^bb56, ^bb60
  ^bb56:  // pred: ^bb55
    llvm.br ^bb57(%12 : i64)
  ^bb57(%227: i64):  // 2 preds: ^bb56, ^bb58
    %228 = llvm.icmp "slt" %227, %22 : i64
    llvm.cond_br %228, ^bb58, ^bb59
  ^bb58:  // pred: ^bb57
    %229 = llvm.getelementptr %209[%212] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %230 = llvm.mul %221, %5 : i64
    %231 = llvm.mul %223, %22 : i64
    %232 = llvm.add %230, %231 : i64
    %233 = llvm.add %232, %227 : i64
    %234 = llvm.getelementptr %229[%233] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %235 = llvm.load %234 : !llvm.ptr -> f32
    %236 = llvm.getelementptr %150[%213] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %237 = llvm.add %232, %225 : i64
    %238 = llvm.getelementptr %236[%237] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %239 = llvm.load %238 : !llvm.ptr -> f32
    %240 = llvm.fmul %235, %15  : f32
    %241 = llvm.fadd %239, %240  : f32
    llvm.store %241, %238 : f32, !llvm.ptr
    %242 = llvm.add %227, %8 : i64
    llvm.br ^bb57(%242 : i64)
  ^bb59:  // pred: ^bb57
    %243 = llvm.add %225, %8 : i64
    llvm.br ^bb55(%243 : i64)
  ^bb60:  // pred: ^bb55
    %244 = llvm.add %223, %8 : i64
    llvm.br ^bb53(%244 : i64)
  ^bb61:  // pred: ^bb53
    %245 = llvm.add %221, %8 : i64
    llvm.br ^bb51(%245 : i64)
  ^bb62:  // pred: ^bb51
    %246 = llvm.intr.stacksave : !llvm.ptr
    %247 = llvm.alloca %8 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %220, %247 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %248 = llvm.insertvalue %2, %1[0] : !llvm.struct<(i64, ptr)> 
    %249 = llvm.insertvalue %247, %248[1] : !llvm.struct<(i64, ptr)> 
    %250 = llvm.alloca %8 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %220, %250 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %251 = llvm.insertvalue %250, %248[1] : !llvm.struct<(i64, ptr)> 
    %252 = llvm.alloca %8 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %249, %252 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %253 = llvm.alloca %8 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %251, %253 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%50, %252, %253) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %246 : !llvm.ptr
    %254 = llvm.add %207, %17 : i64
    llvm.br ^bb49(%254 : i64)
  ^bb63:  // pred: ^bb49
    %255 = llvm.add %205, %17 : i64
    llvm.br ^bb47(%255 : i64)
  ^bb64:  // pred: ^bb47
    %256 = llvm.add %203, %17 : i64
    llvm.br ^bb45(%256 : i64)
  ^bb65:  // pred: ^bb45
    %257 = llvm.alloca %8 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %97, %257 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.call @print_memref_f32(%0, %257) : (i64, !llvm.ptr) -> ()
    %258 = llvm.add %50, %23 : i64
    %259 = llvm.call @malloc(%258) : (i64) -> !llvm.ptr
    %260 = llvm.ptrtoint %259 : !llvm.ptr to i64
    %261 = llvm.add %260, %32 : i64
    %262 = llvm.urem %261, %23  : i64
    %263 = llvm.sub %261, %262 : i64
    %264 = llvm.inttoptr %263 : i64 to !llvm.ptr
    llvm.br ^bb66(%12 : i64)
  ^bb66(%265: i64):  // 2 preds: ^bb65, ^bb67
    %266 = llvm.icmp "slt" %265, %8 : i64
    llvm.cond_br %266, ^bb67, ^bb68
  ^bb67:  // pred: ^bb66
    %267 = llvm.getelementptr %264[%265] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %16, %267 : f32, !llvm.ptr
    %268 = llvm.add %265, %8 : i64
    llvm.br ^bb66(%268 : i64)
  ^bb68:  // pred: ^bb66
    %269 = llvm.getelementptr %24[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %270 = llvm.ptrtoint %269 : !llvm.ptr to i64
    %271 = llvm.add %270, %23 : i64
    %272 = llvm.call @malloc(%271) : (i64) -> !llvm.ptr
    %273 = llvm.ptrtoint %272 : !llvm.ptr to i64
    %274 = llvm.add %273, %32 : i64
    %275 = llvm.urem %274, %23  : i64
    %276 = llvm.sub %274, %275 : i64
    %277 = llvm.inttoptr %276 : i64 to !llvm.ptr
    llvm.br ^bb69(%12 : i64)
  ^bb69(%278: i64):  // 2 preds: ^bb68, ^bb70
    %279 = llvm.icmp "slt" %278, %8 : i64
    llvm.cond_br %279, ^bb70, ^bb71
  ^bb70:  // pred: ^bb69
    %280 = llvm.getelementptr %277[%278] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %10, %280 : i64, !llvm.ptr
    %281 = llvm.add %278, %8 : i64
    llvm.br ^bb69(%281 : i64)
  ^bb71:  // pred: ^bb69
    llvm.br ^bb72(%12 : i64)
  ^bb72(%282: i64):  // 2 preds: ^bb71, ^bb79
    %283 = llvm.icmp "slt" %282, %18 : i64
    llvm.cond_br %283, ^bb73, ^bb80
  ^bb73:  // pred: ^bb72
    llvm.br ^bb74(%12 : i64)
  ^bb74(%284: i64):  // 2 preds: ^bb73, ^bb78
    %285 = llvm.icmp "slt" %284, %8 : i64
    llvm.cond_br %285, ^bb75, ^bb79
  ^bb75:  // pred: ^bb74
    llvm.br ^bb76(%12 : i64)
  ^bb76(%286: i64):  // 2 preds: ^bb75, ^bb77
    %287 = llvm.icmp "slt" %286, %17 : i64
    llvm.cond_br %287, ^bb77, ^bb78
  ^bb77:  // pred: ^bb76
    %288 = llvm.getelementptr %90[%282] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %289 = llvm.mul %284, %18 : i64
    %290 = llvm.add %289, %286 : i64
    %291 = llvm.getelementptr %288[%290] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %292 = llvm.load %291 : !llvm.ptr -> f32
    %293 = llvm.getelementptr %264[%284] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %294 = llvm.load %293 : !llvm.ptr -> f32
    %295 = llvm.getelementptr %277[%284] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %296 = llvm.load %295 : !llvm.ptr -> i64
    %297 = llvm.add %286, %282 : i64
    %298 = llvm.fcmp "ogt" %292, %294 : f32
    %299 = llvm.select %298, %292, %294 : i1, f32
    %300 = llvm.select %298, %297, %296 : i1, i64
    llvm.store %299, %293 : f32, !llvm.ptr
    llvm.store %300, %295 : i64, !llvm.ptr
    %301 = llvm.add %286, %8 : i64
    llvm.br ^bb76(%301 : i64)
  ^bb78:  // pred: ^bb76
    %302 = llvm.add %284, %8 : i64
    llvm.br ^bb74(%302 : i64)
  ^bb79:  // pred: ^bb74
    %303 = llvm.add %282, %17 : i64
    llvm.br ^bb72(%303 : i64)
  ^bb80:  // pred: ^bb72
    %304 = llvm.add %80, %11 : i64
    llvm.br ^bb1(%304, %159 : i64, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)
  ^bb81:  // pred: ^bb1
    llvm.return %69 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  }
}
