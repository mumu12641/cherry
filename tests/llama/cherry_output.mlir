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
  cherry.func @host() {
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
      %7 = cherry.argmax %6#0 dim 1 : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xi64]>
      %8 = cherry.constant(0 : i64) : i64
      %9 = cherry.tensor_get %7[%8] : (!cherry.cherry_tensor<[?xi64]>, i64) -> i64
      %10 = cherry.constant(1 : i64) : i64
      %11 = arith.addi %arg1, %10 : i64
      scf.yield %9, %11, %6#1 : i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>
    }
    cherry.print %5#2 : !cherry.cherry_tensor<[32x2048x128xf32]>
    cherry.return
  }
}

// ==========================================
// Phase: Inliner
// ==========================================
module {
  cherry.func @host() {
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
      %13 = cherry.argmax %9 dim 1 : (!cherry.cherry_tensor<[?xf32]>) -> !cherry.cherry_tensor<[?xi64]>
      %14 = cherry.constant(0 : i64) : i64
      %15 = cherry.tensor_get %13[%14] : (!cherry.cherry_tensor<[?xi64]>, i64) -> i64
      %16 = cherry.constant(1 : i64) : i64
      %17 = arith.addi %arg1, %16 : i64
      scf.yield %15, %17, %12 : i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>
    }
    cherry.print %5#2 : !cherry.cherry_tensor<[32x2048x128xf32]>
    cherry.return
  }
}

// ==========================================
// Phase: Shape Inference
// ==========================================
module {
  cherry.func @host() {
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
      %13 = cherry.argmax %9 dim 1 : (!cherry.cherry_tensor<[1x32000xf32]>) -> !cherry.cherry_tensor<[1xi64]>
      %14 = cherry.constant(0 : i64) : i64
      %15 = cherry.tensor_get %13[%14] : (!cherry.cherry_tensor<[1xi64]>, i64) -> i64
      %16 = cherry.constant(1 : i64) : i64
      %17 = arith.addi %arg1, %16 : i64
      scf.yield %15, %17, %12 : i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>
    }
    cherry.print %5#2 : !cherry.cherry_tensor<[32x2048x128xf32]>
    cherry.return
  }
}

// ==========================================
// Phase: Canonicalizer
// ==========================================
module {
  cherry.func @host() {
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
      %12 = cherry.argmax %9 dim 1 : (!cherry.cherry_tensor<[1x32000xf32]>) -> !cherry.cherry_tensor<[1xi64]>
      %13 = cherry.constant(0 : i64) : i64
      %14 = cherry.tensor_get %12[%13] : (!cherry.cherry_tensor<[1xi64]>, i64) -> i64
      %15 = cherry.constant(1 : i64) : i64
      %16 = arith.addi %arg1, %15 : i64
      scf.yield %14, %16, %11 : i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>
    }
    cherry.print %5#2 : !cherry.cherry_tensor<[32x2048x128xf32]>
    cherry.return
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
  func.func @host() {
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
    %cast = tensor.cast %0#2 : tensor<32x2048x128xf32> to tensor<*xf32>
    call @print_memref_f32(%cast) : (tensor<*xf32>) -> ()
    return
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
  func.func @host() {
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
    %cast = tensor.cast %0#2 : tensor<32x2048x128xf32> to tensor<*xf32>
    call @print_memref_f32(%cast) : (tensor<*xf32>) -> ()
    return
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
  func.func @host() {
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
    %cast = memref.cast %2#1 : memref<32x2048x128xf32> to memref<*xf32>
    call @print_memref_f32(%cast) : (memref<*xf32>) -> ()
    return
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
  func.func @host() {
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
    %cast = memref.cast %2#1 : memref<32x2048x128xf32> to memref<*xf32>
    call @print_memref_f32(%cast) : (memref<*xf32>) -> ()
    return
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
  llvm.func @host() {
    %0 = llvm.mlir.constant(3 : index) : i64
    %1 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %2 = llvm.mlir.constant(3 : i64) : i64
    %3 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %4 = llvm.mlir.addressof @__constant_32x2048x128xf32 : !llvm.ptr
    %5 = llvm.mlir.constant(262144 : index) : i64
    %6 = llvm.mlir.addressof @__constant_8x8xf32 : !llvm.ptr
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(10 : i64) : i64
    %9 = llvm.mlir.constant(0 : i64) : i64
    %10 = llvm.mlir.constant(1 : i64) : i64
    %11 = llvm.mlir.constant(0 : index) : i64
    %12 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %13 = llvm.mlir.constant(5.000000e-01 : f32) : f32
    %14 = llvm.mlir.constant(1.234200e+04 : f32) : f32
    %15 = llvm.mlir.constant(0xFF800000 : f32) : f32
    %16 = llvm.mlir.constant(8 : index) : i64
    %17 = llvm.mlir.constant(32000 : index) : i64
    %18 = llvm.mlir.constant(768 : index) : i64
    %19 = llvm.mlir.constant(32 : index) : i64
    %20 = llvm.mlir.constant(2048 : index) : i64
    %21 = llvm.mlir.constant(128 : index) : i64
    %22 = llvm.mlir.constant(64 : index) : i64
    %23 = llvm.mlir.zero : !llvm.ptr
    %24 = llvm.getelementptr %6[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x array<8 x f32>>
    %25 = llvm.getelementptr %4[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x array<2048 x array<128 x f32>>>
    %26 = llvm.getelementptr %23[8388608] : (!llvm.ptr) -> !llvm.ptr, f32
    %27 = llvm.ptrtoint %26 : !llvm.ptr to i64
    %28 = llvm.add %27, %22 : i64
    %29 = llvm.call @malloc(%28) : (i64) -> !llvm.ptr
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.sub %22, %7 : i64
    %32 = llvm.add %30, %31 : i64
    %33 = llvm.urem %32, %22  : i64
    %34 = llvm.sub %32, %33 : i64
    %35 = llvm.inttoptr %34 : i64 to !llvm.ptr
    %36 = llvm.insertvalue %29, %3[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %37 = llvm.insertvalue %35, %36[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %38 = llvm.insertvalue %11, %37[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %39 = llvm.insertvalue %19, %38[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %40 = llvm.insertvalue %20, %39[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %41 = llvm.insertvalue %21, %40[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %42 = llvm.insertvalue %5, %41[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %43 = llvm.insertvalue %21, %42[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %44 = llvm.insertvalue %7, %43[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %45 = llvm.mul %19, %7 : i64
    %46 = llvm.mul %45, %20 : i64
    %47 = llvm.mul %46, %21 : i64
    %48 = llvm.getelementptr %23[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %49 = llvm.ptrtoint %48 : !llvm.ptr to i64
    %50 = llvm.mul %47, %49 : i64
    "llvm.intr.memcpy"(%35, %25, %50) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb1(%9, %44 : i64, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)
  ^bb1(%51: i64, %52: !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>):  // 2 preds: ^bb0, ^bb80
    %53 = llvm.icmp "slt" %51, %8 : i64
    %54 = llvm.call @malloc(%28) : (i64) -> !llvm.ptr
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.add %55, %31 : i64
    %57 = llvm.urem %56, %22  : i64
    %58 = llvm.sub %56, %57 : i64
    %59 = llvm.inttoptr %58 : i64 to !llvm.ptr
    %60 = llvm.insertvalue %54, %3[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %61 = llvm.insertvalue %59, %60[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %62 = llvm.insertvalue %11, %61[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %63 = llvm.insertvalue %19, %62[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %64 = llvm.insertvalue %20, %63[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %65 = llvm.insertvalue %21, %64[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %66 = llvm.insertvalue %5, %65[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %67 = llvm.insertvalue %21, %66[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %68 = llvm.insertvalue %7, %67[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %69 = llvm.extractvalue %52[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %70 = llvm.mul %69, %7 : i64
    %71 = llvm.extractvalue %52[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %72 = llvm.mul %70, %71 : i64
    %73 = llvm.extractvalue %52[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %74 = llvm.mul %72, %73 : i64
    %75 = llvm.mul %74, %49 : i64
    %76 = llvm.extractvalue %52[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %77 = llvm.extractvalue %52[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %78 = llvm.getelementptr %76[%77] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%59, %78, %75) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.cond_br %53, ^bb2(%51, %68 : i64, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>), ^bb81
  ^bb2(%79: i64, %80: !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>):  // pred: ^bb1
    %81 = llvm.getelementptr %23[32000] : (!llvm.ptr) -> !llvm.ptr, f32
    %82 = llvm.ptrtoint %81 : !llvm.ptr to i64
    %83 = llvm.add %82, %22 : i64
    %84 = llvm.call @malloc(%83) : (i64) -> !llvm.ptr
    %85 = llvm.ptrtoint %84 : !llvm.ptr to i64
    %86 = llvm.add %85, %31 : i64
    %87 = llvm.urem %86, %22  : i64
    %88 = llvm.sub %86, %87 : i64
    %89 = llvm.inttoptr %88 : i64 to !llvm.ptr
    llvm.br ^bb3(%11 : i64)
  ^bb3(%90: i64):  // 2 preds: ^bb2, ^bb10
    %91 = llvm.icmp "slt" %90, %17 : i64
    llvm.cond_br %91, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%11 : i64)
  ^bb5(%92: i64):  // 2 preds: ^bb4, ^bb9
    %93 = llvm.icmp "slt" %92, %7 : i64
    llvm.cond_br %93, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%11 : i64)
  ^bb7(%94: i64):  // 2 preds: ^bb6, ^bb8
    %95 = llvm.icmp "slt" %94, %16 : i64
    llvm.cond_br %95, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %96 = llvm.getelementptr %89[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %97 = llvm.mul %92, %17 : i64
    %98 = llvm.add %97, %94 : i64
    %99 = llvm.getelementptr %96[%98] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %12, %99 : f32, !llvm.ptr
    %100 = llvm.add %94, %7 : i64
    llvm.br ^bb7(%100 : i64)
  ^bb9:  // pred: ^bb7
    %101 = llvm.add %92, %7 : i64
    llvm.br ^bb5(%101 : i64)
  ^bb10:  // pred: ^bb5
    %102 = llvm.mul %7, %7 : i64
    %103 = llvm.mul %102, %16 : i64
    %104 = llvm.mul %103, %49 : i64
    %105 = llvm.getelementptr %89[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%105, %105, %104) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %106 = llvm.add %90, %16 : i64
    llvm.br ^bb3(%106 : i64)
  ^bb11:  // pred: ^bb3
    llvm.br ^bb12(%11 : i64)
  ^bb12(%107: i64):  // 2 preds: ^bb11, ^bb25
    %108 = llvm.icmp "slt" %107, %17 : i64
    llvm.cond_br %108, ^bb13, ^bb26
  ^bb13:  // pred: ^bb12
    llvm.br ^bb14(%11 : i64)
  ^bb14(%109: i64):  // 2 preds: ^bb13, ^bb24
    %110 = llvm.icmp "slt" %109, %18 : i64
    llvm.cond_br %110, ^bb15, ^bb25
  ^bb15:  // pred: ^bb14
    llvm.br ^bb16(%11 : i64)
  ^bb16(%111: i64):  // 2 preds: ^bb15, ^bb23
    %112 = llvm.icmp "slt" %111, %7 : i64
    llvm.cond_br %112, ^bb17, ^bb24
  ^bb17:  // pred: ^bb16
    llvm.br ^bb18(%11 : i64)
  ^bb18(%113: i64):  // 2 preds: ^bb17, ^bb22
    %114 = llvm.icmp "slt" %113, %16 : i64
    llvm.cond_br %114, ^bb19, ^bb23
  ^bb19:  // pred: ^bb18
    llvm.br ^bb20(%11 : i64)
  ^bb20(%115: i64):  // 2 preds: ^bb19, ^bb21
    %116 = llvm.icmp "slt" %115, %16 : i64
    llvm.cond_br %116, ^bb21, ^bb22
  ^bb21:  // pred: ^bb20
    %117 = llvm.mul %115, %16 : i64
    %118 = llvm.add %117, %113 : i64
    %119 = llvm.getelementptr %24[%118] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %120 = llvm.load %119 : !llvm.ptr -> f32
    %121 = llvm.getelementptr %89[%107] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %122 = llvm.mul %111, %17 : i64
    %123 = llvm.add %122, %113 : i64
    %124 = llvm.getelementptr %121[%123] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %125 = llvm.load %124 : !llvm.ptr -> f32
    %126 = llvm.fmul %120, %13  : f32
    %127 = llvm.fadd %125, %126  : f32
    llvm.store %127, %124 : f32, !llvm.ptr
    %128 = llvm.add %115, %7 : i64
    llvm.br ^bb20(%128 : i64)
  ^bb22:  // pred: ^bb20
    %129 = llvm.add %113, %7 : i64
    llvm.br ^bb18(%129 : i64)
  ^bb23:  // pred: ^bb18
    %130 = llvm.add %111, %7 : i64
    llvm.br ^bb16(%130 : i64)
  ^bb24:  // pred: ^bb16
    %131 = llvm.mul %7, %7 : i64
    %132 = llvm.mul %131, %16 : i64
    %133 = llvm.mul %132, %49 : i64
    %134 = llvm.getelementptr %89[%107] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%134, %134, %133) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %135 = llvm.add %109, %16 : i64
    llvm.br ^bb14(%135 : i64)
  ^bb25:  // pred: ^bb14
    %136 = llvm.add %107, %16 : i64
    llvm.br ^bb12(%136 : i64)
  ^bb26:  // pred: ^bb12
    %137 = llvm.call @malloc(%28) : (i64) -> !llvm.ptr
    %138 = llvm.ptrtoint %137 : !llvm.ptr to i64
    %139 = llvm.add %138, %31 : i64
    %140 = llvm.urem %139, %22  : i64
    %141 = llvm.sub %139, %140 : i64
    %142 = llvm.inttoptr %141 : i64 to !llvm.ptr
    %143 = llvm.insertvalue %137, %3[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %144 = llvm.insertvalue %142, %143[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %145 = llvm.insertvalue %11, %144[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %146 = llvm.insertvalue %19, %145[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %147 = llvm.insertvalue %20, %146[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %148 = llvm.insertvalue %21, %147[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %149 = llvm.insertvalue %5, %148[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %150 = llvm.insertvalue %21, %149[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %151 = llvm.insertvalue %7, %150[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb27(%11 : i64)
  ^bb27(%152: i64):  // 2 preds: ^bb26, ^bb43
    %153 = llvm.icmp "slt" %152, %19 : i64
    llvm.cond_br %153, ^bb28, ^bb44
  ^bb28:  // pred: ^bb27
    llvm.br ^bb29(%11 : i64)
  ^bb29(%154: i64):  // 2 preds: ^bb28, ^bb42
    %155 = llvm.icmp "slt" %154, %20 : i64
    llvm.cond_br %155, ^bb30, ^bb43
  ^bb30:  // pred: ^bb29
    llvm.br ^bb31(%11 : i64)
  ^bb31(%156: i64):  // 2 preds: ^bb30, ^bb41
    %157 = llvm.icmp "slt" %156, %21 : i64
    llvm.cond_br %157, ^bb32, ^bb42
  ^bb32:  // pred: ^bb31
    %158 = llvm.mul %152, %5 : i64
    %159 = llvm.mul %154, %21 : i64
    %160 = llvm.add %158, %159 : i64
    %161 = llvm.add %160, %156 : i64
    %162 = llvm.insertvalue %161, %144[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %163 = llvm.insertvalue %16, %162[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %164 = llvm.insertvalue %5, %163[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %165 = llvm.insertvalue %16, %164[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %166 = llvm.insertvalue %21, %165[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %167 = llvm.insertvalue %16, %166[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %168 = llvm.insertvalue %7, %167[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb33(%11 : i64)
  ^bb33(%169: i64):  // 2 preds: ^bb32, ^bb40
    %170 = llvm.icmp "slt" %169, %16 : i64
    llvm.cond_br %170, ^bb34, ^bb41
  ^bb34:  // pred: ^bb33
    llvm.br ^bb35(%11 : i64)
  ^bb35(%171: i64):  // 2 preds: ^bb34, ^bb39
    %172 = llvm.icmp "slt" %171, %16 : i64
    llvm.cond_br %172, ^bb36, ^bb40
  ^bb36:  // pred: ^bb35
    llvm.br ^bb37(%11 : i64)
  ^bb37(%173: i64):  // 2 preds: ^bb36, ^bb38
    %174 = llvm.icmp "slt" %173, %16 : i64
    llvm.cond_br %174, ^bb38, ^bb39
  ^bb38:  // pred: ^bb37
    %175 = llvm.getelementptr %142[%161] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %176 = llvm.mul %169, %5 : i64
    %177 = llvm.mul %171, %21 : i64
    %178 = llvm.add %176, %177 : i64
    %179 = llvm.add %178, %173 : i64
    %180 = llvm.getelementptr %175[%179] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %12, %180 : f32, !llvm.ptr
    %181 = llvm.add %173, %7 : i64
    llvm.br ^bb37(%181 : i64)
  ^bb39:  // pred: ^bb37
    %182 = llvm.add %171, %7 : i64
    llvm.br ^bb35(%182 : i64)
  ^bb40:  // pred: ^bb35
    %183 = llvm.add %169, %7 : i64
    llvm.br ^bb33(%183 : i64)
  ^bb41:  // pred: ^bb33
    %184 = llvm.intr.stacksave : !llvm.ptr
    %185 = llvm.alloca %7 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %168, %185 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %186 = llvm.insertvalue %2, %1[0] : !llvm.struct<(i64, ptr)> 
    %187 = llvm.insertvalue %185, %186[1] : !llvm.struct<(i64, ptr)> 
    %188 = llvm.alloca %7 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %168, %188 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %189 = llvm.insertvalue %188, %186[1] : !llvm.struct<(i64, ptr)> 
    %190 = llvm.alloca %7 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %187, %190 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %191 = llvm.alloca %7 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %189, %191 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%49, %190, %191) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %184 : !llvm.ptr
    %192 = llvm.add %156, %16 : i64
    llvm.br ^bb31(%192 : i64)
  ^bb42:  // pred: ^bb31
    %193 = llvm.add %154, %16 : i64
    llvm.br ^bb29(%193 : i64)
  ^bb43:  // pred: ^bb29
    %194 = llvm.add %152, %16 : i64
    llvm.br ^bb27(%194 : i64)
  ^bb44:  // pred: ^bb27
    llvm.br ^bb45(%11 : i64)
  ^bb45(%195: i64):  // 2 preds: ^bb44, ^bb64
    %196 = llvm.icmp "slt" %195, %19 : i64
    llvm.cond_br %196, ^bb46, ^bb65
  ^bb46:  // pred: ^bb45
    llvm.br ^bb47(%11 : i64)
  ^bb47(%197: i64):  // 2 preds: ^bb46, ^bb63
    %198 = llvm.icmp "slt" %197, %20 : i64
    llvm.cond_br %198, ^bb48, ^bb64
  ^bb48:  // pred: ^bb47
    llvm.br ^bb49(%11 : i64)
  ^bb49(%199: i64):  // 2 preds: ^bb48, ^bb62
    %200 = llvm.icmp "slt" %199, %21 : i64
    llvm.cond_br %200, ^bb50, ^bb63
  ^bb50:  // pred: ^bb49
    %201 = llvm.extractvalue %80[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %202 = llvm.mul %195, %5 : i64
    %203 = llvm.mul %197, %21 : i64
    %204 = llvm.add %202, %203 : i64
    %205 = llvm.add %204, %199 : i64
    %206 = llvm.insertvalue %205, %144[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %207 = llvm.insertvalue %16, %206[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %208 = llvm.insertvalue %5, %207[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %209 = llvm.insertvalue %16, %208[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %210 = llvm.insertvalue %21, %209[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %211 = llvm.insertvalue %16, %210[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %212 = llvm.insertvalue %7, %211[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb51(%11 : i64)
  ^bb51(%213: i64):  // 2 preds: ^bb50, ^bb61
    %214 = llvm.icmp "slt" %213, %16 : i64
    llvm.cond_br %214, ^bb52, ^bb62
  ^bb52:  // pred: ^bb51
    llvm.br ^bb53(%11 : i64)
  ^bb53(%215: i64):  // 2 preds: ^bb52, ^bb60
    %216 = llvm.icmp "slt" %215, %16 : i64
    llvm.cond_br %216, ^bb54, ^bb61
  ^bb54:  // pred: ^bb53
    llvm.br ^bb55(%11 : i64)
  ^bb55(%217: i64):  // 2 preds: ^bb54, ^bb59
    %218 = llvm.icmp "slt" %217, %16 : i64
    llvm.cond_br %218, ^bb56, ^bb60
  ^bb56:  // pred: ^bb55
    llvm.br ^bb57(%11 : i64)
  ^bb57(%219: i64):  // 2 preds: ^bb56, ^bb58
    %220 = llvm.icmp "slt" %219, %21 : i64
    llvm.cond_br %220, ^bb58, ^bb59
  ^bb58:  // pred: ^bb57
    %221 = llvm.getelementptr %201[%204] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %222 = llvm.mul %213, %5 : i64
    %223 = llvm.mul %215, %21 : i64
    %224 = llvm.add %222, %223 : i64
    %225 = llvm.add %224, %219 : i64
    %226 = llvm.getelementptr %221[%225] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %227 = llvm.load %226 : !llvm.ptr -> f32
    %228 = llvm.getelementptr %142[%205] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %229 = llvm.add %224, %217 : i64
    %230 = llvm.getelementptr %228[%229] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %231 = llvm.load %230 : !llvm.ptr -> f32
    %232 = llvm.fmul %227, %14  : f32
    %233 = llvm.fadd %231, %232  : f32
    llvm.store %233, %230 : f32, !llvm.ptr
    %234 = llvm.add %219, %7 : i64
    llvm.br ^bb57(%234 : i64)
  ^bb59:  // pred: ^bb57
    %235 = llvm.add %217, %7 : i64
    llvm.br ^bb55(%235 : i64)
  ^bb60:  // pred: ^bb55
    %236 = llvm.add %215, %7 : i64
    llvm.br ^bb53(%236 : i64)
  ^bb61:  // pred: ^bb53
    %237 = llvm.add %213, %7 : i64
    llvm.br ^bb51(%237 : i64)
  ^bb62:  // pred: ^bb51
    %238 = llvm.intr.stacksave : !llvm.ptr
    %239 = llvm.alloca %7 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %212, %239 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %240 = llvm.insertvalue %2, %1[0] : !llvm.struct<(i64, ptr)> 
    %241 = llvm.insertvalue %239, %240[1] : !llvm.struct<(i64, ptr)> 
    %242 = llvm.alloca %7 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %212, %242 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %243 = llvm.insertvalue %242, %240[1] : !llvm.struct<(i64, ptr)> 
    %244 = llvm.alloca %7 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %241, %244 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %245 = llvm.alloca %7 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %243, %245 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%49, %244, %245) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %238 : !llvm.ptr
    %246 = llvm.add %199, %16 : i64
    llvm.br ^bb49(%246 : i64)
  ^bb63:  // pred: ^bb49
    %247 = llvm.add %197, %16 : i64
    llvm.br ^bb47(%247 : i64)
  ^bb64:  // pred: ^bb47
    %248 = llvm.add %195, %16 : i64
    llvm.br ^bb45(%248 : i64)
  ^bb65:  // pred: ^bb45
    %249 = llvm.add %49, %22 : i64
    %250 = llvm.call @malloc(%249) : (i64) -> !llvm.ptr
    %251 = llvm.ptrtoint %250 : !llvm.ptr to i64
    %252 = llvm.add %251, %31 : i64
    %253 = llvm.urem %252, %22  : i64
    %254 = llvm.sub %252, %253 : i64
    %255 = llvm.inttoptr %254 : i64 to !llvm.ptr
    llvm.br ^bb66(%11 : i64)
  ^bb66(%256: i64):  // 2 preds: ^bb65, ^bb67
    %257 = llvm.icmp "slt" %256, %7 : i64
    llvm.cond_br %257, ^bb67, ^bb68
  ^bb67:  // pred: ^bb66
    %258 = llvm.getelementptr %255[%256] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %15, %258 : f32, !llvm.ptr
    %259 = llvm.add %256, %7 : i64
    llvm.br ^bb66(%259 : i64)
  ^bb68:  // pred: ^bb66
    %260 = llvm.getelementptr %23[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %261 = llvm.ptrtoint %260 : !llvm.ptr to i64
    %262 = llvm.add %261, %22 : i64
    %263 = llvm.call @malloc(%262) : (i64) -> !llvm.ptr
    %264 = llvm.ptrtoint %263 : !llvm.ptr to i64
    %265 = llvm.add %264, %31 : i64
    %266 = llvm.urem %265, %22  : i64
    %267 = llvm.sub %265, %266 : i64
    %268 = llvm.inttoptr %267 : i64 to !llvm.ptr
    llvm.br ^bb69(%11 : i64)
  ^bb69(%269: i64):  // 2 preds: ^bb68, ^bb70
    %270 = llvm.icmp "slt" %269, %7 : i64
    llvm.cond_br %270, ^bb70, ^bb71
  ^bb70:  // pred: ^bb69
    %271 = llvm.getelementptr %268[%269] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %9, %271 : i64, !llvm.ptr
    %272 = llvm.add %269, %7 : i64
    llvm.br ^bb69(%272 : i64)
  ^bb71:  // pred: ^bb69
    llvm.br ^bb72(%11 : i64)
  ^bb72(%273: i64):  // 2 preds: ^bb71, ^bb79
    %274 = llvm.icmp "slt" %273, %17 : i64
    llvm.cond_br %274, ^bb73, ^bb80
  ^bb73:  // pred: ^bb72
    llvm.br ^bb74(%11 : i64)
  ^bb74(%275: i64):  // 2 preds: ^bb73, ^bb78
    %276 = llvm.icmp "slt" %275, %7 : i64
    llvm.cond_br %276, ^bb75, ^bb79
  ^bb75:  // pred: ^bb74
    llvm.br ^bb76(%11 : i64)
  ^bb76(%277: i64):  // 2 preds: ^bb75, ^bb77
    %278 = llvm.icmp "slt" %277, %16 : i64
    llvm.cond_br %278, ^bb77, ^bb78
  ^bb77:  // pred: ^bb76
    %279 = llvm.getelementptr %89[%273] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %280 = llvm.mul %275, %17 : i64
    %281 = llvm.add %280, %277 : i64
    %282 = llvm.getelementptr %279[%281] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %283 = llvm.load %282 : !llvm.ptr -> f32
    %284 = llvm.getelementptr %255[%275] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %285 = llvm.load %284 : !llvm.ptr -> f32
    %286 = llvm.getelementptr %268[%275] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %287 = llvm.load %286 : !llvm.ptr -> i64
    %288 = llvm.add %277, %273 : i64
    %289 = llvm.fcmp "ogt" %283, %285 : f32
    %290 = llvm.select %289, %283, %285 : i1, f32
    %291 = llvm.select %289, %288, %287 : i1, i64
    llvm.store %290, %284 : f32, !llvm.ptr
    llvm.store %291, %286 : i64, !llvm.ptr
    %292 = llvm.add %277, %7 : i64
    llvm.br ^bb76(%292 : i64)
  ^bb78:  // pred: ^bb76
    %293 = llvm.add %275, %7 : i64
    llvm.br ^bb74(%293 : i64)
  ^bb79:  // pred: ^bb74
    %294 = llvm.add %273, %16 : i64
    llvm.br ^bb72(%294 : i64)
  ^bb80:  // pred: ^bb72
    %295 = llvm.add %79, %10 : i64
    llvm.br ^bb1(%295, %151 : i64, !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)
  ^bb81:  // pred: ^bb1
    %296 = llvm.alloca %7 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %68, %296 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    llvm.call @print_memref_f32(%0, %296) : (i64, !llvm.ptr) -> ()
    llvm.return
  }
}
