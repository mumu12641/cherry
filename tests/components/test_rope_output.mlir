// Original IR loaded from file
module {
  func.func @host() {
    %0 = cherry.create_tensor dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]> : tensor<1x4xf32> -> !cherry.cherry_tensor<[1x4xf32]>
    %1 = cherry.constant(1 : i64) : i64
    %2 = cherry.rope %0, %1 : (!cherry.cherry_tensor<[1x4xf32]>, i64) -> !cherry.cherry_tensor<[1x4xf32]>
    cherry.print %2 : !cherry.cherry_tensor<[1x4xf32]>
    return
  }
}

// ==========================================
// Phase: Inliner
// ==========================================
module {
  func.func @host() {
    %0 = cherry.create_tensor dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]> : tensor<1x4xf32> -> !cherry.cherry_tensor<[1x4xf32]>
    %1 = cherry.constant(1 : i64) : i64
    %2 = cherry.rope %0, %1 : (!cherry.cherry_tensor<[1x4xf32]>, i64) -> !cherry.cherry_tensor<[1x4xf32]>
    cherry.print %2 : !cherry.cherry_tensor<[1x4xf32]>
    return
  }
}

// ==========================================
// Phase: Shape Inference
// ==========================================
module {
  func.func @host() {
    %0 = cherry.create_tensor dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]> : tensor<1x4xf32> -> !cherry.cherry_tensor<[1x4xf32]>
    %1 = cherry.constant(1 : i64) : i64
    %2 = cherry.rope %0, %1 : (!cherry.cherry_tensor<[1x4xf32]>, i64) -> !cherry.cherry_tensor<[1x4xf32]>
    cherry.print %2 : !cherry.cherry_tensor<[1x4xf32]>
    return
  }
}

// ==========================================
// Phase: Canonicalizer
// ==========================================
module {
  func.func @host() {
    %0 = cherry.create_tensor dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]> : tensor<1x4xf32> -> !cherry.cherry_tensor<[1x4xf32]>
    %1 = cherry.constant(1 : i64) : i64
    %2 = cherry.rope %0, %1 : (!cherry.cherry_tensor<[1x4xf32]>, i64) -> !cherry.cherry_tensor<[1x4xf32]>
    cherry.print %2 : !cherry.cherry_tensor<[1x4xf32]>
    return
  }
}

// ==========================================
// Phase: Convert to Linalg
// ==========================================
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
module {
  func.func private @printMemrefF32(tensor<*xf32> {bufferization.access = "read"})
  func.func @host() {
    %cst = arith.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]> : tensor<1x4xf32>
    %c1_i64 = arith.constant 1 : i64
    %0 = tensor.empty() : tensor<2xf32>
    %1 = tensor.empty() : tensor<2xf32>
    %2 = arith.uitofp %c1_i64 : i64 to f32
    %3:2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} outs(%0, %1 : tensor<2xf32>, tensor<2xf32>) {
    ^bb0(%out: f32, %out_6: f32):
      %7 = linalg.index 0 : index
      %8 = arith.index_cast %7 : index to i64
      %9 = arith.uitofp %8 : i64 to f32
      %cst_7 = arith.constant 1.000000e+04 : f32
      %cst_8 = arith.constant 4.000000e+00 : f32
      %cst_9 = arith.constant -2.000000e+00 : f32
      %10 = arith.mulf %cst_9, %9 : f32
      %11 = arith.divf %10, %cst_8 : f32
      %12 = math.powf %cst_7, %11 : f32
      %13 = arith.mulf %2, %12 : f32
      %14 = math.cos %13 : f32
      %15 = math.sin %13 : f32
      linalg.yield %14, %15 : f32, f32
    } -> (tensor<2xf32>, tensor<2xf32>)
    %expanded = tensor.expand_shape %cst [[0], [1, 2]] output_shape [1, 2, 2] : tensor<1x4xf32> into tensor<1x2x2xf32>
    %extracted_slice = tensor.extract_slice %expanded[0, 0, 0] [1, 2, 1] [1, 1, 1] : tensor<1x2x2xf32> to tensor<1x2x1xf32>
    %collapsed = tensor.collapse_shape %extracted_slice [[0], [1, 2]] : tensor<1x2x1xf32> into tensor<1x2xf32>
    %extracted_slice_0 = tensor.extract_slice %expanded[0, 0, 1] [1, 2, 1] [1, 1, 1] : tensor<1x2x2xf32> to tensor<1x2x1xf32>
    %collapsed_1 = tensor.collapse_shape %extracted_slice_0 [[0], [1, 2]] : tensor<1x2x1xf32> into tensor<1x2xf32>
    %4 = tensor.empty() : tensor<1x2xf32>
    %5:2 = linalg.generic {indexing_maps = [#map1, #map1, #map2, #map2, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%collapsed, %collapsed_1, %3#0, %3#1 : tensor<1x2xf32>, tensor<1x2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%4, %4 : tensor<1x2xf32>, tensor<1x2xf32>) {
    ^bb0(%in: f32, %in_6: f32, %in_7: f32, %in_8: f32, %out: f32, %out_9: f32):
      %7 = arith.mulf %in, %in_7 : f32
      %8 = arith.mulf %in_6, %in_8 : f32
      %9 = arith.subf %7, %8 : f32
      %10 = arith.mulf %in_6, %in_7 : f32
      %11 = arith.mulf %in, %in_8 : f32
      %12 = arith.addf %10, %11 : f32
      linalg.yield %9, %12 : f32, f32
    } -> (tensor<1x2xf32>, tensor<1x2xf32>)
    %expanded_2 = tensor.expand_shape %5#0 [[0], [1, 2]] output_shape [1, 2, 1] : tensor<1x2xf32> into tensor<1x2x1xf32>
    %expanded_3 = tensor.expand_shape %5#1 [[0], [1, 2]] output_shape [1, 2, 1] : tensor<1x2xf32> into tensor<1x2x1xf32>
    %6 = tensor.empty() : tensor<1x2x2xf32>
    %inserted_slice = tensor.insert_slice %expanded_2 into %6[0, 0, 0] [1, 2, 1] [1, 1, 1] : tensor<1x2x1xf32> into tensor<1x2x2xf32>
    %inserted_slice_4 = tensor.insert_slice %expanded_3 into %inserted_slice[0, 0, 1] [1, 2, 1] [1, 1, 1] : tensor<1x2x1xf32> into tensor<1x2x2xf32>
    %collapsed_5 = tensor.collapse_shape %inserted_slice_4 [[0], [1, 2]] : tensor<1x2x2xf32> into tensor<1x4xf32>
    %cast = tensor.cast %collapsed_5 : tensor<1x4xf32> to tensor<*xf32>
    call @printMemrefF32(%cast) : (tensor<*xf32>) -> ()
    return
  }
}

// ==========================================
// Phase: Linalg Tiling
// ==========================================
#map = affine_map<(d0) -> (-d0 + 2, 8)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0 + d1)>
#map3 = affine_map<(d0) -> (-d0 + 1, 8)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d1, d2)>
module {
  func.func private @printMemrefF32(tensor<*xf32> {bufferization.access = "read"})
  func.func @host() {
    %c8 = arith.constant 8 : index
    %c8_0 = arith.constant 8 : index
    %c8_1 = arith.constant 8 : index
    %c8_2 = arith.constant 8 : index
    %c8_3 = arith.constant 8 : index
    %c8_4 = arith.constant 8 : index
    %cst = arith.constant dense<[[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]]> : tensor<1x2x2xf32>
    %cst_5 = arith.constant -2.000000e+00 : f32
    %cst_6 = arith.constant 4.000000e+00 : f32
    %cst_7 = arith.constant 1.000000e+04 : f32
    %0 = tensor.empty() : tensor<2xf32>
    %1 = tensor.empty() : tensor<2xf32>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c8_8 = arith.constant 8 : index
    %2:2 = scf.for %arg0 = %c0 to %c2 step %c8_8 iter_args(%arg1 = %0, %arg2 = %1) -> (tensor<2xf32>, tensor<2xf32>) {
      %7 = affine.min #map(%arg0)
      %8 = affine.min #map(%arg0)
      %extracted_slice_20 = tensor.extract_slice %arg1[%arg0] [%7] [1] : tensor<2xf32> to tensor<?xf32>
      %extracted_slice_21 = tensor.extract_slice %arg2[%arg0] [%8] [1] : tensor<2xf32> to tensor<?xf32>
      %9:2 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} outs(%extracted_slice_20, %extracted_slice_21 : tensor<?xf32>, tensor<?xf32>) {
      ^bb0(%out: f32, %out_24: f32):
        %10 = linalg.index 0 : index
        %11 = affine.apply #map2(%10, %arg0)
        %12 = arith.index_cast %11 : index to i64
        %13 = arith.uitofp %12 : i64 to f32
        %14 = arith.mulf %13, %cst_5 : f32
        %15 = arith.divf %14, %cst_6 : f32
        %16 = math.powf %cst_7, %15 : f32
        %17 = math.cos %16 : f32
        %18 = math.sin %16 : f32
        linalg.yield %17, %18 : f32, f32
      } -> (tensor<?xf32>, tensor<?xf32>)
      %inserted_slice_22 = tensor.insert_slice %9#0 into %arg1[%arg0] [%7] [1] : tensor<?xf32> into tensor<2xf32>
      %inserted_slice_23 = tensor.insert_slice %9#1 into %arg2[%arg0] [%8] [1] : tensor<?xf32> into tensor<2xf32>
      scf.yield %inserted_slice_22, %inserted_slice_23 : tensor<2xf32>, tensor<2xf32>
    }
    %extracted_slice = tensor.extract_slice %cst[0, 0, 0] [1, 2, 1] [1, 1, 1] : tensor<1x2x2xf32> to tensor<1x2x1xf32>
    %extracted_slice_9 = tensor.extract_slice %cst[0, 0, 1] [1, 2, 1] [1, 1, 1] : tensor<1x2x2xf32> to tensor<1x2x1xf32>
    %expanded = tensor.expand_shape %2#0 [[0, 1]] output_shape [2, 1] : tensor<2xf32> into tensor<2x1xf32>
    %expanded_10 = tensor.expand_shape %2#1 [[0, 1]] output_shape [2, 1] : tensor<2xf32> into tensor<2x1xf32>
    %3 = tensor.empty() : tensor<1x2x1xf32>
    %4 = tensor.empty() : tensor<1x2x1xf32>
    %c0_11 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_12 = arith.constant 8 : index
    %c0_13 = arith.constant 0 : index
    %c2_14 = arith.constant 2 : index
    %c8_15 = arith.constant 8 : index
    %c0_16 = arith.constant 0 : index
    %c1_17 = arith.constant 1 : index
    %c8_18 = arith.constant 8 : index
    %5:2 = scf.for %arg0 = %c0_11 to %c1 step %c8_12 iter_args(%arg1 = %3, %arg2 = %4) -> (tensor<1x2x1xf32>, tensor<1x2x1xf32>) {
      %7:2 = scf.for %arg3 = %c0_13 to %c2_14 step %c8_15 iter_args(%arg4 = %arg1, %arg5 = %arg2) -> (tensor<1x2x1xf32>, tensor<1x2x1xf32>) {
        %8:2 = scf.for %arg6 = %c0_16 to %c1_17 step %c8_18 iter_args(%arg7 = %arg4, %arg8 = %arg5) -> (tensor<1x2x1xf32>, tensor<1x2x1xf32>) {
          %9 = affine.min #map3(%arg0)
          %10 = affine.min #map(%arg3)
          %11 = affine.min #map3(%arg6)
          %12 = affine.min #map3(%arg0)
          %13 = affine.min #map(%arg3)
          %14 = affine.min #map3(%arg6)
          %15 = affine.min #map(%arg3)
          %16 = affine.min #map3(%arg6)
          %17 = affine.min #map(%arg3)
          %18 = affine.min #map3(%arg6)
          %19 = affine.min #map3(%arg0)
          %20 = affine.min #map(%arg3)
          %21 = affine.min #map3(%arg6)
          %22 = affine.min #map3(%arg0)
          %23 = affine.min #map(%arg3)
          %24 = affine.min #map3(%arg6)
          %extracted_slice_20 = tensor.extract_slice %extracted_slice[%arg0, %arg3, %arg6] [%9, %10, %11] [1, 1, 1] : tensor<1x2x1xf32> to tensor<?x?x?xf32>
          %extracted_slice_21 = tensor.extract_slice %extracted_slice_9[%arg0, %arg3, %arg6] [%12, %13, %14] [1, 1, 1] : tensor<1x2x1xf32> to tensor<?x?x?xf32>
          %extracted_slice_22 = tensor.extract_slice %expanded[%arg3, %arg6] [%15, %16] [1, 1] : tensor<2x1xf32> to tensor<?x?xf32>
          %extracted_slice_23 = tensor.extract_slice %expanded_10[%arg3, %arg6] [%17, %18] [1, 1] : tensor<2x1xf32> to tensor<?x?xf32>
          %extracted_slice_24 = tensor.extract_slice %arg7[%arg0, %arg3, %arg6] [%19, %20, %21] [1, 1, 1] : tensor<1x2x1xf32> to tensor<?x?x?xf32>
          %extracted_slice_25 = tensor.extract_slice %arg8[%arg0, %arg3, %arg6] [%22, %23, %24] [1, 1, 1] : tensor<1x2x1xf32> to tensor<?x?x?xf32>
          %25:2 = linalg.generic {indexing_maps = [#map4, #map4, #map5, #map5, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_20, %extracted_slice_21, %extracted_slice_22, %extracted_slice_23 : tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_24, %extracted_slice_25 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %in_28: f32, %in_29: f32, %in_30: f32, %out: f32, %out_31: f32):
            %26 = arith.mulf %in, %in_29 : f32
            %27 = arith.mulf %in_28, %in_30 : f32
            %28 = arith.subf %26, %27 : f32
            %29 = arith.mulf %in_28, %in_29 : f32
            %30 = arith.mulf %in, %in_30 : f32
            %31 = arith.addf %29, %30 : f32
            linalg.yield %28, %31 : f32, f32
          } -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
          %inserted_slice_26 = tensor.insert_slice %25#0 into %arg7[%arg0, %arg3, %arg6] [%19, %20, %21] [1, 1, 1] : tensor<?x?x?xf32> into tensor<1x2x1xf32>
          %inserted_slice_27 = tensor.insert_slice %25#1 into %arg8[%arg0, %arg3, %arg6] [%22, %23, %24] [1, 1, 1] : tensor<?x?x?xf32> into tensor<1x2x1xf32>
          scf.yield %inserted_slice_26, %inserted_slice_27 : tensor<1x2x1xf32>, tensor<1x2x1xf32>
        }
        scf.yield %8#0, %8#1 : tensor<1x2x1xf32>, tensor<1x2x1xf32>
      }
      scf.yield %7#0, %7#1 : tensor<1x2x1xf32>, tensor<1x2x1xf32>
    }
    %6 = tensor.empty() : tensor<1x2x2xf32>
    %inserted_slice = tensor.insert_slice %5#0 into %6[0, 0, 0] [1, 2, 1] [1, 1, 1] : tensor<1x2x1xf32> into tensor<1x2x2xf32>
    %inserted_slice_19 = tensor.insert_slice %5#1 into %inserted_slice[0, 0, 1] [1, 2, 1] [1, 1, 1] : tensor<1x2x1xf32> into tensor<1x2x2xf32>
    %collapsed = tensor.collapse_shape %inserted_slice_19 [[0], [1, 2]] : tensor<1x2x2xf32> into tensor<1x4xf32>
    %cast = tensor.cast %collapsed : tensor<1x4xf32> to tensor<*xf32>
    call @printMemrefF32(%cast) : (tensor<*xf32>) -> ()
    return
  }
}

// ==========================================
// Phase: Bufferization
// ==========================================
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
module {
  memref.global "private" constant @__constant_1x2x2xf32 : memref<1x2x2xf32> = dense<[[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]]> {alignment = 64 : i64}
  func.func private @printMemrefF32(memref<*xf32> {bufferization.access = "read"})
  func.func @host() {
    %cst = arith.constant 1.000000e+04 : f32
    %cst_0 = arith.constant 4.000000e+00 : f32
    %cst_1 = arith.constant -2.000000e+00 : f32
    %0 = memref.get_global @__constant_1x2x2xf32 : memref<1x2x2xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} outs(%alloc, %alloc_2 : memref<2xf32>, memref<2xf32>) {
    ^bb0(%out: f32, %out_10: f32):
      %1 = linalg.index 0 : index
      %2 = arith.index_cast %1 : index to i64
      %3 = arith.uitofp %2 : i64 to f32
      %4 = arith.mulf %3, %cst_1 : f32
      %5 = arith.divf %4, %cst_0 : f32
      %6 = math.powf %cst, %5 : f32
      %7 = math.cos %6 : f32
      %8 = math.sin %6 : f32
      linalg.yield %7, %8 : f32, f32
    }
    %subview = memref.subview %0[0, 0, 0] [1, 2, 1] [1, 1, 1] : memref<1x2x2xf32> to memref<1x2x1xf32, strided<[4, 2, 1]>>
    %subview_3 = memref.subview %0[0, 0, 1] [1, 2, 1] [1, 1, 1] : memref<1x2x2xf32> to memref<1x2x1xf32, strided<[4, 2, 1], offset: 1>>
    %expand_shape = memref.expand_shape %alloc [[0, 1]] output_shape [2, 1] : memref<2xf32> into memref<2x1xf32>
    %expand_shape_4 = memref.expand_shape %alloc_2 [[0, 1]] output_shape [2, 1] : memref<2xf32> into memref<2x1xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x2x1xf32>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x2x1xf32>
    linalg.generic {indexing_maps = [#map1, #map1, #map2, #map2, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview, %subview_3, %expand_shape, %expand_shape_4 : memref<1x2x1xf32, strided<[4, 2, 1]>>, memref<1x2x1xf32, strided<[4, 2, 1], offset: 1>>, memref<2x1xf32>, memref<2x1xf32>) outs(%alloc_5, %alloc_6 : memref<1x2x1xf32>, memref<1x2x1xf32>) {
    ^bb0(%in: f32, %in_10: f32, %in_11: f32, %in_12: f32, %out: f32, %out_13: f32):
      %1 = arith.mulf %in, %in_11 : f32
      %2 = arith.mulf %in_10, %in_12 : f32
      %3 = arith.subf %1, %2 : f32
      %4 = arith.mulf %in_10, %in_11 : f32
      %5 = arith.mulf %in, %in_12 : f32
      %6 = arith.addf %4, %5 : f32
      linalg.yield %3, %6 : f32, f32
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x2x2xf32>
    %subview_8 = memref.subview %alloc_7[0, 0, 0] [1, 2, 1] [1, 1, 1] : memref<1x2x2xf32> to memref<1x2x1xf32, strided<[4, 2, 1]>>
    memref.copy %alloc_5, %subview_8 : memref<1x2x1xf32> to memref<1x2x1xf32, strided<[4, 2, 1]>>
    %subview_9 = memref.subview %alloc_7[0, 0, 1] [1, 2, 1] [1, 1, 1] : memref<1x2x2xf32> to memref<1x2x1xf32, strided<[4, 2, 1], offset: 1>>
    memref.copy %alloc_6, %subview_9 : memref<1x2x1xf32> to memref<1x2x1xf32, strided<[4, 2, 1], offset: 1>>
    %collapse_shape = memref.collapse_shape %alloc_7 [[0], [1, 2]] : memref<1x2x2xf32> into memref<1x4xf32>
    %cast = memref.cast %collapse_shape : memref<1x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
}

// ==========================================
// Phase: linalg to scf
// ==========================================
module {
  memref.global "private" constant @__constant_1x2x2xf32 : memref<1x2x2xf32> = dense<[[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]]> {alignment = 64 : i64}
  func.func private @printMemrefF32(memref<*xf32> {bufferization.access = "read"})
  func.func @host() {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+04 : f32
    %cst_0 = arith.constant 4.000000e+00 : f32
    %cst_1 = arith.constant -2.000000e+00 : f32
    %0 = memref.get_global @__constant_1x2x2xf32 : memref<1x2x2xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2xf32>
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %1 = arith.index_cast %arg0 : index to i64
      %2 = arith.uitofp %1 : i64 to f32
      %3 = arith.mulf %2, %cst_1 : f32
      %4 = arith.divf %3, %cst_0 : f32
      %5 = math.powf %cst, %4 : f32
      %6 = math.cos %5 : f32
      %7 = math.sin %5 : f32
      memref.store %6, %alloc[%arg0] : memref<2xf32>
      memref.store %7, %alloc_2[%arg0] : memref<2xf32>
    }
    %subview = memref.subview %0[0, 0, 0] [1, 2, 1] [1, 1, 1] : memref<1x2x2xf32> to memref<1x2x1xf32, strided<[4, 2, 1]>>
    %subview_3 = memref.subview %0[0, 0, 1] [1, 2, 1] [1, 1, 1] : memref<1x2x2xf32> to memref<1x2x1xf32, strided<[4, 2, 1], offset: 1>>
    %expand_shape = memref.expand_shape %alloc [[0, 1]] output_shape [2, 1] : memref<2xf32> into memref<2x1xf32>
    %expand_shape_4 = memref.expand_shape %alloc_2 [[0, 1]] output_shape [2, 1] : memref<2xf32> into memref<2x1xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x2x1xf32>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x2x1xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c2 step %c1 {
        scf.for %arg2 = %c0 to %c1 step %c1 {
          %1 = memref.load %subview[%arg0, %arg1, %arg2] : memref<1x2x1xf32, strided<[4, 2, 1]>>
          %2 = memref.load %subview_3[%arg0, %arg1, %arg2] : memref<1x2x1xf32, strided<[4, 2, 1], offset: 1>>
          %3 = memref.load %expand_shape[%arg1, %arg2] : memref<2x1xf32>
          %4 = memref.load %expand_shape_4[%arg1, %arg2] : memref<2x1xf32>
          %5 = arith.mulf %1, %3 : f32
          %6 = arith.mulf %2, %4 : f32
          %7 = arith.subf %5, %6 : f32
          %8 = arith.mulf %2, %3 : f32
          %9 = arith.mulf %1, %4 : f32
          %10 = arith.addf %8, %9 : f32
          memref.store %7, %alloc_5[%arg0, %arg1, %arg2] : memref<1x2x1xf32>
          memref.store %10, %alloc_6[%arg0, %arg1, %arg2] : memref<1x2x1xf32>
        }
      }
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x2x2xf32>
    %subview_8 = memref.subview %alloc_7[0, 0, 0] [1, 2, 1] [1, 1, 1] : memref<1x2x2xf32> to memref<1x2x1xf32, strided<[4, 2, 1]>>
    memref.copy %alloc_5, %subview_8 : memref<1x2x1xf32> to memref<1x2x1xf32, strided<[4, 2, 1]>>
    %subview_9 = memref.subview %alloc_7[0, 0, 1] [1, 2, 1] [1, 1, 1] : memref<1x2x2xf32> to memref<1x2x1xf32, strided<[4, 2, 1], offset: 1>>
    memref.copy %alloc_6, %subview_9 : memref<1x2x1xf32> to memref<1x2x1xf32, strided<[4, 2, 1], offset: 1>>
    %collapse_shape = memref.collapse_shape %alloc_7 [[0], [1, 2]] : memref<1x2x2xf32> into memref<1x4xf32>
    %cast = memref.cast %collapse_shape : memref<1x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
}

// ==========================================
// Phase: lower to llvm
// ==========================================
module {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_1x2x2xf32(dense<[[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]]> : tensor<1x2x2xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x array<2 x array<2 x f32>>>
  llvm.func @printMemrefF32(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @host() {
    %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %1 = llvm.mlir.constant(3 : i64) : i64
    %2 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.mlir.constant(64 : index) : i64
    %4 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %5 = llvm.mlir.addressof @__constant_1x2x2xf32 : !llvm.ptr
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.mlir.constant(2 : index) : i64
    %8 = llvm.mlir.constant(0 : index) : i64
    %9 = llvm.mlir.constant(1.000000e+04 : f32) : f32
    %10 = llvm.mlir.constant(4.000000e+00 : f32) : f32
    %11 = llvm.mlir.constant(-2.000000e+00 : f32) : f32
    %12 = llvm.mlir.constant(4 : index) : i64
    %13 = llvm.mlir.zero : !llvm.ptr
    %14 = llvm.getelementptr %5[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<2 x array<2 x f32>>>
    %15 = llvm.getelementptr %13[2] : (!llvm.ptr) -> !llvm.ptr, f32
    %16 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %17 = llvm.add %16, %3 : i64
    %18 = llvm.call @malloc(%17) : (i64) -> !llvm.ptr
    %19 = llvm.ptrtoint %18 : !llvm.ptr to i64
    %20 = llvm.sub %3, %6 : i64
    %21 = llvm.add %19, %20 : i64
    %22 = llvm.urem %21, %3  : i64
    %23 = llvm.sub %21, %22 : i64
    %24 = llvm.inttoptr %23 : i64 to !llvm.ptr
    %25 = llvm.call @malloc(%17) : (i64) -> !llvm.ptr
    %26 = llvm.ptrtoint %25 : !llvm.ptr to i64
    %27 = llvm.add %26, %20 : i64
    %28 = llvm.urem %27, %3  : i64
    %29 = llvm.sub %27, %28 : i64
    %30 = llvm.inttoptr %29 : i64 to !llvm.ptr
    llvm.br ^bb1(%8 : i64)
  ^bb1(%31: i64):  // 2 preds: ^bb0, ^bb2
    %32 = llvm.icmp "slt" %31, %7 : i64
    llvm.cond_br %32, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %33 = llvm.uitofp %31 : i64 to f32
    %34 = llvm.fmul %33, %11  : f32
    %35 = llvm.fdiv %34, %10  : f32
    %36 = llvm.intr.pow(%9, %35)  : (f32, f32) -> f32
    %37 = llvm.intr.cos(%36)  : (f32) -> f32
    %38 = llvm.intr.sin(%36)  : (f32) -> f32
    %39 = llvm.getelementptr %24[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %37, %39 : f32, !llvm.ptr
    %40 = llvm.getelementptr %30[%31] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %38, %40 : f32, !llvm.ptr
    %41 = llvm.add %31, %6 : i64
    llvm.br ^bb1(%41 : i64)
  ^bb3:  // pred: ^bb1
    %42 = llvm.call @malloc(%17) : (i64) -> !llvm.ptr
    %43 = llvm.ptrtoint %42 : !llvm.ptr to i64
    %44 = llvm.add %43, %20 : i64
    %45 = llvm.urem %44, %3  : i64
    %46 = llvm.sub %44, %45 : i64
    %47 = llvm.inttoptr %46 : i64 to !llvm.ptr
    %48 = llvm.insertvalue %42, %4[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %49 = llvm.insertvalue %47, %48[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %50 = llvm.insertvalue %8, %49[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %51 = llvm.insertvalue %6, %50[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %52 = llvm.insertvalue %7, %51[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %53 = llvm.insertvalue %6, %52[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %54 = llvm.insertvalue %7, %53[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %55 = llvm.insertvalue %6, %54[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %56 = llvm.insertvalue %6, %55[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %57 = llvm.call @malloc(%17) : (i64) -> !llvm.ptr
    %58 = llvm.ptrtoint %57 : !llvm.ptr to i64
    %59 = llvm.add %58, %20 : i64
    %60 = llvm.urem %59, %3  : i64
    %61 = llvm.sub %59, %60 : i64
    %62 = llvm.inttoptr %61 : i64 to !llvm.ptr
    %63 = llvm.insertvalue %57, %4[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %64 = llvm.insertvalue %62, %63[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %65 = llvm.insertvalue %8, %64[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %66 = llvm.insertvalue %6, %65[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %67 = llvm.insertvalue %7, %66[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %68 = llvm.insertvalue %6, %67[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %69 = llvm.insertvalue %7, %68[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %70 = llvm.insertvalue %6, %69[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %71 = llvm.insertvalue %6, %70[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb4(%8 : i64)
  ^bb4(%72: i64):  // 2 preds: ^bb3, ^bb11
    %73 = llvm.icmp "slt" %72, %6 : i64
    llvm.cond_br %73, ^bb5, ^bb12
  ^bb5:  // pred: ^bb4
    llvm.br ^bb6(%8 : i64)
  ^bb6(%74: i64):  // 2 preds: ^bb5, ^bb10
    %75 = llvm.icmp "slt" %74, %7 : i64
    llvm.cond_br %75, ^bb7, ^bb11
  ^bb7:  // pred: ^bb6
    llvm.br ^bb8(%8 : i64)
  ^bb8(%76: i64):  // 2 preds: ^bb7, ^bb9
    %77 = llvm.icmp "slt" %76, %6 : i64
    llvm.cond_br %77, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %78 = llvm.mul %72, %12 : i64
    %79 = llvm.mul %74, %7 : i64
    %80 = llvm.add %78, %79 : i64
    %81 = llvm.add %80, %76 : i64
    %82 = llvm.getelementptr %14[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %83 = llvm.load %82 : !llvm.ptr -> f32
    %84 = llvm.getelementptr %14[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %85 = llvm.getelementptr %84[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %86 = llvm.load %85 : !llvm.ptr -> f32
    %87 = llvm.add %74, %76 : i64
    %88 = llvm.getelementptr %24[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %89 = llvm.load %88 : !llvm.ptr -> f32
    %90 = llvm.getelementptr %30[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %91 = llvm.load %90 : !llvm.ptr -> f32
    %92 = llvm.fmul %83, %89  : f32
    %93 = llvm.fmul %86, %91  : f32
    %94 = llvm.fsub %92, %93  : f32
    %95 = llvm.fmul %86, %89  : f32
    %96 = llvm.fmul %83, %91  : f32
    %97 = llvm.fadd %95, %96  : f32
    %98 = llvm.mul %72, %7 : i64
    %99 = llvm.add %98, %74 : i64
    %100 = llvm.add %99, %76 : i64
    %101 = llvm.getelementptr %47[%100] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %94, %101 : f32, !llvm.ptr
    %102 = llvm.getelementptr %62[%100] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %97, %102 : f32, !llvm.ptr
    %103 = llvm.add %76, %6 : i64
    llvm.br ^bb8(%103 : i64)
  ^bb10:  // pred: ^bb8
    %104 = llvm.add %74, %6 : i64
    llvm.br ^bb6(%104 : i64)
  ^bb11:  // pred: ^bb6
    %105 = llvm.add %72, %6 : i64
    llvm.br ^bb4(%105 : i64)
  ^bb12:  // pred: ^bb4
    %106 = llvm.getelementptr %13[4] : (!llvm.ptr) -> !llvm.ptr, f32
    %107 = llvm.ptrtoint %106 : !llvm.ptr to i64
    %108 = llvm.add %107, %3 : i64
    %109 = llvm.call @malloc(%108) : (i64) -> !llvm.ptr
    %110 = llvm.ptrtoint %109 : !llvm.ptr to i64
    %111 = llvm.add %110, %20 : i64
    %112 = llvm.urem %111, %3  : i64
    %113 = llvm.sub %111, %112 : i64
    %114 = llvm.inttoptr %113 : i64 to !llvm.ptr
    %115 = llvm.insertvalue %109, %4[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %116 = llvm.insertvalue %114, %115[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %117 = llvm.insertvalue %8, %116[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %118 = llvm.insertvalue %6, %117[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %119 = llvm.insertvalue %12, %118[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %120 = llvm.insertvalue %7, %119[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %121 = llvm.insertvalue %7, %120[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %122 = llvm.insertvalue %6, %121[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %123 = llvm.insertvalue %6, %122[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %124 = llvm.intr.stacksave : !llvm.ptr
    %125 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %56, %125 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %126 = llvm.insertvalue %1, %0[0] : !llvm.struct<(i64, ptr)> 
    %127 = llvm.insertvalue %125, %126[1] : !llvm.struct<(i64, ptr)> 
    %128 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %123, %128 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %129 = llvm.insertvalue %128, %126[1] : !llvm.struct<(i64, ptr)> 
    %130 = llvm.alloca %6 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %127, %130 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %131 = llvm.alloca %6 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %129, %131 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %132 = llvm.getelementptr %13[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %133 = llvm.ptrtoint %132 : !llvm.ptr to i64
    llvm.call @memrefCopy(%133, %130, %131) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %124 : !llvm.ptr
    %134 = llvm.insertvalue %6, %116[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %135 = llvm.insertvalue %6, %134[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %136 = llvm.insertvalue %12, %135[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %137 = llvm.insertvalue %7, %136[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %138 = llvm.insertvalue %7, %137[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %139 = llvm.insertvalue %6, %138[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %140 = llvm.insertvalue %6, %139[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %141 = llvm.intr.stacksave : !llvm.ptr
    %142 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %71, %142 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %143 = llvm.insertvalue %142, %126[1] : !llvm.struct<(i64, ptr)> 
    %144 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %140, %144 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %145 = llvm.insertvalue %144, %126[1] : !llvm.struct<(i64, ptr)> 
    %146 = llvm.alloca %6 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %143, %146 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %147 = llvm.alloca %6 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %145, %147 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    llvm.call @memrefCopy(%133, %146, %147) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %141 : !llvm.ptr
    %148 = llvm.insertvalue %109, %2[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %149 = llvm.insertvalue %114, %148[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %150 = llvm.insertvalue %8, %149[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %151 = llvm.insertvalue %6, %150[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %152 = llvm.insertvalue %12, %151[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %153 = llvm.insertvalue %12, %152[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %154 = llvm.insertvalue %6, %153[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %155 = llvm.alloca %6 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %154, %155 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.call @printMemrefF32(%7, %155) : (i64, !llvm.ptr) -> ()
    llvm.return
  }
}
