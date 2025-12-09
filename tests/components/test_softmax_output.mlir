// Original IR loaded from file
module {
  cherry.func private @test_softmax(%arg0: !cherry.cherry_tensor<[1x1024x768xf32]>) -> (!cherry.cherry_tensor<[?xi64]>, !cherry.cherry_tensor<[?xf32]>) {
    %0 = cherry.argmax %arg0 dim 1 : (!cherry.cherry_tensor<[1x1024x768xf32]>) -> !cherry.cherry_tensor<[?xi64]>
    %1 = cherry.softmax %arg0 axis 1 : (!cherry.cherry_tensor<[1x1024x768xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %0, %1 : !cherry.cherry_tensor<[?xi64]>, !cherry.cherry_tensor<[?xf32]>
  }
  cherry.func @host() -> (!cherry.cherry_tensor<[?xi64]>, !cherry.cherry_tensor<[?xf32]>) {
    %0 = cherry.create_tensor dense<3.000000e-01> : tensor<1x1024x768xf32> -> !cherry.cherry_tensor<[1x1024x768xf32]>
    %1:2 = cherry.call @test_softmax(%0) : (!cherry.cherry_tensor<[1x1024x768xf32]>) -> (!cherry.cherry_tensor<[?xi64]>, !cherry.cherry_tensor<[?xf32]>)
    cherry.return %1#0, %1#1 : !cherry.cherry_tensor<[?xi64]>, !cherry.cherry_tensor<[?xf32]>
  }
}

// ==========================================
// Phase: Inliner
// ==========================================
module {
  cherry.func @host() -> (!cherry.cherry_tensor<[?xi64]>, !cherry.cherry_tensor<[?xf32]>) {
    %0 = cherry.create_tensor dense<3.000000e-01> : tensor<1x1024x768xf32> -> !cherry.cherry_tensor<[1x1024x768xf32]>
    %1 = cherry.argmax %0 dim 1 : (!cherry.cherry_tensor<[1x1024x768xf32]>) -> !cherry.cherry_tensor<[?xi64]>
    %2 = cherry.softmax %0 axis 1 : (!cherry.cherry_tensor<[1x1024x768xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %1, %2 : !cherry.cherry_tensor<[?xi64]>, !cherry.cherry_tensor<[?xf32]>
  }
}

// ==========================================
// Phase: Shape Inference
// ==========================================
module {
  cherry.func @host() -> (!cherry.cherry_tensor<[1x768xi64]>, !cherry.cherry_tensor<[1x1024x768xf32]>) {
    %0 = cherry.create_tensor dense<3.000000e-01> : tensor<1x1024x768xf32> -> !cherry.cherry_tensor<[1x1024x768xf32]>
    %1 = cherry.argmax %0 dim 1 : (!cherry.cherry_tensor<[1x1024x768xf32]>) -> !cherry.cherry_tensor<[1x768xi64]>
    %2 = cherry.softmax %0 axis 1 : (!cherry.cherry_tensor<[1x1024x768xf32]>) -> !cherry.cherry_tensor<[1x1024x768xf32]>
    cherry.return %1, %2 : !cherry.cherry_tensor<[1x768xi64]>, !cherry.cherry_tensor<[1x1024x768xf32]>
  }
}

// ==========================================
// Phase: Canonicalizer
// ==========================================
module {
  cherry.func @host() -> (!cherry.cherry_tensor<[1x768xi64]>, !cherry.cherry_tensor<[1x1024x768xf32]>) {
    %0 = cherry.create_tensor dense<3.000000e-01> : tensor<1x1024x768xf32> -> !cherry.cherry_tensor<[1x1024x768xf32]>
    %1 = cherry.argmax %0 dim 1 : (!cherry.cherry_tensor<[1x1024x768xf32]>) -> !cherry.cherry_tensor<[1x768xi64]>
    %2 = cherry.softmax %0 axis 1 : (!cherry.cherry_tensor<[1x1024x768xf32]>) -> !cherry.cherry_tensor<[1x1024x768xf32]>
    cherry.return %1, %2 : !cherry.cherry_tensor<[1x768xi64]>, !cherry.cherry_tensor<[1x1024x768xf32]>
  }
}

// ==========================================
// Phase: Convert to Linalg
// ==========================================
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
module {
  func.func @host() -> (tensor<1x768xi64>, tensor<1x1024x768xf32>) {
    %cst = arith.constant dense<3.000000e-01> : tensor<1x1024x768xf32>
    %cst_0 = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<1x768xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x768xf32>) -> tensor<1x768xf32>
    %c0_i64 = arith.constant 0 : i64
    %2 = tensor.empty() : tensor<1x768xi64>
    %3 = linalg.fill ins(%c0_i64 : i64) outs(%2 : tensor<1x768xi64>) -> tensor<1x768xi64>
    %4:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction", "parallel"]} ins(%cst : tensor<1x1024x768xf32>) outs(%1, %3 : tensor<1x768xf32>, tensor<1x768xi64>) {
    ^bb0(%in: f32, %out: f32, %out_3: i64):
      %15 = linalg.index 1 : index
      %16 = arith.index_cast %15 : index to i64
      %17 = arith.cmpf ogt, %in, %out : f32
      %18 = arith.select %17, %in, %out : f32
      %19 = arith.select %17, %16, %out_3 : i64
      linalg.yield %18, %19 : f32, i64
    } -> (tensor<1x768xf32>, tensor<1x768xi64>)
    %5 = tensor.empty() : tensor<1x768xf32>
    %cst_1 = arith.constant 0xFF800000 : f32
    %6 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<1x768xf32>) -> tensor<1x768xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "parallel"]} ins(%cst : tensor<1x1024x768xf32>) outs(%6 : tensor<1x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %15 = arith.maxnumf %in, %out : f32
      linalg.yield %15 : f32
    } -> tensor<1x768xf32>
    %8 = tensor.empty() : tensor<1x1024x768xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst, %7 : tensor<1x1024x768xf32>, tensor<1x768xf32>) outs(%8 : tensor<1x1024x768xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %15 = arith.subf %in, %in_3 : f32
      %16 = math.exp %15 : f32
      linalg.yield %16 : f32
    } -> tensor<1x1024x768xf32>
    %10 = tensor.empty() : tensor<1x768xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %11 = linalg.fill ins(%cst_2 : f32) outs(%10 : tensor<1x768xf32>) -> tensor<1x768xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<1x1024x768xf32>) outs(%11 : tensor<1x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %15 = arith.addf %in, %out : f32
      linalg.yield %15 : f32
    } -> tensor<1x768xf32>
    %13 = tensor.empty() : tensor<1x1024x768xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %12 : tensor<1x1024x768xf32>, tensor<1x768xf32>) outs(%13 : tensor<1x1024x768xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %15 = arith.divf %in, %in_3 : f32
      linalg.yield %15 : f32
    } -> tensor<1x1024x768xf32>
    return %4#1, %14 : tensor<1x768xi64>, tensor<1x1024x768xf32>
  }
}

// ==========================================
// Phase: Linalg Tiling
// ==========================================
#map = affine_map<(d0) -> (-d0 + 1, 8)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1) -> (d0 + d1)>
module {
  func.func @host() -> (tensor<1x768xi64>, tensor<1x1024x768xf32>) {
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
    %cst = arith.constant 0.000000e+00 : f32
    %cst_26 = arith.constant 3.000000e-01 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_27 = arith.constant 0xFF800000 : f32
    %cst_28 = arith.constant dense<3.000000e-01> : tensor<1x1024x768xf32>
    %0 = tensor.empty() : tensor<1x768xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_29 = arith.constant 8 : index
    %c0_30 = arith.constant 0 : index
    %c768 = arith.constant 768 : index
    %c8_31 = arith.constant 8 : index
    %1 = scf.for %arg0 = %c0 to %c1 step %c8_29 iter_args(%arg1 = %0) -> (tensor<1x768xf32>) {
      %15 = scf.for %arg2 = %c0_30 to %c768 step %c8_31 iter_args(%arg3 = %arg1) -> (tensor<1x768xf32>) {
        %16 = affine.min #map(%arg0)
        %extracted_slice = tensor.extract_slice %arg3[%arg0, %arg2] [%16, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
        %17 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice : tensor<?x8xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_27 : f32
        } -> tensor<?x8xf32>
        %inserted_slice = tensor.insert_slice %17 into %arg3[%arg0, %arg2] [%16, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
        scf.yield %inserted_slice : tensor<1x768xf32>
      }
      scf.yield %15 : tensor<1x768xf32>
    }
    %2 = tensor.empty() : tensor<1x768xi64>
    %c0_32 = arith.constant 0 : index
    %c1_33 = arith.constant 1 : index
    %c8_34 = arith.constant 8 : index
    %c0_35 = arith.constant 0 : index
    %c768_36 = arith.constant 768 : index
    %c8_37 = arith.constant 8 : index
    %3 = scf.for %arg0 = %c0_32 to %c1_33 step %c8_34 iter_args(%arg1 = %2) -> (tensor<1x768xi64>) {
      %15 = scf.for %arg2 = %c0_35 to %c768_36 step %c8_37 iter_args(%arg3 = %arg1) -> (tensor<1x768xi64>) {
        %16 = affine.min #map(%arg0)
        %extracted_slice = tensor.extract_slice %arg3[%arg0, %arg2] [%16, 8] [1, 1] : tensor<1x768xi64> to tensor<?x8xi64>
        %17 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice : tensor<?x8xi64>) {
        ^bb0(%out: i64):
          linalg.yield %c0_i64 : i64
        } -> tensor<?x8xi64>
        %inserted_slice = tensor.insert_slice %17 into %arg3[%arg0, %arg2] [%16, 8] [1, 1] : tensor<?x8xi64> into tensor<1x768xi64>
        scf.yield %inserted_slice : tensor<1x768xi64>
      }
      scf.yield %15 : tensor<1x768xi64>
    }
    %c0_38 = arith.constant 0 : index
    %c1_39 = arith.constant 1 : index
    %c8_40 = arith.constant 8 : index
    %c0_41 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c8_42 = arith.constant 8 : index
    %c0_43 = arith.constant 0 : index
    %c768_44 = arith.constant 768 : index
    %c8_45 = arith.constant 8 : index
    %4:2 = scf.for %arg0 = %c0_38 to %c1_39 step %c8_40 iter_args(%arg1 = %1, %arg2 = %3) -> (tensor<1x768xf32>, tensor<1x768xi64>) {
      %15:2 = scf.for %arg3 = %c0_41 to %c1024 step %c8_42 iter_args(%arg4 = %arg1, %arg5 = %arg2) -> (tensor<1x768xf32>, tensor<1x768xi64>) {
        %16:2 = scf.for %arg6 = %c0_43 to %c768_44 step %c8_45 iter_args(%arg7 = %arg4, %arg8 = %arg5) -> (tensor<1x768xf32>, tensor<1x768xi64>) {
          %17 = affine.min #map(%arg0)
          %18 = affine.min #map(%arg0)
          %19 = affine.min #map(%arg0)
          %extracted_slice = tensor.extract_slice %cst_28[%arg0, %arg3, %arg6] [%17, 8, 8] [1, 1, 1] : tensor<1x1024x768xf32> to tensor<?x8x8xf32>
          %extracted_slice_94 = tensor.extract_slice %arg7[%arg0, %arg6] [%18, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
          %extracted_slice_95 = tensor.extract_slice %arg8[%arg0, %arg6] [%19, 8] [1, 1] : tensor<1x768xi64> to tensor<?x8xi64>
          %20:2 = linalg.generic {indexing_maps = [#map2, #map3, #map3], iterator_types = ["parallel", "reduction", "parallel"]} ins(%extracted_slice : tensor<?x8x8xf32>) outs(%extracted_slice_94, %extracted_slice_95 : tensor<?x8xf32>, tensor<?x8xi64>) {
          ^bb0(%in: f32, %out: f32, %out_97: i64):
            %21 = linalg.index 1 : index
            %22 = affine.apply #map4(%21, %arg3)
            %23 = arith.index_cast %22 : index to i64
            %24 = arith.cmpf ogt, %in, %out : f32
            %25 = arith.select %24, %in, %out : f32
            %26 = arith.select %24, %23, %out_97 : i64
            linalg.yield %25, %26 : f32, i64
          } -> (tensor<?x8xf32>, tensor<?x8xi64>)
          %inserted_slice = tensor.insert_slice %20#0 into %arg7[%arg0, %arg6] [%18, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
          %inserted_slice_96 = tensor.insert_slice %20#1 into %arg8[%arg0, %arg6] [%19, 8] [1, 1] : tensor<?x8xi64> into tensor<1x768xi64>
          scf.yield %inserted_slice, %inserted_slice_96 : tensor<1x768xf32>, tensor<1x768xi64>
        }
        scf.yield %16#0, %16#1 : tensor<1x768xf32>, tensor<1x768xi64>
      }
      scf.yield %15#0, %15#1 : tensor<1x768xf32>, tensor<1x768xi64>
    }
    %5 = tensor.empty() : tensor<1x768xf32>
    %c0_46 = arith.constant 0 : index
    %c1_47 = arith.constant 1 : index
    %c8_48 = arith.constant 8 : index
    %c0_49 = arith.constant 0 : index
    %c768_50 = arith.constant 768 : index
    %c8_51 = arith.constant 8 : index
    %6 = scf.for %arg0 = %c0_46 to %c1_47 step %c8_48 iter_args(%arg1 = %5) -> (tensor<1x768xf32>) {
      %15 = scf.for %arg2 = %c0_49 to %c768_50 step %c8_51 iter_args(%arg3 = %arg1) -> (tensor<1x768xf32>) {
        %16 = affine.min #map(%arg0)
        %extracted_slice = tensor.extract_slice %arg3[%arg0, %arg2] [%16, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
        %17 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice : tensor<?x8xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_27 : f32
        } -> tensor<?x8xf32>
        %inserted_slice = tensor.insert_slice %17 into %arg3[%arg0, %arg2] [%16, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
        scf.yield %inserted_slice : tensor<1x768xf32>
      }
      scf.yield %15 : tensor<1x768xf32>
    }
    %c0_52 = arith.constant 0 : index
    %c1_53 = arith.constant 1 : index
    %c8_54 = arith.constant 8 : index
    %c0_55 = arith.constant 0 : index
    %c1024_56 = arith.constant 1024 : index
    %c8_57 = arith.constant 8 : index
    %c0_58 = arith.constant 0 : index
    %c768_59 = arith.constant 768 : index
    %c8_60 = arith.constant 8 : index
    %7 = scf.for %arg0 = %c0_52 to %c1_53 step %c8_54 iter_args(%arg1 = %6) -> (tensor<1x768xf32>) {
      %15 = scf.for %arg2 = %c0_55 to %c1024_56 step %c8_57 iter_args(%arg3 = %arg1) -> (tensor<1x768xf32>) {
        %16 = scf.for %arg4 = %c0_58 to %c768_59 step %c8_60 iter_args(%arg5 = %arg3) -> (tensor<1x768xf32>) {
          %17 = affine.min #map(%arg0)
          %18 = affine.min #map(%arg0)
          %extracted_slice = tensor.extract_slice %cst_28[%arg0, %arg2, %arg4] [%17, 8, 8] [1, 1, 1] : tensor<1x1024x768xf32> to tensor<?x8x8xf32>
          %extracted_slice_94 = tensor.extract_slice %arg5[%arg0, %arg4] [%18, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
          %19 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction", "parallel"]} ins(%extracted_slice : tensor<?x8x8xf32>) outs(%extracted_slice_94 : tensor<?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %20 = arith.maxnumf %in, %out : f32
            linalg.yield %20 : f32
          } -> tensor<?x8xf32>
          %inserted_slice = tensor.insert_slice %19 into %arg5[%arg0, %arg4] [%18, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
          scf.yield %inserted_slice : tensor<1x768xf32>
        }
        scf.yield %16 : tensor<1x768xf32>
      }
      scf.yield %15 : tensor<1x768xf32>
    }
    %8 = tensor.empty() : tensor<1x1024x768xf32>
    %c0_61 = arith.constant 0 : index
    %c1_62 = arith.constant 1 : index
    %c8_63 = arith.constant 8 : index
    %c0_64 = arith.constant 0 : index
    %c1024_65 = arith.constant 1024 : index
    %c8_66 = arith.constant 8 : index
    %c0_67 = arith.constant 0 : index
    %c768_68 = arith.constant 768 : index
    %c8_69 = arith.constant 8 : index
    %9 = scf.for %arg0 = %c0_61 to %c1_62 step %c8_63 iter_args(%arg1 = %8) -> (tensor<1x1024x768xf32>) {
      %15 = scf.for %arg2 = %c0_64 to %c1024_65 step %c8_66 iter_args(%arg3 = %arg1) -> (tensor<1x1024x768xf32>) {
        %16 = scf.for %arg4 = %c0_67 to %c768_68 step %c8_69 iter_args(%arg5 = %arg3) -> (tensor<1x1024x768xf32>) {
          %17 = affine.min #map(%arg0)
          %18 = affine.min #map(%arg0)
          %extracted_slice = tensor.extract_slice %7[%arg0, %arg4] [%17, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
          %extracted_slice_94 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%18, 8, 8] [1, 1, 1] : tensor<1x1024x768xf32> to tensor<?x8x8xf32>
          %19 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<?x8xf32>) outs(%extracted_slice_94 : tensor<?x8x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %20 = arith.subf %cst_26, %in : f32
            %21 = math.exp %20 : f32
            linalg.yield %21 : f32
          } -> tensor<?x8x8xf32>
          %inserted_slice = tensor.insert_slice %19 into %arg5[%arg0, %arg2, %arg4] [%18, 8, 8] [1, 1, 1] : tensor<?x8x8xf32> into tensor<1x1024x768xf32>
          scf.yield %inserted_slice : tensor<1x1024x768xf32>
        }
        scf.yield %16 : tensor<1x1024x768xf32>
      }
      scf.yield %15 : tensor<1x1024x768xf32>
    }
    %10 = tensor.empty() : tensor<1x768xf32>
    %c0_70 = arith.constant 0 : index
    %c1_71 = arith.constant 1 : index
    %c8_72 = arith.constant 8 : index
    %c0_73 = arith.constant 0 : index
    %c768_74 = arith.constant 768 : index
    %c8_75 = arith.constant 8 : index
    %11 = scf.for %arg0 = %c0_70 to %c1_71 step %c8_72 iter_args(%arg1 = %10) -> (tensor<1x768xf32>) {
      %15 = scf.for %arg2 = %c0_73 to %c768_74 step %c8_75 iter_args(%arg3 = %arg1) -> (tensor<1x768xf32>) {
        %16 = affine.min #map(%arg0)
        %extracted_slice = tensor.extract_slice %arg3[%arg0, %arg2] [%16, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
        %17 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice : tensor<?x8xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<?x8xf32>
        %inserted_slice = tensor.insert_slice %17 into %arg3[%arg0, %arg2] [%16, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
        scf.yield %inserted_slice : tensor<1x768xf32>
      }
      scf.yield %15 : tensor<1x768xf32>
    }
    %c0_76 = arith.constant 0 : index
    %c1_77 = arith.constant 1 : index
    %c8_78 = arith.constant 8 : index
    %c0_79 = arith.constant 0 : index
    %c1024_80 = arith.constant 1024 : index
    %c8_81 = arith.constant 8 : index
    %c0_82 = arith.constant 0 : index
    %c768_83 = arith.constant 768 : index
    %c8_84 = arith.constant 8 : index
    %12 = scf.for %arg0 = %c0_76 to %c1_77 step %c8_78 iter_args(%arg1 = %11) -> (tensor<1x768xf32>) {
      %15 = scf.for %arg2 = %c0_79 to %c1024_80 step %c8_81 iter_args(%arg3 = %arg1) -> (tensor<1x768xf32>) {
        %16 = scf.for %arg4 = %c0_82 to %c768_83 step %c8_84 iter_args(%arg5 = %arg3) -> (tensor<1x768xf32>) {
          %17 = affine.min #map(%arg0)
          %18 = affine.min #map(%arg0)
          %extracted_slice = tensor.extract_slice %9[%arg0, %arg2, %arg4] [%17, 8, 8] [1, 1, 1] : tensor<1x1024x768xf32> to tensor<?x8x8xf32>
          %extracted_slice_94 = tensor.extract_slice %arg5[%arg0, %arg4] [%18, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
          %19 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<?x8x8xf32>) outs(%extracted_slice_94 : tensor<?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %20 = arith.addf %in, %out : f32
            linalg.yield %20 : f32
          } -> tensor<?x8xf32>
          %inserted_slice = tensor.insert_slice %19 into %arg5[%arg0, %arg4] [%18, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
          scf.yield %inserted_slice : tensor<1x768xf32>
        }
        scf.yield %16 : tensor<1x768xf32>
      }
      scf.yield %15 : tensor<1x768xf32>
    }
    %13 = tensor.empty() : tensor<1x1024x768xf32>
    %c0_85 = arith.constant 0 : index
    %c1_86 = arith.constant 1 : index
    %c8_87 = arith.constant 8 : index
    %c0_88 = arith.constant 0 : index
    %c1024_89 = arith.constant 1024 : index
    %c8_90 = arith.constant 8 : index
    %c0_91 = arith.constant 0 : index
    %c768_92 = arith.constant 768 : index
    %c8_93 = arith.constant 8 : index
    %14 = scf.for %arg0 = %c0_85 to %c1_86 step %c8_87 iter_args(%arg1 = %13) -> (tensor<1x1024x768xf32>) {
      %15 = scf.for %arg2 = %c0_88 to %c1024_89 step %c8_90 iter_args(%arg3 = %arg1) -> (tensor<1x1024x768xf32>) {
        %16 = scf.for %arg4 = %c0_91 to %c768_92 step %c8_93 iter_args(%arg5 = %arg3) -> (tensor<1x1024x768xf32>) {
          %17 = affine.min #map(%arg0)
          %18 = affine.min #map(%arg0)
          %19 = affine.min #map(%arg0)
          %extracted_slice = tensor.extract_slice %9[%arg0, %arg2, %arg4] [%17, 8, 8] [1, 1, 1] : tensor<1x1024x768xf32> to tensor<?x8x8xf32>
          %extracted_slice_94 = tensor.extract_slice %12[%arg0, %arg4] [%18, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
          %extracted_slice_95 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%19, 8, 8] [1, 1, 1] : tensor<1x1024x768xf32> to tensor<?x8x8xf32>
          %20 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_94 : tensor<?x8x8xf32>, tensor<?x8xf32>) outs(%extracted_slice_95 : tensor<?x8x8xf32>) {
          ^bb0(%in: f32, %in_96: f32, %out: f32):
            %21 = arith.divf %in, %in_96 : f32
            linalg.yield %21 : f32
          } -> tensor<?x8x8xf32>
          %inserted_slice = tensor.insert_slice %20 into %arg5[%arg0, %arg2, %arg4] [%19, 8, 8] [1, 1, 1] : tensor<?x8x8xf32> into tensor<1x1024x768xf32>
          scf.yield %inserted_slice : tensor<1x1024x768xf32>
        }
        scf.yield %16 : tensor<1x1024x768xf32>
      }
      scf.yield %15 : tensor<1x1024x768xf32>
    }
    return %4#1, %14 : tensor<1x768xi64>, tensor<1x1024x768xf32>
  }
}

// ==========================================
// Phase: Bufferization
// ==========================================
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1) -> (d0 + d1)>
module {
  memref.global "private" constant @__constant_1x1024x768xf32 : memref<1x1024x768xf32> = dense<3.000000e-01> {alignment = 64 : i64}
  func.func @host() -> (memref<1x768xi64>, memref<1x1024x768xf32>) {
    %c1024 = arith.constant 1024 : index
    %c768 = arith.constant 768 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 3.000000e-01 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0xFF800000 : f32
    %0 = memref.get_global @__constant_1x1024x768xf32 : memref<1x1024x768xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
    scf.for %arg0 = %c0 to %c768 step %c8 {
      %subview = memref.subview %alloc[0, %arg0] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%subview : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst_1 : f32
      }
      memref.copy %subview, %subview : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x768xi64>
    scf.for %arg0 = %c0 to %c768 step %c8 {
      %subview = memref.subview %alloc_2[0, %arg0] [1, 8] [1, 1] : memref<1x768xi64> to memref<1x8xi64, strided<[768, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%subview : memref<1x8xi64, strided<[768, 1], offset: ?>>) {
      ^bb0(%out: i64):
        linalg.yield %c0_i64 : i64
      }
      memref.copy %subview, %subview : memref<1x8xi64, strided<[768, 1], offset: ?>> to memref<1x8xi64, strided<[768, 1], offset: ?>>
    }
    scf.for %arg0 = %c0 to %c1024 step %c8 {
      scf.for %arg1 = %c0 to %c768 step %c8 {
        %subview = memref.subview %0[0, %arg0, %arg1] [1, 8, 8] [1, 1, 1] : memref<1x1024x768xf32> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
        %subview_7 = memref.subview %alloc[0, %arg1] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        %subview_8 = memref.subview %alloc_2[0, %arg1] [1, 8] [1, 1] : memref<1x768xi64> to memref<1x8xi64, strided<[768, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "reduction", "parallel"]} ins(%subview : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>) outs(%subview_7, %subview_8 : memref<1x8xf32, strided<[768, 1], offset: ?>>, memref<1x8xi64, strided<[768, 1], offset: ?>>) {
        ^bb0(%in: f32, %out: f32, %out_9: i64):
          %1 = linalg.index 1 : index
          %2 = affine.apply #map3(%1, %arg0)
          %3 = arith.index_cast %2 : index to i64
          %4 = arith.cmpf ogt, %in, %out : f32
          %5 = arith.select %4, %in, %out : f32
          %6 = arith.select %4, %3, %out_9 : i64
          linalg.yield %5, %6 : f32, i64
        }
        memref.copy %subview_7, %subview_7 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        memref.copy %subview_8, %subview_8 : memref<1x8xi64, strided<[768, 1], offset: ?>> to memref<1x8xi64, strided<[768, 1], offset: ?>>
      }
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
    scf.for %arg0 = %c0 to %c768 step %c8 {
      %subview = memref.subview %alloc_3[0, %arg0] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%subview : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst_1 : f32
      }
      memref.copy %subview, %subview : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
    }
    scf.for %arg0 = %c0 to %c1024 step %c8 {
      scf.for %arg1 = %c0 to %c768 step %c8 {
        %subview = memref.subview %0[0, %arg0, %arg1] [1, 8, 8] [1, 1, 1] : memref<1x1024x768xf32> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
        %subview_7 = memref.subview %alloc_3[0, %arg1] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction", "parallel"]} ins(%subview : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>) outs(%subview_7 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
        ^bb0(%in: f32, %out: f32):
          %1 = arith.maxnumf %in, %out : f32
          linalg.yield %1 : f32
        }
        memref.copy %subview_7, %subview_7 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
      }
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x1024x768xf32>
    scf.for %arg0 = %c0 to %c1024 step %c8 {
      scf.for %arg1 = %c0 to %c768 step %c8 {
        %subview = memref.subview %alloc_3[0, %arg1] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        %subview_7 = memref.subview %alloc_4[0, %arg0, %arg1] [1, 8, 8] [1, 1, 1] : memref<1x1024x768xf32> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview : memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%subview_7 : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>) {
        ^bb0(%in: f32, %out: f32):
          %1 = arith.subf %cst_0, %in : f32
          %2 = math.exp %1 : f32
          linalg.yield %2 : f32
        }
        memref.copy %subview_7, %subview_7 : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
      }
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
    scf.for %arg0 = %c0 to %c768 step %c8 {
      %subview = memref.subview %alloc_5[0, %arg0] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%subview : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
      ^bb0(%out: f32):
        linalg.yield %cst : f32
      }
      memref.copy %subview, %subview : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
    }
    scf.for %arg0 = %c0 to %c1024 step %c8 {
      scf.for %arg1 = %c0 to %c768 step %c8 {
        %subview = memref.subview %alloc_4[0, %arg0, %arg1] [1, 8, 8] [1, 1, 1] : memref<1x1024x768xf32> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
        %subview_7 = memref.subview %alloc_5[0, %arg1] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>) outs(%subview_7 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
        ^bb0(%in: f32, %out: f32):
          %1 = arith.addf %in, %out : f32
          linalg.yield %1 : f32
        }
        memref.copy %subview_7, %subview_7 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
      }
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x1024x768xf32>
    scf.for %arg0 = %c0 to %c1024 step %c8 {
      scf.for %arg1 = %c0 to %c768 step %c8 {
        %subview = memref.subview %alloc_4[0, %arg0, %arg1] [1, 8, 8] [1, 1, 1] : memref<1x1024x768xf32> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
        %subview_7 = memref.subview %alloc_5[0, %arg1] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        %subview_8 = memref.subview %alloc_6[0, %arg0, %arg1] [1, 8, 8] [1, 1, 1] : memref<1x1024x768xf32> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%subview, %subview_7 : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>, memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%subview_8 : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>) {
        ^bb0(%in: f32, %in_9: f32, %out: f32):
          %1 = arith.divf %in, %in_9 : f32
          linalg.yield %1 : f32
        }
        memref.copy %subview_8, %subview_8 : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
      }
    }
    return %alloc_2, %alloc_6 : memref<1x768xi64>, memref<1x1024x768xf32>
  }
}

// ==========================================
// Phase: linalg to scf
// ==========================================
#map = affine_map<(d0, d1) -> (d0 + d1)>
module {
  memref.global "private" constant @__constant_1x1024x768xf32 : memref<1x1024x768xf32> = dense<3.000000e-01> {alignment = 64 : i64}
  func.func @host() -> (memref<1x768xi64>, memref<1x1024x768xf32>) {
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c768 = arith.constant 768 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 3.000000e-01 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0xFF800000 : f32
    %0 = memref.get_global @__constant_1x1024x768xf32 : memref<1x1024x768xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
    scf.for %arg0 = %c0 to %c768 step %c8 {
      %subview = memref.subview %alloc[0, %arg0] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          memref.store %cst_1, %subview[%arg1, %arg2] : memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
      }
      memref.copy %subview, %subview : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x768xi64>
    scf.for %arg0 = %c0 to %c768 step %c8 {
      %subview = memref.subview %alloc_2[0, %arg0] [1, 8] [1, 1] : memref<1x768xi64> to memref<1x8xi64, strided<[768, 1], offset: ?>>
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          memref.store %c0_i64, %subview[%arg1, %arg2] : memref<1x8xi64, strided<[768, 1], offset: ?>>
        }
      }
      memref.copy %subview, %subview : memref<1x8xi64, strided<[768, 1], offset: ?>> to memref<1x8xi64, strided<[768, 1], offset: ?>>
    }
    scf.for %arg0 = %c0 to %c1024 step %c8 {
      scf.for %arg1 = %c0 to %c768 step %c8 {
        %subview = memref.subview %0[0, %arg0, %arg1] [1, 8, 8] [1, 1, 1] : memref<1x1024x768xf32> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
        %subview_7 = memref.subview %alloc[0, %arg1] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        %subview_8 = memref.subview %alloc_2[0, %arg1] [1, 8] [1, 1] : memref<1x768xi64> to memref<1x8xi64, strided<[768, 1], offset: ?>>
        scf.for %arg2 = %c0 to %c1 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            scf.for %arg4 = %c0 to %c8 step %c1 {
              %1 = memref.load %subview[%arg2, %arg3, %arg4] : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
              %2 = memref.load %subview_7[%arg2, %arg4] : memref<1x8xf32, strided<[768, 1], offset: ?>>
              %3 = memref.load %subview_8[%arg2, %arg4] : memref<1x8xi64, strided<[768, 1], offset: ?>>
              %4 = affine.apply #map(%arg3, %arg0)
              %5 = arith.index_cast %4 : index to i64
              %6 = arith.cmpf ogt, %1, %2 : f32
              %7 = arith.select %6, %1, %2 : f32
              %8 = arith.select %6, %5, %3 : i64
              memref.store %7, %subview_7[%arg2, %arg4] : memref<1x8xf32, strided<[768, 1], offset: ?>>
              memref.store %8, %subview_8[%arg2, %arg4] : memref<1x8xi64, strided<[768, 1], offset: ?>>
            }
          }
        }
        memref.copy %subview_7, %subview_7 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        memref.copy %subview_8, %subview_8 : memref<1x8xi64, strided<[768, 1], offset: ?>> to memref<1x8xi64, strided<[768, 1], offset: ?>>
      }
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
    scf.for %arg0 = %c0 to %c768 step %c8 {
      %subview = memref.subview %alloc_3[0, %arg0] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          memref.store %cst_1, %subview[%arg1, %arg2] : memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
      }
      memref.copy %subview, %subview : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
    }
    scf.for %arg0 = %c0 to %c1024 step %c8 {
      scf.for %arg1 = %c0 to %c768 step %c8 {
        %subview = memref.subview %0[0, %arg0, %arg1] [1, 8, 8] [1, 1, 1] : memref<1x1024x768xf32> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
        %subview_7 = memref.subview %alloc_3[0, %arg1] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        scf.for %arg2 = %c0 to %c1 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            scf.for %arg4 = %c0 to %c8 step %c1 {
              %1 = memref.load %subview[%arg2, %arg3, %arg4] : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
              %2 = memref.load %subview_7[%arg2, %arg4] : memref<1x8xf32, strided<[768, 1], offset: ?>>
              %3 = arith.maxnumf %1, %2 : f32
              memref.store %3, %subview_7[%arg2, %arg4] : memref<1x8xf32, strided<[768, 1], offset: ?>>
            }
          }
        }
        memref.copy %subview_7, %subview_7 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
      }
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x1024x768xf32>
    scf.for %arg0 = %c0 to %c1024 step %c8 {
      scf.for %arg1 = %c0 to %c768 step %c8 {
        %subview = memref.subview %alloc_3[0, %arg1] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        %subview_7 = memref.subview %alloc_4[0, %arg0, %arg1] [1, 8, 8] [1, 1, 1] : memref<1x1024x768xf32> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
        scf.for %arg2 = %c0 to %c1 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            scf.for %arg4 = %c0 to %c8 step %c1 {
              %1 = memref.load %subview[%arg2, %arg4] : memref<1x8xf32, strided<[768, 1], offset: ?>>
              %2 = arith.subf %cst_0, %1 : f32
              %3 = math.exp %2 : f32
              memref.store %3, %subview_7[%arg2, %arg3, %arg4] : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
            }
          }
        }
        memref.copy %subview_7, %subview_7 : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
      }
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
    scf.for %arg0 = %c0 to %c768 step %c8 {
      %subview = memref.subview %alloc_5[0, %arg0] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          memref.store %cst, %subview[%arg1, %arg2] : memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
      }
      memref.copy %subview, %subview : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
    }
    scf.for %arg0 = %c0 to %c1024 step %c8 {
      scf.for %arg1 = %c0 to %c768 step %c8 {
        %subview = memref.subview %alloc_4[0, %arg0, %arg1] [1, 8, 8] [1, 1, 1] : memref<1x1024x768xf32> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
        %subview_7 = memref.subview %alloc_5[0, %arg1] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        scf.for %arg2 = %c0 to %c1 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            scf.for %arg4 = %c0 to %c8 step %c1 {
              %1 = memref.load %subview[%arg2, %arg3, %arg4] : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
              %2 = memref.load %subview_7[%arg2, %arg4] : memref<1x8xf32, strided<[768, 1], offset: ?>>
              %3 = arith.addf %1, %2 : f32
              memref.store %3, %subview_7[%arg2, %arg4] : memref<1x8xf32, strided<[768, 1], offset: ?>>
            }
          }
        }
        memref.copy %subview_7, %subview_7 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
      }
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x1024x768xf32>
    scf.for %arg0 = %c0 to %c1024 step %c8 {
      scf.for %arg1 = %c0 to %c768 step %c8 {
        %subview = memref.subview %alloc_4[0, %arg0, %arg1] [1, 8, 8] [1, 1, 1] : memref<1x1024x768xf32> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
        %subview_7 = memref.subview %alloc_5[0, %arg1] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        %subview_8 = memref.subview %alloc_6[0, %arg0, %arg1] [1, 8, 8] [1, 1, 1] : memref<1x1024x768xf32> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
        scf.for %arg2 = %c0 to %c1 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            scf.for %arg4 = %c0 to %c8 step %c1 {
              %1 = memref.load %subview[%arg2, %arg3, %arg4] : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
              %2 = memref.load %subview_7[%arg2, %arg4] : memref<1x8xf32, strided<[768, 1], offset: ?>>
              %3 = arith.divf %1, %2 : f32
              memref.store %3, %subview_8[%arg2, %arg3, %arg4] : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
            }
          }
        }
        memref.copy %subview_8, %subview_8 : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
      }
    }
    return %alloc_2, %alloc_6 : memref<1x768xi64>, memref<1x1024x768xf32>
  }
}

// ==========================================
// Phase: lower to llvm
// ==========================================
#map = affine_map<(d0, d1) -> (d0 + d1)>
module {
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_1x1024x768xf32(dense<3.000000e-01> : tensor<1x1024x768xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x array<1024 x array<768 x f32>>>
  llvm.func @host() -> !llvm.struct<(struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)> {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(1024 : index) : i64
    %2 = llvm.mlir.constant(768 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(8 : index) : i64
    %5 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %6 = llvm.mlir.constant(3.000000e-01 : f32) : f32
    %7 = llvm.mlir.constant(0 : i64) : i64
    %8 = llvm.mlir.constant(0xFF800000 : f32) : f32
    %9 = llvm.mlir.constant(1 : index) : i64
    %10 = llvm.mlir.constant(1024 : index) : i64
    %11 = llvm.mlir.constant(768 : index) : i64
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.mlir.constant(786432 : index) : i64
    %14 = llvm.mlir.constant(786432 : index) : i64
    %15 = llvm.mlir.zero : !llvm.ptr
    %16 = llvm.getelementptr %15[%14] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %17 = llvm.ptrtoint %16 : !llvm.ptr to i64
    %18 = llvm.mlir.addressof @__constant_1x1024x768xf32 : !llvm.ptr
    %19 = llvm.getelementptr %18[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<1024 x array<768 x f32>>>
    %20 = llvm.mlir.constant(3735928559 : index) : i64
    %21 = llvm.inttoptr %20 : i64 to !llvm.ptr
    %22 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %23 = llvm.insertvalue %21, %22[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %24 = llvm.insertvalue %19, %23[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %25 = llvm.mlir.constant(0 : index) : i64
    %26 = llvm.insertvalue %25, %24[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %27 = llvm.insertvalue %9, %26[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %28 = llvm.insertvalue %10, %27[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %29 = llvm.insertvalue %11, %28[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %30 = llvm.insertvalue %13, %29[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %31 = llvm.insertvalue %11, %30[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %32 = llvm.insertvalue %12, %31[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %33 = builtin.unrealized_conversion_cast %32 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<1x1024x768xf32>
    %34 = llvm.mlir.constant(1 : index) : i64
    %35 = llvm.mlir.constant(768 : index) : i64
    %36 = llvm.mlir.constant(1 : index) : i64
    %37 = llvm.mlir.constant(768 : index) : i64
    %38 = llvm.mlir.zero : !llvm.ptr
    %39 = llvm.getelementptr %38[%37] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.mlir.constant(64 : index) : i64
    %42 = llvm.add %40, %41 : i64
    %43 = llvm.call @malloc(%42) : (i64) -> !llvm.ptr
    %44 = llvm.ptrtoint %43 : !llvm.ptr to i64
    %45 = llvm.mlir.constant(1 : index) : i64
    %46 = llvm.sub %41, %45 : i64
    %47 = llvm.add %44, %46 : i64
    %48 = llvm.urem %47, %41  : i64
    %49 = llvm.sub %47, %48 : i64
    %50 = llvm.inttoptr %49 : i64 to !llvm.ptr
    %51 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %52 = llvm.insertvalue %43, %51[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %53 = llvm.insertvalue %50, %52[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.mlir.constant(0 : index) : i64
    %55 = llvm.insertvalue %54, %53[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.insertvalue %34, %55[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.insertvalue %35, %56[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.insertvalue %35, %57[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.insertvalue %36, %58[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = builtin.unrealized_conversion_cast %59 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<1x768xf32>
    llvm.br ^bb1(%3 : i64)
  ^bb1(%61: i64):  // 2 preds: ^bb0, ^bb8
    %62 = builtin.unrealized_conversion_cast %61 : i64 to index
    %63 = llvm.icmp "slt" %61, %2 : i64
    llvm.cond_br %63, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %subview = memref.subview %60[0, %62] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
    %64 = builtin.unrealized_conversion_cast %subview : memref<1x8xf32, strided<[768, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb3(%3 : i64)
  ^bb3(%65: i64):  // 2 preds: ^bb2, ^bb7
    %66 = llvm.icmp "slt" %65, %0 : i64
    llvm.cond_br %66, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%3 : i64)
  ^bb5(%67: i64):  // 2 preds: ^bb4, ^bb6
    %68 = llvm.icmp "slt" %67, %4 : i64
    llvm.cond_br %68, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %69 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.extractvalue %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.getelementptr %69[%70] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %72 = llvm.mlir.constant(768 : index) : i64
    %73 = llvm.mul %65, %72 : i64
    %74 = llvm.add %73, %67 : i64
    %75 = llvm.getelementptr %71[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %8, %75 : f32, !llvm.ptr
    %76 = llvm.add %67, %0 : i64
    llvm.br ^bb5(%76 : i64)
  ^bb7:  // pred: ^bb5
    %77 = llvm.add %65, %0 : i64
    llvm.br ^bb3(%77 : i64)
  ^bb8:  // pred: ^bb3
    %78 = llvm.mlir.constant(1 : index) : i64
    %79 = llvm.extractvalue %64[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %80 = llvm.mul %78, %79 : i64
    %81 = llvm.extractvalue %64[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %82 = llvm.mul %80, %81 : i64
    %83 = llvm.mlir.zero : !llvm.ptr
    %84 = llvm.getelementptr %83[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %85 = llvm.ptrtoint %84 : !llvm.ptr to i64
    %86 = llvm.mul %82, %85 : i64
    %87 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %88 = llvm.extractvalue %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %89 = llvm.getelementptr %87[%88] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %90 = llvm.extractvalue %64[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.extractvalue %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.getelementptr %90[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%92, %89, %86) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %93 = llvm.add %61, %4 : i64
    llvm.br ^bb1(%93 : i64)
  ^bb9:  // pred: ^bb1
    %94 = llvm.mlir.constant(1 : index) : i64
    %95 = llvm.mlir.constant(768 : index) : i64
    %96 = llvm.mlir.constant(1 : index) : i64
    %97 = llvm.mlir.constant(768 : index) : i64
    %98 = llvm.mlir.zero : !llvm.ptr
    %99 = llvm.getelementptr %98[%97] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %100 = llvm.ptrtoint %99 : !llvm.ptr to i64
    %101 = llvm.mlir.constant(64 : index) : i64
    %102 = llvm.add %100, %101 : i64
    %103 = llvm.call @malloc(%102) : (i64) -> !llvm.ptr
    %104 = llvm.ptrtoint %103 : !llvm.ptr to i64
    %105 = llvm.mlir.constant(1 : index) : i64
    %106 = llvm.sub %101, %105 : i64
    %107 = llvm.add %104, %106 : i64
    %108 = llvm.urem %107, %101  : i64
    %109 = llvm.sub %107, %108 : i64
    %110 = llvm.inttoptr %109 : i64 to !llvm.ptr
    %111 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %112 = llvm.insertvalue %103, %111[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %113 = llvm.insertvalue %110, %112[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %114 = llvm.mlir.constant(0 : index) : i64
    %115 = llvm.insertvalue %114, %113[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %116 = llvm.insertvalue %94, %115[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %117 = llvm.insertvalue %95, %116[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %118 = llvm.insertvalue %95, %117[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %119 = llvm.insertvalue %96, %118[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %120 = builtin.unrealized_conversion_cast %119 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<1x768xi64>
    llvm.br ^bb10(%3 : i64)
  ^bb10(%121: i64):  // 2 preds: ^bb9, ^bb17
    %122 = builtin.unrealized_conversion_cast %121 : i64 to index
    %123 = llvm.icmp "slt" %121, %2 : i64
    llvm.cond_br %123, ^bb11, ^bb18
  ^bb11:  // pred: ^bb10
    %subview_0 = memref.subview %120[0, %122] [1, 8] [1, 1] : memref<1x768xi64> to memref<1x8xi64, strided<[768, 1], offset: ?>>
    %124 = builtin.unrealized_conversion_cast %subview_0 : memref<1x8xi64, strided<[768, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb12(%3 : i64)
  ^bb12(%125: i64):  // 2 preds: ^bb11, ^bb16
    %126 = llvm.icmp "slt" %125, %0 : i64
    llvm.cond_br %126, ^bb13, ^bb17
  ^bb13:  // pred: ^bb12
    llvm.br ^bb14(%3 : i64)
  ^bb14(%127: i64):  // 2 preds: ^bb13, ^bb15
    %128 = llvm.icmp "slt" %127, %4 : i64
    llvm.cond_br %128, ^bb15, ^bb16
  ^bb15:  // pred: ^bb14
    %129 = llvm.extractvalue %124[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %130 = llvm.extractvalue %124[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.getelementptr %129[%130] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %132 = llvm.mlir.constant(768 : index) : i64
    %133 = llvm.mul %125, %132 : i64
    %134 = llvm.add %133, %127 : i64
    %135 = llvm.getelementptr %131[%134] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %7, %135 : i64, !llvm.ptr
    %136 = llvm.add %127, %0 : i64
    llvm.br ^bb14(%136 : i64)
  ^bb16:  // pred: ^bb14
    %137 = llvm.add %125, %0 : i64
    llvm.br ^bb12(%137 : i64)
  ^bb17:  // pred: ^bb12
    %138 = llvm.mlir.constant(1 : index) : i64
    %139 = llvm.extractvalue %124[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %140 = llvm.mul %138, %139 : i64
    %141 = llvm.extractvalue %124[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %142 = llvm.mul %140, %141 : i64
    %143 = llvm.mlir.zero : !llvm.ptr
    %144 = llvm.getelementptr %143[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %145 = llvm.ptrtoint %144 : !llvm.ptr to i64
    %146 = llvm.mul %142, %145 : i64
    %147 = llvm.extractvalue %124[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %148 = llvm.extractvalue %124[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %149 = llvm.getelementptr %147[%148] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %150 = llvm.extractvalue %124[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %151 = llvm.extractvalue %124[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %152 = llvm.getelementptr %150[%151] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    "llvm.intr.memcpy"(%152, %149, %146) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %153 = llvm.add %121, %4 : i64
    llvm.br ^bb10(%153 : i64)
  ^bb18:  // pred: ^bb10
    llvm.br ^bb19(%3 : i64)
  ^bb19(%154: i64):  // 2 preds: ^bb18, ^bb32
    %155 = builtin.unrealized_conversion_cast %154 : i64 to index
    %156 = llvm.icmp "slt" %154, %1 : i64
    llvm.cond_br %156, ^bb20, ^bb33
  ^bb20:  // pred: ^bb19
    llvm.br ^bb21(%3 : i64)
  ^bb21(%157: i64):  // 2 preds: ^bb20, ^bb31
    %158 = builtin.unrealized_conversion_cast %157 : i64 to index
    %159 = llvm.icmp "slt" %157, %2 : i64
    llvm.cond_br %159, ^bb22, ^bb32
  ^bb22:  // pred: ^bb21
    %subview_1 = memref.subview %33[0, %155, %158] [1, 8, 8] [1, 1, 1] : memref<1x1024x768xf32> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
    %160 = builtin.unrealized_conversion_cast %subview_1 : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %subview_2 = memref.subview %60[0, %158] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
    %161 = builtin.unrealized_conversion_cast %subview_2 : memref<1x8xf32, strided<[768, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %subview_3 = memref.subview %120[0, %158] [1, 8] [1, 1] : memref<1x768xi64> to memref<1x8xi64, strided<[768, 1], offset: ?>>
    %162 = builtin.unrealized_conversion_cast %subview_3 : memref<1x8xi64, strided<[768, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb23(%3 : i64)
  ^bb23(%163: i64):  // 2 preds: ^bb22, ^bb30
    %164 = llvm.icmp "slt" %163, %0 : i64
    llvm.cond_br %164, ^bb24, ^bb31
  ^bb24:  // pred: ^bb23
    llvm.br ^bb25(%3 : i64)
  ^bb25(%165: i64):  // 2 preds: ^bb24, ^bb29
    %166 = builtin.unrealized_conversion_cast %165 : i64 to index
    %167 = llvm.icmp "slt" %165, %4 : i64
    llvm.cond_br %167, ^bb26, ^bb30
  ^bb26:  // pred: ^bb25
    llvm.br ^bb27(%3 : i64)
  ^bb27(%168: i64):  // 2 preds: ^bb26, ^bb28
    %169 = llvm.icmp "slt" %168, %4 : i64
    llvm.cond_br %169, ^bb28, ^bb29
  ^bb28:  // pred: ^bb27
    %170 = llvm.extractvalue %160[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %171 = llvm.extractvalue %160[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %172 = llvm.getelementptr %170[%171] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %173 = llvm.mlir.constant(786432 : index) : i64
    %174 = llvm.mul %163, %173 : i64
    %175 = llvm.mlir.constant(768 : index) : i64
    %176 = llvm.mul %165, %175 : i64
    %177 = llvm.add %174, %176 : i64
    %178 = llvm.add %177, %168 : i64
    %179 = llvm.getelementptr %172[%178] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %180 = llvm.load %179 : !llvm.ptr -> f32
    %181 = llvm.extractvalue %161[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %182 = llvm.extractvalue %161[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %183 = llvm.getelementptr %181[%182] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %184 = llvm.mlir.constant(768 : index) : i64
    %185 = llvm.mul %163, %184 : i64
    %186 = llvm.add %185, %168 : i64
    %187 = llvm.getelementptr %183[%186] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %188 = llvm.load %187 : !llvm.ptr -> f32
    %189 = llvm.extractvalue %162[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %190 = llvm.extractvalue %162[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %191 = llvm.getelementptr %189[%190] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %192 = llvm.mlir.constant(768 : index) : i64
    %193 = llvm.mul %163, %192 : i64
    %194 = llvm.add %193, %168 : i64
    %195 = llvm.getelementptr %191[%194] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %196 = llvm.load %195 : !llvm.ptr -> i64
    %197 = affine.apply #map(%166, %155)
    %198 = builtin.unrealized_conversion_cast %197 : index to i64
    %199 = llvm.fcmp "ogt" %180, %188 : f32
    %200 = llvm.select %199, %180, %188 : i1, f32
    %201 = llvm.select %199, %198, %196 : i1, i64
    %202 = llvm.extractvalue %161[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %203 = llvm.extractvalue %161[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %204 = llvm.getelementptr %202[%203] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %205 = llvm.mlir.constant(768 : index) : i64
    %206 = llvm.mul %163, %205 : i64
    %207 = llvm.add %206, %168 : i64
    %208 = llvm.getelementptr %204[%207] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %200, %208 : f32, !llvm.ptr
    %209 = llvm.extractvalue %162[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %210 = llvm.extractvalue %162[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %211 = llvm.getelementptr %209[%210] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %212 = llvm.mlir.constant(768 : index) : i64
    %213 = llvm.mul %163, %212 : i64
    %214 = llvm.add %213, %168 : i64
    %215 = llvm.getelementptr %211[%214] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %201, %215 : i64, !llvm.ptr
    %216 = llvm.add %168, %0 : i64
    llvm.br ^bb27(%216 : i64)
  ^bb29:  // pred: ^bb27
    %217 = llvm.add %165, %0 : i64
    llvm.br ^bb25(%217 : i64)
  ^bb30:  // pred: ^bb25
    %218 = llvm.add %163, %0 : i64
    llvm.br ^bb23(%218 : i64)
  ^bb31:  // pred: ^bb23
    %219 = llvm.mlir.constant(1 : index) : i64
    %220 = llvm.extractvalue %161[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %221 = llvm.mul %219, %220 : i64
    %222 = llvm.extractvalue %161[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %223 = llvm.mul %221, %222 : i64
    %224 = llvm.mlir.zero : !llvm.ptr
    %225 = llvm.getelementptr %224[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %226 = llvm.ptrtoint %225 : !llvm.ptr to i64
    %227 = llvm.mul %223, %226 : i64
    %228 = llvm.extractvalue %161[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %229 = llvm.extractvalue %161[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %230 = llvm.getelementptr %228[%229] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %231 = llvm.extractvalue %161[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %232 = llvm.extractvalue %161[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %233 = llvm.getelementptr %231[%232] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%233, %230, %227) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %234 = llvm.mlir.constant(1 : index) : i64
    %235 = llvm.extractvalue %162[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %236 = llvm.mul %234, %235 : i64
    %237 = llvm.extractvalue %162[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %238 = llvm.mul %236, %237 : i64
    %239 = llvm.mlir.zero : !llvm.ptr
    %240 = llvm.getelementptr %239[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %241 = llvm.ptrtoint %240 : !llvm.ptr to i64
    %242 = llvm.mul %238, %241 : i64
    %243 = llvm.extractvalue %162[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %244 = llvm.extractvalue %162[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %245 = llvm.getelementptr %243[%244] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %246 = llvm.extractvalue %162[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %247 = llvm.extractvalue %162[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %248 = llvm.getelementptr %246[%247] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    "llvm.intr.memcpy"(%248, %245, %242) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %249 = llvm.add %157, %4 : i64
    llvm.br ^bb21(%249 : i64)
  ^bb32:  // pred: ^bb21
    %250 = llvm.add %154, %4 : i64
    llvm.br ^bb19(%250 : i64)
  ^bb33:  // pred: ^bb19
    %251 = llvm.mlir.constant(1 : index) : i64
    %252 = llvm.mlir.constant(768 : index) : i64
    %253 = llvm.mlir.constant(1 : index) : i64
    %254 = llvm.mlir.constant(768 : index) : i64
    %255 = llvm.mlir.zero : !llvm.ptr
    %256 = llvm.getelementptr %255[%254] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %257 = llvm.ptrtoint %256 : !llvm.ptr to i64
    %258 = llvm.mlir.constant(64 : index) : i64
    %259 = llvm.add %257, %258 : i64
    %260 = llvm.call @malloc(%259) : (i64) -> !llvm.ptr
    %261 = llvm.ptrtoint %260 : !llvm.ptr to i64
    %262 = llvm.mlir.constant(1 : index) : i64
    %263 = llvm.sub %258, %262 : i64
    %264 = llvm.add %261, %263 : i64
    %265 = llvm.urem %264, %258  : i64
    %266 = llvm.sub %264, %265 : i64
    %267 = llvm.inttoptr %266 : i64 to !llvm.ptr
    %268 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %269 = llvm.insertvalue %260, %268[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %270 = llvm.insertvalue %267, %269[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %271 = llvm.mlir.constant(0 : index) : i64
    %272 = llvm.insertvalue %271, %270[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %273 = llvm.insertvalue %251, %272[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %274 = llvm.insertvalue %252, %273[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %275 = llvm.insertvalue %252, %274[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %276 = llvm.insertvalue %253, %275[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %277 = builtin.unrealized_conversion_cast %276 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<1x768xf32>
    llvm.br ^bb34(%3 : i64)
  ^bb34(%278: i64):  // 2 preds: ^bb33, ^bb41
    %279 = builtin.unrealized_conversion_cast %278 : i64 to index
    %280 = llvm.icmp "slt" %278, %2 : i64
    llvm.cond_br %280, ^bb35, ^bb42
  ^bb35:  // pred: ^bb34
    %subview_4 = memref.subview %277[0, %279] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
    %281 = builtin.unrealized_conversion_cast %subview_4 : memref<1x8xf32, strided<[768, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb36(%3 : i64)
  ^bb36(%282: i64):  // 2 preds: ^bb35, ^bb40
    %283 = llvm.icmp "slt" %282, %0 : i64
    llvm.cond_br %283, ^bb37, ^bb41
  ^bb37:  // pred: ^bb36
    llvm.br ^bb38(%3 : i64)
  ^bb38(%284: i64):  // 2 preds: ^bb37, ^bb39
    %285 = llvm.icmp "slt" %284, %4 : i64
    llvm.cond_br %285, ^bb39, ^bb40
  ^bb39:  // pred: ^bb38
    %286 = llvm.extractvalue %281[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %287 = llvm.extractvalue %281[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %288 = llvm.getelementptr %286[%287] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %289 = llvm.mlir.constant(768 : index) : i64
    %290 = llvm.mul %282, %289 : i64
    %291 = llvm.add %290, %284 : i64
    %292 = llvm.getelementptr %288[%291] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %8, %292 : f32, !llvm.ptr
    %293 = llvm.add %284, %0 : i64
    llvm.br ^bb38(%293 : i64)
  ^bb40:  // pred: ^bb38
    %294 = llvm.add %282, %0 : i64
    llvm.br ^bb36(%294 : i64)
  ^bb41:  // pred: ^bb36
    %295 = llvm.mlir.constant(1 : index) : i64
    %296 = llvm.extractvalue %281[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %297 = llvm.mul %295, %296 : i64
    %298 = llvm.extractvalue %281[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %299 = llvm.mul %297, %298 : i64
    %300 = llvm.mlir.zero : !llvm.ptr
    %301 = llvm.getelementptr %300[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %302 = llvm.ptrtoint %301 : !llvm.ptr to i64
    %303 = llvm.mul %299, %302 : i64
    %304 = llvm.extractvalue %281[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %305 = llvm.extractvalue %281[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %306 = llvm.getelementptr %304[%305] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %307 = llvm.extractvalue %281[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %308 = llvm.extractvalue %281[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %309 = llvm.getelementptr %307[%308] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%309, %306, %303) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %310 = llvm.add %278, %4 : i64
    llvm.br ^bb34(%310 : i64)
  ^bb42:  // pred: ^bb34
    llvm.br ^bb43(%3 : i64)
  ^bb43(%311: i64):  // 2 preds: ^bb42, ^bb56
    %312 = builtin.unrealized_conversion_cast %311 : i64 to index
    %313 = llvm.icmp "slt" %311, %1 : i64
    llvm.cond_br %313, ^bb44, ^bb57
  ^bb44:  // pred: ^bb43
    llvm.br ^bb45(%3 : i64)
  ^bb45(%314: i64):  // 2 preds: ^bb44, ^bb55
    %315 = builtin.unrealized_conversion_cast %314 : i64 to index
    %316 = llvm.icmp "slt" %314, %2 : i64
    llvm.cond_br %316, ^bb46, ^bb56
  ^bb46:  // pred: ^bb45
    %subview_5 = memref.subview %33[0, %312, %315] [1, 8, 8] [1, 1, 1] : memref<1x1024x768xf32> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
    %317 = builtin.unrealized_conversion_cast %subview_5 : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %subview_6 = memref.subview %277[0, %315] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
    %318 = builtin.unrealized_conversion_cast %subview_6 : memref<1x8xf32, strided<[768, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb47(%3 : i64)
  ^bb47(%319: i64):  // 2 preds: ^bb46, ^bb54
    %320 = llvm.icmp "slt" %319, %0 : i64
    llvm.cond_br %320, ^bb48, ^bb55
  ^bb48:  // pred: ^bb47
    llvm.br ^bb49(%3 : i64)
  ^bb49(%321: i64):  // 2 preds: ^bb48, ^bb53
    %322 = llvm.icmp "slt" %321, %4 : i64
    llvm.cond_br %322, ^bb50, ^bb54
  ^bb50:  // pred: ^bb49
    llvm.br ^bb51(%3 : i64)
  ^bb51(%323: i64):  // 2 preds: ^bb50, ^bb52
    %324 = llvm.icmp "slt" %323, %4 : i64
    llvm.cond_br %324, ^bb52, ^bb53
  ^bb52:  // pred: ^bb51
    %325 = llvm.extractvalue %317[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %326 = llvm.extractvalue %317[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %327 = llvm.getelementptr %325[%326] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %328 = llvm.mlir.constant(786432 : index) : i64
    %329 = llvm.mul %319, %328 : i64
    %330 = llvm.mlir.constant(768 : index) : i64
    %331 = llvm.mul %321, %330 : i64
    %332 = llvm.add %329, %331 : i64
    %333 = llvm.add %332, %323 : i64
    %334 = llvm.getelementptr %327[%333] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %335 = llvm.load %334 : !llvm.ptr -> f32
    %336 = llvm.extractvalue %318[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %337 = llvm.extractvalue %318[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %338 = llvm.getelementptr %336[%337] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %339 = llvm.mlir.constant(768 : index) : i64
    %340 = llvm.mul %319, %339 : i64
    %341 = llvm.add %340, %323 : i64
    %342 = llvm.getelementptr %338[%341] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %343 = llvm.load %342 : !llvm.ptr -> f32
    %344 = llvm.intr.maxnum(%335, %343)  : (f32, f32) -> f32
    %345 = llvm.extractvalue %318[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %346 = llvm.extractvalue %318[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %347 = llvm.getelementptr %345[%346] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %348 = llvm.mlir.constant(768 : index) : i64
    %349 = llvm.mul %319, %348 : i64
    %350 = llvm.add %349, %323 : i64
    %351 = llvm.getelementptr %347[%350] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %344, %351 : f32, !llvm.ptr
    %352 = llvm.add %323, %0 : i64
    llvm.br ^bb51(%352 : i64)
  ^bb53:  // pred: ^bb51
    %353 = llvm.add %321, %0 : i64
    llvm.br ^bb49(%353 : i64)
  ^bb54:  // pred: ^bb49
    %354 = llvm.add %319, %0 : i64
    llvm.br ^bb47(%354 : i64)
  ^bb55:  // pred: ^bb47
    %355 = llvm.mlir.constant(1 : index) : i64
    %356 = llvm.extractvalue %318[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %357 = llvm.mul %355, %356 : i64
    %358 = llvm.extractvalue %318[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %359 = llvm.mul %357, %358 : i64
    %360 = llvm.mlir.zero : !llvm.ptr
    %361 = llvm.getelementptr %360[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %362 = llvm.ptrtoint %361 : !llvm.ptr to i64
    %363 = llvm.mul %359, %362 : i64
    %364 = llvm.extractvalue %318[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %365 = llvm.extractvalue %318[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %366 = llvm.getelementptr %364[%365] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %367 = llvm.extractvalue %318[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %368 = llvm.extractvalue %318[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %369 = llvm.getelementptr %367[%368] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%369, %366, %363) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %370 = llvm.add %314, %4 : i64
    llvm.br ^bb45(%370 : i64)
  ^bb56:  // pred: ^bb45
    %371 = llvm.add %311, %4 : i64
    llvm.br ^bb43(%371 : i64)
  ^bb57:  // pred: ^bb43
    %372 = llvm.mlir.constant(1 : index) : i64
    %373 = llvm.mlir.constant(1024 : index) : i64
    %374 = llvm.mlir.constant(768 : index) : i64
    %375 = llvm.mlir.constant(1 : index) : i64
    %376 = llvm.mlir.constant(786432 : index) : i64
    %377 = llvm.mlir.constant(786432 : index) : i64
    %378 = llvm.mlir.zero : !llvm.ptr
    %379 = llvm.getelementptr %378[%377] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %380 = llvm.ptrtoint %379 : !llvm.ptr to i64
    %381 = llvm.mlir.constant(64 : index) : i64
    %382 = llvm.add %380, %381 : i64
    %383 = llvm.call @malloc(%382) : (i64) -> !llvm.ptr
    %384 = llvm.ptrtoint %383 : !llvm.ptr to i64
    %385 = llvm.mlir.constant(1 : index) : i64
    %386 = llvm.sub %381, %385 : i64
    %387 = llvm.add %384, %386 : i64
    %388 = llvm.urem %387, %381  : i64
    %389 = llvm.sub %387, %388 : i64
    %390 = llvm.inttoptr %389 : i64 to !llvm.ptr
    %391 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %392 = llvm.insertvalue %383, %391[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %393 = llvm.insertvalue %390, %392[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %394 = llvm.mlir.constant(0 : index) : i64
    %395 = llvm.insertvalue %394, %393[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %396 = llvm.insertvalue %372, %395[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %397 = llvm.insertvalue %373, %396[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %398 = llvm.insertvalue %374, %397[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %399 = llvm.insertvalue %376, %398[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %400 = llvm.insertvalue %374, %399[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %401 = llvm.insertvalue %375, %400[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %402 = builtin.unrealized_conversion_cast %401 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<1x1024x768xf32>
    llvm.br ^bb58(%3 : i64)
  ^bb58(%403: i64):  // 2 preds: ^bb57, ^bb71
    %404 = builtin.unrealized_conversion_cast %403 : i64 to index
    %405 = llvm.icmp "slt" %403, %1 : i64
    llvm.cond_br %405, ^bb59, ^bb72
  ^bb59:  // pred: ^bb58
    llvm.br ^bb60(%3 : i64)
  ^bb60(%406: i64):  // 2 preds: ^bb59, ^bb70
    %407 = builtin.unrealized_conversion_cast %406 : i64 to index
    %408 = llvm.icmp "slt" %406, %2 : i64
    llvm.cond_br %408, ^bb61, ^bb71
  ^bb61:  // pred: ^bb60
    %subview_7 = memref.subview %277[0, %407] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
    %409 = builtin.unrealized_conversion_cast %subview_7 : memref<1x8xf32, strided<[768, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %subview_8 = memref.subview %402[0, %404, %407] [1, 8, 8] [1, 1, 1] : memref<1x1024x768xf32> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
    %410 = builtin.unrealized_conversion_cast %subview_8 : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    llvm.br ^bb62(%3 : i64)
  ^bb62(%411: i64):  // 2 preds: ^bb61, ^bb69
    %412 = llvm.icmp "slt" %411, %0 : i64
    llvm.cond_br %412, ^bb63, ^bb70
  ^bb63:  // pred: ^bb62
    llvm.br ^bb64(%3 : i64)
  ^bb64(%413: i64):  // 2 preds: ^bb63, ^bb68
    %414 = llvm.icmp "slt" %413, %4 : i64
    llvm.cond_br %414, ^bb65, ^bb69
  ^bb65:  // pred: ^bb64
    llvm.br ^bb66(%3 : i64)
  ^bb66(%415: i64):  // 2 preds: ^bb65, ^bb67
    %416 = llvm.icmp "slt" %415, %4 : i64
    llvm.cond_br %416, ^bb67, ^bb68
  ^bb67:  // pred: ^bb66
    %417 = llvm.extractvalue %409[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %418 = llvm.extractvalue %409[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %419 = llvm.getelementptr %417[%418] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %420 = llvm.mlir.constant(768 : index) : i64
    %421 = llvm.mul %411, %420 : i64
    %422 = llvm.add %421, %415 : i64
    %423 = llvm.getelementptr %419[%422] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %424 = llvm.load %423 : !llvm.ptr -> f32
    %425 = llvm.fsub %6, %424  : f32
    %426 = llvm.intr.exp(%425)  : (f32) -> f32
    %427 = llvm.extractvalue %410[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %428 = llvm.extractvalue %410[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %429 = llvm.getelementptr %427[%428] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %430 = llvm.mlir.constant(786432 : index) : i64
    %431 = llvm.mul %411, %430 : i64
    %432 = llvm.mlir.constant(768 : index) : i64
    %433 = llvm.mul %413, %432 : i64
    %434 = llvm.add %431, %433 : i64
    %435 = llvm.add %434, %415 : i64
    %436 = llvm.getelementptr %429[%435] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %426, %436 : f32, !llvm.ptr
    %437 = llvm.add %415, %0 : i64
    llvm.br ^bb66(%437 : i64)
  ^bb68:  // pred: ^bb66
    %438 = llvm.add %413, %0 : i64
    llvm.br ^bb64(%438 : i64)
  ^bb69:  // pred: ^bb64
    %439 = llvm.add %411, %0 : i64
    llvm.br ^bb62(%439 : i64)
  ^bb70:  // pred: ^bb62
    %440 = llvm.intr.stacksave : !llvm.ptr
    %441 = llvm.mlir.constant(3 : i64) : i64
    %442 = llvm.mlir.constant(1 : index) : i64
    %443 = llvm.alloca %442 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %410, %443 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %444 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %445 = llvm.insertvalue %441, %444[0] : !llvm.struct<(i64, ptr)> 
    %446 = llvm.insertvalue %443, %445[1] : !llvm.struct<(i64, ptr)> 
    %447 = llvm.mlir.constant(3 : i64) : i64
    %448 = llvm.mlir.constant(1 : index) : i64
    %449 = llvm.alloca %448 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %410, %449 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %450 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %451 = llvm.insertvalue %447, %450[0] : !llvm.struct<(i64, ptr)> 
    %452 = llvm.insertvalue %449, %451[1] : !llvm.struct<(i64, ptr)> 
    %453 = llvm.mlir.constant(1 : index) : i64
    %454 = llvm.alloca %453 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %446, %454 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %455 = llvm.alloca %453 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %452, %455 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %456 = llvm.mlir.zero : !llvm.ptr
    %457 = llvm.getelementptr %456[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %458 = llvm.ptrtoint %457 : !llvm.ptr to i64
    llvm.call @memrefCopy(%458, %454, %455) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %440 : !llvm.ptr
    %459 = llvm.add %406, %4 : i64
    llvm.br ^bb60(%459 : i64)
  ^bb71:  // pred: ^bb60
    %460 = llvm.add %403, %4 : i64
    llvm.br ^bb58(%460 : i64)
  ^bb72:  // pred: ^bb58
    %461 = llvm.mlir.constant(1 : index) : i64
    %462 = llvm.mlir.constant(768 : index) : i64
    %463 = llvm.mlir.constant(1 : index) : i64
    %464 = llvm.mlir.constant(768 : index) : i64
    %465 = llvm.mlir.zero : !llvm.ptr
    %466 = llvm.getelementptr %465[%464] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %467 = llvm.ptrtoint %466 : !llvm.ptr to i64
    %468 = llvm.mlir.constant(64 : index) : i64
    %469 = llvm.add %467, %468 : i64
    %470 = llvm.call @malloc(%469) : (i64) -> !llvm.ptr
    %471 = llvm.ptrtoint %470 : !llvm.ptr to i64
    %472 = llvm.mlir.constant(1 : index) : i64
    %473 = llvm.sub %468, %472 : i64
    %474 = llvm.add %471, %473 : i64
    %475 = llvm.urem %474, %468  : i64
    %476 = llvm.sub %474, %475 : i64
    %477 = llvm.inttoptr %476 : i64 to !llvm.ptr
    %478 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %479 = llvm.insertvalue %470, %478[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %480 = llvm.insertvalue %477, %479[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %481 = llvm.mlir.constant(0 : index) : i64
    %482 = llvm.insertvalue %481, %480[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %483 = llvm.insertvalue %461, %482[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %484 = llvm.insertvalue %462, %483[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %485 = llvm.insertvalue %462, %484[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %486 = llvm.insertvalue %463, %485[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %487 = builtin.unrealized_conversion_cast %486 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<1x768xf32>
    llvm.br ^bb73(%3 : i64)
  ^bb73(%488: i64):  // 2 preds: ^bb72, ^bb80
    %489 = builtin.unrealized_conversion_cast %488 : i64 to index
    %490 = llvm.icmp "slt" %488, %2 : i64
    llvm.cond_br %490, ^bb74, ^bb81
  ^bb74:  // pred: ^bb73
    %subview_9 = memref.subview %487[0, %489] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
    %491 = builtin.unrealized_conversion_cast %subview_9 : memref<1x8xf32, strided<[768, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb75(%3 : i64)
  ^bb75(%492: i64):  // 2 preds: ^bb74, ^bb79
    %493 = llvm.icmp "slt" %492, %0 : i64
    llvm.cond_br %493, ^bb76, ^bb80
  ^bb76:  // pred: ^bb75
    llvm.br ^bb77(%3 : i64)
  ^bb77(%494: i64):  // 2 preds: ^bb76, ^bb78
    %495 = llvm.icmp "slt" %494, %4 : i64
    llvm.cond_br %495, ^bb78, ^bb79
  ^bb78:  // pred: ^bb77
    %496 = llvm.extractvalue %491[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %497 = llvm.extractvalue %491[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %498 = llvm.getelementptr %496[%497] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %499 = llvm.mlir.constant(768 : index) : i64
    %500 = llvm.mul %492, %499 : i64
    %501 = llvm.add %500, %494 : i64
    %502 = llvm.getelementptr %498[%501] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %5, %502 : f32, !llvm.ptr
    %503 = llvm.add %494, %0 : i64
    llvm.br ^bb77(%503 : i64)
  ^bb79:  // pred: ^bb77
    %504 = llvm.add %492, %0 : i64
    llvm.br ^bb75(%504 : i64)
  ^bb80:  // pred: ^bb75
    %505 = llvm.mlir.constant(1 : index) : i64
    %506 = llvm.extractvalue %491[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %507 = llvm.mul %505, %506 : i64
    %508 = llvm.extractvalue %491[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %509 = llvm.mul %507, %508 : i64
    %510 = llvm.mlir.zero : !llvm.ptr
    %511 = llvm.getelementptr %510[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %512 = llvm.ptrtoint %511 : !llvm.ptr to i64
    %513 = llvm.mul %509, %512 : i64
    %514 = llvm.extractvalue %491[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %515 = llvm.extractvalue %491[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %516 = llvm.getelementptr %514[%515] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %517 = llvm.extractvalue %491[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %518 = llvm.extractvalue %491[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %519 = llvm.getelementptr %517[%518] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%519, %516, %513) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %520 = llvm.add %488, %4 : i64
    llvm.br ^bb73(%520 : i64)
  ^bb81:  // pred: ^bb73
    llvm.br ^bb82(%3 : i64)
  ^bb82(%521: i64):  // 2 preds: ^bb81, ^bb95
    %522 = builtin.unrealized_conversion_cast %521 : i64 to index
    %523 = llvm.icmp "slt" %521, %1 : i64
    llvm.cond_br %523, ^bb83, ^bb96
  ^bb83:  // pred: ^bb82
    llvm.br ^bb84(%3 : i64)
  ^bb84(%524: i64):  // 2 preds: ^bb83, ^bb94
    %525 = builtin.unrealized_conversion_cast %524 : i64 to index
    %526 = llvm.icmp "slt" %524, %2 : i64
    llvm.cond_br %526, ^bb85, ^bb95
  ^bb85:  // pred: ^bb84
    %subview_10 = memref.subview %402[0, %522, %525] [1, 8, 8] [1, 1, 1] : memref<1x1024x768xf32> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
    %527 = builtin.unrealized_conversion_cast %subview_10 : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %subview_11 = memref.subview %487[0, %525] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
    %528 = builtin.unrealized_conversion_cast %subview_11 : memref<1x8xf32, strided<[768, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb86(%3 : i64)
  ^bb86(%529: i64):  // 2 preds: ^bb85, ^bb93
    %530 = llvm.icmp "slt" %529, %0 : i64
    llvm.cond_br %530, ^bb87, ^bb94
  ^bb87:  // pred: ^bb86
    llvm.br ^bb88(%3 : i64)
  ^bb88(%531: i64):  // 2 preds: ^bb87, ^bb92
    %532 = llvm.icmp "slt" %531, %4 : i64
    llvm.cond_br %532, ^bb89, ^bb93
  ^bb89:  // pred: ^bb88
    llvm.br ^bb90(%3 : i64)
  ^bb90(%533: i64):  // 2 preds: ^bb89, ^bb91
    %534 = llvm.icmp "slt" %533, %4 : i64
    llvm.cond_br %534, ^bb91, ^bb92
  ^bb91:  // pred: ^bb90
    %535 = llvm.extractvalue %527[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %536 = llvm.extractvalue %527[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %537 = llvm.getelementptr %535[%536] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %538 = llvm.mlir.constant(786432 : index) : i64
    %539 = llvm.mul %529, %538 : i64
    %540 = llvm.mlir.constant(768 : index) : i64
    %541 = llvm.mul %531, %540 : i64
    %542 = llvm.add %539, %541 : i64
    %543 = llvm.add %542, %533 : i64
    %544 = llvm.getelementptr %537[%543] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %545 = llvm.load %544 : !llvm.ptr -> f32
    %546 = llvm.extractvalue %528[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %547 = llvm.extractvalue %528[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %548 = llvm.getelementptr %546[%547] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %549 = llvm.mlir.constant(768 : index) : i64
    %550 = llvm.mul %529, %549 : i64
    %551 = llvm.add %550, %533 : i64
    %552 = llvm.getelementptr %548[%551] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %553 = llvm.load %552 : !llvm.ptr -> f32
    %554 = llvm.fadd %545, %553  : f32
    %555 = llvm.extractvalue %528[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %556 = llvm.extractvalue %528[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %557 = llvm.getelementptr %555[%556] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %558 = llvm.mlir.constant(768 : index) : i64
    %559 = llvm.mul %529, %558 : i64
    %560 = llvm.add %559, %533 : i64
    %561 = llvm.getelementptr %557[%560] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %554, %561 : f32, !llvm.ptr
    %562 = llvm.add %533, %0 : i64
    llvm.br ^bb90(%562 : i64)
  ^bb92:  // pred: ^bb90
    %563 = llvm.add %531, %0 : i64
    llvm.br ^bb88(%563 : i64)
  ^bb93:  // pred: ^bb88
    %564 = llvm.add %529, %0 : i64
    llvm.br ^bb86(%564 : i64)
  ^bb94:  // pred: ^bb86
    %565 = llvm.mlir.constant(1 : index) : i64
    %566 = llvm.extractvalue %528[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %567 = llvm.mul %565, %566 : i64
    %568 = llvm.extractvalue %528[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %569 = llvm.mul %567, %568 : i64
    %570 = llvm.mlir.zero : !llvm.ptr
    %571 = llvm.getelementptr %570[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %572 = llvm.ptrtoint %571 : !llvm.ptr to i64
    %573 = llvm.mul %569, %572 : i64
    %574 = llvm.extractvalue %528[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %575 = llvm.extractvalue %528[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %576 = llvm.getelementptr %574[%575] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %577 = llvm.extractvalue %528[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %578 = llvm.extractvalue %528[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %579 = llvm.getelementptr %577[%578] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%579, %576, %573) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %580 = llvm.add %524, %4 : i64
    llvm.br ^bb84(%580 : i64)
  ^bb95:  // pred: ^bb84
    %581 = llvm.add %521, %4 : i64
    llvm.br ^bb82(%581 : i64)
  ^bb96:  // pred: ^bb82
    %582 = llvm.mlir.constant(1 : index) : i64
    %583 = llvm.mlir.constant(1024 : index) : i64
    %584 = llvm.mlir.constant(768 : index) : i64
    %585 = llvm.mlir.constant(1 : index) : i64
    %586 = llvm.mlir.constant(786432 : index) : i64
    %587 = llvm.mlir.constant(786432 : index) : i64
    %588 = llvm.mlir.zero : !llvm.ptr
    %589 = llvm.getelementptr %588[%587] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %590 = llvm.ptrtoint %589 : !llvm.ptr to i64
    %591 = llvm.mlir.constant(64 : index) : i64
    %592 = llvm.add %590, %591 : i64
    %593 = llvm.call @malloc(%592) : (i64) -> !llvm.ptr
    %594 = llvm.ptrtoint %593 : !llvm.ptr to i64
    %595 = llvm.mlir.constant(1 : index) : i64
    %596 = llvm.sub %591, %595 : i64
    %597 = llvm.add %594, %596 : i64
    %598 = llvm.urem %597, %591  : i64
    %599 = llvm.sub %597, %598 : i64
    %600 = llvm.inttoptr %599 : i64 to !llvm.ptr
    %601 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %602 = llvm.insertvalue %593, %601[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %603 = llvm.insertvalue %600, %602[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %604 = llvm.mlir.constant(0 : index) : i64
    %605 = llvm.insertvalue %604, %603[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %606 = llvm.insertvalue %582, %605[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %607 = llvm.insertvalue %583, %606[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %608 = llvm.insertvalue %584, %607[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %609 = llvm.insertvalue %586, %608[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %610 = llvm.insertvalue %584, %609[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %611 = llvm.insertvalue %585, %610[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %612 = builtin.unrealized_conversion_cast %611 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<1x1024x768xf32>
    llvm.br ^bb97(%3 : i64)
  ^bb97(%613: i64):  // 2 preds: ^bb96, ^bb110
    %614 = builtin.unrealized_conversion_cast %613 : i64 to index
    %615 = llvm.icmp "slt" %613, %1 : i64
    llvm.cond_br %615, ^bb98, ^bb111
  ^bb98:  // pred: ^bb97
    llvm.br ^bb99(%3 : i64)
  ^bb99(%616: i64):  // 2 preds: ^bb98, ^bb109
    %617 = builtin.unrealized_conversion_cast %616 : i64 to index
    %618 = llvm.icmp "slt" %616, %2 : i64
    llvm.cond_br %618, ^bb100, ^bb110
  ^bb100:  // pred: ^bb99
    %subview_12 = memref.subview %402[0, %614, %617] [1, 8, 8] [1, 1, 1] : memref<1x1024x768xf32> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
    %619 = builtin.unrealized_conversion_cast %subview_12 : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %subview_13 = memref.subview %487[0, %617] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
    %620 = builtin.unrealized_conversion_cast %subview_13 : memref<1x8xf32, strided<[768, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %subview_14 = memref.subview %612[0, %614, %617] [1, 8, 8] [1, 1, 1] : memref<1x1024x768xf32> to memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>>
    %621 = builtin.unrealized_conversion_cast %subview_14 : memref<1x8x8xf32, strided<[786432, 768, 1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    llvm.br ^bb101(%3 : i64)
  ^bb101(%622: i64):  // 2 preds: ^bb100, ^bb108
    %623 = llvm.icmp "slt" %622, %0 : i64
    llvm.cond_br %623, ^bb102, ^bb109
  ^bb102:  // pred: ^bb101
    llvm.br ^bb103(%3 : i64)
  ^bb103(%624: i64):  // 2 preds: ^bb102, ^bb107
    %625 = llvm.icmp "slt" %624, %4 : i64
    llvm.cond_br %625, ^bb104, ^bb108
  ^bb104:  // pred: ^bb103
    llvm.br ^bb105(%3 : i64)
  ^bb105(%626: i64):  // 2 preds: ^bb104, ^bb106
    %627 = llvm.icmp "slt" %626, %4 : i64
    llvm.cond_br %627, ^bb106, ^bb107
  ^bb106:  // pred: ^bb105
    %628 = llvm.extractvalue %619[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %629 = llvm.extractvalue %619[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %630 = llvm.getelementptr %628[%629] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %631 = llvm.mlir.constant(786432 : index) : i64
    %632 = llvm.mul %622, %631 : i64
    %633 = llvm.mlir.constant(768 : index) : i64
    %634 = llvm.mul %624, %633 : i64
    %635 = llvm.add %632, %634 : i64
    %636 = llvm.add %635, %626 : i64
    %637 = llvm.getelementptr %630[%636] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %638 = llvm.load %637 : !llvm.ptr -> f32
    %639 = llvm.extractvalue %620[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %640 = llvm.extractvalue %620[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %641 = llvm.getelementptr %639[%640] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %642 = llvm.mlir.constant(768 : index) : i64
    %643 = llvm.mul %622, %642 : i64
    %644 = llvm.add %643, %626 : i64
    %645 = llvm.getelementptr %641[%644] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %646 = llvm.load %645 : !llvm.ptr -> f32
    %647 = llvm.fdiv %638, %646  : f32
    %648 = llvm.extractvalue %621[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %649 = llvm.extractvalue %621[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %650 = llvm.getelementptr %648[%649] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %651 = llvm.mlir.constant(786432 : index) : i64
    %652 = llvm.mul %622, %651 : i64
    %653 = llvm.mlir.constant(768 : index) : i64
    %654 = llvm.mul %624, %653 : i64
    %655 = llvm.add %652, %654 : i64
    %656 = llvm.add %655, %626 : i64
    %657 = llvm.getelementptr %650[%656] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %647, %657 : f32, !llvm.ptr
    %658 = llvm.add %626, %0 : i64
    llvm.br ^bb105(%658 : i64)
  ^bb107:  // pred: ^bb105
    %659 = llvm.add %624, %0 : i64
    llvm.br ^bb103(%659 : i64)
  ^bb108:  // pred: ^bb103
    %660 = llvm.add %622, %0 : i64
    llvm.br ^bb101(%660 : i64)
  ^bb109:  // pred: ^bb101
    %661 = llvm.intr.stacksave : !llvm.ptr
    %662 = llvm.mlir.constant(3 : i64) : i64
    %663 = llvm.mlir.constant(1 : index) : i64
    %664 = llvm.alloca %663 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %621, %664 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %665 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %666 = llvm.insertvalue %662, %665[0] : !llvm.struct<(i64, ptr)> 
    %667 = llvm.insertvalue %664, %666[1] : !llvm.struct<(i64, ptr)> 
    %668 = llvm.mlir.constant(3 : i64) : i64
    %669 = llvm.mlir.constant(1 : index) : i64
    %670 = llvm.alloca %669 x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %621, %670 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
    %671 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
    %672 = llvm.insertvalue %668, %671[0] : !llvm.struct<(i64, ptr)> 
    %673 = llvm.insertvalue %670, %672[1] : !llvm.struct<(i64, ptr)> 
    %674 = llvm.mlir.constant(1 : index) : i64
    %675 = llvm.alloca %674 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %667, %675 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %676 = llvm.alloca %674 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %673, %676 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %677 = llvm.mlir.zero : !llvm.ptr
    %678 = llvm.getelementptr %677[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %679 = llvm.ptrtoint %678 : !llvm.ptr to i64
    llvm.call @memrefCopy(%679, %675, %676) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %661 : !llvm.ptr
    %680 = llvm.add %616, %4 : i64
    llvm.br ^bb99(%680 : i64)
  ^bb110:  // pred: ^bb99
    %681 = llvm.add %613, %4 : i64
    llvm.br ^bb97(%681 : i64)
  ^bb111:  // pred: ^bb97
    %682 = llvm.mlir.undef : !llvm.struct<(struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)>
    %683 = llvm.insertvalue %119, %682[0] : !llvm.struct<(struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)> 
    %684 = llvm.insertvalue %611, %683[1] : !llvm.struct<(struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)> 
    llvm.return %684 : !llvm.struct<(struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>)>
  }
}
