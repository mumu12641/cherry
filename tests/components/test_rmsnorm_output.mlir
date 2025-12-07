// Original IR loaded from file
module {
  cherry.func @main() -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %1 = cherry.create_tensor dense<1.000000e+00> : tensor<8xf32> -> !cherry.cherry_tensor<[8xf32]>
    %2 = cherry.rmsnorm %0 scale %1 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8xf32]> -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %2 : !cherry.cherry_tensor<[?xf32]>
  }
}

// ==========================================
// Phase: Inliner
// ==========================================
module {
  cherry.func @main() -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %1 = cherry.create_tensor dense<1.000000e+00> : tensor<8xf32> -> !cherry.cherry_tensor<[8xf32]>
    %2 = cherry.rmsnorm %0 scale %1 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8xf32]> -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %2 : !cherry.cherry_tensor<[?xf32]>
  }
}

// ==========================================
// Phase: Shape Inference
// ==========================================
module {
  cherry.func @main() -> !cherry.cherry_tensor<[1x4x8xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %1 = cherry.create_tensor dense<1.000000e+00> : tensor<8xf32> -> !cherry.cherry_tensor<[8xf32]>
    %2 = cherry.rmsnorm %0 scale %1 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8xf32]> -> !cherry.cherry_tensor<[1x4x8xf32]>
    cherry.return %2 : !cherry.cherry_tensor<[1x4x8xf32]>
  }
}

// ==========================================
// Phase: Canonicalizer
// ==========================================
module {
  cherry.func @main() -> !cherry.cherry_tensor<[1x4x8xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %1 = cherry.create_tensor dense<1.000000e+00> : tensor<8xf32> -> !cherry.cherry_tensor<[8xf32]>
    %2 = cherry.rmsnorm %0 scale %1 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x4x8xf32]>, !cherry.cherry_tensor<[8xf32]> -> !cherry.cherry_tensor<[1x4x8xf32]>
    cherry.return %2 : !cherry.cherry_tensor<[1x4x8xf32]>
  }
}

// ==========================================
// Phase: Convert to Linalg
// ==========================================
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2)>
module {
  func.func @main() -> tensor<1x4x8xf32> {
    %cst = arith.constant dense<5.000000e-01> : tensor<1x4x8xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<8xf32>
    %0 = tensor.empty() : tensor<1x4xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<1x4xf32>) -> tensor<1x4xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cst : tensor<1x4x8xf32>) outs(%1 : tensor<1x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.mulf %in, %in : f32
      %10 = arith.addf %out, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<1x4xf32>
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %cst, %c2 : tensor<1x4x8xf32>
    %3 = arith.index_cast %dim : index to i64
    %4 = arith.uitofp %3 : i64 to f32
    %cst_2 = arith.constant 9.99999974E-6 : f32
    %5 = tensor.empty() : tensor<1x4xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<1x4xf32>) outs(%5 : tensor<1x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.divf %in, %4 : f32
      %10 = arith.addf %9, %cst_2 : f32
      %11 = math.rsqrt %10 : f32
      linalg.yield %11 : f32
    } -> tensor<1x4xf32>
    %7 = tensor.empty() : tensor<1x4x8xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst, %6, %cst_0 : tensor<1x4x8xf32>, tensor<1x4xf32>, tensor<8xf32>) outs(%7 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %in_3: f32, %in_4: f32, %out: f32):
      %9 = arith.mulf %in, %in_3 : f32
      %10 = arith.mulf %9, %in_4 : f32
      linalg.yield %10 : f32
    } -> tensor<1x4x8xf32>
    return %8 : tensor<1x4x8xf32>
  }
}

// ==========================================
// Phase: Linalg Tiling
// ==========================================
#map = affine_map<(d0) -> (-d0 + 1, 8)>
#map1 = affine_map<(d0) -> (-d0 + 4, 8)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
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
    %cst = arith.constant 5.000000e-01 : f32
    %cst_8 = arith.constant 0.000000e+00 : f32
    %cst_9 = arith.constant 8.000000e+00 : f32
    %cst_10 = arith.constant 9.99999974E-6 : f32
    %cst_11 = arith.constant dense<5.000000e-01> : tensor<1x4x8xf32>
    %0 = tensor.empty() : tensor<1x4xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_12 = arith.constant 8 : index
    %c0_13 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8_14 = arith.constant 8 : index
    %1 = scf.for %arg0 = %c0 to %c1 step %c8_12 iter_args(%arg1 = %0) -> (tensor<1x4xf32>) {
      %5 = scf.for %arg2 = %c0_13 to %c4 step %c8_14 iter_args(%arg3 = %arg1) -> (tensor<1x4xf32>) {
        %6 = affine.min #map(%arg0)
        %7 = affine.min #map1(%arg2)
        %extracted_slice = tensor.extract_slice %arg3[%arg0, %arg2] [%6, %7] [1, 1] : tensor<1x4xf32> to tensor<?x?xf32>
        %8 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice : tensor<?x?xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_8 : f32
        } -> tensor<?x?xf32>
        %inserted_slice = tensor.insert_slice %8 into %arg3[%arg0, %arg2] [%6, %7] [1, 1] : tensor<?x?xf32> into tensor<1x4xf32>
        scf.yield %inserted_slice : tensor<1x4xf32>
      }
      scf.yield %5 : tensor<1x4xf32>
    }
    %c0_15 = arith.constant 0 : index
    %c1_16 = arith.constant 1 : index
    %c8_17 = arith.constant 8 : index
    %c0_18 = arith.constant 0 : index
    %c4_19 = arith.constant 4 : index
    %c8_20 = arith.constant 8 : index
    %c0_21 = arith.constant 0 : index
    %c8_22 = arith.constant 8 : index
    %c8_23 = arith.constant 8 : index
    %2 = scf.for %arg0 = %c0_15 to %c1_16 step %c8_17 iter_args(%arg1 = %1) -> (tensor<1x4xf32>) {
      %5 = scf.for %arg2 = %c0_18 to %c4_19 step %c8_20 iter_args(%arg3 = %arg1) -> (tensor<1x4xf32>) {
        %6 = scf.for %arg4 = %c0_21 to %c8_22 step %c8_23 iter_args(%arg5 = %arg3) -> (tensor<1x4xf32>) {
          %7 = affine.min #map(%arg0)
          %8 = affine.min #map1(%arg2)
          %9 = affine.min #map(%arg0)
          %10 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %cst_11[%arg0, %arg2, %arg4] [%7, %8, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %extracted_slice_33 = tensor.extract_slice %arg5[%arg0, %arg2] [%9, %10] [1, 1] : tensor<1x4xf32> to tensor<?x?xf32>
          %11 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<?x?x8xf32>) outs(%extracted_slice_33 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %12 = arith.mulf %in, %in : f32
            %13 = arith.addf %out, %12 : f32
            linalg.yield %13 : f32
          } -> tensor<?x?xf32>
          %inserted_slice = tensor.insert_slice %11 into %arg5[%arg0, %arg2] [%9, %10] [1, 1] : tensor<?x?xf32> into tensor<1x4xf32>
          scf.yield %inserted_slice : tensor<1x4xf32>
        }
        scf.yield %6 : tensor<1x4xf32>
      }
      scf.yield %5 : tensor<1x4xf32>
    }
    %3 = tensor.empty() : tensor<1x4x8xf32>
    %c0_24 = arith.constant 0 : index
    %c1_25 = arith.constant 1 : index
    %c8_26 = arith.constant 8 : index
    %c0_27 = arith.constant 0 : index
    %c4_28 = arith.constant 4 : index
    %c8_29 = arith.constant 8 : index
    %c0_30 = arith.constant 0 : index
    %c8_31 = arith.constant 8 : index
    %c8_32 = arith.constant 8 : index
    %4 = scf.for %arg0 = %c0_24 to %c1_25 step %c8_26 iter_args(%arg1 = %3) -> (tensor<1x4x8xf32>) {
      %5 = scf.for %arg2 = %c0_27 to %c4_28 step %c8_29 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %6 = scf.for %arg4 = %c0_30 to %c8_31 step %c8_32 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %7 = affine.min #map(%arg0)
          %8 = affine.min #map1(%arg2)
          %9 = affine.min #map(%arg0)
          %10 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %2[%arg0, %arg2] [%7, %8] [1, 1] : tensor<1x4xf32> to tensor<?x?xf32>
          %extracted_slice_33 = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%9, %10, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %11 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<?x?xf32>) outs(%extracted_slice_33 : tensor<?x?x8xf32>) {
          ^bb0(%in: f32, %out: f32):
            %12 = arith.divf %in, %cst_9 : f32
            %13 = arith.addf %12, %cst_10 : f32
            %14 = math.rsqrt %13 : f32
            %15 = arith.mulf %14, %cst : f32
            linalg.yield %15 : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %11 into %arg5[%arg0, %arg2, %arg4] [%9, %10, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %6 : tensor<1x4x8xf32>
      }
      scf.yield %5 : tensor<1x4x8xf32>
    }
    return %4 : tensor<1x4x8xf32>
  }
}

// ==========================================
// Phase: Bufferization
// ==========================================
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  memref.global "private" constant @__constant_1x4x8xf32 : memref<1x4x8xf32> = dense<5.000000e-01> {alignment = 64 : i64}
  func.func @main() -> memref<1x4x8xf32> {
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 8.000000e+00 : f32
    %cst_2 = arith.constant 9.99999974E-6 : f32
    %0 = memref.get_global @__constant_1x4x8xf32 : memref<1x4x8xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%alloc : memref<1x4xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    }
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0 : memref<1x4x8xf32>) outs(%alloc : memref<1x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.mulf %in, %in : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc : memref<1x4xf32>) outs(%alloc_3 : memref<1x4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.divf %in, %cst_1 : f32
      %2 = arith.addf %1, %cst_2 : f32
      %3 = math.rsqrt %2 : f32
      %4 = arith.mulf %3, %cst : f32
      linalg.yield %4 : f32
    }
    return %alloc_3 : memref<1x4x8xf32>
  }
}

// ==========================================
// Phase: linalg to scf
// ==========================================
module {
  memref.global "private" constant @__constant_1x4x8xf32 : memref<1x4x8xf32> = dense<5.000000e-01> {alignment = 64 : i64}
  func.func @main() -> memref<1x4x8xf32> {
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 8.000000e+00 : f32
    %cst_2 = arith.constant 9.99999974E-6 : f32
    %0 = memref.get_global @__constant_1x4x8xf32 : memref<1x4x8xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        memref.store %cst_0, %alloc[%arg0, %arg1] : memref<1x4xf32>
      }
    }
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          %1 = memref.load %0[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
          %2 = memref.load %alloc[%arg0, %arg1] : memref<1x4xf32>
          %3 = arith.mulf %1, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %alloc[%arg0, %arg1] : memref<1x4xf32>
        }
      }
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          %1 = memref.load %alloc[%arg0, %arg1] : memref<1x4xf32>
          %2 = arith.divf %1, %cst_1 : f32
          %3 = arith.addf %2, %cst_2 : f32
          %4 = math.rsqrt %3 : f32
          %5 = arith.mulf %4, %cst : f32
          memref.store %5, %alloc_3[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
        }
      }
    }
    return %alloc_3 : memref<1x4x8xf32>
  }
}

// ==========================================
// Phase: lower to llvm
// ==========================================
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_1x4x8xf32(dense<5.000000e-01> : tensor<1x4x8xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x array<4 x array<8 x f32>>>
  llvm.func @main() -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> {
    %0 = llvm.mlir.constant(8 : index) : i64
    %1 = llvm.mlir.constant(4 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(5.000000e-01 : f32) : f32
    %5 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %6 = llvm.mlir.constant(8.000000e+00 : f32) : f32
    %7 = llvm.mlir.constant(9.99999974E-6 : f32) : f32
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(4 : index) : i64
    %10 = llvm.mlir.constant(8 : index) : i64
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.mlir.constant(32 : index) : i64
    %13 = llvm.mlir.constant(32 : index) : i64
    %14 = llvm.mlir.zero : !llvm.ptr
    %15 = llvm.getelementptr %14[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %16 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %17 = llvm.mlir.addressof @__constant_1x4x8xf32 : !llvm.ptr
    %18 = llvm.getelementptr %17[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<4 x array<8 x f32>>>
    %19 = llvm.mlir.constant(3735928559 : index) : i64
    %20 = llvm.inttoptr %19 : i64 to !llvm.ptr
    %21 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %22 = llvm.insertvalue %20, %21[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %23 = llvm.insertvalue %18, %22[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %24 = llvm.mlir.constant(0 : index) : i64
    %25 = llvm.insertvalue %24, %23[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %26 = llvm.insertvalue %8, %25[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %27 = llvm.insertvalue %9, %26[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %28 = llvm.insertvalue %10, %27[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %29 = llvm.insertvalue %12, %28[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %30 = llvm.insertvalue %10, %29[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %31 = llvm.insertvalue %11, %30[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %32 = llvm.mlir.constant(1 : index) : i64
    %33 = llvm.mlir.constant(4 : index) : i64
    %34 = llvm.mlir.constant(1 : index) : i64
    %35 = llvm.mlir.constant(4 : index) : i64
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
    %49 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %50 = llvm.insertvalue %41, %49[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.insertvalue %48, %50[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.mlir.constant(0 : index) : i64
    %53 = llvm.insertvalue %52, %51[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.insertvalue %32, %53[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.insertvalue %33, %54[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.insertvalue %33, %55[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.insertvalue %34, %56[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%3 : i64)
  ^bb1(%58: i64):  // 2 preds: ^bb0, ^bb5
    %59 = llvm.icmp "slt" %58, %2 : i64
    llvm.cond_br %59, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%3 : i64)
  ^bb3(%60: i64):  // 2 preds: ^bb2, ^bb4
    %61 = llvm.icmp "slt" %60, %1 : i64
    llvm.cond_br %61, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %62 = llvm.extractvalue %57[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.mlir.constant(4 : index) : i64
    %64 = llvm.mul %58, %63 : i64
    %65 = llvm.add %64, %60 : i64
    %66 = llvm.getelementptr %62[%65] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %5, %66 : f32, !llvm.ptr
    %67 = llvm.add %60, %2 : i64
    llvm.br ^bb3(%67 : i64)
  ^bb5:  // pred: ^bb3
    %68 = llvm.add %58, %2 : i64
    llvm.br ^bb1(%68 : i64)
  ^bb6:  // pred: ^bb1
    llvm.br ^bb7(%3 : i64)
  ^bb7(%69: i64):  // 2 preds: ^bb6, ^bb14
    %70 = llvm.icmp "slt" %69, %2 : i64
    llvm.cond_br %70, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%3 : i64)
  ^bb9(%71: i64):  // 2 preds: ^bb8, ^bb13
    %72 = llvm.icmp "slt" %71, %1 : i64
    llvm.cond_br %72, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%3 : i64)
  ^bb11(%73: i64):  // 2 preds: ^bb10, ^bb12
    %74 = llvm.icmp "slt" %73, %0 : i64
    llvm.cond_br %74, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %75 = llvm.extractvalue %31[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %76 = llvm.mlir.constant(32 : index) : i64
    %77 = llvm.mul %69, %76 : i64
    %78 = llvm.mlir.constant(8 : index) : i64
    %79 = llvm.mul %71, %78 : i64
    %80 = llvm.add %77, %79 : i64
    %81 = llvm.add %80, %73 : i64
    %82 = llvm.getelementptr %75[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %83 = llvm.load %82 : !llvm.ptr -> f32
    %84 = llvm.extractvalue %57[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %85 = llvm.mlir.constant(4 : index) : i64
    %86 = llvm.mul %69, %85 : i64
    %87 = llvm.add %86, %71 : i64
    %88 = llvm.getelementptr %84[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %89 = llvm.load %88 : !llvm.ptr -> f32
    %90 = llvm.fmul %83, %83  : f32
    %91 = llvm.fadd %89, %90  : f32
    %92 = llvm.extractvalue %57[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.mlir.constant(4 : index) : i64
    %94 = llvm.mul %69, %93 : i64
    %95 = llvm.add %94, %71 : i64
    %96 = llvm.getelementptr %92[%95] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %91, %96 : f32, !llvm.ptr
    %97 = llvm.add %73, %2 : i64
    llvm.br ^bb11(%97 : i64)
  ^bb13:  // pred: ^bb11
    %98 = llvm.add %71, %2 : i64
    llvm.br ^bb9(%98 : i64)
  ^bb14:  // pred: ^bb9
    %99 = llvm.add %69, %2 : i64
    llvm.br ^bb7(%99 : i64)
  ^bb15:  // pred: ^bb7
    %100 = llvm.mlir.constant(1 : index) : i64
    %101 = llvm.mlir.constant(4 : index) : i64
    %102 = llvm.mlir.constant(8 : index) : i64
    %103 = llvm.mlir.constant(1 : index) : i64
    %104 = llvm.mlir.constant(32 : index) : i64
    %105 = llvm.mlir.constant(32 : index) : i64
    %106 = llvm.mlir.zero : !llvm.ptr
    %107 = llvm.getelementptr %106[%105] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %108 = llvm.ptrtoint %107 : !llvm.ptr to i64
    %109 = llvm.mlir.constant(64 : index) : i64
    %110 = llvm.add %108, %109 : i64
    %111 = llvm.call @malloc(%110) : (i64) -> !llvm.ptr
    %112 = llvm.ptrtoint %111 : !llvm.ptr to i64
    %113 = llvm.mlir.constant(1 : index) : i64
    %114 = llvm.sub %109, %113 : i64
    %115 = llvm.add %112, %114 : i64
    %116 = llvm.urem %115, %109  : i64
    %117 = llvm.sub %115, %116 : i64
    %118 = llvm.inttoptr %117 : i64 to !llvm.ptr
    %119 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %120 = llvm.insertvalue %111, %119[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %121 = llvm.insertvalue %118, %120[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %122 = llvm.mlir.constant(0 : index) : i64
    %123 = llvm.insertvalue %122, %121[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %124 = llvm.insertvalue %100, %123[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %125 = llvm.insertvalue %101, %124[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %126 = llvm.insertvalue %102, %125[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %127 = llvm.insertvalue %104, %126[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %128 = llvm.insertvalue %102, %127[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %129 = llvm.insertvalue %103, %128[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb16(%3 : i64)
  ^bb16(%130: i64):  // 2 preds: ^bb15, ^bb23
    %131 = llvm.icmp "slt" %130, %2 : i64
    llvm.cond_br %131, ^bb17, ^bb24
  ^bb17:  // pred: ^bb16
    llvm.br ^bb18(%3 : i64)
  ^bb18(%132: i64):  // 2 preds: ^bb17, ^bb22
    %133 = llvm.icmp "slt" %132, %1 : i64
    llvm.cond_br %133, ^bb19, ^bb23
  ^bb19:  // pred: ^bb18
    llvm.br ^bb20(%3 : i64)
  ^bb20(%134: i64):  // 2 preds: ^bb19, ^bb21
    %135 = llvm.icmp "slt" %134, %0 : i64
    llvm.cond_br %135, ^bb21, ^bb22
  ^bb21:  // pred: ^bb20
    %136 = llvm.extractvalue %57[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %137 = llvm.mlir.constant(4 : index) : i64
    %138 = llvm.mul %130, %137 : i64
    %139 = llvm.add %138, %132 : i64
    %140 = llvm.getelementptr %136[%139] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %141 = llvm.load %140 : !llvm.ptr -> f32
    %142 = llvm.fdiv %141, %6  : f32
    %143 = llvm.fadd %142, %7  : f32
    %144 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %145 = llvm.intr.sqrt(%143)  : (f32) -> f32
    %146 = llvm.fdiv %144, %145  : f32
    %147 = llvm.fmul %146, %4  : f32
    %148 = llvm.extractvalue %129[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %149 = llvm.mlir.constant(32 : index) : i64
    %150 = llvm.mul %130, %149 : i64
    %151 = llvm.mlir.constant(8 : index) : i64
    %152 = llvm.mul %132, %151 : i64
    %153 = llvm.add %150, %152 : i64
    %154 = llvm.add %153, %134 : i64
    %155 = llvm.getelementptr %148[%154] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %147, %155 : f32, !llvm.ptr
    %156 = llvm.add %134, %2 : i64
    llvm.br ^bb20(%156 : i64)
  ^bb22:  // pred: ^bb20
    %157 = llvm.add %132, %2 : i64
    llvm.br ^bb18(%157 : i64)
  ^bb23:  // pred: ^bb18
    %158 = llvm.add %130, %2 : i64
    llvm.br ^bb16(%158 : i64)
  ^bb24:  // pred: ^bb16
    llvm.return %129 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  }
}
