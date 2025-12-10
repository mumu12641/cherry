// Original IR loaded from file
module {
  cherry.func @test_rmsnorm() -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x768xf32> -> !cherry.cherry_tensor<[1x768xf32]>
    %1 = cherry.create_tensor dense<1.000000e+00> : tensor<1x768xf32> -> !cherry.cherry_tensor<[1x768xf32]>
    %2 = cherry.constant(768 : i64) : i64
    %3 = cherry.reshape %1, %2 : (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>
    %4 = cherry.rmsnorm %0 scale %3 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %4 : !cherry.cherry_tensor<[?xf32]>
  }
}

// ==========================================
// Phase: Inliner
// ==========================================
module {
  cherry.func @test_rmsnorm() -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x768xf32> -> !cherry.cherry_tensor<[1x768xf32]>
    %1 = cherry.create_tensor dense<1.000000e+00> : tensor<1x768xf32> -> !cherry.cherry_tensor<[1x768xf32]>
    %2 = cherry.constant(768 : i64) : i64
    %3 = cherry.reshape %1, %2 : (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>
    %4 = cherry.rmsnorm %0 scale %3 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %4 : !cherry.cherry_tensor<[?xf32]>
  }
}

// ==========================================
// Phase: Shape Inference
// ==========================================
module {
  cherry.func @test_rmsnorm() -> !cherry.cherry_tensor<[1x768xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x768xf32> -> !cherry.cherry_tensor<[1x768xf32]>
    %1 = cherry.create_tensor dense<1.000000e+00> : tensor<1x768xf32> -> !cherry.cherry_tensor<[1x768xf32]>
    %2 = cherry.constant(768 : i64) : i64
    %3 = cherry.reshape %1, %2 : (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>
    %4 = cherry.rmsnorm %0 scale %3 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
    cherry.return %4 : !cherry.cherry_tensor<[1x768xf32]>
  }
}

// ==========================================
// Phase: Canonicalizer
// ==========================================
module {
  cherry.func @test_rmsnorm() -> !cherry.cherry_tensor<[1x768xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x768xf32> -> !cherry.cherry_tensor<[1x768xf32]>
    %1 = cherry.create_tensor dense<1.000000e+00> : tensor<1x768xf32> -> !cherry.cherry_tensor<[1x768xf32]>
    %2 = cherry.constant(768 : i64) : i64
    %3 = cherry.reshape %1, %2 : (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>
    %4 = cherry.rmsnorm %0 scale %3 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
    cherry.return %4 : !cherry.cherry_tensor<[1x768xf32]>
  }
}

// ==========================================
// Phase: Convert to Linalg
// ==========================================
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @test_rmsnorm() -> tensor<1x768xf32> {
    %cst = arith.constant dense<5.000000e-01> : tensor<1x768xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<1x768xf32>
    %c768_i64 = arith.constant 768 : i64
    %c768_i64_1 = arith.constant 768 : i64
    %from_elements = tensor.from_elements %c768_i64_1 : tensor<1xi64>
    %reshape = tensor.reshape %cst_0(%from_elements) : (tensor<1x768xf32>, tensor<1xi64>) -> tensor<768xf32>
    %0 = tensor.empty() : tensor<1xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%cst : tensor<1x768xf32>) outs(%1 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.mulf %in, %in : f32
      %10 = arith.addf %out, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<1xf32>
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %cst, %c1 : tensor<1x768xf32>
    %3 = arith.index_cast %dim : index to i64
    %4 = arith.uitofp %3 : i64 to f32
    %cst_3 = arith.constant 9.99999974E-6 : f32
    %5 = tensor.empty() : tensor<1xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%2 : tensor<1xf32>) outs(%5 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.divf %in, %4 : f32
      %10 = arith.addf %9, %cst_3 : f32
      %11 = math.rsqrt %10 : f32
      linalg.yield %11 : f32
    } -> tensor<1xf32>
    %7 = tensor.empty() : tensor<1x768xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%cst, %6, %reshape : tensor<1x768xf32>, tensor<1xf32>, tensor<768xf32>) outs(%7 : tensor<1x768xf32>) {
    ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
      %9 = arith.mulf %in, %in_4 : f32
      %10 = arith.mulf %9, %in_5 : f32
      linalg.yield %10 : f32
    } -> tensor<1x768xf32>
    return %8 : tensor<1x768xf32>
  }
}

// ==========================================
// Phase: Linalg Tiling
// ==========================================
#map = affine_map<(d0) -> (-d0 + 1, 8)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @test_rmsnorm() -> tensor<1x768xf32> {
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
    %cst_9 = arith.constant 7.680000e+02 : f32
    %cst_10 = arith.constant 9.99999974E-6 : f32
    %cst_11 = arith.constant dense<5.000000e-01> : tensor<1x768xf32>
    %0 = tensor.empty() : tensor<1xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_12 = arith.constant 8 : index
    %1 = scf.for %arg0 = %c0 to %c1 step %c8_12 iter_args(%arg1 = %0) -> (tensor<1xf32>) {
      %5 = affine.min #map(%arg0)
      %extracted_slice = tensor.extract_slice %arg1[%arg0] [%5] [1] : tensor<1xf32> to tensor<?xf32>
      %6 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice : tensor<?xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_8 : f32
      } -> tensor<?xf32>
      %inserted_slice = tensor.insert_slice %6 into %arg1[%arg0] [%5] [1] : tensor<?xf32> into tensor<1xf32>
      scf.yield %inserted_slice : tensor<1xf32>
    }
    %c0_13 = arith.constant 0 : index
    %c1_14 = arith.constant 1 : index
    %c8_15 = arith.constant 8 : index
    %c0_16 = arith.constant 0 : index
    %c768 = arith.constant 768 : index
    %c8_17 = arith.constant 8 : index
    %2 = scf.for %arg0 = %c0_13 to %c1_14 step %c8_15 iter_args(%arg1 = %1) -> (tensor<1xf32>) {
      %5 = scf.for %arg2 = %c0_16 to %c768 step %c8_17 iter_args(%arg3 = %arg1) -> (tensor<1xf32>) {
        %6 = affine.min #map(%arg0)
        %7 = affine.min #map(%arg0)
        %extracted_slice = tensor.extract_slice %cst_11[%arg0, %arg2] [%6, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
        %extracted_slice_24 = tensor.extract_slice %arg3[%arg0] [%7] [1] : tensor<1xf32> to tensor<?xf32>
        %8 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice : tensor<?x8xf32>) outs(%extracted_slice_24 : tensor<?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %9 = arith.mulf %in, %in : f32
          %10 = arith.addf %out, %9 : f32
          linalg.yield %10 : f32
        } -> tensor<?xf32>
        %inserted_slice = tensor.insert_slice %8 into %arg3[%arg0] [%7] [1] : tensor<?xf32> into tensor<1xf32>
        scf.yield %inserted_slice : tensor<1xf32>
      }
      scf.yield %5 : tensor<1xf32>
    }
    %3 = tensor.empty() : tensor<1x768xf32>
    %c0_18 = arith.constant 0 : index
    %c1_19 = arith.constant 1 : index
    %c8_20 = arith.constant 8 : index
    %c0_21 = arith.constant 0 : index
    %c768_22 = arith.constant 768 : index
    %c8_23 = arith.constant 8 : index
    %4 = scf.for %arg0 = %c0_18 to %c1_19 step %c8_20 iter_args(%arg1 = %3) -> (tensor<1x768xf32>) {
      %5 = scf.for %arg2 = %c0_21 to %c768_22 step %c8_23 iter_args(%arg3 = %arg1) -> (tensor<1x768xf32>) {
        %6 = affine.min #map(%arg0)
        %7 = affine.min #map(%arg0)
        %extracted_slice = tensor.extract_slice %2[%arg0] [%6] [1] : tensor<1xf32> to tensor<?xf32>
        %extracted_slice_24 = tensor.extract_slice %arg3[%arg0, %arg2] [%7, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
        %8 = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice : tensor<?xf32>) outs(%extracted_slice_24 : tensor<?x8xf32>) {
        ^bb0(%in: f32, %out: f32):
          %9 = arith.divf %in, %cst_9 : f32
          %10 = arith.addf %9, %cst_10 : f32
          %11 = math.rsqrt %10 : f32
          %12 = arith.mulf %11, %cst : f32
          linalg.yield %12 : f32
        } -> tensor<?x8xf32>
        %inserted_slice = tensor.insert_slice %8 into %arg3[%arg0, %arg2] [%7, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
        scf.yield %inserted_slice : tensor<1x768xf32>
      }
      scf.yield %5 : tensor<1x768xf32>
    }
    return %4 : tensor<1x768xf32>
  }
}

// ==========================================
// Phase: Bufferization
// ==========================================
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  memref.global "private" constant @__constant_1x768xf32 : memref<1x768xf32> = dense<5.000000e-01> {alignment = 64 : i64}
  func.func @test_rmsnorm() -> memref<1x768xf32> {
    %c768 = arith.constant 768 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 7.680000e+02 : f32
    %cst_2 = arith.constant 9.99999974E-6 : f32
    %0 = memref.get_global @__constant_1x768xf32 : memref<1x768xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloc : memref<1xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_0 : f32
    }
    scf.for %arg0 = %c0 to %c768 step %c8 {
      %subview = memref.subview %0[0, %arg0] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%subview : memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%alloc : memref<1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %1 = arith.mulf %in, %in : f32
        %2 = arith.addf %out, %1 : f32
        linalg.yield %2 : f32
      }
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
    scf.for %arg0 = %c0 to %c768 step %c8 {
      %subview = memref.subview %alloc_3[0, %arg0] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%alloc : memref<1xf32>) outs(%subview : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
      ^bb0(%in: f32, %out: f32):
        %1 = arith.divf %in, %cst_1 : f32
        %2 = arith.addf %1, %cst_2 : f32
        %3 = math.rsqrt %2 : f32
        %4 = arith.mulf %3, %cst : f32
        linalg.yield %4 : f32
      }
      memref.copy %subview, %subview : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
    }
    return %alloc_3 : memref<1x768xf32>
  }
}

// ==========================================
// Phase: linalg to scf
// ==========================================
module {
  memref.global "private" constant @__constant_1x768xf32 : memref<1x768xf32> = dense<5.000000e-01> {alignment = 64 : i64}
  func.func @test_rmsnorm() -> memref<1x768xf32> {
    %c1 = arith.constant 1 : index
    %c768 = arith.constant 768 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 7.680000e+02 : f32
    %cst_2 = arith.constant 9.99999974E-6 : f32
    %0 = memref.get_global @__constant_1x768xf32 : memref<1x768xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      memref.store %cst_0, %alloc[%arg0] : memref<1xf32>
    }
    scf.for %arg0 = %c0 to %c768 step %c8 {
      %subview = memref.subview %0[0, %arg0] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          %1 = memref.load %subview[%arg1, %arg2] : memref<1x8xf32, strided<[768, 1], offset: ?>>
          %2 = memref.load %alloc[%arg1] : memref<1xf32>
          %3 = arith.mulf %1, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %alloc[%arg1] : memref<1xf32>
        }
      }
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
    scf.for %arg0 = %c0 to %c768 step %c8 {
      %subview = memref.subview %alloc_3[0, %arg0] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
      scf.for %arg1 = %c0 to %c1 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          %1 = memref.load %alloc[%arg1] : memref<1xf32>
          %2 = arith.divf %1, %cst_1 : f32
          %3 = arith.addf %2, %cst_2 : f32
          %4 = math.rsqrt %3 : f32
          %5 = arith.mulf %4, %cst : f32
          memref.store %5, %subview[%arg1, %arg2] : memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
      }
      memref.copy %subview, %subview : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
    }
    return %alloc_3 : memref<1x768xf32>
  }
}

// ==========================================
// Phase: lower to llvm
// ==========================================
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_1x768xf32(dense<5.000000e-01> : tensor<1x768xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x array<768 x f32>>
  llvm.func @test_rmsnorm() -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
    %0 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.mlir.addressof @__constant_1x768xf32 : !llvm.ptr
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(768 : index) : i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(8 : index) : i64
    %8 = llvm.mlir.constant(5.000000e-01 : f32) : f32
    %9 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %10 = llvm.mlir.constant(7.680000e+02 : f32) : f32
    %11 = llvm.mlir.constant(9.99999974E-6 : f32) : f32
    %12 = llvm.mlir.zero : !llvm.ptr
    %13 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x f32>>
    %14 = llvm.getelementptr %12[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %15 = llvm.ptrtoint %14 : !llvm.ptr to i64
    %16 = llvm.add %15, %1 : i64
    %17 = llvm.call @malloc(%16) : (i64) -> !llvm.ptr
    %18 = llvm.ptrtoint %17 : !llvm.ptr to i64
    %19 = llvm.sub %1, %4 : i64
    %20 = llvm.add %18, %19 : i64
    %21 = llvm.urem %20, %1  : i64
    %22 = llvm.sub %20, %21 : i64
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr
    llvm.br ^bb1(%6 : i64)
  ^bb1(%24: i64):  // 2 preds: ^bb0, ^bb2
    %25 = llvm.icmp "slt" %24, %4 : i64
    llvm.cond_br %25, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %26 = llvm.getelementptr %23[%24] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %9, %26 : f32, !llvm.ptr
    %27 = llvm.add %24, %4 : i64
    llvm.br ^bb1(%27 : i64)
  ^bb3:  // pred: ^bb1
    llvm.br ^bb4(%6 : i64)
  ^bb4(%28: i64):  // 2 preds: ^bb3, ^bb11
    %29 = llvm.icmp "slt" %28, %5 : i64
    llvm.cond_br %29, ^bb5, ^bb12
  ^bb5:  // pred: ^bb4
    llvm.br ^bb6(%6 : i64)
  ^bb6(%30: i64):  // 2 preds: ^bb5, ^bb10
    %31 = llvm.icmp "slt" %30, %4 : i64
    llvm.cond_br %31, ^bb7, ^bb11
  ^bb7:  // pred: ^bb6
    llvm.br ^bb8(%6 : i64)
  ^bb8(%32: i64):  // 2 preds: ^bb7, ^bb9
    %33 = llvm.icmp "slt" %32, %7 : i64
    llvm.cond_br %33, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %34 = llvm.getelementptr %13[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %35 = llvm.mul %30, %5 : i64
    %36 = llvm.add %35, %32 : i64
    %37 = llvm.getelementptr %34[%36] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %38 = llvm.load %37 : !llvm.ptr -> f32
    %39 = llvm.getelementptr %23[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %40 = llvm.load %39 : !llvm.ptr -> f32
    %41 = llvm.fmul %38, %38  : f32
    %42 = llvm.fadd %40, %41  : f32
    llvm.store %42, %39 : f32, !llvm.ptr
    %43 = llvm.add %32, %4 : i64
    llvm.br ^bb8(%43 : i64)
  ^bb10:  // pred: ^bb8
    %44 = llvm.add %30, %4 : i64
    llvm.br ^bb6(%44 : i64)
  ^bb11:  // pred: ^bb6
    %45 = llvm.add %28, %7 : i64
    llvm.br ^bb4(%45 : i64)
  ^bb12:  // pred: ^bb4
    %46 = llvm.getelementptr %12[768] : (!llvm.ptr) -> !llvm.ptr, f32
    %47 = llvm.ptrtoint %46 : !llvm.ptr to i64
    %48 = llvm.add %47, %1 : i64
    %49 = llvm.call @malloc(%48) : (i64) -> !llvm.ptr
    %50 = llvm.ptrtoint %49 : !llvm.ptr to i64
    %51 = llvm.add %50, %19 : i64
    %52 = llvm.urem %51, %1  : i64
    %53 = llvm.sub %51, %52 : i64
    %54 = llvm.inttoptr %53 : i64 to !llvm.ptr
    %55 = llvm.insertvalue %49, %2[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %56 = llvm.insertvalue %54, %55[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.insertvalue %6, %56[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %58 = llvm.insertvalue %4, %57[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.insertvalue %5, %58[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %60 = llvm.insertvalue %5, %59[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.insertvalue %4, %60[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb13(%6 : i64)
  ^bb13(%62: i64):  // 2 preds: ^bb12, ^bb20
    %63 = llvm.icmp "slt" %62, %5 : i64
    llvm.cond_br %63, ^bb14, ^bb21
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%6 : i64)
  ^bb15(%64: i64):  // 2 preds: ^bb14, ^bb19
    %65 = llvm.icmp "slt" %64, %4 : i64
    llvm.cond_br %65, ^bb16, ^bb20
  ^bb16:  // pred: ^bb15
    llvm.br ^bb17(%6 : i64)
  ^bb17(%66: i64):  // 2 preds: ^bb16, ^bb18
    %67 = llvm.icmp "slt" %66, %7 : i64
    llvm.cond_br %67, ^bb18, ^bb19
  ^bb18:  // pred: ^bb17
    %68 = llvm.getelementptr %23[%64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %69 = llvm.load %68 : !llvm.ptr -> f32
    %70 = llvm.fdiv %69, %10  : f32
    %71 = llvm.fadd %70, %11  : f32
    %72 = llvm.intr.sqrt(%71)  : (f32) -> f32
    %73 = llvm.fdiv %0, %72  : f32
    %74 = llvm.fmul %73, %8  : f32
    %75 = llvm.getelementptr %54[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %76 = llvm.mul %64, %5 : i64
    %77 = llvm.add %76, %66 : i64
    %78 = llvm.getelementptr %75[%77] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %74, %78 : f32, !llvm.ptr
    %79 = llvm.add %66, %4 : i64
    llvm.br ^bb17(%79 : i64)
  ^bb19:  // pred: ^bb17
    %80 = llvm.add %64, %4 : i64
    llvm.br ^bb15(%80 : i64)
  ^bb20:  // pred: ^bb15
    %81 = llvm.mul %4, %4 : i64
    %82 = llvm.mul %81, %7 : i64
    %83 = llvm.mul %82, %15 : i64
    %84 = llvm.getelementptr %54[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%84, %84, %83) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %85 = llvm.add %62, %7 : i64
    llvm.br ^bb13(%85 : i64)
  ^bb21:  // pred: ^bb13
    llvm.return %61 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
}
