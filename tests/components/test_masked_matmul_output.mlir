// Original IR loaded from file
module {
  func.func @host() {
    %0 = cherry.create_tensor dense<[[1.000000e+00, 2.000000e+00]]> : tensor<1x2xf32> -> !cherry.cherry_tensor<[1x2xf32]>
    %1 = cherry.create_tensor dense<[[1.000000e+00, 1.000000e+01], [1.000000e+00, 1.000000e+01]]> : tensor<2x2xf32> -> !cherry.cherry_tensor<[2x2xf32]>
    %2 = cherry.constant(0 : i64) : i64
    %3 = cherry.generate_mask %2, [1, 2] : !cherry.cherry_tensor<[1x2xf32]>
    %4 = cherry.masked_matmul %0, %1, %3 : (!cherry.cherry_tensor<[1x2xf32]>, !cherry.cherry_tensor<[2x2xf32]>, !cherry.cherry_tensor<[1x2xf32]>) -> !cherry.cherry_tensor<[1x2xf32]>
    cherry.print %4 : !cherry.cherry_tensor<[1x2xf32]>
    return
  }
}

// ==========================================
// Phase: Inliner
// ==========================================
module {
  func.func @host() {
    %0 = cherry.create_tensor dense<[[1.000000e+00, 2.000000e+00]]> : tensor<1x2xf32> -> !cherry.cherry_tensor<[1x2xf32]>
    %1 = cherry.create_tensor dense<[[1.000000e+00, 1.000000e+01], [1.000000e+00, 1.000000e+01]]> : tensor<2x2xf32> -> !cherry.cherry_tensor<[2x2xf32]>
    %2 = cherry.constant(0 : i64) : i64
    %3 = cherry.generate_mask %2, [1, 2] : !cherry.cherry_tensor<[1x2xf32]>
    %4 = cherry.masked_matmul %0, %1, %3 : (!cherry.cherry_tensor<[1x2xf32]>, !cherry.cherry_tensor<[2x2xf32]>, !cherry.cherry_tensor<[1x2xf32]>) -> !cherry.cherry_tensor<[1x2xf32]>
    cherry.print %4 : !cherry.cherry_tensor<[1x2xf32]>
    return
  }
}

// ==========================================
// Phase: Shape Inference
// ==========================================
module {
  func.func @host() {
    %0 = cherry.create_tensor dense<[[1.000000e+00, 2.000000e+00]]> : tensor<1x2xf32> -> !cherry.cherry_tensor<[1x2xf32]>
    %1 = cherry.create_tensor dense<[[1.000000e+00, 1.000000e+01], [1.000000e+00, 1.000000e+01]]> : tensor<2x2xf32> -> !cherry.cherry_tensor<[2x2xf32]>
    %2 = cherry.constant(0 : i64) : i64
    %3 = cherry.generate_mask %2, [1, 2] : !cherry.cherry_tensor<[1x2xf32]>
    %4 = cherry.masked_matmul %0, %1, %3 : (!cherry.cherry_tensor<[1x2xf32]>, !cherry.cherry_tensor<[2x2xf32]>, !cherry.cherry_tensor<[1x2xf32]>) -> !cherry.cherry_tensor<[1x2xf32]>
    cherry.print %4 : !cherry.cherry_tensor<[1x2xf32]>
    return
  }
}

// ==========================================
// Phase: Canonicalizer
// ==========================================
module {
  func.func @host() {
    %0 = cherry.create_tensor dense<[[1.000000e+00, 2.000000e+00]]> : tensor<1x2xf32> -> !cherry.cherry_tensor<[1x2xf32]>
    %1 = cherry.create_tensor dense<[[1.000000e+00, 1.000000e+01], [1.000000e+00, 1.000000e+01]]> : tensor<2x2xf32> -> !cherry.cherry_tensor<[2x2xf32]>
    %2 = cherry.constant(0 : i64) : i64
    %3 = cherry.generate_mask %2, [1, 2] : !cherry.cherry_tensor<[1x2xf32]>
    %4 = cherry.masked_matmul %0, %1, %3 : (!cherry.cherry_tensor<[1x2xf32]>, !cherry.cherry_tensor<[2x2xf32]>, !cherry.cherry_tensor<[1x2xf32]>) -> !cherry.cherry_tensor<[1x2xf32]>
    cherry.print %4 : !cherry.cherry_tensor<[1x2xf32]>
    return
  }
}

// ==========================================
// Phase: Convert to Linalg
// ==========================================
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func private @printMemrefF32(tensor<*xf32> {bufferization.access = "read"})
  func.func @host() {
    %cst = arith.constant dense<[[1.000000e+00, 2.000000e+00]]> : tensor<1x2xf32>
    %cst_0 = arith.constant dense<[[1.000000e+00, 1.000000e+01], [1.000000e+00, 1.000000e+01]]> : tensor<2x2xf32>
    %c0_i64 = arith.constant 0 : i64
    %0 = tensor.empty() : tensor<1x2xf32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%0 : tensor<1x2xf32>) {
    ^bb0(%out: f32):
      %5 = linalg.index 1 : index
      %6 = arith.index_cast %5 : index to i64
      %7 = arith.cmpi sle, %6, %c0_i64 : i64
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0.000000e+00 : f32
      %8 = arith.select %7, %cst_2, %cst_3 : f32
      linalg.yield %8 : f32
    } -> tensor<1x2xf32>
    %2 = tensor.empty() : tensor<1x2xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<1x2xf32>) -> tensor<1x2xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map2, #map3, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cst, %cst_0, %1 : tensor<1x2xf32>, tensor<2x2xf32>, tensor<1x2xf32>) outs(%3 : tensor<1x2xf32>) {
    ^bb0(%in: f32, %in_2: f32, %in_3: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      %cst_4 = arith.constant -1.000000e+09 : f32
      %cst_5 = arith.constant 5.000000e-01 : f32
      %7 = arith.cmpf ugt, %in_3, %cst_5 : f32
      %8 = arith.select %7, %6, %cst_4 : f32
      linalg.yield %8 : f32
    } -> tensor<1x2xf32>
    %cast = tensor.cast %4 : tensor<1x2xf32> to tensor<*xf32>
    call @printMemrefF32(%cast) : (tensor<*xf32>) -> ()
    return
  }
}

// ==========================================
// Phase: Linalg Tiling
// ==========================================
#map = affine_map<(d0) -> (-d0 + 1, 8)>
#map1 = affine_map<(d0) -> (-d0 + 2, 8)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d0 + d1)>
module {
  func.func private @printMemrefF32(tensor<*xf32> {bufferization.access = "read"})
  func.func @host() {
    %c8 = arith.constant 8 : index
    %c8_0 = arith.constant 8 : index
    %c8_1 = arith.constant 8 : index
    %c8_2 = arith.constant 8 : index
    %c8_3 = arith.constant 8 : index
    %c8_4 = arith.constant 8 : index
    %cst = arith.constant 5.000000e-01 : f32
    %cst_5 = arith.constant -1.000000e+09 : f32
    %cst_6 = arith.constant 0.000000e+00 : f32
    %cst_7 = arith.constant 1.000000e+00 : f32
    %cst_8 = arith.constant dense<[[1.000000e+00, 2.000000e+00]]> : tensor<1x2xf32>
    %cst_9 = arith.constant dense<[[1.000000e+00, 1.000000e+01], [1.000000e+00, 1.000000e+01]]> : tensor<2x2xf32>
    %c0_i64 = arith.constant 0 : i64
    %0 = tensor.empty() : tensor<1x2xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_10 = arith.constant 8 : index
    %c0_11 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c8_12 = arith.constant 8 : index
    %1 = scf.for %arg0 = %c0 to %c1 step %c8_10 iter_args(%arg1 = %0) -> (tensor<1x2xf32>) {
      %3 = scf.for %arg2 = %c0_11 to %c2 step %c8_12 iter_args(%arg3 = %arg1) -> (tensor<1x2xf32>) {
        %4 = affine.min #map(%arg0)
        %5 = affine.min #map1(%arg2)
        %extracted_slice = tensor.extract_slice %arg3[%arg0, %arg2] [%4, %5] [1, 1] : tensor<1x2xf32> to tensor<?x?xf32>
        %6 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice : tensor<?x?xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_6 : f32
        } -> tensor<?x?xf32>
        %inserted_slice = tensor.insert_slice %6 into %arg3[%arg0, %arg2] [%4, %5] [1, 1] : tensor<?x?xf32> into tensor<1x2xf32>
        scf.yield %inserted_slice : tensor<1x2xf32>
      }
      scf.yield %3 : tensor<1x2xf32>
    }
    %c0_13 = arith.constant 0 : index
    %c1_14 = arith.constant 1 : index
    %c8_15 = arith.constant 8 : index
    %c0_16 = arith.constant 0 : index
    %c2_17 = arith.constant 2 : index
    %c8_18 = arith.constant 8 : index
    %c0_19 = arith.constant 0 : index
    %c2_20 = arith.constant 2 : index
    %c8_21 = arith.constant 8 : index
    %2 = scf.for %arg0 = %c0_13 to %c1_14 step %c8_15 iter_args(%arg1 = %1) -> (tensor<1x2xf32>) {
      %3 = scf.for %arg2 = %c0_16 to %c2_17 step %c8_18 iter_args(%arg3 = %arg1) -> (tensor<1x2xf32>) {
        %4 = scf.for %arg4 = %c0_19 to %c2_20 step %c8_21 iter_args(%arg5 = %arg3) -> (tensor<1x2xf32>) {
          %5 = affine.min #map(%arg0)
          %6 = affine.min #map1(%arg4)
          %7 = affine.min #map1(%arg4)
          %8 = affine.min #map1(%arg2)
          %9 = affine.min #map(%arg0)
          %10 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %cst_8[%arg0, %arg4] [%5, %6] [1, 1] : tensor<1x2xf32> to tensor<?x?xf32>
          %extracted_slice_22 = tensor.extract_slice %cst_9[%arg4, %arg2] [%7, %8] [1, 1] : tensor<2x2xf32> to tensor<?x?xf32>
          %extracted_slice_23 = tensor.extract_slice %arg5[%arg0, %arg2] [%9, %10] [1, 1] : tensor<1x2xf32> to tensor<?x?xf32>
          %11 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_22 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_23 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_24: f32, %out: f32):
            %12 = linalg.index 1 : index
            %13 = affine.apply #map6(%12, %arg2)
            %14 = arith.index_cast %13 : index to i64
            %15 = arith.cmpi sle, %14, %c0_i64 : i64
            %16 = arith.select %15, %cst_7, %cst_6 : f32
            %17 = arith.mulf %in, %in_24 : f32
            %18 = arith.addf %out, %17 : f32
            %19 = arith.cmpf ugt, %16, %cst : f32
            %20 = arith.select %19, %18, %cst_5 : f32
            linalg.yield %20 : f32
          } -> tensor<?x?xf32>
          %inserted_slice = tensor.insert_slice %11 into %arg5[%arg0, %arg2] [%9, %10] [1, 1] : tensor<?x?xf32> into tensor<1x2xf32>
          scf.yield %inserted_slice : tensor<1x2xf32>
        }
        scf.yield %4 : tensor<1x2xf32>
      }
      scf.yield %3 : tensor<1x2xf32>
    }
    %cast = tensor.cast %2 : tensor<1x2xf32> to tensor<*xf32>
    call @printMemrefF32(%cast) : (tensor<*xf32>) -> ()
    return
  }
}

// ==========================================
// Phase: Bufferization
// ==========================================
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  memref.global "private" constant @__constant_2x2xf32 : memref<2x2xf32> = dense<[[1.000000e+00, 1.000000e+01], [1.000000e+00, 1.000000e+01]]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x2xf32 : memref<1x2xf32> = dense<[[1.000000e+00, 2.000000e+00]]> {alignment = 64 : i64}
  func.func private @printMemrefF32(memref<*xf32> {bufferization.access = "read"})
  func.func @host() {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant -1.000000e+09 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %0 = memref.get_global @__constant_1x2xf32 : memref<1x2xf32>
    %1 = memref.get_global @__constant_2x2xf32 : memref<2x2xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%alloc : memref<1x2xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst_1 : f32
    }
    linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : memref<1x2xf32>, memref<2x2xf32>) outs(%alloc : memref<1x2xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %2 = linalg.index 1 : index
      %3 = arith.index_cast %2 : index to i64
      %4 = arith.cmpi sle, %3, %c0_i64 : i64
      %5 = arith.select %4, %cst_2, %cst_1 : f32
      %6 = arith.mulf %in, %in_3 : f32
      %7 = arith.addf %out, %6 : f32
      %8 = arith.cmpf ugt, %5, %cst : f32
      %9 = arith.select %8, %7, %cst_0 : f32
      linalg.yield %9 : f32
    }
    %cast = memref.cast %alloc : memref<1x2xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
}

// ==========================================
// Phase: linalg to scf
// ==========================================
module {
  memref.global "private" constant @__constant_2x2xf32 : memref<2x2xf32> = dense<[[1.000000e+00, 1.000000e+01], [1.000000e+00, 1.000000e+01]]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x2xf32 : memref<1x2xf32> = dense<[[1.000000e+00, 2.000000e+00]]> {alignment = 64 : i64}
  func.func private @printMemrefF32(memref<*xf32> {bufferization.access = "read"})
  func.func @host() {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant -1.000000e+09 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %0 = memref.get_global @__constant_1x2xf32 : memref<1x2xf32>
    %1 = memref.get_global @__constant_2x2xf32 : memref<2x2xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c2 step %c1 {
        memref.store %cst_1, %alloc[%arg0, %arg1] : memref<1x2xf32>
      }
    }
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c2 step %c1 {
        scf.for %arg2 = %c0 to %c2 step %c1 {
          %2 = memref.load %0[%arg0, %arg2] : memref<1x2xf32>
          %3 = memref.load %1[%arg2, %arg1] : memref<2x2xf32>
          %4 = memref.load %alloc[%arg0, %arg1] : memref<1x2xf32>
          %5 = arith.index_cast %arg1 : index to i64
          %6 = arith.cmpi sle, %5, %c0_i64 : i64
          %7 = arith.select %6, %cst_2, %cst_1 : f32
          %8 = arith.mulf %2, %3 : f32
          %9 = arith.addf %4, %8 : f32
          %10 = arith.cmpf ugt, %7, %cst : f32
          %11 = arith.select %10, %9, %cst_0 : f32
          memref.store %11, %alloc[%arg0, %arg1] : memref<1x2xf32>
        }
      }
    }
    %cast = memref.cast %alloc : memref<1x2xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
}

// ==========================================
// Phase: lower to llvm
// ==========================================
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_2x2xf32(dense<[[1.000000e+00, 1.000000e+01], [1.000000e+00, 1.000000e+01]]> : tensor<2x2xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<2 x array<2 x f32>>
  llvm.mlir.global private constant @__constant_1x2xf32(dense<[[1.000000e+00, 2.000000e+00]]> : tensor<1x2xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x array<2 x f32>>
  llvm.func @printMemrefF32(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @host() {
    %0 = llvm.mlir.constant(64 : index) : i64
    %1 = llvm.mlir.addressof @__constant_2x2xf32 : !llvm.ptr
    %2 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.mlir.addressof @__constant_1x2xf32 : !llvm.ptr
    %4 = llvm.mlir.constant(2 : index) : i64
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(0 : index) : i64
    %7 = llvm.mlir.constant(0 : i64) : i64
    %8 = llvm.mlir.constant(5.000000e-01 : f32) : f32
    %9 = llvm.mlir.constant(-1.000000e+09 : f32) : f32
    %10 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %11 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %12 = llvm.mlir.zero : !llvm.ptr
    %13 = llvm.getelementptr %3[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<2 x f32>>
    %14 = llvm.getelementptr %1[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x array<2 x f32>>
    %15 = llvm.getelementptr %12[2] : (!llvm.ptr) -> !llvm.ptr, f32
    %16 = llvm.ptrtoint %15 : !llvm.ptr to i64
    %17 = llvm.add %16, %0 : i64
    %18 = llvm.call @malloc(%17) : (i64) -> !llvm.ptr
    %19 = llvm.ptrtoint %18 : !llvm.ptr to i64
    %20 = llvm.sub %0, %5 : i64
    %21 = llvm.add %19, %20 : i64
    %22 = llvm.urem %21, %0  : i64
    %23 = llvm.sub %21, %22 : i64
    %24 = llvm.inttoptr %23 : i64 to !llvm.ptr
    %25 = llvm.insertvalue %18, %2[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.insertvalue %24, %25[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %6, %26[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %5, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.insertvalue %4, %28[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.insertvalue %4, %29[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %5, %30[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%6 : i64)
  ^bb1(%32: i64):  // 2 preds: ^bb0, ^bb5
    %33 = llvm.icmp "slt" %32, %5 : i64
    llvm.cond_br %33, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%6 : i64)
  ^bb3(%34: i64):  // 2 preds: ^bb2, ^bb4
    %35 = llvm.icmp "slt" %34, %4 : i64
    llvm.cond_br %35, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %36 = llvm.mul %32, %4 : i64
    %37 = llvm.add %36, %34 : i64
    %38 = llvm.getelementptr %24[%37] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %10, %38 : f32, !llvm.ptr
    %39 = llvm.add %34, %5 : i64
    llvm.br ^bb3(%39 : i64)
  ^bb5:  // pred: ^bb3
    %40 = llvm.add %32, %5 : i64
    llvm.br ^bb1(%40 : i64)
  ^bb6:  // pred: ^bb1
    llvm.br ^bb7(%6 : i64)
  ^bb7(%41: i64):  // 2 preds: ^bb6, ^bb14
    %42 = llvm.icmp "slt" %41, %5 : i64
    llvm.cond_br %42, ^bb8, ^bb15
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%6 : i64)
  ^bb9(%43: i64):  // 2 preds: ^bb8, ^bb13
    %44 = llvm.icmp "slt" %43, %4 : i64
    llvm.cond_br %44, ^bb10, ^bb14
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%6 : i64)
  ^bb11(%45: i64):  // 2 preds: ^bb10, ^bb12
    %46 = llvm.icmp "slt" %45, %4 : i64
    llvm.cond_br %46, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %47 = llvm.mul %41, %4 : i64
    %48 = llvm.add %47, %45 : i64
    %49 = llvm.getelementptr %13[%48] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %50 = llvm.load %49 : !llvm.ptr -> f32
    %51 = llvm.mul %45, %4 : i64
    %52 = llvm.add %51, %43 : i64
    %53 = llvm.getelementptr %14[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %54 = llvm.load %53 : !llvm.ptr -> f32
    %55 = llvm.add %47, %43 : i64
    %56 = llvm.getelementptr %24[%55] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %57 = llvm.load %56 : !llvm.ptr -> f32
    %58 = llvm.icmp "sle" %43, %7 : i64
    %59 = llvm.select %58, %11, %10 : i1, f32
    %60 = llvm.fmul %50, %54  : f32
    %61 = llvm.fadd %57, %60  : f32
    %62 = llvm.fcmp "ugt" %59, %8 : f32
    %63 = llvm.select %62, %61, %9 : i1, f32
    llvm.store %63, %56 : f32, !llvm.ptr
    %64 = llvm.add %45, %5 : i64
    llvm.br ^bb11(%64 : i64)
  ^bb13:  // pred: ^bb11
    %65 = llvm.add %43, %5 : i64
    llvm.br ^bb9(%65 : i64)
  ^bb14:  // pred: ^bb9
    %66 = llvm.add %41, %5 : i64
    llvm.br ^bb7(%66 : i64)
  ^bb15:  // pred: ^bb7
    %67 = llvm.alloca %5 x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %31, %67 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.call @printMemrefF32(%4, %67) : (i64, !llvm.ptr) -> ()
    llvm.return
  }
}
