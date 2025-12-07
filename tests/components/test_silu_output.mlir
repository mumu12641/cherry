// Original IR loaded from file
module {
  cherry.func @test_silu() -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %1 = cherry.tensor_silu %0 : (!cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %1 : !cherry.cherry_tensor<[?xf32]>
  }
}

// ==========================================
// Phase: Inliner
// ==========================================
module {
  cherry.func @test_silu() -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %1 = cherry.tensor_silu %0 : (!cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %1 : !cherry.cherry_tensor<[?xf32]>
  }
}

// ==========================================
// Phase: Shape Inference
// ==========================================
module {
  cherry.func @test_silu() -> !cherry.cherry_tensor<[1x4x8xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %1 = cherry.tensor_silu %0 : (!cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    cherry.return %1 : !cherry.cherry_tensor<[1x4x8xf32]>
  }
}

// ==========================================
// Phase: Canonicalizer
// ==========================================
module {
  cherry.func @test_silu() -> !cherry.cherry_tensor<[1x4x8xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %1 = cherry.tensor_silu %0 : (!cherry.cherry_tensor<[1x4x8xf32]>) -> !cherry.cherry_tensor<[1x4x8xf32]>
    cherry.return %1 : !cherry.cherry_tensor<[1x4x8xf32]>
  }
}

// ==========================================
// Phase: Convert to Linalg
// ==========================================
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @test_silu() -> tensor<1x4x8xf32> {
    %cst = arith.constant dense<5.000000e-01> : tensor<1x4x8xf32>
    %0 = tensor.empty() : tensor<1x4x8xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst : tensor<1x4x8xf32>) outs(%0 : tensor<1x4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.negf %in : f32
      %3 = math.exp %2 : f32
      %4 = arith.addf %in, %3 : f32
      %5 = arith.divf %in, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<1x4x8xf32>
    return %1 : tensor<1x4x8xf32>
  }
}

// ==========================================
// Phase: Linalg Tiling
// ==========================================
#map = affine_map<(d0) -> (-d0 + 1, 8)>
#map1 = affine_map<(d0) -> (-d0 + 4, 8)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @test_silu() -> tensor<1x4x8xf32> {
    %c8 = arith.constant 8 : index
    %c8_0 = arith.constant 8 : index
    %c8_1 = arith.constant 8 : index
    %cst = arith.constant 0.451862752 : f32
    %0 = tensor.empty() : tensor<1x4x8xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_2 = arith.constant 8 : index
    %c0_3 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8_4 = arith.constant 8 : index
    %c0_5 = arith.constant 0 : index
    %c8_6 = arith.constant 8 : index
    %c8_7 = arith.constant 8 : index
    %1 = scf.for %arg0 = %c0 to %c1 step %c8_2 iter_args(%arg1 = %0) -> (tensor<1x4x8xf32>) {
      %2 = scf.for %arg2 = %c0_3 to %c4 step %c8_4 iter_args(%arg3 = %arg1) -> (tensor<1x4x8xf32>) {
        %3 = scf.for %arg4 = %c0_5 to %c8_6 step %c8_7 iter_args(%arg5 = %arg3) -> (tensor<1x4x8xf32>) {
          %4 = affine.min #map(%arg0)
          %5 = affine.min #map1(%arg2)
          %extracted_slice = tensor.extract_slice %arg5[%arg0, %arg2, %arg4] [%4, %5, 8] [1, 1, 1] : tensor<1x4x8xf32> to tensor<?x?x8xf32>
          %6 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel", "parallel"]} outs(%extracted_slice : tensor<?x?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst : f32
          } -> tensor<?x?x8xf32>
          %inserted_slice = tensor.insert_slice %6 into %arg5[%arg0, %arg2, %arg4] [%4, %5, 8] [1, 1, 1] : tensor<?x?x8xf32> into tensor<1x4x8xf32>
          scf.yield %inserted_slice : tensor<1x4x8xf32>
        }
        scf.yield %3 : tensor<1x4x8xf32>
      }
      scf.yield %2 : tensor<1x4x8xf32>
    }
    return %1 : tensor<1x4x8xf32>
  }
}

// ==========================================
// Phase: Bufferization
// ==========================================
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @test_silu() -> memref<1x4x8xf32> {
    %cst = arith.constant 0.451862752 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%alloc : memref<1x4x8xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    return %alloc : memref<1x4x8xf32>
  }
}

// ==========================================
// Phase: linalg to scf
// ==========================================
module {
  func.func @test_silu() -> memref<1x4x8xf32> {
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.451862752 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    scf.for %arg0 = %c0 to %c1 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        scf.for %arg2 = %c0 to %c8 step %c1 {
          memref.store %cst, %alloc[%arg0, %arg1, %arg2] : memref<1x4x8xf32>
        }
      }
    }
    return %alloc : memref<1x4x8xf32>
  }
}

// ==========================================
// Phase: lower to llvm
// ==========================================
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @test_silu() -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> {
    %0 = llvm.mlir.constant(8 : index) : i64
    %1 = llvm.mlir.constant(4 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(0.451862752 : f32) : f32
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.mlir.constant(4 : index) : i64
    %7 = llvm.mlir.constant(8 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(32 : index) : i64
    %10 = llvm.mlir.constant(32 : index) : i64
    %11 = llvm.mlir.zero : !llvm.ptr
    %12 = llvm.getelementptr %11[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %13 = llvm.ptrtoint %12 : !llvm.ptr to i64
    %14 = llvm.mlir.constant(64 : index) : i64
    %15 = llvm.add %13, %14 : i64
    %16 = llvm.call @malloc(%15) : (i64) -> !llvm.ptr
    %17 = llvm.ptrtoint %16 : !llvm.ptr to i64
    %18 = llvm.mlir.constant(1 : index) : i64
    %19 = llvm.sub %14, %18 : i64
    %20 = llvm.add %17, %19 : i64
    %21 = llvm.urem %20, %14  : i64
    %22 = llvm.sub %20, %21 : i64
    %23 = llvm.inttoptr %22 : i64 to !llvm.ptr
    %24 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %25 = llvm.insertvalue %16, %24[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %26 = llvm.insertvalue %23, %25[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %27 = llvm.mlir.constant(0 : index) : i64
    %28 = llvm.insertvalue %27, %26[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %29 = llvm.insertvalue %5, %28[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %30 = llvm.insertvalue %6, %29[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %31 = llvm.insertvalue %7, %30[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %32 = llvm.insertvalue %9, %31[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %33 = llvm.insertvalue %7, %32[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %34 = llvm.insertvalue %8, %33[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.br ^bb1(%3 : i64)
  ^bb1(%35: i64):  // 2 preds: ^bb0, ^bb8
    %36 = llvm.icmp "slt" %35, %2 : i64
    llvm.cond_br %36, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%3 : i64)
  ^bb3(%37: i64):  // 2 preds: ^bb2, ^bb7
    %38 = llvm.icmp "slt" %37, %1 : i64
    llvm.cond_br %38, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%3 : i64)
  ^bb5(%39: i64):  // 2 preds: ^bb4, ^bb6
    %40 = llvm.icmp "slt" %39, %0 : i64
    llvm.cond_br %40, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %41 = llvm.extractvalue %34[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %42 = llvm.mlir.constant(32 : index) : i64
    %43 = llvm.mul %35, %42 : i64
    %44 = llvm.mlir.constant(8 : index) : i64
    %45 = llvm.mul %37, %44 : i64
    %46 = llvm.add %43, %45 : i64
    %47 = llvm.add %46, %39 : i64
    %48 = llvm.getelementptr %41[%47] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %4, %48 : f32, !llvm.ptr
    %49 = llvm.add %39, %2 : i64
    llvm.br ^bb5(%49 : i64)
  ^bb7:  // pred: ^bb5
    %50 = llvm.add %37, %2 : i64
    llvm.br ^bb3(%50 : i64)
  ^bb8:  // pred: ^bb3
    %51 = llvm.add %35, %2 : i64
    llvm.br ^bb1(%51 : i64)
  ^bb9:  // pred: ^bb1
    llvm.return %34 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  }
}
