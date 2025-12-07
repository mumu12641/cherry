// Original IR loaded from file
module {
  cherry.func @test_tensor_get_set() -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.constant(0 : i64) : i64
    %1 = cherry.constant(1 : i64) : i64
    %2 = cherry.constant(2 : i64) : i64
    %3 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %4 = cherry.tensor_get %3[%0, %1, %2] : (!cherry.cherry_tensor<[1x4x8xf32]>, i64, i64, i64) -> f32
    %5 = cherry.tensor_set %3[%0, %1, %2], %4 : (!cherry.cherry_tensor<[1x4x8xf32]>, i64, i64, i64, f32) -> !cherry.cherry_tensor<[1x4x8xf32]>
    cherry.return %5 : !cherry.cherry_tensor<[1x4x8xf32]>
  }
}

// ==========================================
// Phase: Inliner
// ==========================================
module {
  cherry.func @test_tensor_get_set() -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.constant(0 : i64) : i64
    %1 = cherry.constant(1 : i64) : i64
    %2 = cherry.constant(2 : i64) : i64
    %3 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %4 = cherry.tensor_get %3[%0, %1, %2] : (!cherry.cherry_tensor<[1x4x8xf32]>, i64, i64, i64) -> f32
    %5 = cherry.tensor_set %3[%0, %1, %2], %4 : (!cherry.cherry_tensor<[1x4x8xf32]>, i64, i64, i64, f32) -> !cherry.cherry_tensor<[1x4x8xf32]>
    cherry.return %5 : !cherry.cherry_tensor<[1x4x8xf32]>
  }
}

// ==========================================
// Phase: Shape Inference
// ==========================================
module {
  cherry.func @test_tensor_get_set() -> !cherry.cherry_tensor<[1x4x8xf32]> {
    %0 = cherry.constant(0 : i64) : i64
    %1 = cherry.constant(1 : i64) : i64
    %2 = cherry.constant(2 : i64) : i64
    %3 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %4 = cherry.tensor_get %3[%0, %1, %2] : (!cherry.cherry_tensor<[1x4x8xf32]>, i64, i64, i64) -> f32
    %5 = cherry.tensor_set %3[%0, %1, %2], %4 : (!cherry.cherry_tensor<[1x4x8xf32]>, i64, i64, i64, f32) -> !cherry.cherry_tensor<[1x4x8xf32]>
    cherry.return %5 : !cherry.cherry_tensor<[1x4x8xf32]>
  }
}

// ==========================================
// Phase: Canonicalizer
// ==========================================
module {
  cherry.func @test_tensor_get_set() -> !cherry.cherry_tensor<[1x4x8xf32]> {
    %0 = cherry.constant(0 : i64) : i64
    %1 = cherry.constant(1 : i64) : i64
    %2 = cherry.constant(2 : i64) : i64
    %3 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %4 = cherry.tensor_get %3[%0, %1, %2] : (!cherry.cherry_tensor<[1x4x8xf32]>, i64, i64, i64) -> f32
    %5 = cherry.tensor_set %3[%0, %1, %2], %4 : (!cherry.cherry_tensor<[1x4x8xf32]>, i64, i64, i64, f32) -> !cherry.cherry_tensor<[1x4x8xf32]>
    cherry.return %5 : !cherry.cherry_tensor<[1x4x8xf32]>
  }
}

// ==========================================
// Phase: Convert to Linalg
// ==========================================
module {
  func.func @test_tensor_get_set() -> tensor<1x4x8xf32> {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %cst = arith.constant dense<5.000000e-01> : tensor<1x4x8xf32>
    %0 = arith.index_cast %c0_i64 : i64 to index
    %1 = arith.index_cast %c1_i64 : i64 to index
    %2 = arith.index_cast %c2_i64 : i64 to index
    %extracted = tensor.extract %cst[%0, %1, %2] : tensor<1x4x8xf32>
    %3 = arith.index_cast %c0_i64 : i64 to index
    %4 = arith.index_cast %c1_i64 : i64 to index
    %5 = arith.index_cast %c2_i64 : i64 to index
    %inserted = tensor.insert %extracted into %cst[%3, %4, %5] : tensor<1x4x8xf32>
    return %inserted : tensor<1x4x8xf32>
  }
}

// ==========================================
// Phase: Linalg Tiling
// ==========================================
module {
  func.func @test_tensor_get_set() -> tensor<1x4x8xf32> {
    %cst = arith.constant dense<5.000000e-01> : tensor<1x4x8xf32>
    return %cst : tensor<1x4x8xf32>
  }
}

// ==========================================
// Phase: Bufferization
// ==========================================
module {
  memref.global "private" constant @__constant_1x4x8xf32 : memref<1x4x8xf32> = dense<5.000000e-01> {alignment = 64 : i64}
  func.func @test_tensor_get_set() -> memref<1x4x8xf32> {
    %0 = memref.get_global @__constant_1x4x8xf32 : memref<1x4x8xf32>
    return %0 : memref<1x4x8xf32>
  }
}

// ==========================================
// Phase: linalg to scf
// ==========================================
module {
  memref.global "private" constant @__constant_1x4x8xf32 : memref<1x4x8xf32> = dense<5.000000e-01> {alignment = 64 : i64}
  func.func @test_tensor_get_set() -> memref<1x4x8xf32> {
    %0 = memref.get_global @__constant_1x4x8xf32 : memref<1x4x8xf32>
    return %0 : memref<1x4x8xf32>
  }
}

// ==========================================
// Phase: lower to llvm
// ==========================================
module {
  llvm.mlir.global private constant @__constant_1x4x8xf32(dense<5.000000e-01> : tensor<1x4x8xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x array<4 x array<8 x f32>>>
  llvm.func @test_tensor_get_set() -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(4 : index) : i64
    %2 = llvm.mlir.constant(8 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(32 : index) : i64
    %5 = llvm.mlir.constant(32 : index) : i64
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = llvm.getelementptr %6[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %8 = llvm.ptrtoint %7 : !llvm.ptr to i64
    %9 = llvm.mlir.addressof @__constant_1x4x8xf32 : !llvm.ptr
    %10 = llvm.getelementptr %9[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<4 x array<8 x f32>>>
    %11 = llvm.mlir.constant(3735928559 : index) : i64
    %12 = llvm.inttoptr %11 : i64 to !llvm.ptr
    %13 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %14 = llvm.insertvalue %12, %13[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %15 = llvm.insertvalue %10, %14[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.insertvalue %16, %15[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %18 = llvm.insertvalue %0, %17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.insertvalue %1, %18[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %20 = llvm.insertvalue %2, %19[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %21 = llvm.insertvalue %4, %20[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %22 = llvm.insertvalue %2, %21[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %23 = llvm.insertvalue %3, %22[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.return %23 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  }
}
