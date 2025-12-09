// Original IR loaded from file
module {
  cherry.func private @test_slice(%arg0: !cherry.cherry_tensor<[4x4xf32]>, %arg1: i64) -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.constant(0 : i64) : i64
    %1 = cherry.constant(2 : i64) : i64
    %2 = cherry.tensor_slice %arg0[%0, %1, %arg1, %1] : (!cherry.cherry_tensor<[4x4xf32]>, i64, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %2 : !cherry.cherry_tensor<[?xf32]>
  }
  cherry.func @host() -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<4x4xf32> -> !cherry.cherry_tensor<[4x4xf32]>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %1 = arith.index_cast %c0 : index to i64
    %2 = cherry.call @test_slice(%0, %1) : (!cherry.cherry_tensor<[4x4xf32]>, i64) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %2 : !cherry.cherry_tensor<[?xf32]>
  }
}

// ==========================================
// Phase: Inliner
// ==========================================
module {
  cherry.func @host() -> !cherry.cherry_tensor<[?xf32]> {
    %c0_i64 = arith.constant 0 : i64
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<4x4xf32> -> !cherry.cherry_tensor<[4x4xf32]>
    %1 = cherry.constant(0 : i64) : i64
    %2 = cherry.constant(2 : i64) : i64
    %3 = cherry.tensor_slice %0[%1, %2, %c0_i64, %2] : (!cherry.cherry_tensor<[4x4xf32]>, i64, i64, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %3 : !cherry.cherry_tensor<[?xf32]>
  }
}

// ==========================================
// Phase: Shape Inference
// ==========================================
module {
  cherry.func @host() -> !cherry.cherry_tensor<[2x2xf32]> {
    %c0_i64 = arith.constant 0 : i64
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<4x4xf32> -> !cherry.cherry_tensor<[4x4xf32]>
    %1 = cherry.constant(0 : i64) : i64
    %2 = cherry.constant(2 : i64) : i64
    %3 = cherry.tensor_slice %0[%1, %2, %c0_i64, %2] : (!cherry.cherry_tensor<[4x4xf32]>, i64, i64, i64, i64) -> !cherry.cherry_tensor<[2x2xf32]>
    cherry.return %3 : !cherry.cherry_tensor<[2x2xf32]>
  }
}

// ==========================================
// Phase: Canonicalizer
// ==========================================
module {
  cherry.func @host() -> !cherry.cherry_tensor<[2x2xf32]> {
    %c0_i64 = arith.constant 0 : i64
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<4x4xf32> -> !cherry.cherry_tensor<[4x4xf32]>
    %1 = cherry.constant(0 : i64) : i64
    %2 = cherry.constant(2 : i64) : i64
    %3 = cherry.tensor_slice %0[%1, %2, %c0_i64, %2] : (!cherry.cherry_tensor<[4x4xf32]>, i64, i64, i64, i64) -> !cherry.cherry_tensor<[2x2xf32]>
    cherry.return %3 : !cherry.cherry_tensor<[2x2xf32]>
  }
}

// ==========================================
// Phase: Convert to Linalg
// ==========================================
module {
  func.func @host() -> tensor<2x2xf32> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<5.000000e-01> : tensor<4x4xf32>
    %c0_i64_0 = arith.constant 0 : i64
    %c2_i64 = arith.constant 2 : i64
    %0 = arith.index_cast %c0_i64_0 : i64 to index
    %1 = arith.index_cast %c0_i64 : i64 to index
    %extracted_slice = tensor.extract_slice %cst[%0, %1] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
    return %extracted_slice : tensor<2x2xf32>
  }
}

// ==========================================
// Phase: Linalg Tiling
// ==========================================
module {
  func.func @host() -> tensor<2x2xf32> {
    %cst = arith.constant dense<5.000000e-01> : tensor<2x2xf32>
    return %cst : tensor<2x2xf32>
  }
}

// ==========================================
// Phase: Bufferization
// ==========================================
module {
  memref.global "private" constant @__constant_2x2xf32 : memref<2x2xf32> = dense<5.000000e-01> {alignment = 64 : i64}
  func.func @host() -> memref<2x2xf32> {
    %0 = memref.get_global @__constant_2x2xf32 : memref<2x2xf32>
    return %0 : memref<2x2xf32>
  }
}

// ==========================================
// Phase: linalg to scf
// ==========================================
module {
  memref.global "private" constant @__constant_2x2xf32 : memref<2x2xf32> = dense<5.000000e-01> {alignment = 64 : i64}
  func.func @host() -> memref<2x2xf32> {
    %0 = memref.get_global @__constant_2x2xf32 : memref<2x2xf32>
    return %0 : memref<2x2xf32>
  }
}

// ==========================================
// Phase: lower to llvm
// ==========================================
module {
  llvm.mlir.global private constant @__constant_2x2xf32(dense<5.000000e-01> : tensor<2x2xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<2 x array<2 x f32>>
  llvm.func @host() -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
    %0 = llvm.mlir.constant(2 : index) : i64
    %1 = llvm.mlir.constant(2 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(4 : index) : i64
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.getelementptr %4[%3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %6 = llvm.ptrtoint %5 : !llvm.ptr to i64
    %7 = llvm.mlir.addressof @__constant_2x2xf32 : !llvm.ptr
    %8 = llvm.getelementptr %7[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x array<2 x f32>>
    %9 = llvm.mlir.constant(3735928559 : index) : i64
    %10 = llvm.inttoptr %9 : i64 to !llvm.ptr
    %11 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.insertvalue %10, %11[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %8, %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.mlir.constant(0 : index) : i64
    %15 = llvm.insertvalue %14, %13[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %0, %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %1, %16[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %1, %17[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %2, %18[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.return %19 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
}
