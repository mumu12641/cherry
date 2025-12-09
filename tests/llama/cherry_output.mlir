// Original IR loaded from file
module {
  cherry.func private @llama_forward(%arg0: i64, %arg1: i64, %arg2: !cherry.cherry_tensor<[32x2048x128xf32]>, %arg3: !cherry.cherry_tensor<[32000x768xf32]>) -> !cherry.cherry_tensor<[1x32000xf32]> {
    %0 = cherry.constant(0 : i64) : i64
    %1 = cherry.constant(1 : i64) : i64
    %2 = cherry.constant(768 : i64) : i64
    %3 = cherry.tensor_slice %arg3[%arg0, %1, %0, %2] : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
    %4 = cherry.create_tensor dense<0.000000e+00> : tensor<1x32000xf32> -> !cherry.cherry_tensor<[1x32000xf32]>
    cherry.return %arg2 : !cherry.cherry_tensor<[32x2048x128xf32]>
  }
  cherry.func @host() -> !cherry.cherry_tensor<[1xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
    %1 = cherry.create_tensor dense<0.000000e+00> : tensor<32x2048x128xf32> -> !cherry.cherry_tensor<[32x2048x128xf32]>
    %2 = cherry.constant(1 : i64) : i64
    %3 = cherry.constant(0 : i64) : i64
    %4 = cherry.constant(10 : i64) : i64
    %5:3 = scf.while (%arg0 = %2, %arg1 = %3, %arg2 = %1) : (i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>) -> (i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>) {
      %7 = arith.cmpi slt, %arg1, %4 : i64
      scf.condition(%7) %arg0, %arg1, %arg2 : i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>
    } do {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: !cherry.cherry_tensor<[32x2048x128xf32]>):
      %7 = cherry.call @llama_forward(%arg0, %arg1, %arg2, %0) : (i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>, !cherry.cherry_tensor<[32000x768xf32]>) -> !cherry.cherry_tensor<[32x2048x128xf32]>
      %8 = cherry.constant(999 : i64) : i64
      %9 = cherry.constant(1 : i64) : i64
      %10 = arith.addi %arg1, %9 : i64
      scf.yield %8, %10, %7 : i64, i64, !cherry.cherry_tensor<[32x2048x128xf32]>
    }
    %6 = cherry.create_tensor dense<0.000000e+00> : tensor<1xf32> -> !cherry.cherry_tensor<[1xf32]>
    cherry.return %6 : !cherry.cherry_tensor<[1xf32]>
  }
}

// ==========================================
// Phase: Inliner
// ==========================================
module {
  cherry.func @host() -> !cherry.cherry_tensor<[1xf32]> {
    %0 = cherry.create_tensor dense<0.000000e+00> : tensor<1xf32> -> !cherry.cherry_tensor<[1xf32]>
    cherry.return %0 : !cherry.cherry_tensor<[1xf32]>
  }
}

// ==========================================
// Phase: Shape Inference
// ==========================================
module {
  cherry.func @host() -> !cherry.cherry_tensor<[1xf32]> {
    %0 = cherry.create_tensor dense<0.000000e+00> : tensor<1xf32> -> !cherry.cherry_tensor<[1xf32]>
    cherry.return %0 : !cherry.cherry_tensor<[1xf32]>
  }
}

// ==========================================
// Phase: Canonicalizer
// ==========================================
module {
  cherry.func @host() -> !cherry.cherry_tensor<[1xf32]> {
    %0 = cherry.create_tensor dense<0.000000e+00> : tensor<1xf32> -> !cherry.cherry_tensor<[1xf32]>
    cherry.return %0 : !cherry.cherry_tensor<[1xf32]>
  }
}

// ==========================================
// Phase: Convert to Linalg
// ==========================================
module {
  func.func @host() -> tensor<1xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<1xf32>
    return %cst : tensor<1xf32>
  }
}

// ==========================================
// Phase: Linalg Tiling
// ==========================================
module {
  func.func @host() -> tensor<1xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<1xf32>
    return %cst : tensor<1xf32>
  }
}

// ==========================================
// Phase: Bufferization
// ==========================================
module {
  memref.global "private" constant @__constant_1xf32 : memref<1xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  func.func @host() -> memref<1xf32> {
    %0 = memref.get_global @__constant_1xf32 : memref<1xf32>
    return %0 : memref<1xf32>
  }
}

// ==========================================
// Phase: linalg to scf
// ==========================================
module {
  memref.global "private" constant @__constant_1xf32 : memref<1xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  func.func @host() -> memref<1xf32> {
    %0 = memref.get_global @__constant_1xf32 : memref<1xf32>
    return %0 : memref<1xf32>
  }
}

// ==========================================
// Phase: lower to llvm
// ==========================================
module {
  llvm.mlir.global private constant @__constant_1xf32(dense<0.000000e+00> : tensor<1xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x f32>
  llvm.func @host() -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.getelementptr %2[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %4 = llvm.ptrtoint %3 : !llvm.ptr to i64
    %5 = llvm.mlir.addressof @__constant_1xf32 : !llvm.ptr
    %6 = llvm.getelementptr %5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x f32>
    %7 = llvm.mlir.constant(3735928559 : index) : i64
    %8 = llvm.inttoptr %7 : i64 to !llvm.ptr
    %9 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %10 = llvm.insertvalue %8, %9[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %6, %10[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.mlir.constant(0 : index) : i64
    %13 = llvm.insertvalue %12, %11[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.insertvalue %0, %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.insertvalue %1, %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.return %15 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  }
}
