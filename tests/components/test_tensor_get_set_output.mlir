// Original IR loaded from file
module {
  cherry.func @test_tensor_get_set() -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.constant(0 : i64) : i64
    %1 = cherry.constant(1 : i64) : i64
    %2 = cherry.constant(2 : i64) : i64
    %3 = cherry.constant(3.211000e+03 : f32) : f32
    %4 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %5 = cherry.tensor_set %4[%0, %1, %2], %3 : (!cherry.cherry_tensor<[1x4x8xf32]>, i64, i64, i64, f32) -> !cherry.cherry_tensor<[1x4x8xf32]>
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
    %3 = cherry.constant(3.211000e+03 : f32) : f32
    %4 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %5 = cherry.tensor_set %4[%0, %1, %2], %3 : (!cherry.cherry_tensor<[1x4x8xf32]>, i64, i64, i64, f32) -> !cherry.cherry_tensor<[1x4x8xf32]>
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
    %3 = cherry.constant(3.211000e+03 : f32) : f32
    %4 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %5 = cherry.tensor_set %4[%0, %1, %2], %3 : (!cherry.cherry_tensor<[1x4x8xf32]>, i64, i64, i64, f32) -> !cherry.cherry_tensor<[1x4x8xf32]>
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
    %3 = cherry.constant(3.211000e+03 : f32) : f32
    %4 = cherry.create_tensor dense<5.000000e-01> : tensor<1x4x8xf32> -> !cherry.cherry_tensor<[1x4x8xf32]>
    %5 = cherry.tensor_set %4[%0, %1, %2], %3 : (!cherry.cherry_tensor<[1x4x8xf32]>, i64, i64, i64, f32) -> !cherry.cherry_tensor<[1x4x8xf32]>
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
    %cst = arith.constant 3.211000e+03 : f32
    %cst_0 = arith.constant dense<5.000000e-01> : tensor<1x4x8xf32>
    %0 = arith.index_cast %c0_i64 : i64 to index
    %1 = arith.index_cast %c1_i64 : i64 to index
    %2 = arith.index_cast %c2_i64 : i64 to index
    %inserted = tensor.insert %cst into %cst_0[%0, %1, %2] : tensor<1x4x8xf32>
    return %inserted : tensor<1x4x8xf32>
  }
}

// ==========================================
// Phase: Linalg Tiling
// ==========================================
module {
  func.func @test_tensor_get_set() -> tensor<1x4x8xf32> {
    %cst = arith.constant 3.211000e+03 : f32
    %cst_0 = arith.constant dense<5.000000e-01> : tensor<1x4x8xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %inserted = tensor.insert %cst into %cst_0[%c0, %c1, %c2] : tensor<1x4x8xf32>
    return %inserted : tensor<1x4x8xf32>
  }
}

// ==========================================
// Phase: Bufferization
// ==========================================
module {
  memref.global "private" constant @__constant_1x4x8xf32 : memref<1x4x8xf32> = dense<5.000000e-01> {alignment = 64 : i64}
  func.func @test_tensor_get_set() -> memref<1x4x8xf32> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 3.211000e+03 : f32
    %0 = memref.get_global @__constant_1x4x8xf32 : memref<1x4x8xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    memref.copy %0, %alloc : memref<1x4x8xf32> to memref<1x4x8xf32>
    memref.store %cst, %alloc[%c0, %c1, %c2] : memref<1x4x8xf32>
    return %alloc : memref<1x4x8xf32>
  }
}

// ==========================================
// Phase: linalg to scf
// ==========================================
module {
  memref.global "private" constant @__constant_1x4x8xf32 : memref<1x4x8xf32> = dense<5.000000e-01> {alignment = 64 : i64}
  func.func @test_tensor_get_set() -> memref<1x4x8xf32> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 3.211000e+03 : f32
    %0 = memref.get_global @__constant_1x4x8xf32 : memref<1x4x8xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    memref.copy %0, %alloc : memref<1x4x8xf32> to memref<1x4x8xf32>
    memref.store %cst, %alloc[%c0, %c1, %c2] : memref<1x4x8xf32>
    return %alloc : memref<1x4x8xf32>
  }
}

// ==========================================
// Phase: lower to llvm
// ==========================================
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_1x4x8xf32(dense<5.000000e-01> : tensor<1x4x8xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x array<4 x array<8 x f32>>>
  llvm.func @test_tensor_get_set() -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> {
    %0 = llvm.mlir.constant(2 : index) : i64
    %1 = llvm.mlir.constant(1 : index) : i64
    %2 = llvm.mlir.constant(0 : index) : i64
    %3 = llvm.mlir.constant(3.211000e+03 : f32) : f32
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(4 : index) : i64
    %6 = llvm.mlir.constant(8 : index) : i64
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.mlir.constant(32 : index) : i64
    %9 = llvm.mlir.constant(32 : index) : i64
    %10 = llvm.mlir.zero : !llvm.ptr
    %11 = llvm.getelementptr %10[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %12 = llvm.ptrtoint %11 : !llvm.ptr to i64
    %13 = llvm.mlir.addressof @__constant_1x4x8xf32 : !llvm.ptr
    %14 = llvm.getelementptr %13[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<4 x array<8 x f32>>>
    %15 = llvm.mlir.constant(3735928559 : index) : i64
    %16 = llvm.inttoptr %15 : i64 to !llvm.ptr
    %17 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %18 = llvm.insertvalue %16, %17[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %19 = llvm.insertvalue %14, %18[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %20 = llvm.mlir.constant(0 : index) : i64
    %21 = llvm.insertvalue %20, %19[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %22 = llvm.insertvalue %4, %21[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %23 = llvm.insertvalue %5, %22[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %24 = llvm.insertvalue %6, %23[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %25 = llvm.insertvalue %8, %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %26 = llvm.insertvalue %6, %25[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %27 = llvm.insertvalue %7, %26[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %28 = llvm.mlir.constant(1 : index) : i64
    %29 = llvm.mlir.constant(4 : index) : i64
    %30 = llvm.mlir.constant(8 : index) : i64
    %31 = llvm.mlir.constant(1 : index) : i64
    %32 = llvm.mlir.constant(32 : index) : i64
    %33 = llvm.mlir.constant(32 : index) : i64
    %34 = llvm.mlir.zero : !llvm.ptr
    %35 = llvm.getelementptr %34[%33] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %36 = llvm.ptrtoint %35 : !llvm.ptr to i64
    %37 = llvm.mlir.constant(64 : index) : i64
    %38 = llvm.add %36, %37 : i64
    %39 = llvm.call @malloc(%38) : (i64) -> !llvm.ptr
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.mlir.constant(1 : index) : i64
    %42 = llvm.sub %37, %41 : i64
    %43 = llvm.add %40, %42 : i64
    %44 = llvm.urem %43, %37  : i64
    %45 = llvm.sub %43, %44 : i64
    %46 = llvm.inttoptr %45 : i64 to !llvm.ptr
    %47 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %48 = llvm.insertvalue %39, %47[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %49 = llvm.insertvalue %46, %48[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %50 = llvm.mlir.constant(0 : index) : i64
    %51 = llvm.insertvalue %50, %49[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %52 = llvm.insertvalue %28, %51[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %53 = llvm.insertvalue %29, %52[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %54 = llvm.insertvalue %30, %53[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %55 = llvm.insertvalue %32, %54[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %56 = llvm.insertvalue %30, %55[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %57 = llvm.insertvalue %31, %56[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %58 = llvm.mlir.constant(1 : index) : i64
    %59 = llvm.extractvalue %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %60 = llvm.mul %58, %59 : i64
    %61 = llvm.extractvalue %27[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %62 = llvm.mul %60, %61 : i64
    %63 = llvm.extractvalue %27[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %64 = llvm.mul %62, %63 : i64
    %65 = llvm.mlir.zero : !llvm.ptr
    %66 = llvm.getelementptr %65[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %67 = llvm.ptrtoint %66 : !llvm.ptr to i64
    %68 = llvm.mul %64, %67 : i64
    %69 = llvm.extractvalue %27[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %70 = llvm.extractvalue %27[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %71 = llvm.getelementptr %69[%70] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %72 = llvm.extractvalue %57[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %73 = llvm.extractvalue %57[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %74 = llvm.getelementptr %72[%73] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%74, %71, %68) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %75 = llvm.extractvalue %57[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %76 = llvm.mlir.constant(32 : index) : i64
    %77 = llvm.mul %2, %76 : i64
    %78 = llvm.mlir.constant(8 : index) : i64
    %79 = llvm.mul %1, %78 : i64
    %80 = llvm.add %77, %79 : i64
    %81 = llvm.add %80, %0 : i64
    %82 = llvm.getelementptr %75[%81] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %3, %82 : f32, !llvm.ptr
    llvm.return %57 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  }
}
