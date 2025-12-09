// Original IR loaded from file
module {
  cherry.func private @test_set_slice(%arg0: !cherry.cherry_tensor<[12x1024x768xf32]>, %arg1: !cherry.cherry_tensor<[1x1x768xf32]>, %arg2: i64, %arg3: i64) -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.tensor_set_slice %arg0[%arg2, %arg3], %arg1 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %0 : !cherry.cherry_tensor<[?xf32]>
  }
  cherry.func @host() -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<12x1024x768xf32> -> !cherry.cherry_tensor<[12x1024x768xf32]>
    %1 = cherry.create_tensor dense<3.000000e-01> : tensor<1x1x768xf32> -> !cherry.cherry_tensor<[1x1x768xf32]>
    %2 = cherry.constant(2 : i64) : i64
    %3 = cherry.constant(512 : i64) : i64
    %4 = cherry.call @test_set_slice(%0, %1, %2, %3) : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %4 : !cherry.cherry_tensor<[?xf32]>
  }
}

// ==========================================
// Phase: Inliner
// ==========================================
module {
  cherry.func @host() -> !cherry.cherry_tensor<[?xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<12x1024x768xf32> -> !cherry.cherry_tensor<[12x1024x768xf32]>
    %1 = cherry.create_tensor dense<3.000000e-01> : tensor<1x1x768xf32> -> !cherry.cherry_tensor<[1x1x768xf32]>
    %2 = cherry.constant(2 : i64) : i64
    %3 = cherry.constant(512 : i64) : i64
    %4 = cherry.tensor_set_slice %0[%2, %3], %1 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[?xf32]>
    cherry.return %4 : !cherry.cherry_tensor<[?xf32]>
  }
}

// ==========================================
// Phase: Shape Inference
// ==========================================
module {
  cherry.func @host() -> !cherry.cherry_tensor<[12x1024x768xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<12x1024x768xf32> -> !cherry.cherry_tensor<[12x1024x768xf32]>
    %1 = cherry.create_tensor dense<3.000000e-01> : tensor<1x1x768xf32> -> !cherry.cherry_tensor<[1x1x768xf32]>
    %2 = cherry.constant(2 : i64) : i64
    %3 = cherry.constant(512 : i64) : i64
    %4 = cherry.tensor_set_slice %0[%2, %3], %1 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[12x1024x768xf32]>
    cherry.return %4 : !cherry.cherry_tensor<[12x1024x768xf32]>
  }
}

// ==========================================
// Phase: Canonicalizer
// ==========================================
module {
  cherry.func @host() -> !cherry.cherry_tensor<[12x1024x768xf32]> {
    %0 = cherry.create_tensor dense<5.000000e-01> : tensor<12x1024x768xf32> -> !cherry.cherry_tensor<[12x1024x768xf32]>
    %1 = cherry.create_tensor dense<3.000000e-01> : tensor<1x1x768xf32> -> !cherry.cherry_tensor<[1x1x768xf32]>
    %2 = cherry.constant(2 : i64) : i64
    %3 = cherry.constant(512 : i64) : i64
    %4 = cherry.tensor_set_slice %0[%2, %3], %1 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[12x1024x768xf32]>
    cherry.return %4 : !cherry.cherry_tensor<[12x1024x768xf32]>
  }
}

// ==========================================
// Phase: Convert to Linalg
// ==========================================
module {
  func.func @host() -> tensor<12x1024x768xf32> {
    %cst = arith.constant dense<5.000000e-01> : tensor<12x1024x768xf32>
    %cst_0 = arith.constant dense<3.000000e-01> : tensor<1x1x768xf32>
    %c2_i64 = arith.constant 2 : i64
    %c512_i64 = arith.constant 512 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %c2_i64 : i64 to index
    %1 = arith.index_cast %c512_i64 : i64 to index
    %inserted_slice = tensor.insert_slice %cst_0 into %cst[%0, %1, 0] [1, 1, 768] [1, 1, 1] : tensor<1x1x768xf32> into tensor<12x1024x768xf32>
    return %inserted_slice : tensor<12x1024x768xf32>
  }
}

// ==========================================
// Phase: Linalg Tiling
// ==========================================
module {
  func.func @host() -> tensor<12x1024x768xf32> {
    %cst = arith.constant dense<5.000000e-01> : tensor<12x1024x768xf32>
    %cst_0 = arith.constant dense<3.000000e-01> : tensor<1x1x768xf32>
    %c2 = arith.constant 2 : index
    %c512 = arith.constant 512 : index
    %inserted_slice = tensor.insert_slice %cst_0 into %cst[%c2, %c512, 0] [1, 1, 768] [1, 1, 1] : tensor<1x1x768xf32> into tensor<12x1024x768xf32>
    return %inserted_slice : tensor<12x1024x768xf32>
  }
}

// ==========================================
// Phase: Bufferization
// ==========================================
module {
  memref.global "private" constant @__constant_1x1x768xf32 : memref<1x1x768xf32> = dense<3.000000e-01> {alignment = 64 : i64}
  memref.global "private" constant @__constant_12x1024x768xf32 : memref<12x1024x768xf32> = dense<5.000000e-01> {alignment = 64 : i64}
  func.func @host() -> memref<12x1024x768xf32> {
    %0 = memref.get_global @__constant_12x1024x768xf32 : memref<12x1024x768xf32>
    %1 = memref.get_global @__constant_1x1x768xf32 : memref<1x1x768xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
    memref.copy %0, %alloc : memref<12x1024x768xf32> to memref<12x1024x768xf32>
    %subview = memref.subview %alloc[2, 512, 0] [1, 1, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: 1966080>>
    memref.copy %1, %subview : memref<1x1x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: 1966080>>
    return %alloc : memref<12x1024x768xf32>
  }
}

// ==========================================
// Phase: linalg to scf
// ==========================================
module {
  memref.global "private" constant @__constant_1x1x768xf32 : memref<1x1x768xf32> = dense<3.000000e-01> {alignment = 64 : i64}
  memref.global "private" constant @__constant_12x1024x768xf32 : memref<12x1024x768xf32> = dense<5.000000e-01> {alignment = 64 : i64}
  func.func @host() -> memref<12x1024x768xf32> {
    %0 = memref.get_global @__constant_12x1024x768xf32 : memref<12x1024x768xf32>
    %1 = memref.get_global @__constant_1x1x768xf32 : memref<1x1x768xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
    memref.copy %0, %alloc : memref<12x1024x768xf32> to memref<12x1024x768xf32>
    %subview = memref.subview %alloc[2, 512, 0] [1, 1, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: 1966080>>
    memref.copy %1, %subview : memref<1x1x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: 1966080>>
    return %alloc : memref<12x1024x768xf32>
  }
}

// ==========================================
// Phase: lower to llvm
// ==========================================
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_1x1x768xf32(dense<3.000000e-01> : tensor<1x1x768xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x array<1 x array<768 x f32>>>
  llvm.mlir.global private constant @__constant_12x1024x768xf32(dense<5.000000e-01> : tensor<12x1024x768xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<12 x array<1024 x array<768 x f32>>>
  llvm.func @host() -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> {
    %0 = llvm.mlir.constant(12 : index) : i64
    %1 = llvm.mlir.constant(1024 : index) : i64
    %2 = llvm.mlir.constant(768 : index) : i64
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(786432 : index) : i64
    %5 = llvm.mlir.constant(9437184 : index) : i64
    %6 = llvm.mlir.zero : !llvm.ptr
    %7 = llvm.getelementptr %6[%5] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %8 = llvm.ptrtoint %7 : !llvm.ptr to i64
    %9 = llvm.mlir.addressof @__constant_12x1024x768xf32 : !llvm.ptr
    %10 = llvm.getelementptr %9[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<12 x array<1024 x array<768 x f32>>>
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
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.mlir.constant(768 : index) : i64
    %27 = llvm.mlir.constant(1 : index) : i64
    %28 = llvm.mlir.constant(768 : index) : i64
    %29 = llvm.mlir.constant(768 : index) : i64
    %30 = llvm.mlir.zero : !llvm.ptr
    %31 = llvm.getelementptr %30[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %32 = llvm.ptrtoint %31 : !llvm.ptr to i64
    %33 = llvm.mlir.addressof @__constant_1x1x768xf32 : !llvm.ptr
    %34 = llvm.getelementptr %33[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<1 x array<768 x f32>>>
    %35 = llvm.mlir.constant(3735928559 : index) : i64
    %36 = llvm.inttoptr %35 : i64 to !llvm.ptr
    %37 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %38 = llvm.insertvalue %36, %37[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %39 = llvm.insertvalue %34, %38[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %40 = llvm.mlir.constant(0 : index) : i64
    %41 = llvm.insertvalue %40, %39[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %42 = llvm.insertvalue %24, %41[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %43 = llvm.insertvalue %25, %42[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %44 = llvm.insertvalue %26, %43[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %45 = llvm.insertvalue %28, %44[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %46 = llvm.insertvalue %26, %45[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %47 = llvm.insertvalue %27, %46[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %48 = llvm.mlir.constant(12 : index) : i64
    %49 = llvm.mlir.constant(1024 : index) : i64
    %50 = llvm.mlir.constant(768 : index) : i64
    %51 = llvm.mlir.constant(1 : index) : i64
    %52 = llvm.mlir.constant(786432 : index) : i64
    %53 = llvm.mlir.constant(9437184 : index) : i64
    %54 = llvm.mlir.zero : !llvm.ptr
    %55 = llvm.getelementptr %54[%53] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %56 = llvm.ptrtoint %55 : !llvm.ptr to i64
    %57 = llvm.mlir.constant(64 : index) : i64
    %58 = llvm.add %56, %57 : i64
    %59 = llvm.call @malloc(%58) : (i64) -> !llvm.ptr
    %60 = llvm.ptrtoint %59 : !llvm.ptr to i64
    %61 = llvm.mlir.constant(1 : index) : i64
    %62 = llvm.sub %57, %61 : i64
    %63 = llvm.add %60, %62 : i64
    %64 = llvm.urem %63, %57  : i64
    %65 = llvm.sub %63, %64 : i64
    %66 = llvm.inttoptr %65 : i64 to !llvm.ptr
    %67 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %68 = llvm.insertvalue %59, %67[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %69 = llvm.insertvalue %66, %68[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %70 = llvm.mlir.constant(0 : index) : i64
    %71 = llvm.insertvalue %70, %69[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %72 = llvm.insertvalue %48, %71[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %73 = llvm.insertvalue %49, %72[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %74 = llvm.insertvalue %50, %73[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %75 = llvm.insertvalue %52, %74[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %76 = llvm.insertvalue %50, %75[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %77 = llvm.insertvalue %51, %76[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %78 = builtin.unrealized_conversion_cast %77 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<12x1024x768xf32>
    %79 = llvm.mlir.constant(1 : index) : i64
    %80 = llvm.extractvalue %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %81 = llvm.mul %79, %80 : i64
    %82 = llvm.extractvalue %23[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %83 = llvm.mul %81, %82 : i64
    %84 = llvm.extractvalue %23[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %85 = llvm.mul %83, %84 : i64
    %86 = llvm.mlir.zero : !llvm.ptr
    %87 = llvm.getelementptr %86[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %88 = llvm.ptrtoint %87 : !llvm.ptr to i64
    %89 = llvm.mul %85, %88 : i64
    %90 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %91 = llvm.extractvalue %23[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %92 = llvm.getelementptr %90[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %93 = llvm.extractvalue %77[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %94 = llvm.extractvalue %77[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %95 = llvm.getelementptr %93[%94] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%95, %92, %89) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %subview = memref.subview %78[2, 512, 0] [1, 1, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: 1966080>>
    %96 = builtin.unrealized_conversion_cast %subview : memref<1x1x768xf32, strided<[786432, 768, 1], offset: 1966080>> to !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %97 = llvm.mlir.constant(1 : index) : i64
    %98 = llvm.extractvalue %47[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %99 = llvm.mul %97, %98 : i64
    %100 = llvm.extractvalue %47[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %101 = llvm.mul %99, %100 : i64
    %102 = llvm.extractvalue %47[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %103 = llvm.mul %101, %102 : i64
    %104 = llvm.mlir.zero : !llvm.ptr
    %105 = llvm.getelementptr %104[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %106 = llvm.ptrtoint %105 : !llvm.ptr to i64
    %107 = llvm.mul %103, %106 : i64
    %108 = llvm.extractvalue %47[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %109 = llvm.extractvalue %47[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %110 = llvm.getelementptr %108[%109] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %111 = llvm.extractvalue %96[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %112 = llvm.extractvalue %96[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %113 = llvm.getelementptr %111[%112] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%113, %110, %107) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.return %77 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  }
}
