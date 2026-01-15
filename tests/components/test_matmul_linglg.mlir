module {
  memref.global "private" constant @__constant_1x768xf32 : memref<1x768xf32> = dense<1.100000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_8x8xf32 : memref<8x8xf32> = dense<1.000000e-01> {alignment = 64 : i64}
  func.func private @printMemrefF32(memref<*xf32> {bufferization.access = "read"})
  func.func @host() {
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c768 = arith.constant 768 : index
    %0 = memref.get_global @__constant_8x8xf32 : memref<8x8xf32>
    %1 = memref.get_global @__constant_1x768xf32 : memref<1x768xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
    scf.for %arg0 = %c0 to %c768 step %c8 {
      %subview = memref.subview %alloc[0, %arg0] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
      linalg.fill ins(%cst : f32) outs(%subview : memref<1x8xf32, strided<[768, 1], offset: ?>>)
    }
    scf.for %arg0 = %c0 to %c768 step %c8 {
      scf.for %arg1 = %c0 to %c768 step %c8 {
        %subview = memref.subview %1[0, %arg1] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        %subview_0 = memref.subview %alloc[0, %arg0] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        linalg.matmul ins(%subview, %0 : memref<1x8xf32, strided<[768, 1], offset: ?>>, memref<8x8xf32>) outs(%subview_0 : memref<1x8xf32, strided<[768, 1], offset: ?>>)
      }
    }
    %cast = memref.cast %alloc : memref<1x768xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
}