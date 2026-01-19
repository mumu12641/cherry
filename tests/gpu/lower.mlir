module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @square(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) -> memref<10x10xf32> {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %c10_1 = arith.constant 10 : index
    %c10_2 = arith.constant 10 : index
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %c10_1, %arg9 = %c10_2, %arg10 = %c1_0) threads(%arg5, %arg6, %arg7) in (%arg11 = %c1_0, %arg12 = %c1_0, %arg13 = %c1_0) {
      %0 = arith.muli %arg2, %c1 : index
      %1 = arith.addi %0, %c0 : index
      %2 = arith.muli %arg3, %c1 : index
      %3 = arith.addi %2, %c0 : index
      %4 = memref.load %arg0[%1, %3] : memref<10x10xf32>
      %5 = arith.mulf %4, %4 : f32
      memref.store %5, %arg1[%1, %3] : memref<10x10xf32>
      gpu.terminator
    } {SCFToGPU_visited}
    return %arg1 : memref<10x10xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<10x10xf32>
    %cast = memref.cast %alloc : memref<10x10xf32> to memref<*xf32>
    gpu.host_register %cast : memref<*xf32>
    %c1_0 = arith.constant 1 : index
    %c10_1 = arith.constant 10 : index
    %c10_2 = arith.constant 10 : index
    gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %c10_1, %arg7 = %c10_2, %arg8 = %c1_0) threads(%arg3, %arg4, %arg5) in (%arg9 = %c1_0, %arg10 = %c1_0, %arg11 = %c1_0) {
      %1 = arith.muli %arg0, %c1 : index
      %2 = arith.addi %1, %c0 : index
      %3 = arith.muli %arg1, %c1 : index
      %4 = arith.addi %3, %c0 : index
      memref.store %cst, %alloc[%2, %4] : memref<10x10xf32>
      gpu.terminator
    } {SCFToGPU_visited}
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<10x10xf32>
    %cast_4 = memref.cast %alloc_3 : memref<10x10xf32> to memref<*xf32>
    gpu.host_register %cast_4 : memref<*xf32>
    %0 = call @square(%alloc, %alloc_3) : (memref<10x10xf32>, memref<10x10xf32>) -> memref<10x10xf32>
    %cast_5 = memref.cast %0 : memref<10x10xf32> to memref<*xf32>
    call @printMemrefF32(%cast_5) : (memref<*xf32>) -> ()
    return
  }
}

