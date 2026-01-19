#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
module {
  func.func @square(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) -> memref<10x10xf32> {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %0 = affine.apply #map(%c10)[%c0, %c1]
    %1 = affine.apply #map(%c10)[%c0, %c1]
    gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %0, %arg9 = %1, %arg10 = %c1_0) threads(%arg5, %arg6, %arg7) in (%arg11 = %c1_0, %arg12 = %c1_0, %arg13 = %c1_0) {
      %2 = affine.apply #map1(%arg2)[%c1, %c0]
      %3 = affine.apply #map1(%arg3)[%c1, %c0]
      %4 = memref.load %arg0[%2, %3] : memref<10x10xf32>
      %5 = arith.mulf %4, %4 : f32
      memref.store %5, %arg1[%2, %3] : memref<10x10xf32>
      gpu.terminator
    } {SCFToGPU_visited}
    return %arg1 : memref<10x10xf32>
  }
}

