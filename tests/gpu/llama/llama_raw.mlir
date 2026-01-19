#map = affine_map<(d0) -> ()>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<(d0, d1) -> (d1)>
#map5 = affine_map<(d0, d1) -> ()>
#map6 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map9 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map10 = affine_map<(d0, d1, d2) -> (d2)>
#map11 = affine_map<(d0, d1) -> (d1, d0)>
#map12 = affine_map<(d0, d1) -> (32, d0 - d1)>
#map13 = affine_map<(d0, d1) -> (128, d0 - d1)>
#map14 = affine_map<(d0) -> (-d0 + 64, 32)>
#map15 = affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>
module {
  memref.global "private" constant @__constant_49xi8 : memref<49xi8> = dense<[47, 104, 111, 109, 101, 47, 110, 120, 47, 121, 99, 121, 47, 112, 98, 47, 99, 104, 101, 114, 114, 121, 47, 116, 101, 115, 116, 115, 47, 108, 108, 97, 109, 97, 47, 116, 111, 107, 101, 110, 105, 122, 101, 114, 46, 98, 105, 110, 0]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_62xi8 : memref<62xi8> = dense<[47, 104, 111, 109, 101, 47, 110, 120, 47, 121, 99, 121, 47, 112, 98, 47, 99, 104, 101, 114, 114, 121, 47, 117, 116, 105, 108, 115, 47, 115, 116, 111, 114, 105, 101, 115, 49, 49, 48, 77, 47, 116, 111, 107, 101, 110, 95, 101, 109, 98, 101, 100, 100, 105, 110, 103, 115, 46, 98, 105, 110, 0]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_67xi8_0 : memref<67xi8> = dense<[47, 104, 111, 109, 101, 47, 110, 120, 47, 121, 99, 121, 47, 112, 98, 47, 99, 104, 101, 114, 114, 121, 47, 117, 116, 105, 108, 115, 47, 115, 116, 111, 114, 105, 101, 115, 49, 49, 48, 77, 47, 108, 97, 121, 101, 114, 115, 95, 114, 109, 115, 95, 97, 116, 116, 95, 119, 101, 105, 103, 104, 116, 46, 98, 105, 110, 0]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_55xi8_5 : memref<55xi8> = dense<[47, 104, 111, 109, 101, 47, 110, 120, 47, 121, 99, 121, 47, 112, 98, 47, 99, 104, 101, 114, 114, 121, 47, 117, 116, 105, 108, 115, 47, 115, 116, 111, 114, 105, 101, 115, 49, 49, 48, 77, 47, 108, 97, 121, 101, 114, 115, 95, 119, 113, 46, 98, 105, 110, 0]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_55xi8_4 : memref<55xi8> = dense<[47, 104, 111, 109, 101, 47, 110, 120, 47, 121, 99, 121, 47, 112, 98, 47, 99, 104, 101, 114, 114, 121, 47, 117, 116, 105, 108, 115, 47, 115, 116, 111, 114, 105, 101, 115, 49, 49, 48, 77, 47, 108, 97, 121, 101, 114, 115, 95, 119, 107, 46, 98, 105, 110, 0]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_55xi8_3 : memref<55xi8> = dense<[47, 104, 111, 109, 101, 47, 110, 120, 47, 121, 99, 121, 47, 112, 98, 47, 99, 104, 101, 114, 114, 121, 47, 117, 116, 105, 108, 115, 47, 115, 116, 111, 114, 105, 101, 115, 49, 49, 48, 77, 47, 108, 97, 121, 101, 114, 115, 95, 119, 118, 46, 98, 105, 110, 0]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_55xi8_2 : memref<55xi8> = dense<[47, 104, 111, 109, 101, 47, 110, 120, 47, 121, 99, 121, 47, 112, 98, 47, 99, 104, 101, 114, 114, 121, 47, 117, 116, 105, 108, 115, 47, 115, 116, 111, 114, 105, 101, 115, 49, 49, 48, 77, 47, 108, 97, 121, 101, 114, 115, 95, 119, 111, 46, 98, 105, 110, 0]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_67xi8 : memref<67xi8> = dense<[47, 104, 111, 109, 101, 47, 110, 120, 47, 121, 99, 121, 47, 112, 98, 47, 99, 104, 101, 114, 114, 121, 47, 117, 116, 105, 108, 115, 47, 115, 116, 111, 114, 105, 101, 115, 49, 49, 48, 77, 47, 108, 97, 121, 101, 114, 115, 95, 114, 109, 115, 95, 102, 102, 110, 95, 119, 101, 105, 103, 104, 116, 46, 98, 105, 110, 0]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_55xi8_1 : memref<55xi8> = dense<[47, 104, 111, 109, 101, 47, 110, 120, 47, 121, 99, 121, 47, 112, 98, 47, 99, 104, 101, 114, 114, 121, 47, 117, 116, 105, 108, 115, 47, 115, 116, 111, 114, 105, 101, 115, 49, 49, 48, 77, 47, 108, 97, 121, 101, 114, 115, 95, 119, 49, 46, 98, 105, 110, 0]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_55xi8_0 : memref<55xi8> = dense<[47, 104, 111, 109, 101, 47, 110, 120, 47, 121, 99, 121, 47, 112, 98, 47, 99, 104, 101, 114, 114, 121, 47, 117, 116, 105, 108, 115, 47, 115, 116, 111, 114, 105, 101, 115, 49, 49, 48, 77, 47, 108, 97, 121, 101, 114, 115, 95, 119, 50, 46, 98, 105, 110, 0]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_55xi8 : memref<55xi8> = dense<[47, 104, 111, 109, 101, 47, 110, 120, 47, 121, 99, 121, 47, 112, 98, 47, 99, 104, 101, 114, 114, 121, 47, 117, 116, 105, 108, 115, 47, 115, 116, 111, 114, 105, 101, 115, 49, 49, 48, 77, 47, 108, 97, 121, 101, 114, 115, 95, 119, 51, 46, 98, 105, 110, 0]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_60xi8 : memref<60xi8> = dense<[47, 104, 111, 109, 101, 47, 110, 120, 47, 121, 99, 121, 47, 112, 98, 47, 99, 104, 101, 114, 114, 121, 47, 117, 116, 105, 108, 115, 47, 115, 116, 111, 114, 105, 101, 115, 49, 49, 48, 77, 47, 102, 105, 110, 97, 108, 95, 114, 109, 115, 95, 110, 111, 114, 109, 46, 98, 105, 110, 0]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_57xi8 : memref<57xi8> = dense<[47, 104, 111, 109, 101, 47, 110, 120, 47, 121, 99, 121, 47, 112, 98, 47, 99, 104, 101, 114, 114, 121, 47, 117, 116, 105, 108, 115, 47, 115, 116, 111, 114, 105, 101, 115, 49, 49, 48, 77, 47, 111, 117, 116, 112, 117, 116, 95, 119, 99, 108, 115, 46, 98, 105, 110, 0]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_12x1024x768xf32 : memref<12x1024x768xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x12x64xf32 : memref<1x12x64xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3xi64_1 : memref<3xi64> = dense<[1, 12, 64]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_2xi64 : memref<2xi64> = dense<[1, 768]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3xi64_0 : memref<3xi64> = dense<[1, 1, 768]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3xi64 : memref<3xi64> = dense<[1, 1, 64]> {alignment = 64 : i64}
  func.func private @free_tokenizer()
  func.func private @end(i64)
  func.func private @decode(i64, i64)
  func.func private @start()
  func.func private @cherry_read_weight_2d_768_32000_f32(memref<?xi8, strided<[?], offset: ?>> {bufferization.access = "read"}, i64, i64) -> memref<768x32000xf32>
  func.func private @cherry_read_weight_1d_768_f32(memref<?xi8, strided<[?], offset: ?>> {bufferization.access = "read"}, i64) -> memref<768xf32>
  func.func private @cherry_read_weight_3d_12_2048_768_f32(memref<?xi8, strided<[?], offset: ?>> {bufferization.access = "read"}, i64, i64, i64) -> memref<12x2048x768xf32>
  func.func private @cherry_read_weight_3d_12_768_2048_f32(memref<?xi8, strided<[?], offset: ?>> {bufferization.access = "read"}, i64, i64, i64) -> memref<12x768x2048xf32>
  func.func private @cherry_read_weight_3d_12_768_768_f32(memref<?xi8, strided<[?], offset: ?>> {bufferization.access = "read"}, i64, i64, i64) -> memref<12x768x768xf32>
  func.func private @cherry_read_weight_2d_12_768_f32(memref<?xi8, strided<[?], offset: ?>> {bufferization.access = "read"}, i64, i64) -> memref<12x768xf32>
  func.func private @cherry_read_weight_2d_32000_768_f32(memref<?xi8, strided<[?], offset: ?>> {bufferization.access = "read"}, i64, i64) -> memref<32000x768xf32>
  func.func private @build_tokenizer(i64, memref<?xi8, strided<[?], offset: ?>>)
  func.func @host() {
    %c32000_i64 = arith.constant 32000 : i64
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %c0 = arith.constant 0 : index
    %c768_i64 = arith.constant 768 : i64
    %c12_i64 = arith.constant 12 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %cst = arith.constant 1.250000e-01 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 9.99999974E-6 : f32
    %cst_2 = arith.constant 1.000000e+04 : f32
    %cst_3 = arith.constant 6.400000e+01 : f32
    %cst_4 = arith.constant -2.000000e+00 : f32
    %cst_5 = arith.constant -1.000000e+09 : f32
    %cst_6 = arith.constant 0xFF800000 : f32
    %cst_7 = arith.constant 1.000000e+00 : f32
    %cst_8 = arith.constant 7.680000e+02 : f32
    %c32000 = arith.constant 32000 : index
    %c2048 = arith.constant 2048 : index
    %c1024 = arith.constant 1024 : index
    %c64 = arith.constant 64 : index
    %c768 = arith.constant 768 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %0 = memref.get_global @__constant_3xi64 : memref<3xi64>
    %1 = memref.get_global @__constant_3xi64_0 : memref<3xi64>
    %2 = memref.get_global @__constant_2xi64 : memref<2xi64>
    %3 = memref.get_global @__constant_3xi64_1 : memref<3xi64>
    %4 = memref.get_global @__constant_1x12x64xf32 : memref<1x12x64xf32>
    %5 = memref.get_global @__constant_12x1024x768xf32 : memref<12x1024x768xf32>
    %6 = memref.get_global @__constant_57xi8 : memref<57xi8>
    %7 = memref.get_global @__constant_60xi8 : memref<60xi8>
    %8 = memref.get_global @__constant_55xi8 : memref<55xi8>
    %9 = memref.get_global @__constant_55xi8_0 : memref<55xi8>
    %10 = memref.get_global @__constant_55xi8_1 : memref<55xi8>
    %11 = memref.get_global @__constant_67xi8 : memref<67xi8>
    %12 = memref.get_global @__constant_55xi8_2 : memref<55xi8>
    %13 = memref.get_global @__constant_55xi8_3 : memref<55xi8>
    %14 = memref.get_global @__constant_55xi8_4 : memref<55xi8>
    %15 = memref.get_global @__constant_55xi8_5 : memref<55xi8>
    %16 = memref.get_global @__constant_67xi8_0 : memref<67xi8>
    %17 = memref.get_global @__constant_62xi8 : memref<62xi8>
    %18 = memref.get_global @__constant_49xi8 : memref<49xi8>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<49xi8>
    memref.copy %18, %alloc : memref<49xi8> to memref<49xi8>
    %cast = memref.cast %alloc : memref<49xi8> to memref<?xi8, strided<[?], offset: ?>>
    call @build_tokenizer(%c32000_i64, %cast) : (i64, memref<?xi8, strided<[?], offset: ?>>) -> ()
    %cast_9 = memref.cast %17 : memref<62xi8> to memref<?xi8, strided<[?], offset: ?>>
    %19 = call @cherry_read_weight_2d_32000_768_f32(%cast_9, %c32000_i64, %c768_i64) : (memref<?xi8, strided<[?], offset: ?>>, i64, i64) -> memref<32000x768xf32>
    %cast_10 = memref.cast %16 : memref<67xi8> to memref<?xi8, strided<[?], offset: ?>>
    %20 = call @cherry_read_weight_2d_12_768_f32(%cast_10, %c12_i64, %c768_i64) : (memref<?xi8, strided<[?], offset: ?>>, i64, i64) -> memref<12x768xf32>
    %cast_11 = memref.cast %15 : memref<55xi8> to memref<?xi8, strided<[?], offset: ?>>
    %21 = call @cherry_read_weight_3d_12_768_768_f32(%cast_11, %c12_i64, %c768_i64, %c768_i64) : (memref<?xi8, strided<[?], offset: ?>>, i64, i64, i64) -> memref<12x768x768xf32>
    %cast_12 = memref.cast %14 : memref<55xi8> to memref<?xi8, strided<[?], offset: ?>>
    %22 = call @cherry_read_weight_3d_12_768_768_f32(%cast_12, %c12_i64, %c768_i64, %c768_i64) : (memref<?xi8, strided<[?], offset: ?>>, i64, i64, i64) -> memref<12x768x768xf32>
    %cast_13 = memref.cast %13 : memref<55xi8> to memref<?xi8, strided<[?], offset: ?>>
    %23 = call @cherry_read_weight_3d_12_768_768_f32(%cast_13, %c12_i64, %c768_i64, %c768_i64) : (memref<?xi8, strided<[?], offset: ?>>, i64, i64, i64) -> memref<12x768x768xf32>
    %cast_14 = memref.cast %12 : memref<55xi8> to memref<?xi8, strided<[?], offset: ?>>
    %24 = call @cherry_read_weight_3d_12_768_768_f32(%cast_14, %c12_i64, %c768_i64, %c768_i64) : (memref<?xi8, strided<[?], offset: ?>>, i64, i64, i64) -> memref<12x768x768xf32>
    %cast_15 = memref.cast %11 : memref<67xi8> to memref<?xi8, strided<[?], offset: ?>>
    %25 = call @cherry_read_weight_2d_12_768_f32(%cast_15, %c12_i64, %c768_i64) : (memref<?xi8, strided<[?], offset: ?>>, i64, i64) -> memref<12x768xf32>
    %cast_16 = memref.cast %10 : memref<55xi8> to memref<?xi8, strided<[?], offset: ?>>
    %26 = call @cherry_read_weight_3d_12_768_2048_f32(%cast_16, %c12_i64, %c768_i64, %c2048_i64) : (memref<?xi8, strided<[?], offset: ?>>, i64, i64, i64) -> memref<12x768x2048xf32>
    %cast_17 = memref.cast %9 : memref<55xi8> to memref<?xi8, strided<[?], offset: ?>>
    %27 = call @cherry_read_weight_3d_12_2048_768_f32(%cast_17, %c12_i64, %c2048_i64, %c768_i64) : (memref<?xi8, strided<[?], offset: ?>>, i64, i64, i64) -> memref<12x2048x768xf32>
    %cast_18 = memref.cast %8 : memref<55xi8> to memref<?xi8, strided<[?], offset: ?>>
    %28 = call @cherry_read_weight_3d_12_768_2048_f32(%cast_18, %c12_i64, %c768_i64, %c2048_i64) : (memref<?xi8, strided<[?], offset: ?>>, i64, i64, i64) -> memref<12x768x2048xf32>
    %cast_19 = memref.cast %7 : memref<60xi8> to memref<?xi8, strided<[?], offset: ?>>
    %29 = call @cherry_read_weight_1d_768_f32(%cast_19, %c768_i64) : (memref<?xi8, strided<[?], offset: ?>>, i64) -> memref<768xf32>
    %cast_20 = memref.cast %6 : memref<57xi8> to memref<?xi8, strided<[?], offset: ?>>
    %30 = call @cherry_read_weight_2d_768_32000_f32(%cast_20, %c768_i64, %c32000_i64) : (memref<?xi8, strided<[?], offset: ?>>, i64, i64) -> memref<768x32000xf32>
    call @start() : () -> ()
    %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
    memref.copy %5, %alloc_21 : memref<12x1024x768xf32> to memref<12x1024x768xf32>
    %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
    memref.copy %5, %alloc_22 : memref<12x1024x768xf32> to memref<12x1024x768xf32>
    %31:2 = scf.while (%arg0 = %c1_i64, %arg1 = %c0_i64) : (i64, i64) -> (i64, i64) {
      %32 = arith.cmpi slt, %arg1, %c128_i64 : i64
      scf.condition(%32) %arg0, %arg1 : i64, i64
    } do {
    ^bb0(%arg0: i64, %arg1: i64):
      %32 = arith.addi %arg1, %c1_i64 : i64
      %33 = arith.index_cast %arg0 : i64 to index
      %subview = memref.subview %19[%33, 0] [1, 768] [1, 1] : memref<32000x768xf32> to memref<1x768xf32, strided<[768, 1], offset: ?>>
      %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
      memref.copy %subview, %alloc_23 : memref<1x768xf32, strided<[768, 1], offset: ?>> to memref<1x768xf32>
      %34 = scf.for %arg2 = %c0 to %c12 step %c1 iter_args(%arg3 = %alloc_23) -> (memref<1x768xf32>) {
        %subview_30 = memref.subview %20[%arg2, 0] [1, 768] [1, 1] : memref<12x768xf32> to memref<768xf32, strided<[1], offset: ?>>
        %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
        linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%cst_0 : f32) outs(%alloc_31 : memref<1xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        }
        scf.for %arg4 = %c0 to %c768 step %c128 {
          %subview_99 = memref.subview %arg3[0, %arg4] [1, 128] [1, 1] : memref<1x768xf32> to memref<1x128xf32, strided<[768, 1], offset: ?>>
          scf.for %arg5 = %c0 to %c128 step %c32 {
            %subview_100 = memref.subview %subview_99[0, %arg5] [1, 32] [1, 1] : memref<1x128xf32, strided<[768, 1], offset: ?>> to memref<1x32xf32, strided<[768, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%subview_100 : memref<1x32xf32, strided<[768, 1], offset: ?>>) outs(%alloc_31 : memref<1xf32>) {
            ^bb0(%in: f32, %out: f32):
              %38 = arith.mulf %in, %in : f32
              %39 = arith.addf %out, %38 : f32
              linalg.yield %39 : f32
            }
          }
        }
        %alloc_32 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_31 : memref<1xf32>) outs(%alloc_32 : memref<1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %38 = arith.divf %in, %cst_8 : f32
          %39 = arith.addf %38, %cst_1 : f32
          %40 = math.rsqrt %39 : f32
          linalg.yield %40 : f32
        }
        %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg4 = %c0 to %c768 step %c32 {
          %subview_99 = memref.subview %arg3[0, %arg4] [1, 32] [1, 1] : memref<1x768xf32> to memref<1x32xf32, strided<[768, 1], offset: ?>>
          %subview_100 = memref.subview %subview_30[%arg4] [32] [1] : memref<768xf32, strided<[1], offset: ?>> to memref<32xf32, strided<[1], offset: ?>>
          %subview_101 = memref.subview %alloc_33[0, %arg4] [1, 32] [1, 1] : memref<1x768xf32> to memref<1x32xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map2, #map3, #map4, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_99, %alloc_32, %subview_100 : memref<1x32xf32, strided<[768, 1], offset: ?>>, memref<1xf32>, memref<32xf32, strided<[1], offset: ?>>) outs(%subview_101 : memref<1x32xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%in: f32, %in_102: f32, %in_103: f32, %out: f32):
            %38 = arith.mulf %in, %in_102 : f32
            %39 = arith.mulf %38, %in_103 : f32
            linalg.yield %39 : f32
          }
        }
        %subview_34 = memref.subview %21[%arg2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<12x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: ?>>
        %subview_35 = memref.subview %22[%arg2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<12x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: ?>>
        %subview_36 = memref.subview %23[%arg2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<12x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: ?>>
        %alloc_37 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg4 = %c0 to %c768 step %c32 {
          %subview_99 = memref.subview %alloc_37[0, %arg4] [1, 32] [1, 1] : memref<1x768xf32> to memref<1x32xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : f32) outs(%subview_99 : memref<1x32xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
        }
        %alloc_38 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        memref.copy %alloc_37, %alloc_38 : memref<1x768xf32> to memref<1x768xf32>
        scf.for %arg4 = %c0 to %c768 step %c128 {
          scf.for %arg5 = %c0 to %c768 step %c128 {
            %subview_99 = memref.subview %alloc_33[0, %arg5] [1, 128] [1, 1] : memref<1x768xf32> to memref<1x128xf32, strided<[768, 1], offset: ?>>
            %subview_100 = memref.subview %subview_34[%arg5, %arg4] [128, 128] [1, 1] : memref<768x768xf32, strided<[768, 1], offset: ?>> to memref<128x128xf32, strided<[768, 1], offset: ?>>
            %subview_101 = memref.subview %alloc_38[0, %arg4] [1, 128] [1, 1] : memref<1x768xf32> to memref<1x128xf32, strided<[768, 1], offset: ?>>
            scf.for %arg6 = %c0 to %c128 step %c32 {
              scf.for %arg7 = %c0 to %c128 step %c32 {
                %subview_102 = memref.subview %subview_99[0, %arg7] [1, 32] [1, 1] : memref<1x128xf32, strided<[768, 1], offset: ?>> to memref<1x32xf32, strided<[768, 1], offset: ?>>
                %subview_103 = memref.subview %subview_100[%arg7, %arg6] [32, 32] [1, 1] : memref<128x128xf32, strided<[768, 1], offset: ?>> to memref<32x32xf32, strided<[768, 1], offset: ?>>
                %subview_104 = memref.subview %subview_101[0, %arg6] [1, 32] [1, 1] : memref<1x128xf32, strided<[768, 1], offset: ?>> to memref<1x32xf32, strided<[768, 1], offset: ?>>
                linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_102, %subview_103 : memref<1x32xf32, strided<[768, 1], offset: ?>>, memref<32x32xf32, strided<[768, 1], offset: ?>>) outs(%subview_104 : memref<1x32xf32, strided<[768, 1], offset: ?>>) {
                ^bb0(%in: f32, %in_105: f32, %out: f32):
                  %38 = arith.mulf %in, %in_105 : f32
                  %39 = arith.addf %out, %38 : f32
                  linalg.yield %39 : f32
                }
              }
            }
          }
        }
        %alloc_39 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg4 = %c0 to %c768 step %c32 {
          %subview_99 = memref.subview %alloc_39[0, %arg4] [1, 32] [1, 1] : memref<1x768xf32> to memref<1x32xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : f32) outs(%subview_99 : memref<1x32xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
        }
        %alloc_40 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        memref.copy %alloc_39, %alloc_40 : memref<1x768xf32> to memref<1x768xf32>
        scf.for %arg4 = %c0 to %c768 step %c128 {
          scf.for %arg5 = %c0 to %c768 step %c128 {
            %subview_99 = memref.subview %alloc_33[0, %arg5] [1, 128] [1, 1] : memref<1x768xf32> to memref<1x128xf32, strided<[768, 1], offset: ?>>
            %subview_100 = memref.subview %subview_35[%arg5, %arg4] [128, 128] [1, 1] : memref<768x768xf32, strided<[768, 1], offset: ?>> to memref<128x128xf32, strided<[768, 1], offset: ?>>
            %subview_101 = memref.subview %alloc_40[0, %arg4] [1, 128] [1, 1] : memref<1x768xf32> to memref<1x128xf32, strided<[768, 1], offset: ?>>
            scf.for %arg6 = %c0 to %c128 step %c32 {
              scf.for %arg7 = %c0 to %c128 step %c32 {
                %subview_102 = memref.subview %subview_99[0, %arg7] [1, 32] [1, 1] : memref<1x128xf32, strided<[768, 1], offset: ?>> to memref<1x32xf32, strided<[768, 1], offset: ?>>
                %subview_103 = memref.subview %subview_100[%arg7, %arg6] [32, 32] [1, 1] : memref<128x128xf32, strided<[768, 1], offset: ?>> to memref<32x32xf32, strided<[768, 1], offset: ?>>
                %subview_104 = memref.subview %subview_101[0, %arg6] [1, 32] [1, 1] : memref<1x128xf32, strided<[768, 1], offset: ?>> to memref<1x32xf32, strided<[768, 1], offset: ?>>
                linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_102, %subview_103 : memref<1x32xf32, strided<[768, 1], offset: ?>>, memref<32x32xf32, strided<[768, 1], offset: ?>>) outs(%subview_104 : memref<1x32xf32, strided<[768, 1], offset: ?>>) {
                ^bb0(%in: f32, %in_105: f32, %out: f32):
                  %38 = arith.mulf %in, %in_105 : f32
                  %39 = arith.addf %out, %38 : f32
                  linalg.yield %39 : f32
                }
              }
            }
          }
        }
        %alloc_41 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg4 = %c0 to %c768 step %c32 {
          %subview_99 = memref.subview %alloc_41[0, %arg4] [1, 32] [1, 1] : memref<1x768xf32> to memref<1x32xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : f32) outs(%subview_99 : memref<1x32xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
        }
        %alloc_42 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        memref.copy %alloc_41, %alloc_42 : memref<1x768xf32> to memref<1x768xf32>
        scf.for %arg4 = %c0 to %c768 step %c128 {
          scf.for %arg5 = %c0 to %c768 step %c128 {
            %subview_99 = memref.subview %alloc_33[0, %arg5] [1, 128] [1, 1] : memref<1x768xf32> to memref<1x128xf32, strided<[768, 1], offset: ?>>
            %subview_100 = memref.subview %subview_36[%arg5, %arg4] [128, 128] [1, 1] : memref<768x768xf32, strided<[768, 1], offset: ?>> to memref<128x128xf32, strided<[768, 1], offset: ?>>
            %subview_101 = memref.subview %alloc_42[0, %arg4] [1, 128] [1, 1] : memref<1x768xf32> to memref<1x128xf32, strided<[768, 1], offset: ?>>
            scf.for %arg6 = %c0 to %c128 step %c32 {
              scf.for %arg7 = %c0 to %c128 step %c32 {
                %subview_102 = memref.subview %subview_99[0, %arg7] [1, 32] [1, 1] : memref<1x128xf32, strided<[768, 1], offset: ?>> to memref<1x32xf32, strided<[768, 1], offset: ?>>
                %subview_103 = memref.subview %subview_100[%arg7, %arg6] [32, 32] [1, 1] : memref<128x128xf32, strided<[768, 1], offset: ?>> to memref<32x32xf32, strided<[768, 1], offset: ?>>
                %subview_104 = memref.subview %subview_101[0, %arg6] [1, 32] [1, 1] : memref<1x128xf32, strided<[768, 1], offset: ?>> to memref<1x32xf32, strided<[768, 1], offset: ?>>
                linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_102, %subview_103 : memref<1x32xf32, strided<[768, 1], offset: ?>>, memref<32x32xf32, strided<[768, 1], offset: ?>>) outs(%subview_104 : memref<1x32xf32, strided<[768, 1], offset: ?>>) {
                ^bb0(%in: f32, %in_105: f32, %out: f32):
                  %38 = arith.mulf %in, %in_105 : f32
                  %39 = arith.addf %out, %38 : f32
                  linalg.yield %39 : f32
                }
              }
            }
          }
        }
        %reshape = memref.reshape %alloc_38(%3) : (memref<1x768xf32>, memref<3xi64>) -> memref<1x12x64xf32>
        %alloc_43 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
        %alloc_44 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
        %36 = arith.uitofp %arg1 : i64 to f32
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} outs(%alloc_43, %alloc_44 : memref<32xf32>, memref<32xf32>) {
        ^bb0(%out: f32, %out_99: f32):
          %38 = linalg.index 0 : index
          %39 = arith.index_cast %38 : index to i64
          %40 = arith.uitofp %39 : i64 to f32
          %41 = arith.mulf %40, %cst_4 : f32
          %42 = arith.divf %41, %cst_3 : f32
          %43 = math.powf %cst_2, %42 : f32
          %44 = arith.mulf %36, %43 : f32
          %45 = math.cos %44 : f32
          %46 = math.sin %44 : f32
          linalg.yield %45, %46 : f32, f32
        }
        %expand_shape = memref.expand_shape %reshape [[0], [1], [2, 3]] output_shape [1, 12, 32, 2] : memref<1x12x64xf32> into memref<1x12x32x2xf32>
        %subview_45 = memref.subview %expand_shape[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        %collapse_shape = memref.collapse_shape %subview_45 [[0], [1], [2, 3]] : memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>> into memref<1x12x32xf32, strided<[768, 64, 2]>>
        %subview_46 = memref.subview %expand_shape[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        %collapse_shape_47 = memref.collapse_shape %subview_46 [[0], [1], [2, 3]] : memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>> into memref<1x12x32xf32, strided<[768, 64, 2], offset: 1>>
        %alloc_48 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32xf32>
        %alloc_49 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32xf32>
        linalg.generic {indexing_maps = [#map9, #map9, #map10, #map10, #map9, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapse_shape, %collapse_shape_47, %alloc_43, %alloc_44 : memref<1x12x32xf32, strided<[768, 64, 2]>>, memref<1x12x32xf32, strided<[768, 64, 2], offset: 1>>, memref<32xf32>, memref<32xf32>) outs(%alloc_48, %alloc_49 : memref<1x12x32xf32>, memref<1x12x32xf32>) {
        ^bb0(%in: f32, %in_99: f32, %in_100: f32, %in_101: f32, %out: f32, %out_102: f32):
          %38 = arith.mulf %in, %in_100 : f32
          %39 = arith.mulf %in_99, %in_101 : f32
          %40 = arith.subf %38, %39 : f32
          %41 = arith.mulf %in_99, %in_100 : f32
          %42 = arith.mulf %in, %in_101 : f32
          %43 = arith.addf %41, %42 : f32
          linalg.yield %40, %43 : f32, f32
        }
        %expand_shape_50 = memref.expand_shape %alloc_48 [[0], [1], [2, 3]] output_shape [1, 12, 32, 1] : memref<1x12x32xf32> into memref<1x12x32x1xf32>
        %expand_shape_51 = memref.expand_shape %alloc_49 [[0], [1], [2, 3]] output_shape [1, 12, 32, 1] : memref<1x12x32xf32> into memref<1x12x32x1xf32>
        %alloc_52 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x2xf32>
        %subview_53 = memref.subview %alloc_52[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        memref.copy %expand_shape_50, %subview_53 : memref<1x12x32x1xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        %alloc_54 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x2xf32>
        memref.copy %alloc_52, %alloc_54 : memref<1x12x32x2xf32> to memref<1x12x32x2xf32>
        %subview_55 = memref.subview %alloc_54[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        memref.copy %expand_shape_51, %subview_55 : memref<1x12x32x1xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        %collapse_shape_56 = memref.collapse_shape %alloc_54 [[0], [1], [2, 3]] : memref<1x12x32x2xf32> into memref<1x12x64xf32>
        %reshape_57 = memref.reshape %collapse_shape_56(%2) : (memref<1x12x64xf32>, memref<2xi64>) -> memref<1x768xf32>
        %reshape_58 = memref.reshape %alloc_40(%3) : (memref<1x768xf32>, memref<3xi64>) -> memref<1x12x64xf32>
        %alloc_59 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
        %alloc_60 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} outs(%alloc_59, %alloc_60 : memref<32xf32>, memref<32xf32>) {
        ^bb0(%out: f32, %out_99: f32):
          %38 = linalg.index 0 : index
          %39 = arith.index_cast %38 : index to i64
          %40 = arith.uitofp %39 : i64 to f32
          %41 = arith.mulf %40, %cst_4 : f32
          %42 = arith.divf %41, %cst_3 : f32
          %43 = math.powf %cst_2, %42 : f32
          %44 = arith.mulf %36, %43 : f32
          %45 = math.cos %44 : f32
          %46 = math.sin %44 : f32
          linalg.yield %45, %46 : f32, f32
        }
        %expand_shape_61 = memref.expand_shape %reshape_58 [[0], [1], [2, 3]] output_shape [1, 12, 32, 2] : memref<1x12x64xf32> into memref<1x12x32x2xf32>
        %subview_62 = memref.subview %expand_shape_61[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        %collapse_shape_63 = memref.collapse_shape %subview_62 [[0], [1], [2, 3]] : memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>> into memref<1x12x32xf32, strided<[768, 64, 2]>>
        %subview_64 = memref.subview %expand_shape_61[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        %collapse_shape_65 = memref.collapse_shape %subview_64 [[0], [1], [2, 3]] : memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>> into memref<1x12x32xf32, strided<[768, 64, 2], offset: 1>>
        %alloc_66 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32xf32>
        %alloc_67 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32xf32>
        linalg.generic {indexing_maps = [#map9, #map9, #map10, #map10, #map9, #map9], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapse_shape_63, %collapse_shape_65, %alloc_59, %alloc_60 : memref<1x12x32xf32, strided<[768, 64, 2]>>, memref<1x12x32xf32, strided<[768, 64, 2], offset: 1>>, memref<32xf32>, memref<32xf32>) outs(%alloc_66, %alloc_67 : memref<1x12x32xf32>, memref<1x12x32xf32>) {
        ^bb0(%in: f32, %in_99: f32, %in_100: f32, %in_101: f32, %out: f32, %out_102: f32):
          %38 = arith.mulf %in, %in_100 : f32
          %39 = arith.mulf %in_99, %in_101 : f32
          %40 = arith.subf %38, %39 : f32
          %41 = arith.mulf %in_99, %in_100 : f32
          %42 = arith.mulf %in, %in_101 : f32
          %43 = arith.addf %41, %42 : f32
          linalg.yield %40, %43 : f32, f32
        }
        %expand_shape_68 = memref.expand_shape %alloc_66 [[0], [1], [2, 3]] output_shape [1, 12, 32, 1] : memref<1x12x32xf32> into memref<1x12x32x1xf32>
        %expand_shape_69 = memref.expand_shape %alloc_67 [[0], [1], [2, 3]] output_shape [1, 12, 32, 1] : memref<1x12x32xf32> into memref<1x12x32x1xf32>
        %alloc_70 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x2xf32>
        %subview_71 = memref.subview %alloc_70[0, 0, 0, 0] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        memref.copy %expand_shape_68, %subview_71 : memref<1x12x32x1xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1]>>
        %alloc_72 = memref.alloc() {alignment = 64 : i64} : memref<1x12x32x2xf32>
        memref.copy %alloc_70, %alloc_72 : memref<1x12x32x2xf32> to memref<1x12x32x2xf32>
        %subview_73 = memref.subview %alloc_72[0, 0, 0, 1] [1, 12, 32, 1] [1, 1, 1, 1] : memref<1x12x32x2xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        memref.copy %expand_shape_69, %subview_73 : memref<1x12x32x1xf32> to memref<1x12x32x1xf32, strided<[768, 64, 2, 1], offset: 1>>
        %collapse_shape_74 = memref.collapse_shape %alloc_72 [[0], [1], [2, 3]] : memref<1x12x32x2xf32> into memref<1x12x64xf32>
        %reshape_75 = memref.reshape %collapse_shape_74(%1) : (memref<1x12x64xf32>, memref<3xi64>) -> memref<1x1x768xf32>
        %37 = arith.index_cast %arg1 : i64 to index
        %subview_76 = memref.subview %alloc_21[%arg2, %37, 0] [1, 1, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        memref.copy %reshape_75, %subview_76 : memref<1x1x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        %reshape_77 = memref.reshape %alloc_42(%1) : (memref<1x768xf32>, memref<3xi64>) -> memref<1x1x768xf32>
        %subview_78 = memref.subview %alloc_22[%arg2, %37, 0] [1, 1, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        memref.copy %reshape_77, %subview_78 : memref<1x1x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        %subview_79 = memref.subview %alloc_21[%arg2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: ?>>
        %subview_80 = memref.subview %alloc_22[%arg2, 0, 0] [1, 1024, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1024x768xf32, strided<[768, 1], offset: ?>>
        %alloc_81 = memref.alloc() {alignment = 64 : i64} : memref<1x12x64xf32>
        memref.copy %4, %alloc_81 : memref<1x12x64xf32> to memref<1x12x64xf32>
        scf.for %arg4 = %c0 to %c12 step %c1 {
          %38 = arith.index_cast %arg4 : index to i64
          %39 = arith.muli %38, %c64_i64 : i64
          %40 = arith.index_cast %39 : i64 to index
          %subview_99 = memref.subview %reshape_57[0, %40] [1, 64] [1, 1] : memref<1x768xf32> to memref<1x64xf32, strided<[768, 1], offset: ?>>
          %subview_100 = memref.subview %subview_79[0, %40] [1024, 64] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: ?>> to memref<1024x64xf32, strided<[768, 1], offset: ?>>
          %alloc_101 = memref.alloc() {alignment = 64 : i64} : memref<64x1024xf32>
          scf.for %arg5 = %c0 to %c64 step %c32 {
            scf.for %arg6 = %c0 to %c1024 step %c32 {
              %subview_116 = memref.subview %subview_100[%arg6, %arg5] [32, 32] [1, 1] : memref<1024x64xf32, strided<[768, 1], offset: ?>> to memref<32x32xf32, strided<[768, 1], offset: ?>>
              %subview_117 = memref.subview %alloc_101[%arg5, %arg6] [32, 32] [1, 1] : memref<64x1024xf32> to memref<32x32xf32, strided<[1024, 1], offset: ?>>
              linalg.generic {indexing_maps = [#map11, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_116 : memref<32x32xf32, strided<[768, 1], offset: ?>>) outs(%subview_117 : memref<32x32xf32, strided<[1024, 1], offset: ?>>) {
              ^bb0(%in: f32, %out: f32):
                linalg.yield %in : f32
              }
            }
          }
          %41 = arith.index_cast %32 : i64 to index
          %subview_102 = memref.subview %alloc_101[0, 0] [64, %41] [1, 1] : memref<64x1024xf32> to memref<64x?xf32, strided<[1024, 1]>>
          %alloc_103 = memref.alloc(%41) {alignment = 64 : i64} : memref<1x?xf32>
          scf.for %arg5 = %c0 to %41 step %c32 {
            %42 = affine.min #map12(%41, %arg5)
            %subview_116 = memref.subview %alloc_103[0, %arg5] [1, %42] [1, 1] : memref<1x?xf32> to memref<1x?xf32, strided<[?, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : f32) outs(%subview_116 : memref<1x?xf32, strided<[?, 1], offset: ?>>) {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
          }
          scf.for %arg5 = %c0 to %41 step %c128 {
            %42 = affine.min #map13(%41, %arg5)
            %subview_116 = memref.subview %subview_102[0, %arg5] [64, %42] [1, 1] : memref<64x?xf32, strided<[1024, 1]>> to memref<64x?xf32, strided<[1024, 1], offset: ?>>
            %subview_117 = memref.subview %alloc_103[0, %arg5] [1, %42] [1, 1] : memref<1x?xf32> to memref<1x?xf32, strided<[?, 1], offset: ?>>
            scf.for %arg6 = %c0 to %42 step %c32 {
              scf.for %arg7 = %c0 to %c64 step %c32 {
                %43 = affine.min #map14(%arg7)
                %44 = affine.min #map12(%42, %arg6)
                %subview_118 = memref.subview %subview_99[0, %arg7] [1, %43] [1, 1] : memref<1x64xf32, strided<[768, 1], offset: ?>> to memref<1x?xf32, strided<[768, 1], offset: ?>>
                %subview_119 = memref.subview %subview_116[%arg7, %arg6] [%43, %44] [1, 1] : memref<64x?xf32, strided<[1024, 1], offset: ?>> to memref<?x?xf32, strided<[1024, 1], offset: ?>>
                %subview_120 = memref.subview %subview_117[0, %arg6] [1, %44] [1, 1] : memref<1x?xf32, strided<[?, 1], offset: ?>> to memref<1x?xf32, strided<[?, 1], offset: ?>>
                linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_118, %subview_119 : memref<1x?xf32, strided<[768, 1], offset: ?>>, memref<?x?xf32, strided<[1024, 1], offset: ?>>) outs(%subview_120 : memref<1x?xf32, strided<[?, 1], offset: ?>>) {
                ^bb0(%in: f32, %in_121: f32, %out: f32):
                  %45 = arith.mulf %in, %in_121 : f32
                  %46 = arith.addf %out, %45 : f32
                  linalg.yield %46 : f32
                }
              }
            }
          }
          %alloc_104 = memref.alloc() {alignment = 64 : i64} : memref<1x1024xf32>
          scf.for %arg5 = %c0 to %c1024 step %c32 {
            %subview_116 = memref.subview %alloc_104[0, %arg5] [1, 32] [1, 1] : memref<1x1024xf32> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%cst_5 : f32) outs(%subview_116 : memref<1x32xf32, strided<[1024, 1], offset: ?>>) {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
          }
          %subview_105 = memref.subview %alloc_104[0, 0] [1, %41] [1, 1] : memref<1x1024xf32> to memref<1x?xf32, strided<[1024, 1]>>
          memref.copy %alloc_103, %subview_105 : memref<1x?xf32> to memref<1x?xf32, strided<[1024, 1]>>
          %alloc_106 = memref.alloc() {alignment = 64 : i64} : memref<1x1024xf32>
          scf.for %arg5 = %c0 to %c1024 step %c32 {
            %subview_116 = memref.subview %alloc_104[0, %arg5] [1, 32] [1, 1] : memref<1x1024xf32> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
            %subview_117 = memref.subview %alloc_106[0, %arg5] [1, 32] [1, 1] : memref<1x1024xf32> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_116 : memref<1x32xf32, strided<[1024, 1], offset: ?>>) outs(%subview_117 : memref<1x32xf32, strided<[1024, 1], offset: ?>>) {
            ^bb0(%in: f32, %out: f32):
              %42 = arith.mulf %in, %cst : f32
              linalg.yield %42 : f32
            }
          }
          %alloc_107 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
          linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%cst_6 : f32) outs(%alloc_107 : memref<1xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          scf.for %arg5 = %c0 to %c1024 step %c128 {
            %subview_116 = memref.subview %alloc_106[0, %arg5] [1, 128] [1, 1] : memref<1x1024xf32> to memref<1x128xf32, strided<[1024, 1], offset: ?>>
            scf.for %arg6 = %c0 to %c128 step %c32 {
              %subview_117 = memref.subview %subview_116[0, %arg6] [1, 32] [1, 1] : memref<1x128xf32, strided<[1024, 1], offset: ?>> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
              linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%subview_117 : memref<1x32xf32, strided<[1024, 1], offset: ?>>) outs(%alloc_107 : memref<1xf32>) {
              ^bb0(%in: f32, %out: f32):
                %42 = arith.maxnumf %in, %out : f32
                linalg.yield %42 : f32
              }
            }
          }
          %alloc_108 = memref.alloc() {alignment = 64 : i64} : memref<1x1024xf32>
          scf.for %arg5 = %c0 to %c1024 step %c32 {
            %subview_116 = memref.subview %alloc_106[0, %arg5] [1, 32] [1, 1] : memref<1x1024xf32> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
            %subview_117 = memref.subview %alloc_108[0, %arg5] [1, 32] [1, 1] : memref<1x1024xf32> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_116, %alloc_107 : memref<1x32xf32, strided<[1024, 1], offset: ?>>, memref<1xf32>) outs(%subview_117 : memref<1x32xf32, strided<[1024, 1], offset: ?>>) {
            ^bb0(%in: f32, %in_118: f32, %out: f32):
              %42 = arith.subf %in, %in_118 : f32
              %43 = math.exp %42 : f32
              linalg.yield %43 : f32
            }
          }
          %alloc_109 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
          linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%cst_0 : f32) outs(%alloc_109 : memref<1xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          scf.for %arg5 = %c0 to %c1024 step %c32 {
            %subview_116 = memref.subview %alloc_108[0, %arg5] [1, 32] [1, 1] : memref<1x1024xf32> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%subview_116 : memref<1x32xf32, strided<[1024, 1], offset: ?>>) outs(%alloc_109 : memref<1xf32>) {
            ^bb0(%in: f32, %out: f32):
              %42 = arith.addf %in, %out : f32
              linalg.yield %42 : f32
            }
          }
          %alloc_110 = memref.alloc() {alignment = 64 : i64} : memref<1x1024xf32>
          scf.for %arg5 = %c0 to %c1024 step %c32 {
            %subview_116 = memref.subview %alloc_108[0, %arg5] [1, 32] [1, 1] : memref<1x1024xf32> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
            %subview_117 = memref.subview %alloc_110[0, %arg5] [1, 32] [1, 1] : memref<1x1024xf32> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_116, %alloc_109 : memref<1x32xf32, strided<[1024, 1], offset: ?>>, memref<1xf32>) outs(%subview_117 : memref<1x32xf32, strided<[1024, 1], offset: ?>>) {
            ^bb0(%in: f32, %in_118: f32, %out: f32):
              %42 = arith.divf %in, %in_118 : f32
              linalg.yield %42 : f32
            }
          }
          %subview_111 = memref.subview %subview_80[0, %40] [1024, 64] [1, 1] : memref<1024x768xf32, strided<[768, 1], offset: ?>> to memref<1024x64xf32, strided<[768, 1], offset: ?>>
          %alloc_112 = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32>
          scf.for %arg5 = %c0 to %c64 step %c32 {
            %subview_116 = memref.subview %alloc_112[0, %arg5] [1, 32] [1, 1] : memref<1x64xf32> to memref<1x32xf32, strided<[64, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : f32) outs(%subview_116 : memref<1x32xf32, strided<[64, 1], offset: ?>>) {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
          }
          %alloc_113 = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32>
          memref.copy %alloc_112, %alloc_113 : memref<1x64xf32> to memref<1x64xf32>
          scf.for %arg5 = %c0 to %c1024 step %c128 {
            %subview_116 = memref.subview %alloc_110[0, %arg5] [1, 128] [1, 1] : memref<1x1024xf32> to memref<1x128xf32, strided<[1024, 1], offset: ?>>
            %subview_117 = memref.subview %subview_111[%arg5, 0] [128, 64] [1, 1] : memref<1024x64xf32, strided<[768, 1], offset: ?>> to memref<128x64xf32, strided<[768, 1], offset: ?>>
            scf.for %arg6 = %c0 to %c64 step %c32 {
              scf.for %arg7 = %c0 to %c128 step %c32 {
                %42 = affine.min #map14(%arg6)
                %subview_118 = memref.subview %subview_116[0, %arg7] [1, 32] [1, 1] : memref<1x128xf32, strided<[1024, 1], offset: ?>> to memref<1x32xf32, strided<[1024, 1], offset: ?>>
                %subview_119 = memref.subview %subview_117[%arg7, %arg6] [32, %42] [1, 1] : memref<128x64xf32, strided<[768, 1], offset: ?>> to memref<32x?xf32, strided<[768, 1], offset: ?>>
                %subview_120 = memref.subview %alloc_113[0, %arg6] [1, %42] [1, 1] : memref<1x64xf32> to memref<1x?xf32, strided<[64, 1], offset: ?>>
                linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_118, %subview_119 : memref<1x32xf32, strided<[1024, 1], offset: ?>>, memref<32x?xf32, strided<[768, 1], offset: ?>>) outs(%subview_120 : memref<1x?xf32, strided<[64, 1], offset: ?>>) {
                ^bb0(%in: f32, %in_121: f32, %out: f32):
                  %43 = arith.mulf %in, %in_121 : f32
                  %44 = arith.addf %out, %43 : f32
                  linalg.yield %44 : f32
                }
              }
            }
          }
          %reshape_114 = memref.reshape %alloc_113(%0) : (memref<1x64xf32>, memref<3xi64>) -> memref<1x1x64xf32>
          %subview_115 = memref.subview %alloc_81[0, %arg4, 0] [1, 1, 64] [1, 1, 1] : memref<1x12x64xf32> to memref<1x1x64xf32, strided<[768, 64, 1], offset: ?>>
          memref.copy %reshape_114, %subview_115 : memref<1x1x64xf32> to memref<1x1x64xf32, strided<[768, 64, 1], offset: ?>>
        }
        %reshape_82 = memref.reshape %alloc_81(%2) : (memref<1x12x64xf32>, memref<2xi64>) -> memref<1x768xf32>
        %subview_83 = memref.subview %24[%arg2, 0, 0] [1, 768, 768] [1, 1, 1] : memref<12x768x768xf32> to memref<768x768xf32, strided<[768, 1], offset: ?>>
        %alloc_84 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg4 = %c0 to %c768 step %c32 {
          %subview_99 = memref.subview %alloc_84[0, %arg4] [1, 32] [1, 1] : memref<1x768xf32> to memref<1x32xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : f32) outs(%subview_99 : memref<1x32xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
        }
        scf.for %arg4 = %c0 to %c768 step %c128 {
          scf.for %arg5 = %c0 to %c768 step %c128 {
            %subview_99 = memref.subview %reshape_82[0, %arg5] [1, 128] [1, 1] : memref<1x768xf32> to memref<1x128xf32, strided<[768, 1], offset: ?>>
            %subview_100 = memref.subview %subview_83[%arg5, %arg4] [128, 128] [1, 1] : memref<768x768xf32, strided<[768, 1], offset: ?>> to memref<128x128xf32, strided<[768, 1], offset: ?>>
            %subview_101 = memref.subview %alloc_84[0, %arg4] [1, 128] [1, 1] : memref<1x768xf32> to memref<1x128xf32, strided<[768, 1], offset: ?>>
            scf.for %arg6 = %c0 to %c128 step %c32 {
              scf.for %arg7 = %c0 to %c128 step %c32 {
                %subview_102 = memref.subview %subview_99[0, %arg7] [1, 32] [1, 1] : memref<1x128xf32, strided<[768, 1], offset: ?>> to memref<1x32xf32, strided<[768, 1], offset: ?>>
                %subview_103 = memref.subview %subview_100[%arg7, %arg6] [32, 32] [1, 1] : memref<128x128xf32, strided<[768, 1], offset: ?>> to memref<32x32xf32, strided<[768, 1], offset: ?>>
                %subview_104 = memref.subview %subview_101[0, %arg6] [1, 32] [1, 1] : memref<1x128xf32, strided<[768, 1], offset: ?>> to memref<1x32xf32, strided<[768, 1], offset: ?>>
                linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_102, %subview_103 : memref<1x32xf32, strided<[768, 1], offset: ?>>, memref<32x32xf32, strided<[768, 1], offset: ?>>) outs(%subview_104 : memref<1x32xf32, strided<[768, 1], offset: ?>>) {
                ^bb0(%in: f32, %in_105: f32, %out: f32):
                  %38 = arith.mulf %in, %in_105 : f32
                  %39 = arith.addf %out, %38 : f32
                  linalg.yield %39 : f32
                }
              }
            }
          }
        }
        %alloc_85 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg4 = %c0 to %c768 step %c32 {
          %subview_99 = memref.subview %arg3[0, %arg4] [1, 32] [1, 1] : memref<1x768xf32> to memref<1x32xf32, strided<[768, 1], offset: ?>>
          %subview_100 = memref.subview %alloc_84[0, %arg4] [1, 32] [1, 1] : memref<1x768xf32> to memref<1x32xf32, strided<[768, 1], offset: ?>>
          %subview_101 = memref.subview %alloc_85[0, %arg4] [1, 32] [1, 1] : memref<1x768xf32> to memref<1x32xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_99, %subview_100 : memref<1x32xf32, strided<[768, 1], offset: ?>>, memref<1x32xf32, strided<[768, 1], offset: ?>>) outs(%subview_101 : memref<1x32xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%in: f32, %in_102: f32, %out: f32):
            %38 = arith.addf %in, %in_102 : f32
            linalg.yield %38 : f32
          }
        }
        %subview_86 = memref.subview %25[%arg2, 0] [1, 768] [1, 1] : memref<12x768xf32> to memref<768xf32, strided<[1], offset: ?>>
        %alloc_87 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
        linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%cst_0 : f32) outs(%alloc_87 : memref<1xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        }
        scf.for %arg4 = %c0 to %c768 step %c128 {
          %subview_99 = memref.subview %alloc_85[0, %arg4] [1, 128] [1, 1] : memref<1x768xf32> to memref<1x128xf32, strided<[768, 1], offset: ?>>
          scf.for %arg5 = %c0 to %c128 step %c32 {
            %subview_100 = memref.subview %subview_99[0, %arg5] [1, 32] [1, 1] : memref<1x128xf32, strided<[768, 1], offset: ?>> to memref<1x32xf32, strided<[768, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%subview_100 : memref<1x32xf32, strided<[768, 1], offset: ?>>) outs(%alloc_87 : memref<1xf32>) {
            ^bb0(%in: f32, %out: f32):
              %38 = arith.mulf %in, %in : f32
              %39 = arith.addf %out, %38 : f32
              linalg.yield %39 : f32
            }
          }
        }
        %alloc_88 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
        linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_87 : memref<1xf32>) outs(%alloc_88 : memref<1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %38 = arith.divf %in, %cst_8 : f32
          %39 = arith.addf %38, %cst_1 : f32
          %40 = math.rsqrt %39 : f32
          linalg.yield %40 : f32
        }
        %alloc_89 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg4 = %c0 to %c768 step %c32 {
          %subview_99 = memref.subview %alloc_85[0, %arg4] [1, 32] [1, 1] : memref<1x768xf32> to memref<1x32xf32, strided<[768, 1], offset: ?>>
          %subview_100 = memref.subview %subview_86[%arg4] [32] [1] : memref<768xf32, strided<[1], offset: ?>> to memref<32xf32, strided<[1], offset: ?>>
          %subview_101 = memref.subview %alloc_89[0, %arg4] [1, 32] [1, 1] : memref<1x768xf32> to memref<1x32xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map2, #map3, #map4, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_99, %alloc_88, %subview_100 : memref<1x32xf32, strided<[768, 1], offset: ?>>, memref<1xf32>, memref<32xf32, strided<[1], offset: ?>>) outs(%subview_101 : memref<1x32xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%in: f32, %in_102: f32, %in_103: f32, %out: f32):
            %38 = arith.mulf %in, %in_102 : f32
            %39 = arith.mulf %38, %in_103 : f32
            linalg.yield %39 : f32
          }
        }
        %subview_90 = memref.subview %26[%arg2, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<12x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: ?>>
        %subview_91 = memref.subview %28[%arg2, 0, 0] [1, 768, 2048] [1, 1, 1] : memref<12x768x2048xf32> to memref<768x2048xf32, strided<[2048, 1], offset: ?>>
        %alloc_92 = memref.alloc() {alignment = 64 : i64} : memref<1x2048xf32>
        scf.for %arg4 = %c0 to %c2048 step %c32 {
          %subview_99 = memref.subview %alloc_92[0, %arg4] [1, 32] [1, 1] : memref<1x2048xf32> to memref<1x32xf32, strided<[2048, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : f32) outs(%subview_99 : memref<1x32xf32, strided<[2048, 1], offset: ?>>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
        }
        scf.for %arg4 = %c0 to %c2048 step %c128 {
          scf.for %arg5 = %c0 to %c768 step %c128 {
            %subview_99 = memref.subview %alloc_89[0, %arg5] [1, 128] [1, 1] : memref<1x768xf32> to memref<1x128xf32, strided<[768, 1], offset: ?>>
            %subview_100 = memref.subview %subview_90[%arg5, %arg4] [128, 128] [1, 1] : memref<768x2048xf32, strided<[2048, 1], offset: ?>> to memref<128x128xf32, strided<[2048, 1], offset: ?>>
            %subview_101 = memref.subview %alloc_92[0, %arg4] [1, 128] [1, 1] : memref<1x2048xf32> to memref<1x128xf32, strided<[2048, 1], offset: ?>>
            scf.for %arg6 = %c0 to %c128 step %c32 {
              scf.for %arg7 = %c0 to %c128 step %c32 {
                %subview_102 = memref.subview %subview_99[0, %arg7] [1, 32] [1, 1] : memref<1x128xf32, strided<[768, 1], offset: ?>> to memref<1x32xf32, strided<[768, 1], offset: ?>>
                %subview_103 = memref.subview %subview_100[%arg7, %arg6] [32, 32] [1, 1] : memref<128x128xf32, strided<[2048, 1], offset: ?>> to memref<32x32xf32, strided<[2048, 1], offset: ?>>
                %subview_104 = memref.subview %subview_101[0, %arg6] [1, 32] [1, 1] : memref<1x128xf32, strided<[2048, 1], offset: ?>> to memref<1x32xf32, strided<[2048, 1], offset: ?>>
                linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_102, %subview_103 : memref<1x32xf32, strided<[768, 1], offset: ?>>, memref<32x32xf32, strided<[2048, 1], offset: ?>>) outs(%subview_104 : memref<1x32xf32, strided<[2048, 1], offset: ?>>) {
                ^bb0(%in: f32, %in_105: f32, %out: f32):
                  %38 = arith.mulf %in, %in_105 : f32
                  %39 = arith.addf %out, %38 : f32
                  linalg.yield %39 : f32
                }
              }
            }
          }
        }
        %alloc_93 = memref.alloc() {alignment = 64 : i64} : memref<1x2048xf32>
        scf.for %arg4 = %c0 to %c2048 step %c32 {
          %subview_99 = memref.subview %alloc_93[0, %arg4] [1, 32] [1, 1] : memref<1x2048xf32> to memref<1x32xf32, strided<[2048, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : f32) outs(%subview_99 : memref<1x32xf32, strided<[2048, 1], offset: ?>>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
        }
        scf.for %arg4 = %c0 to %c2048 step %c128 {
          scf.for %arg5 = %c0 to %c768 step %c128 {
            %subview_99 = memref.subview %alloc_89[0, %arg5] [1, 128] [1, 1] : memref<1x768xf32> to memref<1x128xf32, strided<[768, 1], offset: ?>>
            %subview_100 = memref.subview %subview_91[%arg5, %arg4] [128, 128] [1, 1] : memref<768x2048xf32, strided<[2048, 1], offset: ?>> to memref<128x128xf32, strided<[2048, 1], offset: ?>>
            %subview_101 = memref.subview %alloc_93[0, %arg4] [1, 128] [1, 1] : memref<1x2048xf32> to memref<1x128xf32, strided<[2048, 1], offset: ?>>
            scf.for %arg6 = %c0 to %c128 step %c32 {
              scf.for %arg7 = %c0 to %c128 step %c32 {
                %subview_102 = memref.subview %subview_99[0, %arg7] [1, 32] [1, 1] : memref<1x128xf32, strided<[768, 1], offset: ?>> to memref<1x32xf32, strided<[768, 1], offset: ?>>
                %subview_103 = memref.subview %subview_100[%arg7, %arg6] [32, 32] [1, 1] : memref<128x128xf32, strided<[2048, 1], offset: ?>> to memref<32x32xf32, strided<[2048, 1], offset: ?>>
                %subview_104 = memref.subview %subview_101[0, %arg6] [1, 32] [1, 1] : memref<1x128xf32, strided<[2048, 1], offset: ?>> to memref<1x32xf32, strided<[2048, 1], offset: ?>>
                linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_102, %subview_103 : memref<1x32xf32, strided<[768, 1], offset: ?>>, memref<32x32xf32, strided<[2048, 1], offset: ?>>) outs(%subview_104 : memref<1x32xf32, strided<[2048, 1], offset: ?>>) {
                ^bb0(%in: f32, %in_105: f32, %out: f32):
                  %38 = arith.mulf %in, %in_105 : f32
                  %39 = arith.addf %out, %38 : f32
                  linalg.yield %39 : f32
                }
              }
            }
          }
        }
        %alloc_94 = memref.alloc() {alignment = 64 : i64} : memref<1x2048xf32>
        scf.for %arg4 = %c0 to %c2048 step %c32 {
          %subview_99 = memref.subview %alloc_92[0, %arg4] [1, 32] [1, 1] : memref<1x2048xf32> to memref<1x32xf32, strided<[2048, 1], offset: ?>>
          %subview_100 = memref.subview %alloc_94[0, %arg4] [1, 32] [1, 1] : memref<1x2048xf32> to memref<1x32xf32, strided<[2048, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_99 : memref<1x32xf32, strided<[2048, 1], offset: ?>>) outs(%subview_100 : memref<1x32xf32, strided<[2048, 1], offset: ?>>) {
          ^bb0(%in: f32, %out: f32):
            %38 = arith.negf %in : f32
            %39 = math.exp %38 : f32
            %40 = arith.addf %39, %cst_7 : f32
            %41 = arith.divf %in, %40 : f32
            linalg.yield %41 : f32
          }
        }
        %alloc_95 = memref.alloc() {alignment = 64 : i64} : memref<1x2048xf32>
        scf.for %arg4 = %c0 to %c2048 step %c32 {
          %subview_99 = memref.subview %alloc_94[0, %arg4] [1, 32] [1, 1] : memref<1x2048xf32> to memref<1x32xf32, strided<[2048, 1], offset: ?>>
          %subview_100 = memref.subview %alloc_93[0, %arg4] [1, 32] [1, 1] : memref<1x2048xf32> to memref<1x32xf32, strided<[2048, 1], offset: ?>>
          %subview_101 = memref.subview %alloc_95[0, %arg4] [1, 32] [1, 1] : memref<1x2048xf32> to memref<1x32xf32, strided<[2048, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_99, %subview_100 : memref<1x32xf32, strided<[2048, 1], offset: ?>>, memref<1x32xf32, strided<[2048, 1], offset: ?>>) outs(%subview_101 : memref<1x32xf32, strided<[2048, 1], offset: ?>>) {
          ^bb0(%in: f32, %in_102: f32, %out: f32):
            %38 = arith.mulf %in, %in_102 : f32
            linalg.yield %38 : f32
          }
        }
        %subview_96 = memref.subview %27[%arg2, 0, 0] [1, 2048, 768] [1, 1, 1] : memref<12x2048x768xf32> to memref<2048x768xf32, strided<[768, 1], offset: ?>>
        %alloc_97 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg4 = %c0 to %c768 step %c32 {
          %subview_99 = memref.subview %alloc_97[0, %arg4] [1, 32] [1, 1] : memref<1x768xf32> to memref<1x32xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : f32) outs(%subview_99 : memref<1x32xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
        }
        scf.for %arg4 = %c0 to %c768 step %c128 {
          scf.for %arg5 = %c0 to %c2048 step %c128 {
            %subview_99 = memref.subview %alloc_95[0, %arg5] [1, 128] [1, 1] : memref<1x2048xf32> to memref<1x128xf32, strided<[2048, 1], offset: ?>>
            %subview_100 = memref.subview %subview_96[%arg5, %arg4] [128, 128] [1, 1] : memref<2048x768xf32, strided<[768, 1], offset: ?>> to memref<128x128xf32, strided<[768, 1], offset: ?>>
            %subview_101 = memref.subview %alloc_97[0, %arg4] [1, 128] [1, 1] : memref<1x768xf32> to memref<1x128xf32, strided<[768, 1], offset: ?>>
            scf.for %arg6 = %c0 to %c128 step %c32 {
              scf.for %arg7 = %c0 to %c128 step %c32 {
                %subview_102 = memref.subview %subview_99[0, %arg7] [1, 32] [1, 1] : memref<1x128xf32, strided<[2048, 1], offset: ?>> to memref<1x32xf32, strided<[2048, 1], offset: ?>>
                %subview_103 = memref.subview %subview_100[%arg7, %arg6] [32, 32] [1, 1] : memref<128x128xf32, strided<[768, 1], offset: ?>> to memref<32x32xf32, strided<[768, 1], offset: ?>>
                %subview_104 = memref.subview %subview_101[0, %arg6] [1, 32] [1, 1] : memref<1x128xf32, strided<[768, 1], offset: ?>> to memref<1x32xf32, strided<[768, 1], offset: ?>>
                linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_102, %subview_103 : memref<1x32xf32, strided<[2048, 1], offset: ?>>, memref<32x32xf32, strided<[768, 1], offset: ?>>) outs(%subview_104 : memref<1x32xf32, strided<[768, 1], offset: ?>>) {
                ^bb0(%in: f32, %in_105: f32, %out: f32):
                  %38 = arith.mulf %in, %in_105 : f32
                  %39 = arith.addf %out, %38 : f32
                  linalg.yield %39 : f32
                }
              }
            }
          }
        }
        %alloc_98 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg4 = %c0 to %c768 step %c32 {
          %subview_99 = memref.subview %alloc_85[0, %arg4] [1, 32] [1, 1] : memref<1x768xf32> to memref<1x32xf32, strided<[768, 1], offset: ?>>
          %subview_100 = memref.subview %alloc_97[0, %arg4] [1, 32] [1, 1] : memref<1x768xf32> to memref<1x32xf32, strided<[768, 1], offset: ?>>
          %subview_101 = memref.subview %alloc_98[0, %arg4] [1, 32] [1, 1] : memref<1x768xf32> to memref<1x32xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_99, %subview_100 : memref<1x32xf32, strided<[768, 1], offset: ?>>, memref<1x32xf32, strided<[768, 1], offset: ?>>) outs(%subview_101 : memref<1x32xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%in: f32, %in_102: f32, %out: f32):
            %38 = arith.addf %in, %in_102 : f32
            linalg.yield %38 : f32
          }
        }
        scf.yield %alloc_98 : memref<1x768xf32>
      }
      %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%cst_0 : f32) outs(%alloc_24 : memref<1xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      scf.for %arg2 = %c0 to %c768 step %c128 {
        %subview_30 = memref.subview %34[0, %arg2] [1, 128] [1, 1] : memref<1x768xf32> to memref<1x128xf32, strided<[768, 1], offset: ?>>
        scf.for %arg3 = %c0 to %c128 step %c32 {
          %subview_31 = memref.subview %subview_30[0, %arg3] [1, 32] [1, 1] : memref<1x128xf32, strided<[768, 1], offset: ?>> to memref<1x32xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%subview_31 : memref<1x32xf32, strided<[768, 1], offset: ?>>) outs(%alloc_24 : memref<1xf32>) {
          ^bb0(%in: f32, %out: f32):
            %36 = arith.mulf %in, %in : f32
            %37 = arith.addf %out, %36 : f32
            linalg.yield %37 : f32
          }
        }
      }
      %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%alloc_24 : memref<1xf32>) outs(%alloc_25 : memref<1xf32>) {
      ^bb0(%in: f32, %out: f32):
        %36 = arith.divf %in, %cst_8 : f32
        %37 = arith.addf %36, %cst_1 : f32
        %38 = math.rsqrt %37 : f32
        linalg.yield %38 : f32
      }
      %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
      scf.for %arg2 = %c0 to %c768 step %c32 {
        %subview_30 = memref.subview %34[0, %arg2] [1, 32] [1, 1] : memref<1x768xf32> to memref<1x32xf32, strided<[768, 1], offset: ?>>
        %subview_31 = memref.subview %29[%arg2] [32] [1] : memref<768xf32> to memref<32xf32, strided<[1], offset: ?>>
        %subview_32 = memref.subview %alloc_26[0, %arg2] [1, 32] [1, 1] : memref<1x768xf32> to memref<1x32xf32, strided<[768, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map2, #map3, #map4, #map2], iterator_types = ["parallel", "parallel"]} ins(%subview_30, %alloc_25, %subview_31 : memref<1x32xf32, strided<[768, 1], offset: ?>>, memref<1xf32>, memref<32xf32, strided<[1], offset: ?>>) outs(%subview_32 : memref<1x32xf32, strided<[768, 1], offset: ?>>) {
        ^bb0(%in: f32, %in_33: f32, %in_34: f32, %out: f32):
          %36 = arith.mulf %in, %in_33 : f32
          %37 = arith.mulf %36, %in_34 : f32
          linalg.yield %37 : f32
        }
      }
      %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<1x32000xf32>
      scf.for %arg2 = %c0 to %c32000 step %c32 {
        %subview_30 = memref.subview %alloc_27[0, %arg2] [1, 32] [1, 1] : memref<1x32000xf32> to memref<1x32xf32, strided<[32000, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : f32) outs(%subview_30 : memref<1x32xf32, strided<[32000, 1], offset: ?>>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        }
      }
      scf.for %arg2 = %c0 to %c32000 step %c128 {
        scf.for %arg3 = %c0 to %c768 step %c128 {
          %subview_30 = memref.subview %alloc_26[0, %arg3] [1, 128] [1, 1] : memref<1x768xf32> to memref<1x128xf32, strided<[768, 1], offset: ?>>
          %subview_31 = memref.subview %30[%arg3, %arg2] [128, 128] [1, 1] : memref<768x32000xf32> to memref<128x128xf32, strided<[32000, 1], offset: ?>>
          %subview_32 = memref.subview %alloc_27[0, %arg2] [1, 128] [1, 1] : memref<1x32000xf32> to memref<1x128xf32, strided<[32000, 1], offset: ?>>
          scf.for %arg4 = %c0 to %c128 step %c32 {
            scf.for %arg5 = %c0 to %c128 step %c32 {
              %subview_33 = memref.subview %subview_30[0, %arg5] [1, 32] [1, 1] : memref<1x128xf32, strided<[768, 1], offset: ?>> to memref<1x32xf32, strided<[768, 1], offset: ?>>
              %subview_34 = memref.subview %subview_31[%arg5, %arg4] [32, 32] [1, 1] : memref<128x128xf32, strided<[32000, 1], offset: ?>> to memref<32x32xf32, strided<[32000, 1], offset: ?>>
              %subview_35 = memref.subview %subview_32[0, %arg4] [1, 32] [1, 1] : memref<1x128xf32, strided<[32000, 1], offset: ?>> to memref<1x32xf32, strided<[32000, 1], offset: ?>>
              linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_33, %subview_34 : memref<1x32xf32, strided<[768, 1], offset: ?>>, memref<32x32xf32, strided<[32000, 1], offset: ?>>) outs(%subview_35 : memref<1x32xf32, strided<[32000, 1], offset: ?>>) {
              ^bb0(%in: f32, %in_36: f32, %out: f32):
                %36 = arith.mulf %in, %in_36 : f32
                %37 = arith.addf %out, %36 : f32
                linalg.yield %37 : f32
              }
            }
          }
        }
      }
      %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%cst_6 : f32) outs(%alloc_28 : memref<1xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      }
      %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<1xi64>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%c0_i64 : i64) outs(%alloc_29 : memref<1xi64>) {
      ^bb0(%in: i64, %out: i64):
        linalg.yield %in : i64
      }
      scf.for %arg2 = %c0 to %c32000 step %c128 {
        %subview_30 = memref.subview %alloc_27[0, %arg2] [1, 128] [1, 1] : memref<1x32000xf32> to memref<1x128xf32, strided<[32000, 1], offset: ?>>
        scf.for %arg3 = %c0 to %c128 step %c32 {
          %subview_31 = memref.subview %subview_30[0, %arg3] [1, 32] [1, 1] : memref<1x128xf32, strided<[32000, 1], offset: ?>> to memref<1x32xf32, strided<[32000, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map2, #map3, #map3], iterator_types = ["parallel", "reduction"]} ins(%subview_31 : memref<1x32xf32, strided<[32000, 1], offset: ?>>) outs(%alloc_28, %alloc_29 : memref<1xf32>, memref<1xi64>) {
          ^bb0(%in: f32, %out: f32, %out_32: i64):
            %36 = linalg.index 1 : index
            %37 = affine.apply #map15(%arg2, %36, %arg3)
            %38 = arith.index_cast %37 : index to i64
            %39 = arith.cmpf ogt, %in, %out : f32
            %40 = arith.select %39, %in, %out : f32
            %41 = arith.select %39, %38, %out_32 : i64
            linalg.yield %40, %41 : f32, i64
          }
        }
      }
      %35 = memref.load %alloc_29[%c0] : memref<1xi64>
      func.call @decode(%arg0, %35) : (i64, i64) -> ()
      scf.yield %35, %32 : i64, i64
    }
    call @end(%c128_i64) : (i64) -> ()
    call @free_tokenizer() : () -> ()
    return
  }
}