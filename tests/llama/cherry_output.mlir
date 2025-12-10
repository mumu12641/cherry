// Original IR loaded from file
module {
  cherry.func private @llama_forward(%arg0: i64, %arg1: i64, %arg2: !cherry.cherry_tensor<[12x1024x768xf32]>, %arg3: !cherry.cherry_tensor<[12x1024x768xf32]>, %arg4: !cherry.cherry_tensor<[32000x768xf32]>, %arg5: !cherry.cherry_tensor<[12x768xf32]>, %arg6: !cherry.cherry_tensor<[12x768x768xf32]>, %arg7: !cherry.cherry_tensor<[12x768x768xf32]>, %arg8: !cherry.cherry_tensor<[12x768x768xf32]>, %arg9: !cherry.cherry_tensor<[12x768x768xf32]>, %arg10: !cherry.cherry_tensor<[12x768xf32]>, %arg11: !cherry.cherry_tensor<[12x768x2048xf32]>, %arg12: !cherry.cherry_tensor<[12x2048x768xf32]>, %arg13: !cherry.cherry_tensor<[12x768x2048xf32]>, %arg14: !cherry.cherry_tensor<[1x768xf32]>, %arg15: !cherry.cherry_tensor<[32000x768xf32]>) -> (!cherry.cherry_tensor<[1x32000xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) {
    %0 = cherry.constant(0 : i64) : i64
    %1 = cherry.tensor_slice %arg4[%arg0, %0] sizes [1, 768] : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %2 = cherry.constant(1 : i64) : i64
    %c12 = arith.constant 12 : index
    %3 = cherry.constant(768 : i64) : i64
    %4 = cherry.constant(12 : i64) : i64
    %5 = cherry.constant(64 : i64) : i64
    %6 = cherry.constant(1.250000e-01 : f32) : f32
    %7 = cherry.constant(1024 : i64) : i64
    %8 = cherry.constant(32000 : i64) : i64
    %9:3 = scf.for %arg16 = %c0 to %c12 step %c1 iter_args(%arg17 = %1, %arg18 = %arg2, %arg19 = %arg3) -> (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) {
      %12 = arith.index_cast %arg16 : index to i64
      %13 = cherry.tensor_slice %arg5[%12, %0] sizes [1, 768] : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
      %14 = cherry.reshape %13, %3 : (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>
      %15 = cherry.rmsnorm %arg17 scale %14 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
      %16 = cherry.tensor_slice %arg6[%12, %0, %0] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
      %17 = cherry.reshape %16, %3, %3 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
      %18 = cherry.tensor_slice %arg7[%12, %0, %0] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
      %19 = cherry.reshape %18, %3, %3 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
      %20 = cherry.tensor_slice %arg8[%12, %0, %0] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
      %21 = cherry.reshape %20, %3, %3 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
      %22 = cherry.matmul %15, %17 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
      %23 = cherry.matmul %15, %19 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
      %24 = cherry.matmul %15, %21 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
      %25 = cherry.reshape %23, %2, %2, %3 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x768xf32]>
      %26 = cherry.tensor_set_slice %arg18[%12, %arg1], %25 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[12x1024x768xf32]>
      %27 = cherry.reshape %24, %2, %2, %3 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x768xf32]>
      %28 = cherry.tensor_set_slice %arg19[%12, %arg1], %27 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[12x1024x768xf32]>
      %29 = cherry.tensor_slice %26[%12, %0, %0] sizes [1, 1024, 768] : (!cherry.cherry_tensor<[12x1024x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1024x768xf32]>
      %30 = cherry.reshape %29, %7, %3 : (!cherry.cherry_tensor<[1x1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x768xf32]>
      %31 = cherry.tensor_slice %28[%12, %0, %0] sizes [1, 1024, 768] : (!cherry.cherry_tensor<[12x1024x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1024x768xf32]>
      %32 = cherry.reshape %31, %7, %3 : (!cherry.cherry_tensor<[1x1024x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1024x768xf32]>
      scf.yield %22, %26, %28 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
    }
    %10 = cherry.reshape %arg15, %3, %8 : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x32000xf32]>
    %11 = cherry.matmul %9#0, %10 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x32000xf32]>) -> !cherry.cherry_tensor<[1x32000xf32]>
    cherry.return %11, %9#1, %9#2 : !cherry.cherry_tensor<[1x32000xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
  }
  cherry.func @host() {
    %0 = cherry.create_tensor dense<2.000000e+00> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
    %1 = cherry.create_tensor dense<3.000000e+00> : tensor<12x768xf32> -> !cherry.cherry_tensor<[12x768xf32]>
    %2 = cherry.create_tensor dense<4.000000e+00> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %3 = cherry.create_tensor dense<5.000000e+00> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %4 = cherry.create_tensor dense<6.000000e+00> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %5 = cherry.create_tensor dense<7.000000e+00> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %6 = cherry.create_tensor dense<8.000000e+00> : tensor<12x768xf32> -> !cherry.cherry_tensor<[12x768xf32]>
    %7 = cherry.create_tensor dense<9.000000e+00> : tensor<12x768x2048xf32> -> !cherry.cherry_tensor<[12x768x2048xf32]>
    %8 = cherry.create_tensor dense<1.000000e+01> : tensor<12x2048x768xf32> -> !cherry.cherry_tensor<[12x2048x768xf32]>
    %9 = cherry.create_tensor dense<1.100000e+01> : tensor<12x768x2048xf32> -> !cherry.cherry_tensor<[12x768x2048xf32]>
    %10 = cherry.create_tensor dense<1.200000e+01> : tensor<1x768xf32> -> !cherry.cherry_tensor<[1x768xf32]>
    %11 = cherry.create_tensor dense<1.300000e+01> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
    %12 = cherry.create_tensor dense<0.000000e+00> : tensor<12x1024x768xf32> -> !cherry.cherry_tensor<[12x1024x768xf32]>
    %13 = cherry.create_tensor dense<0.000000e+00> : tensor<12x1024x768xf32> -> !cherry.cherry_tensor<[12x1024x768xf32]>
    %14 = cherry.constant(1 : i64) : i64
    %15 = cherry.constant(0 : i64) : i64
    %16 = cherry.constant(10 : i64) : i64
    %17:4 = scf.while (%arg0 = %14, %arg1 = %15, %arg2 = %12, %arg3 = %13) : (i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) -> (i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) {
      %18 = arith.cmpi slt, %arg1, %16 : i64
      scf.condition(%18) %arg0, %arg1, %arg2, %arg3 : i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
    } do {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: !cherry.cherry_tensor<[12x1024x768xf32]>, %arg3: !cherry.cherry_tensor<[12x1024x768xf32]>):
      %18:3 = cherry.call @llama_forward(%arg0, %arg1, %arg2, %arg3, %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11) : (i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[32000x768xf32]>, !cherry.cherry_tensor<[12x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768x768xf32]>, !cherry.cherry_tensor<[12x768xf32]>, !cherry.cherry_tensor<[12x768x2048xf32]>, !cherry.cherry_tensor<[12x2048x768xf32]>, !cherry.cherry_tensor<[12x768x2048xf32]>, !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[32000x768xf32]>) -> (!cherry.cherry_tensor<[1x32000xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>)
      %19 = cherry.argmax %18#0 dim 1 : (!cherry.cherry_tensor<[1x32000xf32]>) -> !cherry.cherry_tensor<[1xi64]>
      cherry.print %19 : !cherry.cherry_tensor<[1xi64]>
      %20 = cherry.constant(0 : i64) : i64
      %21 = cherry.tensor_get %19[%20] : (!cherry.cherry_tensor<[1xi64]>, i64) -> i64
      %22 = cherry.constant(1 : i64) : i64
      %23 = arith.addi %arg1, %22 : i64
      scf.yield %21, %23, %18#1, %18#2 : i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
    }
    cherry.return
  }
}

// ==========================================
// Phase: Inliner
// ==========================================
module {
  cherry.func @host() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %0 = cherry.create_tensor dense<2.000000e+00> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
    %1 = cherry.create_tensor dense<3.000000e+00> : tensor<12x768xf32> -> !cherry.cherry_tensor<[12x768xf32]>
    %2 = cherry.create_tensor dense<4.000000e+00> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %3 = cherry.create_tensor dense<5.000000e+00> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %4 = cherry.create_tensor dense<6.000000e+00> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %5 = cherry.create_tensor dense<1.300000e+01> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
    %6 = cherry.create_tensor dense<0.000000e+00> : tensor<12x1024x768xf32> -> !cherry.cherry_tensor<[12x1024x768xf32]>
    %7 = cherry.create_tensor dense<0.000000e+00> : tensor<12x1024x768xf32> -> !cherry.cherry_tensor<[12x1024x768xf32]>
    %8 = cherry.constant(1 : i64) : i64
    %9 = cherry.constant(0 : i64) : i64
    %10 = cherry.constant(10 : i64) : i64
    %11:4 = scf.while (%arg0 = %8, %arg1 = %9, %arg2 = %6, %arg3 = %7) : (i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) -> (i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) {
      %12 = arith.cmpi slt, %arg1, %10 : i64
      scf.condition(%12) %arg0, %arg1, %arg2, %arg3 : i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
    } do {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: !cherry.cherry_tensor<[12x1024x768xf32]>, %arg3: !cherry.cherry_tensor<[12x1024x768xf32]>):
      %12 = cherry.constant(0 : i64) : i64
      %13 = cherry.tensor_slice %0[%arg0, %12] sizes [1, 768] : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
      %14 = cherry.constant(1 : i64) : i64
      %15 = cherry.constant(768 : i64) : i64
      %16 = cherry.constant(32000 : i64) : i64
      %17:3 = scf.for %arg4 = %c0 to %c12 step %c1 iter_args(%arg5 = %13, %arg6 = %arg2, %arg7 = %arg3) -> (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) {
        %25 = arith.index_cast %arg4 : index to i64
        %26 = cherry.tensor_slice %1[%25, %12] sizes [1, 768] : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %27 = cherry.reshape %26, %15 : (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>
        %28 = cherry.rmsnorm %arg5 scale %27 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
        %29 = cherry.tensor_slice %2[%25, %12, %12] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %30 = cherry.reshape %29, %15, %15 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %31 = cherry.tensor_slice %3[%25, %12, %12] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %32 = cherry.reshape %31, %15, %15 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %33 = cherry.tensor_slice %4[%25, %12, %12] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %34 = cherry.reshape %33, %15, %15 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %35 = cherry.matmul %28, %30 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %36 = cherry.matmul %28, %32 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %37 = cherry.matmul %28, %34 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %38 = cherry.reshape %36, %14, %14, %15 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x768xf32]>
        %39 = cherry.tensor_set_slice %arg6[%25, %arg1], %38 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[12x1024x768xf32]>
        %40 = cherry.reshape %37, %14, %14, %15 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x768xf32]>
        %41 = cherry.tensor_set_slice %arg7[%25, %arg1], %40 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[12x1024x768xf32]>
        scf.yield %35, %39, %41 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
      }
      %18 = cherry.reshape %5, %15, %16 : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x32000xf32]>
      %19 = cherry.matmul %17#0, %18 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x32000xf32]>) -> !cherry.cherry_tensor<[1x32000xf32]>
      %20 = cherry.argmax %19 dim 1 : (!cherry.cherry_tensor<[1x32000xf32]>) -> !cherry.cherry_tensor<[1xi64]>
      cherry.print %20 : !cherry.cherry_tensor<[1xi64]>
      %21 = cherry.constant(0 : i64) : i64
      %22 = cherry.tensor_get %20[%21] : (!cherry.cherry_tensor<[1xi64]>, i64) -> i64
      %23 = cherry.constant(1 : i64) : i64
      %24 = arith.addi %arg1, %23 : i64
      scf.yield %22, %24, %17#1, %17#2 : i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
    }
    cherry.return
  }
}

// ==========================================
// Phase: Shape Inference
// ==========================================
module {
  cherry.func @host() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %0 = cherry.create_tensor dense<2.000000e+00> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
    %1 = cherry.create_tensor dense<3.000000e+00> : tensor<12x768xf32> -> !cherry.cherry_tensor<[12x768xf32]>
    %2 = cherry.create_tensor dense<4.000000e+00> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %3 = cherry.create_tensor dense<5.000000e+00> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %4 = cherry.create_tensor dense<6.000000e+00> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %5 = cherry.create_tensor dense<1.300000e+01> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
    %6 = cherry.create_tensor dense<0.000000e+00> : tensor<12x1024x768xf32> -> !cherry.cherry_tensor<[12x1024x768xf32]>
    %7 = cherry.create_tensor dense<0.000000e+00> : tensor<12x1024x768xf32> -> !cherry.cherry_tensor<[12x1024x768xf32]>
    %8 = cherry.constant(1 : i64) : i64
    %9 = cherry.constant(0 : i64) : i64
    %10 = cherry.constant(10 : i64) : i64
    %11:4 = scf.while (%arg0 = %8, %arg1 = %9, %arg2 = %6, %arg3 = %7) : (i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) -> (i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) {
      %12 = arith.cmpi slt, %arg1, %10 : i64
      scf.condition(%12) %arg0, %arg1, %arg2, %arg3 : i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
    } do {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: !cherry.cherry_tensor<[12x1024x768xf32]>, %arg3: !cherry.cherry_tensor<[12x1024x768xf32]>):
      %12 = cherry.constant(0 : i64) : i64
      %13 = cherry.tensor_slice %0[%arg0, %12] sizes [1, 768] : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
      %14 = cherry.constant(1 : i64) : i64
      %15 = cherry.constant(768 : i64) : i64
      %16 = cherry.constant(32000 : i64) : i64
      %17:3 = scf.for %arg4 = %c0 to %c12 step %c1 iter_args(%arg5 = %13, %arg6 = %arg2, %arg7 = %arg3) -> (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) {
        %25 = arith.index_cast %arg4 : index to i64
        %26 = cherry.tensor_slice %1[%25, %12] sizes [1, 768] : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %27 = cherry.reshape %26, %15 : (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>
        %28 = cherry.rmsnorm %arg5 scale %27 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
        %29 = cherry.tensor_slice %2[%25, %12, %12] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %30 = cherry.reshape %29, %15, %15 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %31 = cherry.tensor_slice %3[%25, %12, %12] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %32 = cherry.reshape %31, %15, %15 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %33 = cherry.tensor_slice %4[%25, %12, %12] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %34 = cherry.reshape %33, %15, %15 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %35 = cherry.matmul %28, %30 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %36 = cherry.matmul %28, %32 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %37 = cherry.matmul %28, %34 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %38 = cherry.reshape %36, %14, %14, %15 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x768xf32]>
        %39 = cherry.tensor_set_slice %arg6[%25, %arg1], %38 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[12x1024x768xf32]>
        %40 = cherry.reshape %37, %14, %14, %15 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x768xf32]>
        %41 = cherry.tensor_set_slice %arg7[%25, %arg1], %40 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[12x1024x768xf32]>
        scf.yield %35, %39, %41 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
      }
      %18 = cherry.reshape %5, %15, %16 : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x32000xf32]>
      %19 = cherry.matmul %17#0, %18 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x32000xf32]>) -> !cherry.cherry_tensor<[1x32000xf32]>
      %20 = cherry.argmax %19 dim 1 : (!cherry.cherry_tensor<[1x32000xf32]>) -> !cherry.cherry_tensor<[1xi64]>
      cherry.print %20 : !cherry.cherry_tensor<[1xi64]>
      %21 = cherry.constant(0 : i64) : i64
      %22 = cherry.tensor_get %20[%21] : (!cherry.cherry_tensor<[1xi64]>, i64) -> i64
      %23 = cherry.constant(1 : i64) : i64
      %24 = arith.addi %arg1, %23 : i64
      scf.yield %22, %24, %17#1, %17#2 : i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
    }
    cherry.return
  }
}

// ==========================================
// Phase: Canonicalizer
// ==========================================
module {
  cherry.func @host() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %0 = cherry.create_tensor dense<2.000000e+00> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
    %1 = cherry.create_tensor dense<3.000000e+00> : tensor<12x768xf32> -> !cherry.cherry_tensor<[12x768xf32]>
    %2 = cherry.create_tensor dense<4.000000e+00> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %3 = cherry.create_tensor dense<5.000000e+00> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %4 = cherry.create_tensor dense<6.000000e+00> : tensor<12x768x768xf32> -> !cherry.cherry_tensor<[12x768x768xf32]>
    %5 = cherry.create_tensor dense<1.300000e+01> : tensor<32000x768xf32> -> !cherry.cherry_tensor<[32000x768xf32]>
    %6 = cherry.create_tensor dense<0.000000e+00> : tensor<12x1024x768xf32> -> !cherry.cherry_tensor<[12x1024x768xf32]>
    %7 = cherry.create_tensor dense<0.000000e+00> : tensor<12x1024x768xf32> -> !cherry.cherry_tensor<[12x1024x768xf32]>
    %8 = cherry.constant(1 : i64) : i64
    %9 = cherry.constant(0 : i64) : i64
    %10 = cherry.constant(10 : i64) : i64
    %11:4 = scf.while (%arg0 = %8, %arg1 = %9, %arg2 = %6, %arg3 = %7) : (i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) -> (i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) {
      %12 = arith.cmpi slt, %arg1, %10 : i64
      scf.condition(%12) %arg0, %arg1, %arg2, %arg3 : i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
    } do {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: !cherry.cherry_tensor<[12x1024x768xf32]>, %arg3: !cherry.cherry_tensor<[12x1024x768xf32]>):
      %12 = cherry.constant(0 : i64) : i64
      %13 = cherry.tensor_slice %0[%arg0, %12] sizes [1, 768] : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
      %14 = cherry.constant(1 : i64) : i64
      %15 = cherry.constant(768 : i64) : i64
      %16 = cherry.constant(32000 : i64) : i64
      %17:3 = scf.for %arg4 = %c0 to %c12 step %c1 iter_args(%arg5 = %13, %arg6 = %arg2, %arg7 = %arg3) -> (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>) {
        %25 = arith.index_cast %arg4 : index to i64
        %26 = cherry.tensor_slice %1[%25, %12] sizes [1, 768] : (!cherry.cherry_tensor<[12x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[1x768xf32]>
        %27 = cherry.reshape %26, %15 : (!cherry.cherry_tensor<[1x768xf32]>, i64) -> !cherry.cherry_tensor<[768xf32]>
        %28 = cherry.rmsnorm %arg5 scale %27 eps 9.99999974E-6 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768xf32]> -> !cherry.cherry_tensor<[1x768xf32]>
        %29 = cherry.tensor_slice %2[%25, %12, %12] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %30 = cherry.reshape %29, %15, %15 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %31 = cherry.tensor_slice %3[%25, %12, %12] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %32 = cherry.reshape %31, %15, %15 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %33 = cherry.tensor_slice %4[%25, %12, %12] sizes [1, 768, 768] : (!cherry.cherry_tensor<[12x768x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x768x768xf32]>
        %34 = cherry.reshape %33, %15, %15 : (!cherry.cherry_tensor<[1x768x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x768xf32]>
        %35 = cherry.matmul %28, %30 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %36 = cherry.matmul %28, %32 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %37 = cherry.matmul %28, %34 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x768xf32]>) -> !cherry.cherry_tensor<[1x768xf32]>
        %38 = cherry.reshape %36, %14, %14, %15 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x768xf32]>
        %39 = cherry.tensor_set_slice %arg6[%25, %arg1], %38 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[12x1024x768xf32]>
        %40 = cherry.reshape %37, %14, %14, %15 : (!cherry.cherry_tensor<[1x768xf32]>, i64, i64, i64) -> !cherry.cherry_tensor<[1x1x768xf32]>
        %41 = cherry.tensor_set_slice %arg7[%25, %arg1], %40 : (!cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[1x1x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[12x1024x768xf32]>
        scf.yield %35, %39, %41 : !cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
      }
      %18 = cherry.reshape %5, %15, %16 : (!cherry.cherry_tensor<[32000x768xf32]>, i64, i64) -> !cherry.cherry_tensor<[768x32000xf32]>
      %19 = cherry.matmul %17#0, %18 : (!cherry.cherry_tensor<[1x768xf32]>, !cherry.cherry_tensor<[768x32000xf32]>) -> !cherry.cherry_tensor<[1x32000xf32]>
      %20 = cherry.argmax %19 dim 1 : (!cherry.cherry_tensor<[1x32000xf32]>) -> !cherry.cherry_tensor<[1xi64]>
      cherry.print %20 : !cherry.cherry_tensor<[1xi64]>
      %21 = cherry.constant(0 : i64) : i64
      %22 = cherry.tensor_get %20[%21] : (!cherry.cherry_tensor<[1xi64]>, i64) -> i64
      %23 = cherry.constant(1 : i64) : i64
      %24 = arith.addi %arg1, %23 : i64
      scf.yield %22, %24, %17#1, %17#2 : i64, i64, !cherry.cherry_tensor<[12x1024x768xf32]>, !cherry.cherry_tensor<[12x1024x768xf32]>
    }
    cherry.return
  }
}

// ==========================================
// Phase: Convert to Linalg
// ==========================================
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func private @printMemrefI64(tensor<*xi64> {bufferization.access = "read"})
  func.func @host() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %cst = arith.constant dense<2.000000e+00> : tensor<32000x768xf32>
    %cst_0 = arith.constant dense<3.000000e+00> : tensor<12x768xf32>
    %cst_1 = arith.constant dense<4.000000e+00> : tensor<12x768x768xf32>
    %cst_2 = arith.constant dense<5.000000e+00> : tensor<12x768x768xf32>
    %cst_3 = arith.constant dense<6.000000e+00> : tensor<12x768x768xf32>
    %cst_4 = arith.constant dense<1.300000e+01> : tensor<32000x768xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<12x1024x768xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<12x1024x768xf32>
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c10_i64 = arith.constant 10 : i64
    %0:4 = scf.while (%arg0 = %c1_i64, %arg1 = %c0_i64, %arg2 = %cst_5, %arg3 = %cst_6) : (i64, i64, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>) -> (i64, i64, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>) {
      %1 = arith.cmpi slt, %arg1, %c10_i64 : i64
      scf.condition(%1) %arg0, %arg1, %arg2, %arg3 : i64, i64, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>
    } do {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: tensor<12x1024x768xf32>, %arg3: tensor<12x1024x768xf32>):
      %c0_i64_7 = arith.constant 0 : i64
      %1 = arith.index_cast %arg0 : i64 to index
      %2 = arith.index_cast %c0_i64_7 : i64 to index
      %extracted_slice = tensor.extract_slice %cst[%1, %2] [1, 768] [1, 1] : tensor<32000x768xf32> to tensor<1x768xf32>
      %c1_i64_8 = arith.constant 1 : i64
      %c768_i64 = arith.constant 768 : i64
      %c32000_i64 = arith.constant 32000 : i64
      %3:3 = scf.for %arg4 = %c0 to %c12 step %c1 iter_args(%arg5 = %extracted_slice, %arg6 = %arg2, %arg7 = %arg3) -> (tensor<1x768xf32>, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>) {
        %14 = arith.index_cast %arg4 : index to i64
        %15 = arith.index_cast %14 : i64 to index
        %16 = arith.index_cast %c0_i64_7 : i64 to index
        %extracted_slice_16 = tensor.extract_slice %cst_0[%15, %16] [1, 768] [1, 1] : tensor<12x768xf32> to tensor<1x768xf32>
        %c768_i64_17 = arith.constant 768 : i64
        %from_elements_18 = tensor.from_elements %c768_i64_17 : tensor<1xi64>
        %reshape_19 = tensor.reshape %extracted_slice_16(%from_elements_18) : (tensor<1x768xf32>, tensor<1xi64>) -> tensor<768xf32>
        %17 = tensor.empty() : tensor<1xf32>
        %cst_20 = arith.constant 0.000000e+00 : f32
        %18 = linalg.fill ins(%cst_20 : f32) outs(%17 : tensor<1xf32>) -> tensor<1xf32>
        %19 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg5 : tensor<1x768xf32>) outs(%18 : tensor<1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %48 = arith.mulf %in, %in : f32
          %49 = arith.addf %out, %48 : f32
          linalg.yield %49 : f32
        } -> tensor<1xf32>
        %c1_21 = arith.constant 1 : index
        %dim = tensor.dim %arg5, %c1_21 : tensor<1x768xf32>
        %20 = arith.index_cast %dim : index to i64
        %21 = arith.uitofp %20 : i64 to f32
        %cst_22 = arith.constant 9.99999974E-6 : f32
        %22 = tensor.empty() : tensor<1xf32>
        %23 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%19 : tensor<1xf32>) outs(%22 : tensor<1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %48 = arith.divf %in, %21 : f32
          %49 = arith.addf %48, %cst_22 : f32
          %50 = math.rsqrt %49 : f32
          linalg.yield %50 : f32
        } -> tensor<1xf32>
        %24 = tensor.empty() : tensor<1x768xf32>
        %25 = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%arg5, %23, %reshape_19 : tensor<1x768xf32>, tensor<1xf32>, tensor<768xf32>) outs(%24 : tensor<1x768xf32>) {
        ^bb0(%in: f32, %in_56: f32, %in_57: f32, %out: f32):
          %48 = arith.mulf %in, %in_56 : f32
          %49 = arith.mulf %48, %in_57 : f32
          linalg.yield %49 : f32
        } -> tensor<1x768xf32>
        %26 = arith.index_cast %14 : i64 to index
        %27 = arith.index_cast %c0_i64_7 : i64 to index
        %28 = arith.index_cast %c0_i64_7 : i64 to index
        %extracted_slice_23 = tensor.extract_slice %cst_1[%26, %27, %28] [1, 768, 768] [1, 1, 1] : tensor<12x768x768xf32> to tensor<1x768x768xf32>
        %c768_i64_24 = arith.constant 768 : i64
        %c768_i64_25 = arith.constant 768 : i64
        %from_elements_26 = tensor.from_elements %c768_i64_24, %c768_i64_25 : tensor<2xi64>
        %reshape_27 = tensor.reshape %extracted_slice_23(%from_elements_26) : (tensor<1x768x768xf32>, tensor<2xi64>) -> tensor<768x768xf32>
        %29 = arith.index_cast %14 : i64 to index
        %30 = arith.index_cast %c0_i64_7 : i64 to index
        %31 = arith.index_cast %c0_i64_7 : i64 to index
        %extracted_slice_28 = tensor.extract_slice %cst_2[%29, %30, %31] [1, 768, 768] [1, 1, 1] : tensor<12x768x768xf32> to tensor<1x768x768xf32>
        %c768_i64_29 = arith.constant 768 : i64
        %c768_i64_30 = arith.constant 768 : i64
        %from_elements_31 = tensor.from_elements %c768_i64_29, %c768_i64_30 : tensor<2xi64>
        %reshape_32 = tensor.reshape %extracted_slice_28(%from_elements_31) : (tensor<1x768x768xf32>, tensor<2xi64>) -> tensor<768x768xf32>
        %32 = arith.index_cast %14 : i64 to index
        %33 = arith.index_cast %c0_i64_7 : i64 to index
        %34 = arith.index_cast %c0_i64_7 : i64 to index
        %extracted_slice_33 = tensor.extract_slice %cst_3[%32, %33, %34] [1, 768, 768] [1, 1, 1] : tensor<12x768x768xf32> to tensor<1x768x768xf32>
        %c768_i64_34 = arith.constant 768 : i64
        %c768_i64_35 = arith.constant 768 : i64
        %from_elements_36 = tensor.from_elements %c768_i64_34, %c768_i64_35 : tensor<2xi64>
        %reshape_37 = tensor.reshape %extracted_slice_33(%from_elements_36) : (tensor<1x768x768xf32>, tensor<2xi64>) -> tensor<768x768xf32>
        %35 = tensor.empty() : tensor<1x768xf32>
        %cst_38 = arith.constant 0.000000e+00 : f32
        %36 = linalg.fill ins(%cst_38 : f32) outs(%35 : tensor<1x768xf32>) -> tensor<1x768xf32>
        %37 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%25, %reshape_27 : tensor<1x768xf32>, tensor<768x768xf32>) outs(%36 : tensor<1x768xf32>) {
        ^bb0(%in: f32, %in_56: f32, %out: f32):
          %48 = arith.mulf %in, %in_56 : f32
          %49 = arith.addf %out, %48 : f32
          linalg.yield %49 : f32
        } -> tensor<1x768xf32>
        %38 = tensor.empty() : tensor<1x768xf32>
        %cst_39 = arith.constant 0.000000e+00 : f32
        %39 = linalg.fill ins(%cst_39 : f32) outs(%38 : tensor<1x768xf32>) -> tensor<1x768xf32>
        %40 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%25, %reshape_32 : tensor<1x768xf32>, tensor<768x768xf32>) outs(%39 : tensor<1x768xf32>) {
        ^bb0(%in: f32, %in_56: f32, %out: f32):
          %48 = arith.mulf %in, %in_56 : f32
          %49 = arith.addf %out, %48 : f32
          linalg.yield %49 : f32
        } -> tensor<1x768xf32>
        %41 = tensor.empty() : tensor<1x768xf32>
        %cst_40 = arith.constant 0.000000e+00 : f32
        %42 = linalg.fill ins(%cst_40 : f32) outs(%41 : tensor<1x768xf32>) -> tensor<1x768xf32>
        %43 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%25, %reshape_37 : tensor<1x768xf32>, tensor<768x768xf32>) outs(%42 : tensor<1x768xf32>) {
        ^bb0(%in: f32, %in_56: f32, %out: f32):
          %48 = arith.mulf %in, %in_56 : f32
          %49 = arith.addf %out, %48 : f32
          linalg.yield %49 : f32
        } -> tensor<1x768xf32>
        %c1_i64_41 = arith.constant 1 : i64
        %c1_i64_42 = arith.constant 1 : i64
        %c768_i64_43 = arith.constant 768 : i64
        %from_elements_44 = tensor.from_elements %c1_i64_41, %c1_i64_42, %c768_i64_43 : tensor<3xi64>
        %reshape_45 = tensor.reshape %40(%from_elements_44) : (tensor<1x768xf32>, tensor<3xi64>) -> tensor<1x1x768xf32>
        %c0_46 = arith.constant 0 : index
        %c1_47 = arith.constant 1 : index
        %44 = arith.index_cast %14 : i64 to index
        %45 = arith.index_cast %arg1 : i64 to index
        %inserted_slice = tensor.insert_slice %reshape_45 into %arg6[%44, %45, 0] [1, 1, 768] [1, 1, 1] : tensor<1x1x768xf32> into tensor<12x1024x768xf32>
        %c1_i64_48 = arith.constant 1 : i64
        %c1_i64_49 = arith.constant 1 : i64
        %c768_i64_50 = arith.constant 768 : i64
        %from_elements_51 = tensor.from_elements %c1_i64_48, %c1_i64_49, %c768_i64_50 : tensor<3xi64>
        %reshape_52 = tensor.reshape %43(%from_elements_51) : (tensor<1x768xf32>, tensor<3xi64>) -> tensor<1x1x768xf32>
        %c0_53 = arith.constant 0 : index
        %c1_54 = arith.constant 1 : index
        %46 = arith.index_cast %14 : i64 to index
        %47 = arith.index_cast %arg1 : i64 to index
        %inserted_slice_55 = tensor.insert_slice %reshape_52 into %arg7[%46, %47, 0] [1, 1, 768] [1, 1, 1] : tensor<1x1x768xf32> into tensor<12x1024x768xf32>
        scf.yield %37, %inserted_slice, %inserted_slice_55 : tensor<1x768xf32>, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>
      }
      %c768_i64_9 = arith.constant 768 : i64
      %c32000_i64_10 = arith.constant 32000 : i64
      %from_elements = tensor.from_elements %c768_i64_9, %c32000_i64_10 : tensor<2xi64>
      %reshape = tensor.reshape %cst_4(%from_elements) : (tensor<32000x768xf32>, tensor<2xi64>) -> tensor<768x32000xf32>
      %4 = tensor.empty() : tensor<1x32000xf32>
      %cst_11 = arith.constant 0.000000e+00 : f32
      %5 = linalg.fill ins(%cst_11 : f32) outs(%4 : tensor<1x32000xf32>) -> tensor<1x32000xf32>
      %6 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3#0, %reshape : tensor<1x768xf32>, tensor<768x32000xf32>) outs(%5 : tensor<1x32000xf32>) {
      ^bb0(%in: f32, %in_16: f32, %out: f32):
        %14 = arith.mulf %in, %in_16 : f32
        %15 = arith.addf %out, %14 : f32
        linalg.yield %15 : f32
      } -> tensor<1x32000xf32>
      %cst_12 = arith.constant 0xFF800000 : f32
      %7 = tensor.empty() : tensor<1xf32>
      %8 = linalg.fill ins(%cst_12 : f32) outs(%7 : tensor<1xf32>) -> tensor<1xf32>
      %c0_i64_13 = arith.constant 0 : i64
      %9 = tensor.empty() : tensor<1xi64>
      %10 = linalg.fill ins(%c0_i64_13 : i64) outs(%9 : tensor<1xi64>) -> tensor<1xi64>
      %11:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%6 : tensor<1x32000xf32>) outs(%8, %10 : tensor<1xf32>, tensor<1xi64>) {
      ^bb0(%in: f32, %out: f32, %out_16: i64):
        %14 = linalg.index 1 : index
        %15 = arith.index_cast %14 : index to i64
        %16 = arith.cmpf ogt, %in, %out : f32
        %17 = arith.select %16, %in, %out : f32
        %18 = arith.select %16, %15, %out_16 : i64
        linalg.yield %17, %18 : f32, i64
      } -> (tensor<1xf32>, tensor<1xi64>)
      %cast = tensor.cast %11#1 : tensor<1xi64> to tensor<*xi64>
      func.call @printMemrefI64(%cast) : (tensor<*xi64>) -> ()
      %c0_i64_14 = arith.constant 0 : i64
      %12 = arith.index_cast %c0_i64_14 : i64 to index
      %extracted = tensor.extract %11#1[%12] : tensor<1xi64>
      %c1_i64_15 = arith.constant 1 : i64
      %13 = arith.addi %arg1, %c1_i64_15 : i64
      scf.yield %extracted, %13, %3#1, %3#2 : i64, i64, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>
    }
    return
  }
}

// ==========================================
// Phase: Linalg Tiling
// ==========================================
#map = affine_map<(d0) -> (-d0 + 1, 8)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d0 + d1)>
module {
  func.func private @printMemrefI64(tensor<*xi64> {bufferization.access = "read"})
  func.func @host() {
    %c8 = arith.constant 8 : index
    %c8_0 = arith.constant 8 : index
    %c8_1 = arith.constant 8 : index
    %c8_2 = arith.constant 8 : index
    %c8_3 = arith.constant 8 : index
    %c8_4 = arith.constant 8 : index
    %c8_5 = arith.constant 8 : index
    %c8_6 = arith.constant 8 : index
    %c8_7 = arith.constant 8 : index
    %c8_8 = arith.constant 8 : index
    %c8_9 = arith.constant 8 : index
    %c8_10 = arith.constant 8 : index
    %c8_11 = arith.constant 8 : index
    %c8_12 = arith.constant 8 : index
    %c8_13 = arith.constant 8 : index
    %c8_14 = arith.constant 8 : index
    %c8_15 = arith.constant 8 : index
    %c8_16 = arith.constant 8 : index
    %c8_17 = arith.constant 8 : index
    %c8_18 = arith.constant 8 : index
    %c8_19 = arith.constant 8 : index
    %c8_20 = arith.constant 8 : index
    %c8_21 = arith.constant 8 : index
    %c8_22 = arith.constant 8 : index
    %c8_23 = arith.constant 8 : index
    %c8_24 = arith.constant 8 : index
    %c8_25 = arith.constant 8 : index
    %c8_26 = arith.constant 8 : index
    %c8_27 = arith.constant 8 : index
    %c8_28 = arith.constant 8 : index
    %c8_29 = arith.constant 8 : index
    %c8_30 = arith.constant 8 : index
    %c8_31 = arith.constant 8 : index
    %c8_32 = arith.constant 8 : index
    %c8_33 = arith.constant 8 : index
    %c8_34 = arith.constant 8 : index
    %c8_35 = arith.constant 8 : index
    %c8_36 = arith.constant 8 : index
    %c8_37 = arith.constant 8 : index
    %c8_38 = arith.constant 8 : index
    %c8_39 = arith.constant 8 : index
    %c8_40 = arith.constant 8 : index
    %cst = arith.constant 0xFF800000 : f32
    %cst_41 = arith.constant 1.300000e+01 : f32
    %cst_42 = arith.constant 6.000000e+00 : f32
    %cst_43 = arith.constant 5.000000e+00 : f32
    %cst_44 = arith.constant 4.000000e+00 : f32
    %cst_45 = arith.constant 3.000000e+00 : f32
    %cst_46 = arith.constant 0.000000e+00 : f32
    %cst_47 = arith.constant dense<[1, 1, 768]> : tensor<3xi64>
    %cst_48 = arith.constant 7.680000e+02 : f32
    %cst_49 = arith.constant dense<2.000000e+00> : tensor<1x768xf32>
    %cst_50 = arith.constant 9.99999974E-6 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %cst_51 = arith.constant dense<0.000000e+00> : tensor<12x1024x768xf32>
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c10_i64 = arith.constant 10 : i64
    %0:4 = scf.while (%arg0 = %c1_i64, %arg1 = %c0_i64, %arg2 = %cst_51, %arg3 = %cst_51) : (i64, i64, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>) -> (i64, i64, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>) {
      %1 = arith.cmpi slt, %arg1, %c10_i64 : i64
      scf.condition(%1) %arg0, %arg1, %arg2, %arg3 : i64, i64, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>
    } do {
    ^bb0(%arg0: i64, %arg1: i64, %arg2: tensor<12x1024x768xf32>, %arg3: tensor<12x1024x768xf32>):
      %1:3 = scf.for %arg4 = %c0 to %c12 step %c1 iter_args(%arg5 = %cst_49, %arg6 = %arg2, %arg7 = %arg3) -> (tensor<1x768xf32>, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>) {
        %11 = arith.index_cast %arg4 : index to i64
        %12 = tensor.empty() : tensor<1xf32>
        %c0_77 = arith.constant 0 : index
        %c1_78 = arith.constant 1 : index
        %c8_79 = arith.constant 8 : index
        %13 = scf.for %arg8 = %c0_77 to %c1_78 step %c8_79 iter_args(%arg9 = %12) -> (tensor<1xf32>) {
          %30 = affine.min #map(%arg8)
          %extracted_slice = tensor.extract_slice %arg9[%arg8] [%30] [1] : tensor<1xf32> to tensor<?xf32>
          %31 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice : tensor<?xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_46 : f32
          } -> tensor<?xf32>
          %inserted_slice_139 = tensor.insert_slice %31 into %arg9[%arg8] [%30] [1] : tensor<?xf32> into tensor<1xf32>
          scf.yield %inserted_slice_139 : tensor<1xf32>
        }
        %c0_80 = arith.constant 0 : index
        %c1_81 = arith.constant 1 : index
        %c8_82 = arith.constant 8 : index
        %c0_83 = arith.constant 0 : index
        %c768_84 = arith.constant 768 : index
        %c8_85 = arith.constant 8 : index
        %14 = scf.for %arg8 = %c0_80 to %c1_81 step %c8_82 iter_args(%arg9 = %13) -> (tensor<1xf32>) {
          %30 = scf.for %arg10 = %c0_83 to %c768_84 step %c8_85 iter_args(%arg11 = %arg9) -> (tensor<1xf32>) {
            %31 = affine.min #map(%arg8)
            %32 = affine.min #map(%arg8)
            %extracted_slice = tensor.extract_slice %arg5[%arg8, %arg10] [%31, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %extracted_slice_139 = tensor.extract_slice %arg11[%arg8] [%32] [1] : tensor<1xf32> to tensor<?xf32>
            %33 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice : tensor<?x8xf32>) outs(%extracted_slice_139 : tensor<?xf32>) {
            ^bb0(%in: f32, %out: f32):
              %34 = arith.mulf %in, %in : f32
              %35 = arith.addf %out, %34 : f32
              linalg.yield %35 : f32
            } -> tensor<?xf32>
            %inserted_slice_140 = tensor.insert_slice %33 into %arg11[%arg8] [%32] [1] : tensor<?xf32> into tensor<1xf32>
            scf.yield %inserted_slice_140 : tensor<1xf32>
          }
          scf.yield %30 : tensor<1xf32>
        }
        %15 = tensor.empty() : tensor<1x768xf32>
        %c0_86 = arith.constant 0 : index
        %c1_87 = arith.constant 1 : index
        %c8_88 = arith.constant 8 : index
        %c0_89 = arith.constant 0 : index
        %c768_90 = arith.constant 768 : index
        %c8_91 = arith.constant 8 : index
        %16 = scf.for %arg8 = %c0_86 to %c1_87 step %c8_88 iter_args(%arg9 = %15) -> (tensor<1x768xf32>) {
          %30 = scf.for %arg10 = %c0_89 to %c768_90 step %c8_91 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %31 = affine.min #map(%arg8)
            %32 = affine.min #map(%arg8)
            %33 = affine.min #map(%arg8)
            %extracted_slice = tensor.extract_slice %arg5[%arg8, %arg10] [%31, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %extracted_slice_139 = tensor.extract_slice %14[%arg8] [%32] [1] : tensor<1xf32> to tensor<?xf32>
            %extracted_slice_140 = tensor.extract_slice %arg11[%arg8, %arg10] [%33, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %34 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_139 : tensor<?x8xf32>, tensor<?xf32>) outs(%extracted_slice_140 : tensor<?x8xf32>) {
            ^bb0(%in: f32, %in_142: f32, %out: f32):
              %35 = arith.divf %in_142, %cst_48 : f32
              %36 = arith.addf %35, %cst_50 : f32
              %37 = math.rsqrt %36 : f32
              %38 = arith.mulf %in, %37 : f32
              %39 = arith.mulf %38, %cst_45 : f32
              linalg.yield %39 : f32
            } -> tensor<?x8xf32>
            %inserted_slice_141 = tensor.insert_slice %34 into %arg11[%arg8, %arg10] [%33, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
            scf.yield %inserted_slice_141 : tensor<1x768xf32>
          }
          scf.yield %30 : tensor<1x768xf32>
        }
        %17 = tensor.empty() : tensor<1x768xf32>
        %c0_92 = arith.constant 0 : index
        %c1_93 = arith.constant 1 : index
        %c8_94 = arith.constant 8 : index
        %c0_95 = arith.constant 0 : index
        %c768_96 = arith.constant 768 : index
        %c8_97 = arith.constant 8 : index
        %18 = scf.for %arg8 = %c0_92 to %c1_93 step %c8_94 iter_args(%arg9 = %17) -> (tensor<1x768xf32>) {
          %30 = scf.for %arg10 = %c0_95 to %c768_96 step %c8_97 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %31 = affine.min #map(%arg8)
            %extracted_slice = tensor.extract_slice %arg11[%arg8, %arg10] [%31, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %32 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice : tensor<?x8xf32>) {
            ^bb0(%out: f32):
              linalg.yield %cst_46 : f32
            } -> tensor<?x8xf32>
            %inserted_slice_139 = tensor.insert_slice %32 into %arg11[%arg8, %arg10] [%31, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
            scf.yield %inserted_slice_139 : tensor<1x768xf32>
          }
          scf.yield %30 : tensor<1x768xf32>
        }
        %c0_98 = arith.constant 0 : index
        %c1_99 = arith.constant 1 : index
        %c8_100 = arith.constant 8 : index
        %c0_101 = arith.constant 0 : index
        %c768_102 = arith.constant 768 : index
        %c8_103 = arith.constant 8 : index
        %c0_104 = arith.constant 0 : index
        %c768_105 = arith.constant 768 : index
        %c8_106 = arith.constant 8 : index
        %19 = scf.for %arg8 = %c0_98 to %c1_99 step %c8_100 iter_args(%arg9 = %18) -> (tensor<1x768xf32>) {
          %30 = scf.for %arg10 = %c0_101 to %c768_102 step %c8_103 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %31 = scf.for %arg12 = %c0_104 to %c768_105 step %c8_106 iter_args(%arg13 = %arg11) -> (tensor<1x768xf32>) {
              %32 = affine.min #map(%arg8)
              %33 = affine.min #map(%arg8)
              %extracted_slice = tensor.extract_slice %16[%arg8, %arg12] [%32, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
              %extracted_slice_139 = tensor.extract_slice %arg13[%arg8, %arg10] [%33, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
              %34 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<?x8xf32>) outs(%extracted_slice_139 : tensor<?x8xf32>) {
              ^bb0(%in: f32, %out: f32):
                %35 = arith.mulf %in, %cst_44 : f32
                %36 = arith.addf %out, %35 : f32
                linalg.yield %36 : f32
              } -> tensor<?x8xf32>
              %inserted_slice_140 = tensor.insert_slice %34 into %arg13[%arg8, %arg10] [%33, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
              scf.yield %inserted_slice_140 : tensor<1x768xf32>
            }
            scf.yield %31 : tensor<1x768xf32>
          }
          scf.yield %30 : tensor<1x768xf32>
        }
        %20 = tensor.empty() : tensor<1x768xf32>
        %c0_107 = arith.constant 0 : index
        %c1_108 = arith.constant 1 : index
        %c8_109 = arith.constant 8 : index
        %c0_110 = arith.constant 0 : index
        %c768_111 = arith.constant 768 : index
        %c8_112 = arith.constant 8 : index
        %21 = scf.for %arg8 = %c0_107 to %c1_108 step %c8_109 iter_args(%arg9 = %20) -> (tensor<1x768xf32>) {
          %30 = scf.for %arg10 = %c0_110 to %c768_111 step %c8_112 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %31 = affine.min #map(%arg8)
            %extracted_slice = tensor.extract_slice %arg11[%arg8, %arg10] [%31, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %32 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice : tensor<?x8xf32>) {
            ^bb0(%out: f32):
              linalg.yield %cst_46 : f32
            } -> tensor<?x8xf32>
            %inserted_slice_139 = tensor.insert_slice %32 into %arg11[%arg8, %arg10] [%31, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
            scf.yield %inserted_slice_139 : tensor<1x768xf32>
          }
          scf.yield %30 : tensor<1x768xf32>
        }
        %c0_113 = arith.constant 0 : index
        %c1_114 = arith.constant 1 : index
        %c8_115 = arith.constant 8 : index
        %c0_116 = arith.constant 0 : index
        %c768_117 = arith.constant 768 : index
        %c8_118 = arith.constant 8 : index
        %c0_119 = arith.constant 0 : index
        %c768_120 = arith.constant 768 : index
        %c8_121 = arith.constant 8 : index
        %22 = scf.for %arg8 = %c0_113 to %c1_114 step %c8_115 iter_args(%arg9 = %21) -> (tensor<1x768xf32>) {
          %30 = scf.for %arg10 = %c0_116 to %c768_117 step %c8_118 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %31 = scf.for %arg12 = %c0_119 to %c768_120 step %c8_121 iter_args(%arg13 = %arg11) -> (tensor<1x768xf32>) {
              %32 = affine.min #map(%arg8)
              %33 = affine.min #map(%arg8)
              %extracted_slice = tensor.extract_slice %16[%arg8, %arg12] [%32, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
              %extracted_slice_139 = tensor.extract_slice %arg13[%arg8, %arg10] [%33, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
              %34 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<?x8xf32>) outs(%extracted_slice_139 : tensor<?x8xf32>) {
              ^bb0(%in: f32, %out: f32):
                %35 = arith.mulf %in, %cst_43 : f32
                %36 = arith.addf %out, %35 : f32
                linalg.yield %36 : f32
              } -> tensor<?x8xf32>
              %inserted_slice_140 = tensor.insert_slice %34 into %arg13[%arg8, %arg10] [%33, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
              scf.yield %inserted_slice_140 : tensor<1x768xf32>
            }
            scf.yield %31 : tensor<1x768xf32>
          }
          scf.yield %30 : tensor<1x768xf32>
        }
        %23 = tensor.empty() : tensor<1x768xf32>
        %c0_122 = arith.constant 0 : index
        %c1_123 = arith.constant 1 : index
        %c8_124 = arith.constant 8 : index
        %c0_125 = arith.constant 0 : index
        %c768_126 = arith.constant 768 : index
        %c8_127 = arith.constant 8 : index
        %24 = scf.for %arg8 = %c0_122 to %c1_123 step %c8_124 iter_args(%arg9 = %23) -> (tensor<1x768xf32>) {
          %30 = scf.for %arg10 = %c0_125 to %c768_126 step %c8_127 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %31 = affine.min #map(%arg8)
            %extracted_slice = tensor.extract_slice %arg11[%arg8, %arg10] [%31, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %32 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice : tensor<?x8xf32>) {
            ^bb0(%out: f32):
              linalg.yield %cst_46 : f32
            } -> tensor<?x8xf32>
            %inserted_slice_139 = tensor.insert_slice %32 into %arg11[%arg8, %arg10] [%31, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
            scf.yield %inserted_slice_139 : tensor<1x768xf32>
          }
          scf.yield %30 : tensor<1x768xf32>
        }
        %c0_128 = arith.constant 0 : index
        %c1_129 = arith.constant 1 : index
        %c8_130 = arith.constant 8 : index
        %c0_131 = arith.constant 0 : index
        %c768_132 = arith.constant 768 : index
        %c8_133 = arith.constant 8 : index
        %c0_134 = arith.constant 0 : index
        %c768_135 = arith.constant 768 : index
        %c8_136 = arith.constant 8 : index
        %25 = scf.for %arg8 = %c0_128 to %c1_129 step %c8_130 iter_args(%arg9 = %24) -> (tensor<1x768xf32>) {
          %30 = scf.for %arg10 = %c0_131 to %c768_132 step %c8_133 iter_args(%arg11 = %arg9) -> (tensor<1x768xf32>) {
            %31 = scf.for %arg12 = %c0_134 to %c768_135 step %c8_136 iter_args(%arg13 = %arg11) -> (tensor<1x768xf32>) {
              %32 = affine.min #map(%arg8)
              %33 = affine.min #map(%arg8)
              %extracted_slice = tensor.extract_slice %16[%arg8, %arg12] [%32, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
              %extracted_slice_139 = tensor.extract_slice %arg13[%arg8, %arg10] [%33, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
              %34 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<?x8xf32>) outs(%extracted_slice_139 : tensor<?x8xf32>) {
              ^bb0(%in: f32, %out: f32):
                %35 = arith.mulf %in, %cst_42 : f32
                %36 = arith.addf %out, %35 : f32
                linalg.yield %36 : f32
              } -> tensor<?x8xf32>
              %inserted_slice_140 = tensor.insert_slice %34 into %arg13[%arg8, %arg10] [%33, 8] [1, 1] : tensor<?x8xf32> into tensor<1x768xf32>
              scf.yield %inserted_slice_140 : tensor<1x768xf32>
            }
            scf.yield %31 : tensor<1x768xf32>
          }
          scf.yield %30 : tensor<1x768xf32>
        }
        %reshape = tensor.reshape %22(%cst_47) : (tensor<1x768xf32>, tensor<3xi64>) -> tensor<1x1x768xf32>
        %26 = arith.index_cast %11 : i64 to index
        %27 = arith.index_cast %arg1 : i64 to index
        %inserted_slice = tensor.insert_slice %reshape into %arg6[%26, %27, 0] [1, 1, 768] [1, 1, 1] : tensor<1x1x768xf32> into tensor<12x1024x768xf32>
        %reshape_137 = tensor.reshape %25(%cst_47) : (tensor<1x768xf32>, tensor<3xi64>) -> tensor<1x1x768xf32>
        %28 = arith.index_cast %11 : i64 to index
        %29 = arith.index_cast %arg1 : i64 to index
        %inserted_slice_138 = tensor.insert_slice %reshape_137 into %arg7[%28, %29, 0] [1, 1, 768] [1, 1, 1] : tensor<1x1x768xf32> into tensor<12x1024x768xf32>
        scf.yield %19, %inserted_slice, %inserted_slice_138 : tensor<1x768xf32>, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>
      }
      %2 = tensor.empty() : tensor<1x32000xf32>
      %c0_52 = arith.constant 0 : index
      %c1_53 = arith.constant 1 : index
      %c8_54 = arith.constant 8 : index
      %c0_55 = arith.constant 0 : index
      %c32000 = arith.constant 32000 : index
      %c8_56 = arith.constant 8 : index
      %3 = scf.for %arg4 = %c0_52 to %c1_53 step %c8_54 iter_args(%arg5 = %2) -> (tensor<1x32000xf32>) {
        %11 = scf.for %arg6 = %c0_55 to %c32000 step %c8_56 iter_args(%arg7 = %arg5) -> (tensor<1x32000xf32>) {
          %12 = affine.min #map(%arg4)
          %extracted_slice = tensor.extract_slice %arg7[%arg4, %arg6] [%12, 8] [1, 1] : tensor<1x32000xf32> to tensor<?x8xf32>
          %13 = linalg.generic {indexing_maps = [#map2], iterator_types = ["parallel", "parallel"]} outs(%extracted_slice : tensor<?x8xf32>) {
          ^bb0(%out: f32):
            linalg.yield %cst_46 : f32
          } -> tensor<?x8xf32>
          %inserted_slice = tensor.insert_slice %13 into %arg7[%arg4, %arg6] [%12, 8] [1, 1] : tensor<?x8xf32> into tensor<1x32000xf32>
          scf.yield %inserted_slice : tensor<1x32000xf32>
        }
        scf.yield %11 : tensor<1x32000xf32>
      }
      %c0_57 = arith.constant 0 : index
      %c1_58 = arith.constant 1 : index
      %c8_59 = arith.constant 8 : index
      %c0_60 = arith.constant 0 : index
      %c32000_61 = arith.constant 32000 : index
      %c8_62 = arith.constant 8 : index
      %c0_63 = arith.constant 0 : index
      %c768 = arith.constant 768 : index
      %c8_64 = arith.constant 8 : index
      %4 = scf.for %arg4 = %c0_57 to %c1_58 step %c8_59 iter_args(%arg5 = %3) -> (tensor<1x32000xf32>) {
        %11 = scf.for %arg6 = %c0_60 to %c32000_61 step %c8_62 iter_args(%arg7 = %arg5) -> (tensor<1x32000xf32>) {
          %12 = scf.for %arg8 = %c0_63 to %c768 step %c8_64 iter_args(%arg9 = %arg7) -> (tensor<1x32000xf32>) {
            %13 = affine.min #map(%arg4)
            %14 = affine.min #map(%arg4)
            %extracted_slice = tensor.extract_slice %1#0[%arg4, %arg8] [%13, 8] [1, 1] : tensor<1x768xf32> to tensor<?x8xf32>
            %extracted_slice_77 = tensor.extract_slice %arg9[%arg4, %arg6] [%14, 8] [1, 1] : tensor<1x32000xf32> to tensor<?x8xf32>
            %15 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice : tensor<?x8xf32>) outs(%extracted_slice_77 : tensor<?x8xf32>) {
            ^bb0(%in: f32, %out: f32):
              %16 = arith.mulf %in, %cst_41 : f32
              %17 = arith.addf %out, %16 : f32
              linalg.yield %17 : f32
            } -> tensor<?x8xf32>
            %inserted_slice = tensor.insert_slice %15 into %arg9[%arg4, %arg6] [%14, 8] [1, 1] : tensor<?x8xf32> into tensor<1x32000xf32>
            scf.yield %inserted_slice : tensor<1x32000xf32>
          }
          scf.yield %12 : tensor<1x32000xf32>
        }
        scf.yield %11 : tensor<1x32000xf32>
      }
      %5 = tensor.empty() : tensor<1xf32>
      %c0_65 = arith.constant 0 : index
      %c1_66 = arith.constant 1 : index
      %c8_67 = arith.constant 8 : index
      %6 = scf.for %arg4 = %c0_65 to %c1_66 step %c8_67 iter_args(%arg5 = %5) -> (tensor<1xf32>) {
        %11 = affine.min #map(%arg4)
        %extracted_slice = tensor.extract_slice %arg5[%arg4] [%11] [1] : tensor<1xf32> to tensor<?xf32>
        %12 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice : tensor<?xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<?xf32>
        %inserted_slice = tensor.insert_slice %12 into %arg5[%arg4] [%11] [1] : tensor<?xf32> into tensor<1xf32>
        scf.yield %inserted_slice : tensor<1xf32>
      }
      %7 = tensor.empty() : tensor<1xi64>
      %c0_68 = arith.constant 0 : index
      %c1_69 = arith.constant 1 : index
      %c8_70 = arith.constant 8 : index
      %8 = scf.for %arg4 = %c0_68 to %c1_69 step %c8_70 iter_args(%arg5 = %7) -> (tensor<1xi64>) {
        %11 = affine.min #map(%arg4)
        %extracted_slice = tensor.extract_slice %arg5[%arg4] [%11] [1] : tensor<1xi64> to tensor<?xi64>
        %12 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%extracted_slice : tensor<?xi64>) {
        ^bb0(%out: i64):
          linalg.yield %c0_i64 : i64
        } -> tensor<?xi64>
        %inserted_slice = tensor.insert_slice %12 into %arg5[%arg4] [%11] [1] : tensor<?xi64> into tensor<1xi64>
        scf.yield %inserted_slice : tensor<1xi64>
      }
      %c0_71 = arith.constant 0 : index
      %c1_72 = arith.constant 1 : index
      %c8_73 = arith.constant 8 : index
      %c0_74 = arith.constant 0 : index
      %c32000_75 = arith.constant 32000 : index
      %c8_76 = arith.constant 8 : index
      %9:2 = scf.for %arg4 = %c0_71 to %c1_72 step %c8_73 iter_args(%arg5 = %6, %arg6 = %8) -> (tensor<1xf32>, tensor<1xi64>) {
        %11:2 = scf.for %arg7 = %c0_74 to %c32000_75 step %c8_76 iter_args(%arg8 = %arg5, %arg9 = %arg6) -> (tensor<1xf32>, tensor<1xi64>) {
          %12 = affine.min #map(%arg4)
          %13 = affine.min #map(%arg4)
          %14 = affine.min #map(%arg4)
          %extracted_slice = tensor.extract_slice %4[%arg4, %arg7] [%12, 8] [1, 1] : tensor<1x32000xf32> to tensor<?x8xf32>
          %extracted_slice_77 = tensor.extract_slice %arg8[%arg4] [%13] [1] : tensor<1xf32> to tensor<?xf32>
          %extracted_slice_78 = tensor.extract_slice %arg9[%arg4] [%14] [1] : tensor<1xi64> to tensor<?xi64>
          %15:2 = linalg.generic {indexing_maps = [#map2, #map3, #map3], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice : tensor<?x8xf32>) outs(%extracted_slice_77, %extracted_slice_78 : tensor<?xf32>, tensor<?xi64>) {
          ^bb0(%in: f32, %out: f32, %out_80: i64):
            %16 = linalg.index 1 : index
            %17 = affine.apply #map6(%16, %arg7)
            %18 = arith.index_cast %17 : index to i64
            %19 = arith.cmpf ogt, %in, %out : f32
            %20 = arith.select %19, %in, %out : f32
            %21 = arith.select %19, %18, %out_80 : i64
            linalg.yield %20, %21 : f32, i64
          } -> (tensor<?xf32>, tensor<?xi64>)
          %inserted_slice = tensor.insert_slice %15#0 into %arg8[%arg4] [%13] [1] : tensor<?xf32> into tensor<1xf32>
          %inserted_slice_79 = tensor.insert_slice %15#1 into %arg9[%arg4] [%14] [1] : tensor<?xi64> into tensor<1xi64>
          scf.yield %inserted_slice, %inserted_slice_79 : tensor<1xf32>, tensor<1xi64>
        }
        scf.yield %11#0, %11#1 : tensor<1xf32>, tensor<1xi64>
      }
      %cast = tensor.cast %9#1 : tensor<1xi64> to tensor<*xi64>
      func.call @printMemrefI64(%cast) : (tensor<*xi64>) -> ()
      %extracted = tensor.extract %9#1[%c0] : tensor<1xi64>
      %10 = arith.addi %arg1, %c1_i64 : i64
      scf.yield %extracted, %10, %1#1, %1#2 : i64, i64, tensor<12x1024x768xf32>, tensor<12x1024x768xf32>
    }
    return
  }
}

// ==========================================
// Phase: Bufferization
// ==========================================
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d0 + d1)>
module {
  memref.global "private" constant @__constant_12x1024x768xf32 : memref<12x1024x768xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x768xf32 : memref<1x768xf32> = dense<2.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3xi64 : memref<3xi64> = dense<[1, 1, 768]> {alignment = 64 : i64}
  func.func private @printMemrefI64(memref<*xi64> {bufferization.access = "read"})
  func.func @host() {
    %c10_i64 = arith.constant 10 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c12 = arith.constant 12 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 9.99999974E-6 : f32
    %cst_0 = arith.constant 7.680000e+02 : f32
    %c32000 = arith.constant 32000 : index
    %c768 = arith.constant 768 : index
    %c8 = arith.constant 8 : index
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 1.300000e+01 : f32
    %cst_3 = arith.constant 6.000000e+00 : f32
    %cst_4 = arith.constant 5.000000e+00 : f32
    %cst_5 = arith.constant 4.000000e+00 : f32
    %cst_6 = arith.constant 3.000000e+00 : f32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_3xi64 : memref<3xi64>
    %1 = memref.get_global @__constant_1x768xf32 : memref<1x768xf32>
    %2 = memref.get_global @__constant_12x1024x768xf32 : memref<12x1024x768xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
    memref.copy %2, %alloc : memref<12x1024x768xf32> to memref<12x1024x768xf32>
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
    memref.copy %2, %alloc_8 : memref<12x1024x768xf32> to memref<12x1024x768xf32>
    %3 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %4 = arith.cmpi slt, %arg0, %c10_i64 : i64
      scf.condition(%4) %arg0 : i64
    } do {
    ^bb0(%arg0: i64):
      %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
      memref.copy %1, %alloc_9 : memref<1x768xf32> to memref<1x768xf32>
      %4 = scf.for %arg1 = %c0 to %c12 step %c1 iter_args(%arg2 = %alloc_9) -> (memref<1x768xf32>) {
        %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
        linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloc_13 : memref<1xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst_7 : f32
        }
        scf.for %arg3 = %c0 to %c768 step %c8 {
          %subview_22 = memref.subview %arg2[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%subview_22 : memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%alloc_13 : memref<1xf32>) {
          ^bb0(%in: f32, %out: f32):
            %7 = arith.mulf %in, %in : f32
            %8 = arith.addf %out, %7 : f32
            linalg.yield %8 : f32
          }
        }
        %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg3 = %c0 to %c768 step %c8 {
          %subview_22 = memref.subview %arg2[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          %subview_23 = memref.subview %alloc_14[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%subview_22, %alloc_13 : memref<1x8xf32, strided<[768, 1], offset: ?>>, memref<1xf32>) outs(%subview_23 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%in: f32, %in_24: f32, %out: f32):
            %7 = arith.divf %in_24, %cst_0 : f32
            %8 = arith.addf %7, %cst : f32
            %9 = math.rsqrt %8 : f32
            %10 = arith.mulf %in, %9 : f32
            %11 = arith.mulf %10, %cst_6 : f32
            linalg.yield %11 : f32
          }
          memref.copy %subview_23, %subview_23 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg3 = %c0 to %c768 step %c8 {
          %subview_22 = memref.subview %alloc_15[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview_22 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%out: f32):
            linalg.yield %cst_7 : f32
          }
          memref.copy %subview_22, %subview_22 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        scf.for %arg3 = %c0 to %c768 step %c8 {
          scf.for %arg4 = %c0 to %c768 step %c8 {
            %subview_22 = memref.subview %alloc_14[0, %arg4] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            %subview_23 = memref.subview %alloc_15[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_22 : memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%subview_23 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
            ^bb0(%in: f32, %out: f32):
              %7 = arith.mulf %in, %cst_5 : f32
              %8 = arith.addf %out, %7 : f32
              linalg.yield %8 : f32
            }
            memref.copy %subview_23, %subview_23 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          }
        }
        %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg3 = %c0 to %c768 step %c8 {
          %subview_22 = memref.subview %alloc_16[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview_22 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%out: f32):
            linalg.yield %cst_7 : f32
          }
          memref.copy %subview_22, %subview_22 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        memref.copy %alloc_16, %alloc_17 : memref<1x768xf32> to memref<1x768xf32>
        scf.for %arg3 = %c0 to %c768 step %c8 {
          scf.for %arg4 = %c0 to %c768 step %c8 {
            %subview_22 = memref.subview %alloc_14[0, %arg4] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            %subview_23 = memref.subview %alloc_17[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_22 : memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%subview_23 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
            ^bb0(%in: f32, %out: f32):
              %7 = arith.mulf %in, %cst_4 : f32
              %8 = arith.addf %out, %7 : f32
              linalg.yield %8 : f32
            }
            memref.copy %subview_23, %subview_23 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          }
        }
        %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg3 = %c0 to %c768 step %c8 {
          %subview_22 = memref.subview %alloc_18[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview_22 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
          ^bb0(%out: f32):
            linalg.yield %cst_7 : f32
          }
          memref.copy %subview_22, %subview_22 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        memref.copy %alloc_18, %alloc_19 : memref<1x768xf32> to memref<1x768xf32>
        scf.for %arg3 = %c0 to %c768 step %c8 {
          scf.for %arg4 = %c0 to %c768 step %c8 {
            %subview_22 = memref.subview %alloc_14[0, %arg4] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            %subview_23 = memref.subview %alloc_19[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview_22 : memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%subview_23 : memref<1x8xf32, strided<[768, 1], offset: ?>>) {
            ^bb0(%in: f32, %out: f32):
              %7 = arith.mulf %in, %cst_3 : f32
              %8 = arith.addf %out, %7 : f32
              linalg.yield %8 : f32
            }
            memref.copy %subview_23, %subview_23 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          }
        }
        %reshape = memref.reshape %alloc_17(%0) : (memref<1x768xf32>, memref<3xi64>) -> memref<1x1x768xf32>
        %6 = arith.index_cast %arg0 : i64 to index
        %subview = memref.subview %alloc[%arg1, %6, 0] [1, 1, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        memref.copy %reshape, %subview : memref<1x1x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        %reshape_20 = memref.reshape %alloc_19(%0) : (memref<1x768xf32>, memref<3xi64>) -> memref<1x1x768xf32>
        %subview_21 = memref.subview %alloc_8[%arg1, %6, 0] [1, 1, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        memref.copy %reshape_20, %subview_21 : memref<1x1x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        scf.yield %alloc_15 : memref<1x768xf32>
      }
      %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x32000xf32>
      scf.for %arg1 = %c0 to %c32000 step %c8 {
        %subview = memref.subview %alloc_10[0, %arg1] [1, 8] [1, 1] : memref<1x32000xf32> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%subview : memref<1x8xf32, strided<[32000, 1], offset: ?>>) {
        ^bb0(%out: f32):
          linalg.yield %cst_7 : f32
        }
        memref.copy %subview, %subview : memref<1x8xf32, strided<[32000, 1], offset: ?>> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
      }
      scf.for %arg1 = %c0 to %c32000 step %c8 {
        scf.for %arg2 = %c0 to %c768 step %c8 {
          %subview = memref.subview %4[0, %arg2] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          %subview_13 = memref.subview %alloc_10[0, %arg1] [1, 8] [1, 1] : memref<1x32000xf32> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%subview : memref<1x8xf32, strided<[768, 1], offset: ?>>) outs(%subview_13 : memref<1x8xf32, strided<[32000, 1], offset: ?>>) {
          ^bb0(%in: f32, %out: f32):
            %6 = arith.mulf %in, %cst_2 : f32
            %7 = arith.addf %out, %6 : f32
            linalg.yield %7 : f32
          }
          memref.copy %subview_13, %subview_13 : memref<1x8xf32, strided<[32000, 1], offset: ?>> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
        }
      }
      %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloc_11 : memref<1xf32>) {
      ^bb0(%out: f32):
        linalg.yield %cst_1 : f32
      }
      %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1xi64>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%alloc_12 : memref<1xi64>) {
      ^bb0(%out: i64):
        linalg.yield %c0_i64 : i64
      }
      scf.for %arg1 = %c0 to %c32000 step %c8 {
        %subview = memref.subview %alloc_10[0, %arg1] [1, 8] [1, 1] : memref<1x32000xf32> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
        linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "reduction"]} ins(%subview : memref<1x8xf32, strided<[32000, 1], offset: ?>>) outs(%alloc_11, %alloc_12 : memref<1xf32>, memref<1xi64>) {
        ^bb0(%in: f32, %out: f32, %out_13: i64):
          %6 = linalg.index 1 : index
          %7 = affine.apply #map5(%6, %arg1)
          %8 = arith.index_cast %7 : index to i64
          %9 = arith.cmpf ogt, %in, %out : f32
          %10 = arith.select %9, %in, %out : f32
          %11 = arith.select %9, %8, %out_13 : i64
          linalg.yield %10, %11 : f32, i64
        }
      }
      %cast = memref.cast %alloc_12 : memref<1xi64> to memref<*xi64>
      func.call @printMemrefI64(%cast) : (memref<*xi64>) -> ()
      %5 = arith.addi %arg0, %c1_i64 : i64
      scf.yield %5 : i64
    }
    return
  }
}

// ==========================================
// Phase: linalg to scf
// ==========================================
#map = affine_map<(d0, d1) -> (d0 + d1)>
module {
  memref.global "private" constant @__constant_12x1024x768xf32 : memref<12x1024x768xf32> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x768xf32 : memref<1x768xf32> = dense<2.000000e+00> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3xi64 : memref<3xi64> = dense<[1, 1, 768]> {alignment = 64 : i64}
  func.func private @printMemrefI64(memref<*xi64> {bufferization.access = "read"})
  func.func @host() {
    %c10_i64 = arith.constant 10 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c12 = arith.constant 12 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 9.99999974E-6 : f32
    %cst_0 = arith.constant 7.680000e+02 : f32
    %c32000 = arith.constant 32000 : index
    %c768 = arith.constant 768 : index
    %c8 = arith.constant 8 : index
    %cst_1 = arith.constant 0xFF800000 : f32
    %cst_2 = arith.constant 1.300000e+01 : f32
    %cst_3 = arith.constant 6.000000e+00 : f32
    %cst_4 = arith.constant 5.000000e+00 : f32
    %cst_5 = arith.constant 4.000000e+00 : f32
    %cst_6 = arith.constant 3.000000e+00 : f32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_3xi64 : memref<3xi64>
    %1 = memref.get_global @__constant_1x768xf32 : memref<1x768xf32>
    %2 = memref.get_global @__constant_12x1024x768xf32 : memref<12x1024x768xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
    memref.copy %2, %alloc : memref<12x1024x768xf32> to memref<12x1024x768xf32>
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<12x1024x768xf32>
    memref.copy %2, %alloc_8 : memref<12x1024x768xf32> to memref<12x1024x768xf32>
    %3 = scf.while (%arg0 = %c0_i64) : (i64) -> i64 {
      %4 = arith.cmpi slt, %arg0, %c10_i64 : i64
      scf.condition(%4) %arg0 : i64
    } do {
    ^bb0(%arg0: i64):
      %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
      memref.copy %1, %alloc_9 : memref<1x768xf32> to memref<1x768xf32>
      %4 = scf.for %arg1 = %c0 to %c12 step %c1 iter_args(%arg2 = %alloc_9) -> (memref<1x768xf32>) {
        %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
        scf.for %arg3 = %c0 to %c1 step %c1 {
          memref.store %cst_7, %alloc_13[%arg3] : memref<1xf32>
        }
        scf.for %arg3 = %c0 to %c768 step %c8 {
          %subview_22 = memref.subview %arg2[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          scf.for %arg4 = %c0 to %c1 step %c1 {
            scf.for %arg5 = %c0 to %c8 step %c1 {
              %7 = memref.load %subview_22[%arg4, %arg5] : memref<1x8xf32, strided<[768, 1], offset: ?>>
              %8 = memref.load %alloc_13[%arg4] : memref<1xf32>
              %9 = arith.mulf %7, %7 : f32
              %10 = arith.addf %8, %9 : f32
              memref.store %10, %alloc_13[%arg4] : memref<1xf32>
            }
          }
        }
        %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg3 = %c0 to %c768 step %c8 {
          %subview_22 = memref.subview %arg2[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          %subview_23 = memref.subview %alloc_14[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          scf.for %arg4 = %c0 to %c1 step %c1 {
            scf.for %arg5 = %c0 to %c8 step %c1 {
              %7 = memref.load %subview_22[%arg4, %arg5] : memref<1x8xf32, strided<[768, 1], offset: ?>>
              %8 = memref.load %alloc_13[%arg4] : memref<1xf32>
              %9 = arith.divf %8, %cst_0 : f32
              %10 = arith.addf %9, %cst : f32
              %11 = math.rsqrt %10 : f32
              %12 = arith.mulf %7, %11 : f32
              %13 = arith.mulf %12, %cst_6 : f32
              memref.store %13, %subview_23[%arg4, %arg5] : memref<1x8xf32, strided<[768, 1], offset: ?>>
            }
          }
          memref.copy %subview_23, %subview_23 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg3 = %c0 to %c768 step %c8 {
          %subview_22 = memref.subview %alloc_15[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          scf.for %arg4 = %c0 to %c1 step %c1 {
            scf.for %arg5 = %c0 to %c8 step %c1 {
              memref.store %cst_7, %subview_22[%arg4, %arg5] : memref<1x8xf32, strided<[768, 1], offset: ?>>
            }
          }
          memref.copy %subview_22, %subview_22 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        scf.for %arg3 = %c0 to %c768 step %c8 {
          scf.for %arg4 = %c0 to %c768 step %c8 {
            %subview_22 = memref.subview %alloc_14[0, %arg4] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            %subview_23 = memref.subview %alloc_15[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            scf.for %arg5 = %c0 to %c1 step %c1 {
              scf.for %arg6 = %c0 to %c8 step %c1 {
                scf.for %arg7 = %c0 to %c8 step %c1 {
                  %7 = memref.load %subview_22[%arg5, %arg7] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                  %8 = memref.load %subview_23[%arg5, %arg6] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                  %9 = arith.mulf %7, %cst_5 : f32
                  %10 = arith.addf %8, %9 : f32
                  memref.store %10, %subview_23[%arg5, %arg6] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                }
              }
            }
            memref.copy %subview_23, %subview_23 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          }
        }
        %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg3 = %c0 to %c768 step %c8 {
          %subview_22 = memref.subview %alloc_16[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          scf.for %arg4 = %c0 to %c1 step %c1 {
            scf.for %arg5 = %c0 to %c8 step %c1 {
              memref.store %cst_7, %subview_22[%arg4, %arg5] : memref<1x8xf32, strided<[768, 1], offset: ?>>
            }
          }
          memref.copy %subview_22, %subview_22 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        memref.copy %alloc_16, %alloc_17 : memref<1x768xf32> to memref<1x768xf32>
        scf.for %arg3 = %c0 to %c768 step %c8 {
          scf.for %arg4 = %c0 to %c768 step %c8 {
            %subview_22 = memref.subview %alloc_14[0, %arg4] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            %subview_23 = memref.subview %alloc_17[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            scf.for %arg5 = %c0 to %c1 step %c1 {
              scf.for %arg6 = %c0 to %c8 step %c1 {
                scf.for %arg7 = %c0 to %c8 step %c1 {
                  %7 = memref.load %subview_22[%arg5, %arg7] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                  %8 = memref.load %subview_23[%arg5, %arg6] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                  %9 = arith.mulf %7, %cst_4 : f32
                  %10 = arith.addf %8, %9 : f32
                  memref.store %10, %subview_23[%arg5, %arg6] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                }
              }
            }
            memref.copy %subview_23, %subview_23 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          }
        }
        %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        scf.for %arg3 = %c0 to %c768 step %c8 {
          %subview_22 = memref.subview %alloc_18[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          scf.for %arg4 = %c0 to %c1 step %c1 {
            scf.for %arg5 = %c0 to %c8 step %c1 {
              memref.store %cst_7, %subview_22[%arg4, %arg5] : memref<1x8xf32, strided<[768, 1], offset: ?>>
            }
          }
          memref.copy %subview_22, %subview_22 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
        }
        %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<1x768xf32>
        memref.copy %alloc_18, %alloc_19 : memref<1x768xf32> to memref<1x768xf32>
        scf.for %arg3 = %c0 to %c768 step %c8 {
          scf.for %arg4 = %c0 to %c768 step %c8 {
            %subview_22 = memref.subview %alloc_14[0, %arg4] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            %subview_23 = memref.subview %alloc_19[0, %arg3] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
            scf.for %arg5 = %c0 to %c1 step %c1 {
              scf.for %arg6 = %c0 to %c8 step %c1 {
                scf.for %arg7 = %c0 to %c8 step %c1 {
                  %7 = memref.load %subview_22[%arg5, %arg7] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                  %8 = memref.load %subview_23[%arg5, %arg6] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                  %9 = arith.mulf %7, %cst_3 : f32
                  %10 = arith.addf %8, %9 : f32
                  memref.store %10, %subview_23[%arg5, %arg6] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                }
              }
            }
            memref.copy %subview_23, %subview_23 : memref<1x8xf32, strided<[768, 1], offset: ?>> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          }
        }
        %reshape = memref.reshape %alloc_17(%0) : (memref<1x768xf32>, memref<3xi64>) -> memref<1x1x768xf32>
        %6 = arith.index_cast %arg0 : i64 to index
        %subview = memref.subview %alloc[%arg1, %6, 0] [1, 1, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        memref.copy %reshape, %subview : memref<1x1x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        %reshape_20 = memref.reshape %alloc_19(%0) : (memref<1x768xf32>, memref<3xi64>) -> memref<1x1x768xf32>
        %subview_21 = memref.subview %alloc_8[%arg1, %6, 0] [1, 1, 768] [1, 1, 1] : memref<12x1024x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        memref.copy %reshape_20, %subview_21 : memref<1x1x768xf32> to memref<1x1x768xf32, strided<[786432, 768, 1], offset: ?>>
        scf.yield %alloc_15 : memref<1x768xf32>
      }
      %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x32000xf32>
      scf.for %arg1 = %c0 to %c32000 step %c8 {
        %subview = memref.subview %alloc_10[0, %arg1] [1, 8] [1, 1] : memref<1x32000xf32> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
        scf.for %arg2 = %c0 to %c1 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            memref.store %cst_7, %subview[%arg2, %arg3] : memref<1x8xf32, strided<[32000, 1], offset: ?>>
          }
        }
        memref.copy %subview, %subview : memref<1x8xf32, strided<[32000, 1], offset: ?>> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
      }
      scf.for %arg1 = %c0 to %c32000 step %c8 {
        scf.for %arg2 = %c0 to %c768 step %c8 {
          %subview = memref.subview %4[0, %arg2] [1, 8] [1, 1] : memref<1x768xf32> to memref<1x8xf32, strided<[768, 1], offset: ?>>
          %subview_13 = memref.subview %alloc_10[0, %arg1] [1, 8] [1, 1] : memref<1x32000xf32> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
          scf.for %arg3 = %c0 to %c1 step %c1 {
            scf.for %arg4 = %c0 to %c8 step %c1 {
              scf.for %arg5 = %c0 to %c8 step %c1 {
                %6 = memref.load %subview[%arg3, %arg5] : memref<1x8xf32, strided<[768, 1], offset: ?>>
                %7 = memref.load %subview_13[%arg3, %arg4] : memref<1x8xf32, strided<[32000, 1], offset: ?>>
                %8 = arith.mulf %6, %cst_2 : f32
                %9 = arith.addf %7, %8 : f32
                memref.store %9, %subview_13[%arg3, %arg4] : memref<1x8xf32, strided<[32000, 1], offset: ?>>
              }
            }
          }
          memref.copy %subview_13, %subview_13 : memref<1x8xf32, strided<[32000, 1], offset: ?>> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
        }
      }
      %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
      scf.for %arg1 = %c0 to %c1 step %c1 {
        memref.store %cst_1, %alloc_11[%arg1] : memref<1xf32>
      }
      %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1xi64>
      scf.for %arg1 = %c0 to %c1 step %c1 {
        memref.store %c0_i64, %alloc_12[%arg1] : memref<1xi64>
      }
      scf.for %arg1 = %c0 to %c32000 step %c8 {
        %subview = memref.subview %alloc_10[0, %arg1] [1, 8] [1, 1] : memref<1x32000xf32> to memref<1x8xf32, strided<[32000, 1], offset: ?>>
        scf.for %arg2 = %c0 to %c1 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            %6 = memref.load %subview[%arg2, %arg3] : memref<1x8xf32, strided<[32000, 1], offset: ?>>
            %7 = memref.load %alloc_11[%arg2] : memref<1xf32>
            %8 = memref.load %alloc_12[%arg2] : memref<1xi64>
            %9 = affine.apply #map(%arg3, %arg1)
            %10 = arith.index_cast %9 : index to i64
            %11 = arith.cmpf ogt, %6, %7 : f32
            %12 = arith.select %11, %6, %7 : f32
            %13 = arith.select %11, %10, %8 : i64
            memref.store %12, %alloc_11[%arg2] : memref<1xf32>
            memref.store %13, %alloc_12[%arg2] : memref<1xi64>
          }
        }
      }
      %cast = memref.cast %alloc_12 : memref<1xi64> to memref<*xi64>
      func.call @printMemrefI64(%cast) : (memref<*xi64>) -> ()
      %5 = arith.addi %arg0, %c1_i64 : i64
      scf.yield %5 : i64
    }
    return
  }
}

// ==========================================
// Phase: lower to llvm
// ==========================================
module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.mlir.global private constant @__constant_12x1024x768xf32(dense<0.000000e+00> : tensor<12x1024x768xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<12 x array<1024 x array<768 x f32>>>
  llvm.mlir.global private constant @__constant_1x768xf32(dense<2.000000e+00> : tensor<1x768xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<1 x array<768 x f32>>
  llvm.mlir.global private constant @__constant_3xi64(dense<[1, 1, 768]> : tensor<3xi64>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<3 x i64>
  llvm.func @printMemrefI64(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @host() {
    %0 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.addressof @__constant_12x1024x768xf32 : !llvm.ptr
    %3 = llvm.mlir.constant(786432 : index) : i64
    %4 = llvm.mlir.constant(1024 : index) : i64
    %5 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.mlir.addressof @__constant_1x768xf32 : !llvm.ptr
    %7 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.mlir.constant(10 : i64) : i64
    %9 = llvm.mlir.constant(0 : i64) : i64
    %10 = llvm.mlir.constant(1 : i64) : i64
    %11 = llvm.mlir.constant(12 : index) : i64
    %12 = llvm.mlir.constant(1 : index) : i64
    %13 = llvm.mlir.constant(0 : index) : i64
    %14 = llvm.mlir.constant(9.99999974E-6 : f32) : f32
    %15 = llvm.mlir.constant(7.680000e+02 : f32) : f32
    %16 = llvm.mlir.constant(32000 : index) : i64
    %17 = llvm.mlir.constant(768 : index) : i64
    %18 = llvm.mlir.constant(8 : index) : i64
    %19 = llvm.mlir.constant(0xFF800000 : f32) : f32
    %20 = llvm.mlir.constant(1.300000e+01 : f32) : f32
    %21 = llvm.mlir.constant(6.000000e+00 : f32) : f32
    %22 = llvm.mlir.constant(5.000000e+00 : f32) : f32
    %23 = llvm.mlir.constant(4.000000e+00 : f32) : f32
    %24 = llvm.mlir.constant(3.000000e+00 : f32) : f32
    %25 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %26 = llvm.mlir.zero : !llvm.ptr
    %27 = llvm.getelementptr %6[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x array<768 x f32>>
    %28 = llvm.getelementptr %2[0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<12 x array<1024 x array<768 x f32>>>
    %29 = llvm.getelementptr %26[9437184] : (!llvm.ptr) -> !llvm.ptr, f32
    %30 = llvm.ptrtoint %29 : !llvm.ptr to i64
    %31 = llvm.add %30, %1 : i64
    %32 = llvm.call @malloc(%31) : (i64) -> !llvm.ptr
    %33 = llvm.ptrtoint %32 : !llvm.ptr to i64
    %34 = llvm.sub %1, %12 : i64
    %35 = llvm.add %33, %34 : i64
    %36 = llvm.urem %35, %1  : i64
    %37 = llvm.sub %35, %36 : i64
    %38 = llvm.inttoptr %37 : i64 to !llvm.ptr
    %39 = llvm.mul %11, %12 : i64
    %40 = llvm.mul %39, %4 : i64
    %41 = llvm.mul %40, %17 : i64
    %42 = llvm.getelementptr %26[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %43 = llvm.ptrtoint %42 : !llvm.ptr to i64
    %44 = llvm.mul %41, %43 : i64
    "llvm.intr.memcpy"(%38, %28, %44) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %45 = llvm.call @malloc(%31) : (i64) -> !llvm.ptr
    %46 = llvm.ptrtoint %45 : !llvm.ptr to i64
    %47 = llvm.add %46, %34 : i64
    %48 = llvm.urem %47, %1  : i64
    %49 = llvm.sub %47, %48 : i64
    %50 = llvm.inttoptr %49 : i64 to !llvm.ptr
    "llvm.intr.memcpy"(%50, %28, %44) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb1(%9 : i64)
  ^bb1(%51: i64):  // 2 preds: ^bb0, ^bb137
    %52 = llvm.icmp "slt" %51, %8 : i64
    llvm.cond_br %52, ^bb2(%51 : i64), ^bb138
  ^bb2(%53: i64):  // pred: ^bb1
    %54 = llvm.getelementptr %26[768] : (!llvm.ptr) -> !llvm.ptr, f32
    %55 = llvm.ptrtoint %54 : !llvm.ptr to i64
    %56 = llvm.add %55, %1 : i64
    %57 = llvm.call @malloc(%56) : (i64) -> !llvm.ptr
    %58 = llvm.ptrtoint %57 : !llvm.ptr to i64
    %59 = llvm.add %58, %34 : i64
    %60 = llvm.urem %59, %1  : i64
    %61 = llvm.sub %59, %60 : i64
    %62 = llvm.inttoptr %61 : i64 to !llvm.ptr
    %63 = llvm.insertvalue %57, %5[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %64 = llvm.insertvalue %62, %63[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.insertvalue %13, %64[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.insertvalue %12, %65[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.insertvalue %17, %66[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.insertvalue %17, %67[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %69 = llvm.insertvalue %12, %68[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.mul %12, %12 : i64
    %71 = llvm.mul %70, %17 : i64
    %72 = llvm.mul %71, %43 : i64
    "llvm.intr.memcpy"(%62, %27, %72) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb3(%13, %69 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb3(%73: i64, %74: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>):  // 2 preds: ^bb2, ^bb97
    %75 = llvm.icmp "slt" %73, %11 : i64
    llvm.cond_br %75, ^bb4, ^bb98
  ^bb4:  // pred: ^bb3
    %76 = llvm.add %43, %1 : i64
    %77 = llvm.call @malloc(%76) : (i64) -> !llvm.ptr
    %78 = llvm.ptrtoint %77 : !llvm.ptr to i64
    %79 = llvm.add %78, %34 : i64
    %80 = llvm.urem %79, %1  : i64
    %81 = llvm.sub %79, %80 : i64
    %82 = llvm.inttoptr %81 : i64 to !llvm.ptr
    llvm.br ^bb5(%13 : i64)
  ^bb5(%83: i64):  // 2 preds: ^bb4, ^bb6
    %84 = llvm.icmp "slt" %83, %12 : i64
    llvm.cond_br %84, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %85 = llvm.getelementptr %82[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %25, %85 : f32, !llvm.ptr
    %86 = llvm.add %83, %12 : i64
    llvm.br ^bb5(%86 : i64)
  ^bb7:  // pred: ^bb5
    llvm.br ^bb8(%13 : i64)
  ^bb8(%87: i64):  // 2 preds: ^bb7, ^bb15
    %88 = llvm.icmp "slt" %87, %17 : i64
    llvm.cond_br %88, ^bb9, ^bb16
  ^bb9:  // pred: ^bb8
    %89 = llvm.extractvalue %74[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb10(%13 : i64)
  ^bb10(%90: i64):  // 2 preds: ^bb9, ^bb14
    %91 = llvm.icmp "slt" %90, %12 : i64
    llvm.cond_br %91, ^bb11, ^bb15
  ^bb11:  // pred: ^bb10
    llvm.br ^bb12(%13 : i64)
  ^bb12(%92: i64):  // 2 preds: ^bb11, ^bb13
    %93 = llvm.icmp "slt" %92, %18 : i64
    llvm.cond_br %93, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    %94 = llvm.getelementptr %89[%87] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %95 = llvm.mul %90, %17 : i64
    %96 = llvm.add %95, %92 : i64
    %97 = llvm.getelementptr %94[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %98 = llvm.load %97 : !llvm.ptr -> f32
    %99 = llvm.getelementptr %82[%90] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %100 = llvm.load %99 : !llvm.ptr -> f32
    %101 = llvm.fmul %98, %98  : f32
    %102 = llvm.fadd %100, %101  : f32
    llvm.store %102, %99 : f32, !llvm.ptr
    %103 = llvm.add %92, %12 : i64
    llvm.br ^bb12(%103 : i64)
  ^bb14:  // pred: ^bb12
    %104 = llvm.add %90, %12 : i64
    llvm.br ^bb10(%104 : i64)
  ^bb15:  // pred: ^bb10
    %105 = llvm.add %87, %18 : i64
    llvm.br ^bb8(%105 : i64)
  ^bb16:  // pred: ^bb8
    %106 = llvm.call @malloc(%56) : (i64) -> !llvm.ptr
    %107 = llvm.ptrtoint %106 : !llvm.ptr to i64
    %108 = llvm.add %107, %34 : i64
    %109 = llvm.urem %108, %1  : i64
    %110 = llvm.sub %108, %109 : i64
    %111 = llvm.inttoptr %110 : i64 to !llvm.ptr
    llvm.br ^bb17(%13 : i64)
  ^bb17(%112: i64):  // 2 preds: ^bb16, ^bb24
    %113 = llvm.icmp "slt" %112, %17 : i64
    llvm.cond_br %113, ^bb18, ^bb25
  ^bb18:  // pred: ^bb17
    %114 = llvm.extractvalue %74[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb19(%13 : i64)
  ^bb19(%115: i64):  // 2 preds: ^bb18, ^bb23
    %116 = llvm.icmp "slt" %115, %12 : i64
    llvm.cond_br %116, ^bb20, ^bb24
  ^bb20:  // pred: ^bb19
    llvm.br ^bb21(%13 : i64)
  ^bb21(%117: i64):  // 2 preds: ^bb20, ^bb22
    %118 = llvm.icmp "slt" %117, %18 : i64
    llvm.cond_br %118, ^bb22, ^bb23
  ^bb22:  // pred: ^bb21
    %119 = llvm.getelementptr %114[%112] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %120 = llvm.mul %115, %17 : i64
    %121 = llvm.add %120, %117 : i64
    %122 = llvm.getelementptr %119[%121] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %123 = llvm.load %122 : !llvm.ptr -> f32
    %124 = llvm.getelementptr %82[%115] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %125 = llvm.load %124 : !llvm.ptr -> f32
    %126 = llvm.fdiv %125, %15  : f32
    %127 = llvm.fadd %126, %14  : f32
    %128 = llvm.intr.sqrt(%127)  : (f32) -> f32
    %129 = llvm.fdiv %0, %128  : f32
    %130 = llvm.fmul %123, %129  : f32
    %131 = llvm.fmul %130, %24  : f32
    %132 = llvm.getelementptr %111[%112] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %133 = llvm.getelementptr %132[%121] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %131, %133 : f32, !llvm.ptr
    %134 = llvm.add %117, %12 : i64
    llvm.br ^bb21(%134 : i64)
  ^bb23:  // pred: ^bb21
    %135 = llvm.add %115, %12 : i64
    llvm.br ^bb19(%135 : i64)
  ^bb24:  // pred: ^bb19
    %136 = llvm.mul %70, %18 : i64
    %137 = llvm.mul %136, %43 : i64
    %138 = llvm.getelementptr %111[%112] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%138, %138, %137) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %139 = llvm.add %112, %18 : i64
    llvm.br ^bb17(%139 : i64)
  ^bb25:  // pred: ^bb17
    %140 = llvm.call @malloc(%56) : (i64) -> !llvm.ptr
    %141 = llvm.ptrtoint %140 : !llvm.ptr to i64
    %142 = llvm.add %141, %34 : i64
    %143 = llvm.urem %142, %1  : i64
    %144 = llvm.sub %142, %143 : i64
    %145 = llvm.inttoptr %144 : i64 to !llvm.ptr
    %146 = llvm.insertvalue %140, %5[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %147 = llvm.insertvalue %145, %146[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %148 = llvm.insertvalue %13, %147[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %149 = llvm.insertvalue %12, %148[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %150 = llvm.insertvalue %17, %149[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %151 = llvm.insertvalue %17, %150[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %152 = llvm.insertvalue %12, %151[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb26(%13 : i64)
  ^bb26(%153: i64):  // 2 preds: ^bb25, ^bb33
    %154 = llvm.icmp "slt" %153, %17 : i64
    llvm.cond_br %154, ^bb27, ^bb34
  ^bb27:  // pred: ^bb26
    llvm.br ^bb28(%13 : i64)
  ^bb28(%155: i64):  // 2 preds: ^bb27, ^bb32
    %156 = llvm.icmp "slt" %155, %12 : i64
    llvm.cond_br %156, ^bb29, ^bb33
  ^bb29:  // pred: ^bb28
    llvm.br ^bb30(%13 : i64)
  ^bb30(%157: i64):  // 2 preds: ^bb29, ^bb31
    %158 = llvm.icmp "slt" %157, %18 : i64
    llvm.cond_br %158, ^bb31, ^bb32
  ^bb31:  // pred: ^bb30
    %159 = llvm.getelementptr %145[%153] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %160 = llvm.mul %155, %17 : i64
    %161 = llvm.add %160, %157 : i64
    %162 = llvm.getelementptr %159[%161] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %25, %162 : f32, !llvm.ptr
    %163 = llvm.add %157, %12 : i64
    llvm.br ^bb30(%163 : i64)
  ^bb32:  // pred: ^bb30
    %164 = llvm.add %155, %12 : i64
    llvm.br ^bb28(%164 : i64)
  ^bb33:  // pred: ^bb28
    %165 = llvm.mul %70, %18 : i64
    %166 = llvm.mul %165, %43 : i64
    %167 = llvm.getelementptr %145[%153] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%167, %167, %166) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %168 = llvm.add %153, %18 : i64
    llvm.br ^bb26(%168 : i64)
  ^bb34:  // pred: ^bb26
    llvm.br ^bb35(%13 : i64)
  ^bb35(%169: i64):  // 2 preds: ^bb34, ^bb48
    %170 = llvm.icmp "slt" %169, %17 : i64
    llvm.cond_br %170, ^bb36, ^bb49
  ^bb36:  // pred: ^bb35
    llvm.br ^bb37(%13 : i64)
  ^bb37(%171: i64):  // 2 preds: ^bb36, ^bb47
    %172 = llvm.icmp "slt" %171, %17 : i64
    llvm.cond_br %172, ^bb38, ^bb48
  ^bb38:  // pred: ^bb37
    llvm.br ^bb39(%13 : i64)
  ^bb39(%173: i64):  // 2 preds: ^bb38, ^bb46
    %174 = llvm.icmp "slt" %173, %12 : i64
    llvm.cond_br %174, ^bb40, ^bb47
  ^bb40:  // pred: ^bb39
    llvm.br ^bb41(%13 : i64)
  ^bb41(%175: i64):  // 2 preds: ^bb40, ^bb45
    %176 = llvm.icmp "slt" %175, %18 : i64
    llvm.cond_br %176, ^bb42, ^bb46
  ^bb42:  // pred: ^bb41
    llvm.br ^bb43(%13 : i64)
  ^bb43(%177: i64):  // 2 preds: ^bb42, ^bb44
    %178 = llvm.icmp "slt" %177, %18 : i64
    llvm.cond_br %178, ^bb44, ^bb45
  ^bb44:  // pred: ^bb43
    %179 = llvm.getelementptr %111[%171] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %180 = llvm.mul %173, %17 : i64
    %181 = llvm.add %180, %177 : i64
    %182 = llvm.getelementptr %179[%181] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %183 = llvm.load %182 : !llvm.ptr -> f32
    %184 = llvm.getelementptr %145[%169] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %185 = llvm.add %180, %175 : i64
    %186 = llvm.getelementptr %184[%185] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %187 = llvm.load %186 : !llvm.ptr -> f32
    %188 = llvm.fmul %183, %23  : f32
    %189 = llvm.fadd %187, %188  : f32
    llvm.store %189, %186 : f32, !llvm.ptr
    %190 = llvm.add %177, %12 : i64
    llvm.br ^bb43(%190 : i64)
  ^bb45:  // pred: ^bb43
    %191 = llvm.add %175, %12 : i64
    llvm.br ^bb41(%191 : i64)
  ^bb46:  // pred: ^bb41
    %192 = llvm.add %173, %12 : i64
    llvm.br ^bb39(%192 : i64)
  ^bb47:  // pred: ^bb39
    %193 = llvm.mul %70, %18 : i64
    %194 = llvm.mul %193, %43 : i64
    %195 = llvm.getelementptr %145[%169] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%195, %195, %194) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %196 = llvm.add %171, %18 : i64
    llvm.br ^bb37(%196 : i64)
  ^bb48:  // pred: ^bb37
    %197 = llvm.add %169, %18 : i64
    llvm.br ^bb35(%197 : i64)
  ^bb49:  // pred: ^bb35
    %198 = llvm.call @malloc(%56) : (i64) -> !llvm.ptr
    %199 = llvm.ptrtoint %198 : !llvm.ptr to i64
    %200 = llvm.add %199, %34 : i64
    %201 = llvm.urem %200, %1  : i64
    %202 = llvm.sub %200, %201 : i64
    %203 = llvm.inttoptr %202 : i64 to !llvm.ptr
    llvm.br ^bb50(%13 : i64)
  ^bb50(%204: i64):  // 2 preds: ^bb49, ^bb57
    %205 = llvm.icmp "slt" %204, %17 : i64
    llvm.cond_br %205, ^bb51, ^bb58
  ^bb51:  // pred: ^bb50
    llvm.br ^bb52(%13 : i64)
  ^bb52(%206: i64):  // 2 preds: ^bb51, ^bb56
    %207 = llvm.icmp "slt" %206, %12 : i64
    llvm.cond_br %207, ^bb53, ^bb57
  ^bb53:  // pred: ^bb52
    llvm.br ^bb54(%13 : i64)
  ^bb54(%208: i64):  // 2 preds: ^bb53, ^bb55
    %209 = llvm.icmp "slt" %208, %18 : i64
    llvm.cond_br %209, ^bb55, ^bb56
  ^bb55:  // pred: ^bb54
    %210 = llvm.getelementptr %203[%204] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %211 = llvm.mul %206, %17 : i64
    %212 = llvm.add %211, %208 : i64
    %213 = llvm.getelementptr %210[%212] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %25, %213 : f32, !llvm.ptr
    %214 = llvm.add %208, %12 : i64
    llvm.br ^bb54(%214 : i64)
  ^bb56:  // pred: ^bb54
    %215 = llvm.add %206, %12 : i64
    llvm.br ^bb52(%215 : i64)
  ^bb57:  // pred: ^bb52
    %216 = llvm.mul %70, %18 : i64
    %217 = llvm.mul %216, %43 : i64
    %218 = llvm.getelementptr %203[%204] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%218, %218, %217) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %219 = llvm.add %204, %18 : i64
    llvm.br ^bb50(%219 : i64)
  ^bb58:  // pred: ^bb50
    %220 = llvm.call @malloc(%56) : (i64) -> !llvm.ptr
    %221 = llvm.ptrtoint %220 : !llvm.ptr to i64
    %222 = llvm.add %221, %34 : i64
    %223 = llvm.urem %222, %1  : i64
    %224 = llvm.sub %222, %223 : i64
    %225 = llvm.inttoptr %224 : i64 to !llvm.ptr
    "llvm.intr.memcpy"(%225, %203, %72) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb59(%13 : i64)
  ^bb59(%226: i64):  // 2 preds: ^bb58, ^bb72
    %227 = llvm.icmp "slt" %226, %17 : i64
    llvm.cond_br %227, ^bb60, ^bb73
  ^bb60:  // pred: ^bb59
    llvm.br ^bb61(%13 : i64)
  ^bb61(%228: i64):  // 2 preds: ^bb60, ^bb71
    %229 = llvm.icmp "slt" %228, %17 : i64
    llvm.cond_br %229, ^bb62, ^bb72
  ^bb62:  // pred: ^bb61
    llvm.br ^bb63(%13 : i64)
  ^bb63(%230: i64):  // 2 preds: ^bb62, ^bb70
    %231 = llvm.icmp "slt" %230, %12 : i64
    llvm.cond_br %231, ^bb64, ^bb71
  ^bb64:  // pred: ^bb63
    llvm.br ^bb65(%13 : i64)
  ^bb65(%232: i64):  // 2 preds: ^bb64, ^bb69
    %233 = llvm.icmp "slt" %232, %18 : i64
    llvm.cond_br %233, ^bb66, ^bb70
  ^bb66:  // pred: ^bb65
    llvm.br ^bb67(%13 : i64)
  ^bb67(%234: i64):  // 2 preds: ^bb66, ^bb68
    %235 = llvm.icmp "slt" %234, %18 : i64
    llvm.cond_br %235, ^bb68, ^bb69
  ^bb68:  // pred: ^bb67
    %236 = llvm.getelementptr %111[%228] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %237 = llvm.mul %230, %17 : i64
    %238 = llvm.add %237, %234 : i64
    %239 = llvm.getelementptr %236[%238] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %240 = llvm.load %239 : !llvm.ptr -> f32
    %241 = llvm.getelementptr %225[%226] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %242 = llvm.add %237, %232 : i64
    %243 = llvm.getelementptr %241[%242] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %244 = llvm.load %243 : !llvm.ptr -> f32
    %245 = llvm.fmul %240, %22  : f32
    %246 = llvm.fadd %244, %245  : f32
    llvm.store %246, %243 : f32, !llvm.ptr
    %247 = llvm.add %234, %12 : i64
    llvm.br ^bb67(%247 : i64)
  ^bb69:  // pred: ^bb67
    %248 = llvm.add %232, %12 : i64
    llvm.br ^bb65(%248 : i64)
  ^bb70:  // pred: ^bb65
    %249 = llvm.add %230, %12 : i64
    llvm.br ^bb63(%249 : i64)
  ^bb71:  // pred: ^bb63
    %250 = llvm.mul %70, %18 : i64
    %251 = llvm.mul %250, %43 : i64
    %252 = llvm.getelementptr %225[%226] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%252, %252, %251) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %253 = llvm.add %228, %18 : i64
    llvm.br ^bb61(%253 : i64)
  ^bb72:  // pred: ^bb61
    %254 = llvm.add %226, %18 : i64
    llvm.br ^bb59(%254 : i64)
  ^bb73:  // pred: ^bb59
    %255 = llvm.call @malloc(%56) : (i64) -> !llvm.ptr
    %256 = llvm.ptrtoint %255 : !llvm.ptr to i64
    %257 = llvm.add %256, %34 : i64
    %258 = llvm.urem %257, %1  : i64
    %259 = llvm.sub %257, %258 : i64
    %260 = llvm.inttoptr %259 : i64 to !llvm.ptr
    llvm.br ^bb74(%13 : i64)
  ^bb74(%261: i64):  // 2 preds: ^bb73, ^bb81
    %262 = llvm.icmp "slt" %261, %17 : i64
    llvm.cond_br %262, ^bb75, ^bb82
  ^bb75:  // pred: ^bb74
    llvm.br ^bb76(%13 : i64)
  ^bb76(%263: i64):  // 2 preds: ^bb75, ^bb80
    %264 = llvm.icmp "slt" %263, %12 : i64
    llvm.cond_br %264, ^bb77, ^bb81
  ^bb77:  // pred: ^bb76
    llvm.br ^bb78(%13 : i64)
  ^bb78(%265: i64):  // 2 preds: ^bb77, ^bb79
    %266 = llvm.icmp "slt" %265, %18 : i64
    llvm.cond_br %266, ^bb79, ^bb80
  ^bb79:  // pred: ^bb78
    %267 = llvm.getelementptr %260[%261] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %268 = llvm.mul %263, %17 : i64
    %269 = llvm.add %268, %265 : i64
    %270 = llvm.getelementptr %267[%269] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %25, %270 : f32, !llvm.ptr
    %271 = llvm.add %265, %12 : i64
    llvm.br ^bb78(%271 : i64)
  ^bb80:  // pred: ^bb78
    %272 = llvm.add %263, %12 : i64
    llvm.br ^bb76(%272 : i64)
  ^bb81:  // pred: ^bb76
    %273 = llvm.mul %70, %18 : i64
    %274 = llvm.mul %273, %43 : i64
    %275 = llvm.getelementptr %260[%261] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%275, %275, %274) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %276 = llvm.add %261, %18 : i64
    llvm.br ^bb74(%276 : i64)
  ^bb82:  // pred: ^bb74
    %277 = llvm.call @malloc(%56) : (i64) -> !llvm.ptr
    %278 = llvm.ptrtoint %277 : !llvm.ptr to i64
    %279 = llvm.add %278, %34 : i64
    %280 = llvm.urem %279, %1  : i64
    %281 = llvm.sub %279, %280 : i64
    %282 = llvm.inttoptr %281 : i64 to !llvm.ptr
    "llvm.intr.memcpy"(%282, %260, %72) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    llvm.br ^bb83(%13 : i64)
  ^bb83(%283: i64):  // 2 preds: ^bb82, ^bb96
    %284 = llvm.icmp "slt" %283, %17 : i64
    llvm.cond_br %284, ^bb84, ^bb97
  ^bb84:  // pred: ^bb83
    llvm.br ^bb85(%13 : i64)
  ^bb85(%285: i64):  // 2 preds: ^bb84, ^bb95
    %286 = llvm.icmp "slt" %285, %17 : i64
    llvm.cond_br %286, ^bb86, ^bb96
  ^bb86:  // pred: ^bb85
    llvm.br ^bb87(%13 : i64)
  ^bb87(%287: i64):  // 2 preds: ^bb86, ^bb94
    %288 = llvm.icmp "slt" %287, %12 : i64
    llvm.cond_br %288, ^bb88, ^bb95
  ^bb88:  // pred: ^bb87
    llvm.br ^bb89(%13 : i64)
  ^bb89(%289: i64):  // 2 preds: ^bb88, ^bb93
    %290 = llvm.icmp "slt" %289, %18 : i64
    llvm.cond_br %290, ^bb90, ^bb94
  ^bb90:  // pred: ^bb89
    llvm.br ^bb91(%13 : i64)
  ^bb91(%291: i64):  // 2 preds: ^bb90, ^bb92
    %292 = llvm.icmp "slt" %291, %18 : i64
    llvm.cond_br %292, ^bb92, ^bb93
  ^bb92:  // pred: ^bb91
    %293 = llvm.getelementptr %111[%285] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %294 = llvm.mul %287, %17 : i64
    %295 = llvm.add %294, %291 : i64
    %296 = llvm.getelementptr %293[%295] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %297 = llvm.load %296 : !llvm.ptr -> f32
    %298 = llvm.getelementptr %282[%283] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %299 = llvm.add %294, %289 : i64
    %300 = llvm.getelementptr %298[%299] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %301 = llvm.load %300 : !llvm.ptr -> f32
    %302 = llvm.fmul %297, %21  : f32
    %303 = llvm.fadd %301, %302  : f32
    llvm.store %303, %300 : f32, !llvm.ptr
    %304 = llvm.add %291, %12 : i64
    llvm.br ^bb91(%304 : i64)
  ^bb93:  // pred: ^bb91
    %305 = llvm.add %289, %12 : i64
    llvm.br ^bb89(%305 : i64)
  ^bb94:  // pred: ^bb89
    %306 = llvm.add %287, %12 : i64
    llvm.br ^bb87(%306 : i64)
  ^bb95:  // pred: ^bb87
    %307 = llvm.mul %70, %18 : i64
    %308 = llvm.mul %307, %43 : i64
    %309 = llvm.getelementptr %282[%283] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%309, %309, %308) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %310 = llvm.add %285, %18 : i64
    llvm.br ^bb85(%310 : i64)
  ^bb96:  // pred: ^bb85
    %311 = llvm.add %283, %18 : i64
    llvm.br ^bb83(%311 : i64)
  ^bb97:  // pred: ^bb83
    %312 = llvm.mul %73, %3 : i64
    %313 = llvm.mul %53, %17 : i64
    %314 = llvm.add %312, %313 : i64
    %315 = llvm.mul %70, %12 : i64
    %316 = llvm.mul %315, %17 : i64
    %317 = llvm.mul %316, %43 : i64
    %318 = llvm.getelementptr %38[%314] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%318, %225, %317) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %319 = llvm.getelementptr %50[%314] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%319, %282, %317) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %320 = llvm.add %73, %12 : i64
    llvm.br ^bb3(%320, %152 : i64, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
  ^bb98:  // pred: ^bb3
    %321 = llvm.getelementptr %26[32000] : (!llvm.ptr) -> !llvm.ptr, f32
    %322 = llvm.ptrtoint %321 : !llvm.ptr to i64
    %323 = llvm.add %322, %1 : i64
    %324 = llvm.call @malloc(%323) : (i64) -> !llvm.ptr
    %325 = llvm.ptrtoint %324 : !llvm.ptr to i64
    %326 = llvm.add %325, %34 : i64
    %327 = llvm.urem %326, %1  : i64
    %328 = llvm.sub %326, %327 : i64
    %329 = llvm.inttoptr %328 : i64 to !llvm.ptr
    llvm.br ^bb99(%13 : i64)
  ^bb99(%330: i64):  // 2 preds: ^bb98, ^bb106
    %331 = llvm.icmp "slt" %330, %16 : i64
    llvm.cond_br %331, ^bb100, ^bb107
  ^bb100:  // pred: ^bb99
    llvm.br ^bb101(%13 : i64)
  ^bb101(%332: i64):  // 2 preds: ^bb100, ^bb105
    %333 = llvm.icmp "slt" %332, %12 : i64
    llvm.cond_br %333, ^bb102, ^bb106
  ^bb102:  // pred: ^bb101
    llvm.br ^bb103(%13 : i64)
  ^bb103(%334: i64):  // 2 preds: ^bb102, ^bb104
    %335 = llvm.icmp "slt" %334, %18 : i64
    llvm.cond_br %335, ^bb104, ^bb105
  ^bb104:  // pred: ^bb103
    %336 = llvm.getelementptr %329[%330] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %337 = llvm.mul %332, %16 : i64
    %338 = llvm.add %337, %334 : i64
    %339 = llvm.getelementptr %336[%338] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %25, %339 : f32, !llvm.ptr
    %340 = llvm.add %334, %12 : i64
    llvm.br ^bb103(%340 : i64)
  ^bb105:  // pred: ^bb103
    %341 = llvm.add %332, %12 : i64
    llvm.br ^bb101(%341 : i64)
  ^bb106:  // pred: ^bb101
    %342 = llvm.mul %70, %18 : i64
    %343 = llvm.mul %342, %43 : i64
    %344 = llvm.getelementptr %329[%330] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%344, %344, %343) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %345 = llvm.add %330, %18 : i64
    llvm.br ^bb99(%345 : i64)
  ^bb107:  // pred: ^bb99
    llvm.br ^bb108(%13 : i64)
  ^bb108(%346: i64):  // 2 preds: ^bb107, ^bb121
    %347 = llvm.icmp "slt" %346, %16 : i64
    llvm.cond_br %347, ^bb109, ^bb122
  ^bb109:  // pred: ^bb108
    llvm.br ^bb110(%13 : i64)
  ^bb110(%348: i64):  // 2 preds: ^bb109, ^bb120
    %349 = llvm.icmp "slt" %348, %17 : i64
    llvm.cond_br %349, ^bb111, ^bb121
  ^bb111:  // pred: ^bb110
    %350 = llvm.extractvalue %74[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb112(%13 : i64)
  ^bb112(%351: i64):  // 2 preds: ^bb111, ^bb119
    %352 = llvm.icmp "slt" %351, %12 : i64
    llvm.cond_br %352, ^bb113, ^bb120
  ^bb113:  // pred: ^bb112
    llvm.br ^bb114(%13 : i64)
  ^bb114(%353: i64):  // 2 preds: ^bb113, ^bb118
    %354 = llvm.icmp "slt" %353, %18 : i64
    llvm.cond_br %354, ^bb115, ^bb119
  ^bb115:  // pred: ^bb114
    llvm.br ^bb116(%13 : i64)
  ^bb116(%355: i64):  // 2 preds: ^bb115, ^bb117
    %356 = llvm.icmp "slt" %355, %18 : i64
    llvm.cond_br %356, ^bb117, ^bb118
  ^bb117:  // pred: ^bb116
    %357 = llvm.getelementptr %350[%348] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %358 = llvm.mul %351, %17 : i64
    %359 = llvm.add %358, %355 : i64
    %360 = llvm.getelementptr %357[%359] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %361 = llvm.load %360 : !llvm.ptr -> f32
    %362 = llvm.getelementptr %329[%346] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %363 = llvm.mul %351, %16 : i64
    %364 = llvm.add %363, %353 : i64
    %365 = llvm.getelementptr %362[%364] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %366 = llvm.load %365 : !llvm.ptr -> f32
    %367 = llvm.fmul %361, %20  : f32
    %368 = llvm.fadd %366, %367  : f32
    llvm.store %368, %365 : f32, !llvm.ptr
    %369 = llvm.add %355, %12 : i64
    llvm.br ^bb116(%369 : i64)
  ^bb118:  // pred: ^bb116
    %370 = llvm.add %353, %12 : i64
    llvm.br ^bb114(%370 : i64)
  ^bb119:  // pred: ^bb114
    %371 = llvm.add %351, %12 : i64
    llvm.br ^bb112(%371 : i64)
  ^bb120:  // pred: ^bb112
    %372 = llvm.mul %70, %18 : i64
    %373 = llvm.mul %372, %43 : i64
    %374 = llvm.getelementptr %329[%346] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    "llvm.intr.memcpy"(%374, %374, %373) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %375 = llvm.add %348, %18 : i64
    llvm.br ^bb110(%375 : i64)
  ^bb121:  // pred: ^bb110
    %376 = llvm.add %346, %18 : i64
    llvm.br ^bb108(%376 : i64)
  ^bb122:  // pred: ^bb108
    %377 = llvm.add %43, %1 : i64
    %378 = llvm.call @malloc(%377) : (i64) -> !llvm.ptr
    %379 = llvm.ptrtoint %378 : !llvm.ptr to i64
    %380 = llvm.add %379, %34 : i64
    %381 = llvm.urem %380, %1  : i64
    %382 = llvm.sub %380, %381 : i64
    %383 = llvm.inttoptr %382 : i64 to !llvm.ptr
    llvm.br ^bb123(%13 : i64)
  ^bb123(%384: i64):  // 2 preds: ^bb122, ^bb124
    %385 = llvm.icmp "slt" %384, %12 : i64
    llvm.cond_br %385, ^bb124, ^bb125
  ^bb124:  // pred: ^bb123
    %386 = llvm.getelementptr %383[%384] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %19, %386 : f32, !llvm.ptr
    %387 = llvm.add %384, %12 : i64
    llvm.br ^bb123(%387 : i64)
  ^bb125:  // pred: ^bb123
    %388 = llvm.getelementptr %26[1] : (!llvm.ptr) -> !llvm.ptr, i64
    %389 = llvm.ptrtoint %388 : !llvm.ptr to i64
    %390 = llvm.add %389, %1 : i64
    %391 = llvm.call @malloc(%390) : (i64) -> !llvm.ptr
    %392 = llvm.ptrtoint %391 : !llvm.ptr to i64
    %393 = llvm.add %392, %34 : i64
    %394 = llvm.urem %393, %1  : i64
    %395 = llvm.sub %393, %394 : i64
    %396 = llvm.inttoptr %395 : i64 to !llvm.ptr
    %397 = llvm.insertvalue %391, %7[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %398 = llvm.insertvalue %396, %397[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %399 = llvm.insertvalue %13, %398[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %400 = llvm.insertvalue %12, %399[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %401 = llvm.insertvalue %12, %400[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.br ^bb126(%13 : i64)
  ^bb126(%402: i64):  // 2 preds: ^bb125, ^bb127
    %403 = llvm.icmp "slt" %402, %12 : i64
    llvm.cond_br %403, ^bb127, ^bb128
  ^bb127:  // pred: ^bb126
    %404 = llvm.getelementptr %396[%402] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %9, %404 : i64, !llvm.ptr
    %405 = llvm.add %402, %12 : i64
    llvm.br ^bb126(%405 : i64)
  ^bb128:  // pred: ^bb126
    llvm.br ^bb129(%13 : i64)
  ^bb129(%406: i64):  // 2 preds: ^bb128, ^bb136
    %407 = llvm.icmp "slt" %406, %16 : i64
    llvm.cond_br %407, ^bb130, ^bb137
  ^bb130:  // pred: ^bb129
    llvm.br ^bb131(%13 : i64)
  ^bb131(%408: i64):  // 2 preds: ^bb130, ^bb135
    %409 = llvm.icmp "slt" %408, %12 : i64
    llvm.cond_br %409, ^bb132, ^bb136
  ^bb132:  // pred: ^bb131
    llvm.br ^bb133(%13 : i64)
  ^bb133(%410: i64):  // 2 preds: ^bb132, ^bb134
    %411 = llvm.icmp "slt" %410, %18 : i64
    llvm.cond_br %411, ^bb134, ^bb135
  ^bb134:  // pred: ^bb133
    %412 = llvm.getelementptr %329[%406] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %413 = llvm.mul %408, %16 : i64
    %414 = llvm.add %413, %410 : i64
    %415 = llvm.getelementptr %412[%414] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %416 = llvm.load %415 : !llvm.ptr -> f32
    %417 = llvm.getelementptr %383[%408] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %418 = llvm.load %417 : !llvm.ptr -> f32
    %419 = llvm.getelementptr %396[%408] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %420 = llvm.load %419 : !llvm.ptr -> i64
    %421 = llvm.add %410, %406 : i64
    %422 = llvm.fcmp "ogt" %416, %418 : f32
    %423 = llvm.select %422, %416, %418 : i1, f32
    %424 = llvm.select %422, %421, %420 : i1, i64
    llvm.store %423, %417 : f32, !llvm.ptr
    llvm.store %424, %419 : i64, !llvm.ptr
    %425 = llvm.add %410, %12 : i64
    llvm.br ^bb133(%425 : i64)
  ^bb135:  // pred: ^bb133
    %426 = llvm.add %408, %12 : i64
    llvm.br ^bb131(%426 : i64)
  ^bb136:  // pred: ^bb131
    %427 = llvm.add %406, %18 : i64
    llvm.br ^bb129(%427 : i64)
  ^bb137:  // pred: ^bb129
    %428 = llvm.alloca %12 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %401, %428 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    llvm.call @printMemrefI64(%12, %428) : (i64, !llvm.ptr) -> ()
    %429 = llvm.add %53, %10 : i64
    llvm.br ^bb1(%429 : i64)
  ^bb138:  // pred: ^bb1
    llvm.return
  }
}
